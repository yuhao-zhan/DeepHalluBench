import argparse
from fileinput import filename
import json
import os
import shutil
from typing import List, Dict, Any
import logging
import time
import concurrent.futures

import sys
if len(sys.argv) > 1 and '--gpu_ids' in sys.argv:
    try:
        gpu_ids_idx = sys.argv.index('--gpu_ids') + 1
        if gpu_ids_idx < len(sys.argv):
            gpu_ids_str = sys.argv[gpu_ids_idx]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
    except (ValueError, IndexError):
        pass

from decomposition import decompose_workflow_to_cache_auto, decompose_report_to_cache_auto, decompose_query
from claim_checking_LLM import process_claims_and_urls_new
from action_checking import process_actions_and_memory_new
from overall_noise_domination import noise_domination_detection
from process_filtered_misaligned_with_modified_urls import process_single_file as process_filtered_misaligned_single_file, collect_from_filtered_files, compute_modified_targets, load_web_cache as load_web_cache_for_filtered, get_summary_citations as get_summary_citations_for_filtered
# Import from local utils module - use absolute path to avoid multiprocessing conflicts
import sys
import importlib.util
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import with explicit module path to avoid conflicts
spec = importlib.util.spec_from_file_location("local_utils", os.path.join(current_dir, "utils.py"))
local_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_utils)

# Now import the function we need
is_url = local_utils.is_url
from fixed_thre_claim_link_to_query import find_relevant_queries_for_claims
from rerank_chunk_score import IntegratedChunkScorer
from judge_HC_against_memory import MemoryJudge

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sepcify process name
import setproctitle
setproctitle.setproctitle('evaluate_for_Qwen')

# Global BGEScorer instance to avoid loading reranker models multiple times
# This ensures that the expensive reranker model loading happens only once,
# and the same instance is reused across all scoring operations
_global_bge_scorer = None


def ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _deep_merge(dest: dict, src: dict) -> dict:
    """
    Recursively merge src into dest without dropping existing keys.
    - For dict values: merge recursively
    - For lists and scalars: src overwrites dest
    """
    for key, src_val in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(src_val, dict):
            _deep_merge(dest[key], src_val)
        else:
            dest[key] = src_val
    return dest


def _has_memory_processing(result_path: str) -> bool:
    """
    Return True if the combined result file already contains memory-based judgments.
    """
    if not os.path.exists(result_path):
        return False
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False

    def _claims_have_memory(blocks: List[Dict[str, Any]]) -> bool:
        if not blocks:
            return False
        for block in blocks:
            for claim in block.get("claim_results", []):
                source = claim.get("processing_source")
                if source in ("Memory_LLM", "Memory_NLI"):
                    return True
        return False

    return (
        _claims_have_memory(data.get("chain_of_research_results")) or
        _claims_have_memory(data.get("report_results"))
    )


def get_global_bge_scorer(num_gpus=4, gpu_ids=None):
    global _global_bge_scorer
    if _global_bge_scorer is None:
        print("🔧 Initializing global BGEScorer...")
        from reranker_scoring import BGEScorer
        _global_bge_scorer = BGEScorer(num_gpus=num_gpus, gpu_ids=gpu_ids)
    return _global_bge_scorer


def monitor_memory_usage():
    # Memory monitoring disabled to reduce output
    pass


_global_models = {}

def initialize_global_models():
    global _global_models
    if not _global_models:
        print("🔧 Initializing global models for memory efficiency…")
    # Models will be initialized lazily when first needed
    _global_models['initialized'] = True


def create_global_url_mapping(all_urls: List[str]) -> Dict[str, int]:
    url_mapping = {}
    for idx, url in enumerate(all_urls):
        url_mapping[url] = idx
    return url_mapping


def score_all_chunks_upfront(
    sub_queries: List[str], 
    all_urls: List[str], 
    web_content_cache: Dict[str, str], 
    cache_file: str, 
    url_mapping: Dict[str, int],
    num_gpus: int = 4,
    gpu_ids: List[int] = None
) -> Dict[str, Any]:
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
            if 'chunk_score' in all_content:
                return all_content['chunk_score']

    print("🔄 Scoring all chunks upfront...")
    
    # Get the global BGEScorer instance (loaded only once)
    global_bge_scorer = get_global_bge_scorer(num_gpus, gpu_ids)
    
    scorer = IntegratedChunkScorer(
        sbert_model='all-MiniLM-L6-v2',
        ner_threshold=0.5,
        c=6.0,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        reranker_instance=global_bge_scorer  # Pass the global instance directly
    )
    
    scoring_results = scorer.score_chunks_sync(
        queries=sub_queries,
        urls=all_urls,  # Use all_urls to score everything upfront
        web_content_cache=web_content_cache,
        cache_file=cache_file,
        url_mapping=url_mapping
    )
    
    # Memory cleanup after scoring
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return scoring_results


def count_all_claims(cache_data: dict) -> int:
    """
    Count all claims from claim_list in iterations and atomic_claims in report paragraphs.
    Returns the total count of all claims (including duplicates).
    """
    all_claims = []
    
    # Extract claims from iterations
    iterations = cache_data.get('iterations', [])
    for i, iteration in enumerate(iterations):
        claim_list = iteration.get(f'claim_list_{i+1}', [])
        all_claims.extend(claim_list)
    
    # Extract claims from report paragraphs
    report_paragraphs = cache_data.get('report', [])
    for para in report_paragraphs:
        atomic_claims = para.get('atomic_claims', [])
        all_claims.extend(atomic_claims)
    
    return len(all_claims)


def map_all_claims_to_queries_upfront(
    cache_data: dict,
    cache_file: str
) -> Dict[str, Any]:

    ensure_parent_dir(cache_file)

    # Check if claim-query mappings already exist in cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                all_content = json.load(f)
            if 'related_query' in all_content and len(all_content['related_query']) > 0:
                return all_content['related_query']
        except Exception as e:
            pass
    
    # Get all claims from cache data
    all_claims = []
    
    # Extract claims from iterations
    iterations = cache_data.get('iterations', [])
    for i, iteration in enumerate(iterations):
        claim_list = iteration.get(f'claim_list_{i+1}', [])
        all_claims.extend(claim_list)
    
    # Extract claims from report paragraphs
    report_paragraphs = cache_data.get('report', [])
    for para in report_paragraphs:
        atomic_claims = para.get('atomic_claims', [])
        all_claims.extend(atomic_claims)
    
    # Remove duplicates while preserving order
    unique_claims = []
    seen_claims = set()
    for claim in all_claims:
        if claim not in seen_claims:
            unique_claims.append(claim)
            seen_claims.add(claim)
    
    # Get query list from cache
    query_list = cache_data.get('query_list', [])
    
    try:
        # Find relevant queries for all claims at once
        selected_queries = find_relevant_queries_for_claims(unique_claims, query_list, fixed_threshold=0.0001)
        
        # Convert the result format to use claim text as keys
        claim_query_mappings = {}
        for claim_idx, claim_info in selected_queries.items():
            claim_text = claim_info['claim']
            claim_query_mappings[claim_text] = claim_info
        
    except Exception as e:
        print(f"❌ Error mapping claims to queries: {e}")
        claim_query_mappings = {}
    
    # Save the mappings to cache file
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
        
        # Add the new claim-query mappings
        if 'related_query' not in all_content:
            all_content['related_query'] = {}
        all_content['related_query'].update(claim_query_mappings)
        
        # Write back to cache file
        ensure_parent_dir(cache_file)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"❌ Error saving claim-query mappings to cache: {e}")
    
    return claim_query_mappings


def main():
    start_time = time.time()
    
    # Set HF_ENDPOINT environment variable for Hugging Face mirror
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"🔧 Set HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    
    parser = argparse.ArgumentParser(
        description='Semantic Fact Check Evaluation Pipeline'
    )
    parser.add_argument('input_json', help='Path to input JSON file')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use (default: 3)')
    parser.add_argument('--gpu_ids', type=str, default="0,1", help='Comma-separated GPU IDs to use (e.g., "0,1,3"). Default: "0,1,3"')
    parser.add_argument('--skip-noise-domination', action='store_true', help='Skip noise domination detection')
    args = parser.parse_args()
    
    # Parse GPU IDs for internal use
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            print(f"🔧 Internal GPU Configuration: {args.num_gpus} GPUs, IDs: {gpu_ids}")
            
            # Note: Not setting CUDA_VISIBLE_DEVICES to allow direct GPU ID access
            gpu_ids_str = ','.join(map(str, gpu_ids))
            print(f"🎯 Will use GPU IDs directly: {gpu_ids_str}")
            
        except ValueError:
            print(f"❌ Invalid GPU IDs format: {args.gpu_ids}. Use comma-separated integers (e.g., '0,1,3')")
            return
    else:
        gpu_ids = list(range(args.num_gpus))
        print(f"🔧 Internal GPU Configuration: {args.num_gpus} GPUs, using first {args.num_gpus} GPUs")
        
        # Note: Not setting CUDA_VISIBLE_DEVICES for default case either
        gpu_ids_str = ','.join(map(str, gpu_ids))
        print(f"🎯 Will use GPU IDs directly: {gpu_ids_str}")

    # Initial memory check
    print(f"\n{'='*50}")
    print("INITIAL MEMORY STATUS")
    print(f"{'='*50}")
    monitor_memory_usage()
    
    # Initialize global BGEScorer early to avoid repeated model loading
    print(f"\n{'='*50}")
    print("INITIALIZING GLOBAL BGESCORER (ONCE ONLY)")
    print(f"{'='*50}")
    get_global_bge_scorer(args.num_gpus, gpu_ids)  # This will load the reranker models once

    # read the query and report from the input json
    with open(args.input_json, 'r', encoding='utf-8') as f:
        text = json.load(f)
        query = text.get('query', '')
        report = text.get('final_report', '')
        all_urls = text.get('all_source_links', [])
        urls = text.get('summary_citations', [])
    
    # Create cache file path
    web_content_cache_file = f"../web_content_cache/benchmark/Tongyi_DR_30B/cache_{os.path.basename(args.input_json)}"
    
    # Create output file path
    output_file = f"../results/benchmark/Tongyi_DR_30B/before_update/results_{os.path.basename(args.input_json)}"
    ensure_parent_dir(output_file)

    # # If result file is not empty, skip the evaluation
    # if os.path.exists(output_file):
    #     with open(output_file, 'r', encoding='utf-8') as f:
    #         final_data = json.load(f)
    #         if final_data.get('chain_of_research_results', []) and final_data.get('report_results', []):
    #             print(f"✅ Skipping evaluation because result file already exists: {output_file}")
    #             print(f"🎯 PROCESS_COMPLETED: Results already exist, skipping evaluation")
    #             return
    
    # Create global URL mapping for consistent chunk IDs
    url_mapping = create_global_url_mapping(all_urls)
    
    # Initialize the results file with basic structure
    # initialize_results_file(output_file, query, report, all_urls, urls)
    
    # Load web content from cache
    print(f"\n📥 Loading web content from cache...")
    if os.path.exists(web_content_cache_file):
        with open(web_content_cache_file, 'r', encoding='utf-8') as f:
            web_content_cache = json.load(f)
        print(f"✅ Loaded {len(web_content_cache)} URLs from cache")
    else:
        web_content_cache = {}
        print(f"⚠️ No cache file found. Skipping evaluation.")
        return None
    
    # Memory check after web content fetching
    print(f"\n💾 Memory status after web content fetching:")
    monitor_memory_usage()

    cache_file = f"../json_cache/benchmark/Tongyi_DR_30B/before_update/cache_{os.path.basename(args.input_json)}"
    
    print(f"\n🔍 Decomposing query...")
    ensure_parent_dir(cache_file)
    sub_queries = decompose_query(query, cache_file)

    # Decompose the query to get the atomic actions
    print(f"\n🔍 Decomposing workflow...")
    decompose_workflow_to_cache_auto(args.input_json, cache_file)
    
    # Decompose report paragraphs to cache if not already done
    print(f"\n📝 Decomposing report paragraphs to cache...")
    decompose_report_to_cache_auto(report, query, cache_file)
    
    # Read the updated cache file
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    # Count all claims to decide if we should skip chunk scoring
    total_claims = count_all_claims(cache_data)
    print(f"\n📊 Total claims found: {total_claims}")
    skip_chunk_scoring = total_claims < 5
    
    if skip_chunk_scoring:
        print(f"⚠️ Less than 5 claims found ({total_claims}). Skipping chunk_score computation and claim processing, going directly to action checking.")
        scoring_results = None
        claim_query_mappings = {}
    else:
        print(f"\n🔄 Scoring all chunks upfront...")
        scoring_results = score_all_chunks_upfront(sub_queries, all_urls, web_content_cache, cache_file, url_mapping, args.num_gpus, gpu_ids)

        print(f"\n🔄 Mapping all claims to their related queries upfront...")
        claim_query_mappings = map_all_claims_to_queries_upfront(cache_data, cache_file)

    print(f"\n🔄 Processing Chain of Research iterations...")

    # Initialize the final results structure
    final_results = {
        "query": query,
        "report": report,
        "all_urls": all_urls,
        "summary_urls": urls,
        "summary": {
            "total_iterations": len(cache_data.get('iterations', [])),
            "total_paragraphs": len(cache_data.get('report', [])),
            "processed_iterations": 0,
            "processed_paragraphs": 0
        }
    }

    # Judge the claims
    print(f"\n🔄 Judging the claims...")
    claim_results = None
    
    # Check if claim results already exist and are not empty
    skip_claim_verification = False
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        chain_results = existing_data.get('chain_of_research_results', [])
        report_results = existing_data.get('report_results', [])
        
        # Check if either chain_of_research_results or report_results is non-empty
        if (chain_results and len(chain_results) > 0 and 
            any(len(item.get('claim_results', [])) > 0 for item in chain_results)) or \
           (report_results and len(report_results) > 0 and 
            any(len(item.get('claim_results', [])) > 0 for item in report_results)):
            skip_claim_verification = True
            print(f"✅ Skipping claim verification - non-empty results already exist")
            claim_results = existing_data
    
    # Skip claim verification if we're skipping chunk scoring
    if skip_chunk_scoring:
        skip_claim_verification = True
        print(f"⚠️ Skipping claim verification - chunk scoring was skipped due to insufficient claims")
        claim_results = None
    
    if not skip_claim_verification:
        try:
            claim_results = process_claims_and_urls_new(query, cache_file, output_file, all_urls, web_content_cache, 0.4, 5, args.num_gpus, gpu_ids, url_mapping)
            if claim_results:
                # Merge claim results into final results
                final_results.update(claim_results)
                print(f"✅ Claim results merged into final results")
        except Exception as e:
            print(f"❌ Error in claim processing: {e}")
            claim_results = None
    else:
        # Use existing claim results
        if claim_results:
            final_results.update(claim_results)
            print(f"✅ Existing claim results merged into final results")

    # Second-turn rejudging for NotSupport claims
    print("🔄 Second-turn rejudging for NotSupport claims...")

    # Check if output_file already contains SECOND_TURN_LLM processing
    skip_second_turn_rejudging = False
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            def _has_second_turn_llm(blocks: List[Dict[str, Any]]) -> bool:
                if not blocks:
                    return False
                for block in blocks:
                    for claim in block.get("claim_results", []):
                        source = claim.get("processing_source")
                        if source == "SECOND_TURN_LLM":
                            return True
                return False
            
            if (_has_second_turn_llm(existing_data.get("chain_of_research_results")) or
                _has_second_turn_llm(existing_data.get("report_results"))):
                skip_second_turn_rejudging = True
        except Exception as e:
            pass
    
    if not skip_second_turn_rejudging:
        try:
            from second_turn_LLM_for_NotSupport import collect_notsupport_claims, rejudge_notsupport_claims_parallel, update_results_file
            
            notsupport_claims = collect_notsupport_claims(output_file)
            
            if notsupport_claims:
                updated_claim_results = rejudge_notsupport_claims_parallel(
                    notsupport_claims=notsupport_claims,
                    cache_file=cache_file,
                    query=query,
                    num_cores=64
                )
                
                update_results_file(output_file, updated_claim_results)
                
                with open(output_file, 'r', encoding='utf-8') as f:
                    updated_data = json.load(f)
                
                if 'chain_of_research_results' in updated_data:
                    final_results['chain_of_research_results'] = updated_data['chain_of_research_results']
                if 'report_results' in updated_data:
                    final_results['report_results'] = updated_data['report_results']
                
        except Exception as e:
            print(f"❌ Error in second-turn rejudging: {e}")

    
    # Process filtered misaligned claims with modified target URLs
    print("🔄 Processing filtered misaligned claims...")
    
    # Define paths
    filtered_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/utils/benchmark/Tongyi_DR_30B/filtered_misaligned_claims"
    misaligned_output_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/utils/benchmark/Tongyi_DR_30B/misaligned_output"
    misaligned_cache_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/utils/benchmark/Tongyi_DR_30B/cache"
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(misaligned_output_dir, exist_ok=True)
    os.makedirs(misaligned_cache_dir, exist_ok=True)
    file_id = os.path.basename(args.input_json).replace('.json', '')
    filtered_file_path = os.path.join(filtered_dir, f"filtered_{file_id}.json")
    
    # Check if intermediate misaligned files already exist
    misaligned_cache_file = os.path.join(misaligned_cache_dir, f"cache_{file_id}.json")
    misaligned_result_file = os.path.join(misaligned_output_dir, f"{file_id}_modified_target_results.json")
    skip_misaligned_processing = os.path.exists(misaligned_cache_file) and os.path.exists(misaligned_result_file)
    
    if not skip_misaligned_processing:
        # Generate filtered file if it doesn't exist
        if not os.path.exists(filtered_file_path):
            os.makedirs(filtered_dir, exist_ok=True)
            
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'evaluation_and_analysis', 'scripts'))
            from filter_misattributed_related_claims import process_result_file, save_json
            
            filtered_data = process_result_file(output_file, args.input_json)
            
            if filtered_data['statistics']['type1_count'] > 0 or filtered_data['statistics']['type2_count'] > 0:
                save_json(filtered_data, filtered_file_path)
        
        # Process filtered claims if they exist
        if os.path.exists(filtered_file_path):
            with open(filtered_file_path, 'r', encoding='utf-8') as f:
                filtered_data = json.load(f)
            
            type1_claims = filtered_data.get('type1_claims', [])
            
            if type1_claims:
                filtered_claims = [{
                    'file_id': file_id,
                    'claim': claim_data.get('claim', ''),
                    'original_target_urls': claim_data.get('target_urls', []),
                    'query': query
                } for claim_data in type1_claims]
                
                summary_citations_for_filtered = get_summary_citations_for_filtered(args.input_json)
                web_cache_for_filtered = load_web_cache_for_filtered(web_content_cache_file)
                
                process_filtered_misaligned_single_file(
                    file_id, filtered_claims, summary_citations_for_filtered, web_cache_for_filtered,
                    misaligned_output_dir, 0.4, 5, args.num_gpus, gpu_ids,
                    cache_dir=misaligned_cache_dir, url_mapping=url_mapping,
                    output_path=None, skip_first_round=False
                )

    # Combine the results and cache files
    print("🔄 Combining the results and cache files...")
    
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'evaluation_and_analysis', 'scripts'))
    from combine_cache_files import process_cache_pair
    from combine_attr_updated_claim_with_prev import process_file_pair as process_results_pair
    from pathlib import Path
    
    # Define paths for combined output
    updated_cache_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/benchmark/Tongyi_DR_30B/after_update"
    combined_output_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/benchmark/Tongyi_DR_30B/after_update"
    os.makedirs(updated_cache_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)
    updated_cache_path = os.path.join(updated_cache_dir, f"cache_{file_id}.json")
    combined_target_path = os.path.join(combined_output_dir, f"{file_id}_combined.json")
    
    # Combine cache files if misaligned cache exists
    if os.path.exists(misaligned_cache_file):
        debug_info_cache = {"errors": [], "warnings": [], "stats": {}}
        process_cache_pair(Path(cache_file), Path(misaligned_cache_file), Path(updated_cache_dir), debug_info_cache)
        cache_file = updated_cache_path
    else:
        if os.path.exists(cache_file):
            try:
                if os.path.abspath(cache_file) != os.path.abspath(updated_cache_path):
                    shutil.copy2(cache_file, updated_cache_path)
                cache_file = updated_cache_path
            except Exception as copy_err:
                print(f"❌ Failed to copy cache file: {copy_err}")
    
    # Combine results files if misaligned results exist
    skip_combining_due_to_memory = os.path.exists(combined_target_path) and _has_memory_processing(combined_target_path)

    if os.path.exists(misaligned_result_file):
        if not skip_combining_due_to_memory:
            debug_info_results = {"errors": [], "warnings": [], "not_found_claims": [], "stats": {}}
            existing_combined_before = None
            if os.path.exists(combined_target_path):
                try:
                    with open(combined_target_path, 'r', encoding='utf-8') as f:
                        existing_combined_before = json.load(f)
                except Exception:
                    existing_combined_before = None
            process_results_pair(Path(output_file), Path(misaligned_result_file), Path(combined_output_dir), debug_info_results)
            try:
                if os.path.exists(combined_target_path):
                    with open(combined_target_path, 'r', encoding='utf-8') as f:
                        new_combined = json.load(f)
                    if existing_combined_before is not None:
                        merged_combined = _deep_merge(existing_combined_before, new_combined)
                        with open(combined_target_path, 'w', encoding='utf-8') as f:
                            json.dump(merged_combined, f, ensure_ascii=False, indent=2)
            except Exception as _e:
                pass
            output_file = combined_target_path
        else:
            output_file = combined_target_path
    else:
        if skip_combining_due_to_memory:
            output_file = combined_target_path
        elif os.path.exists(output_file):
            try:
                if os.path.abspath(output_file) != os.path.abspath(combined_target_path):
                    shutil.copy2(output_file, combined_target_path)
                output_file = combined_target_path
            except Exception as copy_err:
                print(f"❌ Failed to copy results file: {copy_err}")

    print("🔄 Judging NotSupport claims against memory...")
    memory_judge_summary = None
    try:
        results_dir = os.path.dirname(output_file)
        cache_dir = os.path.dirname(cache_file)
        result_filename = os.path.basename(output_file)

        if result_filename.endswith("_combined.json"):
            file_id_for_memory = result_filename[:-len("_combined.json")]
            target_memory_path = output_file
        else:
            file_id_for_memory = None
            target_memory_path = None

        if file_id_for_memory and target_memory_path:
            if not _has_memory_processing(target_memory_path) and os.path.exists(target_memory_path):
                llm_workers = min(32, max(1, (os.cpu_count() or 1)))
                similarity_gpu_ids = gpu_ids if gpu_ids is not None else []

                memory_judge = MemoryJudge(
                    results_dir=results_dir,
                    cache_dir=cache_dir,
                    output_dir=results_dir,
                    similarity_top_k=10,
                    nli_threshold=0.99,
                    llm_workers=llm_workers,
                    similarity_gpu_ids=similarity_gpu_ids,
                )

                memory_judge_summary = memory_judge.process_single_file(file_id_for_memory)

                if memory_judge_summary:
                    final_results['memory_judgment_summary'] = memory_judge_summary

            if os.path.exists(target_memory_path):
                with open(target_memory_path, 'r', encoding='utf-8') as f:
                    memory_updated_data = json.load(f)
                _deep_merge(final_results, memory_updated_data)
    except Exception as e:
        print(f"❌ Error during memory judgment: {e}")


    # Judge the noise domination
    print("🔄 Judging the noise domination...")
    
    skip_noise_domination = args.skip_noise_domination
    
    # Check if noise domination results already exist and are not empty
    if not skip_noise_domination:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            nd_analysis = existing_data.get('nd_analysis', {})
            hallucination_score = existing_data.get('hallucination_score')
            
            if nd_analysis and (
                nd_analysis.get('total_nd', 0) > 0 or 
                nd_analysis.get('unmapped_cluster_details', []) or
                hallucination_score is not None
            ):
                skip_noise_domination = True
            
            # Merge ALL existing noise domination results
            nd_fields = [
                'hallucination_score', 'nd_analysis', 'clusters', 'mapped_clusters', 
                'unmapped_clusters', 'total_clusters', 'total_mapped', 'total_unmapped',
                'total_entailed_chunks', 'entailed_iteration_analysis', 
                'mapped_unmapped_feature_comparison', 'iteration_statistics', 
                'citation_statistics', 'iteration_level_nd_analysis'
            ]
            
            for field in nd_fields:
                if field in existing_data:
                    final_results[field] = existing_data[field]
    
    if not skip_noise_domination:
        try:
            noise_domination_detection(output_file, cache_file, args.input_json, args.num_gpus, gpu_ids)
            with open(output_file, 'r', encoding='utf-8') as f:
                updated_results = json.load(f)
            
            nd_fields = [
                'hallucination_score', 'nd_analysis', 'clusters', 'mapped_clusters', 
                'unmapped_clusters', 'total_clusters', 'total_mapped', 'total_unmapped',
                'total_entailed_chunks', 'entailed_iteration_analysis', 
                'mapped_unmapped_feature_comparison', 'iteration_statistics', 
                'citation_statistics', 'iteration_level_nd_analysis'
            ]
            
            for field in nd_fields:
                if field in updated_results:
                    final_results[field] = updated_results[field]
        except Exception as e:
            print(f"❌ Error in noise domination detection: {e}")

    # Judge the actions
    print("🔄 Judging the actions...")

    action_results = process_actions_and_memory_new(cache_file, args.input_json, output_file, top_k=10, num_gpus=args.num_gpus, gpu_ids=gpu_ids)
    if action_results:
        final_results.update(action_results)

    # Update summary with processed counts
    if claim_results and 'chain_of_research_results' in claim_results:
        final_results['summary']['processed_iterations'] = len(claim_results.get('chain_of_research_results', []))
    if claim_results and 'report_results' in claim_results:
        final_results['summary']['processed_paragraphs'] = len(claim_results.get('report_results', []))

    # Save the complete final results
    ensure_parent_dir(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"✅ Completed in {(end_time - start_time) / 60:.2f} minutes")
    
    # Clean up multiprocessing resources to avoid SIGTERM errors
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Force cleanup of global BGEScorer
        global _global_bge_scorer
        if _global_bge_scorer is not None:
            try:
                if hasattr(_global_bge_scorer, 'stop_multi_process_pool'):
                    _global_bge_scorer.stop_multi_process_pool()
                elif hasattr(_global_bge_scorer, 'stop_self_pool'):
                    _global_bge_scorer.stop_self_pool()
            except Exception as e:
                pass
            finally:
                _global_bge_scorer = None
    except Exception as e:
        pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Final cleanup
        try:
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass