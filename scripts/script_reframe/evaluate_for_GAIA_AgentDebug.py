import argparse
from fileinput import filename
import json
import os
from typing import List, Dict, Any
import logging
import time
import concurrent.futures
import shutil

# Set CUDA_VISIBLE_DEVICES BEFORE any imports to restrict GPU usage
import sys
if len(sys.argv) > 1 and '--gpu_ids' in sys.argv:
    try:
        gpu_ids_idx = sys.argv.index('--gpu_ids') + 1
        if gpu_ids_idx < len(sys.argv):
            gpu_ids_str = sys.argv[gpu_ids_idx]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
            print(f"🎯 Set CUDA_VISIBLE_DEVICES to: {gpu_ids_str}")
    except (ValueError, IndexError):
        pass

from decomposition import decompose_workflow_to_cache_auto, decompose_report_to_cache_auto, decompose_query
from claim_checking_LLM import process_claims_and_urls_new
from action_checking import process_actions_and_memory_new
from overall_noise_domination import noise_domination_detection
from process_filtered_misaligned_with_modified_urls import process_single_file as process_filtered_misaligned_single_file, collect_from_filtered_files, compute_modified_targets, load_web_cache as load_web_cache_for_filtered, get_summary_citations as get_summary_citations_for_filtered
from judge_HC_against_memory import MemoryJudge
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sepcify process name
import setproctitle
setproctitle.setproctitle('Yuhao_evaluate')

# Global BGEScorer instance to avoid loading reranker models multiple times
# This ensures that the expensive reranker model loading happens only once,
# and the same instance is reused across all scoring operations
_global_bge_scorer = None


def ensure_parent_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _has_memory_processing(result_path: str) -> bool:
    """
    Return True if the result file already contains memory-based judgments.
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
                if claim.get("processing_source") in ("Memory_LLM", "Memory_NLI"):
                    return True
        return False

    return (
        _claims_have_memory(data.get("chain_of_research_results")) or
        _claims_have_memory(data.get("report_results"))
    )


def get_global_bge_scorer(num_gpus=4, gpu_ids=None):
    global _global_bge_scorer
    if _global_bge_scorer is None:
        print("🔧 Initializing global BGEScorer (this will load reranker models once)...")
        print("🎯 This is the ONLY time you'll see 'initial target device' messages!")
        from reranker_scoring import BGEScorer
        _global_bge_scorer = BGEScorer(num_gpus=num_gpus, gpu_ids=gpu_ids)
        print("✅ Global BGEScorer initialized successfully")
    return _global_bge_scorer


def monitor_memory_usage():
    try:
        import psutil
        import torch
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"💾 System Memory: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"🎮 GPU {i}: {allocated:.2f}GB / {total:.2f}GB used")
                
                # Warning if GPU memory is high
                if allocated / total > 0.8:
                    print(f"⚠️ GPU {i} memory usage is high ({allocated/total*100:.1f}%)")
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024**3
        print(f"🔄 Process Memory: {process_memory:.2f}GB")
        
    except ImportError:
        print("📊 Memory monitoring not available (psutil not installed)")


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
    print(f"🌐 Created global URL mapping for {len(all_urls)} URLs")
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
        print(f"✅ Skipping chunk scoring because cache file already exists: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
            if 'chunk_score' in all_content:
                return all_content['chunk_score']
            else:
                print(f"❌ No chunk scores found in cache file: {cache_file}")


    print(f"\n{'='*50}")
    print("SCORING ALL CHUNKS UPFRONT")
    print(f"{'='*50}")
    
    # Memory check before scoring
    print(f"💾 Memory status before chunk scoring:")
    monitor_memory_usage()
    
    # Initialize the global scorer with the pre-loaded BGEScorer instance
    print("🔧 Initializing IntegratedChunkScorer for upfront scoring...")
    print("📝 Note: Using global BGEScorer instance to avoid reloading reranker models")
    print("🚀 This ensures 'initial target device' appears only once!")
    
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
    
    print(f"📊 Scoring {len(sub_queries)} queries against chunks from {len(all_urls)} URLs...")
    print(f"💾 Web content cache contains {len(web_content_cache)} documents")
    
    # Score all chunks against all queries
    print(f"🚀 Starting upfront scoring of all chunks...")
    print(f"💾 Using pre-loaded BGEScorer instance (no model reloading)")
    scoring_results = scorer.score_chunks_sync(
        queries=sub_queries,
        urls=all_urls,  # Use all_urls to score everything upfront
        web_content_cache=web_content_cache,
        cache_file=cache_file,
        url_mapping=url_mapping
    )
    
    print(f"✅ Upfront scoring completed!")
    print(f"📊 Scored {len(scoring_results.get('detailed_chunk_scores', []))} chunks")
    print(f"📊 Results saved to cache file: {cache_file}")
    
    # Memory check after scoring
    print(f"💾 Memory status after chunk scoring:")
    monitor_memory_usage()
    
    # Memory cleanup after scoring
    print(f"🧹 Memory cleanup after upfront scoring")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return scoring_results


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
                print(f"✅ Skipping claim-to-query mapping because 'related_query' already exists in cache")
                return all_content['related_query']
        except Exception as e:
            print(f"⚠️ Error checking cache for claim-query mappings: {e}")
    
    print(f"\n{'='*50}")
    print("MAPPING ALL CLAIMS TO QUERIES UPFRONT")
    print(f"{'='*50}")
    
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
    
    print(f"📊 Found {len(unique_claims)} unique claims to map")
    
    # Get query list from cache
    query_list = cache_data.get('query_list', [])
    print(f"📊 Mapping against {len(query_list)} queries")
    
    # Map all claims to their related queries in one call
    print(f"  📝 Mapping {len(unique_claims)} claims to queries...")
    
    try:
        # Find relevant queries for all claims at once
        selected_queries = find_relevant_queries_for_claims(unique_claims, query_list, fixed_threshold=0.0001)
        
        # Convert the result format to use claim text as keys
        claim_query_mappings = {}
        for claim_idx, claim_info in selected_queries.items():
            claim_text = claim_info['claim']
            claim_query_mappings[claim_text] = claim_info
        
        print(f"✅ Completed mapping {len(claim_query_mappings)} claims to queries")
        
    except Exception as e:
        print(f"⚠️ Error mapping claims to queries: {e}")
        claim_query_mappings = {}
    
    # Save the mappings to cache file
    # The cache structure will now include:
    # - chunk_score: Pre-computed chunk scores
    # - related_query: Pre-computed claim-to-query mappings
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
        
        print(f"💾 Claim-query mappings saved to cache file: {cache_file}")
        print(f"📊 Cache now contains both chunk scores and claim-query mappings")
        
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
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use (default: 3)')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3", help='Comma-separated GPU IDs to use (e.g., "0,1,3"). Default: "0,1,3"')
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
    web_content_cache_file = f"../web_content_cache/GAIA_AgentDebug/cache_{os.path.basename(args.input_json)}"
    
    # Create output file path
    output_file = f"../results/GAIA_AgentDebug/before_update/{os.path.basename(args.input_json).replace('.json', '')}_combined.json"
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

    cache_file = f"../json_cache/GAIA_AgentDebug/before_update/cache_{os.path.basename(args.input_json)}"
    
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
    
    print(f"\n🔄 Scoring all chunks upfront...")
    # scoring_results = score_all_chunks_upfront(sub_queries, all_urls, web_content_cache, cache_file, url_mapping, args.num_gpus, gpu_ids)

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
    print(f"\n🔄 Second-turn rejudging for NotSupport claims...")
    
    try:
        # Import the second-turn rejudging function
        from second_turn_LLM_for_NotSupport import collect_notsupport_claims, rejudge_notsupport_claims_parallel, update_results_file
        
        # Collect all NotSupport claims from current results
        notsupport_claims = collect_notsupport_claims(output_file)
        
        if notsupport_claims:
            print(f"📊 Found {len(notsupport_claims)} NotSupport claims for second-turn rejudging")
            
            # Re-judge NotSupport claims against unprocessed chunks
            updated_claim_results = rejudge_notsupport_claims_parallel(
                notsupport_claims=notsupport_claims,
                cache_file=cache_file,
                query=query,
                num_cores=64
            )
            
            # Update the results file with new judgments
            update_results_file(output_file, updated_claim_results)
            
            # Count changed judgments
            changed_judgments = 0
            for claim_text, result in updated_claim_results.items():
                if result.get('final_judgment') != 'NotSupport':
                    changed_judgments += 1
            
            print(f"✅ Second-turn rejudging completed:")
            print(f"  - Total NotSupport claims processed: {len(updated_claim_results)}")
            print(f"  - Claims with changed judgments: {changed_judgments}")
            print(f"  - Claims remaining NotSupport: {len(updated_claim_results) - changed_judgments}")
            
            # Reload the updated results for subsequent processing
            with open(output_file, 'r', encoding='utf-8') as f:
                updated_data = json.load(f)
            
            # Update final_results with the updated claim results
            if 'chain_of_research_results' in updated_data:
                final_results['chain_of_research_results'] = updated_data['chain_of_research_results']
            if 'report_results' in updated_data:
                final_results['report_results'] = updated_data['report_results']
            
            print(f"✅ Updated final results with second-turn rejudging results")
        else:
            print(f"✅ No NotSupport claims found for second-turn rejudging")
            
    except Exception as e:
        print(f"❌ Error in second-turn rejudging: {e}")
        import traceback
        traceback.print_exc()

    
    # Processing filtered misaligned claims is disabled (deprecated step).

    print(f"\n🔄 Judging NotSupport claims against memory...")
    memory_judge_summary = None
    try:
        if not os.path.exists(output_file):
            print(f"⚠️ Skipping memory judgment - result file missing: {output_file}")
        else:
            file_id_for_memory = os.path.basename(args.input_json).replace('.json', '')
            results_dir = os.path.dirname(output_file)
            cache_dir = os.path.dirname(cache_file)
            combined_output_path = output_file

            if _has_memory_processing(combined_output_path):
                print(f"✅ Skipping memory judgment - Memory_LLM/NLI already present in {combined_output_path}")
            else:
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
                if memory_judge_summary.get('total_support') == -1:
                    print(f"⚠️ Skipping memory judgment - it has been processed")
                elif memory_judge_summary:
                    final_results['memory_judgment_summary'] = memory_judge_summary
                    print(f"✅ Memory judgment completed: relabeled {memory_judge_summary.get('total_relabeled', 0)} claims")

            if os.path.exists(combined_output_path):
                with open(combined_output_path, 'r', encoding='utf-8') as f:
                    memory_updated_data = json.load(f)
                for key in ('chain_of_research_results', 'report_results', 'memory_updates'):
                    if key in memory_updated_data:
                        final_results[key] = memory_updated_data[key]
    except Exception as e:
        print(f"❌ Error during memory judgment: {e}")

    # Judge the noise domination
    # print(f"\n🔄 Judging the noise domination...")
    
    # Check if noise domination results already exist and are not empty
    # skip_noise_domination = False
    # if os.path.exists(output_file):
    #     with open(output_file, 'r', encoding='utf-8') as f:
    #         existing_data = json.load(f)
    #     nd_analysis = existing_data.get('nd_analysis', {})
    #     hallucination_score = existing_data.get('hallucination_score')
        
    #     # Check if nd_analysis is non-empty (has meaningful data)
    #     if nd_analysis and (
    #         nd_analysis.get('total_nd', 0) > 0 or 
    #         nd_analysis.get('unmapped_cluster_details', []) or
    #         hallucination_score is not None
    #     ):
    #         skip_noise_domination = True
    #         print(f"✅ Skipping noise domination - non-empty nd_analysis already exists")
            
    #         # Merge ALL existing noise domination results
    #         nd_fields = [
    #             'hallucination_score', 'nd_analysis', 'clusters', 'mapped_clusters', 
    #             'unmapped_clusters', 'total_clusters', 'total_mapped', 'total_unmapped',
    #             'total_entailed_chunks', 'entailed_iteration_analysis', 
    #             'mapped_unmapped_feature_comparison', 'iteration_statistics', 
    #             'citation_statistics', 'iteration_level_nd_analysis'
    #         ]
            
    #         nd_fields_merged = []
    #         for field in nd_fields:
    #             if field in existing_data:
    #                 final_results[field] = existing_data[field]
    #                 nd_fields_merged.append(field)
            
    #         print(f"✅ Existing noise domination results merged into final results: {', '.join(nd_fields_merged)}")
    
    # if not skip_noise_domination:
    #     try:
    #         noise_domination_detection(output_file, cache_file, args.input_json, args.num_gpus, gpu_ids)
    #         # Read the updated results file to get ALL noise domination results
    #         with open(output_file, 'r', encoding='utf-8') as f:
    #             updated_results = json.load(f)
            
    #         # Merge ALL ND-related fields into final results
    #         nd_fields = [
    #             'hallucination_score', 'nd_analysis', 'clusters', 'mapped_clusters', 
    #             'unmapped_clusters', 'total_clusters', 'total_mapped', 'total_unmapped',
    #             'total_entailed_chunks', 'entailed_iteration_analysis', 
    #             'mapped_unmapped_feature_comparison', 'iteration_statistics', 
    #             'citation_statistics', 'iteration_level_nd_analysis'
    #         ]
            
    #         nd_fields_merged = []
    #         for field in nd_fields:
    #             if field in updated_results:
    #                 final_results[field] = updated_results[field]
    #                 nd_fields_merged.append(field)
            
    #         print(f"✅ Noise domination results merged into final results: {', '.join(nd_fields_merged)}")
    #     except Exception as e:
    #         print(f"❌ Error in noise domination detection: {e}")

    # Judge the actions
    print(f"\n🔄 Judging the actions...")
    action_results = None
    
    # Check if action results already exist and are not empty
    skip_action_checking = False
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        hallucinated_actions = existing_data.get('hallucinated_actions', {})
        
        # Check if hallucinated_actions is non-empty
        if hallucinated_actions and hallucinated_actions.get('total_actions', 0) > 0:
            skip_action_checking = True
            print(f"✅ Skipping action checking - non-empty hallucinated_actions already exist")
            action_results = existing_data
    
    if not skip_action_checking:
        try:
            action_results = process_actions_and_memory_new(
                cache_file,
                args.input_json,
                output_file,
                top_k=10,
                num_gpus=args.num_gpus,
                gpu_ids=gpu_ids,
                mode="AgentDebug"
            )
            if action_results:
                # Merge action results into final results
                final_results.update(action_results)
                print(f"✅ Action results merged into final results")
        except Exception as e:
            print(f"❌ Error in action processing: {e}")
            action_results = None
    else:
        # Use existing action results
        if action_results:
            final_results.update(action_results)
            print(f"✅ Existing action results merged into final results")

    # Update summary with processed counts
    if claim_results and 'chain_of_research_results' in claim_results:
        final_results['summary']['processed_iterations'] = len(claim_results.get('chain_of_research_results', []))
    if claim_results and 'report_results' in claim_results:
        final_results['summary']['processed_paragraphs'] = len(claim_results.get('report_results', []))

    # Save the complete final results
    print(f"\n💾 Saving complete final results to: {output_file}")
    ensure_parent_dir(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Complete results saved successfully")

    # Final memory status
    print(f"\n💾 Final memory status:")
    monitor_memory_usage()
    
    print(f"\n✅ Complete results saved to: {output_file}")
    
    # Read final file to show summary
    with open(output_file, 'r', encoding='utf-8') as f:
        final_data = json.load(f)
    
    print(f"📊 Final Summary:")
    if 'summary' in final_data:
        print(f"  - Chain of Research iterations: {final_data['summary']['total_iterations']}")
        print(f"  - Report paragraphs: {final_data['summary']['total_paragraphs']}")
        print(f"  - Successfully processed iterations: {final_data['summary']['processed_iterations']}")
        print(f"  - Successfully processed paragraphs: {final_data['summary']['processed_paragraphs']}")
    else:
        print(f"  - Summary section not found in results file")
        
        # Calculate and display hallucination degree metrics
        print(f"\n🎯 HALLUCINATION DEGREE METRICS:")
        print(f"{'='*60}")
        
        # Type 1: NotSupported claim ratio from process_claims_and_urls_new
        print(f"📊 Type 1 - Claim Verification Results:")
        if claim_results and 'chain_of_research_results' in claim_results:
            chain_claims = claim_results['chain_of_research_results'][0]['claim_results'] if claim_results['chain_of_research_results'] else []
            report_claims = claim_results['report_results'][0]['claim_results'] if claim_results['report_results'] else []
            all_claims = chain_claims + report_claims
            
            total_claims = len(all_claims)
            not_supported_claims = sum(1 for claim in all_claims if claim.get('final_judgment') == 'contradicted')
            not_supported_ratio = not_supported_claims / total_claims if total_claims > 0 else 0
            
            print(f"  - Total claims processed: {total_claims}")
            print(f"  - NotSupported claims: {not_supported_claims}")
            print(f"  - NotSupported claim ratio: {not_supported_ratio:.4f} ({not_supported_ratio*100:.2f}%)")
        else:
            print(f"  - No claim results available")
        
        # Type 2: NWIS and ND ratios from noise_domination_detection
        print(f"\n📊 Type 2 - Noise Domination Analysis:")
        if 'hallucination_score' in final_data and 'nd_analysis' in final_data:
            nwis_score = final_data['hallucination_score']
            nd_analysis = final_data['nd_analysis']
            
            print(f"  - NWIS Score: {nwis_score:.4f}")
            print(f"  - Document-level ND (All) ratio: {nd_analysis.get('document_level_nd_all_ratio', 0):.4f} ({nd_analysis.get('document_level_nd_all_ratio', 0)*100:.2f}%)")
            print(f"  - Document-level ND (Partial) ratio: {nd_analysis.get('document_level_nd_partial_ratio', 0):.4f} ({nd_analysis.get('document_level_nd_partial_ratio', 0)*100:.2f}%)")
            print(f"  - Chunk-level ND ratio: {nd_analysis.get('chunk_level_nd_ratio', 0):.4f} ({nd_analysis.get('chunk_level_nd_ratio', 0)*100:.2f}%)")
            
            # Calculate average ND ratio
            avg_nd_ratio = (nd_analysis.get('document_level_nd_all_ratio', 0) + 
                           nd_analysis.get('document_level_nd_partial_ratio', 0) + 
                           nd_analysis.get('chunk_level_nd_ratio', 0)) / 3
            print(f"  - Average ND ratio: {avg_nd_ratio:.4f} ({avg_nd_ratio*100:.2f}%)")
        else:
            print(f"  - No ND analysis available")
        
        # Type 3: Hallucinated actions and query coverage from process_actions_and_memory_new
        print(f"\n📊 Type 3 - Action Verification Results:")
        if action_results and 'hallucinated_actions' in action_results and 'missed_queries' in action_results:
            hallucinated_actions = action_results['hallucinated_actions']
            missed_queries = action_results['missed_queries']
            
            total_actions = hallucinated_actions.get('total_actions', 0)
            not_support_actions = hallucinated_actions.get('not_support_count', 0)
            hallucinated_action_ratio = not_support_actions / total_actions if total_actions > 0 else 0
            
            total_queries = missed_queries.get('total_queries', 0)
            missed_query_count = missed_queries.get('missed_count', 0)
            coverage_rate = missed_queries.get('coverage_rate', 0)
            
            print(f"  - Total actions processed: {total_actions}")
            print(f"  - NotSupport actions: {not_support_actions}")
            print(f"  - Hallucinated action ratio: {hallucinated_action_ratio:.4f} ({hallucinated_action_ratio*100:.2f}%)")
            print(f"  - Total queries analyzed: {total_queries}")
            print(f"  - Missed queries: {missed_query_count}")
            print(f"  - Query coverage rate: {coverage_rate:.4f} ({coverage_rate*100:.2f}%)")
        else:
            print(f"  - No action results available")
        
        print(f"\n🎯 SUMMARY OF HALLUCINATION DEGREES:")
        print(f"{'='*60}")
        if final_results and 'chain_of_research_results' in final_results:
            chain_claims = final_results['chain_of_research_results'][0]['claim_results'] if final_results['chain_of_research_results'] else []
            report_claims = final_results['report_results'][0]['claim_results'] if final_results['report_results'] else []
            all_claims = chain_claims + report_claims
            total_claims = len(all_claims)
            not_supported_claims = sum(1 for claim in all_claims if claim.get('final_judgment') == 'NotSupport')
            not_supported_ratio = not_supported_claims / total_claims if total_claims > 0 else 0
            print(f"Type 1 - NotSupported claim ratio (after second-turn rejudging): {not_supported_ratio:.4f} ({not_supported_ratio*100:.2f}%)")
        
        if 'hallucination_score' in final_data and 'nd_analysis' in final_data:
            nwis_score = final_data['hallucination_score']
            nd_analysis = final_data['nd_analysis']
            avg_nd_ratio = (nd_analysis.get('document_level_nd_all_ratio', 0) + 
                           nd_analysis.get('document_level_nd_partial_ratio', 0) + 
                           nd_analysis.get('chunk_level_nd_ratio', 0)) / 3
            print(f"Type 2 - NWIS Score: {nwis_score:.4f}, Average ND ratio: {avg_nd_ratio:.4f} ({avg_nd_ratio*100:.2f}%)")
        
        if final_results and 'hallucinated_actions' in final_results and 'missed_queries' in final_results:
            hallucinated_actions = final_results['hallucinated_actions']
            missed_queries = final_results['missed_queries']
            total_actions = hallucinated_actions.get('total_actions', 0)
            not_support_actions = hallucinated_actions.get('not_support_count', 0)
            hallucinated_action_ratio = not_support_actions / total_actions if total_actions > 0 else 0
            coverage_rate = missed_queries.get('coverage_rate', 0)
            print(f"Type 3 - Hallucinated action ratio: {hallucinated_action_ratio:.4f} ({hallucinated_action_ratio*100:.2f}%), Query coverage rate: {coverage_rate:.4f} ({coverage_rate*100:.2f}%)")

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 60} minutes")
    
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
                # Try to properly close the reranker
                if hasattr(_global_bge_scorer, 'stop_multi_process_pool'):
                    _global_bge_scorer.stop_multi_process_pool()
                elif hasattr(_global_bge_scorer, 'stop_self_pool'):
                    _global_bge_scorer.stop_self_pool()
            except Exception as e:
                print(f"⚠️ Warning: Error during reranker cleanup: {e}")
            finally:
                _global_bge_scorer = None
        
        print("🧹 Cleanup completed")
    except Exception as e:
        print(f"⚠️ Warning: Error during cleanup: {e}")


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