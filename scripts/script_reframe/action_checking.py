#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import warnings
from typing import List, Dict, Tuple, Any, Optional

# Add the parent directory to sys.path to import from reproduce.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add paths for importing reranking and NLI modules
sys.path.append('/data/zyh/DeepResearch/HalluBench_backup_0828/action_checking/scripts')
sys.path.append('/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/scripts_new')
sys.path.append('/data/zyh/DeepResearch/HalluBench_backup_0828/claim_verification/top_scripts/models')
sys.path.append('/data/zyh/DeepResearch/HalluBench_backup_0828/claim_verification/top_scripts')

# Import reranking and NLI modules
try:
    from reranker_scoring import BGEScorer
    from memory_config import get_memory_config, get_gpu_batch_size
    from nli import nli_score_batch_parallel, initialize_nli_models_once
    from config import API_KEYS, BASE_URL, extract_json_from_content
    RERANKING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import reranking/NLI modules: {e}")
    RERANKING_AVAILABLE = False

warnings.filterwarnings('ignore')

    
    # Load query from input file
def load_query_from_json(file_path: str) -> str:
    """Load the query text from the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('query', '')
    except Exception as e:
        print(f"❌ Error loading query from {file_path}: {e}")
        return ""


# Import functions from other modules instead of duplicating them
try:
    from rerank_ato_query_and_action import (
        load_data_from_cache, compute_reranking_scores, analyze_query_action_coverage
    )
    from nli_against_query import (
        create_action_query_pairs, run_nli_validation, run_two_stage_judgment,
        load_structured_data_from_cache, save_results, flatten_actions_with_iterations
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import from external modules: {e}")
    IMPORT_SUCCESS = False




def process_integrated_action_checking(cache_file_path: str, input_json_path: str, output_file_path: str, 
                                     num_gpus: int = 4, gpu_ids: List[int] = None, nli_threshold: float = 0.95,
                                     mode: str = "normal") -> Dict[str, Any]:
    """
    Integrated action checking pipeline: Hallucinated Action Detection + Missed Query Detection
    
    Args:
        cache_file_path: Path to the cache file with queries and actions
        input_json_path: Path to the input JSON file with query
        output_file_path: Path to save the results
        num_gpus: Number of GPUs to use (default: 4)
        gpu_ids: Specific GPU IDs to use (e.g., [0, 1, 3]). If None, uses first num_gpus GPUs
        nli_threshold: NLI threshold for high-confidence determination (default: 0.99)
        mode: Prompt style to use when querying LLM ("normal" or "AgentDebug")
    
    Returns:
        Dict containing hallucinated_actions and missed_queries results
    """
    print(f"🚀 Starting integrated action processing pipeline (Hallucinated Action Detection + Missed Query Detection)")
    
    # Map GPU IDs if CUDA_VISIBLE_DEVICES is set
    if gpu_ids is not None and "CUDA_VISIBLE_DEVICES" in os.environ:
        # When CUDA_VISIBLE_DEVICES is set, GPU IDs are remapped to 0, 1, 2, ...
        mapped_gpu_ids = list(range(len(gpu_ids)))
        print(f"🔄 GPU ID mapping: {gpu_ids} -> {mapped_gpu_ids} (due to CUDA_VISIBLE_DEVICES)")
        gpu_ids = mapped_gpu_ids
    
    # Load query
    query = load_query_from_json(input_json_path)
    if not query:
        print("❌ No query found")
        return {"hallucinated_actions": [], "missed_queries": []}
    
    # Step 1: Load structured data
    print(f"📋 Step 1: Loading structured data...")
    if not IMPORT_SUCCESS:
        print("❌ External modules not available, returning empty results")
        return {"hallucinated_actions": [], "missed_queries": []}
    
    # Load action and claim lists from cache file
    action_lists, claim_lists = load_structured_data_from_cache(cache_file_path)
    if not action_lists:
        print("❌ No action lists found")
        return {"hallucinated_actions": [], "missed_queries": []}
    
    # Load query list for missed query detection (for reranking)
    query_list, _ = load_data_from_cache(cache_file_path)
    
    # Flatten all actions for processing while keeping iteration indices
    all_actions, action_iteration_indices = flatten_actions_with_iterations(action_lists)
    
    print(f"✅ Total actions loaded: {len(all_actions)}")
    print(f"✅ Total claim lists loaded: {len(claim_lists)}")
    
    # Helper to persist incremental results without overwriting other fields
    def write_output_file(data: Dict[str, Any]):
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

    # Load existing results if present
    existing_data: Dict[str, Any] = {}
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Failed to read existing results from {output_file_path}: {e}")
            existing_data = {}

    # Step 2: Detect hallucinated actions using NLI + LLM
    print(f"🔄 Step 2: Detecting hallucinated actions using two-stage judgment (NLI + LLM)...")
    
    hallucinated_actions_results: List[Dict[str, Any]] = []
    existing_hallucinated = existing_data.get('hallucinated_actions', {})
    skip_hallucinated = bool(existing_hallucinated.get('results'))

    if skip_hallucinated:
        print("✅ Skipping hallucinated action detection - existing hallucinated_actions found in output")
        hallucinated_actions_results = existing_hallucinated.get('results', [])
    elif RERANKING_AVAILABLE and all_actions:
        # Create action-query pairs and run NLI validation
        action_query_pairs = create_action_query_pairs(all_actions, query)
        nli_scores = run_nli_validation(action_query_pairs, num_gpus=num_gpus, gpu_ids=gpu_ids)
        
        # Run two-stage judgment process
        reranker_gpu_ids = None
        reranker_num_gpus = 4
        if gpu_ids:
            reranker_gpu_ids = gpu_ids[:4]
            reranker_num_gpus = len(reranker_gpu_ids)
        else:
            reranker_num_gpus = min(4, num_gpus)

        hallucinated_actions_results = run_two_stage_judgment(
            all_actions,
            query,
            action_lists,
            claim_lists,
            nli_scores,
            API_KEYS[0] if API_KEYS else "",
            action_iteration_indices,
            nli_threshold,
            mode=mode,
            reranker_num_gpus=reranker_num_gpus,
            reranker_gpu_ids=reranker_gpu_ids,
            cache_file_path=cache_file_path
        )
        
        hallucinated_actions_payload = {
            "total_actions": len(hallucinated_actions_results),
            "support_count": sum(1 for r in hallucinated_actions_results if r['judgment']['label'] == 'Support'),
            "not_support_count": sum(1 for r in hallucinated_actions_results if r['judgment']['label'] == 'NotSupport'),
            "nli_determined": sum(1 for r in hallucinated_actions_results if r['judgment'].get('decision_source') == 'NLI'),
            "llm_determined": sum(1 for r in hallucinated_actions_results if r['judgment'].get('decision_source') == 'LLM'),
            "results": hallucinated_actions_results
        }
        existing_data['hallucinated_actions'] = hallucinated_actions_payload
        write_output_file(existing_data)
        print(f"✅ Hallucinated action detection completed: {len(hallucinated_actions_results)} actions processed")
    else:
        print("⚠️ Skipping hallucinated action detection (modules not available or no actions)")
    
    # Step 3: Detect missed queries using reranking
    print(f"🔄 Step 3: Detecting missed queries using reranking and clustering...")
    
    missed_queries_results: Dict[str, Any] = existing_data.get('missed_queries', {})
    skip_missed_queries = bool(missed_queries_results)
    if skip_missed_queries:
        print("✅ Skipping missed query detection - existing missed_queries found in output")
    elif RERANKING_AVAILABLE and query_list and action_lists:
        # Compute reranking scores for all query-action pairs
        reranking_scores = compute_reranking_scores(query_list, action_lists, num_gpus=num_gpus, gpu_ids=gpu_ids)
        
        if reranking_scores:
            # Analyze coverage and identify missed queries
            missed_queries_results = analyze_query_action_coverage(reranking_scores)
            
            existing_data['missed_queries'] = {
                "total_queries": missed_queries_results.get('total_queries', 0),
                "missed_count": len(missed_queries_results.get('missed_queries', [])),
                "coverage_rate": missed_queries_results.get('coverage_rate', 0.0),
                "missed_queries": missed_queries_results.get('missed_queries', []),
                "first_cluster_queries": missed_queries_results.get('first_cluster_queries', {})
            }
            missed_queries_results = existing_data['missed_queries']
            write_output_file(existing_data)
            print(f"✅ Missed query detection completed: {missed_queries_results.get('total_queries', 0)} queries analyzed")
        else:
            print("⚠️ No reranking scores computed")
    else:
        print("⚠️ Skipping missed query detection (modules not available or no data)")
    
    # Step 4: Create final output
    print(f"📊 Step 4: Organizing final results...")
    
    current_hallucinated = existing_data.get('hallucinated_actions', {
        "total_actions": len(hallucinated_actions_results),
        "support_count": sum(1 for r in hallucinated_actions_results if r['judgment']['label'] == 'Support'),
        "not_support_count": sum(1 for r in hallucinated_actions_results if r['judgment']['label'] == 'NotSupport'),
        "nli_determined": sum(1 for r in hallucinated_actions_results if r['judgment'].get('decision_source') == 'NLI'),
        "llm_determined": sum(1 for r in hallucinated_actions_results if r['judgment'].get('decision_source') == 'LLM'),
        "results": hallucinated_actions_results
    })
    current_missed = existing_data.get('missed_queries', {
        "total_queries": missed_queries_results.get('total_queries', 0) if missed_queries_results else 0,
        "missed_count": len(missed_queries_results.get('missed_queries', [])) if missed_queries_results else 0,
        "coverage_rate": missed_queries_results.get('coverage_rate', 0.0) if missed_queries_results else 0.0,
        "missed_queries": missed_queries_results.get('missed_queries', []) if missed_queries_results else [],
        "first_cluster_queries": missed_queries_results.get('first_cluster_queries', {}) if missed_queries_results else {}
    })
    
    print(f"✅ Integrated processing pipeline completed:")
    print(f"  - Detected {current_hallucinated.get('total_actions', 0)} hallucinated actions")
    print(f"  - Found {len(current_missed.get('missed_queries', []))} missed queries")
    
    return {
        "hallucinated_actions": current_hallucinated,
        "missed_queries": current_missed
    }


# Standalone function for easy import (backward compatibility)
def process_actions_and_memory_new(cache_file_path: str, input_json_path: str, output_file_path: str,
                                 top_k: int = 5, num_apis: int = 20, num_gpus: int = 4, 
                                 gpu_ids: List[int] = None, nli_threshold: float = 0.95,
                                 mode: str = "normal") -> Dict[str, Any]:
    """Legacy function name for backward compatibility."""
    return process_integrated_action_checking(
        cache_file_path,
        input_json_path,
        output_file_path,
        num_gpus,
        gpu_ids,
        nli_threshold,
        mode=mode
    )

if __name__ == "__main__":
    # GPU Configuration - specify which GPUs to use
    num_gpus = 3
    gpu_ids = [0, 1, 3]  # Use specific GPUs: cuda:0, cuda:1, cuda:3
    # Alternative: gpu_ids = None  # Use first num_gpus GPUs automatically
    nli_threshold = 0.95
    
    # Force CUDA to only use specified GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"🔒 Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    cache_file_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/cache_ai_job_seeking.json"
    input_json_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/temp/ai_job_seeking.json"
    output_file_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/claim_and_action/action_checking_results_ai_job_seeking.json"
    
    print(f"🚀 Starting integrated action checking pipeline:")
    print(f"  - Hallucinated Action Detection (NLI + LLM)")
    print(f"  - Missed Query Detection (Reranking + Clustering)")
    print(f"  - GPUs: {num_gpus}, GPU IDs: {gpu_ids}, NLI Threshold: {nli_threshold}")
    print(f"  - Output: {output_file_path}")
    
    time_start = time.time()
    results = process_integrated_action_checking(
        cache_file_path,
        input_json_path,
        output_file_path,
        num_gpus,
        gpu_ids,
        nli_threshold,
        mode="normal"
    )
    time_end = time.time()
    
    print(f"\n🎉 Processing completed in {(time_end - time_start) / 60:.2f} minutes")
    print(f"\n📊 Final Results Summary:")
    print(f"  - Hallucinated Actions: {results.get('hallucinated_actions', {}).get('total_actions', 0)} total")
    print(f"    * Support: {results.get('hallucinated_actions', {}).get('support_count', 0)}")
    print(f"    * Not Support: {results.get('hallucinated_actions', {}).get('not_support_count', 0)}")
    print(f"  - Missed Queries: {results.get('missed_queries', {}).get('missed_count', 0)} out of {results.get('missed_queries', {}).get('total_queries', 0)}")
    print(f"    * Coverage Rate: {results.get('missed_queries', {}).get('coverage_rate', 0.0):.2%}")
