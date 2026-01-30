#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import os
import logging
import json
import math
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict, Counter
import warnings
from datetime import datetime
import sys
import time

# Disable tokenizers parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import multiprocessing as mp
import concurrent.futures as futures
from openai import OpenAI
# Import from local utils module - use absolute path to avoid multiprocessing conflicts
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import with explicit module path to avoid conflicts
import importlib.util
spec = importlib.util.spec_from_file_location("local_utils", os.path.join(current_dir, "utils.py"))
local_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_utils)

# Now import the functions we need
OptimizedContextLocator = local_utils.OptimizedContextLocator
find_target_url = local_utils.find_target_url

# Import new claim verification modules
import sys
sys.path.append('/data/zyh/DeepResearch/HalluBench_backup_0828/claim_verification/top_scripts')
from models.nli import nli_score_batch_parallel, initialize_nli_models_once
from models.llm_for_HalluBench import llm_judge_claim_emphasized_with_retry as llm_judge_claim_with_retry
from core.judgment import check_high_confidence_judgment, finalize_claim_judgment
from config import API_KEYS, BASE_URL

# Import similarity filtering functionality
from similarity_filtering import SimilarityFilter

# Add the parent directory to sys.path to import from reproduce.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
Reuse common implementations from claim_checking_LLM to avoid duplication.
Only keep modified-target logic locally.
"""
from claim_checking_LLM import (
    compute_top_k_chunks_for_claims as base_compute_top_k_chunks_for_claims,
    validate_chunks_from_target_urls as base_validate_chunks_from_target_urls,
    process_claims_parallel as base_process_claims_parallel,
)


def _to_binary_label(label: str) -> str:
    if not isinstance(label, str):
        return "NotSupport"
    normalized = label.strip().lower()
    if normalized in ["support", "entailed"]:
        return "Support"
    return "NotSupport"

def collect_claims_and_urls(cache_file_path: str, summary_citations: List[str]) -> List[Dict[str, Any]]:

    print(f"📋 Collecting claims and URLs from cache file: {cache_file_path}")
    
    collected_data = []
    
    try:
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Collect claims from iterations
        iterations = cache_data.get('iterations', [])
        print(f"🔄 Processing {len(iterations)} iterations for claim collection")
        
        for i, iteration in enumerate(iterations):
            # Extract data for this iteration using numbered keys (following evaluate.py pattern)
            claim_list = iteration.get(f'claim_list_{i+1}', [])
            search_list = iteration.get(f'search_list_{i+1}', [])
            
            # If not the first iteration, add all previous search items to current search list
            if i > 0:
                for j in range(i):
                    prev_search_list = cache_data['iterations'][j].get(f'search_list_{j+1}', [])
                    search_list = search_list + [item for item in prev_search_list if item not in search_list]
            
            # Filter out claims with no relevant queries (following evaluate.py pattern)
            claim_query_mappings = cache_data.get('related_query', {})
            filtered_claims = []
            for claim in claim_list:
                if claim_query_mappings.get(claim, {}).get('relevant_queries', []) == []:
                    print(f"⚠️ No relevant queries found for claim: {claim}")
                else:
                    filtered_claims.append(claim)

            # Add each claim with its target URLs
            for claim in filtered_claims:
                collected_data.append({
                    'claim': claim,
                    'target_urls': search_list,
                    'source_type': 'iteration',
                    'iteration_index': i + 1
                })            
        
        # Collect claims from report paragraphs
        report_paragraphs = cache_data.get('report', [])
        print(f"📄 Processing {len(report_paragraphs)} report paragraphs for claim collection")
        
        for para_idx, para in enumerate(report_paragraphs):
            atomic_claims = para.get('atomic_claims', [])
            
            # Filter out claims with no relevant queries (following evaluate.py pattern)
            claim_query_mappings = cache_data.get('related_query', {})
            filtered_claims = []
            for claim in atomic_claims:
                if claim_query_mappings.get(claim, {}).get('relevant_queries', []) == []:
                    print(f"⚠️ No relevant queries found for report claim: {claim}")
                else:
                    filtered_claims.append(claim)
                
            # Collect target URLs for each claim individually
            for atomic_claim in filtered_claims:
                claim_target_urls = find_target_url(atomic_claim, atomic_claims)
                if claim_target_urls:
                    print(f"✅ Found target URLs {claim_target_urls} for report claim: {atomic_claim}")
                else:
                    print(f"⚠️ No target URLs found for report claim: {atomic_claim}")
                    claim_target_urls = summary_citations
                
                # Add each claim with its specific target URLs
                collected_data.append({
                    'claim': atomic_claim,
                    'target_urls': claim_target_urls, 
                    'source_type': 'report',
                    'paragraph_index': para_idx + 1
                })
        
        print(f"✅ Collected {len(collected_data)} claims total:")
        iteration_claims = [d for d in collected_data if d['source_type'] == 'iteration']
        report_claims = [d for d in collected_data if d['source_type'] == 'report']
        print(f"  - Iteration claims: {len(iteration_claims)}")
        print(f"  - Report claims: {len(report_claims)}")
        
        return collected_data
        
    except Exception as e:
        print(f"❌ Error collecting claims and URLs: {e}")
        return []


def compute_top_k_chunks_for_claims(collected_claims: List[Dict[str, Any]], web_content: Dict[str, str], 
                                   similarity_threshold: float = 0.4, top_k: int = 5, 
                                   num_gpus: int = 4, gpu_ids: List[int] = None, 
                                   url_mapping: Dict[str, int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Delegate to shared implementation to avoid duplication."""
    return base_compute_top_k_chunks_for_claims(
        collected_claims=collected_claims,
        web_content=web_content,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        url_mapping=url_mapping
    )


def validate_chunks_from_target_urls(final_results: Dict[str, Any]) -> None:
    """Delegate to shared implementation to avoid duplication."""
    return base_validate_chunks_from_target_urls(final_results)


def load_chunks_from_cache_standalone(cache_file: str, urls: List[str], web_content: Dict[str, str]) -> Tuple[List[str], List[Dict], Dict[str, List[int]]]:

    all_chunks = []
    chunk_metadata = []
    url_to_chunks = {}
    
    if not cache_file or not os.path.exists(cache_file):
        return all_chunks, chunk_metadata, url_to_chunks
        
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        chunk_scores = cache_data.get('chunk_score', {})
        if chunk_scores:
            print(f"✅ Loading pre-computed chunks from cache to ensure chunk ID consistency")
            
            for chunk_id, chunk_data in chunk_scores.items():
                if isinstance(chunk_data, dict) and 'chunk_text' in chunk_data:
                    # Extract URL from chunk data
                    url = chunk_data.get('url', '')
                    
                    # Filter by URLs if specified
                    if urls is not None and url not in urls:
                        continue
                    
                    chunk_idx = len(all_chunks)
                    all_chunks.append(chunk_data['chunk_text'])
                    
                    if url:
                        chunk_metadata.append({
                            'chunk_id': chunk_id,
                            'source_url': url,
                            'chunk_index': chunk_idx,
                            'chunk_length': len(chunk_data['chunk_text']),
                            'sentence_count': chunk_data.get('sentence_count', 0),
                            'sentence_indices': chunk_data.get('sentence_indices', [])
                        })
                        
                        # Track chunk indices for this URL
                        if url not in url_to_chunks:
                            url_to_chunks[url] = []
                        url_to_chunks[url].append(chunk_idx)
            
            print(f"✅ Successfully loaded {len(all_chunks)} chunks from cache for {len(urls)} specified URLs")
            
    except Exception as e:
        print(f"⚠️ Error loading chunk cache: {e}")
        
    return all_chunks, chunk_metadata, url_to_chunks


def process_claims_parallel(query: str, collected_claims: List[Dict[str, Any]], web_content: Dict[str, str], 
                           cache_file: str, similarity_threshold: float = 0.4, 
                           top_k: int = 5, num_gpus: int = 4, gpu_ids: List[int] = None, 
                           url_mapping: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Delegate to shared implementation in claim_checking_LLM.py.
    The base implementation includes:
    - NLI scoring for all claim-chunk pairs
    - First-round LLM processing with dynamic k selection
    - Second-round LLM for NotSupport claims using all cached top_k chunks
    """
    return base_process_claims_parallel(
        query=query,
        collected_claims=collected_claims,
        web_content=web_content,
        cache_file=cache_file,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        url_mapping=url_mapping
    )




class LLMClaimChecker:
    
    def __init__(self):
        self.context_locator = OptimizedContextLocator()
    
    def process_claims_and_urls_new(self, query: str, cache_file_path: str, output_file_path: str, summary_citations: List[str], web_content_cache: Dict[str, str], similarity_threshold: float = 0.4, top_k: int = 5, num_gpus: int = 4, gpu_ids: List[int] = None) -> Dict[str, Any]:
        print(f"🚀 Starting new parallel claim processing pipeline")
        
        # Step 1: Collect all claims and their target URLs
        print(f"📋 Step 1: Collecting claims and URLs...")
        collected_claims = collect_claims_and_urls(cache_file_path, summary_citations)

        # ONLY use first 5 claims for testing
        # collected_claims = collected_claims[:5]
        # ONLY process claim "Five relevant job openings have been identified at OpenAI."
        # collected_claims = [claim for claim in collected_claims if claim['claim'] == "Five relevant job openings have been identified at OpenAI."]
        # if not collected_claims:
        #     print("❌ No claims collected, returning empty results")
        #     return []
        
        # Step 2: Process all claims in parallel
        print(f"🔄 Step 2: Processing {len(collected_claims)} claims in parallel...")
        results = process_claims_parallel(
            query, collected_claims, web_content_cache, cache_file_path, 
            similarity_threshold, top_k, num_gpus, gpu_ids
        )
        
        # Organize results by source type for proper format
        chain_of_research_results = []
        report_results = []
        
        for claim, result in results.items():
            if result['source_type'] == 'iteration':
                chain_of_research_results.append(result)
            elif result['source_type'] == 'report':
                report_results.append(result)
        
        # Create the final output structure
        final_output = {
            "chain_of_research_results": [
                {
                    "claim_results": chain_of_research_results
                }
            ],
            "report_results": [
                {
                    "claim_results": report_results
                }
            ]
        }
        
        print(f"✅ Parallel processing pipeline completed. Processed {len(results)} claims:")
        print(f"  - Iteration claims: {len(chain_of_research_results)}")
        print(f"  - Report claims: {len(report_results)}")
        
        # Load existing results if file exists, otherwise create new structure
        existing_data = {}
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                print(f"📂 Loaded existing results file with keys: {list(existing_data.keys())}")
            except Exception as e:
                print(f"⚠️ Error loading existing results file: {e}")
                existing_data = {}
        
        # Merge claim verification results into existing data
        existing_data.update(final_output)
        
        # Save merged results
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        
        print(f"💾 Successfully saved merged results to {output_file_path}")
        
        return final_output


# Standalone function for easy import
def process_claims_and_urls_new(query: str, cache_file_path: str, output_file_path: str, summary_citations: List[str], web_content_cache: Dict[str, str], similarity_threshold: float = 0.4, top_k: int = 5, num_gpus: int = 4, gpu_ids: List[int] = None) -> Dict[str, Any]:
    checker = LLMClaimChecker()
    return checker.process_claims_and_urls_new(query, cache_file_path, output_file_path, summary_citations, web_content_cache, similarity_threshold, top_k, num_gpus, gpu_ids)

if __name__ == "__main__":
    cache_file_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/reframe/cache_ai_job_seeking.json"
    input_json_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/temp/ai_job_seeking.json"
    web_cache_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/web_content_cache/train_gemini/cache_ai_job_seeking.json"
    output_file_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe/claim_result_ai_job_seeking.json"
    with open(input_json_path, "r") as f:
        data = json.load(f)
    summary_citations = data["summary_citations"]
    query = data["query"]
    
    with open(web_cache_path, "r") as f:
        web_content_cache = json.load(f)
    similarity_threshold = 0.4
    top_k = 5
    num_gpus = 4
    gpu_ids = [0, 1, 2, 3]  # Use specific GPUs: cuda:0, cuda:1, cuda:3
    time_start = time.time()
    process_claims_and_urls_new(query, cache_file_path, output_file_path, summary_citations, web_content_cache, similarity_threshold, top_k, num_gpus, gpu_ids)
    time_end = time.time()
    print(f"Time taken: {(time_end - time_start) / 60} minutes")