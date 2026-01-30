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
                from utils import is_url
                if is_url(claim):
                    continue
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
                from utils import is_url
                if is_url(atomic_claim):
                    continue
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
    """
    Compute top_k_chunks for all claims using similarity filtering + reranking.
    
    Args:
        collected_claims: List of claim data dictionaries
        web_content: Dictionary mapping URLs to their content
        similarity_threshold: Minimum similarity score threshold
        top_k: Number of top chunks to select per claim
        num_gpus: Number of GPUs to use
        gpu_ids: List of specific GPU IDs to use
        url_mapping: Dictionary mapping URLs to their index
        
    Returns:
        Dictionary mapping each claim to its top_k_chunks
    """
    print(f"🔍 Computing top_k_chunks for {len(collected_claims)} claims using similarity filtering + reranking")
    
    # Initialize similarity filter
    similarity_filter = SimilarityFilter(
        similarity_threshold=similarity_threshold,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids
    )
    
    # Extract all unique URLs from collected claims
    all_urls = set()
    for claim_data in collected_claims:
        target_urls = claim_data.get('target_urls', [])
        all_urls.update(target_urls)
    
    print(f"📊 Processing {len(all_urls)} unique URLs for chunk extraction")
    
    # Extract chunks from all URLs
    all_chunks = []
    url_to_chunks = {}
    
    for url in all_urls:
        if url in web_content and web_content[url].strip():
            content = web_content[url]
            chunks = similarity_filter.extract_chunks_from_content(content, url)
            
            # Normalize chunk_id format to match chunk_score format: {url_index}-{chunk_id}
            if url_mapping and url in url_mapping:
                url_index = url_mapping[url]
                for chunk in chunks:
                    original_chunk_id = chunk.get('chunk_id', '')
                    # Only add prefix if not already present
                    if not original_chunk_id.startswith(f"{url_index}-"):
                        chunk['chunk_id'] = f"{url_index}-{original_chunk_id}"
            
            all_chunks.extend(chunks)
            url_to_chunks[url] = chunks
            # print(f"📄 Extracted {len(chunks)} chunks from {url}")
        else:
            print(f"⚠️ No content found for URL: {url}")
            url_to_chunks[url] = []
    
    print(f"📈 Total chunks extracted: {len(all_chunks)}")
    
    # Create claim-chunk mapping (each claim only processes its own target URLs' chunks)
    claim_chunk_mapping = {}
    
    for claim_data in collected_claims:
        claim = claim_data['claim']
        target_urls = claim_data.get('target_urls', [])
        
        # Collect chunks from this claim's target URLs only
        claim_chunks = []
        for url in target_urls:
            if url in url_to_chunks:
                claim_chunks.extend(url_to_chunks[url])
        
        claim_chunk_mapping[claim] = claim_chunks
        # print(f"📊 Claim '{claim[:100]}...' has {len(claim_chunks)} chunks from {len(target_urls)} URLs")
    
    # Apply similarity filtering + reranking
    print(f"🚀 Applying similarity filtering + reranking for {len(claim_chunk_mapping)} claims...")
    
    try:
        results = similarity_filter.filter_chunks_by_similarity_with_mapping(
            claim_chunk_mapping=claim_chunk_mapping,
            apply_reranking=True,
            top_k=top_k
        )
        
        # Extract top_k_chunks from results
        top_k_chunks = {}
        for claim, claim_data in results.items():
            if 'top_chunks' in claim_data and claim_data['top_chunks']:
                top_k_chunks[claim] = claim_data['top_chunks']
                print(f"✅ Selected {len(claim_data['top_chunks'])} top chunks for claim: {claim[:100]}...")
            else:
                top_k_chunks[claim] = []
                print(f"⚠️ No chunks selected for claim: {claim[:100]}...")
        
        print(f"✅ Successfully computed top_k_chunks for {len(top_k_chunks)} claims")
        return top_k_chunks
        
    except Exception as e:
        print(f"❌ Error in similarity filtering + reranking: {e}")
        # Return empty chunks for all claims as fallback
        return {claim_data['claim']: [] for claim_data in collected_claims}


def validate_chunks_from_target_urls(final_results: Dict[str, Any]) -> None:
    print(f"🔍 Validating that all chunks come from their claim's target_urls...")
    
    validation_errors = []
    
    for claim, result in final_results.items():
        target_urls = set(result.get('target_urls', []))
        relevant_chunks = result.get('relevant_chunks', [])
        
        for chunk in relevant_chunks:
            chunk_source_url = chunk.get('source_url', '')
            if chunk_source_url and chunk_source_url not in target_urls:
                validation_errors.append({
                    'claim': claim[:100] + "..." if len(claim) > 100 else claim,
                    'chunk_source_url': chunk_source_url,
                    'target_urls': list(target_urls),
                    'chunk_id': chunk.get('chunk_id', 'unknown')
                })
    
    if validation_errors:
        print(f"❌ VALIDATION FAILED: Found {len(validation_errors)} chunks from wrong URLs!")
        for error in validation_errors[:5]:  # Show first 5 errors
            print(f"   Claim: {error['claim']}")
            print(f"   Chunk URL: {error['chunk_source_url']}")
            print(f"   Expected URLs: {error['target_urls']}")
            print(f"   Chunk ID: {error['chunk_id']}")
            print()
        if len(validation_errors) > 5:
            print(f"   ... and {len(validation_errors) - 5} more errors")
    else:
        print(f"✅ VALIDATION PASSED: All chunks come from their claim's target_urls!")






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
    print(f"🚀 Starting parallel processing of {len(collected_claims)} claims using {num_gpus} GPUs")
    
    # Step 0: Initialize NLI models globally
    print(f"🧠 Step 0: Initializing NLI models...")
    initialize_nli_models_once(num_gpus, gpu_ids)
    
    # Step 1: Check if top_k_chunks already exist in cache
    print(f"🔍 Step 1: Checking for cached top_k_chunks...")
    
    cached_top_chunks = None
    cache_data = None
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        if 'top_k_chunks' in cache_data and cache_data['top_k_chunks']:
            cached_top_chunks = cache_data['top_k_chunks']
            print(f"✅ Found cached top_k_chunks for {len(cached_top_chunks)} claims")
        else:
            print(f"❌ No cached top_k_chunks found! Computing them now using similarity filtering + reranking...")
            
            # Compute top_k_chunks using similarity filtering + reranking
            cached_top_chunks = compute_top_k_chunks_for_claims(
                collected_claims=collected_claims,
                web_content=web_content,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                url_mapping=url_mapping
            )
            
            # Store computed top_k_chunks back to cache file
            print(f"💾 Storing computed top_k_chunks to cache file...")
            cache_data['top_k_chunks'] = cached_top_chunks
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Successfully stored top_k_chunks for {len(cached_top_chunks)} claims to cache")
            
    except Exception as e:
        print(f"❌ Error reading/writing cache file: {e}")
        return {}
    
    # Prepare claim-to-data mapping for later use
    claim_to_data_mapping = {}
    for claim_data in collected_claims:
        claim_to_data_mapping[claim_data['claim']] = claim_data
    
    # Load top_k_chunks for all claims
    claim_chunk_mapping = {}
    total_chunks = 0
    
    for claim_data in collected_claims:
        claim = claim_data['claim']
        if claim in cached_top_chunks:
            top_chunks = cached_top_chunks[claim]
            if top_chunks:
                # Convert top_k_chunks to the format expected by NLI/LLM processing
                chunks_for_this_claim = []
                for chunk in top_chunks:
                    chunk_data = {
                        'chunk_id': chunk.get('chunk_id', 'unknown'),
                        'chunk_text': chunk.get('chunk_text', ''),
                        'source_url': chunk.get('source_url', ''),
                        'chunk_index': chunk.get('position', 0),
                        'chunk_length': chunk.get('length', len(chunk.get('chunk_text', ''))),
                        'similarity_score': chunk.get('similarity_score', 0.0),
                        'rerank_score': chunk.get('rerank_score', 0.0)
                    }
                    chunks_for_this_claim.append(chunk_data)
                
                claim_chunk_mapping[claim] = chunks_for_this_claim
                total_chunks += len(chunks_for_this_claim)
                print(f"📊 Loaded {len(chunks_for_this_claim)} top chunks for claim: {claim[:100]}...")
            else:
                print(f"⚠️ No chunks found for claim: {claim[:100]}...")
                claim_chunk_mapping[claim] = []
        else:
            print(f"⚠️ No cached chunks found for claim: {claim[:100]}...")
            claim_chunk_mapping[claim] = []
    
    print(f"📈 TOP_K_CHUNKS LOADING STATS:")
    print(f"   • Total claims: {len(claim_chunk_mapping)}")
    print(f"   • Total chunks loaded: {total_chunks}")
    
    # Step 2: NLI-first then LLM processing
    print(f"🧠 Step 2: NLI-first then LLM processing")
    
    final_results = {}
    global_total_tokens = 0
    overall_start_time = time.time()
    
    # STAGE 1: NLI scoring for all claim-chunk pairs
    print(f"🧠 STAGE 1: NLI scoring for all claim-chunk pairs...")
    
    all_claim_chunk_pairs = []
    claim_chunk_indices = {}
    
    for claim, chunks in claim_chunk_mapping.items():
        claim_chunk_indices[claim] = []
        for chunk in chunks:
            pair_idx = len(all_claim_chunk_pairs)
            all_claim_chunk_pairs.append((claim, chunk['chunk_text']))
            claim_chunk_indices[claim].append(pair_idx)
    
    print(f"📊 Processing {len(all_claim_chunk_pairs)} claim-chunk pairs with NLI...")
    
    # Run NLI scoring in parallel
    nli_scores = nli_score_batch_parallel(all_claim_chunk_pairs)
    
    # STAGE 2: Process each claim with NLI results
    print(f"🧠 STAGE 2: Processing claims with NLI results...")
    
    # Collect claims that need LLM processing (not determined by NLI confidence)
    claims_needing_llm = []
    # Store all chunks_with_nli data for later access
    all_chunks_with_nli_data = {}
    
    for claim, chunk_indices in claim_chunk_indices.items():
        claim_data = claim_to_data_mapping.get(claim, {})
        chunks = claim_chunk_mapping[claim]
        
        # Prepare chunks with NLI scores
        chunks_with_nli = []
        for i, chunk_idx in enumerate(chunk_indices):
            if chunk_idx < len(nli_scores):
                chunk = chunks[i]
                nli_score = nli_scores[chunk_idx]
                
                # Determine judgment from NLI scores
                judgment_scores = {
                    'entailment': nli_score.get('entailment', 0.0),
                    'neutral': nli_score.get('neutral', 0.0),
                    'contradiction': nli_score.get('contradiction', 0.0)
                }
                
                # Get the highest scoring judgment
                max_judgment = max(judgment_scores, key=judgment_scores.get)
                max_score = judgment_scores[max_judgment]
                
                # Map NLI labels to our judgment format
                if max_judgment == 'entailment':
                    judgment = 'entailed'
                elif max_judgment == 'contradiction':
                    judgment = 'contradicted'
                else:
                    judgment = 'neutral'
                
                chunks_with_nli.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_url': chunk['source_url'],
                    'chunk_index': chunk['chunk_index'],
                    'chunk_text': chunk['chunk_text'],
                    'chunk_length': chunk['chunk_length'],
                    'judgment': judgment,
                    'confidence': max_score,
                    'nli_scores': nli_score,
                    'rerank_score': chunk.get('rerank_score', 0.0),
                    'similarity_score': chunk.get('similarity_score', 0.0)
                })
        
        # Store chunks_with_nli data for this claim
        all_chunks_with_nli_data[claim] = chunks_with_nli
        
        # Check for high confidence judgments first
        # Adapt chunks to the expected format for high confidence check
        adapted_chunks = []
        for chunk in chunks_with_nli:
            # Get the specific judgment score (entailment, contradiction, or neutral)
            chunk_nli_scores = chunk.get('nli_scores', {})  # Renamed to avoid conflict
            judgment = chunk['judgment']
            
            # Map judgment back to NLI score key
            if judgment == 'entailed':
                judgment_score = chunk_nli_scores.get('entailment', 0.0)
                nli_judgment = 'entailment'  # Map to NLI format
            elif judgment == 'contradicted':
                judgment_score = chunk_nli_scores.get('contradiction', 0.0)
                nli_judgment = 'contradiction'  # Map to NLI format
            else:  # neutral
                judgment_score = chunk_nli_scores.get('neutral', 0.0)
                nli_judgment = 'neutral'  # Map to NLI format
            
            # Create a simple sentence score structure
            adapted_chunk = {
                'chunk_id': chunk['chunk_id'],
                'source_url': chunk['source_url'],
                'top_sentence_scores': [{
                    'judgment': nli_judgment,  # Use NLI format
                    'judgment_score': judgment_score,
                    'sentence_idx': 0
                }]
            }
            adapted_chunks.append(adapted_chunk)
        
        high_conf_result = check_high_confidence_judgment(adapted_chunks)
        
        if high_conf_result:
            # Use high confidence result directly
            relevant_chunks = high_conf_result.get('relevant_chunks', [])
            processed_chunks = []
            all_nli_scores = {}
            
            # Process all relevant chunks
            for rel_chunk in relevant_chunks:
                chunk_id = rel_chunk.get('chunk_id', '')
                
                # Find corresponding chunk in chunks_with_nli to get additional data
                for chunk in chunks_with_nli:
                    if chunk.get('chunk_id') == chunk_id:
                        processed_chunk = {
                            'chunk_id': chunk_id,
                            'judgment': _to_binary_label(rel_chunk['judgment']),
                            'confidence': rel_chunk['judgment_score'],
                            'rerank_score': chunk.get('rerank_score', 0.0),
                            'similarity_score': chunk.get('similarity_score', 0.0),
                            'chunk_text': chunk.get('chunk_text', ''),
                            'source_url': rel_chunk.get('source_url', ''),
                            'sentence_idx': rel_chunk.get('sentence_idx', -1)
                        }
                        processed_chunks.append(processed_chunk)
                        
                        # Collect NLI scores for this chunk
                        if chunk.get('nli_scores'):
                            all_nli_scores[chunk_id] = chunk.get('nli_scores', {})
                        break
            
            final_results[claim] = {
                'claim': claim,
                'final_judgment': _to_binary_label(high_conf_result['judgment']),
                'relevant_chunks': processed_chunks,
                'all_judged_chunks': processed_chunks,  # For NLI high confidence, relevant chunks = all judged chunks
                'source_type': claim_data.get('source_type', 'unknown'),
                'target_urls': claim_data.get('target_urls', []),
                'total_tokens_used': 0,
                'processing_source': 'NLI_HIGH_CONFIDENCE',
                'high_confidence_skip': True,
                'nli_scores': all_nli_scores  # Save NLI scores of all relevant chunks
            }
        else:
            # This claim needs LLM processing - collect it
            claims_needing_llm.append({
                'claim': claim,
                'claim_data': claim_data,
                'chunks_with_nli': chunks_with_nli
            })
        
        # Add iteration/paragraph index if available
        if claim_data.get('source_type') == 'iteration':
            if claim in final_results:
                final_results[claim]['iteration_index'] = claim_data.get('iteration_index')
        elif claim_data.get('source_type') == 'report':
            if claim in final_results:
                final_results[claim]['paragraph_index'] = claim_data.get('paragraph_index')
    
    print(f"📊 NLI processing completed: {len(final_results)} claims determined by NLI, {len(claims_needing_llm)} claims need LLM processing")
    
    # Handle claims that were processed by NLI but didn't have high confidence (should be rare)
    nli_processed_claims = set(final_results.keys())
    all_claims = set(claim_to_data_mapping.keys())
    missing_claims = all_claims - nli_processed_claims - {info['claim'] for info in claims_needing_llm}
    
    if missing_claims:
        print(f"⚠️ Found {len(missing_claims)} claims that were processed by NLI but not included in results")
        for claim in missing_claims:
            claim_data = claim_to_data_mapping[claim]
            # These should be claims with low confidence NLI results
            final_results[claim] = {
                'claim': claim,
                'final_judgment': 'NotSupport',  # Binary default
                'relevant_chunks': [],
                'all_judged_chunks': [],  # No chunks were judged by LLM
                'source_type': claim_data.get('source_type', 'unknown'),
                'target_urls': claim_data.get('target_urls', []),
                'total_tokens_used': 0,
                'processing_source': 'NLI_LOW_CONFIDENCE',
                'nli_scores': {}  # No NLI scores for low confidence claims
            }
    
    # STAGE 3: LLM processing for claims not determined by NLI
    if claims_needing_llm:
        print(f"🧠 STAGE 3: LLM processing for {len(claims_needing_llm)} claims...")
        
        # Load cached top k docs for unprocessed claims with dynamic k selection
        from core.data_processing import load_cached_top_k_docs
        
        # Load dynamic top k chunks for each claim that needs LLM processing
        dynamic_claim_chunks = load_cached_top_k_docs(cache_file)
        
        # Prepare all chunk args for LLM processing
        all_chunk_args = []
        claim_chunk_mapping_llm = {}
        
        for claim_info in claims_needing_llm:
            claim = claim_info['claim']
            claim_data = claim_info['claim_data']
            chunks_with_nli = claim_info['chunks_with_nli']
            
            # Get dynamic top k chunks for this specific claim
            if claim in dynamic_claim_chunks:
                dynamic_chunks = dynamic_claim_chunks[claim]
                # print(f"📊 Processing claim for LLM: {claim[:100]}... with {len(dynamic_chunks)} dynamic chunks")
                
                if len(dynamic_chunks) == 0:
                    # No chunks passed dynamic k selection - set to neutral directly
                    print(f"⚠️ No chunks passed dynamic k selection for claim: {claim[:100]}... Setting to NotSupport")
                    
                    # Store result as NotSupport (binary)
                    final_results[claim] = {
                        'claim': claim,
                        'final_judgment': 'NotSupport',
                        'relevant_chunks': [],
                        'all_judged_chunks': [],  # No chunks were judged by LLM
                        'source_type': claim_data.get('source_type', 'unknown'),
                        'target_urls': claim_data.get('target_urls', []),
                        'total_tokens_used': 0,
                        'processing_source': 'NO_DYNAMIC_CHUNKS',
                        'nli_scores': {}  # No NLI scores for claims without dynamic chunks
                    }
                    
                    # Add iteration/paragraph index if available
                    if claim_data.get('source_type') == 'iteration':
                        final_results[claim]['iteration_index'] = claim_data.get('iteration_index')
                    elif claim_data.get('source_type') == 'report':
                        final_results[claim]['paragraph_index'] = claim_data.get('paragraph_index')
                    
                    print(f"✅ Set to NotSupport for claim: {claim[:100]}... (no chunks passed dynamic k selection)")
                else:
                    # Process chunks with LLM
                    for chunk_idx, chunk in enumerate(dynamic_chunks):
                        chunk_text = chunk.get('chunk_text', '')
                        chunk_id = chunk.get('chunk_id', '')
                        
                        if not chunk_text or not chunk_id:
                            print(f"⚠️ Skipping chunk with missing text or ID")
                            continue
                        
                        chunk_meta = {
                            'chunk_id': chunk_id,
                            'source_url': chunk.get('source_url', ''),
                            'chunk_index': chunk.get('chunk_index', chunk_idx),
                            'chunk_length': chunk.get('chunk_length', len(chunk_text)),
                            'rerank_score': chunk.get('rerank_score', 0.0),
                            'similarity_score': chunk.get('similarity_score', 0.0),
                            'claim': claim
                        }
                        
                        all_chunk_args.append((claim, chunk_text, query, chunk_meta))
                        
                        # Create unique chunk_id to avoid conflicts between different claims
                        unique_chunk_id = f"{hash(claim)}_{chunk_id}"
                        chunk_meta['chunk_id'] = unique_chunk_id
                        chunk_meta['original_chunk_id'] = chunk_id
                        
                        claim_chunk_mapping_llm[unique_chunk_id] = (claim, chunk_meta, chunk_idx)
            else:
                print(f"⚠️ No dynamic chunks found for claim: {claim[:100]}...")
        
        print(f"📊 Total claim-chunk pairs for LLM processing: {len(all_chunk_args)}")
        
        # Process all claim-chunk pairs in parallel using LLM
        if all_chunk_args:
            print(f"🚀 Starting batch LLM processing...")
            try:
                # Use the parallel processing function from eval_fever_for_HalluBench.py
                from evaluation.eval_fever_for_HalluBench import process_llm_judgments_emphasized_parallel
                parallel_results = process_llm_judgments_emphasized_parallel(all_chunk_args, num_cores=64)
                # for result in parallel_results:
                #     print(result['chunk_text'])
                #     print(f"🔍 LLM result: {result}")
                # print(f"✅ Batch LLM processing completed: {len(parallel_results)} results")
            except Exception as e:
                print(f"❌ Error in batch LLM processing: {e}")
                print(f"🔄 Falling back to single LLM processing...")
                parallel_results = []
                for chunk_args in all_chunk_args:
                    claim_text, chunk_text, query_text, chunk_meta = chunk_args
                    api_client = OpenAI(api_key=API_KEYS[0], base_url=BASE_URL)
                    llm_result = llm_judge_claim_with_retry(api_client, claim_text, chunk_text, query_text)
                    parallel_results.append({
                        'chunk_id': chunk_meta['chunk_id'],
                        'source_url': chunk_meta.get('source_url', ''),
                        'chunk_index': chunk_meta['chunk_index'],
                        'judgment': llm_result['judgment'],
                        'evidence': llm_result['evidence'],
                        'confidence': llm_result['confidence'],
                        'chunk_text': chunk_text,
                        'tokens_used': llm_result.get('tokens_used', 0),
                        'chunk_meta': chunk_meta
                    })
        else:
            parallel_results = []
        
        # Process LLM results and finalize judgments
        claim_results_llm = {}
        
        # Group LLM results by claim
        for result in parallel_results:
            if result and isinstance(result, dict) and not result.get('error'):
                chunk_id = result['chunk_id']
                if chunk_id in claim_chunk_mapping_llm:
                    claim, chunk_meta, chunk_idx = claim_chunk_mapping_llm[chunk_id]
                    
                    if claim not in claim_results_llm:
                        claim_results_llm[claim] = {
                            'chunk_scores': [],
                            'total_tokens': 0
                        }
                    
                    claim_results_llm[claim]['chunk_scores'].append({
                        'chunk_id': chunk_meta.get('original_chunk_id', result['chunk_id']),
                        'source_url': result.get('source_url', ''),
                        'chunk_index': result['chunk_index'],
                        'judgment': _to_binary_label(result['judgment']),
                        'evidence': result['evidence'],
                        'confidence': result['confidence'],
                        'chunk_text': result['chunk_text'],
                        'rerank_score': chunk_meta.get('rerank_score', 0),
                        'similarity_score': chunk_meta.get('similarity_score', 0)
                    })
                    claim_results_llm[claim]['total_tokens'] += result.get('tokens_used', 0)
        
        # Finalize judgments for LLM-processed claims (parallel per-claim)
        def _finalize_one(item):
            claim, claim_result = item
            claim_chunk_scores = claim_result['chunk_scores']
            claim_tokens = claim_result['total_tokens']
            claim_data = claim_to_data_mapping.get(claim, {})

            # Find the original chunks_with_nli for this claim to get NLI scores
            chunks_with_nli = None
            for claim_info in claims_needing_llm:
                if claim_info['claim'] == claim:
                    chunks_with_nli = claim_info['chunks_with_nli']
                    break

            # Use finalize_claim_judgment (not with_concat)
            final_judgment, relevant_chunks = finalize_claim_judgment(claim_chunk_scores)
            final_judgment = _to_binary_label(final_judgment)
            mapped_relevant_chunks = []
            for rc in (relevant_chunks or []):
                if isinstance(rc, dict):
                    rc_copy = {**rc}
                    if 'judgment' in rc_copy:
                        rc_copy['judgment'] = _to_binary_label(rc_copy['judgment'])
                    mapped_relevant_chunks.append(rc_copy)
                else:
                    mapped_relevant_chunks.append(rc)

            # Save ALL chunks that were judged by LLM (not just relevant chunks)
            all_judged_chunks = []
            for chunk_score in claim_chunk_scores:
                all_judged_chunks.append(chunk_score)

            result_obj = {
                'claim': claim,
                'final_judgment': final_judgment,
                'relevant_chunks': mapped_relevant_chunks,  # Keep the relevant chunks for compatibility
                'all_judged_chunks': all_judged_chunks,  # Save ALL chunks that were judged by LLM
                'source_type': claim_data.get('source_type', 'unknown'),
                'target_urls': claim_data.get('target_urls', []),
                'total_tokens_used': claim_tokens,
                'processing_source': 'LLM',
                'nli_scores': {}  # No NLI scores for LLM-processed claims
            }
            # Add iteration/paragraph index if available
            if claim_data.get('source_type') == 'iteration':
                result_obj['iteration_index'] = claim_data.get('iteration_index')
            elif claim_data.get('source_type') == 'report':
                result_obj['paragraph_index'] = claim_data.get('paragraph_index')

            return claim, result_obj, claim_tokens, final_judgment

        with futures.ThreadPoolExecutor(max_workers=min(32, len(claim_results_llm) or 1)) as executor:
            for claim, result_obj, tokens_used, final_judgment in executor.map(_finalize_one, claim_results_llm.items()):
                print(f"🔄 Finalizing judgment for LLM-processed claim: {claim[:100]}...")
                final_results[claim] = result_obj
                global_total_tokens += tokens_used
                print(f"✅ Completed LLM processing for claim: {claim[:100]}... (judgment: {final_judgment})")
    
    print(f"✅ NLI-first then LLM processing completed for {len(final_results)} claims")
    
    # Validate that all chunks come from their claim's target_urls
    validate_chunks_from_target_urls(final_results)
    
    # Display total tokens used across all claims
    print(f"💰 Total tokens consumed across all claims: {global_total_tokens}")
    print(f"💰 Average tokens per claim: {global_total_tokens / len(final_results) if final_results else 0:.1f}")
    
    # Performance summary
    overall_time = time.time() - overall_start_time
    print(f"🚀 Parallel processing performance summary:")
    print(f"   • Total claims processed: {len(final_results)}")
    print(f"   • Total tokens consumed: {global_total_tokens}")
    print(f"   • CPU cores utilized: {min(256, mp.cpu_count())}")
    print(f"   • Total processing time: {overall_time:.2f}s")
    print(f"   • Average time per claim: {overall_time / len(final_results) if final_results else 0:.2f}s")
    print(f"   • Processing completed successfully!")
    
    # Memory cleanup
    print(f"🧹 Final memory cleanup after parallel processing")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return final_results


class LLMClaimChecker:
    
    def __init__(self):
        self.context_locator = OptimizedContextLocator()
    
    def process_claims_and_urls_new(self, query: str, cache_file_path: str, output_file_path: str, summary_citations: List[str], web_content_cache: Dict[str, str], similarity_threshold: float = 0.4, top_k: int = 5, num_gpus: int = 4, gpu_ids: List[int] = None, url_mapping: Dict[str, int] = None) -> Dict[str, Any]:
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
            similarity_threshold, top_k, num_gpus, gpu_ids, url_mapping
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
def process_claims_and_urls_new(query: str, cache_file_path: str, output_file_path: str, summary_citations: List[str], web_content_cache: Dict[str, str], similarity_threshold: float = 0.4, top_k: int = 5, num_gpus: int = 4, gpu_ids: List[int] = None, url_mapping: Dict[str, int] = None) -> Dict[str, Any]:
    checker = LLMClaimChecker()
    return checker.process_claims_and_urls_new(query, cache_file_path, output_file_path, summary_citations, web_content_cache, similarity_threshold, top_k, num_gpus, gpu_ids, url_mapping)

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