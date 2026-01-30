#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
import logging
from typing import List, Dict, Any, Tuple, Set
import concurrent.futures as futures
from collections import defaultdict

# Add the parent directory to sys.path to import from claim_checking_LLM.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import LLM judgment functions
from claim_checking_LLM import _to_binary_label
from models.llm_for_HalluBench import llm_judge_claim_emphasized_with_retry as llm_judge_claim_with_retry
from config import API_KEYS, BASE_URL
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_notsupport_claims(results_file: str) -> List[Dict[str, Any]]:
    """
    Collect all claims with final_judgment = "NotSupport" from the results file.
    
    Args:
        results_file: Path to the results JSON file
        
    Returns:
        List of claim dictionaries that have NotSupport judgment
    """
    print(f"🔍 Collecting NotSupport claims from: {results_file}")
    
    notsupport_claims = []
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Check chain_of_research_results
        chain_results = results_data.get('chain_of_research_results', [])
        for chain_result in chain_results:
            claim_results = chain_result.get('claim_results', [])
            for claim in claim_results:
                if claim.get('final_judgment') == 'NotSupport':
                    notsupport_claims.append({
                        'claim': claim,
                        'source_type': 'iteration',
                        'source_index': len(notsupport_claims)
                    })
        
        # Check report_results
        report_results = results_data.get('report_results', [])
        for report_result in report_results:
            claim_results = report_result.get('claim_results', [])
            for claim in claim_results:
                if claim.get('final_judgment') == 'NotSupport':
                    notsupport_claims.append({
                        'claim': claim,
                        'source_type': 'report',
                        'source_index': len(notsupport_claims)
                    })
        
        print(f"✅ Found {len(notsupport_claims)} NotSupport claims:")
        iteration_claims = [c for c in notsupport_claims if c['source_type'] == 'iteration']
        report_claims = [c for c in notsupport_claims if c['source_type'] == 'report']
        print(f"  - Iteration claims: {len(iteration_claims)}")
        print(f"  - Report claims: {len(report_claims)}")
        
        return notsupport_claims
        
    except Exception as e:
        print(f"❌ Error collecting NotSupport claims: {e}")
        return []



def collect_notsupport_claims_for_misattribution(results_file: str) -> List[Dict[str, Any]]:
    """
    Collect all claims with final_judgment = "NotSupport" from the results file.
    
    Args:
        results_file: Path to the results JSON file
        
    Returns:
        List of claim dictionaries that have NotSupport judgment
    """
    print(f"🔍 Collecting NotSupport claims from: {results_file}")
    
    notsupport_claims = []
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Check chain_of_research_results
        claim_results = results_data.get('claims', [])
        for claim in claim_results:
            if claim.get('final_judgment') == 'NotSupport':
                notsupport_claims.append(claim)
        
        print(f"✅ Found {len(notsupport_claims)} NotSupport claims:")
        return notsupport_claims
        
    except Exception as e:
        print(f"❌ Error collecting NotSupport claims: {e}")
        return []



def get_unprocessed_chunks_for_claim(claim_data: Dict[str, Any], cache_file: str) -> List[Dict[str, Any]]:
    """
    Get unprocessed chunks for a NotSupport claim by comparing top_k_chunks with all_judged_chunks.
    
    Args:
        claim_data: The claim data dictionary (can be wrapped or unwrapped)
        cache_file: Path to the cache file containing top_k_chunks
        
    Returns:
        List of unprocessed chunks
    """
    # Handle both wrapped and unwrapped claim data
    # Wrapped (from collect_notsupport_claims): {'claim': {claim_dict}, 'source_type': '...', 'source_index': ...}
    # Unwrapped (from collect_notsupport_claims_for_misattribution): {claim_dict} directly
    if 'source_type' in claim_data and 'source_index' in claim_data:
        # Wrapped format - extract the claim dict
        claim_obj = claim_data['claim']
    else:
        # Unwrapped format - claim_data IS the claim dict
        claim_obj = claim_data
    
    # Now claim_obj is the actual claim dict with fields: 'claim' (text), 'final_judgment', etc.
    claim_text = claim_obj['claim']  # This is the claim text string
    all_judged_chunks = claim_obj.get('all_judged_chunks', [])
    
    # Get chunk IDs that have already been judged
    judged_chunk_ids = set()
    for chunk in all_judged_chunks:
        chunk_id = chunk.get('chunk_id', '')
        source_url = chunk.get('source_url', '')
        if chunk_id and source_url:
            judged_chunk_ids.add(f"{source_url}_{chunk_id}")
    
    # print(f"🔍 Getting unprocessed chunks for claim: {claim_text[:100]}...")
    # print(f"📊 Already judged {len(judged_chunk_ids)} chunks")
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        top_k_chunks_data = cache_data.get('top_k_chunks', {})
        
        # Find the claim in top_k_chunks
        claim_chunks = None
        for cached_claim, chunks in top_k_chunks_data.items():
            if cached_claim == claim_text:
                claim_chunks = chunks
                break
        
        if not claim_chunks:
            print(f"⚠️ No top_k_chunks found for claim: {claim_text[:100]}...")
            return []
        
        # Filter out chunks that have already been judged
        unprocessed_chunks = []
        for chunk in claim_chunks:
            chunk_id = chunk.get('chunk_id', '')
            source_url = chunk.get('source_url', '')
            if chunk_id and source_url:
                if f"{source_url}_{chunk_id}" not in judged_chunk_ids:
                    unprocessed_chunks.append(chunk)
        
        # print(f"📊 Found {len(unprocessed_chunks)} unprocessed chunks out of {len(claim_chunks)} total chunks")
        
        return unprocessed_chunks
        
    except Exception as e:
        print(f"❌ Error getting unprocessed chunks: {e}")
        return []


def process_claim_chunk_pair(args: Tuple[str, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process a single claim-chunk pair with LLM judgment.
    
    Args:
        args: Tuple of (claim_text, chunk_text, query_text, chunk_meta)
        
    Returns:
        Dictionary containing the LLM judgment result
    """
    claim_text, chunk_text, query_text, chunk_meta = args
    
    try:
        # Use the first available API key
        api_client = OpenAI(api_key=API_KEYS[0], base_url=BASE_URL)
        
        # Get LLM judgment
        llm_result = llm_judge_claim_with_retry(api_client, claim_text, chunk_text, query_text)
        
        return {
            'chunk_id': chunk_meta['chunk_id'],
            'source_url': chunk_meta.get('source_url', ''),
            'chunk_index': chunk_meta.get('chunk_index', 0),
            'judgment': _to_binary_label(llm_result['judgment']),
            'evidence': llm_result['evidence'],
            'confidence': llm_result['confidence'],
            'chunk_text': chunk_text,
            'tokens_used': llm_result.get('tokens_used', 0),
            'rerank_score': chunk_meta.get('rerank_score', 0.0),
            'similarity_score': chunk_meta.get('similarity_score', 0.0),
            'claim_text': claim_text  # Add claim text for grouping
        }
        
    except Exception as e:
        print(f"❌ Error processing claim-chunk pair: {e}")
        return {
            'chunk_id': chunk_meta['chunk_id'],
            'source_url': chunk_meta.get('source_url', ''),
            'chunk_index': chunk_meta.get('chunk_index', 0),
            'judgment': 'NotSupport',  # Default to NotSupport on error
            'evidence': f"Error during processing: {str(e)}",
            'confidence': 0.0,
            'chunk_text': chunk_text,
            'tokens_used': 0,
            'rerank_score': chunk_meta.get('rerank_score', 0.0),
            'similarity_score': chunk_meta.get('similarity_score', 0.0),
            'claim_text': claim_text,  # Add claim text for grouping
            'error': str(e)
        }


def finalize_claim_judgment_from_chunks(chunk_scores: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Finalize the judgment for a claim based on its chunk scores.
    This follows the same pattern as in claim_checking_LLM.py.
    
    Args:
        chunk_scores: List of chunk score dictionaries
        
    Returns:
        Tuple of (final_judgment, relevant_chunks)
    """
    if not chunk_scores:
        return 'NotSupport', []
    
    # Count judgments
    judgment_counts = defaultdict(int)
    for chunk in chunk_scores:
        judgment = chunk.get('judgment', 'NotSupport')
        judgment_counts[judgment] += 1
    
    # Determine final judgment: Support if at least one chunk supports the claim
    total_chunks = len(chunk_scores)
    support_count = judgment_counts.get('Support', 0)
    notsupport_count = judgment_counts.get('NotSupport', 0)
    
    # If at least one chunk supports the claim, it's Support
    if support_count > 0:
        final_judgment = 'Support'
    else:
        final_judgment = 'NotSupport'
    
    # Select relevant chunks: only chunks that match the final judgment
    if final_judgment == 'Support':
        # For Support claims, only include chunks that are also Support
        relevant_chunks = [chunk for chunk in chunk_scores if chunk.get('judgment') == 'Support']
    else:
        # For NotSupport claims, include all chunks (or could be refined further)
        relevant_chunks = chunk_scores
    
    # print(f"📊 Final judgment: {final_judgment} (Support: {support_count}/{total_chunks}, NotSupport: {notsupport_count}/{total_chunks})")
    
    return final_judgment, relevant_chunks


def collect_all_claim_chunk_pairs(
    notsupport_claims: List[Dict[str, Any]], 
    cache_file: str, 
    query: str
) -> Tuple[List[Tuple[str, str, str, Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    """
    Collect all claim-chunk pairs that need to be processed in parallel.
    
    Args:
        notsupport_claims: List of NotSupport claim data
        cache_file: Path to cache file containing top_k_chunks
        query: Original query text
        
    Returns:
        Tuple of (all_chunk_args, claim_to_original_data_mapping)
    """
    print(f"🔍 Collecting all claim-chunk pairs for parallel processing...")
    
    all_chunk_args = []
    claim_to_original_data_mapping = {}
    
    for claim_data in notsupport_claims:
        # Handle both wrapped and unwrapped claim data
        # Wrapped (from collect_notsupport_claims): {'claim': {claim_dict}, 'source_type': '...', 'source_index': ...}
        # Unwrapped (from collect_notsupport_claims_for_misattribution): {claim_dict} directly
        if 'source_type' in claim_data and 'source_index' in claim_data:
            # Wrapped format - extract the claim dict
            claim_obj = claim_data['claim']
        else:
            # Unwrapped format - claim_data IS the claim dict
            claim_obj = claim_data
        
        # Now claim_obj is the actual claim dict with fields: 'claim' (text), 'final_judgment', etc.
        claim_text = claim_obj['claim']  # This is the claim text string
        # print(f"📋 Collecting chunks for claim: {claim_text[:100]}...")
        
        # Store original claim data for later use (the unwrapped claim dict)
        claim_to_original_data_mapping[claim_text] = claim_obj
        
        # Get unprocessed chunks for this claim
        unprocessed_chunks = get_unprocessed_chunks_for_claim(claim_data, cache_file)
        
        if not unprocessed_chunks:
            print(f"⚠️ No unprocessed chunks found for claim: {claim_text[:100]}...")
            continue
        
        # Prepare chunk arguments for this claim
        for chunk in unprocessed_chunks:
            chunk_meta = {
                'chunk_id': chunk.get('chunk_id', ''),
                'source_url': chunk.get('source_url', ''),
                'chunk_index': chunk.get('position', 0),
                'rerank_score': chunk.get('rerank_score', 0.0),
                'similarity_score': chunk.get('similarity_score', 0.0),
                'claim': claim_text  # Add claim text to metadata for grouping later
            }
            all_chunk_args.append((claim_text, chunk.get('chunk_text', ''), query, chunk_meta))
        
        # print(f"📊 Added {len(unprocessed_chunks)} chunks for claim: {claim_text[:100]}...")
    
    print(f"📈 Total claim-chunk pairs collected: {len(all_chunk_args)}")
    return all_chunk_args, claim_to_original_data_mapping


def rejudge_notsupport_claims_parallel(
    notsupport_claims: List[Dict[str, Any]], 
    cache_file: str, 
    query: str,
    num_cores: int = 64
) -> Dict[str, Dict[str, Any]]:
    """
    Re-judge all NotSupport claims against their unprocessed chunks in parallel.
    
    Args:
        notsupport_claims: List of NotSupport claim data
        cache_file: Path to cache file containing top_k_chunks
        query: Original query text
        num_cores: Number of CPU cores to use for parallel processing
        
    Returns:
        Dictionary mapping claim text to updated claim results
    """
    print(f"🚀 Starting parallel re-judgment of {len(notsupport_claims)} NotSupport claims")
    
    # Step 1: Collect all claim-chunk pairs upfront
    print(f"\n{'='*50}")
    print("STEP 1: COLLECTING ALL CLAIM-CHUNK PAIRS")
    print(f"{'='*50}")
    
    all_chunk_args, claim_to_original_data_mapping = collect_all_claim_chunk_pairs(
        notsupport_claims, cache_file, query
    )
    
    if not all_chunk_args:
        print("⚠️ No claim-chunk pairs to process. Returning original results.")
        # Handle both wrapped and unwrapped formats
        result = {}
        for claim_data in notsupport_claims:
            if 'source_type' in claim_data and 'source_index' in claim_data:
                # Wrapped format: extract claim dict
                claim_obj = claim_data['claim']
                claim_text = claim_obj['claim']
                result[claim_text] = claim_obj
            else:
                # Unwrapped format: claim_data IS the claim dict
                claim_text = claim_data['claim']
                result[claim_text] = claim_data
        return result
    
    # Step 2: Process all claim-chunk pairs in parallel
    print(f"\n{'='*50}")
    print("STEP 2: PROCESSING ALL CLAIM-CHUNK PAIRS IN PARALLEL")
    print(f"{'='*50}")
    
    print(f"🚀 Processing {len(all_chunk_args)} claim-chunk pairs using {num_cores} cores...")
    
    parallel_results = []
    if all_chunk_args:
        with futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
            parallel_results = list(executor.map(process_claim_chunk_pair, all_chunk_args))
    
    print(f"✅ Parallel processing completed: {len(parallel_results)} results")
    
    # Step 3: Group results by claim and finalize judgments
    print(f"\n{'='*50}")
    print("STEP 3: GROUPING RESULTS AND FINALIZING JUDGMENTS")
    print(f"{'='*50}")
    
    # Group results by claim
    claim_results = defaultdict(list)
    total_tokens_used = 0
    
    for result in parallel_results:
        if result and isinstance(result, dict) and not result.get('error'):
            claim_text = result.get('claim_text', '')
            if claim_text:
                claim_results[claim_text].append(result)
                total_tokens_used += result.get('tokens_used', 0)
    
    # Finalize judgments for each claim
    updated_results = {}
    
    for claim_text in claim_to_original_data_mapping.keys():
        
        original_claim_data = claim_to_original_data_mapping[claim_text]
        new_chunk_results = claim_results.get(claim_text, [])
        
        # Combine with existing judged chunks
        existing_chunks = original_claim_data.get('all_judged_chunks', [])
        all_chunk_scores = existing_chunks + new_chunk_results
        
        # Finalize judgment
        final_judgment, relevant_chunks = finalize_claim_judgment_from_chunks(all_chunk_scores)
        
        # Calculate tokens used for this claim
        claim_tokens = sum(result.get('tokens_used', 0) for result in new_chunk_results)
        
        # Create updated claim result
        updated_claim_result = {
            'claim': claim_text,
            'final_judgment': final_judgment,
            'relevant_chunks': relevant_chunks,
            'all_judged_chunks': all_chunk_scores,
            'source_type': original_claim_data.get('source_type', 'unknown'),
            'target_urls': original_claim_data.get('target_urls', []),
            'total_tokens_used': original_claim_data.get('total_tokens_used', 0) + claim_tokens,
            'processing_source': 'SECOND_TURN_LLM',
            'nli_scores': original_claim_data.get('nli_scores', {}),
            'second_turn_processed': True,
            'second_turn_chunks_processed': len(new_chunk_results)
        }
        
        # Add iteration/paragraph index if available
        if original_claim_data.get('iteration_index') is not None:
            updated_claim_result['iteration_index'] = original_claim_data['iteration_index']
        if original_claim_data.get('paragraph_index') is not None:
            updated_claim_result['paragraph_index'] = original_claim_data['paragraph_index']
        
        updated_results[claim_text] = updated_claim_result
        
        # print(f"✅ Updated judgment for claim: {claim_text[:100]}... (was: NotSupport, now: {final_judgment})")
    
    print(f"💰 Total tokens used in second turn: {total_tokens_used}")
    print(f"✅ Completed re-judgment of {len(updated_results)} claims")
    
    return updated_results


def update_results_file(results_file: str, updated_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Update the results file with the new judgments for NotSupport claims.
    
    Args:
        results_file: Path to the results JSON file
        updated_results: Dictionary mapping claim text to updated claim results
    """
    print(f"💾 Updating results file: {results_file}")
    
    try:
        # Load existing results
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        updated_count = 0
        
        # Update chain_of_research_results
        chain_results = results_data.get('chain_of_research_results', [])
        for chain_result in chain_results:
            claim_results = chain_result.get('claim_results', [])
            for i, claim in enumerate(claim_results):
                claim_text = claim.get('claim', '')
                if claim_text in updated_results:
                    claim_results[i] = updated_results[claim_text]
                    updated_count += 1
                    print(f"✅ Updated chain claim: {claim_text[:100]}...")
        
        # Update report_results
        report_results = results_data.get('report_results', [])
        for report_result in report_results:
            claim_results = report_result.get('claim_results', [])
            for i, claim in enumerate(claim_results):
                claim_text = claim.get('claim', '')
                if claim_text in updated_results:
                    claim_results[i] = updated_results[claim_text]
                    updated_count += 1
                    # print(f"✅ Updated report claim: {claim_text[:100]}...")
        
        # Save updated results
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Successfully updated {updated_count} claims in results file")
        
    except Exception as e:
        print(f"❌ Error updating results file: {e}")


def main():
    """
    Main function to execute the second turn LLM processing for NotSupport claims.
    """
    start_time = time.time()
    
    # Set HF_ENDPOINT environment variable for Hugging Face mirror
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"🔧 Set HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    
    # File paths (adjust these based on your specific case)
    results_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe/results_3a6f5321868afb24193d6755ff10a551.json"
    cache_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/reframe/cache_3a6f5321868afb24193d6755ff10a551.json"
    
    # Load query from results file
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        query = results_data.get('query', '')
        print(f"📝 Loaded query: {query[:100]}...")
    except Exception as e:
        print(f"❌ Error loading query from results file: {e}")
        return
    
    # Step 1: Collect all NotSupport claims
    print(f"\n{'='*60}")
    print("STEP 1: COLLECTING NOTSUPPORT CLAIMS")
    print(f"{'='*60}")
    
    notsupport_claims = collect_notsupport_claims(results_file)

    # Use the first 5 claims for testing
    # notsupport_claims = notsupport_claims[:5]
    
    if not notsupport_claims:
        print("✅ No NotSupport claims found. Nothing to process.")
        return
    
    # Step 2: Re-judge NotSupport claims against unprocessed chunks
    print(f"\n{'='*60}")
    print("STEP 2: RE-JUDGING NOTSUPPORT CLAIMS")
    print(f"{'='*60}")
    
    updated_results = rejudge_notsupport_claims_parallel(
        notsupport_claims=notsupport_claims,
        cache_file=cache_file,
        query=query,
        num_cores=64
    )
    
    # Step 3: Update results file
    print(f"\n{'='*60}")
    print("STEP 3: UPDATING RESULTS FILE")
    print(f"{'='*60}")
    
    update_results_file(results_file, updated_results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    changed_judgments = 0
    for claim_text, result in updated_results.items():
        if result.get('final_judgment') != 'NotSupport':
            changed_judgments += 1
            print(f"🔄 Changed: {claim_text[:100]}... -> {result.get('final_judgment')}")
    
    print(f"📊 Total NotSupport claims processed: {len(updated_results)}")
    print(f"📊 Claims with changed judgments: {changed_judgments}")
    print(f"📊 Claims remaining NotSupport: {len(updated_results) - changed_judgments}")
    
    end_time = time.time()
    print(f"⏱️ Total processing time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
