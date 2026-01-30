#!/usr/bin/env python3
"""
Script to correct chunk_id format in cache and result files.
Correct format: {url_index}-chunk_{chunk_index}
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

# Define the four directories
CACHE1_DIR = '/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/Tianyu_sampled/gemini/after_update'
CACHE2_DIR = '/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/Tianyu_sampled/gemini/before_update'
RESULT1_DIR = '/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Tianyu_sampled/gemini/after_update'
RESULT2_DIR = '/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Tianyu_sampled/gemini/before_update'


def extract_url_index_mapping(cache_data: dict) -> Dict[str, int]:
    """
    Extract URL to url_index mapping from chunk_score in cache file.
    
    Args:
        cache_data: The loaded JSON data from cache file
        
    Returns:
        Dictionary mapping URL to url_index
    """
    url_to_index = {}
    
    if "chunk_score" not in cache_data:
        return url_to_index
    
    chunk_score = cache_data["chunk_score"]
    
    for chunk_key, chunk_info in chunk_score.items():
        # Parse chunk_key format: "{url_index}-chunk_{chunk_index}"
        match = re.match(r'^(\d+)-chunk_\d+$', chunk_key)
        if match:
            url_index = int(match.group(1))
            url = chunk_info.get("url")
            if url:
                url_to_index[url] = url_index
    
    return url_to_index


def needs_correction(chunk_id: str) -> bool:
    """
    Check if chunk_id needs correction.
    Correct format: {url_index}-chunk_{chunk_index}
    Incorrect format: chunk_{chunk_index} (missing url_index prefix)
    """
    # If it doesn't start with a digit, it needs correction
    return not re.match(r'^\d+-chunk_\d+$', chunk_id)


def correct_chunk_id(chunk_id: str, source_url: str, url_mapping: Dict[str, int]) -> Optional[str]:
    """
    Correct chunk_id by adding url_index prefix based on source_url.
    
    Args:
        chunk_id: Current chunk_id (may be incorrect)
        source_url: The source URL for this chunk
        url_mapping: Mapping from URL to url_index
        
    Returns:
        Corrected chunk_id or None if correction not possible
    """
    if not needs_correction(chunk_id):
        return None  # Already correct
    
    # Find url_index from mapping
    url_index = url_mapping.get(source_url)
    if url_index is None:
        return None  # Cannot find mapping
    
    # Extract chunk_index from current chunk_id
    match = re.match(r'^chunk_(\d+)$', chunk_id)
    if not match:
        return None  # Unexpected format
    
    chunk_index = match.group(1)
    return f"{url_index}-chunk_{chunk_index}"


def correct_cache_file(cache_path: str, url_mapping: Dict[str, int]) -> bool:
    """
    Correct chunk_id in cache file's top_k_chunks.
    
    Returns:
        True if any corrections were made, False otherwise
    """
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {cache_path}: {e}")
        return False
    
    corrected = False
    
    if "top_k_chunks" in data:
        top_k_chunks = data["top_k_chunks"]
        for claim, chunks in top_k_chunks.items():
            if not isinstance(chunks, list):
                continue
            
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                
                chunk_id = chunk.get("chunk_id")
                source_url = chunk.get("source_url")
                
                if chunk_id and source_url:
                    corrected_id = correct_chunk_id(chunk_id, source_url, url_mapping)
                    if corrected_id:
                        chunk["chunk_id"] = corrected_id
                        corrected = True
    
    if corrected:
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Corrected cache file: {cache_path}")
            return True
        except Exception as e:
            print(f"Error saving {cache_path}: {e}")
            return False
    
    return False


def correct_result_file(result_path: str, url_mapping: Dict[str, int]) -> bool:
    """
    Correct chunk_id in result file's chain_of_research_results and report_results.
    
    Returns:
        True if any corrections were made, False otherwise
    """
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {result_path}: {e}")
        return False
    
    corrected = False
    
    def correct_chunks_in_list(chunks_list):
        """Helper to correct chunks in a list"""
        nonlocal corrected
        if not isinstance(chunks_list, list):
            return
        
        for chunk in chunks_list:
            if not isinstance(chunk, dict):
                continue
            
            chunk_id = chunk.get("chunk_id")
            source_url = chunk.get("source_url")
            
            if chunk_id and source_url:
                corrected_id = correct_chunk_id(chunk_id, source_url, url_mapping)
                if corrected_id:
                    chunk["chunk_id"] = corrected_id
                    corrected = True
    
    # Correct chain_of_research_results
    if "chain_of_research_results" in data:
        chain_results = data["chain_of_research_results"]
        if isinstance(chain_results, list):
            for result in chain_results:
                if not isinstance(result, dict):
                    continue
                
                if "claim_results" in result:
                    for claim_result in result["claim_results"]:
                        if not isinstance(claim_result, dict):
                            continue
                        
                        # Correct relevant_chunks
                        if "relevant_chunks" in claim_result:
                            correct_chunks_in_list(claim_result["relevant_chunks"])
                        
                        # Correct all_judged_chunks
                        if "all_judged_chunks" in claim_result:
                            correct_chunks_in_list(claim_result["all_judged_chunks"])
    
    # Correct report_results
    if "report_results" in data:
        report_results = data["report_results"]
        if isinstance(report_results, list):
            for result in report_results:
                if not isinstance(result, dict):
                    continue
                
                if "claim_results" in result:
                    for claim_result in result["claim_results"]:
                        if not isinstance(claim_result, dict):
                            continue
                        
                        # Correct relevant_chunks
                        if "relevant_chunks" in claim_result:
                            correct_chunks_in_list(claim_result["relevant_chunks"])
                        
                        # Correct all_judged_chunks
                        if "all_judged_chunks" in claim_result:
                            correct_chunks_in_list(claim_result["all_judged_chunks"])
    
    if corrected:
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Corrected result file: {result_path}")
            return True
        except Exception as e:
            print(f"Error saving {result_path}: {e}")
            return False
    
    return False


def get_file_id_from_cache_path(cache_path: str) -> Optional[str]:
    """Extract file ID from cache file path (e.g., cache_0b6546d8eb8ea9048175be8d0c8faed8.json -> 0b6546d8eb8ea9048175be8d0c8faed8)"""
    match = re.search(r'cache_([a-f0-9]+)\.json$', cache_path)
    if match:
        return match.group(1)
    return None


def get_file_id_from_result_path(result_path: str) -> Optional[str]:
    """Extract file ID from result file path (e.g., 0b6546d8eb8ea9048175be8d0c8faed8_combined.json -> 0b6546d8eb8ea9048175be8d0c8faed8)"""
    match = re.search(r'([a-f0-9]+)_combined\.json$', result_path)
    if match:
        return match.group(1)
    return None


def process_file_group(file_id: str):
    """
    Process a group of four files with the same file_id.
    """
    print(f"\nProcessing file group: {file_id}")
    
    # Find cache files
    cache1_path = os.path.join(CACHE1_DIR, f"cache_{file_id}.json")
    cache2_path = os.path.join(CACHE2_DIR, f"cache_{file_id}.json")
    
    # Find result files
    result1_path = os.path.join(RESULT1_DIR, f"{file_id}_combined.json")
    result2_path = os.path.join(RESULT2_DIR, f"{file_id}_combined.json")
    
    # Try to get mapping from cache1 first, then cache2
    url_mapping = {}
    cache_file_used = None
    
    for cache_path in [cache1_path, cache2_path]:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                url_mapping = extract_url_index_mapping(cache_data)
                if url_mapping:
                    cache_file_used = cache_path
                    print(f"  Using mapping from: {cache_path} ({len(url_mapping)} URLs)")
                    break
            except Exception as e:
                print(f"  Error reading {cache_path}: {e}")
                continue
    
    if not url_mapping:
        print(f"  Warning: Could not extract URL mapping for {file_id}, skipping...")
        return
    
    # Correct cache files
    # for cache_path in [cache1_path, cache2_path]:
    #     if os.path.exists(cache_path):
    #         correct_cache_file(cache_path, url_mapping)
    
    # Correct result files
    for result_path in [result1_path, result2_path]:
        if os.path.exists(result_path) and result_path.endswith('Art_1_combined.json'):
            print(f"  Correcting result file: {result_path}")
            correct_result_file(result_path, url_mapping)


def main():
    """Main function to process all files."""
    print("Starting chunk_id correction...")
    
    # Collect all file IDs from cache1 directory
    file_ids = set()
    
    if os.path.exists(CACHE1_DIR):
        for filename in os.listdir(CACHE1_DIR):
            if filename.startswith("cache_") and filename.endswith(".json"):
                file_id = get_file_id_from_cache_path(os.path.join(CACHE1_DIR, filename))
                if file_id:
                    file_ids.add(file_id)
    
    # Also collect from cache2 directory
    if os.path.exists(CACHE2_DIR):
        for filename in os.listdir(CACHE2_DIR):
            if filename.startswith("cache_") and filename.endswith(".json"):
                file_id = get_file_id_from_cache_path(os.path.join(CACHE2_DIR, filename))
                if file_id:
                    file_ids.add(file_id)
    
    print(f"Found {len(file_ids)} unique file IDs to process")
    
    # Process each file group
    for file_id in sorted(file_ids):
        process_file_group(file_id)
    
    print("\nCorrection completed!")


if __name__ == "__main__":
    main()

