#!/usr/bin/env python3
"""
Script to delete JSON cache files containing 'Error: Connection error.' 
and their corresponding result files.
"""

import os
import json
from pathlib import Path


def extract_file_id(cache_filename):
    """Extract file_id from cache filename (remove 'cache_' prefix and '.json' suffix)"""
    if cache_filename.startswith("cache_"):
        return cache_filename.replace("cache_", "").replace(".json", "")
    return cache_filename.replace(".json", "")


def file_contains_error(filepath, error_string="Error: Connection error."):
    """
    Check if a JSON file contains the error string.
    Uses efficient string search without loading entire file into memory.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read file in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                if error_string in chunk:
                    return True
        return False
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return False


def find_result_file(result_dir, file_id):
    """
    Find the result file for a given file_id.
    Tries both naming patterns: {file_id}_combined.json and results_{file_id}.json
    """
    result_path = Path(result_dir)
    
    # Try both naming patterns
    patterns = [
        f"{file_id}_combined.json",
        f"results_{file_id}.json"
    ]
    
    for pattern in patterns:
        result_file = result_path / pattern
        if result_file.exists():
            return result_file
    
    return None


def process_directory(cache_dir, result_dir, error_string="Error: Connection error."):
    """
    Process a directory pair: find cache files with errors and delete them along with result files.
    
    Args:
        cache_dir: Directory containing cache JSON files
        result_dir: Directory containing result JSON files
        error_string: String to search for in cache files
    """
    cache_path = Path(cache_dir)
    result_path = Path(result_dir)
    
    if not cache_path.exists():
        print(f"Warning: Cache directory does not exist: {cache_dir}")
        return []
    
    if not result_path.exists():
        print(f"Warning: Result directory does not exist: {result_dir}")
        return []
    
    deleted_files = []
    
    # Get all JSON files in cache directory
    cache_files = list(cache_path.glob("cache_*.json"))
    print(f"\nScanning {len(cache_files)} cache files in {cache_dir}...")
    
    for cache_file in cache_files:
        file_id = extract_file_id(cache_file.name)
        result_file = find_result_file(result_dir, file_id)
        
        # Check if cache file contains the error
        if file_contains_error(cache_file, error_string):
            print(f"\nFound error in: {cache_file.name}")
            print(f"  File ID: {file_id}")
            
            # Delete cache file
            try:
                cache_file.unlink()
                print(f"  ✓ Deleted cache file: {cache_file.name}")
                deleted_files.append(("cache", cache_file, file_id))
            except Exception as e:
                print(f"  ✗ Error deleting cache file {cache_file.name}: {e}")
            
            # Delete corresponding result file if it exists
            if result_file:
                try:
                    result_file.unlink()
                    print(f"  ✓ Deleted result file: {result_file.name}")
                    deleted_files.append(("result", result_file, file_id))
                except Exception as e:
                    print(f"  ✗ Error deleting result file {result_file.name}: {e}")
            else:
                print(f"  - Result file not found for file_id: {file_id}")
    
    return deleted_files


def main():
    """Main function to process all directories."""
    base_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector"
    error_string = "Error: Connection error."
    
    # Define directory pairs
    directory_pairs = [
        (
            f"{base_dir}/json_cache/benchmark/Perplexity/after_update",
            f"{base_dir}/results/benchmark/Perplexity/after_update"
        ),
        (
            f"{base_dir}/json_cache/benchmark/Perplexity/before_update",
            f"{base_dir}/results/benchmark/Perplexity/before_update"
        ),
    ]
    
    print("=" * 80)
    print("Clearing files with 'Error: Connection error.'")
    print("=" * 80)
    
    all_deleted = []
    
    for cache_dir, result_dir in directory_pairs:
        print(f"\n{'=' * 80}")
        print(f"Processing: {cache_dir}")
        print(f"{'=' * 80}")
        deleted = process_directory(cache_dir, result_dir, error_string)
        all_deleted.extend(deleted)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files deleted: {len(all_deleted)}")
    
    cache_count = sum(1 for f in all_deleted if f[0] == "cache")
    result_count = sum(1 for f in all_deleted if f[0] == "result")
    
    print(f"  - Cache files: {cache_count}")
    print(f"  - Result files: {result_count}")
    
    if all_deleted:
        print(f"\nDeleted file IDs:")
        file_ids = set(f[2] for f in all_deleted)
        for file_id in sorted(file_ids):
            print(f"  - {file_id}")
    else:
        print("\nNo files with errors found.")


if __name__ == "__main__":
    main()
