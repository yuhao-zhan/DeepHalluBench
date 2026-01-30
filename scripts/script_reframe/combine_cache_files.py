#!/usr/bin/env python3
"""
Combine original cache files with updated cache files.
Only merge the top_k_chunks field - if a claim appears in the updated cache file,
replace the original entry with the updated one.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


def extract_file_id(filename: str) -> str:
    """Extract file ID from filename."""
    # Both: cache_0b6546d8eb8ea9048175be8d0c8faed8.json
    if filename.startswith("cache_"):
        return filename.replace("cache_", "").replace(".json", "")
    return filename.replace(".json", "")


def process_cache_pair(
    original_file: Path,
    updated_file: Path,
    output_dir: Path,
    debug_info: Dict
) -> bool:
    """Process a pair of cache files and combine their top_k_chunks."""
    file_id = extract_file_id(original_file.name)
    
    print(f"\n{'='*80}")
    print(f"Processing cache file ID: {file_id}")
    print(f"Original: {original_file.name}")
    print(f"Updated: {updated_file.name}")
    
    # Load original cache file
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load original cache file: {e}")
        debug_info["errors"].append(f"{file_id}: Failed to load original - {e}")
        return False
    
    # Load updated cache file
    try:
        with open(updated_file, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load updated cache file: {e}")
        debug_info["errors"].append(f"{file_id}: Failed to load updated - {e}")
        return False
    
    # Get top_k_chunks from both files
    original_top_k = original_data.get("top_k_chunks", {})
    updated_top_k = updated_data.get("top_k_chunks", {})
    
    print(f"Original top_k_chunks has {len(original_top_k)} claims")
    print(f"Updated top_k_chunks has {len(updated_top_k)} claims")
    
    # Create a copy of original data for modification
    combined_data = json.loads(json.dumps(original_data))  # Deep copy
    
    # Statistics
    stats = {
        "total_original_claims": len(original_top_k),
        "total_updated_claims": len(updated_top_k),
        "replaced": 0,
        "added": 0,
        "kept_original": 0
    }
    
    # Merge top_k_chunks - ONLY update this field
    combined_top_k = combined_data.get("top_k_chunks", {})
    
    # First, keep all original claims
    for claim_key in original_top_k:
        if claim_key not in updated_top_k:
            # Keep original if not in updated
            stats["kept_original"] += 1
        else:
            # Replace with updated version
            combined_top_k[claim_key] = updated_top_k[claim_key]
            stats["replaced"] += 1
            
            if stats["replaced"] <= 3:
                print(f"\n  ✓ Replaced claim: '{claim_key[:60]}...'")
                print(f"    Original chunks: {len(original_top_k[claim_key])}")
                print(f"    Updated chunks: {len(updated_top_k[claim_key])}")
    
    # Add new claims from updated that don't exist in original
    for claim_key in updated_top_k:
        if claim_key not in original_top_k:
            combined_top_k[claim_key] = updated_top_k[claim_key]
            stats["added"] += 1
            
            if stats["added"] <= 3:
                print(f"\n  + Added new claim: '{claim_key[:60]}...'")
                print(f"    Chunks: {len(updated_top_k[claim_key])}")
    
    # Update the combined data - ONLY top_k_chunks field
    combined_data["top_k_chunks"] = combined_top_k
    
    # Save combined cache file
    output_file = output_dir / f"cache_{file_id}.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved combined cache to: {output_file.name}")
        print(f"  Statistics:")
        print(f"    - Original claims: {stats['total_original_claims']}")
        print(f"    - Updated claims: {stats['total_updated_claims']}")
        print(f"    - Replaced: {stats['replaced']}")
        print(f"    - Added (new): {stats['added']}")
        print(f"    - Kept original: {stats['kept_original']}")
        print(f"    - Total after merge: {len(combined_top_k)}")
    except Exception as e:
        print(f"ERROR: Failed to save output file: {e}")
        debug_info["errors"].append(f"{file_id}: Failed to save - {e}")
        return False
    
    debug_info["stats"][file_id] = stats
    return True


def main():
    # Directories
    original_dir = Path("/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/reframe")
    updated_dir = Path("/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/cache")
    output_dir = Path("/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/updated_whole_results")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug information
    debug_info = {
        "processed_files": [],
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Get all updated cache files
    updated_files = sorted(updated_dir.glob("cache_*.json"))
    print(f"Found {len(updated_files)} updated cache files")
    
    # Process each updated cache file
    processed = 0
    for updated_file in updated_files:
        file_id = extract_file_id(updated_file.name)
        
        # Find corresponding original cache file
        original_file = original_dir / f"cache_{file_id}.json"
        
        if not original_file.exists():
            print(f"WARNING: Original cache file not found for {file_id}: {original_file.name}")
            debug_info["warnings"].append(f"{file_id}: Original cache file not found")
            continue
        
        if process_cache_pair(original_file, updated_file, output_dir, debug_info):
            processed += 1
            debug_info["processed_files"].append(file_id)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total cache files processed: {processed}")
    print(f"Total errors: {len(debug_info['errors'])}")
    print(f"Total warnings: {len(debug_info['warnings'])}")
    
    # Print detailed stats
    if debug_info["stats"]:
        print(f"\nDetailed Statistics:")
        total_replaced = sum(s["replaced"] for s in debug_info["stats"].values())
        total_added = sum(s["added"] for s in debug_info["stats"].values())
        total_kept = sum(s["kept_original"] for s in debug_info["stats"].values())
        print(f"  Total replaced claims: {total_replaced}")
        print(f"  Total added (new) claims: {total_added}")
        print(f"  Total kept original claims: {total_kept}")
    
    # Print errors if any
    if debug_info["errors"]:
        print(f"\nErrors:")
        for error in debug_info["errors"][:10]:  # First 10
            print(f"  - {error}")
        if len(debug_info["errors"]) > 10:
            print(f"  ... and {len(debug_info['errors']) - 10} more errors")
    
    # Print warnings if any
    if debug_info["warnings"]:
        print(f"\nWarnings:")
        for warning in debug_info["warnings"][:10]:  # First 10
            print(f"  - {warning}")
        if len(debug_info["warnings"]) > 10:
            print(f"  ... and {len(debug_info['warnings']) - 10} more warnings")
    
    # Save debug info to file
    debug_file = output_dir / "_cache_debug_info.json"
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)
    print(f"\nDebug information saved to: {debug_file}")


if __name__ == "__main__":
    main()

