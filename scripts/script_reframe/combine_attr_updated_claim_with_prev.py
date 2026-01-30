#!/usr/bin/env python3
"""
Combine original results with updated claim results from misattribute rejudge.
For every claim in _modified_target_results, replace the corresponding claim in the original result file.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


def extract_file_id(filename: str) -> str:
    """Extract file ID from filename."""
    # Original: results_0b6546d8eb8ea9048175be8d0c8faed8.json
    # Updated: 0b6546d8eb8ea9048175be8d0c8faed8_modified_target_results.json
    if filename.startswith("results_"):
        return filename.replace("results_", "").replace(".json", "")
    elif filename.endswith("_modified_target_results.json"):
        return filename.replace("_modified_target_results.json", "")
    return filename.replace(".json", "")


def find_matching_claim(claim_text: str, original_claims: List[Dict]) -> int:
    """Find the index of matching claim in original claims list by claim text."""
    for idx, orig_claim in enumerate(original_claims):
        if orig_claim.get("claim") == claim_text:
            return idx
    return -1


def replace_claim_in_original(original_claim: Dict, updated_claim: Dict) -> Dict:
    """Replace original claim data with updated claim data, preserving original structure where needed."""
    # Copy the updated claim data - this will replace the original claim entirely
    # The updated claim has the correct attribution information from the rejudge process
    new_claim = updated_claim.copy()
    
    # The updated claim structure may differ slightly (e.g., sentence_idx vs chunk_index,
    # different field names in chunks), but we want to use the updated structure as it
    # contains the corrected attribution information
    return new_claim


def process_file_pair(
    original_file: Path,
    updated_file: Path,
    output_dir: Path,
    debug_info: Dict
) -> bool:
    """Process a pair of files and combine them."""
    file_id = extract_file_id(original_file.name)
    
    print(f"\n{'='*80}")
    print(f"Processing file ID: {file_id}")
    print(f"Original: {original_file.name}")
    print(f"Updated: {updated_file.name}")
    
    # Check if combined output file already exists - if so, use it as base to preserve existing attributes
    output_file = output_dir / f"{file_id}_combined.json"
    base_file = None
    if output_file.exists():
        print(f"ℹ️  Found existing combined file: {output_file.name}")
        print(f"   Will use it as base to preserve existing attributes (e.g., nd_analysis, hallucination_score)")
        base_file = output_file
    else:
        base_file = original_file
    
    # Load base file (either existing combined file or original file)
    try:
        with open(base_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load base file ({base_file}): {e}")
        debug_info["errors"].append(f"{file_id}: Failed to load base file - {e}")
        return False
    
    # Load updated file
    try:
        with open(updated_file, 'r', encoding='utf-8') as f:
            updated_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load updated file: {e}")
        debug_info["errors"].append(f"{file_id}: Failed to load updated - {e}")
        return False
    
    # Get claims from both files
    # Original: report_results[0].claim_results
    # Updated: claims (top-level)
    original_claims = []
    if "report_results" in original_data and len(original_data["report_results"]) > 0:
        original_claims = original_data["report_results"][0].get("claim_results", [])
    else:
        print(f"WARNING: No report_results found in original file")
        debug_info["warnings"].append(f"{file_id}: No report_results in original")
    
    updated_claims = updated_data.get("claims", [])
    
    print(f"Original file has {len(original_claims)} claims")
    print(f"Updated file has {len(updated_claims)} claims")
    
    # Create a copy of original data for modification
    combined_data = json.loads(json.dumps(original_data))  # Deep copy
    
    # Statistics
    stats = {
        "total_original": len(original_claims),
        "total_updated": len(updated_claims),
        "matched": 0,
        "not_found": 0,
        "replaced": 0,
        "added": 0
    }
    
    # Match and replace claims - ONLY update report_results section
    if "report_results" in combined_data and len(combined_data["report_results"]) > 0:
        combined_claims = combined_data["report_results"][0].get("claim_results", [])
        
        for updated_claim in updated_claims:
            updated_claim_text = updated_claim.get("claim", "")
            if not updated_claim_text:
                print(f"WARNING: Empty claim text in updated file")
                continue
            
            # Find matching claim in original
            orig_idx = find_matching_claim(updated_claim_text, combined_claims)
            
            if orig_idx >= 0:
                stats["matched"] += 1
                # Replace the claim
                old_claim = combined_claims[orig_idx].copy()
                combined_claims[orig_idx] = replace_claim_in_original(old_claim, updated_claim)
                stats["replaced"] += 1
                
                # Debug output for first few replacements
                if stats["replaced"] <= 3:
                    print(f"\n  ✓ Matched & Replaced claim {stats['matched']}: '{updated_claim_text[:60]}...'")
                    print(f"    Original judgment: {old_claim.get('final_judgment', 'N/A')}")
                    print(f"    Updated judgment: {updated_claim.get('final_judgment', 'N/A')}")
                    print(f"    Original chunks: {len(old_claim.get('relevant_chunks', []))}")
                    print(f"    Updated chunks: {len(updated_claim.get('relevant_chunks', []))}")
            else:
                stats["not_found"] += 1
                # Add as new claim since it's not found in original (likely an atomic/decomposed claim)
                combined_claims.append(updated_claim)
                stats["added"] += 1
                
                # Debug output for first few additions
                if stats["added"] <= 3:
                    print(f"\n  + Added new claim {stats['added']}: '{updated_claim_text[:60]}...'")
                    print(f"    Judgment: {updated_claim.get('final_judgment', 'N/A')}")
                    print(f"    Chunks: {len(updated_claim.get('relevant_chunks', []))}")
                
                debug_info["not_found_claims"].append(f"{file_id}: {updated_claim_text[:80]}")
        
        # Update the combined data
        combined_data["report_results"][0]["claim_results"] = combined_claims
    
    # Save combined result (output_file was already defined earlier)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved combined result to: {output_file.name}")
        print(f"  Statistics:")
        print(f"    - Total original claims: {stats['total_original']}")
        print(f"    - Total updated claims: {stats['total_updated']}")
        print(f"    - Matched: {stats['matched']}")
        print(f"    - Replaced: {stats['replaced']}")
        print(f"    - Added (new): {stats['added']}")
        print(f"    - Not found: {stats['not_found']}")
    except Exception as e:
        print(f"ERROR: Failed to save output file: {e}")
        debug_info["errors"].append(f"{file_id}: Failed to save - {e}")
        return False
    
    debug_info["stats"][file_id] = stats
    return True


def main():
    # Directories
    original_dir = Path("/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe")
    updated_dir = Path("/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/results")
    output_dir = Path("/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/updated_whole_results")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Debug information
    debug_info = {
        "processed_files": [],
        "errors": [],
        "warnings": [],
        "not_found_claims": [],
        "stats": {}
    }
    
    # Get all updated files
    updated_files = sorted(updated_dir.glob("*_modified_target_results.json"))
    print(f"Found {len(updated_files)} updated result files")
    
    # Process each updated file
    processed = 0
    for updated_file in updated_files:
        file_id = extract_file_id(updated_file.name)
        
        # Find corresponding original file
        original_file = original_dir / f"results_{file_id}.json"
        
        if not original_file.exists():
            print(f"WARNING: Original file not found for {file_id}: {original_file.name}")
            debug_info["warnings"].append(f"{file_id}: Original file not found")
            continue
        
        if process_file_pair(original_file, updated_file, output_dir, debug_info):
            processed += 1
            debug_info["processed_files"].append(file_id)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {processed}")
    print(f"Total errors: {len(debug_info['errors'])}")
    print(f"Total warnings: {len(debug_info['warnings'])}")
    print(f"Claims not found: {len(debug_info['not_found_claims'])}")
    
    # Print detailed stats
    if debug_info["stats"]:
        print(f"\nDetailed Statistics:")
        total_matched = sum(s["matched"] for s in debug_info["stats"].values())
        total_replaced = sum(s["replaced"] for s in debug_info["stats"].values())
        total_added = sum(s.get("added", 0) for s in debug_info["stats"].values())
        total_not_found = sum(s["not_found"] for s in debug_info["stats"].values())
        print(f"  Total matched claims: {total_matched}")
        print(f"  Total replaced claims: {total_replaced}")
        print(f"  Total added (new) claims: {total_added}")
        print(f"  Total not found (but added): {total_not_found}")
    
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
    
    # Print not found claims if any
    if debug_info["not_found_claims"]:
        print(f"\nClaims not found in original files (first 10):")
        for claim in debug_info["not_found_claims"][:10]:
            print(f"  - {claim}")
        if len(debug_info["not_found_claims"]) > 10:
            print(f"  ... and {len(debug_info['not_found_claims']) - 10} more")
    
    # Save debug info to file
    debug_file = output_dir / "_debug_info.json"
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)
    print(f"\nDebug information saved to: {debug_file}")


if __name__ == "__main__":
    main()

