#!/usr/bin/env python3
"""
Script to filter misaligned related claims from result JSON files.

This script identifies two types of problematic claims:
1. Claims with < 3 target URLs and final judgment NotSupport
2. Claims with > 2 target URLs, final judgment Support, but all supporting chunks 
   are from all_source_links and not in summary_citations
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str) -> None:
    """Save JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def check_type1_claim(claim: Dict) -> bool:
    """
    Check if claim is Type 1: target_urls < 3 and final_judgment is NotSupport.
    
    Args:
        claim: The claim dictionary
    
    Returns:
        True if the claim matches Type 1 criteria
    """
    target_urls = claim.get('target_urls', [])
    final_judgment = claim.get('final_judgment', '')
    
    return len(target_urls) < 3 and final_judgment == 'NotSupport'


def check_type2_claim(claim: Dict, all_source_links: List[str], summary_citations: List[str]) -> bool:
    """
    Check if claim is Type 2: target_urls > 2, final_judgment is Support,
    and all supporting chunks are from all_source_links but NOT in summary_citations.
    
    Args:
        claim: The claim dictionary
        all_source_links: All source links from raw input
        summary_citations: Summary citations from raw input
    
    Returns:
        True if the claim matches Type 2 criteria
    """
    target_urls = claim.get('target_urls', [])
    final_judgment = claim.get('final_judgment', '')
    all_judged_chunks = claim.get('all_judged_chunks', [])
    
    # Must have > 2 target URLs and final judgment Support
    if len(target_urls) <= 2 or final_judgment != 'Support':
        return False
    
    # Find all chunks with Support judgment
    support_chunks = [chunk for chunk in all_judged_chunks if chunk.get('judgment') == 'Support']
    
    # If no support chunks, return False
    if not support_chunks:
        return False
    
    # Convert to sets for faster lookup
    all_source_links_set = set(all_source_links)
    summary_citations_set = set(summary_citations)
    
    # Check if ALL support chunks are from all_source_links but NOT in summary_citations
    for chunk in support_chunks:
        source_url = chunk.get('source_url', '')
        
        # If source_url is not in all_source_links, return False
        if source_url not in all_source_links_set:
            return False
        
        # If source_url IS in summary_citations, return False (we want ONLY from all_source_links, NOT in summary_citations)
        if source_url in summary_citations_set:
            return False
    
    # All support chunks are from all_source_links and NOT in summary_citations
    return True


def process_result_file(result_path: str, raw_path: str) -> Dict[str, Any]:
    """
    Process a single result file and its corresponding raw input file.
    
    Args:
        result_path: Path to the result JSON file
        raw_path: Path to the raw input JSON file
    
    Returns:
        Dictionary containing filtered claims
    """
    # Load both files
    result_data = load_json(result_path)
    raw_data = load_json(raw_path)
    
    # Get all_source_links and summary_citations from raw data
    all_source_links = raw_data.get('all_source_links', [])
    summary_citations = raw_data.get('summary_citations', [])
    
    # Initialize output structure
    filtered_data = {
        'query': result_data.get('query', ''),
        'report': result_data.get('report', ''),
        'type1_claims': [],  # < 3 URLs and NotSupport
        'type2_claims': [],  # > 2 URLs, Support, but only from all_source_links (not in summary_citations)
        'statistics': {
            'type1_count': 0,
            'type2_count': 0
        }
    }
    
    # Process chain_of_research_results
    for iteration in result_data.get('chain_of_research_results', []):
        for claim in iteration.get('claim_results', []):
            # Check Type 1
            if check_type1_claim(claim):
                filtered_data['type1_claims'].append(claim)
                filtered_data['statistics']['type1_count'] += 1
            
            # Check Type 2
            elif check_type2_claim(claim, all_source_links, summary_citations):
                filtered_data['type2_claims'].append(claim)
                filtered_data['statistics']['type2_count'] += 1
    
    # Process report_results
    for paragraph in result_data.get('report_results', []):
        for claim in paragraph.get('claim_results', []):
            # Check Type 1
            if check_type1_claim(claim):
                filtered_data['type1_claims'].append(claim)
                filtered_data['statistics']['type1_count'] += 1
            
            # Check Type 2
            elif check_type2_claim(claim, all_source_links, summary_citations):
                filtered_data['type2_claims'].append(claim)
                filtered_data['statistics']['type2_count'] += 1
    
    return filtered_data


def main():
    """Main function to process all result files."""
    # Define paths
    result_dir = Path('/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe')
    raw_dir = Path('/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/Tianyu_ReportBench/json_new_summary_cite')
    output_dir = Path('/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/filtered_misaligned_claims')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track overall statistics
    overall_stats = {
        'total_files_processed': 0,
        'total_type1_claims': 0,
        'total_type2_claims': 0,
        'files_with_type1': 0,
        'files_with_type2': 0
    }
    
    # Process all result files
    result_files = sorted(result_dir.glob('results_*.json'))
    
    print(f"Found {len(result_files)} result files to process")
    
    for result_file in result_files:
        # Extract the hash from filename (results_<hash>.json -> <hash>.json)
        filename = result_file.name
        hash_name = filename.replace('results_', '')
        
        # Construct raw file path
        raw_file = raw_dir / hash_name
        
        # Check if raw file exists
        if not raw_file.exists():
            print(f"Warning: Raw file not found for {filename}, skipping")
            continue
        
        try:
            # Process the file pair
            filtered_data = process_result_file(str(result_file), str(raw_file))
            
            # Only save if there are any filtered claims
            if filtered_data['statistics']['type1_count'] > 0 or filtered_data['statistics']['type2_count'] > 0:
                output_file = output_dir / f"filtered_{hash_name}"
                save_json(filtered_data, str(output_file))
                
                # Update overall statistics
                if filtered_data['statistics']['type1_count'] > 0:
                    overall_stats['files_with_type1'] += 1
                if filtered_data['statistics']['type2_count'] > 0:
                    overall_stats['files_with_type2'] += 1
                
                print(f"Processed {filename}: Type1={filtered_data['statistics']['type1_count']}, Type2={filtered_data['statistics']['type2_count']}")
            
            # Update overall statistics
            overall_stats['total_files_processed'] += 1
            overall_stats['total_type1_claims'] += filtered_data['statistics']['type1_count']
            overall_stats['total_type2_claims'] += filtered_data['statistics']['type2_count']
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Save overall statistics
    stats_file = output_dir / 'overall_statistics.json'
    save_json(overall_stats, str(stats_file))
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files processed: {overall_stats['total_files_processed']}")
    print(f"Total Type 1 claims (< 3 URLs + NotSupport): {overall_stats['total_type1_claims']}")
    print(f"Total Type 2 claims (> 2 URLs + Support + only from all_source_links): {overall_stats['total_type2_claims']}")
    print(f"Files with Type 1 claims: {overall_stats['files_with_type1']}")
    print(f"Files with Type 2 claims: {overall_stats['files_with_type2']}")
    print(f"\nOutput directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

