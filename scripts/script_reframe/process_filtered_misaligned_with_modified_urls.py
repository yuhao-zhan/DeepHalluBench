#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process filtered misaligned claims with modified target URLs.
For each claim: target_urls = summary_citations - original_target_urls (from filtered file)
"""

import os
import json
import glob
from typing import List, Dict, Any
from collections import defaultdict

# Import the existing pipeline
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from claim_checking_LLM_modified_target_url import process_claims_parallel
from second_turn_LLM_for_NotSupport import collect_notsupport_claims_for_misattribution, rejudge_notsupport_claims_parallel, update_results_file
import setproctitle
setproctitle.setproctitle("Yuhao_misalignment")


def collect_from_filtered_files(filtered_dir: str) -> List[Dict[str, Any]]:
    """Collect claims from all filtered JSON files."""
    print(f"📋 Collecting claims from filtered files in: {filtered_dir}")
    
    all_claims = []
    filtered_files = glob.glob(os.path.join(filtered_dir, "filtered_*.json"))
    print(f"📂 Found {len(filtered_files)} filtered files")
    
    for filtered_file in filtered_files:
        filename = os.path.basename(filtered_file)
        file_id = filename.replace('filtered_', '').replace('.json', '')
        
        with open(filtered_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        query = data.get('query', '')
        type1_claims = data.get('type1_claims', [])
        
        print(f"📄 {filename}: {len(type1_claims)} type1_claims")
        
        # Collect each claim with metadata
        for claim_data in type1_claims:
            all_claims.append({
                'file_id': file_id,
                'claim': claim_data.get('claim', ''),
                'original_target_urls': claim_data.get('target_urls', []),
                'query': query
            })
    
    print(f"✅ Collected {len(all_claims)} claims from {len(filtered_files)} files")
    return all_claims


def compute_modified_targets(summary_citations: List[str], original_targets: List[str]) -> List[str]:
    """Compute modified target URLs: summary_citations - original_targets"""
    summary_set = set(summary_citations)
    original_set = set(original_targets)
    return list(summary_set - original_set)


def check_second_turn_completed(results_file: str) -> bool:
    """Check if second-turn judging has already been completed for all NotSupport claims."""
    if not os.path.exists(results_file):
        return False
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    claims = results_data.get('claims', [])
    notsupport_claims = [c for c in claims if c.get('final_judgment') == 'NotSupport']
    
    if not notsupport_claims:
        return True
    
    # Check if all NotSupport claims have been processed in second turn
    return all(claim.get('second_turn_processed', False) for claim in notsupport_claims)


def update_results_file_for_misattribution(results_file: str, updated_results: Dict[str, Dict[str, Any]]) -> None:
    """Update the results file with the new judgments for NotSupport claims."""
    print(f"💾 Updating results file: {results_file}")
    
    # Load existing results
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    # Update claims in the 'claims' field
    claims = results_data.get('claims', [])
    updated_count = 0
    for i, claim in enumerate(claims):
        claim_text = claim.get('claim', '')
        if claim_text in updated_results:
            claims[i] = updated_results[claim_text]
            updated_count += 1
            print(f"✅ Updated claim: {claim_text[:100]}...")
    
    results_data['claims'] = claims
    
    # Save updated results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Successfully updated {updated_count} claims")


def load_web_cache(web_cache_file: str) -> Dict[str, str]:
    """Load web content cache from file path."""
    if os.path.exists(web_cache_file):
        with open(web_cache_file, 'r', encoding='utf-8') as f:
            web_content = json.load(f)
        print(f"✅ Loaded web cache: {len(web_content)} URLs")
        return web_content
    
    print(f"⚠️ Web cache file not found: {web_cache_file}")
    return {}


def get_summary_citations(raw_json_file: str) -> List[str]:
    """Get summary_citations from the original data file."""
    if os.path.exists(raw_json_file):
        with open(raw_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary_citations = data.get('all_source_links', [])
        print(f"✅ Loaded {len(summary_citations)} summary_citations")
        return summary_citations
    
    print(f"⚠️ Raw JSON file not found: {raw_json_file}")
    return []


def process_single_file(file_id: str, claims: List[Dict], summary_citations: List[str], 
                       web_content: Dict[str, str], output_dir: str,
                       similarity_threshold: float, top_k: int, num_gpus: int, gpu_ids: List[int],
                       cache_dir: str, url_mapping: Dict[str, int] = None, 
                       output_path: str = None, skip_first_round: bool = False):
    """Process all claims from one file with modified target URLs.
    
    Args:
        cache_dir: Directory to store cache files
        skip_first_round: If True, skip first-round judging and only do second-turn judging.
        output_path: Path to existing results file (required if skip_first_round is True).
    """
    
    print(f"\n{'='*80}")
    print(f"📁 Processing file: {file_id}")
    print(f"   Claims: {len(claims)}")
    print(f"   Summary citations: {len(summary_citations)}")
    if skip_first_round:
        print(f"   ⏩ Skipping first-round judging (using existing results)")
    print(f"{'='*80}")
    
    query = claims[0]['query'] if claims else ""
    
    # Create cache file path
    cache_file_path = os.path.join(cache_dir, f"cache_{file_id}.json")
    
    # Ensure cache file exists
    if not os.path.exists(cache_file_path):
        minimal_cache = {
            'query': query,
            'iterations': [],
            'report': [],
            'related_query': {}
        }
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(minimal_cache, f, indent=2, ensure_ascii=False)
    
    results = {}
    
    # First-round judging (skip if skip_first_round is True)
    if not skip_first_round:
        # Prepare claims with modified target URLs
        prepared_claims = []
        for claim_data in claims:
            claim = claim_data['claim']
            original_targets = claim_data['original_target_urls']
            modified_targets = compute_modified_targets(summary_citations, original_targets)
            
            # Log the modification
            print(f"   Claim: {claim[:60]}...")
            print(f"      Original targets: {len(original_targets)}")
            print(f"      Modified targets: {len(modified_targets)}")
            
            prepared_claims.append({
                'claim': claim,
                'target_urls': modified_targets,
                'source_type': 'filtered_misaligned',
                'source_file': file_id
            })
        
        # Process claims using the existing parallel pipeline
        print(f"🚀 Processing {len(prepared_claims)} claims with modified target URLs...")
        
        results = process_claims_parallel(
            query, prepared_claims, web_content, cache_file_path,
            similarity_threshold, top_k, num_gpus, gpu_ids, url_mapping
        )
        
        # Save results
        output_data = {
            'file_id': file_id,
            'query': query,
            'total_claims': len(results),
            'claims': list(results.values())
        }
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{file_id}_modified_target_results.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved results to {output_path}")
        
    # Second-turn rejudging for NotSupport claims
    print(f"\n🔄 Second-turn rejudging for NotSupport claims in {file_id}...")
    notsupport_claims = collect_notsupport_claims_for_misattribution(output_path)
    
    if notsupport_claims:
        print(f"📊 Found {len(notsupport_claims)} NotSupport claims for second-turn rejudging")
        
        # Re-judge NotSupport claims against unprocessed chunks
        updated_claim_results = rejudge_notsupport_claims_parallel(
            notsupport_claims=notsupport_claims,
            cache_file=cache_file_path,
            query=query,
            num_cores=64
        )
        
        # Update the results file with new judgments
        update_results_file_for_misattribution(output_path, updated_claim_results)
        
        # Count changed judgments
        changed_judgments = sum(1 for result in updated_claim_results.values() 
                               if result.get('final_judgment') != 'NotSupport')
        
        print(f"✅ Second-turn rejudging completed:")
        print(f"  - Total NotSupport claims processed: {len(updated_claim_results)}")
        print(f"  - Claims with changed judgments: {changed_judgments}")
        print(f"  - Claims remaining NotSupport: {len(updated_claim_results) - changed_judgments}")
    else:
        print(f"✅ No NotSupport claims for second-turn rejudging")
    
    return results
            
        

def main():
    """Main entry point - NOTE: This is kept for standalone testing only."""
    print("⚠️ Warning: This script should be called from evaluate.py with proper paths")
    print("⚠️ For standalone testing, please configure paths below:")
    
    # Configuration - These should be passed from evaluate.py
    filtered_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/filtered_misaligned_claims"
    output_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/results"
    cache_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/cache"
    web_cache_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/web_content_cache/train_gemini"
    raw_json_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/Tianyu_ReportBench/json_new_summary_cite"
    
    similarity_threshold = 0.4
    top_k = 5
    num_gpus = 1
    gpu_ids = [2]
    
    # Collect claims
    all_claims = collect_from_filtered_files(filtered_dir)
    
    if not all_claims:
        print("❌ No claims collected, exiting")
        return
    
    # Group by file_id
    claims_by_file = defaultdict(list)
    for claim_data in all_claims:
        file_id = claim_data['file_id']
        claims_by_file[file_id].append(claim_data)
    
    print(f"\n📊 Processing {len(claims_by_file)} files with {len(all_claims)} total claims")
    
    # Process each file
    total_results = {}
    
    for file_id, claims in claims_by_file.items():
        output_result_path = os.path.join(output_dir, f"{file_id}_modified_target_results.json")
        result_file_exists = os.path.exists(output_result_path)
        
        # Check if we need to skip first-round or entire processing
        if result_file_exists:
            if check_second_turn_completed(output_result_path):
                print(f"⏩ Result file exists and second-turn completed for {file_id}, skipping...")
                continue
            else:
                print(f"⏩ Result file exists for {file_id}, skipping first-round but running second-turn...")
        
        # Get paths for this file
        raw_json_file = os.path.join(raw_json_dir, f"{file_id}.json")
        web_cache_file = os.path.join(web_cache_dir, f"cache_{file_id}.json")
        
        # Get summary_citations and web content
        summary_citations = get_summary_citations(raw_json_file)
        if not summary_citations:
            print(f"⚠️ No summary_citations for {file_id}, skipping")
            continue
        
        web_content = load_web_cache(web_cache_file)
        if not result_file_exists and not web_content:
            print(f"⚠️ No web cache for {file_id}, skipping")
            continue
        
        # Build url_mapping for this file (if needed)
        # For standalone mode, we need to create url_mapping from raw JSON
        with open(raw_json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        all_urls = raw_data.get('all_source_links', [])
        url_mapping = {url: idx for idx, url in enumerate(all_urls)}
        
        # Process this file
        file_results = process_single_file(
            file_id, claims, summary_citations, web_content,
            output_dir, similarity_threshold, top_k, num_gpus, gpu_ids,
            cache_dir=cache_dir, url_mapping=url_mapping,
            output_path=output_result_path if result_file_exists else None,
            skip_first_round=result_file_exists
        )
        
        if file_results:
            total_results.update(file_results)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"   Files processed: {len(claims_by_file)}")
    print(f"   Total claims processed: {len(total_results)}")
    print(f"   Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed / 60:.2f} minutes")

