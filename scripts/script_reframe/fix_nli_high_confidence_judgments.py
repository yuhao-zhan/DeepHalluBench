#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, Any, List

def _to_binary_label(label: str) -> str:
    """Convert judgment label to binary format."""
    if not isinstance(label, str):
        return "NotSupport"
    normalized = label.strip().lower()
    if normalized in ["support", "entailed"]:
        return "Support"
    return "NotSupport"

def determine_judgment_from_nli_scores(nli_scores: Dict[str, float]) -> str:
    """
    Determine judgment from NLI scores.
    
    Args:
        nli_scores: Dictionary with 'entailment', 'neutral', 'contradiction' scores
        
    Returns:
        Binary judgment: 'Support' or 'NotSupport'
    """
    if not nli_scores:
        return "NotSupport"
    
    entailment = nli_scores.get('entailment', 0.0)
    neutral = nli_scores.get('neutral', 0.0)
    contradiction = nli_scores.get('contradiction', 0.0)
    
    # Find the highest scoring judgment
    max_score = max(entailment, neutral, contradiction)
    
    if max_score == entailment and entailment > 0.5:  # High confidence entailment
        return "Support"
    elif max_score == contradiction and contradiction > 0.5:  # High confidence contradiction
        return "NotSupport"
    else:  # Neutral or low confidence
        return "NotSupport"

def fix_nli_high_confidence_claims(results_file: str) -> None:
    """
    Fix judgment inconsistencies in NLI_HIGH_CONFIDENCE claims.
    
    Args:
        results_file: Path to the results JSON file
    """
    print(f"🔧 Fixing NLI_HIGH_CONFIDENCE claims in: {results_file}")
    
    # Load the results file
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results file: {e}")
        return
    
    fixed_count = 0
    total_nli_claims = 0
    
    # Process chain_of_research_results
    if "chain_of_research_results" in results_data:
        chain_results = results_data["chain_of_research_results"]
        for chain_item in chain_results:
            if isinstance(chain_item, dict) and "claim_results" in chain_item:
                claim_results = chain_item["claim_results"]
                for claim_data in claim_results:
                    if isinstance(claim_data, dict) and claim_data.get("processing_source") == "NLI_HIGH_CONFIDENCE":
                        total_nli_claims += 1
                        fixed = fix_single_nli_claim(claim_data)
                        if fixed:
                            fixed_count += 1
    
    # Process report_results
    if "report_results" in results_data:
        report_results = results_data["report_results"]
        for report_item in report_results:
            if isinstance(report_item, dict) and "claim_results" in report_item:
                claim_results = report_item["claim_results"]
                for claim_data in claim_results:
                    if isinstance(claim_data, dict) and claim_data.get("processing_source") == "NLI_HIGH_CONFIDENCE":
                        total_nli_claims += 1
                        fixed = fix_single_nli_claim(claim_data)
                        if fixed:
                            fixed_count += 1
    
    print(f"📊 Fixed {fixed_count} out of {total_nli_claims} NLI_HIGH_CONFIDENCE claims")
    
    # Save the corrected results
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Successfully saved corrected results to: {results_file}")
    except Exception as e:
        print(f"❌ Error saving results file: {e}")

def fix_single_nli_claim(claim_data: Dict[str, Any]) -> bool:
    """
    Fix a single NLI_HIGH_CONFIDENCE claim.
    
    Args:
        claim_data: The claim data dictionary
        
    Returns:
        True if the claim was fixed, False otherwise
    """
    claim_text = claim_data.get("claim", "")
    final_judgment = claim_data.get("final_judgment", "")
    nli_scores = claim_data.get("nli_scores", {})
    relevant_chunks = claim_data.get("relevant_chunks", [])
    all_judged_chunks = claim_data.get("all_judged_chunks", [])
    
    if not nli_scores:
        print(f"⚠️ No NLI scores found for claim: {claim_text[:100]}...")
        return False
    
    # Determine the correct judgment from NLI scores
    correct_judgment = None
    
    # Check each chunk's NLI scores
    for chunk_id, chunk_nli_scores in nli_scores.items():
        chunk_judgment = determine_judgment_from_nli_scores(chunk_nli_scores)
        if chunk_judgment == "Support":
            correct_judgment = "Support"
            break
    
    if correct_judgment is None:
        correct_judgment = "NotSupport"
    
    # Check if correction is needed
    needs_fix = False
    
    # Check relevant_chunks
    for chunk in relevant_chunks:
        if chunk.get("judgment") != correct_judgment:
            needs_fix = True
            break
    
    # Check all_judged_chunks
    if not needs_fix:
        for chunk in all_judged_chunks:
            if chunk.get("judgment") != correct_judgment:
                needs_fix = True
                break
    
    if not needs_fix:
        print(f"✅ No fix needed for claim: {claim_text[:100]}...")
        return False
    
    # Apply the fix
    print(f"🔧 Fixing claim: {claim_text[:100]}...")
    print(f"   Final judgment: {final_judgment} -> {correct_judgment}")
    
    # Update relevant_chunks
    for chunk in relevant_chunks:
        chunk["judgment"] = correct_judgment
    
    # Update all_judged_chunks
    for chunk in all_judged_chunks:
        chunk["judgment"] = correct_judgment
    
    # Update final_judgment if needed
    if final_judgment != correct_judgment:
        claim_data["final_judgment"] = correct_judgment
        print(f"   Updated final_judgment: {final_judgment} -> {correct_judgment}")
    
    print(f"✅ Fixed claim: {claim_text[:100]}...")
    return True

def main():
    """Main function to fix NLI_HIGH_CONFIDENCE claims."""
    results_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe/results_3b082586d614ef01655c508b42f68410.json"
    
    print(f"🚀 Starting NLI_HIGH_CONFIDENCE claims correction...")
    print(f"📁 Results file: {results_file}")
    
    # Create backup
    backup_file = results_file + ".backup"
    try:
        with open(results_file, 'r', encoding='utf-8') as src:
            with open(backup_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"💾 Created backup: {backup_file}")
    except Exception as e:
        print(f"⚠️ Could not create backup: {e}")
    
    # Fix the claims
    fix_nli_high_confidence_claims(results_file)
    
    print(f"✅ NLI_HIGH_CONFIDENCE claims correction completed!")

if __name__ == "__main__":
    main()
