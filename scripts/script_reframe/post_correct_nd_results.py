"""
Post-correction script for ND analysis results.

This script corrects ND analysis results by:
1. Re-extracting entailed claims with the fixed extract_entailed_claims_and_chunks function
2. Rebuilding entailed_chunk_memory with only Support chunks
3. Keeping existing clusters (both overall and iteration-level)
4. Re-labeling is_entailed and is_mapped flags for chunks and clusters
5. Recalculating all ND-related metrics (hallucination_score, nd_analysis, etc.)
"""

import json
from typing import Dict, List, Set, Any, Tuple, Optional
import sys
import os
import glob
from pathlib import Path

# Add the script_reframe directory to path to import from overall_noise_domination
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from overall_noise_domination import (
    extract_entailed_claims_and_chunks,
    EntailedChunkMemory,
    calculate_hallucination_score_with_nd_analysis,
    map_entailed_chunks_to_clusters,
    load_summary_citations,
    normalize_url_for_matching,
    analyze_entailed_chunks_iterations_from_memory,
    calculate_iteration_statistics_from_clusters,
    calculate_citation_statistics_from_clusters
)


def rebuild_entailed_memory(results: Dict[str, Any], verbose: bool = True) -> EntailedChunkMemory:
    """Rebuild entailed chunk memory using the fixed extraction function."""
    if verbose:
        print("🔧 Rebuilding entailed chunk memory with fixed extraction...")
    
    entailed_claims_data = extract_entailed_claims_and_chunks(results)
    if not entailed_claims_data:
        if verbose:
            print("⚠️ No entailed claims found in results file")
        return EntailedChunkMemory()
    
    entailed_chunk_memory = EntailedChunkMemory()
    if verbose:
        print(f"📊 Found {len(entailed_claims_data)} entailed claims")
    
    for claim_data in entailed_claims_data:
        relevant_chunks = claim_data["relevant_chunks"]
        entailed_chunk_memory.add_entailed_chunks(relevant_chunks)
    
    if verbose:
        print(f"✅ Total entailed chunks stored: {entailed_chunk_memory.get_entailed_chunk_count()}")
    return entailed_chunk_memory


def correct_cluster_chunks(
    cluster: Dict[str, Any], 
    entailed_chunk_memory: EntailedChunkMemory
) -> Tuple[bool, int]:
    """
    Correct is_entailed flags for chunks in a cluster and determine if cluster is mapped.
    
    Returns:
        (is_mapped, num_entailed_chunks)
    """
    has_entailed_chunk = False
    num_entailed = 0
    
    for chunk in cluster.get("chunks", []):
        chunk_id = chunk.get("chunk_id", "")
        url = chunk.get("url", "")
        
        # Check if chunk is entailed using the corrected memory
        is_entailed = entailed_chunk_memory.is_chunk_entailed(chunk_id, url)
        
        # Update is_entailed flag
        chunk["is_entailed"] = is_entailed
        if "features" in chunk:
            chunk["features"]["is_entailed"] = is_entailed
        
        if is_entailed:
            has_entailed_chunk = True
            num_entailed += 1
    
    return has_entailed_chunk, num_entailed


def correct_overall_clusters(
    results: Dict[str, Any],
    entailed_chunk_memory: EntailedChunkMemory,
    summary_citations: Set[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """Correct overall clusters and recalculate ND analysis."""
    if verbose:
        print("\n📊 Correcting overall clusters...")
    
    if "clusters" not in results:
        if verbose:
            print("⚠️ No clusters found in results")
        return results
    
    clusters = results["clusters"]
    if verbose:
        print(f"🔍 Processing {len(clusters)} overall clusters...")
    
    # Build chunks_with_avg_scores for ND calculation
    chunks_with_avg_scores = []
    for cluster in clusters:
        for chunk in cluster.get("chunks", []):
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("chunk_text", "")
            url = chunk.get("url", "")
            score = cluster.get("score", 0.0)  # Use cluster score as chunk score approximation
            chunks_with_avg_scores.append((chunk_id, chunk_text, url, score))
    
    # Rebuild clusters_with_scores structure for mapping
    clusters_with_scores = []
    mapped_cluster_indices = []
    unmapped_cluster_indices = []
    
    for rank, cluster in enumerate(clusters):
        # Correct chunks in this cluster
        is_mapped, num_entailed = correct_cluster_chunks(cluster, entailed_chunk_memory)
        
        # Update is_mapped flag
        cluster["is_mapped"] = is_mapped
        if is_mapped:
            mapped_cluster_indices.append(rank)
        else:
            unmapped_cluster_indices.append(rank)
        
        # Rebuild cluster_chunks structure for mapping function
        cluster_chunks = [(chunk["chunk_id"], chunk["chunk_text"]) for chunk in cluster.get("chunks", [])]
        cluster_score = cluster.get("score", 0.0)
        clusters_with_scores.append((cluster_chunks, cluster_score))
        
        if num_entailed > 0 and verbose:
            print(f"  ✅ Cluster {rank}: {num_entailed} entailed chunks, is_mapped={is_mapped}")
    
    if verbose:
        print(f"📈 Mapped clusters: {len(mapped_cluster_indices)}, Unmapped clusters: {len(unmapped_cluster_indices)}")
    
    # Recalculate ND analysis
    if verbose:
        print("🔄 Recalculating ND analysis for overall clusters...")
    nd_results = calculate_hallucination_score_with_nd_analysis(
        mapped_cluster_indices,
        unmapped_cluster_indices,
        clusters_with_scores,
        summary_citations,
        chunks_with_avg_scores
    )
    
    # Update results
    results["hallucination_score"] = nd_results["wis_score"]
    results["nd_analysis"] = nd_results["nd_analysis"]
    results["mapped_clusters"] = sorted(mapped_cluster_indices)
    results["unmapped_clusters"] = sorted(unmapped_cluster_indices)
    results["total_clusters"] = len(clusters)
    results["total_mapped"] = len(mapped_cluster_indices)
    results["total_unmapped"] = len(unmapped_cluster_indices)
    
    if verbose:
        print(f"✅ Overall ND analysis corrected: WIS Score = {nd_results['wis_score']:.4f}")
    
    # Recalculate iteration statistics from corrected clusters
    if verbose:
        print("🔄 Recalculating iteration statistics...")
    iteration_stats = calculate_iteration_statistics_from_clusters(clusters)
    results["iteration_statistics"] = iteration_stats
    
    # Recalculate citation statistics from corrected clusters
    if verbose:
        print("🔄 Recalculating citation statistics...")
    citation_stats = calculate_citation_statistics_from_clusters(clusters)
    results["citation_statistics"] = citation_stats
    
    # Recalculate mapped_unmapped_feature_comparison from corrected clusters
    if verbose:
        print("🔄 Recalculating mapped_unmapped_feature_comparison...")
    mapped_chunks_features = []
    unmapped_chunks_features = []
    
    for cluster in clusters:
        for chunk in cluster.get("chunks", []):
            features = {
                'chunk_id': chunk.get('chunk_id', ''),
                'url': chunk.get('url', ''),
                'is_entailed': chunk.get('is_entailed', False),
                'in_citations': chunk.get('in_citations', False),
                'iteration': chunk.get('iteration'),
                'iterations': chunk.get('iterations', [])
            }
            
            if chunk.get('is_entailed', False):
                mapped_chunks_features.append(features)
            else:
                unmapped_chunks_features.append(features)
    
    feature_comparison = {
        'mapped_count': len(mapped_chunks_features),
        'unmapped_count': len(unmapped_chunks_features)
    }
    results["mapped_unmapped_feature_comparison"] = feature_comparison
    
    return results


def correct_iteration_clusters(
    iteration_results: Dict[str, Any],
    entailed_chunk_memory: EntailedChunkMemory,
    summary_citations: Set[str],
    iteration_idx: int
) -> Dict[str, Any]:
    """Correct clusters for a single iteration and recalculate ND analysis."""
    if "clusters" not in iteration_results:
        return iteration_results
    
    clusters = iteration_results["clusters"]
    
    # Build chunks_with_avg_scores for this iteration
    chunks_with_avg_scores = []
    for cluster in clusters:
        for chunk in cluster.get("chunks", []):
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("chunk_text", "")
            url = chunk.get("url", "")
            score = cluster.get("score", 0.0)
            chunks_with_avg_scores.append((chunk_id, chunk_text, url, score))
    
    # Rebuild clusters_with_scores structure
    clusters_with_scores = []
    mapped_cluster_indices = []
    unmapped_cluster_indices = []
    
    for rank, cluster in enumerate(clusters):
        # Correct chunks in this cluster
        is_mapped, num_entailed = correct_cluster_chunks(cluster, entailed_chunk_memory)
        
        # Update is_mapped flag
        cluster["is_mapped"] = is_mapped
        if is_mapped:
            mapped_cluster_indices.append(rank)
        else:
            unmapped_cluster_indices.append(rank)
        
        # Rebuild cluster_chunks structure
        cluster_chunks = [(chunk["chunk_id"], chunk["chunk_text"]) for chunk in cluster.get("chunks", [])]
        cluster_score = cluster.get("score", 0.0)
        clusters_with_scores.append((cluster_chunks, cluster_score))
    
    # Recalculate ND analysis for this iteration
    nd_results = calculate_hallucination_score_with_nd_analysis(
        mapped_cluster_indices,
        unmapped_cluster_indices,
        clusters_with_scores,
        summary_citations,
        chunks_with_avg_scores
    )
    
    # Update iteration results
    iteration_results["mapped_clusters"] = sorted(mapped_cluster_indices)
    iteration_results["unmapped_clusters"] = sorted(unmapped_cluster_indices)
    iteration_results["mapped_cluster_count"] = len(mapped_cluster_indices)
    iteration_results["unmapped_cluster_count"] = len(unmapped_cluster_indices)
    iteration_results["hallucination_score"] = nd_results["wis_score"]
    iteration_results["nd_analysis"] = nd_results["nd_analysis"]
    
    return iteration_results


def correct_iteration_level_nd_analysis(
    results: Dict[str, Any],
    entailed_chunk_memory: EntailedChunkMemory,
    summary_citations: Set[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """Correct iteration-level ND analysis."""
    if verbose:
        print("\n📊 Correcting iteration-level ND analysis...")
    
    if "iteration_level_nd_analysis" not in results:
        print("⚠️ No iteration_level_nd_analysis found in results")
        return results
    
    iteration_nd_analysis = results["iteration_level_nd_analysis"]
    
    if "iteration_results" not in iteration_nd_analysis:
        print("⚠️ No iteration_results found in iteration_level_nd_analysis")
        return results
    
    iteration_results_dict = iteration_nd_analysis["iteration_results"]
    if verbose:
        print(f"🔍 Processing {len(iteration_results_dict)} iterations...")
    
    corrected_iterations = {}
    
    for iter_key, iter_data in iteration_results_dict.items():
        iter_idx = iter_data.get("iteration_index", int(iter_key))
        if verbose:
            print(f"  🔄 Processing Iteration {iter_idx}...")
        
        corrected_iter = correct_iteration_clusters(
            iter_data,
            entailed_chunk_memory,
            summary_citations,
            iter_idx
        )
        
        corrected_iterations[iter_key] = corrected_iter
        
        if verbose:
            print(f"    ✅ Iteration {iter_idx}: HS = {corrected_iter['hallucination_score']:.4f}, "
                  f"Mapped = {corrected_iter['mapped_cluster_count']}, "
                  f"Unmapped = {corrected_iter['unmapped_cluster_count']}")
    
    # Update aggregate statistics
    if corrected_iterations:
        avg_hs = sum(r["hallucination_score"] for r in corrected_iterations.values()) / len(corrected_iterations)
        total_mapped = sum(r["mapped_cluster_count"] for r in corrected_iterations.values())
        total_unmapped = sum(r["unmapped_cluster_count"] for r in corrected_iterations.values())
        
        iteration_nd_analysis["iteration_results"] = corrected_iterations
        if "aggregate_statistics" not in iteration_nd_analysis:
            iteration_nd_analysis["aggregate_statistics"] = {}
        
        iteration_nd_analysis["aggregate_statistics"].update({
            "total_iterations": len(corrected_iterations),
            "average_hallucination_score": avg_hs,
            "total_mapped_clusters": total_mapped,
            "total_unmapped_clusters": total_unmapped,
            "avg_clusters_per_iteration": sum(r["total_clusters"] for r in corrected_iterations.values()) / len(corrected_iterations)
        })
    
    if verbose:
        print(f"✅ Iteration-level ND analysis corrected for {len(corrected_iterations)} iterations")
    
    return results


def recalculate_entailed_iteration_analysis(
    results: Dict[str, Any],
    entailed_chunk_memory: EntailedChunkMemory,
    verbose: bool = True
) -> Dict[str, Any]:
    """Recalculate entailed_iteration_analysis using corrected entailed_chunk_memory."""
    if verbose:
        print("\n📊 Recalculating entailed_iteration_analysis...")
    
    # Build url_to_iterations mapping from results file
    url_to_iterations = {}
    if "clusters" in results:
        # Extract iteration information from clusters
        for cluster in results.get("clusters", []):
            for chunk in cluster.get("chunks", []):
                url = chunk.get("url", "")
                iterations = chunk.get("iterations", [])
                if url and iterations:
                    # Store both original and normalized URL mappings
                    normalized_url = normalize_url_for_matching(url)
                    # Store in normalized URL (primary)
                    if normalized_url not in url_to_iterations:
                        url_to_iterations[normalized_url] = []
                    url_to_iterations[normalized_url].extend(iterations)
                    # Also store in original URL (for exact matching)
                    if url != normalized_url and url not in url_to_iterations:
                        url_to_iterations[url] = []
                    if url != normalized_url:
                        url_to_iterations[url].extend(iterations)
        
        # Deduplicate and sort iterations for each URL
        for url in url_to_iterations:
            url_to_iterations[url] = sorted(set(url_to_iterations[url]))
    
    # Use the function from overall_noise_domination to analyze
    entailed_iteration_analysis = analyze_entailed_chunks_iterations_from_memory(
        entailed_chunk_memory,
        url_to_iterations
    )
    
    if verbose:
        print(f"✅ Recalculated entailed_iteration_analysis: {entailed_iteration_analysis.get('total_entailed_chunks', 0)} entailed chunks")
        print(f"   - Chunks with iterations: {entailed_iteration_analysis.get('chunks_with_iterations', 0)}")
        print(f"   - Chunks without iterations: {entailed_iteration_analysis.get('chunks_without_iterations', 0)}")
    
    return entailed_iteration_analysis


def extract_before_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract statistics before correction."""
    stats = {
        'total_entailed_chunks': results.get('total_entailed_chunks', 0),
        'hallucination_score': results.get('hallucination_score', 0.0),
        'total_mapped': results.get('total_mapped', 0),
        'total_unmapped': results.get('total_unmapped', 0),
        'total_clusters': results.get('total_clusters', 0),
    }
    
    # Entailed iteration analysis
    if 'entailed_iteration_analysis' in results:
        eia = results['entailed_iteration_analysis']
        stats['entailed_iteration_total'] = eia.get('total_entailed_chunks', 0)
        stats['entailed_iteration_with'] = eia.get('chunks_with_iterations', 0)
        stats['entailed_iteration_without'] = eia.get('chunks_without_iterations', 0)
    
    # Mapped/unmapped feature comparison
    if 'mapped_unmapped_feature_comparison' in results:
        comp = results['mapped_unmapped_feature_comparison']
        stats['mapped_count'] = comp.get('mapped_count', 0)
        stats['unmapped_count'] = comp.get('unmapped_count', 0)
    
    # Iteration-level ND analysis
    if 'iteration_level_nd_analysis' in results:
        agg_stats = results['iteration_level_nd_analysis'].get('aggregate_statistics', {})
        if agg_stats:
            stats['iteration_level_avg_hs'] = agg_stats.get('average_hallucination_score', 0.0)
            stats['iteration_level_total'] = agg_stats.get('total_iterations', 0)
    
    return stats


def post_correct_nd_results(results_file: str, raw_json_file: str = None, json_file_path: str = None, 
                           verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Post-correct ND analysis results in the given results file.
    
    Args:
        results_file: Path to the results JSON file to correct
        raw_json_file: Optional path to raw JSON file for summary_citations
        json_file_path: Optional path to cache JSON file for url_to_iterations mapping
        verbose: Whether to print detailed output
    
    Returns:
        Dict containing before/after statistics and changes, or None if error
    """
    if verbose:
        print("=" * 80)
        print("🔧 POST-CORRECTING ND ANALYSIS RESULTS")
        print("=" * 80)
        print(f"📁 Results file: {results_file}")
    
    # Load results file
    if verbose:
        print("\n📂 Loading results file...")
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        if verbose:
            print(f"✅ Loaded results file with {len(results)} top-level keys")
    except Exception as e:
        if verbose:
            print(f"❌ Error loading results file: {e}")
        return None
    
    # Extract before statistics
    before_stats = extract_before_stats(results)
    
    # Load summary_citations if raw_json_file is provided
    summary_citations = set()
    if raw_json_file:
        if verbose:
            print(f"📂 Loading summary_citations from: {raw_json_file}")
        summary_citations = load_summary_citations(raw_json_file)
        if verbose:
            print(f"✅ Loaded {len(summary_citations)} summary citations")
    elif "summary_urls" in results:
        # Fallback to summary_urls in results file
        summary_citations = set(results.get("summary_urls", []))
        if verbose:
            print(f"✅ Using {len(summary_citations)} summary_urls from results file")
    
    # Step 1: Rebuild entailed memory with fixed extraction
    entailed_chunk_memory = rebuild_entailed_memory(results, verbose=verbose)
    results["total_entailed_chunks"] = entailed_chunk_memory.get_entailed_chunk_count()
    
    # Step 2: Correct overall clusters (this also updates iteration_statistics, citation_statistics, mapped_unmapped_feature_comparison)
    results = correct_overall_clusters(results, entailed_chunk_memory, summary_citations, verbose=verbose)
    
    # Step 3: Recalculate entailed_iteration_analysis
    results["entailed_iteration_analysis"] = recalculate_entailed_iteration_analysis(results, entailed_chunk_memory, verbose=verbose)
    
    # Step 4: Correct iteration-level clusters
    results = correct_iteration_level_nd_analysis(results, entailed_chunk_memory, summary_citations, verbose=verbose)
    
    # Extract after statistics
    after_stats = extract_before_stats(results)
    
    # Calculate changes
    changes = {
        'total_entailed_chunks': after_stats['total_entailed_chunks'] - before_stats['total_entailed_chunks'],
        'hallucination_score': after_stats['hallucination_score'] - before_stats['hallucination_score'],
        'total_mapped': after_stats['total_mapped'] - before_stats['total_mapped'],
        'total_unmapped': after_stats['total_unmapped'] - before_stats['total_unmapped'],
    }
    
    if 'entailed_iteration_total' in before_stats:
        changes['entailed_iteration_total'] = after_stats.get('entailed_iteration_total', 0) - before_stats['entailed_iteration_total']
    
    if 'mapped_count' in before_stats:
        changes['mapped_count'] = after_stats.get('mapped_count', 0) - before_stats['mapped_count']
        changes['unmapped_count'] = after_stats.get('unmapped_count', 0) - before_stats['unmapped_count']
    
    if 'iteration_level_avg_hs' in before_stats:
        changes['iteration_level_avg_hs'] = after_stats.get('iteration_level_avg_hs', 0.0) - before_stats.get('iteration_level_avg_hs', 0.0)
    
    # Save corrected results
    if verbose:
        print(f"\n💾 Saving corrected results to: {results_file}")
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"✅ Corrected results saved successfully!")
    except Exception as e:
        if verbose:
            print(f"❌ Error saving corrected results: {e}")
        return None
    
    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("📊 CORRECTION SUMMARY")
        print("=" * 80)
        print(f"✅ Total entailed chunks: {results.get('total_entailed_chunks', 0)} (change: {changes['total_entailed_chunks']:+d})")
        print(f"✅ Overall WIS Score: {results.get('hallucination_score', 0):.4f} (change: {changes['hallucination_score']:+.4f})")
        print(f"✅ Overall mapped clusters: {results.get('total_mapped', 0)} (change: {changes['total_mapped']:+d})")
        print(f"✅ Overall unmapped clusters: {results.get('total_unmapped', 0)} (change: {changes['total_unmapped']:+d})")
        
        if "entailed_iteration_analysis" in results:
            eia = results["entailed_iteration_analysis"]
            print(f"✅ Entailed iteration analysis: {eia.get('total_entailed_chunks', 0)} entailed chunks")
            print(f"   - Chunks with iterations: {eia.get('chunks_with_iterations', 0)}")
            print(f"   - Chunks without iterations: {eia.get('chunks_without_iterations', 0)}")
        
        if "mapped_unmapped_feature_comparison" in results:
            comp = results["mapped_unmapped_feature_comparison"]
            print(f"✅ Mapped/Unmapped comparison: Mapped={comp.get('mapped_count', 0)}, Unmapped={comp.get('unmapped_count', 0)}")
        
        if "iteration_level_nd_analysis" in results:
            agg_stats = results["iteration_level_nd_analysis"].get("aggregate_statistics", {})
            if agg_stats:
                print(f"✅ Iteration-level average HS: {agg_stats.get('average_hallucination_score', 0):.4f}")
                print(f"✅ Total iterations: {agg_stats.get('total_iterations', 0)}")
        
        print("=" * 80)
    
    # Return statistics
    return {
        'filename': os.path.basename(results_file),
        'before': before_stats,
        'after': after_stats,
        'changes': changes
    }


def batch_process_directory(directory: str, raw_json_base_dir: str = None, verbose: bool = True):
    """
    Batch process all JSON files in a directory.
    
    Args:
        directory: Directory containing result JSON files
        raw_json_base_dir: Optional base directory for raw JSON files (to find summary_citations)
        verbose: Whether to print detailed output for each file
    """
    print("=" * 80)
    print("🔄 BATCH PROCESSING ND RESULTS CORRECTION")
    print("=" * 80)
    print(f"📁 Processing directory: {directory}")
    
    # Find all JSON files
    pattern = os.path.join(directory, "*_combined.json")
    result_files = glob.glob(pattern)
    result_files.sort()
    
    print(f"\n📊 Found {len(result_files)} result files to process\n")
    
    all_statistics = []
    success_count = 0
    error_count = 0
    
    for idx, results_file in enumerate(result_files, 1):
        filename = os.path.basename(results_file)
        print(f"\n[{idx}/{len(result_files)}] Processing: {filename}")
        print("-" * 80)
        
        # Try to find corresponding raw JSON file
        raw_json_file = None
        if raw_json_base_dir:
            # Extract base name (remove _combined.json suffix)
            base_name = filename.replace("_combined.json", ".json")
            potential_raw_file = os.path.join(raw_json_base_dir, base_name)
            if os.path.exists(potential_raw_file):
                raw_json_file = potential_raw_file
        
        # Process file
        stats = post_correct_nd_results(results_file, raw_json_file, None, verbose=verbose)
        
        if stats:
            all_statistics.append(stats)
            success_count += 1
        else:
            error_count += 1
            print(f"❌ Failed to process {filename}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count} files")
    print(f"❌ Failed: {error_count} files")
    print(f"📁 Total files: {len(result_files)}")
    
    if all_statistics:
        print("\n" + "=" * 80)
        print("📈 AGGREGATE STATISTICS")
        print("=" * 80)
        
        # Calculate aggregate changes
        total_entailed_change = sum(s['changes']['total_entailed_chunks'] for s in all_statistics)
        total_hs_change = sum(s['changes']['hallucination_score'] for s in all_statistics)
        total_mapped_change = sum(s['changes']['total_mapped'] for s in all_statistics)
        total_unmapped_change = sum(s['changes']['total_unmapped'] for s in all_statistics)
        
        avg_entailed_change = total_entailed_change / len(all_statistics)
        avg_hs_change = total_hs_change / len(all_statistics)
        avg_mapped_change = total_mapped_change / len(all_statistics)
        avg_unmapped_change = total_unmapped_change / len(all_statistics)
        
        print(f"\n📊 Total Changes (Sum across all files):")
        print(f"  - Total Entailed Chunks: {total_entailed_change:+d}")
        print(f"  - Overall Hallucination Score: {total_hs_change:+.4f}")
        print(f"  - Mapped Clusters: {total_mapped_change:+d}")
        print(f"  - Unmapped Clusters: {total_unmapped_change:+d}")
        
        print(f"\n📊 Average Changes (Per file):")
        print(f"  - Total Entailed Chunks: {avg_entailed_change:+.2f}")
        print(f"  - Overall Hallucination Score: {avg_hs_change:+.4f}")
        print(f"  - Mapped Clusters: {avg_mapped_change:+.2f}")
        print(f"  - Unmapped Clusters: {avg_unmapped_change:+.2f}")
        
        # Print detailed statistics table
        print(f"\n{'=' * 80}")
        print("📋 DETAILED FILE STATISTICS")
        print("=" * 80)
        print(f"{'Filename':<50} {'Entailed':<12} {'HS Change':<12} {'Mapped':<10} {'Unmapped':<10}")
        print("-" * 80)
        
        for stats in all_statistics:
            filename = stats['filename']
            changes = stats['changes']
            print(f"{filename:<50} {changes['total_entailed_chunks']:>+11} "
                  f"{changes['hallucination_score']:>+11.4f} {changes['total_mapped']:>+9} "
                  f"{changes['total_unmapped']:>+9}")
        
        print("=" * 80)
    
    return all_statistics


def main():
    """Main function."""
    # Directory to process
    results_directory = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Tianyu_sampled/gemini/after_update"
    
    # Optional: base directory for raw JSON files (for summary_citations)
    # If None, will use summary_urls from results file
    raw_json_base_dir = None  # Set to actual path if available
    
    # Batch process all files in directory
    batch_process_directory(results_directory, raw_json_base_dir, verbose=True)


if __name__ == "__main__":
    main()

