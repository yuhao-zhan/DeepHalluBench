#!/usr/bin/env python3
"""
ND Analysis Script - Standalone Noise Domination Analysis
Decouples ND processing from evaluate.py and provides detailed analysis
"""

import json
import os
import sys
from typing import List, Dict, Set, Tuple, Any, Optional
from pathlib import Path
from collections import Counter
import setproctitle
setproctitle.setproctitle('Yuhao_ND_analysis')

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'HalluDetector' / 'script_reframe'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'claim_verification' / 'top_scripts'))

# Import from overall_noise_domination module (runtime import, linter may not resolve it)
from overall_noise_domination import (  # type: ignore
    extract_chunks_with_scores,
    compute_average_scores,
    cluster_and_rank_chunks,
    extract_entailed_claims_and_chunks,
    EntailedChunkMemory,
    map_entailed_chunks_to_clusters,
    calculate_hallucination_score_with_nd_analysis,
    load_summary_citations,
    normalize_url_for_matching
)


def load_url_to_iteration_mapping(cache_file: str) -> Dict[str, List[int]]:
    """
    Load the mapping from URLs to iteration indices.
    Returns: Dict mapping normalized URL to list of iteration indices where it appears
    """
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        url_to_iterations = {}
        iterations = cache_data.get('iterations', [])
        
        for iter_idx, iteration in enumerate(iterations, start=1):
            # Check for search_list_{index} in the iteration
            search_key = f'search_list_{iter_idx}'
            if search_key in iteration:
                urls = iteration[search_key]
                # De-duplicate URLs within the same iteration to avoid repeated iteration entries
                unique_urls_in_iteration = set(urls)
                for url in unique_urls_in_iteration:
                    # Normalize URL for matching
                    normalized_url = normalize_url_for_matching(url)
                    if normalized_url not in url_to_iterations:
                        url_to_iterations[normalized_url] = []
                    # Append iteration index only once
                    if not url_to_iterations[normalized_url] or url_to_iterations[normalized_url][-1] != iter_idx:
                        url_to_iterations[normalized_url].append(iter_idx)
        
        # Ensure each URL maps to a unique, sorted list of iterations
        for key in list(url_to_iterations.keys()):
            url_to_iterations[key] = sorted(set(url_to_iterations[key]))

        print(f"📊 Loaded URL-to-iteration mapping: {len(url_to_iterations)} unique URLs across iterations")
        return url_to_iterations
        
    except Exception as e:
        print(f"⚠️ Error loading URL-to-iteration mapping: {e}")
        return {}


def analyze_entailed_chunks_iterations(entailed_chunk_memory: EntailedChunkMemory, 
                                       url_to_iterations: Dict[str, List[int]]) -> Dict[str, Any]:
    """
    Analyze which iterations entailed chunks were retrieved from.
    
    Args:
        entailed_chunk_memory: Memory containing entailed chunks
        url_to_iterations: Mapping from URLs to iteration indices
    
    Returns:
        Dict containing iteration distribution statistics
    """
    entailed_details = entailed_chunk_memory.get_entailed_chunk_details()
    
    all_iterations = []
    chunk_iteration_details = []
    
    for chunk_key, chunk_info in entailed_details.items():
        url = chunk_info['source_url']
        normalized_url = normalize_url_for_matching(url)
        
        # Find iterations for this URL
        iterations = url_to_iterations.get(normalized_url, [])
        if not iterations:
            # Try exact URL match
            iterations = url_to_iterations.get(url, [])
        
        if iterations:
            all_iterations.extend(iterations)
            chunk_iteration_details.append({
                'chunk_id': chunk_info['chunk_id'],
                'url': url,
                'normalized_url': normalized_url,
                'iterations': iterations,
                'first_iteration': min(iterations) if iterations else None,
                'last_iteration': max(iterations) if iterations else None
            })
        else:
            chunk_iteration_details.append({
                'chunk_id': chunk_info['chunk_id'],
                'url': url,
                'normalized_url': normalized_url,
                'iterations': [],
                'first_iteration': None,
                'last_iteration': None
            })
    
    # Calculate distribution statistics
    iteration_distribution = {}
    for iteration in all_iterations:
        iteration_distribution[iteration] = iteration_distribution.get(iteration, 0) + 1
    
    chunks_with_iterations = len([c for c in chunk_iteration_details if c['iterations']])
    chunks_without_iterations = len([c for c in chunk_iteration_details if not c['iterations']])
    
    return {
        'total_entailed_chunks': len(entailed_details),
        'chunks_with_iterations': chunks_with_iterations,
        'chunks_without_iterations': chunks_without_iterations,
        'iteration_distribution': sorted(iteration_distribution.items()),
        'chunk_details': chunk_iteration_details
    }


def compare_mapped_unmapped_features(mapped_features: List[Dict[str, Any]], 
                                       unmapped_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare features between mapped and unmapped chunks to find distinguishing characteristics.
    
    Args:
        mapped_features: List of feature dicts for mapped chunks
        unmapped_features: List of feature dicts for unmapped chunks
    
    Returns:
        Dict containing comparison statistics
    """
    if not mapped_features and not unmapped_features:
        return {'status': 'no_data'}
    
    comparison = {
        'mapped_count': len(mapped_features),
        'unmapped_count': len(unmapped_features),
        'feature_averages': {}
    }
    
    # Calculate position distribution (front/middle/back)
    position_features = ['document_position']
    for feature in position_features:
        mapped_front = sum(1 for f in mapped_features if f.get(feature) == 'front')
        mapped_middle = sum(1 for f in mapped_features if f.get(feature) == 'middle')
        mapped_back = sum(1 for f in mapped_features if f.get(feature) == 'back')
        
        unmapped_front = sum(1 for f in unmapped_features if f.get(feature) == 'front')
        unmapped_middle = sum(1 for f in unmapped_features if f.get(feature) == 'middle')
        unmapped_back = sum(1 for f in unmapped_features if f.get(feature) == 'back')
        
        comparison['feature_averages'][feature] = {
            'mapped': {
                'front': mapped_front,
                'middle': mapped_middle,
                'back': mapped_back,
                'front_pct': mapped_front / len(mapped_features) if mapped_features else 0,
                'middle_pct': mapped_middle / len(mapped_features) if mapped_features else 0,
                'back_pct': mapped_back / len(mapped_features) if mapped_features else 0
            },
            'unmapped': {
                'front': unmapped_front,
                'middle': unmapped_middle,
                'back': unmapped_back,
                'front_pct': unmapped_front / len(unmapped_features) if unmapped_features else 0,
                'middle_pct': unmapped_middle / len(unmapped_features) if unmapped_features else 0,
                'back_pct': unmapped_back / len(unmapped_features) if unmapped_features else 0
            }
        }
    
    # Find most distinguishing features (largest differences)
    distinguishing_features = []
    for feature, stats in comparison['feature_averages'].items():
        if 'avg' in stats:
            diff = abs(stats['mapped_avg'] - stats['unmapped_avg'])
            distinguishing_features.append((feature, diff, stats))
        elif feature == 'document_position':
            # Calculate difference for position distribution
            mapped_dist = stats['mapped']
            unmapped_dist = stats['unmapped']
            front_diff = abs(mapped_dist['front_pct'] - unmapped_dist['front_pct'])
            middle_diff = abs(mapped_dist['middle_pct'] - unmapped_dist['middle_pct'])
            back_diff = abs(mapped_dist['back_pct'] - unmapped_dist['back_pct'])
            max_diff = max(front_diff, middle_diff, back_diff)
            distinguishing_features.append((feature, max_diff, stats))
    
    # Sort by difference and take top 5
    distinguishing_features.sort(key=lambda x: x[1], reverse=True)
    comparison['top_distinguishing_features'] = [
        {'feature': feat, 'difference': diff, 'stats': stats}
        for feat, diff, stats in distinguishing_features[:5]
    ]
    
    return comparison


def get_chunk_position_in_document(
    chunk_id: str,
    url: str,
    chunk_scores: Dict[str, Any],
    url_to_url_index: Optional[Dict[str, int]] = None
) -> str:
    """Get the position of a chunk in its document (front/middle/back).

    Args:
        chunk_id: Chunk ID (e.g., "chunk_9")
        url: URL of the document
        chunk_scores: Dictionary of chunk_score data from cache where keys follow
            the pattern "<url_index>-chunk_<idx>"
        url_to_url_index: Optional mapping from URL to its index in the cache

    Returns:
        "front", "middle", or "back"
    """

    try:
        import re

        match = re.search(r"chunk_(\d+)", chunk_id)
        if not match:
            return "middle"

        chunk_num = int(match.group(1))

        normalized_url = normalize_url_for_matching(url)
        url_chunk_indices: List[int] = []

        url_index = None
        if url_to_url_index and url in url_to_url_index:
            url_index = url_to_url_index[url]

        for chunk_key, chunk_info in chunk_scores.items():
            chunk_url = chunk_info.get("url", "")
            chunk_url_normalized = normalize_url_for_matching(chunk_url)

            url_matches = chunk_url == url or chunk_url_normalized == normalized_url

            if not url_matches and url_index is not None:
                key_match = re.match(r"^(\d+)-chunk_", str(chunk_key))
                if key_match and int(key_match.group(1)) == url_index:
                    url_matches = True

            if not url_matches:
                continue

            chunk_id_from_cache = chunk_info.get("chunk_id_original", "")
            if not chunk_id_from_cache:
                key_chunk_match = re.search(r"chunk_(\d+)", str(chunk_key))
                if key_chunk_match:
                    chunk_id_from_cache = f"chunk_{key_chunk_match.group(1)}"

            cache_match = re.search(r"chunk_(\d+)", chunk_id_from_cache)
            if cache_match:
                url_chunk_indices.append(int(cache_match.group(1)))

        if not url_chunk_indices:
            return "middle"

        total_chunks = max(url_chunk_indices) + 1
        position_ratio = chunk_num / total_chunks if total_chunks > 0 else 0.5

        if position_ratio < 0.33:
            return "front"
        if position_ratio < 0.67:
            return "middle"
        return "back"

    except Exception as exc:
        print(f"⚠️ Error calculating chunk position for {chunk_id}: {exc}")
        return "middle"


def extract_chunk_features(chunk_data: Dict[str, Any], 
                          chunk_id: str, 
                          url: str,
                          cache_file: str,
                          chunk_scores: Dict[str, Any],
                          chunk_text: str = '',
                          url_to_url_index: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Extract features for a chunk to distinguish mapped vs unmapped chunks.
    
    Args:
        chunk_data: Chunk data from cache
        chunk_id: Chunk ID
        url: URL of the chunk
        cache_file: Path to cache file
        chunk_scores: Dictionary of chunk_score data from cache
        chunk_text: Chunk text
    
    Returns:
        Dict containing extracted features
    """
    features = {
        'chunk_id': chunk_id,
        'url': url,
        'chunk_text': chunk_text
    }
    
    # Get chunk text from chunk_data if not provided
    if not chunk_text and chunk_data:
        features['chunk_text'] = chunk_data.get('chunk_text', '')
    
    # Get document position (front/middle/back)
    document_position = get_chunk_position_in_document(chunk_id, url, chunk_scores, url_to_url_index)
    features['document_position'] = document_position
    
    return features


def check_chunk_in_support_claims(chunk_id: str, url: str, results: Dict[str, Any]) -> bool:
    """Check if this chunk appears in any claim judged as Support."""
    for iteration in results.get('chain_of_research_results', []):
        for claim in iteration.get('claim_results', []):
            if claim.get('final_judgment') == 'Support':
                for judged_chunk in claim.get('all_judged_chunks', []):
                    if judged_chunk.get('chunk_id') == chunk_id and judged_chunk.get('source_url') == url:
                        return True
    
    for paragraph in results.get('report_results', []):
        for claim in paragraph.get('claim_results', []):
            if claim.get('final_judgment') == 'Support':
                for judged_chunk in claim.get('all_judged_chunks', []):
                    if judged_chunk.get('chunk_id') == chunk_id and judged_chunk.get('source_url') == url:
                        return True
    
    return False


def check_chunk_in_notsupport_claims(chunk_id: str, url: str, results: Dict[str, Any]) -> bool:
    """Check if this chunk appears in any claim judged as NotSupport."""
    for iteration in results.get('chain_of_research_results', []):
        for claim in iteration.get('claim_results', []):
            if claim.get('final_judgment') == 'NotSupport':
                for judged_chunk in claim.get('all_judged_chunks', []):
                    if judged_chunk.get('chunk_id') == chunk_id and judged_chunk.get('source_url') == url:
                        return True
    
    for paragraph in results.get('report_results', []):
        for claim in paragraph.get('claim_results', []):
            if claim.get('final_judgment') == 'NotSupport':
                for judged_chunk in claim.get('all_judged_chunks', []):
                    if judged_chunk.get('chunk_id') == chunk_id and judged_chunk.get('source_url') == url:
                        return True
    
    return False


def analyze_iteration_level_nd(
    cache_file: str,
    entailed_chunk_memory: EntailedChunkMemory,
    url_to_iterations: Dict[str, List[int]],
    summary_citations: List[str],
    num_gpus: int = 1,
    logical_gpu_ids: List[int] = None
) -> Dict[str, Any]:
    """
    Analyze ND at iteration level - cluster and compute ND for each iteration separately.
    
    Args:
        cache_file: Path to cache JSON file
        entailed_chunk_memory: Memory containing entailed chunks
        url_to_iterations: Mapping from URLs to iteration indices
        summary_citations: List of cited URLs from summary
        num_gpus: Number of GPUs to use
        logical_gpu_ids: Logical GPU IDs after CUDA_VISIBLE_DEVICES mapping
    
    Returns:
        Dict containing iteration-level ND analysis results
    """
    print(f"\n📊 Starting iteration-level ND analysis...")
    
    # Load cache data to get iterations and their URLs
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading cache file: {e}")
        return {}
    
    iterations_data = cache_data.get('iterations', [])
    chunk_scores = cache_data.get('chunk_score', {})
    
    # Build reverse mapping: iteration -> list of URLs
    iteration_to_urls = {}
    for iter_idx, iteration in enumerate(iterations_data, start=1):
        search_key = f'search_list_{iter_idx}'
        if search_key in iteration:
            urls = iteration[search_key]
            # Normalize and deduplicate URLs
            normalized_urls = list(set([normalize_url_for_matching(url) for url in urls]))
            iteration_to_urls[iter_idx] = {
                'urls': urls,
                'normalized_urls': normalized_urls
            }
    
    print(f"✅ Found {len(iteration_to_urls)} iterations to analyze")
    
    # Extract all chunks with scores first (for full context)
    all_chunks = extract_chunks_with_scores(cache_file)
    if not all_chunks:
        print("❌ No chunks found")
        return {}
    
    all_chunks_avg = compute_average_scores(all_chunks)
    
    iteration_results = {}
    
    for iter_idx, iter_data in iteration_to_urls.items():
        print(f"\n  🔄 Processing Iteration {iter_idx} ({len(iter_data['urls'])} URLs)...")
        
        # Filter chunks that belong to URLs in this iteration
        iteration_chunks = []
        for chunk_id, chunk_text, url, score in all_chunks_avg:
            normalized_url = normalize_url_for_matching(url)
            # Check if this chunk's URL is in the current iteration
            if url in iter_data['urls'] or normalized_url in iter_data['normalized_urls']:
                iteration_chunks.append((chunk_id, chunk_text, url, score))
        
        if not iteration_chunks:
            print(f"    ⚠️ No chunks found for iteration {iter_idx}")
            continue
        
        print(f"    ✅ Found {len(iteration_chunks)} chunks in iteration {iter_idx}")
        
        # Cluster chunks for this iteration only
        try:
            iteration_clusters = cluster_and_rank_chunks(
                iteration_chunks,
                method="umap_hdbscan_tuned",
                num_gpus=num_gpus,
                gpu_ids=logical_gpu_ids
            )
        except Exception as e:
            print(f"    ❌ Error clustering iteration {iter_idx}: {e}")
            continue
        
        print(f"    ✅ Created {len(iteration_clusters)} clusters for iteration {iter_idx}")
        
        # Map entailed chunks to iteration-specific clusters
        iteration_mapping = map_entailed_chunks_to_clusters(
            iteration_clusters,
            entailed_chunk_memory,
            iteration_chunks
        )
        
        # Calculate ND scores for this iteration
        iteration_nd = calculate_hallucination_score_with_nd_analysis(
            iteration_mapping.get('mapped_clusters', []),
            iteration_mapping.get('unmapped_clusters', []),
            iteration_clusters,
            summary_citations,
            iteration_chunks
        )
        
        # Build detailed cluster information for this iteration
        detailed_iteration_clusters = []
        for rank, (cluster_chunks, cluster_score) in enumerate(iteration_clusters):
            cluster_detail = {
                'rank': rank,
                'score': cluster_score,
                'is_mapped': rank in iteration_mapping.get('mapped_clusters', []),
                'chunks': []
            }
            
            seen_chunks = set()
            for chunk_id, chunk_text in cluster_chunks:
                # Find URL for this chunk
                url = "unknown_url"
                for cid, ctext, curl, cscore in iteration_chunks:
                    if cid == chunk_id and ctext.strip() == chunk_text.strip():
                        url = curl
                        break
                
                chunk_key = (chunk_id, url)
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)
                
                # Check if chunk is entailed
                is_entailed = entailed_chunk_memory.is_chunk_entailed(chunk_id, url)
                in_citations = url in summary_citations
                
                cluster_detail['chunks'].append({
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'url': url,
                    'is_entailed': is_entailed,
                    'in_citations': in_citations
                })
            
            detailed_iteration_clusters.append(cluster_detail)
        
        # Store results for this iteration
        iteration_results[iter_idx] = {
            'iteration_index': iter_idx,
            'total_urls': len(iter_data['urls']),
            'total_chunks': len(iteration_chunks),
            'total_clusters': len(iteration_clusters),
            'mapped_clusters': sorted(iteration_mapping.get('mapped_clusters', [])),
            'unmapped_clusters': sorted(iteration_mapping.get('unmapped_clusters', [])),
            'mapped_cluster_count': len(iteration_mapping.get('mapped_clusters', [])),
            'unmapped_cluster_count': len(iteration_mapping.get('unmapped_clusters', [])),
            'hallucination_score': iteration_nd['wis_score'],
            'nd_analysis': iteration_nd['nd_analysis'],
            'urls': iter_data['urls'],
            'clusters': detailed_iteration_clusters
        }
        
        print(f"    ✅ Iteration {iter_idx} - HS: {iteration_nd['wis_score']:.4f}, "
              f"Mapped: {len(iteration_mapping.get('mapped_clusters', []))}, "
              f"Unmapped: {len(iteration_mapping.get('unmapped_clusters', []))}")
    
    print(f"\n✅ Iteration-level analysis complete: {len(iteration_results)} iterations processed")
    
    # Calculate aggregate statistics across iterations
    if iteration_results:
        avg_hs = sum(r['hallucination_score'] for r in iteration_results.values()) / len(iteration_results)
        total_mapped = sum(r['mapped_cluster_count'] for r in iteration_results.values())
        total_unmapped = sum(r['unmapped_cluster_count'] for r in iteration_results.values())
        
        aggregate_stats = {
            'total_iterations': len(iteration_results),
            'average_hallucination_score': avg_hs,
            'total_mapped_clusters': total_mapped,
            'total_unmapped_clusters': total_unmapped,
            'avg_clusters_per_iteration': sum(r['total_clusters'] for r in iteration_results.values()) / len(iteration_results)
        }
    else:
        aggregate_stats = {}
    
    return {
        'iteration_results': iteration_results,
        'aggregate_statistics': aggregate_stats
    }


def analyze_single_file(
    cache_file: str,
    results_file: str,
    raw_json_file: str,
    output_file: str = None,
    num_gpus: int = 4,
    gpu_ids: List[int] = None,
    logical_gpu_ids: List[int] = None
) -> Dict[str, Any]:
    """
    Analyze a single file for ND (Noise Domination) metrics.
    
    Args:
        cache_file: Path to cache JSON file
        results_file: Path to results JSON file (for reading entailed chunks)
        raw_json_file: Path to raw JSON file
        output_file: Path to output analysis file (to check existing analysis)
        num_gpus: Number of GPUs to use
        gpu_ids: List of GPU IDs to use
    
    Returns:
        Dict containing analysis results
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING: {os.path.basename(cache_file)}")
    print(f"{'='*80}")
    
    # Check if OUTPUT file already has nd_analysis to skip processing
    skip_overall = False
    skip_iteration = False
    existing_output_data = {}
    
    if output_file and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_output_data = json.load(f)
            
            # Check if overall nd_analysis exists and is non-empty in OUTPUT file
            if existing_output_data.get('nd_analysis') and len(existing_output_data.get('nd_analysis', {})) > 0:
                print("⏭️  Overall ND analysis already exists in OUTPUT file, skipping overall processing")
                skip_overall = True
            
            # Check if iteration-level nd_analysis exists and is non-empty in OUTPUT file
            if existing_output_data.get('iteration_level_nd_analysis') and len(existing_output_data.get('iteration_level_nd_analysis', {})) > 0:
                print("⏭️  Iteration-level ND analysis already exists in OUTPUT file, skipping iteration-level processing")
                skip_iteration = True
        except Exception as e:
            print(f"⚠️ Could not read existing output file: {e}")
            existing_output_data = {}
    
    # If both are already done, return early
    if skip_overall and skip_iteration:
        print("✅ All analysis already complete, skipping file")
        return None
    
    # Load summary_citations from raw JSON file
    summary_citations = load_summary_citations(raw_json_file)
    
    # Load results file and extract entailed chunks (needed for both overall and iteration-level)
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load results file: {e}")
        return None
    
    # Extract entailed claims and their chunks
    entailed_claims_data = extract_entailed_claims_and_chunks(results)
    if not entailed_claims_data:
        print("⚠️ No entailed claims found in results file")
        return None
    
    # Initialize EntailedChunkMemory and store entailed chunks
    entailed_chunk_memory = EntailedChunkMemory()
    for claim_data in entailed_claims_data:
        relevant_chunks = claim_data["relevant_chunks"]
        entailed_chunk_memory.add_entailed_chunks(relevant_chunks)
    
    print(f"✅ Found {entailed_chunk_memory.get_entailed_chunk_count()} entailed chunks")
    
    # Load URL-to-iteration mapping
    url_to_iterations = load_url_to_iteration_mapping(cache_file)
    
    # ===== OVERALL ND ANALYSIS =====
    nd_results = None
    mapping_results = None
    chunks_with_avg_scores = None
    clusters_with_scores = None
    existing_hallucination_score = None
    existing_nd_analysis = None
    
    if not skip_overall:
        print("\n🔄 Running overall ND analysis...")
        
        # Extract chunks with scores
        chunks_with_scores = extract_chunks_with_scores(cache_file)
        if not chunks_with_scores:
            print("❌ No chunks with scores were extracted. Skipping overall analysis.")
        else:
            # Compute average scores for each chunk
            chunks_with_avg_scores = compute_average_scores(chunks_with_scores)
            
            # Cluster chunks and rank by score
            clusters_with_scores = cluster_and_rank_chunks(
                chunks_with_avg_scores,
                method="umap_hdbscan_tuned",
                num_gpus=num_gpus or 1,
                gpu_ids=logical_gpu_ids
            )
            
            # Analyze entailed chunks' iteration distribution
            print(f"\n📊 Analyzing iteration distribution of entailed chunks...")
            entailed_iteration_analysis = analyze_entailed_chunks_iterations(entailed_chunk_memory, url_to_iterations)
            
            # Map entailed chunks to clusters
            mapping_results = map_entailed_chunks_to_clusters(
                clusters_with_scores,
                entailed_chunk_memory,
                chunks_with_avg_scores
            )
            
            # Calculate hallucination score with ND analysis
            nd_results = calculate_hallucination_score_with_nd_analysis(
                mapping_results.get('mapped_clusters', []),
                mapping_results.get('unmapped_clusters', []),
                clusters_with_scores,
                summary_citations,
                chunks_with_avg_scores
            )
    else:
        # Load existing overall analysis from OUTPUT file - PRESERVE existing values!
        print("📥 Loading existing overall ND analysis from OUTPUT file...")
        existing_hallucination_score = existing_output_data.get('hallucination_score')
        existing_nd_analysis = existing_output_data.get('nd_analysis', {})
        
        # Still need to extract chunks for iteration-level analysis
        chunks_with_scores = extract_chunks_with_scores(cache_file)
        if chunks_with_scores:
            chunks_with_avg_scores = compute_average_scores(chunks_with_scores)
            clusters_with_scores = cluster_and_rank_chunks(
                chunks_with_avg_scores,
                method="umap_hdbscan_tuned",
                num_gpus=num_gpus or 1,
                gpu_ids=logical_gpu_ids
            )
            entailed_iteration_analysis = analyze_entailed_chunks_iterations(entailed_chunk_memory, url_to_iterations)
            mapping_results = map_entailed_chunks_to_clusters(
                clusters_with_scores,
                entailed_chunk_memory,
                chunks_with_avg_scores
            )
    
    # ===== ITERATION-LEVEL ND ANALYSIS =====
    iteration_level_nd = None
    if not skip_iteration:
        print("\n🔄 Running iteration-level ND analysis...")
        iteration_level_nd = analyze_iteration_level_nd(
            cache_file,
            entailed_chunk_memory,
            url_to_iterations,
            summary_citations,
            num_gpus=num_gpus or 1,
            logical_gpu_ids=logical_gpu_ids
        )
    else:
        # Load existing iteration-level analysis from OUTPUT file - PRESERVE existing values!
        print("📥 Loading existing iteration-level ND analysis from OUTPUT file...")
        iteration_level_nd = existing_output_data.get('iteration_level_nd_analysis', {})
    
    # If both analyses were skipped and loaded from existing results, we still need to build the output
    if skip_overall and not chunks_with_avg_scores:
        # Extract chunks for building detailed clusters
        chunks_with_scores = extract_chunks_with_scores(cache_file)
        if chunks_with_scores:
            chunks_with_avg_scores = compute_average_scores(chunks_with_scores)
            clusters_with_scores = cluster_and_rank_chunks(
                chunks_with_avg_scores,
                method="umap_hdbscan_tuned",
                num_gpus=num_gpus or 1,
                gpu_ids=logical_gpu_ids
            )
            entailed_iteration_analysis = analyze_entailed_chunks_iterations(entailed_chunk_memory, url_to_iterations)
            mapping_results = map_entailed_chunks_to_clusters(
                clusters_with_scores,
                entailed_chunk_memory,
                chunks_with_avg_scores
            )
    
    # Safety check
    if not chunks_with_avg_scores or not clusters_with_scores:
        print("❌ Failed to extract or cluster chunks")
        return None
    
    # Build (chunk_id, chunk_text) -> (url, score) mapping
    # chunk_id alone is not unique across URLs, so we need both chunk_id and chunk_text
    chunk_key_to_info = {}
    for chunk_id, chunk_text, url, score in chunks_with_avg_scores:
        chunk_key = (chunk_id, chunk_text.strip())
        chunk_key_to_info[chunk_key] = {
            'url': url,
            'score': score,
            'chunk_text': chunk_text.strip()
        }
    
    # Load cache file to extract chunk features and build URL index to URL mapping
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        chunk_scores = cache_data.get('chunk_score', {})
        
        # Build URL index to URL mapping from chunk_score keys
        # Format: "url_index-chunk_id" (e.g., "0-chunk_0")
        url_index_to_url = {}
        for chunk_key_str, chunk_info in chunk_scores.items():
            # Parse chunk_key like "0-chunk_0" to get url_index
            import re
            match = re.match(r'^(\d+)-chunk_', chunk_key_str)
            if match:
                url_index = int(match.group(1))
                url = chunk_info.get('url', '')
                if url and url_index not in url_index_to_url:
                    url_index_to_url[url_index] = url
        
        # Also build reverse mapping: URL -> url_index
        url_to_url_index = {url: idx for idx, url in url_index_to_url.items()}
        
    except Exception as e:
        print(f"⚠️ Error loading cache file for chunk features: {e}")
        chunk_scores = {}
        url_index_to_url = {}
        url_to_url_index = {}
    
    # Build detailed cluster information with iteration data and features
    mapped_chunks_features = []
    unmapped_chunks_features = []
    
    detailed_clusters = []
    for rank, (cluster_chunks, cluster_score) in enumerate(clusters_with_scores):
        cluster_details = {
            'rank': rank,
            'score': cluster_score,
            'is_mapped': rank in mapping_results.get('mapped_clusters', []),
            'chunks': []
        }
        # Deduplicate chunks within the same cluster by (chunk_id, url)
        seen_chunk_keys = set()

        for chunk_id, chunk_text in cluster_chunks:
            # Find URL by matching (chunk_id, chunk_text) tuple
            chunk_key = (chunk_id, chunk_text.strip())
            chunk_info = chunk_key_to_info.get(chunk_key)
            
            if not chunk_info:
                # Try to find by matching chunk_text only if exact match fails
                # This can happen if text has minor whitespace differences
                url = "unknown_url"
                for (cid, ctext), info in chunk_key_to_info.items():
                    if cid == chunk_id and ctext.strip() == chunk_text.strip():
                        url = info['url']
                        chunk_info = info
                        break
            else:
                url = chunk_info['url']
            
            normalized_url = normalize_url_for_matching(url)

            # Skip duplicates of the same (chunk_id, url) within this cluster
            dedup_key = (chunk_id, url)
            if dedup_key in seen_chunk_keys:
                continue
            seen_chunk_keys.add(dedup_key)
            
            # Find iteration(s) for this URL
            iterations = url_to_iterations.get(normalized_url, [])
            if not iterations:
                # Try exact URL match
                iterations = url_to_iterations.get(url, [])
            # Choose a single canonical iteration index (earliest appearance)
            iteration_index = min(iterations) if iterations else None
            
            # Get chunk data from cache using url_index-chunk_id format
            chunk_data = None
            if url in url_to_url_index:
                url_index = url_to_url_index[url]
                chunk_key_str = f"{url_index}-{chunk_id}"
                chunk_data = chunk_scores.get(chunk_key_str)
            
            # If not found, try to find by matching chunk_id and url in chunk_info
            if not chunk_data:
                for chunk_key_str, chunk_info_cache in chunk_scores.items():
                    if (chunk_info_cache.get('url') == url and 
                        chunk_info_cache.get('chunk_id_original') == chunk_id):
                        chunk_data = chunk_info_cache
                        break
            
            # Extract features
            features = extract_chunk_features(
                chunk_data if chunk_data else {},
                chunk_id,
                url,
                cache_file,
                chunk_scores,
                chunk_text,
                url_to_url_index=url_to_url_index
            )
            # Store single iteration and a single-element list for compatibility
            features['iteration'] = iteration_index
            features['iterations'] = [iteration_index] if iteration_index is not None else []
            features['is_entailed'] = entailed_chunk_memory.is_chunk_entailed(chunk_id, url)
            features['in_citations'] = url in summary_citations
            
            cluster_details['chunks'].append({
                'chunk_id': chunk_id,
                'chunk_text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                'url': url,
                'iteration': iteration_index,
                'iterations': [iteration_index] if iteration_index is not None else [],
                'is_entailed': features['is_entailed'],
                'in_citations': features['in_citations'],
                'features': features
            })
            
            # Separate mapped and unmapped chunks for feature analysis
            if features['is_entailed']:
                mapped_chunks_features.append(features)
            else:
                unmapped_chunks_features.append(features)
        
        detailed_clusters.append(cluster_details)
    
    # Analyze features for mapped vs unmapped chunks
    feature_comparison = compare_mapped_unmapped_features(mapped_chunks_features, unmapped_chunks_features)
    
    # Build comprehensive analysis results
    # Use existing values if overall analysis was skipped, otherwise use newly computed values
    if skip_overall and existing_hallucination_score is not None:
        # Use existing values - DO NOT overwrite!
        hallucination_score = existing_hallucination_score
        nd_analysis_data = existing_nd_analysis
        print(f"📥 Using existing hallucination_score: {hallucination_score}")
        
        # Also preserve other existing fields if they exist from OUTPUT file
        existing_clusters = existing_output_data.get('clusters', detailed_clusters)
        existing_mapped = existing_output_data.get('mapped_clusters', sorted(mapping_results.get('mapped_clusters', [])) if mapping_results else [])
        existing_unmapped = existing_output_data.get('unmapped_clusters', sorted(mapping_results.get('unmapped_clusters', [])) if mapping_results else [])
        existing_total_clusters = existing_output_data.get('total_clusters', len(clusters_with_scores))
        existing_total_mapped = existing_output_data.get('total_mapped', len(mapping_results.get('mapped_clusters', [])) if mapping_results else 0)
        existing_total_unmapped = existing_output_data.get('total_unmapped', len(mapping_results.get('unmapped_clusters', [])) if mapping_results else 0)
        existing_entailed_iter = existing_output_data.get('entailed_iteration_analysis', entailed_iteration_analysis)
        existing_feature_comp = existing_output_data.get('mapped_unmapped_feature_comparison', feature_comparison)
        existing_iter_stats = existing_output_data.get('iteration_statistics', None)
        existing_citation_stats = existing_output_data.get('citation_statistics', None)
    elif nd_results and isinstance(nd_results, dict):
        # Use newly computed values
        hallucination_score = nd_results.get('wis_score', 0)
        nd_analysis_data = nd_results.get('nd_analysis', {})
        existing_clusters = detailed_clusters
        existing_mapped = sorted(mapping_results.get('mapped_clusters', [])) if mapping_results else []
        existing_unmapped = sorted(mapping_results.get('unmapped_clusters', [])) if mapping_results else []
        existing_total_clusters = len(clusters_with_scores)
        existing_total_mapped = len(mapping_results.get('mapped_clusters', [])) if mapping_results else 0
        existing_total_unmapped = len(mapping_results.get('unmapped_clusters', [])) if mapping_results else 0
        existing_entailed_iter = entailed_iteration_analysis
        existing_feature_comp = feature_comparison
        existing_iter_stats = None
        existing_citation_stats = None
    elif isinstance(nd_results, dict):
        hallucination_score = nd_results.get('wis_score', 0)
        nd_analysis_data = nd_results
        existing_clusters = detailed_clusters
        existing_mapped = sorted(mapping_results.get('mapped_clusters', [])) if mapping_results else []
        existing_unmapped = sorted(mapping_results.get('unmapped_clusters', [])) if mapping_results else []
        existing_total_clusters = len(clusters_with_scores)
        existing_total_mapped = len(mapping_results.get('mapped_clusters', [])) if mapping_results else 0
        existing_total_unmapped = len(mapping_results.get('unmapped_clusters', [])) if mapping_results else 0
        existing_entailed_iter = entailed_iteration_analysis
        existing_feature_comp = feature_comparison
        existing_iter_stats = None
        existing_citation_stats = None
    else:
        hallucination_score = 0
        nd_analysis_data = {}
        existing_clusters = detailed_clusters
        existing_mapped = []
        existing_unmapped = []
        existing_total_clusters = len(clusters_with_scores) if clusters_with_scores else 0
        existing_total_mapped = 0
        existing_total_unmapped = 0
        existing_entailed_iter = entailed_iteration_analysis
        existing_feature_comp = feature_comparison
        existing_iter_stats = None
        existing_citation_stats = None
    
    analysis_results = {
        'file_id': os.path.basename(cache_file).replace('cache_', '').replace('.json', ''),
        'hallucination_score': hallucination_score if hallucination_score is not None else 0,
        'nd_analysis': nd_analysis_data,
        'clusters': existing_clusters,
        'mapped_clusters': existing_mapped,
        'unmapped_clusters': existing_unmapped,
        'total_clusters': existing_total_clusters,
        'total_mapped': existing_total_mapped,
        'total_unmapped': existing_total_unmapped,
        'total_entailed_chunks': entailed_chunk_memory.get_entailed_chunk_count(),
        'entailed_iteration_analysis': existing_entailed_iter,
        'mapped_unmapped_feature_comparison': existing_feature_comp,
        'iteration_level_nd_analysis': iteration_level_nd if iteration_level_nd else {}
    }
    
    # Calculate iteration statistics (use existing if available)
    if existing_iter_stats is not None:
        analysis_results['iteration_statistics'] = existing_iter_stats
    else:
        iteration_stats = calculate_iteration_statistics(detailed_clusters)
        analysis_results['iteration_statistics'] = iteration_stats
    
    # Calculate citation statistics (use existing if available)
    if existing_citation_stats is not None:
        analysis_results['citation_statistics'] = existing_citation_stats
    else:
        citation_stats = calculate_citation_statistics(detailed_clusters)
        analysis_results['citation_statistics'] = citation_stats
    
    print(f"✅ Analysis complete: Hallucination Score = {analysis_results['hallucination_score']:.4f}")
    
    # Print iteration distribution summary
    if 'entailed_iteration_analysis' in analysis_results:
        iteration_analysis = analysis_results['entailed_iteration_analysis']
        print(f"\n📊 Entailed Chunks Iteration Analysis:")
        print(f"  - Total entailed chunks: {iteration_analysis['total_entailed_chunks']}")
        print(f"  - Chunks with iterations: {iteration_analysis['chunks_with_iterations']}")
        print(f"  - Chunks without iterations: {iteration_analysis['chunks_without_iterations']}")
        if iteration_analysis['iteration_distribution']:
            print(f"  - Iteration distribution: {dict(iteration_analysis['iteration_distribution'])}")
    
    # Print iteration-level ND analysis summary
    if 'iteration_level_nd_analysis' in analysis_results and analysis_results['iteration_level_nd_analysis']:
        iter_nd = analysis_results['iteration_level_nd_analysis']
        if 'aggregate_statistics' in iter_nd and iter_nd['aggregate_statistics']:
            agg_stats = iter_nd['aggregate_statistics']
            print(f"\n🔬 Iteration-Level ND Analysis Summary:")
            print(f"  - Total iterations analyzed: {agg_stats.get('total_iterations', 0)}")
            print(f"  - Average hallucination score per iteration: {agg_stats.get('average_hallucination_score', 0):.4f}")
            print(f"  - Total mapped clusters across iterations: {agg_stats.get('total_mapped_clusters', 0)}")
            print(f"  - Total unmapped clusters across iterations: {agg_stats.get('total_unmapped_clusters', 0)}")
            print(f"  - Average clusters per iteration: {agg_stats.get('avg_clusters_per_iteration', 0):.2f}")
    
    # Print feature comparison summary
    if 'mapped_unmapped_feature_comparison' in analysis_results:
        feature_comp = analysis_results['mapped_unmapped_feature_comparison']
        if 'top_distinguishing_features' in feature_comp:
            print(f"\n🔍 Top Distinguishing Features (mapped vs unmapped chunks):")
            for i, feat_info in enumerate(feature_comp['top_distinguishing_features'][:3], 1):
                print(f"  {i}. {feat_info['feature']}: difference = {feat_info['difference']:.4f}")
                print(f"     Mapped avg = {feat_info['stats'].get('mapped_avg', feat_info['stats'].get('mapped_percentage', 0)):.4f}")
                print(f"     Unmapped avg = {feat_info['stats'].get('unmapped_avg', feat_info['stats'].get('unmapped_percentage', 0)):.4f}")
    
    return analysis_results


def calculate_iteration_statistics(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about iteration distribution across clusters.
    """
    all_iterations = []
    cluster_iteration_counts = []
    
    for cluster in clusters:
        cluster_iterations = []
        for chunk in cluster['chunks']:
            if 'iteration' in chunk and chunk['iteration'] is not None:
                cluster_iterations.append(chunk['iteration'])
            else:
                cluster_iterations.extend(chunk.get('iterations', []))
        
        if cluster_iterations:
            all_iterations.extend(cluster_iterations)
            # Count unique iterations per cluster
            unique_iterations = len(set(cluster_iterations))
            cluster_iteration_counts.append({
                'cluster_rank': cluster['rank'],
                'unique_iterations': unique_iterations,
                'total_urls': len(cluster['chunks']),
                'iteration_list': list(set(cluster_iterations))
            })
    
    if not all_iterations:
        return {
            'total_unique_iterations': 0,
            'avg_iterations_per_cluster': 0,
            'iteration_frequency': {},
            'clusters_with_no_iterations': len([c for c in clusters if not any(ch.get('iterations') for ch in c['chunks'])]),
            'cluster_details': cluster_iteration_counts
        }
    
    # Calculate frequency of each iteration
    iteration_frequency = {}
    for iteration in all_iterations:
        iteration_frequency[iteration] = iteration_frequency.get(iteration, 0) + 1
    
    return {
        'total_unique_iterations': len(set(all_iterations)),
        'max_iteration': max(all_iterations) if all_iterations else 0,
        'min_iteration': min(all_iterations) if all_iterations else 0,
        'avg_iterations_per_cluster': len(all_iterations) / len(clusters) if clusters else 0,
        'iteration_frequency': iteration_frequency,
        'clusters_with_iterations': len([c for c in clusters if any(ch.get('iterations') for ch in c['chunks'])]),
        'clusters_without_iterations': len([c for c in clusters if not any(ch.get('iterations') for ch in c['chunks'])]),
        'cluster_details': cluster_iteration_counts
    }


def calculate_citation_statistics(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about citation coverage across clusters.
    """
    total_chunks = 0
    chunks_in_citations = 0
    total_urls = 0
    urls_in_citations = 0
    
    for cluster in clusters:
        cluster_urls = set()
        cluster_urls_in_citations = 0
        
        for chunk in cluster['chunks']:
            total_chunks += 1
            cluster_urls.add(chunk['url'])
            
            if chunk.get('in_citations', False):
                chunks_in_citations += 1
                cluster_urls_in_citations += 1
        
        total_urls += len(cluster_urls)
        if cluster_urls_in_citations > 0:
            urls_in_citations += 1
    
    return {
        'total_chunks': total_chunks,
        'chunks_in_citations': chunks_in_citations,
        'citation_coverage': chunks_in_citations / total_chunks if total_chunks > 0 else 0,
        'total_unique_urls': total_urls,
        'clusters_with_cited_urls': urls_in_citations,
        'clusters_without_cited_urls': len(clusters) - urls_in_citations
    }


def process_all_files(
    cache_dir: str,
    results_dir: str,
    raw_json_dir: str,
    output_dir: str,
    num_gpus: int = 4,
    gpu_ids: List[int] = None,
    logical_gpu_ids: List[int] = None
):
    """
    Process all files in the directories.
    
    Args:
        cache_dir: Directory containing cache JSON files
        results_dir: Directory containing results JSON files
        raw_json_dir: Directory containing raw JSON files
        output_dir: Directory to save analysis results
        num_gpus: Number of GPUs to use
        gpu_ids: List of GPU IDs to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all cache files
    cache_files = sorted(Path(cache_dir).glob('cache_*.json'))
    print(f"🔍 Found {len(cache_files)} cache files to process")

    # ONLY process the first 5 cache files for testing
    # cache_files = cache_files[:2]
    
    all_results = []
    
    for cache_file in cache_files:
        file_id = cache_file.stem.replace('cache_', '')
        
        # Determine corresponding files
        # results_file = Path(results_dir) / f'results_{file_id}.json'
        results_file = Path(results_dir) / f'{file_id}_combined.json'
        raw_json_file = Path(raw_json_dir) / f'{file_id}.json'
        
        # Check if required files exist
        if not results_file.exists():
            print(f"⚠️ Skipping {file_id}: results file not found")
            continue
        if not raw_json_file.exists():
            print(f"⚠️ Skipping {file_id}: raw JSON file not found")
            continue
        
        # Determine output file path
        output_file = Path(output_dir) / f'analysis_{file_id}.json'
        
        # Process the file (pass output_file to check for existing analysis)
        try:
            analysis_result = analyze_single_file(
                str(cache_file),
                str(results_file),
                str(raw_json_file),
                output_file=str(output_file),
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
                logical_gpu_ids=logical_gpu_ids
            )
            
            if analysis_result:
                all_results.append(analysis_result)
                
                # Save individual result (output_file already defined above)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result, f, indent=2, ensure_ascii=False)
                print(f"💾 Saved analysis to: {output_file}")
                
        except Exception as e:
            print(f"❌ Error processing {file_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate summary report
    generate_summary_report(all_results, output_dir)
    
    print(f"\n✅ Processing complete! Processed {len(all_results)} files")


def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: str):
    """
    Generate a summary report with aggregated statistics.
    """
    if not all_results:
        print("⚠️ No results to generate summary report")
        return
    
    summary = {
        'total_files_processed': len(all_results),
        'files': {}
    }
    
    # Aggregate statistics
    total_hallucination_score = 0
    total_document_nd_all = 0
    total_document_nd_partial = 0
    total_chunk_nd = 0
    total_nd = 0
    entailed_iterations_all = []
    all_distinguishing_features = {}
    
    for result in all_results:
        file_id = result['file_id']
        
        # Store individual file summary
        hs_val = result.get('hallucination_score', 0)
        nd_data = result.get('nd_analysis', {})
        summary['files'][file_id] = {
            'hallucination_score': hs_val if hs_val is not None else 0,
            'total_clusters': result.get('total_clusters', 0),
            'total_mapped': result.get('total_mapped', 0),
            'total_unmapped': result.get('total_unmapped', 0),
            'total_entailed_chunks': result.get('total_entailed_chunks', 0),
            'nd_all_ratio': nd_data.get('document_level_nd_all_ratio', 0) if nd_data else 0,
            'nd_partial_ratio': nd_data.get('document_level_nd_partial_ratio', 0) if nd_data else 0,
            'nd_chunk_ratio': nd_data.get('chunk_level_nd_ratio', 0) if nd_data else 0
        }
        
        # Collect entailed iteration data
        if 'entailed_iteration_analysis' in result:
            iter_analysis = result['entailed_iteration_analysis']
            for iteration, count in iter_analysis.get('iteration_distribution', []):
                entailed_iterations_all.extend([iteration] * count)
        
        # Collect distinguishing features
        if 'mapped_unmapped_feature_comparison' in result:
            feature_comp = result['mapped_unmapped_feature_comparison']
            for feat_info in feature_comp.get('top_distinguishing_features', []):
                feat_name = feat_info['feature']
                if feat_name not in all_distinguishing_features:
                    all_distinguishing_features[feat_name] = {'differences': [], 'count': 0}
                all_distinguishing_features[feat_name]['differences'].append(feat_info['difference'])
                all_distinguishing_features[feat_name]['count'] += 1
        
        # Accumulate for global averages
        hs_value = result.get('hallucination_score', 0)
        total_hallucination_score += hs_value if hs_value is not None else 0
        
        nd_analysis = result.get('nd_analysis', {})
        if nd_analysis:
            total_document_nd_all += nd_analysis.get('document_level_nd_all', 0)
            total_document_nd_partial += nd_analysis.get('document_level_nd_partial', 0)
            total_chunk_nd += nd_analysis.get('chunk_level_nd', 0)
            total_nd += nd_analysis.get('total_nd', 0)
    
    # Calculate averages
    summary['average_hallucination_score'] = total_hallucination_score / len(all_results) if all_results else 0
    
    if total_nd > 0:
        summary['average_nd_ratios'] = {
            'document_level_nd_all': total_document_nd_all / total_nd,
            'document_level_nd_partial': total_document_nd_partial / total_nd,
            'chunk_level_nd': total_chunk_nd / total_nd
        }
    
    # Calculate iteration distribution across all files
    if entailed_iterations_all:
        iteration_counter = Counter(entailed_iterations_all)
        summary['overall_iteration_distribution'] = dict(sorted(iteration_counter.items()))
        summary['total_entailed_chunks_with_iterations'] = len(entailed_iterations_all)
    
    # Calculate average distinguishing features across all files
    if all_distinguishing_features:
        summary['overall_distinguishing_features'] = {}
        for feat_name, feat_data in all_distinguishing_features.items():
            avg_diff = sum(feat_data['differences']) / len(feat_data['differences']) if feat_data['differences'] else 0
            summary['overall_distinguishing_features'][feat_name] = {
                'average_difference': avg_diff,
                'files_with_feature': feat_data['count'],
                'total_occurrences': len(feat_data['differences'])
            }
    
    # Save summary report
    summary_file = Path(output_dir) / 'summary_report.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Summary Report")
    print(f"=" * 80)
    print(f"Total files processed: {len(all_results)}")
    print(f"Average hallucination score: {summary['average_hallucination_score']:.4f}")
    if 'average_nd_ratios' in summary:
        print(f"Average ND ratios:")
        print(f"  - Document-level ND (All): {summary['average_nd_ratios']['document_level_nd_all']:.4f}")
        print(f"  - Document-level ND (Partial): {summary['average_nd_ratios']['document_level_nd_partial']:.4f}")
        print(f"  - Chunk-level ND: {summary['average_nd_ratios']['chunk_level_nd']:.4f}")
    
    if 'overall_iteration_distribution' in summary:
        print(f"\n📊 Overall Entailed Chunks Iteration Distribution:")
        print(f"  Total entailed chunks with iterations: {summary['total_entailed_chunks_with_iterations']}")
        print(f"  Distribution: {summary['overall_iteration_distribution']}")
    
    if 'overall_distinguishing_features' in summary:
        print(f"\n🔍 Overall Distinguishing Features:")
        sorted_features = sorted(summary['overall_distinguishing_features'].items(), 
                                key=lambda x: x[1]['average_difference'], reverse=True)
        for feat_name, feat_data in sorted_features[:3]:
            print(f"  - {feat_name}: avg_diff={feat_data['average_difference']:.4f} (in {feat_data['files_with_feature']} files)")
    
    print(f"\n💾 Saved summary to: {summary_file}")


def main():
    """Main entry point for the script."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Standalone ND Analysis Script'
    )
    # parser.add_argument('--cache_dir', type=str,
    #                    default='/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/reframe',
    #                    help='Directory containing cache JSON files')
    parser.add_argument('--cache_dir', type=str,
                       default='/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/updated_cache',
                       help='Directory containing cache JSON files')
    # parser.add_argument('--results_dir', type=str,
    #                    default='/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe',
    #                    help='Directory containing results JSON files')
    parser.add_argument('--results_dir', type=str,
                       default='/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/updated_whole_results',
                       help='Directory containing results JSON files')
    parser.add_argument('--raw_json_dir', type=str,
                       default='/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/Tianyu_ReportBench/json_new_summary_cite',
                       help='Directory containing raw JSON files')
    parser.add_argument('--output_dir', type=str,
                       default='/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/nd_analysis_results/updated_claim_cache_json',
                       help='Directory to save analysis results')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3",
                       help='Comma-separated GPU IDs (e.g., "0,1,2,3")')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        except ValueError:
            print("⚠️ Invalid GPU IDs format. Using default [0,1,2,3]")
            gpu_ids = [2, 3]
    
    # Set CUDA_VISIBLE_DEVICES to restrict visible GPUs to the specified ones
    # This is required by chunk_clustering.py which expects logical GPU IDs (0, 1, ...)
    # After setting, PyTorch will remap physical GPU IDs to logical IDs (0, 1, ...)
    if gpu_ids:
        gpu_ids_str = ','.join(map(str, gpu_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        print(f"🎯 Set CUDA_VISIBLE_DEVICES={gpu_ids_str}")
        print(f"   Physical GPUs {gpu_ids} will be mapped to logical GPUs 0-{len(gpu_ids)-1}")
        # After setting CUDA_VISIBLE_DEVICES, all subsequent GPU references should use logical IDs (0, 1, ...)
        # So we need to update gpu_ids for functions that expect logical IDs
        logical_gpu_ids = list(range(len(gpu_ids)))
    else:
        logical_gpu_ids = None
    
    # Set HF_ENDPOINT environment variable
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"🔧 Set HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    
    # Process all files
    process_all_files(
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        raw_json_dir=args.raw_json_dir,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        gpu_ids=gpu_ids,
        logical_gpu_ids=logical_gpu_ids
    )


if __name__ == '__main__':
    main()

