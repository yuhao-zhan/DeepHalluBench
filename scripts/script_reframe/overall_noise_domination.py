import json
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Any
from chunk_clustering import extract_chunk_texts, cluster_chunks_umap_hdbscan_tuned
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp


class EntailedChunkMemory:
    """Memory to track chunks that have been previously verified as Support for claims."""
    
    def __init__(self):
        # Store chunks by their unique identifiers (url + chunk_id)
        self.entailed_chunks: Set[str] = set()
        # Store detailed chunk information for matching
        self.entailed_chunk_details: Dict[str, Dict[str, Any]] = {}
    
    def add_entailed_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            # Check if chunk has any positive score (score, rerank_score, or similarity_score)
            # OR if it has a positive confidence (for entailed chunks from results file)
            score = chunk.get('score', 0)
            rerank_score = chunk.get('rerank_score', 0)
            similarity_score = chunk.get('similarity_score', 0)
            confidence = chunk.get('confidence', 0)
            
            # Add chunk if it has any positive score OR positive confidence
            if score > 0 or rerank_score > 0 or similarity_score > 0 or confidence > 0:
                # chunk_id format is {url_index}-chunk_{chunk_index} (e.g., "18-chunk_6")
                # This is already unique within a file, so we use it directly as the key
                chunk_id = chunk.get('chunk_id', '')
                url = chunk.get('source_url', '')
                if chunk_id:
                    # Use chunk_id directly as the key (it's already unique within the file)
                    self.entailed_chunks.add(chunk_id)
                    self.entailed_chunk_details[chunk_id] = {
                        'chunk_id': chunk_id,
                        'source_url': url,
                        'chunk_text': chunk.get('chunk_text', ''),
                        'score': score,
                        'rerank_score': rerank_score,
                        'similarity_score': similarity_score,
                        'confidence': confidence
                    }
    
    def is_chunk_entailed(self, chunk_id: str, url: str = None) -> bool:
        """
        Check if a chunk is entailed.
        
        Args:
            chunk_id: The chunk ID (format: {url_index}-chunk_{chunk_index}, e.g., "18-chunk_6")
            url: Optional URL (kept for backward compatibility, but not used since chunk_id is unique)
        
        Returns:
            True if the chunk is entailed, False otherwise
        """
        # chunk_id format is {url_index}-chunk_{chunk_index}, which is unique within a file
        # So we can directly check if it's in entailed_chunks
        return chunk_id in self.entailed_chunks
    
    def get_entailed_chunk_details(self) -> Dict[str, Dict[str, Any]]:
        return self.entailed_chunk_details.copy()
    
    def get_entailed_chunk_count(self) -> int:
        return len(self.entailed_chunks)


def extract_chunks_with_scores(json_file_path: str) -> List[Tuple[str, str, str, Dict[str, float]]]:
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks_with_scores = []
        
        if 'chunk_score' in data:
            chunk_score_data = data['chunk_score']
            
            for chunk_key, chunk_info in chunk_score_data.items():
                if 'chunk_text' in chunk_info and 'scores' in chunk_info and 'url' in chunk_info:
                    chunk_text = chunk_info['chunk_text']
                    url = chunk_info['url']
                    scores = chunk_info['scores']
                    
                    # Extract combined scores for each query
                    combined_scores = {}
                    for query_id, query_scores in scores.items():
                        if 'combined' in query_scores:
                            combined_scores[query_id] = query_scores['combined']
                    
                    # Keep the original chunk_key WITH prefix (e.g., "0-chunk_0", "44-chunk_5")
                    # This matches the format used in claim verification results
                    chunk_id = chunk_key
                    
                    chunks_with_scores.append((chunk_id, chunk_text, url, combined_scores))
                    # print(f"Extracted chunk {len(chunks_with_scores)}: {chunk_id} | URL: {url}")
                else:
                    print(f"Warning: Missing chunk_text, scores, or url for {chunk_key}")
        else:
            print("Error: 'chunk_score' attribute not found in the JSON file")
            return []
        
        return chunks_with_scores
        
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []


def compute_average_scores(chunks_with_scores: List[Tuple[str, str, str, Dict[str, float]]]) -> List[Tuple[str, str, str, float]]:
    chunks_with_avg_scores = []
    
    for chunk_id, chunk_text, url, scores in chunks_with_scores:
        if scores:
            # Calculate average of all combined scores
            avg_score = sum(scores.values()) / len(scores)
            chunks_with_avg_scores.append((chunk_id, chunk_text, url, avg_score))
        else:
            # If no scores, assign 0
            chunks_with_avg_scores.append((chunk_id, chunk_text, url, 0.0))
    
    # Sort by average score in descending order
    chunks_with_avg_scores.sort(key=lambda x: x[3], reverse=True)
    
    return chunks_with_avg_scores


def cluster_and_rank_chunks(chunks_with_avg_scores: List[Tuple[str, str, str, float]], 
                           method: str = "umap_hdbscan_tuned",
                           num_gpus: int = 1,
                           gpu_ids: List[int] = None) -> List[Tuple[List[tuple], float]]:
    chunks_for_clustering = []
    chunk_key_to_score = {}  # Map (chunk_id, url) to score for later lookup
    chunk_key_to_url = {}    # Map (chunk_id, url) to URL for later lookup
    
    for chunk_id, chunk_text, url, score in chunks_with_avg_scores:
        chunks_for_clustering.append((chunk_id, chunk_text))
        chunk_key = f"{url}_{chunk_id}"
        chunk_key_to_score[chunk_key] = score
        chunk_key_to_url[chunk_key] = url
    
    print(f"Clustering {len(chunks_for_clustering)} chunks using {method} method...")
    clusters = cluster_chunks_umap_hdbscan_tuned(chunks_for_clustering, num_gpus=num_gpus, gpu_ids=gpu_ids)
    
    # For each cluster, find the highest average score among its chunks
    clusters_with_scores = []
    
    for cluster_idx, cluster in enumerate(clusters):
        # Find the highest score among chunks in this cluster
        cluster_score = 0.0
        print(f"\nCluster {cluster_idx + 1}:")
        for chunk_id, chunk_text in cluster:
            # Find the URL for this chunk by matching chunk_id and text
            # This is not ideal but necessary due to clustering limitations
            url = "unknown_url"
            for orig_chunk_id, orig_chunk_text, orig_url, orig_score in chunks_with_avg_scores:
                if (chunk_id == orig_chunk_id and 
                    chunk_text.strip() == orig_chunk_text.strip()):
                    url = orig_url
                    score = orig_score
                    break
            else:
                score = 0.0
            
            # Use the max score among chunks in this cluster
            cluster_score = max(cluster_score, score)
        
        clusters_with_scores.append((cluster, cluster_score))
    
    # Sort clusters by descending score
    clusters_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    return clusters_with_scores


def save_ranked_clusters(clusters_with_scores: List[Tuple[List[tuple], float]], 
                        output_file_path: str):
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("RANKED CLUSTERS BY SCORE (DESCENDING ORDER)\n")
            f.write("=" * 60 + "\n\n")
            
            for i, (cluster, score) in enumerate(clusters_with_scores):
                f.write(f"=== CLUSTER {i+1} (Score: {score:.6f}) ===\n")
                for j, (chunk_id, chunk_text) in enumerate(cluster):
                    f.write(f"{j+1}. Chunk ID: {chunk_id}, Text: {chunk_text}\n")
                f.write("\n")
        
        print(f"Ranked clusters saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving to file: {e}")


def print_cluster_summary(clusters_with_scores: List[Tuple[List[tuple], float]]):
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY (Ranked by Score)")
    print("=" * 60)
    
    total_chunks = sum(len(cluster) for cluster, _ in clusters_with_scores)
    
    for i, (cluster, score) in enumerate(clusters_with_scores):
        print(f"Cluster {i+1}: {len(cluster)} chunks, Score: {score:.6f}")
    
    print(f"\nTotal clusters: {len(clusters_with_scores)}")
    print(f"Total chunks: {total_chunks}")
    
    if clusters_with_scores:
        print(f"Highest cluster score: {clusters_with_scores[0][1]:.6f}")
        print(f"Lowest cluster score: {clusters_with_scores[-1][1]:.6f}")


def print_detailed_cluster_debug(clusters_with_scores: List[Tuple[List[tuple], float]], 
                                mapped_clusters: List[int], unmapped_clusters: List[int],
                                chunks_with_avg_scores: List[Tuple[str, str, str, float]]):
    """Print detailed debug information about clusters for debugging high hallucination scores."""
    print("\n" + "=" * 80)
    print("DETAILED CLUSTER DEBUG INFORMATION")
    print("=" * 80)
    
    print(f"📊 Total clusters: {len(clusters_with_scores)}")
    print(f"📊 Mapped clusters: {len(mapped_clusters)} - {mapped_clusters}")
    print(f"📊 Unmapped clusters: {len(unmapped_clusters)} - {unmapped_clusters}")
    
    print(f"\n🔍 DETAILED CLUSTER RANKINGS:")
    print("-" * 80)
    
    for rank, (cluster_chunks, cluster_score) in enumerate(clusters_with_scores):
        is_mapped = rank in mapped_clusters
        is_unmapped = rank in unmapped_clusters
        status = "MAPPED" if is_mapped else "UNMAPPED"
        
        for i, (chunk_id, chunk_text) in enumerate(cluster_chunks):
            # Find the URL for this chunk
            url = "unknown_url"
            for orig_chunk_id, orig_chunk_text, orig_url, orig_score in chunks_with_avg_scores:
                if (chunk_id == orig_chunk_id and 
                    chunk_text.strip() == orig_chunk_text.strip()):
                    url = orig_url
                    break
            
            # Truncate chunk text for readability
            display_text = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            # print(f"   {i+1}. Chunk ID: {chunk_id}")
            # print(f"      URL: {url}")
            # print(f"      Text: {display_text}")
            # print()
    
    for rank in mapped_clusters:
        if rank < len(clusters_with_scores):
            cluster_chunks, cluster_score = clusters_with_scores[rank]
            for i, (chunk_id, chunk_text) in enumerate(cluster_chunks[:3]):  # Show first 3 chunks
                # Find the URL for this chunk
                url = "unknown_url"
                for orig_chunk_id, orig_chunk_text, orig_url, orig_score in chunks_with_avg_scores:
                    if (chunk_id == orig_chunk_id and 
                        chunk_text.strip() == orig_chunk_text.strip()):
                        url = orig_url
                        break
    
    for rank in unmapped_clusters:
        if rank < len(clusters_with_scores):
            cluster_chunks, cluster_score = clusters_with_scores[rank]
            for i, (chunk_id, chunk_text) in enumerate(cluster_chunks[:3]):  # Show first 3 chunks
                # Find the URL for this chunk
                url = "unknown_url"
                for orig_chunk_id, orig_chunk_text, orig_url, orig_score in chunks_with_avg_scores:
                    if (chunk_id == orig_chunk_id and 
                        chunk_text.strip() == orig_chunk_text.strip()):
                        url = orig_url
                        break
                
                display_text = chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
                # print(f"   {i+1}. {chunk_id} | {url} | {display_text}")
            # if len(cluster_chunks) > 3:
            #     print(f"   ... and {len(cluster_chunks) - 3} more chunks")
            # print()


def extract_entailed_claims_and_chunks(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract entailed claims and their chunks from results.
    
    IMPORTANT: If final_judgment is "Support", we only consider chunks with judgment="Support" 
    in all_judged_chunks as entailed chunks. Chunks with judgment="NotSupport" should NOT be 
    included in entailed memory, even if final_judgment is "Support".
    """
    entailed_claims = []
    
    # First, locate Support claims from chain_of_research_results
    if "chain_of_research_results" in results:
        chain_results = results["chain_of_research_results"]
        
        for chain_item in chain_results:
            if isinstance(chain_item, dict) and "claim_results" in chain_item:
                claim_results = chain_item["claim_results"]
                
                for claim_data in claim_results:
                    if isinstance(claim_data, dict) and claim_data.get("final_judgment") in ["Support", "entailed"]:
                        claim_text = claim_data.get("claim", "")
                        final_judgment = claim_data.get("final_judgment", "")
                        all_judged_chunks = claim_data.get("all_judged_chunks", [])
                        
                        # Filter only chunks with judgment="Support" or judgment="entailed"
                        support_chunks = [
                            chunk for chunk in all_judged_chunks
                            if isinstance(chunk, dict) and chunk.get("judgment") in ["Support", "entailed"]
                        ]
                        
                        if claim_text and support_chunks:
                            entailed_claims.append({
                                "claim": claim_text,
                                "relevant_chunks": support_chunks,  # Only Support chunks
                                "source": "chain_of_research_results"
                            })

    # Then locate Support claims from report_results
    if "report_results" in results:
        report_results = results["report_results"]
        for report_item in report_results:
            if isinstance(report_item, dict) and "claim_results" in report_item:
                claim_results = report_item["claim_results"]
                for claim_data in claim_results:
                    if isinstance(claim_data, dict) and claim_data.get("final_judgment") in ["Support", "entailed"]:
                        claim_text = claim_data.get("claim", "")
                        final_judgment = claim_data.get("final_judgment", "")
                        all_judged_chunks = claim_data.get("all_judged_chunks", [])
                        
                        # Filter only chunks with judgment="Support" or judgment="entailed"
                        support_chunks = [
                            chunk for chunk in all_judged_chunks
                            if isinstance(chunk, dict) and chunk.get("judgment") in ["Support", "entailed"]
                        ]
                        
                        if claim_text and support_chunks:
                            entailed_claims.append({
                                "claim": claim_text,
                                "relevant_chunks": support_chunks,  # Only Support chunks
                                "source": "report_results"
                            })
    
    return entailed_claims


def normalize_url_for_matching(url: str) -> str:
    """
    Normalize URL for consistent matching by removing special characters but keeping structure.
    """
    if not url:
        return url
    
    import re
    
    # Remove fragment identifiers first
    # if '#' in url:
    #     url = url.split('#')[0]
    
    # Decode URL encoding
    import urllib.parse
    try:
        url = urllib.parse.unquote(url)
    except Exception:
        pass
    
    # Remove trailing slashes
    normalized = url.rstrip('/')
    
    # Replace semicolons with ampersands in query parameters
    # if ';' in url and '?' in url:
    #     base_url, query_part = url.split('?', 1)
    #     query_part = query_part.replace(';', '&')
    #     url = f"{base_url}?{query_part}"
    
    # # Remove all non-alphanumeric characters except dots, slashes, colons, question marks, and underscores
    # # This preserves the basic URL structure while removing problematic characters
    # normalized = re.sub(r'[^a-zA-Z0-9./:?_]', '', url)
    
    return normalized


def _urls_are_similar(url1: str, url2: str) -> bool:
    """
    Check if two URLs are similar enough to be considered the same.
    Uses alphanumeric-only matching with additional smart matching for common patterns.
    """
    if not url1 or not url2:
        return False
    
    # Normalize both URLs by removing all non-alphanumeric characters
    norm_url1 = normalize_url_for_matching(url1)
    norm_url2 = normalize_url_for_matching(url2)
    
    # Direct match after normalization
    if norm_url1 == norm_url2:
        return True
    
    # For ndupress.ndu.edu URLs, try to match by article ID and title
    if 'ndupress.ndu.edu' in norm_url1 and 'ndupress.ndu.edu' in norm_url2:
        # Extract article ID (numeric part) and title from both URLs
        import re
        
        # Find article ID in both URLs
        id1_match = re.search(r'(\d+)', norm_url1)
        id2_match = re.search(r'(\d+)', norm_url2)
        
        if id1_match and id2_match and id1_match.group(1) == id2_match.group(1):
            # Extract the title part (after the article ID)
            title1 = norm_url1[id1_match.end():]
            title2 = norm_url2[id2_match.end():]
            
            # If titles are similar enough, consider them the same
            if title1 == title2 or abs(len(title1) - len(title2)) <= 5:
                return True
    
    return False


def map_entailed_chunks_to_clusters(
    clusters_with_scores: List[Tuple[List[tuple], float]], 
    entailed_chunk_memory: EntailedChunkMemory,
    chunks_with_avg_scores: List[Tuple[str, str, str, float]]
) -> Dict[str, Any]:
    
    # Get all entailed chunks
    entailed_chunks = entailed_chunk_memory.get_entailed_chunk_details()
    
    if len(entailed_chunks) == 0:
        return {
            "mapped_clusters": [],
            "unmapped_clusters": list(range(len(clusters_with_scores)))
        }
    
    print(f"🚀 Starting optimized mapping with {len(entailed_chunks)} entailed chunks and {len(clusters_with_scores)} clusters")
    
    # Pre-build lookup tables for O(1) access
    print("🔧 Building lookup tables...")
    
    # Build (chunk_id, normalized_text) -> URL mapping to disambiguate duplicate chunk_ids across URLs
    chunk_pair_to_url: Dict[Tuple[str, str], str] = {}
    for chunk_id, chunk_text, url, score in chunks_with_avg_scores:
        normalized_url = normalize_url_for_matching(url)
        chunk_pair_to_url[(chunk_id, str(chunk_text).strip())] = normalized_url
    
    # Build cluster chunk mapping: cluster_idx -> set of unique chunk_ids
    # chunk_id format is {url_index}-chunk_{chunk_index}, which is already unique within the file
    cluster_chunk_mapping: Dict[int, Set[str]] = {}
    for cluster_idx, (cluster_chunks, cluster_score) in enumerate(clusters_with_scores):
        chunk_ids: Set[str] = set()
        for cluster_chunk_id, cluster_chunk_text in cluster_chunks:
            # chunk_id is already unique, use it directly
            if cluster_chunk_id:
                chunk_ids.add(cluster_chunk_id)
        cluster_chunk_mapping[cluster_idx] = chunk_ids
    
    print(f"✅ Built lookup tables: {len(chunk_pair_to_url)} chunk pairs, {len(cluster_chunk_mapping)} clusters")
    
    # Debug: Print sample chunk keys for debugging
    print("🔍 Sample chunk keys in clusters:")
    for i, (cluster_idx, chunk_keys) in enumerate(list(cluster_chunk_mapping.items())[:3]):
        print(f"  Cluster {cluster_idx}: {list(chunk_keys)[:2]}...")  # Show first 2 keys
    
    # Convert entailed chunks to list for parallel processing
    entailed_chunks_list = []
    for chunk_id, chunk_info in entailed_chunks.items():
        entailed_chunks_list.append((chunk_id, chunk_info))
    
    # Use multiprocessing to map entailed chunks to clusters
    from multiprocessing import Pool, cpu_count
    import functools
    
    # Create a partial function with the lookup tables
    map_func = functools.partial(
        _map_single_entailed_chunk_to_clusters,
        cluster_chunk_mapping=cluster_chunk_mapping
    )
    
    # Use 128 cores as requested
    num_cores = min(128, cpu_count())
    print(f"🚀 Using {num_cores} CPU cores for parallel mapping...")
    
    mapped_clusters = set()
    
    with Pool(processes=num_cores) as pool:
        # Process entailed chunks in parallel
        results = pool.map(map_func, entailed_chunks_list)
        
        # Collect results
        for cluster_indices in results:
            if cluster_indices:
                mapped_clusters.update(cluster_indices)
    
    print(f"✅ Parallel mapping completed! Found {len(mapped_clusters)} mapped clusters")
    
    # Identify unmapped clusters
    unmapped_clusters = set(range(len(clusters_with_scores))) - mapped_clusters
    
    return {
        "mapped_clusters": sorted(list(mapped_clusters)),
        "unmapped_clusters": sorted(list(unmapped_clusters))
    }


def _map_single_entailed_chunk_to_clusters(
    entailed_chunk_data: tuple,
    cluster_chunk_mapping: dict
) -> list:
    """
    Map a single entailed chunk to its corresponding clusters.
    This function is designed to be used with multiprocessing.
    """
    chunk_id, chunk_info = entailed_chunk_data
    chunk_url = chunk_info.get('source_url', '')
    
    mapped_clusters = []
    
    # Find clusters that contain this chunk_id (chunk_id is unique within the file)
    for cluster_idx, cluster_chunk_ids in cluster_chunk_mapping.items():
        if chunk_id in cluster_chunk_ids:
            mapped_clusters.append(cluster_idx)
    
    if mapped_clusters:
        for idx in mapped_clusters:
            print(f"✅ Mapped entailed chunk {chunk_id} from {chunk_url} to cluster {idx}")
    else:
        # print(f"❌ Could not find cluster for entailed chunk {chunk_id} from {chunk_url}")
        # Debug: print available chunk_ids for this chunk_id
        debug_chunk_ids = []
        for cluster_idx, cluster_chunk_ids in cluster_chunk_mapping.items():
            if chunk_id in cluster_chunk_ids:
                debug_chunk_ids.append(f"Cluster {cluster_idx}")
        if not debug_chunk_ids:
            print(f"   Chunk {chunk_id} not found in any cluster")
            # Show some sample chunk_ids for debugging
            sample_chunk_ids = []
            for cluster_idx, cluster_chunk_ids in list(cluster_chunk_mapping.items())[:2]:
                sample_chunk_ids.extend(list(cluster_chunk_ids)[:2])
            print(f"   Sample cluster chunk_ids: {sample_chunk_ids}")
    
    return mapped_clusters


def analyze_cluster_distribution(mapping_results: Dict[str, Any]) -> Dict[str, Any]:
    return {"placeholder": "not_used"}


def save_mapping_analysis(
    mapping_results: Dict[str, Any], 
    analysis_results: Dict[str, Any], 
    output_file: str
):
    pass  # Not needed for ND analysis


def calculate_iteration_statistics_from_clusters(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                'total_chunks': len(cluster['chunks']),
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


def calculate_citation_statistics_from_clusters(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def analyze_entailed_chunks_iterations_from_memory(
    entailed_chunk_memory: EntailedChunkMemory,
    url_to_iterations: Dict[str, List[int]]
) -> Dict[str, Any]:
    """
    Analyze which iterations entailed chunks were retrieved from.
    
    Args:
        entailed_chunk_memory: Memory containing entailed chunks
        url_to_iterations: Mapping from URLs to iteration indices
    
    Returns:
        Dict containing iteration distribution statistics
    """
    entailed_details = entailed_chunk_memory.get_entailed_chunk_details()
    
    # Track unique chunks per iteration (not iteration occurrences)
    # Key: iteration index, Value: set of chunk_ids
    iteration_to_chunks = {}
    chunk_iteration_details = []
    
    for chunk_key, chunk_info in entailed_details.items():
        chunk_id = chunk_info['chunk_id']
        url = chunk_info['source_url']
        normalized_url = normalize_url_for_matching(url)
        
        # Find iterations for this URL
        iterations = url_to_iterations.get(normalized_url, [])
        if not iterations:
            # Try exact URL match
            iterations = url_to_iterations.get(url, [])
        
        if iterations:
            # For each iteration this chunk appears in, add the chunk_id to that iteration's set
            for iteration in iterations:
                if iteration not in iteration_to_chunks:
                    iteration_to_chunks[iteration] = set()
                iteration_to_chunks[iteration].add(chunk_id)
            
            chunk_iteration_details.append({
                'chunk_id': chunk_id,
                'url': url,
                'normalized_url': normalized_url,
                'iterations': iterations,
                'first_iteration': min(iterations) if iterations else None,
                'last_iteration': max(iterations) if iterations else None
            })
        else:
            chunk_iteration_details.append({
                'chunk_id': chunk_id,
                'url': url,
                'normalized_url': normalized_url,
                'iterations': [],
                'first_iteration': None,
                'last_iteration': None
            })
    
    # Calculate distribution statistics: count unique chunks per iteration
    iteration_distribution = {}
    for iteration, chunk_set in iteration_to_chunks.items():
        iteration_distribution[iteration] = len(chunk_set)
    
    chunks_with_iterations = len([c for c in chunk_iteration_details if c['iterations']])
    chunks_without_iterations = len([c for c in chunk_iteration_details if not c['iterations']])
    
    return {
        'total_entailed_chunks': len(entailed_details),
        'chunks_with_iterations': chunks_with_iterations,
        'chunks_without_iterations': chunks_without_iterations,
        'iteration_distribution': sorted(iteration_distribution.items()),
        'chunk_details': chunk_iteration_details
    }


def analyze_iteration_level_nd(
    json_file_path: str,
    entailed_chunk_memory: EntailedChunkMemory,
    summary_citations: Set[str],
    url_to_iterations: Dict[str, List[int]],
    num_gpus: int = 1,
    gpu_ids: List[int] = None
) -> Dict[str, Any]:
    """
    Analyze ND at iteration level - cluster and compute ND for each iteration separately.
    
    Args:
        json_file_path: Path to cache JSON file
        entailed_chunk_memory: Memory containing entailed chunks
        summary_citations: Set of cited URLs from summary
        url_to_iterations: Mapping from URLs to iteration indices
        num_gpus: Number of GPUs to use
        gpu_ids: List of specific GPU IDs to use
        
    Returns:
        Dict containing iteration-level ND analysis results
    """
    print(f"\n📊 Starting iteration-level ND analysis...")
    
    # Load cache data to get iterations and their URLs
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading cache file: {e}")
        return {}
    
    iterations_data = cache_data.get('iterations', [])
    
    # Build reverse mapping: iteration -> list of URLs (using provided url_to_iterations)
    iteration_to_urls = {}
    # If ONLY one iteration, use the provided url_to_iterations, skip iteration-level ND analysis
    if len(iterations_data) == 1:
        print(f"✅ Found only one iteration, skipping iteration-level ND analysis")
        return {
            'iteration_results': {},
            'aggregate_statistics': {}
        }
    
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
    all_chunks = extract_chunks_with_scores(json_file_path)
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
                gpu_ids=gpu_ids
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
                
                # Find normalized URL for iteration lookup
                normalized_url = normalize_url_for_matching(url)
                
                # Find iterations for this URL (for this iteration-specific analysis, it should be current iteration)
                url_iterations = url_to_iterations.get(normalized_url, [])
                if not url_iterations:
                    url_iterations = url_to_iterations.get(url, [])
                
                # For iteration-level analysis, the iteration should be the current iter_idx
                iteration_index = iter_idx if iter_idx in url_iterations else (min(url_iterations) if url_iterations else None)
                
                # Build features for this chunk
                features = {
                    'chunk_id': chunk_id,
                    'url': url,
                    'chunk_text': chunk_text,
                    'document_position': 'middle',  # Default value
                    'iteration': iteration_index,
                    'iterations': url_iterations if url_iterations else [],
                    'is_entailed': is_entailed,
                    'in_citations': in_citations
                }
                
                cluster_detail['chunks'].append({
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'url': url,
                    'iteration': iteration_index,
                    'iterations': url_iterations if url_iterations else [],
                    'is_entailed': is_entailed,
                    'in_citations': in_citations,
                    'features': features
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
            'nd_analysis': iteration_nd['nd_analysis'],  # Keep complete nd_analysis including unmapped_cluster_details
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


def noise_domination_detection(results_file: str, json_file_path: str, raw_json_file: str, num_gpus: int = None, gpu_ids: List[int] = None):
    
    print("=== ND Analysis ===")
    
    # Load results file first to check if nd_analysis already exists
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Failed to load results file: {e}")
        return
    
    # Check if global level ND analysis already exists and is not empty
    existing_nd_analysis = results.get('nd_analysis', {})
    existing_hallucination_score = results.get('hallucination_score')
    skip_global_nd = False
    
    if existing_nd_analysis and (
        existing_nd_analysis.get('total_nd', 0) > 0 or 
        existing_nd_analysis.get('unmapped_cluster_details', []) or
        existing_hallucination_score is not None
    ):
        skip_global_nd = True
        print("✅ Global level ND analysis already exists, skipping global level ND computation...")
        hallucination_score = existing_hallucination_score
        nd_analysis = existing_nd_analysis
        clusters_with_scores = None
        mapping_results = None
        chunks_with_avg_scores = None
        entailed_chunk_memory = None
        detailed_clusters = results.get('clusters', [])
    else:
        # Load summary_citations from raw JSON file
        summary_citations = load_summary_citations(raw_json_file)
        
        # Extract chunks with scores
        chunks_with_scores = extract_chunks_with_scores(json_file_path)
        if not chunks_with_scores:
            print("No chunks with scores were extracted. Exiting.")
            return
        
        # Compute average scores for each chunk
        chunks_with_avg_scores = compute_average_scores(chunks_with_scores)
        
        # Print chunk information for ND analysis
        print(f"\n=== CHUNK URL MAPPING ===")
        # for chunk_id, chunk_text, url, score in chunks_with_avg_scores:
            # print(f"Chunk: {chunk_id} | URL: {url} | Score: {score:.6f}")
        
        # Cluster chunks and rank by score
        clusters_with_scores = cluster_and_rank_chunks(
            chunks_with_avg_scores,
            method="umap_hdbscan_tuned",
            num_gpus=num_gpus or 1,
            gpu_ids=gpu_ids
        )
        
        # Extract entailed claims and their chunks
        entailed_claims_data = extract_entailed_claims_and_chunks(results)
        if not entailed_claims_data:
            print("No entailed claims found in results file")
            return
        
        # Initialize EntailedChunkMemory and store entailed chunks
        entailed_chunk_memory = EntailedChunkMemory()
        print(f"\n=== ENTAILED CHUNKS DEBUG ===")
        print(f"Found {len(entailed_claims_data)} entailed claims")
        for claim_data in entailed_claims_data:
            relevant_chunks = claim_data["relevant_chunks"]
            entailed_chunk_memory.add_entailed_chunks(relevant_chunks)
            print(f"Claim: {claim_data['claim'][:100]}...")
            for chunk in relevant_chunks:
                print(f"  - Entailed Chunk: {chunk.get('chunk_id', 'unknown')} | URL: {chunk.get('source_url', 'unknown')} | Confidence: {chunk.get('confidence', 'N/A')}")
        
        print(f"\nTotal entailed chunks stored: {entailed_chunk_memory.get_entailed_chunk_count()}")
        entailed_details = entailed_chunk_memory.get_entailed_chunk_details()
        # print(f"Sample entailed chunks:")
        # for i, (chunk_key, chunk_info) in enumerate(list(entailed_details.items())[:5]):
        #     print(f"  {i+1}. {chunk_key} -> {chunk_info}")
        
        # Map entailed chunks to clusters
        mapping_results = map_entailed_chunks_to_clusters(clusters_with_scores, entailed_chunk_memory, chunks_with_avg_scores)

        # Print detailed debug information
        print_detailed_cluster_debug(
            clusters_with_scores, 
            mapping_results.get('mapped_clusters', []), 
            mapping_results.get('unmapped_clusters', []),
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
        
        hallucination_score = nd_results['wis_score']
        nd_analysis = nd_results['nd_analysis']
        
        print(f"WIS Score: {hallucination_score:.4f}")
        print(f"\n每个unmapped cluster的ND类型比例:")
        print("=" * 80)
        
        for detail in nd_analysis['unmapped_cluster_details']:
            cluster_rank = detail['cluster_rank']
            total_chunks = detail['total_chunks']
            doc_ratio = detail['document_level_nd_ratio']
            chunk_ratio = detail['chunk_level_nd_ratio']
            
            print(f"Cluster {cluster_rank + 1} (包含 {total_chunks} 个chunks):")
            print(f"  - Document-level ND: {doc_ratio:.2%}")
            print(f"  - Chunk-level ND: {chunk_ratio:.2%}")
            print(f"  - 总计: {doc_ratio + chunk_ratio:.2%}")
            
            # 显示每个chunk的详细信息
            # print(f"  Chunk详情:")
            for chunk_detail in detail['chunk_details']:
                chunk_id = chunk_detail['chunk_id']
                chunk_url = chunk_detail['chunk_url']
                chunk_in_citations = chunk_detail['chunk_in_citations']
                # print(f"    - {chunk_id}: URL={chunk_url}, 在citations={'是' if chunk_in_citations else '否'}")
                # print(f"      ND比例: Document={chunk_detail['document_nd_ratio']:.2%}, Chunk={chunk_detail['chunk_nd_ratio']:.2%}")
            # print()
        
        print(f"总体统计:")
        print(f"  - Document-level ND: {nd_analysis['document_level_nd_ratio']:.2%}")
        print(f"  - Chunk-level ND: {nd_analysis['chunk_level_nd_ratio']:.2%}")
        print(f"  - 总计: {nd_analysis['document_level_nd_ratio'] + nd_analysis['chunk_level_nd_ratio']:.2%}")
        
        print(f"\n平均ND比例 (跨unmapped clusters平均):")
        print(f"  - 平均Document-level ND: {nd_analysis['avg_document_level_nd_ratio']:.2%}")
        print(f"  - 平均Chunk-level ND: {nd_analysis['avg_chunk_level_nd_ratio']:.2%}")

    # ===== LOAD URL-TO-ITERATION MAPPING =====
    print(f"\n📊 Loading URL-to-iteration mapping...")
    url_to_iterations = {}
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        iterations = cache_data.get('iterations', [])
        for iter_idx, iteration in enumerate(iterations, start=1):
            search_key = f'search_list_{iter_idx}'
            if search_key in iteration:
                urls = iteration[search_key]
                unique_urls_in_iteration = set(urls)
                for url in unique_urls_in_iteration:
                    normalized_url = normalize_url_for_matching(url)
                    if normalized_url not in url_to_iterations:
                        url_to_iterations[normalized_url] = []
                    if not url_to_iterations[normalized_url] or url_to_iterations[normalized_url][-1] != iter_idx:
                        url_to_iterations[normalized_url].append(iter_idx)
        
        # Ensure each URL maps to a unique, sorted list of iterations
        for key in list(url_to_iterations.keys()):
            url_to_iterations[key] = sorted(set(url_to_iterations[key]))
        
        print(f"✅ Loaded URL-to-iteration mapping: {len(url_to_iterations)} unique URLs")
    except Exception as e:
        print(f"⚠️ Error loading URL-to-iteration mapping: {e}")
    
    # If skipped global ND, load necessary data from existing results
    if skip_global_nd:
        # Load summary_citations for later use
        summary_citations = load_summary_citations(raw_json_file)
        
        # Load entailed chunks for iteration level ND
        entailed_claims_data = extract_entailed_claims_and_chunks(results)
        if entailed_claims_data:
            entailed_chunk_memory = EntailedChunkMemory()
            for claim_data in entailed_claims_data:
                relevant_chunks = claim_data["relevant_chunks"]
                entailed_chunk_memory.add_entailed_chunks(relevant_chunks)
        else:
            entailed_chunk_memory = EntailedChunkMemory()
        
        # Load mapping results from existing results
        mapping_results = {
            'mapped_clusters': results.get('mapped_clusters', []),
            'unmapped_clusters': results.get('unmapped_clusters', [])
        }
        
        # Load other statistics from existing results
        iteration_stats = results.get('iteration_statistics', {})
        citation_stats = results.get('citation_statistics', {})
        entailed_iteration_analysis = results.get('entailed_iteration_analysis', {})
        feature_comparison = results.get('mapped_unmapped_feature_comparison', {})
    else:
        # ===== BUILD DETAILED CLUSTER INFORMATION =====
        print(f"\n🔧 Building detailed cluster information...")
        
        # Load summary_citations from raw JSON file
        summary_citations = load_summary_citations(raw_json_file)
        
        # Build (chunk_id, chunk_text) -> (url, score) mapping
        chunk_key_to_info = {}
        for chunk_id, chunk_text, url, score in chunks_with_avg_scores:
            chunk_key = (chunk_id, chunk_text.strip())
            chunk_key_to_info[chunk_key] = {
                'url': url,
                'score': score,
                'chunk_text': chunk_text.strip()
            }
        
        # Load cache data for URL to url_index mapping and chunk features
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            chunk_scores = cache_data.get('chunk_score', {})
            
            # Build URL index to URL mapping from chunk_score keys
            url_index_to_url = {}
            for chunk_key_str, chunk_info_item in chunk_scores.items():
                import re
                match = re.match(r'^(\d+)-chunk_', chunk_key_str)
                if match:
                    url_index = int(match.group(1))
                    url_from_cache = chunk_info_item.get('url', '')
                    if url_from_cache and url_index not in url_index_to_url:
                        url_index_to_url[url_index] = url_from_cache
            
            url_to_url_index = {url_val: idx for idx, url_val in url_index_to_url.items()}
        except Exception as e:
            print(f"⚠️ Error loading cache for chunk features: {e}")
            chunk_scores = {}
            url_to_url_index = {}
        
        # Build detailed cluster information with entailment status, iterations, and features
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
                    url = "unknown_url"
                    for (cid, ctext), info in chunk_key_to_info.items():
                        if cid == chunk_id and ctext.strip() == chunk_text.strip():
                            url = info['url']
                            chunk_info = info
                            break
                else:
                    url = chunk_info['url']
                
                # Skip duplicates of the same (chunk_id, url) within this cluster
                dedup_key = (chunk_id, url)
                if dedup_key in seen_chunk_keys:
                    continue
                seen_chunk_keys.add(dedup_key)
                
                normalized_url = normalize_url_for_matching(url)
                
                # Find iteration(s) for this URL
                iterations = url_to_iterations.get(normalized_url, [])
                if not iterations:
                    iterations = url_to_iterations.get(url, [])
                iteration_index = min(iterations) if iterations else None
                
                # Get chunk data from cache using url_index-chunk_id format
                chunk_data = None
                if url in url_to_url_index:
                    url_index = url_to_url_index[url]
                    chunk_key_str = f"{url_index}-{chunk_id}"
                    chunk_data = chunk_scores.get(chunk_key_str)
                
                # If not found, try to find by matching chunk_id and url
                if not chunk_data:
                    for chunk_key_str, chunk_info_cache in chunk_scores.items():
                        if (chunk_info_cache.get('url') == url and 
                            chunk_info_cache.get('chunk_id_original') == chunk_id):
                            chunk_data = chunk_info_cache
                            break
                
                # Extract features
                features = {
                    'chunk_id': chunk_id,
                    'url': url,
                    'chunk_text': chunk_text,
                    'document_position': 'middle',  # Default value
                    'iteration': iteration_index,
                    'iterations': iterations if iterations else [],
                    'is_entailed': entailed_chunk_memory.is_chunk_entailed(chunk_id, url),
                    'in_citations': url in summary_citations
                }
                
                # Check if chunk is entailed
                is_entailed = entailed_chunk_memory.is_chunk_entailed(chunk_id, url)
                in_citations = url in summary_citations
                
                cluster_details['chunks'].append({
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                    'url': url,
                    'iteration': iteration_index,
                    'iterations': iterations if iterations else [],
                    'is_entailed': is_entailed,
                    'in_citations': in_citations,
                    'features': features
                })
            
            detailed_clusters.append(cluster_details)
        
        # ===== CALCULATE ITERATION STATISTICS =====
        print(f"\n📊 Calculating iteration statistics...")
        iteration_stats = calculate_iteration_statistics_from_clusters(detailed_clusters)
        
        # ===== CALCULATE CITATION STATISTICS =====
        print(f"\n📊 Calculating citation statistics...")
        citation_stats = calculate_citation_statistics_from_clusters(detailed_clusters)
        
        # ===== ANALYZE ENTAILED CHUNKS ITERATIONS =====
        print(f"\n📊 Analyzing entailed chunks iteration distribution...")
        entailed_iteration_analysis = analyze_entailed_chunks_iterations_from_memory(
            entailed_chunk_memory,
            url_to_iterations
        )
        
        # ===== MAPPED VS UNMAPPED FEATURE COMPARISON =====
        print(f"\n🔍 Comparing mapped vs unmapped chunk features...")
        mapped_chunks_features = []
        unmapped_chunks_features = []
        
        for cluster in detailed_clusters:
            for chunk in cluster['chunks']:
                features = {
                    'chunk_id': chunk['chunk_id'],
                    'url': chunk['url'],
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

    # ===== ITERATION-LEVEL ND ANALYSIS =====
    # Check if iteration level ND analysis already exists
    existing_iteration_level_nd = results.get('iteration_level_nd_analysis', {})
    skip_iteration_nd = False
    
    if existing_iteration_level_nd and (
        existing_iteration_level_nd.get('iteration_results', {}) or
        existing_iteration_level_nd.get('aggregate_statistics', {})
    ):
        skip_iteration_nd = True
        print("✅ Iteration level ND analysis already exists, skipping iteration level ND computation...")
        iteration_level_nd = existing_iteration_level_nd
    else:
        print(f"\n🔄 Running iteration-level ND analysis...")
        # Load summary_citations if not already loaded
        if skip_global_nd:
            summary_citations = load_summary_citations(raw_json_file)
        
        iteration_level_nd = analyze_iteration_level_nd(
            json_file_path,
            entailed_chunk_memory,
            summary_citations,
            url_to_iterations,
            num_gpus=num_gpus or 1,
            gpu_ids=gpu_ids
        )

    # ===== SAVE ALL RESULTS =====
    print(f"\n💾 Saving comprehensive ND analysis results...")
    
    # Use the results dict that was loaded at the beginning of the function
    # Preserve important existing fields
    existing_keys = set(results.keys())
    important_keys = {'hallucinated_actions', 'missed_queries', 'query', 'report', 'all_urls', 
                    'summary_urls', 'summary', 'chain_of_research_results', 'report_results'}
    preserved_keys = existing_keys & important_keys
    if preserved_keys:
        print(f"📌 Preserving existing fields: {', '.join(sorted(preserved_keys))}")
    
    # Add/update only ND analysis fields (will not overwrite existing non-ND fields)
    if not skip_global_nd:
        results['hallucination_score'] = hallucination_score
        results['nd_analysis'] = nd_analysis
        results['clusters'] = detailed_clusters
        results['mapped_clusters'] = sorted(mapping_results.get('mapped_clusters', []))
        results['unmapped_clusters'] = sorted(mapping_results.get('unmapped_clusters', []))
        results['total_clusters'] = len(clusters_with_scores) if clusters_with_scores else results.get('total_clusters', 0)
        results['total_mapped'] = len(mapping_results.get('mapped_clusters', []))
        results['total_unmapped'] = len(mapping_results.get('unmapped_clusters', []))
        results['total_entailed_chunks'] = entailed_chunk_memory.get_entailed_chunk_count() if entailed_chunk_memory else results.get('total_entailed_chunks', 0)
        results['entailed_iteration_analysis'] = entailed_iteration_analysis
        results['mapped_unmapped_feature_comparison'] = feature_comparison
        results['iteration_statistics'] = iteration_stats
        results['citation_statistics'] = citation_stats
    
    # Add iteration-level ND analysis results
    if not skip_iteration_nd and iteration_level_nd:
        results['iteration_level_nd_analysis'] = iteration_level_nd
        print(f"\n📊 Iteration-Level ND Analysis Summary:")
        if 'aggregate_statistics' in iteration_level_nd and iteration_level_nd['aggregate_statistics']:
            agg_stats = iteration_level_nd['aggregate_statistics']
            # print(f"  - Total iterations analyzed: {agg_stats.get('total_iterations', 0)}")
            # print(f"  - Average hallucination score per iteration: {agg_stats.get('average_hallucination_score', 0):.4f}")
            # print(f"  - Total mapped clusters across iterations: {agg_stats.get('total_mapped_clusters', 0)}")
            # print(f"  - Total unmapped clusters across iterations: {agg_stats.get('total_unmapped_clusters', 0)}")
            # print(f"  - Average clusters per iteration: {agg_stats.get('avg_clusters_per_iteration', 0):.2f}")
    
    # Write back to file, preserving all existing fields
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Verify preservation
    final_keys = set(results.keys())
    nd_keys = {'hallucination_score', 'nd_analysis', 'clusters', 'mapped_clusters', 'unmapped_clusters',
               'total_clusters', 'total_mapped', 'total_unmapped', 'total_entailed_chunks',
               'entailed_iteration_analysis', 'mapped_unmapped_feature_comparison', 
               'iteration_statistics', 'citation_statistics', 'iteration_level_nd_analysis'}
    non_nd_keys = final_keys - nd_keys
    
    print(f"\n✅ Comprehensive ND analysis results saved to: {results_file}")
    if not skip_global_nd:
        print(f"   - Saved {len(detailed_clusters)} clusters")
        print(f"   - Saved {results.get('total_mapped', 0)} mapped clusters")
        print(f"   - Saved {results.get('total_unmapped', 0)} unmapped clusters")
        print(f"   - Saved {results.get('total_entailed_chunks', 0)} entailed chunks")
        print(f"   - Saved iteration statistics")
        print(f"   - Saved citation statistics")
    if iteration_level_nd and not skip_iteration_nd:
        print(f"   - Saved iteration-level ND analysis")
    if non_nd_keys:
        print(f"   - Preserved {len(non_nd_keys)} existing non-ND fields: {', '.join(sorted(list(non_nd_keys)[:5]))}{'...' if len(non_nd_keys) > 5 else ''}")
    

def load_summary_citations(raw_json_file: str) -> Set[str]:
    """Load summary_citations from the raw JSON file."""
    try:
        with open(raw_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary_citations = set(data.get('summary_citations', []))
        return summary_citations
    except Exception as e:
        print(f"❌ Error loading summary_citations: {e}")
        return set()




def calculate_hallucination_score_with_nd_analysis_parallel(mapped_clusters, unmapped_clusters, 
                                                          clusters_with_scores, summary_citations, chunks_with_avg_scores,
                                                          num_processes=128):
    """
    Calculate hallucination score with detailed noise domination (ND) analysis at chunk-level.
    Optimized version with parallel processing and algorithmic improvements.
    
    Args:
        mapped_clusters: List of cluster indices that contain entailed chunks
        unmapped_clusters: List of cluster indices that don't contain entailed chunks
        clusters_with_scores: List of (cluster_chunks, cluster_score) tuples
        summary_citations: Set of URLs from summary_citations
        chunks_with_avg_scores: List of (chunk_id, chunk_text, url, score) tuples
        num_processes: Number of CPU processes to use (default: 128)
    
    Returns:
        Dict containing WIS score and ND analysis
    """
    
    # Pre-compute chunk_id to URL mapping for O(1) lookup
    chunk_id_to_url = {}
    for chunk_id, chunk_text, url, score in chunks_with_avg_scores:
        chunk_id_to_url[chunk_id] = url
    
    # Prepare data for parallel processing
    unmapped_cluster_data = []
    for u_rank in unmapped_clusters:
        unmapped_cluster_chunks, unmapped_cluster_score = clusters_with_scores[u_rank]
        unmapped_cluster_data.append({
            'u_rank': u_rank,
            'chunks': unmapped_cluster_chunks,
            'score': unmapped_cluster_score
        })
    
    # Process unmapped clusters in parallel
    if len(unmapped_cluster_data) > 0:
        # Use multiprocessing for parallel computation
        with Pool(processes=min(num_processes, len(unmapped_cluster_data), cpu_count())) as pool:
            # Create partial function with fixed arguments
            process_func = partial(
                process_unmapped_cluster,
                mapped_clusters=mapped_clusters,
                clusters_with_scores=clusters_with_scores,
                summary_citations=summary_citations,
                chunk_id_to_url=chunk_id_to_url
            )
            
            # Process clusters in parallel
            results = pool.map(process_func, unmapped_cluster_data)
    else:
        results = []
    
    # Aggregate results
    actual_wis = 0.0
    nd_analysis = {
        'document_level_nd': 0,
        'chunk_level_nd': 0,
        'total_nd': 0,
        'unmapped_cluster_details': []
    }
    
    for result in results:
        # wis_contribution already includes multiplication by chunk count
        actual_wis += result['wis_contribution']
        nd_analysis['document_level_nd'] += result['document_nd_count']
        nd_analysis['chunk_level_nd'] += result['chunk_nd_count']
        nd_analysis['total_nd'] += result['total_inversions']
        nd_analysis['unmapped_cluster_details'].append(result['cluster_details'])
    
    # Calculate WIS score
    num_unmapped = len(unmapped_clusters)
    num_mapped = len(mapped_clusters)
    
    if actual_wis == 0 or num_unmapped == 0 or num_mapped == 0:
        wis_score = 0.0
    else:
        # Calculate maximum possible WIS
        # Optimal arrangement to maximize penalty: 
        # Assume the first num_unmapped clusters are unmapped, and all mapped clusters are after them.
        # In this optimal case, each unmapped cluster at rank i (1-based) has:
        # - rank = i (among the first num_unmapped positions)
        # - inversion_count = num_mapped (all mapped clusters are after all unmapped clusters)
        # - contribution = (num_mapped / i) * num_chunks
        
        max_wis = 0.0
        for i in range(num_unmapped):
            cluster_chunks, _ = clusters_with_scores[i]
            num_chunks = len(cluster_chunks)
            rank_for_calc = i + 1  # 1-based rank (first num_unmapped clusters: ranks 1, 2, ..., num_unmapped)
            inversion_count = num_mapped  # All mapped clusters are after all unmapped clusters
            max_wis += (inversion_count / rank_for_calc) * num_chunks
        
        wis_score = min(actual_wis / max_wis, 1.0)
    
    # Calculate ND ratios
    if nd_analysis['total_nd'] > 0:
        nd_analysis['document_level_nd_ratio'] = nd_analysis['document_level_nd'] / nd_analysis['total_nd']
        nd_analysis['chunk_level_nd_ratio'] = nd_analysis['chunk_level_nd'] / nd_analysis['total_nd']
    else:
        nd_analysis['document_level_nd_ratio'] = 0.0
        nd_analysis['chunk_level_nd_ratio'] = 0.0
    
    # Calculate average ratios across clusters
    if nd_analysis['unmapped_cluster_details']:
        total_doc_ratio = sum(detail['document_level_nd_ratio'] for detail in nd_analysis['unmapped_cluster_details'])
        total_chunk_ratio = sum(detail['chunk_level_nd_ratio'] for detail in nd_analysis['unmapped_cluster_details'])
        
        num_clusters = len(nd_analysis['unmapped_cluster_details'])
        nd_analysis['avg_document_level_nd_ratio'] = total_doc_ratio / num_clusters
        nd_analysis['avg_chunk_level_nd_ratio'] = total_chunk_ratio / num_clusters
    else:
        nd_analysis['avg_document_level_nd_ratio'] = 0.0
        nd_analysis['avg_chunk_level_nd_ratio'] = 0.0
    
    return {
        'wis_score': wis_score,
        'nd_analysis': nd_analysis
    }


def process_unmapped_cluster(cluster_data, mapped_clusters, clusters_with_scores, 
                            summary_citations, chunk_id_to_url):
    """
    Process a single unmapped cluster for ND analysis.
    This function is designed to be called in parallel.
    """
    u_rank = cluster_data['u_rank']
    unmapped_cluster_chunks = cluster_data['chunks']
    unmapped_cluster_score = cluster_data['score']
    
    # Count inversions: how many mapped clusters rank after this unmapped cluster
    rank_for_calc = u_rank + 1
    inversion_count = sum(1 for m_rank in mapped_clusters if u_rank < m_rank)
    num_chunks = len(unmapped_cluster_chunks)
    
    # WIS contribution: (inversion_count / rank) * num_chunks
    wis_contribution = (inversion_count / rank_for_calc * num_chunks) if inversion_count > 0 else 0.0
    
    # Initialize cluster-level counters
    cluster_document_nd = 0
    cluster_chunk_nd = 0
    chunk_nd_details = []
    
    # Process each chunk in the unmapped cluster
    for chunk_id, chunk_text in unmapped_cluster_chunks:
        # Get chunk URL efficiently
        chunk_url = chunk_id_to_url.get(chunk_id, "unknown_url")
        chunk_in_citations = chunk_url in summary_citations
        
        # Initialize chunk-level counters
        document_nd_count = 0
        chunk_nd_count = 0
        
        # Count inversions for this chunk (each mapped cluster counts as 1)
        for m_rank in mapped_clusters:
            if u_rank < m_rank:  # This is an inversion
                if not chunk_in_citations:
                    # Case 1: Document-level ND (unmapped chunk URL is NOT in summary_citations)
                    document_nd_count += 1
                else:
                    # Case 2: Chunk-level ND (unmapped chunk URL is in summary_citations)
                    chunk_nd_count += 1
        
        total_chunk_inversions = document_nd_count + chunk_nd_count
        
        # Always record chunk details, even if inversions are 0 (for visibility and debugging)
        if total_chunk_inversions > 0:
            # Calculate ratios for this chunk
            document_nd_ratio = document_nd_count / total_chunk_inversions
            chunk_nd_ratio = chunk_nd_count / total_chunk_inversions
        else:
            # No inversions, set ratios to 0
            document_nd_ratio = 0.0
            chunk_nd_ratio = 0.0
        
        chunk_nd_details.append({
            'chunk_id': chunk_id,
            'chunk_url': chunk_url,
            'chunk_in_citations': chunk_in_citations,
            'document_nd_count': document_nd_count,
            'chunk_nd_count': chunk_nd_count,
            'total_inversions': total_chunk_inversions,
            'document_nd_ratio': document_nd_ratio,
            'chunk_nd_ratio': chunk_nd_ratio
        })
        
        # Accumulate for cluster totals
        cluster_document_nd += document_nd_count
        cluster_chunk_nd += chunk_nd_count
    
    # Total inversions for the cluster: inversion_count * num_chunks
    cluster_total_inversions = inversion_count * num_chunks
    
    # Prepare cluster details
    cluster_details = {
        'cluster_rank': u_rank,
        'cluster_score': unmapped_cluster_score,
        'total_chunks': num_chunks,
        'document_level_nd': cluster_document_nd,
        'chunk_level_nd': cluster_chunk_nd,
        'total_inversions': cluster_total_inversions,
        'chunk_details': chunk_nd_details
    }
    
    # Calculate cluster ratios
    if cluster_total_inversions > 0:
        cluster_details['document_level_nd_ratio'] = cluster_document_nd / cluster_total_inversions
        cluster_details['chunk_level_nd_ratio'] = cluster_chunk_nd / cluster_total_inversions
    else:
        cluster_details['document_level_nd_ratio'] = 0.0
        cluster_details['chunk_level_nd_ratio'] = 0.0
    
    return {
        'wis_contribution': wis_contribution,
        'document_nd_count': cluster_document_nd,
        'chunk_nd_count': cluster_chunk_nd,
        'total_inversions': cluster_total_inversions,
        'cluster_details': cluster_details
    }


def calculate_hallucination_score_with_nd_analysis(mapped_clusters, unmapped_clusters, 
                                                 clusters_with_scores, summary_citations, chunks_with_avg_scores):
    """
    Calculate hallucination score with detailed noise domination (ND) analysis at chunk-level.
    This is the original implementation - kept for backward compatibility.
    For better performance, use calculate_hallucination_score_with_nd_analysis_parallel instead.
    
    Args:
        mapped_clusters: List of cluster indices that contain entailed chunks
        unmapped_clusters: List of cluster indices that don't contain entailed chunks
        clusters_with_scores: List of (cluster_chunks, cluster_score) tuples
        summary_citations: Set of URLs from summary_citations
        chunks_with_avg_scores: List of (chunk_id, chunk_text, url, score) tuples
    
    Returns:
        Dict containing WIS score and ND analysis
    """
    
    # Use the parallel version by default for better performance
    return calculate_hallucination_score_with_nd_analysis_parallel(
        mapped_clusters, unmapped_clusters, clusters_with_scores, 
        summary_citations, chunks_with_avg_scores, num_processes=128
    )


def calculate_hallucination_score(mapped_clusters, unmapped_clusters):
    """Legacy function for backward compatibility."""
    actual_wis = 0.0
    for u_rank in unmapped_clusters:
        # Avoid division by zero for rank 0. Treat rank 0 as rank 1 for penalty calculation.
        rank_for_calc = u_rank + 1
        inversion_count = sum(1 for m_rank in mapped_clusters if u_rank < m_rank)
        if inversion_count > 0:
            actual_wis += inversion_count / rank_for_calc

    if actual_wis == 0:
        return 0.0

    # 2. Calculate the maximum possible WIS (denominator)
    num_unmapped = len(unmapped_clusters)
    num_mapped = len(mapped_clusters)

    if num_unmapped == 0 or num_mapped == 0:
        return 0.0

    # The worst-case ranks for unmapped items are {1, 2, ..., num_unmapped}
    # We calculate the harmonic sum: 1/1 + 1/2 + ... + 1/num_unmapped
    harmonic_sum = sum(1.0/i for i in range(1, num_unmapped + 1))
    max_wis = num_mapped * harmonic_sum

    if max_wis == 0:
        return 0.0 # Should not happen if lists are not empty, but for safety

    # 3. Return the normalized score
    nwis = actual_wis / max_wis
    print(f"Actual WIS: {actual_wis:.4f}, Max WIS: {max_wis:.4f}, NWIS: {nwis:.4f}")
    
    return min(nwis, 1.0) # Cap at 1.0 to handle any floating point inaccuracies


# def plot_nd_ratios_by_rank(nd_analysis: Dict[str, Any], output_path: str):
#     """
#     Generate a plot showing ND ratios by cluster rank.
    
#     Args:
#         nd_analysis: ND analysis results containing cluster details
#         output_path: Path to save the plot
#     """
#     if not nd_analysis['unmapped_cluster_details']:
#         print("No unmapped cluster details available for plotting.")
#         return
    
#     # Extract data for plotting
#     cluster_ranks = []
#     doc_all_ratios = []
#     doc_partial_ratios = []
#     chunk_nd_ratios = []
    
#     for detail in nd_analysis['unmapped_cluster_details']:
#         cluster_ranks.append(detail['cluster_rank'] + 1)  # Convert to 1-based ranking
#         doc_all_ratios.append(detail['document_level_nd_all_ratio'])
#         doc_partial_ratios.append(detail['document_level_nd_partial_ratio'])
#         chunk_nd_ratios.append(detail['chunk_level_nd_ratio'])
    
#     # Create the plot
#     plt.figure(figsize=(12, 8))
    
#     plt.plot(cluster_ranks, doc_all_ratios, 'o-', label='Document-level ND (All)', linewidth=2, markersize=6)
#     plt.plot(cluster_ranks, doc_partial_ratios, 's-', label='Document-level ND (Partial)', linewidth=2, markersize=6)
#     plt.plot(cluster_ranks, chunk_nd_ratios, '^-', label='Chunk-level ND', linewidth=2, markersize=6)
    
#     plt.xlabel('Unmapped Cluster Rank', fontsize=12)
#     plt.ylabel('ND Ratio', fontsize=12)
#     plt.title('Noise Domination Ratios by Cluster Rank (Chunk-level Analysis)', fontsize=14)
#     plt.legend(fontsize=11)
#     plt.grid(True, alpha=0.3)
    
#     # Set y-axis to percentage format
#     plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
#     # Add summary statistics as text
#     avg_doc_all = nd_analysis['avg_document_level_nd_all_ratio']
#     avg_doc_partial = nd_analysis['avg_document_level_nd_partial_ratio']
#     avg_chunk_nd = nd_analysis['avg_chunk_level_nd_ratio']
    
#     stats_text = f'Average Ratios:\nDoc-All: {avg_doc_all:.1%}\nDoc-Partial: {avg_doc_partial:.1%}\nChunk-ND: {avg_chunk_nd:.1%}'
#     plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
#              fontsize=10)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     print(f"ND ratios plot saved to: {output_path}")
#     plt.close()


def main():
    # File paths
    json_file_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/reframe/cache_ai_job_seeking.json"
    results_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/reframe/results_ai_job_seeking.json"
    raw_json_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/temp/ai_job_seeking.json"
    noise_domination_detection(results_file, json_file_path, raw_json_file, num_gpus=4, gpu_ids=[0, 1, 2, 3])
   
if __name__ == "__main__":
    main()
