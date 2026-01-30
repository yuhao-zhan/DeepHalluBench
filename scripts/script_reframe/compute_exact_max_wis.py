import json
import os
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional
import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def load_clusters_from_json(json_file: str) -> Tuple[List[Dict], int, int]:
    """
    Load clusters information from JSON file.
    
    Returns:
        (clusters, num_mapped, num_unmapped)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data.get('clusters', [])
    
    # Count mapped and unmapped clusters
    num_mapped = sum(1 for c in clusters if c.get('is_mapped', False))
    num_unmapped = sum(1 for c in clusters if not c.get('is_mapped', False))
    
    return clusters, num_mapped, num_unmapped


def compute_wis_for_combination(
    unmapped_ranks: List[int],
    all_clusters: List[Dict],
    mapped_ranks: List[int]
) -> float:
    """
    Compute WIS for a specific combination of unmapped clusters.
    
    Args:
        unmapped_ranks: List of cluster ranks (0-based) selected as unmapped
        all_clusters: All clusters with their information
        mapped_ranks: List of cluster ranks that are actually mapped
    
    Returns:
        WIS value for this combination
    """
    wis = 0.0
    
    for u_rank in unmapped_ranks:
        # Get cluster information
        cluster = all_clusters[u_rank]
        num_chunks = len(cluster.get('chunks', []))
        
        # Calculate inversion_count: how many mapped clusters rank after this unmapped cluster
        # Note: In this combination, mapped_ranks are the clusters that are NOT in unmapped_ranks
        inversion_count = sum(1 for m_rank in mapped_ranks if u_rank < m_rank)
        
        # WIS contribution: (inversion_count / rank) * num_chunks
        # rank is 1-based (u_rank + 1)
        rank_for_calc = u_rank + 1
        if inversion_count > 0:
            wis += (inversion_count / rank_for_calc) * num_chunks
    
    return wis


def generate_pruned_combinations(
    total_clusters: int,
    num_unmapped: int,
    max_unmapped_rank: Optional[int] = None
) -> List[Tuple[int, ...]]:
    """
    Generate pruned combinations where unmapped clusters are only from top ranks.
    
    Args:
        total_clusters: Total number of clusters
        num_unmapped: Number of unmapped clusters to select
        max_unmapped_rank: Maximum rank to consider for unmapped clusters.
                          If None, uses heuristic: min(total_clusters, num_unmapped * 3)
    
    Returns:
        List of combinations (tuples of ranks)
    """
    if max_unmapped_rank is None:
        # Heuristic: only consider top ranks for unmapped clusters
        # Reason: WIS contribution is (inversion_count / rank) * num_chunks
        # Low ranks (top positions) maximize this by having small denominator
        # We allow some flexibility to account for clusters with many chunks:
        # Consider up to 2x num_unmapped or num_unmapped + 20, whichever is smaller
        max_unmapped_rank = min(
            total_clusters - 1,
            max(num_unmapped - 1, min(num_unmapped * 2 - 1, num_unmapped + 20 - 1))
        )
    
    # Only consider ranks from 0 to max_unmapped_rank
    candidate_ranks = list(range(max_unmapped_rank + 1))
    
    # If we don't have enough candidates, use all ranks
    if len(candidate_ranks) < num_unmapped:
        candidate_ranks = list(range(total_clusters))
    
    # Generate combinations from pruned candidate set
    return list(combinations(candidate_ranks, num_unmapped))


def process_combination_batch_wrapper(args: Tuple[List[Tuple[int, ...]], List[Dict], List[int]]) -> Tuple[float, List[int]]:
    """
    Wrapper function for multiprocessing that unpacks arguments.
    
    Args:
        args: Tuple of (combo_batch, clusters, all_ranks)
    
    Returns:
        (max_wis, best_combination)
    """
    combo_batch, clusters, all_ranks = args
    return process_combination_batch(combo_batch, clusters, all_ranks)


def process_combination_batch(
    combo_batch: List[Tuple[int, ...]],
    clusters: List[Dict],
    all_ranks: List[int]
) -> Tuple[float, List[int]]:
    """
    Process a batch of combinations and return the best WIS and combination.
    
    Args:
        combo_batch: List of combinations to process
        clusters: All clusters
        all_ranks: All possible ranks
    
    Returns:
        (max_wis, best_combination)
    """
    max_wis = -1.0
    best_combination = None
    
    for combo in combo_batch:
        unmapped_combo = list(combo)
        mapped_combo = [r for r in all_ranks if r not in unmapped_combo]
        
        wis = compute_wis_for_combination(unmapped_combo, clusters, mapped_combo)
        
        if wis > max_wis:
            max_wis = wis
            best_combination = sorted(unmapped_combo)
    
    return max_wis, best_combination


def find_max_wis_combination(
    clusters: List[Dict],
    num_unmapped: int,
    num_mapped: int,
    use_tqdm: bool = False,
    num_cores: int = 128,
    max_unmapped_rank: Optional[int] = None
) -> Tuple[float, List[int], bool]:
    """
    Enumerate pruned combinations and find the one with maximum WIS using parallel processing.
    
    Args:
        clusters: All clusters
        num_unmapped: Number of unmapped clusters
        num_mapped: Number of mapped clusters
        use_tqdm: Whether to use tqdm for progress bar
        num_cores: Number of CPU cores to use for parallel processing
        max_unmapped_rank: Maximum rank to consider for unmapped clusters (pruning parameter)
    
    Returns:
        (max_wis, best_combination, is_approximation_case)
    """
    total_clusters = len(clusters)
    
    if num_unmapped == 0 or num_mapped == 0:
        return 0.0, [], False
    
    # The approximation case: unmapped clusters are ranks 0 to num_unmapped-1
    approximation_case = list(range(num_unmapped))
    
    # Generate pruned combinations
    log = tqdm.write if use_tqdm else print
    
    # Calculate original combination count for comparison
    try:
        original_count = math.comb(total_clusters, num_unmapped)
        original_count_str = f"{original_count:,}"
    except (OverflowError, ValueError):
        original_count = None
        original_count_str = "too large to calculate"
    
    log(f"  Generating pruned combinations (only considering top ranks for unmapped clusters)...")
    
    pruned_combinations = generate_pruned_combinations(total_clusters, num_unmapped, max_unmapped_rank)
    num_combinations = len(pruned_combinations)
    
    if original_count:
        reduction_ratio = (1 - num_combinations / original_count) * 100
        log(f"  Reduced from C({total_clusters}, {num_unmapped}) = {original_count_str} to {num_combinations:,} combinations ({reduction_ratio:.2f}% reduction)")
    else:
        log(f"  Generated {num_combinations:,} pruned combinations (original count was {original_count_str})")
    
    if num_combinations == 0:
        log(f"  No valid combinations after pruning")
        return 0.0, [], False
    
    all_ranks = list(range(total_clusters))
    
    # Split combinations into batches for parallel processing
    batch_size = max(1, num_combinations // (num_cores * 4))  # 4 batches per core
    batches = []
    for i in range(0, num_combinations, batch_size):
        batches.append(pruned_combinations[i:i + batch_size])
    
    log(f"  Processing {len(batches)} batches across {num_cores} cores...")
    
    # Prepare arguments for multiprocessing
    process_args = [(batch, clusters, all_ranks) for batch in batches]
    
    max_wis = -1.0
    best_combination = None
    
    with Pool(processes=min(num_cores, cpu_count(), len(batches))) as pool:
        if use_tqdm:
            results = list(tqdm(
                pool.imap(process_combination_batch_wrapper, process_args),
                total=len(batches),
                desc="  Processing batches",
                leave=False
            ))
        else:
            results = pool.map(process_combination_batch_wrapper, process_args)
    
    # Find the best result across all batches
    for wis, combo in results:
        if wis > max_wis:
            max_wis = wis
            best_combination = combo
    
    # Check if best_combination matches the approximation case
    is_approximation = (sorted(best_combination) == sorted(approximation_case)) if best_combination else False
    
    return max_wis, best_combination, is_approximation


def process_json_file(json_file: str, use_tqdm: bool = False) -> Dict[str, Any]:
    """
    Process a single JSON file to find maximum WIS.
    """
    log = tqdm.write if use_tqdm else print
    
    log(f"\nProcessing: {os.path.basename(json_file)}")
    
    try:
        clusters, num_mapped, num_unmapped = load_clusters_from_json(json_file)
        
        if len(clusters) == 0:
            log(f"  No clusters found")
            return {
                'file': os.path.basename(json_file),
                'error': 'No clusters found'
            }
        
        log(f"  Total clusters: {len(clusters)}")
        log(f"  Mapped: {num_mapped}, Unmapped: {num_unmapped}")
        
        if num_unmapped == 0 or num_mapped == 0:
            log(f"  Skipping: no unmapped or mapped clusters")
            return {
                'file': os.path.basename(json_file),
                'total_clusters': len(clusters),
                'num_mapped': num_mapped,
                'num_unmapped': num_unmapped,
                'max_wis': 0.0,
                'best_combination': [],
                'is_approximation_case': False,
                'skipped': True
            }
        
        # Compute approximation WIS (current method: unmapped clusters are ranks 0 to num_unmapped-1)
        approximation_unmapped = list(range(num_unmapped))
        approximation_mapped = list(range(num_unmapped, len(clusters)))
        approximation_wis = compute_wis_for_combination(
            approximation_unmapped,
            clusters,
            approximation_mapped
        )
        
        # Find exact maximum WIS with parallel processing and pruning
        max_wis, best_combination, is_approximation = find_max_wis_combination(
            clusters, num_unmapped, num_mapped, 
            use_tqdm=use_tqdm, 
            num_cores=128
        )
        
        log(f"  Approximation WIS: {approximation_wis:.6f}")
        log(f"  Maximum WIS: {max_wis:.6f}")
        log(f"  Best combination: {best_combination}")
        log(f"  Is approximation case: {is_approximation}")
        
        return {
            'file': os.path.basename(json_file),
            'total_clusters': len(clusters),
            'num_mapped': num_mapped,
            'num_unmapped': num_unmapped,
            'approximation_wis': approximation_wis,
            'max_wis': max_wis,
            'best_combination': best_combination,
            'is_approximation_case': is_approximation,
            'difference': max_wis - approximation_wis,
            'relative_difference': (max_wis - approximation_wis) / max_wis if max_wis > 0 else 0.0
        }
        
    except Exception as e:
        log = tqdm.write if use_tqdm else print
        log(f"  Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return {
            'file': os.path.basename(json_file),
            'error': str(e)
        }


def main():
    # Directory containing JSON files
    input_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Mind2Web2/gemini/after_update"
    output_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Mind2Web2/gemini/exact_max_wis_results.json"
    
    # Get all JSON files
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file with progress bar
    results = []
    for json_file in tqdm(json_files, desc="Processing files"):
        result = process_json_file(json_file, use_tqdm=True)
        results.append(result)
    
    # Save results
    output_data = {
        'summary': {
            'total_files': len(json_files),
            'processed_files': len([r for r in results if 'error' not in r]),
            'files_with_errors': len([r for r in results if 'error' in r]),
            'approximation_correct': len([r for r in results if r.get('is_approximation_case', False)]),
            'approximation_incorrect': len([r for r in results if not r.get('is_approximation_case', False) and 'error' not in r])
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total files: {output_data['summary']['total_files']}")
    print(f"  Processed: {output_data['summary']['processed_files']}")
    print(f"  Errors: {output_data['summary']['files_with_errors']}")
    print(f"  Approximation correct: {output_data['summary']['approximation_correct']}")
    print(f"  Approximation incorrect: {output_data['summary']['approximation_incorrect']}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

