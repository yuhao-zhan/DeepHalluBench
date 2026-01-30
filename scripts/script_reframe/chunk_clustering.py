import numpy as np
import json
import os
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import umap
import hdbscan
from multiprocessing import Pool, cpu_count
import time
import torch
import concurrent.futures
import threading

# Suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*force_all_finite.*")
warnings.filterwarnings("ignore", message=".*ensure_all_finite.*")


def _encode_texts_on_device(args):
    """
    Encode a shard of texts on a specific physical CUDA device.
    Args: (texts, device_idx, batch_size, start_idx)
        device_idx: Physical GPU ID (e.g., 0, 1, 2, 3)
    Returns: (start_idx, embeddings_np)
    """
    texts, device_idx, batch_size, start_idx = args
    try:
        from sentence_transformers import SentenceTransformer
        if torch.cuda.is_available():
            # Use physical GPU ID directly
            torch.cuda.set_device(device_idx)
        # Lazy, thread-safe per-device model cache
        global _MODEL_CACHE, _MODEL_LOCK
        try:
            _MODEL_CACHE
        except NameError:
            _MODEL_CACHE = {}
        try:
            _MODEL_LOCK
        except NameError:
            _MODEL_LOCK = threading.Lock()
        with _MODEL_LOCK:
            if device_idx not in _MODEL_CACHE:
                # Prefer direct device initialization to avoid meta->cuda copy
                try:
                    model = SentenceTransformer("BAAI/bge-m3", device=f"cuda:{device_idx}") if torch.cuda.is_available() else SentenceTransformer("BAAI/bge-m3")
                except Exception as e1:
                    # Fallback: construct then move; try to_empty if available
                    model = SentenceTransformer("BAAI/bge-m3")
                    if torch.cuda.is_available():
                        try:
                            if hasattr(model, 'to_empty'):
                                model = model.to_empty(device=f"cuda:{device_idx}")
                            else:
                                model.to(torch.device(f"cuda:{device_idx}"))
                        except Exception as e2:
                            raise
                _MODEL_CACHE[device_idx] = model
            else:
                model = _MODEL_CACHE[device_idx]
        embs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
            convert_to_numpy=True
        )
        return (start_idx, embs)
    except Exception as e:
        print(f"❌ Encoding shard failed on device cuda:{device_idx}: {e}")
        raise


def extract_chunk_texts(json_file_path):
    """
    Extract all chunk_text values and their IDs from the 'chunk_score' attribute in the JSON file.
    
    Args:
        json_file_path: Path to the JSON file
        
    Returns:
        List of tuples (chunk_id, chunk_text)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunk_data = []
        
        # Check if chunk_score exists in the data
        if 'chunk_score' in data:
            chunk_score_data = data['chunk_score']
            
            # Iterate through all entries in chunk_score
            for chunk_id, chunk_info in chunk_score_data.items():
                if 'chunk_text' in chunk_info:
                    chunk_data.append((chunk_id, chunk_info['chunk_text']))
                    print(f"Extracted chunk {len(chunk_data)}: {chunk_id}")
                else:
                    print(f"Warning: No chunk_text found for {chunk_id}")
        else:
            print("Error: 'chunk_score' attribute not found in the JSON file")
            return []
        
        print(f"\nTotal chunks extracted: {len(chunk_data)}")
        return chunk_data
        
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def cluster_chunks_umap_hdbscan(chunks: List[tuple], 
                                n_neighbors: int = 15, 
                                min_cluster_size: int = 3, 
                                min_samples: int = 2,
                                cluster_selection_epsilon: float = 0.0) -> List[List[tuple]]:
    """
    Cluster chunks using UMAP for dimensionality reduction and HDBSCAN for density-based clustering.
    This approach provides better differentiation and avoids overly large clusters.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text) to cluster
        n_neighbors: Number of neighbors for UMAP (controls local vs global structure)
        min_cluster_size: Minimum size of clusters for HDBSCAN
        min_samples: Minimum samples in neighborhood for HDBSCAN
        cluster_selection_epsilon: Distance threshold for cluster selection in HDBSCAN
    
    Returns:
        List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return [chunks]
    
    # Extract just the text for embedding
    chunk_texts = [chunk[1] for chunk in chunks]
    
    # Load the BGE-M3 model
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    # Convert chunks to embeddings
    print("Converting chunks to embeddings...")
    embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Apply UMAP for dimensionality reduction
    print("Applying UMAP dimensionality reduction...")
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=min(50, len(chunks) - 1),  # Reduce to max 50 dimensions or chunks-1
        metric='cosine',
        random_state=42
    )
    
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    
    # Apply HDBSCAN clustering
    print("Applying HDBSCAN density-based clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean'
    )
    
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # Group chunks by cluster labels
    clusters = {}
    noise_chunks = []
    
    for i, label in enumerate(cluster_labels):
        if label == -1:  # Noise points
            noise_chunks.append(chunks[i])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[i])
    
    # Convert to list of lists
    result_clusters = list(clusters.values())
    
    # Add noise chunks as individual clusters if any
    for noise_chunk in noise_chunks:
        result_clusters.append([noise_chunk])
    
    print(f"Created {len(result_clusters)} clusters from {len(chunks)} chunks")
    print(f"Noise points (individual clusters): {len(noise_chunks)}")
    
    return result_clusters


def cluster_chunks_fast_parallel(chunks: List[tuple], 
                                similarity_threshold: float = 0.75,
                                max_cluster_size: int = 10) -> List[List[tuple]]:
    """
    Fast parallel clustering using cosine similarity with multiprocessing.
    Much faster than UMAP+HDBSCAN for large datasets.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text) to cluster
        similarity_threshold: Threshold for determining if chunks are similar
        max_cluster_size: Maximum allowed cluster size
    
    Returns:
        List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return [chunks]
    
    print(f"Fast parallel clustering with {cpu_count()} CPU cores...")
    start_time = time.time()
    
    # Extract just the text for embedding
    chunk_texts = [chunk[1] for chunk in chunks]
    
    # Load the BGE-M3 model
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    # Convert chunks to embeddings with progress bar
    print("Converting chunks to embeddings...")
    embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Calculate pairwise cosine similarities in parallel
    print("Calculating similarity matrix in parallel...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create clusters based on similarity threshold
    clusters = []
    used_chunks = set()
    
    # Sort chunks by similarity to others (most similar first)
    chunk_similarities = np.sum(similarity_matrix, axis=1)
    sorted_indices = np.argsort(chunk_similarities)[::-1]
    
    for i in sorted_indices:
        if i in used_chunks:
            continue
            
        # Start a new cluster with this chunk
        current_cluster = [chunks[i]]
        used_chunks.add(i)
        
        # Find all chunks similar to this one
        for j in sorted_indices:
            if j in used_chunks:
                continue
                
            if similarity_matrix[i][j] >= similarity_threshold:
                current_cluster.append(chunks[j])
                used_chunks.add(j)
                
                # Limit cluster size
                if len(current_cluster) >= max_cluster_size:
                    break
        
        clusters.append(current_cluster)
    
    end_time = time.time()
    print(f"Fast clustering completed in {end_time - start_time:.2f} seconds")
    print(f"Created {len(clusters)} clusters from {len(chunks)} chunks")
    
    return clusters


def cluster_chunks_umap_hdbscan_tuned(chunks: List[tuple], 
                                     target_cluster_size: int = 5,
                                     max_cluster_size: int = 15,
                                     num_gpus: int = 1,
                                     gpu_ids: List[int] = None) -> List[List[tuple]]:
    """
    Tuned UMAP+HDBSCAN clustering that attempts to create clusters of reasonable size.
    Automatically adjusts parameters to avoid overly large clusters.
    Supports multi-GPU parallel processing for faster embedding generation.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text) to cluster
        target_cluster_size: Target size for clusters
        max_cluster_size: Maximum allowed cluster size
        num_gpus: Number of GPUs to use for parallel processing
        gpu_ids: List of specific GPU IDs to use (if None, uses first num_gpus)
    
    Returns:
        List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return [chunks]
    
    # Extract just the text for embedding
    chunk_texts = [chunk[1] for chunk in chunks]
    
    print("🤖 Preparing BGE-M3 encoding...")
    embed_start = time.time()
    device = "cuda" if torch.cuda.is_available() and num_gpus > 0 else "cpu"
    if gpu_ids is None:
        gpu_ids = list(range(max(1, num_gpus)))
    
    if device == "cuda" and len(gpu_ids) > 1 and len(chunk_texts) > 0:
        # Use physical GPU IDs directly (not logical indices)
        # This allows us to use specific GPUs without CUDA_VISIBLE_DEVICES remapping
        print(f"🚀 Sharded multi-GPU encoding on physical GPUs: {gpu_ids}")
        num_workers = min(len(gpu_ids), len(chunk_texts))
        # Partition texts into contiguous shards
        shard_sizes = []
        base = len(chunk_texts) // num_workers
        rem = len(chunk_texts) % num_workers
        for i in range(num_workers):
            shard_sizes.append(base + (1 if i < rem else 0))
        tasks = []
        start = 0
        for i in range(num_workers):
            end = start + shard_sizes[i]
            texts_shard = chunk_texts[start:end]
            # Use physical GPU ID directly
            physical_device_idx = gpu_ids[i]
            per_gpu_batch = 32
            tasks.append((texts_shard, physical_device_idx, per_gpu_batch, start))
            start = end
        # Encode shards in parallel threads (CUDA safe without fork)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_encode_texts_on_device, t) for t in tasks]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        embeddings = np.concatenate([emb for (_, emb) in results], axis=0)
    else:
        # Single-device path
        if device == "cuda" and gpu_ids and len(gpu_ids) > 0:
            # Use the first specified GPU ID
            device_str = f"cuda:{gpu_ids[0]}"
        else:
            device_str = "cuda:0" if device == "cuda" else "cpu"
        print(f"🎯 Single-device encoding on {device_str}")
        model = SentenceTransformer("BAAI/bge-m3", device=device_str)
        embeddings = model.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
            device=device_str
        )
    
    embed_time = time.time() - embed_start
    print(f"✅ Generated embeddings shape: {embeddings.shape} in {embed_time:.2f} seconds")
    
    # Clean up GPU memory after embedding generation
    if device == "cuda":
        try:
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 GPU memory cleaned up after embedding generation")
        except Exception:
            pass
    
    # Apply UMAP for dimensionality reduction with multi-core acceleration
    print("🗺️  Applying UMAP dimensionality reduction with multi-core acceleration...")
    umap_start = time.time()
    n_jobs = min(128, cpu_count())  # Use up to 128 cores
    print(f"🔧 Using {n_jobs} CPU cores for UMAP")
    
    # Ensure n_components is strictly less than number of samples to avoid sparse matrix issues
    # For small datasets, use a conservative n_components (max 2-3)
    # For larger datasets, use min(50, len(chunks) - 2) to ensure n_components < N
    num_samples = len(chunks)
    if num_samples <= 3:
        # For very small datasets, skip UMAP or use minimal reduction
        n_components = max(1, num_samples - 1)
    elif num_samples <= 10:
        # For small datasets, use at most 2-3 components
        n_components = min(3, num_samples - 1)
    else:
        # For larger datasets, use min(50, num_samples - 2) to ensure n_components < N
        n_components = min(50, max(2, num_samples - 2))
    
    # Ensure n_neighbors is also properly bounded
    n_neighbors_val = min(15, max(2, num_samples - 1))
    
    print(f"📊 UMAP config: n_components={n_components}, n_neighbors={n_neighbors_val}, num_samples={num_samples}")
    
    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors_val,
        min_dist=0.1,
        n_components=n_components,
        metric='cosine',
        random_state=42,
        n_jobs=n_jobs,
        verbose=True
    )
    
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    umap_time = time.time() - umap_start
    print(f"✅ UMAP reduced embeddings shape: {reduced_embeddings.shape} in {umap_time:.2f} seconds")
    
    # Use fixed HDBSCAN parameters
    min_cluster_size = 2
    min_samples = 1
    epsilon = 0
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
        core_dist_n_jobs=n_jobs,
        cluster_selection_method='eom'
    )
    
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # Group chunks by cluster labels
    clusters = {}
    noise_chunks = []
    
    for i, label in enumerate(cluster_labels):
        if label == -1:  # Noise points
            noise_chunks.append(chunks[i])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[i])
    
    # Convert to list of lists
    result_clusters = list(clusters.values())
    
    # Add noise chunks as individual clusters
    for noise_chunk in noise_chunks:
        result_clusters.append([noise_chunk])
    
    print(f"Created {len(result_clusters)} clusters from {len(chunks)} chunks")
    print(f"Noise points (individual clusters): {len(noise_chunks)}")
    print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")
    print(f"Cluster sizes: {[len(cluster) for cluster in result_clusters]}")
    
    return result_clusters


def cluster_chunks(chunks: List[tuple], similarity_threshold: float = 0.7) -> List[List[tuple]]:
    """
    Cluster chunks based on semantic similarity using BGE-M3 embeddings.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text) to cluster
        similarity_threshold: Threshold for determining if chunks are similar (0.0 to 1.0)
    
    Returns:
        List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return [chunks]
    
    # Extract just the text for embedding
    chunk_texts = [chunk[1] for chunk in chunks]
    
    # Load the BGE-M3 model
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    # Convert chunks to embeddings
    print("Converting chunks to embeddings...")
    embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Calculate pairwise cosine similarities
    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create a distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Use hierarchical clustering with automatic cluster determination
    # We'll use a distance threshold approach to determine clusters
    print("Performing hierarchical clustering...")
    
    # Start with a conservative distance threshold
    distance_threshold = 1 - similarity_threshold
    
    # Use AgglomerativeClustering with distance_threshold to automatically determine clusters
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Let the algorithm determine number of clusters
        distance_threshold=distance_threshold,
        linkage='average',
        metric='precomputed'
    )
    
    # Fit the clustering model
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Group chunks by cluster labels
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[i])
    
    # Convert to list of lists
    result_clusters = list(clusters.values())
    
    print(f"Created {len(result_clusters)} clusters from {len(chunks)} chunks")
    
    return result_clusters


def cluster_chunks_adaptive(chunks: List[tuple], min_similarity: float = 0.6, max_similarity: float = 0.9) -> List[List[tuple]]:
    """
    Adaptive clustering that tries different similarity thresholds to find optimal clustering.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text) to cluster
        min_similarity: Minimum similarity threshold to try
        max_similarity: Maximum similarity threshold to try
    
    Returns:
        List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return [chunks]
    
    # Extract just the text for embedding
    chunk_texts = [chunk[1] for chunk in chunks]
    
    # Load the BGE-M3 model
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    # Convert chunks to embeddings
    print("Converting chunks to embeddings...")
    embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Calculate pairwise cosine similarities
    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create a distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Try different similarity thresholds to find optimal clustering
    best_clusters = None
    best_score = -1
    
    # Test different thresholds
    thresholds = np.linspace(min_similarity, max_similarity, 10)
    
    for threshold in thresholds:
        distance_threshold = 1 - threshold
        print(f"Trying threshold: {distance_threshold}")
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        n_clusters = len(set(cluster_labels))
        
        if n_clusters == 0:
            continue
            
        # Calculate clustering quality score
        # We want to balance between having reasonable cluster sizes and good separation
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
        avg_cluster_size = np.mean(cluster_sizes)
        size_variance = np.var(cluster_sizes)
        
        # Score based on cluster balance and number of clusters
        # Prefer moderate number of clusters with balanced sizes
        if n_clusters == 1 and len(chunks) > 1:
            # If all chunks are very similar, single cluster is good
            score = 1.0
        elif n_clusters > len(chunks) // 2:
            # Too many clusters
            score = 0.0
        else:
            # Balance between cluster count and size distribution
            size_balance = 1.0 / (1.0 + size_variance / (avg_cluster_size ** 2))
            cluster_count_score = 1.0 / (1.0 + abs(n_clusters - len(chunks) // 4))
            score = size_balance * cluster_count_score
        
        if score > best_score:
            best_score = score
            best_clusters = cluster_labels
    
    if best_clusters is None:
        # Fallback to single cluster if no good clustering found
        return [chunks]
    
    # Group chunks by best cluster labels
    clusters = {}
    for i, label in enumerate(best_clusters):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[i])
    
    result_clusters = list(clusters.values())
    
    print(f"Created {len(result_clusters)} clusters from {len(chunks)} chunks (best score: {best_score:.3f})")
    
    return result_clusters


def cluster_chunks_simple(chunks: List[tuple], similarity_threshold: float = 0.75) -> List[List[tuple]]:
    """
    Simple clustering approach using a fixed similarity threshold.
    This is faster but less adaptive than the other methods.
    
    Args:
        chunks: List of tuples (chunk_id, chunk_text) to cluster
        similarity_threshold: Threshold for determining if chunks are similar
    
    Returns:
        List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    """
    if not chunks:
        return []
    
    if len(chunks) == 1:
        return [chunks]
    
    # Extract just the text for embedding
    chunk_texts = [chunk[1] for chunk in chunks]
    
    # Load the BGE-M3 model
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    # Convert chunks to embeddings
    print("Converting chunks to embeddings...")
    embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    
    # Calculate pairwise cosine similarities
    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create clusters based on similarity threshold
    clusters = []
    used_chunks = set()
    
    for i, chunk in enumerate(chunks):
        if i in used_chunks:
            continue
            
        # Start a new cluster with this chunk
        current_cluster = [chunk]
        used_chunks.add(i)
        
        # Find all chunks similar to this one
        for j, other_chunk in enumerate(chunks):
            if j in used_chunks:
                continue
                
            if similarity_matrix[i][j] >= similarity_threshold:
                current_cluster.append(other_chunk)
                used_chunks.add(j)
        
        clusters.append(current_cluster)
    
    print(f"Created {len(clusters)} clusters from {len(chunks)} chunks")
    
    return clusters


# Only keep the UMAP+HDBSCAN tuned method


def create_chunk_id_to_cluster_mapping(clusters: List[List[tuple]]) -> dict:
    """
    Create a mapping from chunk_id to cluster index for easy lookup.
    
    Args:
        clusters: List of clusters, where each cluster is a list of tuples (chunk_id, chunk_text)
    
    Returns:
        Dictionary mapping chunk_id to cluster index
    """
    chunk_to_cluster = {}
    for cluster_idx, cluster in enumerate(clusters):
        for chunk_id, chunk_text in cluster:
            chunk_to_cluster[chunk_id] = cluster_idx
    return chunk_to_cluster


if __name__ == "__main__":
    # Load full dataset from JSON file
    print("=== LOADING FULL DATASET ===")
    json_file_path = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/train_gemini/cache_ai_job_seeking.json"
    
    print("📂 Extracting chunks from JSON file...")
    chunk_data = extract_chunk_texts(json_file_path)
    
    if not chunk_data:
        print("❌ No chunks extracted from JSON file!")
        exit(1)
    
    print(f"✅ Extracted {len(chunk_data)} chunks total")
    print("📝 Sample chunks:")
    for i, (chunk_id, text) in enumerate(chunk_data[:3]):
        print(f"  {i+1}. {chunk_id}: {text[:50]}...")
    
    # Cluster all chunks using UMAP+HDBSCAN tuned with GPU acceleration
    print(f"\n=== STARTING UMAP+HDBSCAN CLUSTERING ===")
    print(f"📊 Input: {len(chunk_data)} chunks")
    print(f"🎯 Method: umap_hdbscan_tuned")
    print(f"🚀 Using GPU acceleration and multi-core processing")
    
    start_time = time.time()
    clusters = cluster_chunks_umap_hdbscan_tuned(chunk_data)
    end_time = time.time()
    
    print(f"⏱️  Clustering completed in {end_time - start_time:.2f} seconds")
    
    print(f"\n✅ Clustering completed! Created {len(clusters)} clusters.")
    
    # Print cluster size distribution
    cluster_sizes = [len(cluster) for cluster in clusters]
    print(f"📊 Cluster sizes: {cluster_sizes}")
    print(f"📈 Average cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"📊 Max cluster size: {max(cluster_sizes)}")
    print(f"📊 Min cluster size: {min(cluster_sizes)}")
    
    # Create mapping for easy lookup
    chunk_to_cluster = create_chunk_id_to_cluster_mapping(clusters)
    
    # Save clustered results
    output_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/test/clustering_results_ai_jobs_umap_hdbscan_full.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"UMAP+HDBSCAN CLUSTERING RESULTS\n")
        f.write(f"Total chunks: {len(chunk_data)}\n")
        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"Cluster sizes: {cluster_sizes}\n")
        f.write("="*50 + "\n\n")
        
        for i, cluster in enumerate(clusters):
            f.write(f"=== CLUSTER {i+1} (Size: {len(cluster)}) ===\n")
            for j, (chunk_id, chunk_text) in enumerate(cluster):
                f.write(f"{j+1}. [{chunk_id}] {chunk_text[:200]}...\n")  # Truncate long text
            f.write("\n")
    
    print(f"💾 Clustering results saved to {output_file}")
    
    # Test the mapping
    print(f"\n🧪 Testing chunk mapping...")
    test_chunk_id = chunk_data[0][0] if chunk_data else None
    if test_chunk_id:
        cluster_idx = chunk_to_cluster.get(test_chunk_id)
        print(f"Chunk {test_chunk_id} maps to cluster {cluster_idx}")
    
    print("\n🎉 Script completed successfully!")
