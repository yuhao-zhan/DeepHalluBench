#!/usr/bin/env python3
"""
Similarity Filtering Script

This script processes claims against web content chunks to find relevant matches above a similarity threshold.
It uses the OptimizedContextLocator to chunk web content and the BAAI/bge-m3 model for similarity computation.

Input: list of claims, list of URLs, web content file
Output: JSON file with claims and their relevant chunks above similarity threshold
"""

import json
import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
import torch
import concurrent.futures
import threading

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from local utils module - use absolute path to avoid multiprocessing conflicts
import os
import sys
import importlib.util
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import with explicit module path to avoid conflicts
spec = importlib.util.spec_from_file_location("local_utils", os.path.join(current_dir, "utils.py"))
local_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_utils)

# Now import the function we need
OptimizedContextLocator = local_utils.OptimizedContextLocator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiGPUManager:
    """Manages multiple GPUs with MAXIMUM memory utilization."""
    
    def __init__(self, num_gpus: int = 4, gpu_ids: List[int] = None):
        """
        Initialize MultiGPUManager with specific GPU IDs.
        
        Args:
            num_gpus: Number of GPUs to use (default: 4)
            gpu_ids: List of specific GPU IDs to use (e.g., [0, 1, 3]). If None, uses first num_gpus.
        """
        if gpu_ids is not None:
            # Use specific GPU IDs
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            self.gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < available_gpus]
            self.num_gpus = len(self.gpu_ids)
            logger.info(f"Using specific GPU IDs: {self.gpu_ids}")
            
            # DO NOT set CUDA_VISIBLE_DEVICES to allow using original GPU IDs
            logger.info(f"Using original GPU IDs without CUDA_VISIBLE_DEVICES restriction")
        else:
            # Use first num_gpus
            self.num_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
            self.gpu_ids = list(range(self.num_gpus))
            logger.info(f"Using first {self.num_gpus} GPUs: {self.gpu_ids}")
        
        self.gpu_models = {}  # Store models per GPU
        self.gpu_rerankers = {}  # Store rerankers per GPU
        self.gpu_locks = {}  # Thread locks for each GPU
        self.gpu_memory_info = {}  # Store memory info for each GPU
        
        if self.num_gpus > 0:
            self._initialize_gpus()
            self._calculate_optimal_batch_sizes()
    
    def _initialize_gpus(self):
        """Initialize models on each GPU with MAXIMUM memory allocation."""
        logger.info(f"🚀 Initializing {self.num_gpus} GPUs with MAXIMUM memory utilization")
        
        # Use the original specified GPU IDs directly for device assignment
        for i, original_gpu_id in enumerate(self.gpu_ids):
            # Use the original GPU ID for device assignment to maintain consistency
            remapped_gpu_id = original_gpu_id
            try:
                torch.cuda.set_device(remapped_gpu_id)
                
                # Get GPU memory information
                props = torch.cuda.get_device_properties(remapped_gpu_id)
                total_memory = props.total_memory
                self.gpu_memory_info[original_gpu_id] = {
                    'total_memory': total_memory,
                    'name': props.name
                }
                logger.info(f"GPU {original_gpu_id} (remapped to {remapped_gpu_id}): {props.name} with {total_memory/1024**3:.2f}GB total memory")
                
                # Initialize embedding model on this GPU
                logger.info(f"Loading embedding model on GPU {original_gpu_id} (remapped to {remapped_gpu_id})")
                try:
                    # First try the standard approach
                    embedding_model = SentenceTransformer("BAAI/bge-m3", device=f'cuda:{remapped_gpu_id}')
                    self.gpu_models[original_gpu_id] = embedding_model
                    logger.info(f"✅ Embedding model loaded on GPU {original_gpu_id}")
                except Exception as e:
                    logger.warning(f"Standard loading failed for GPU {original_gpu_id}: {e}")
                    # Try alternative approach with to_empty
                    try:
                        embedding_model = SentenceTransformer("BAAI/bge-m3")
                        embedding_model = embedding_model.to_empty(device=f'cuda:{remapped_gpu_id}')
                        self.gpu_models[original_gpu_id] = embedding_model
                        logger.info(f"✅ Embedding model loaded on GPU {original_gpu_id} (alternative method)")
                    except Exception as e2:
                        logger.error(f"Alternative method also failed for GPU {original_gpu_id}: {e2}")
                        continue
                
                # Initialize reranker on this GPU
                try:
                    from FlagEmbedding import FlagReranker
                    logger.info(f"Loading reranker on GPU {original_gpu_id} (remapped to {remapped_gpu_id})")
                    # FlagReranker uses 'devices' parameter (plural), not 'device'
                    # Pass the list of GPU IDs to avoid using all GPUs
                    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices=[f'cuda:{remapped_gpu_id}'])
                    self.gpu_rerankers[original_gpu_id] = reranker
                    logger.info(f"✅ Reranker loaded on GPU {original_gpu_id}")
                except ImportError:
                    logger.warning(f"FlagEmbedding not available on GPU {original_gpu_id}")
                except Exception as e:
                    logger.error(f"Failed to load reranker on GPU {original_gpu_id}: {e}")
                
                # Create lock for this GPU
                self.gpu_locks[original_gpu_id] = threading.Lock()
                
                logger.info(f"GPU {original_gpu_id} initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize GPU {original_gpu_id}: {e}")
                continue
        
        logger.info(f"Successfully initialized {len(self.gpu_models)} GPUs")
        
        # If no GPUs were initialized, fall back to CPU
        if not self.gpu_models:
            logger.warning("No GPUs initialized, falling back to CPU")
            try:
                embedding_model = SentenceTransformer("BAAI/bge-m3", device='cpu')
                self.gpu_models[-1] = embedding_model  # Use -1 to indicate CPU
                logger.info("✅ Fallback: Embedding model loaded on CPU")
            except Exception as e:
                logger.error(f"Failed to load embedding model on CPU: {e}")
    
    def _calculate_optimal_batch_sizes(self):
        """Calculate optimal batch sizes based on available GPU memory."""
        if not self.gpu_memory_info:
            return
        
        logger.info("🧮 Calculating optimal batch sizes for MAXIMUM GPU memory utilization (80% usage)...")
        
        for gpu_id, memory_info in self.gpu_memory_info.items():
            total_memory_gb = memory_info['total_memory'] / 1024**3
            
            # Estimate memory usage per text item for BAAI/bge-m3
            # BGE-M3 has 1024 dimensions, float32 = 4 bytes per dimension
            # Plus overhead for model weights, intermediate computations, and gradients
            estimated_memory_per_item_mb = 5.0  # MB per text item (conservative estimate)
            
            # Use 80% of available memory for maximum utilization
            available_memory_gb = total_memory_gb * 0.8
            available_memory_mb = available_memory_gb * 1024
            
            optimal_batch_size = int(available_memory_mb / estimated_memory_per_item_mb)
            
            # Set reasonable bounds - minimum 2000, maximum 100000
            optimal_batch_size = max(2000, min(optimal_batch_size, 100000))
            
            memory_info['optimal_batch_size'] = optimal_batch_size
            logger.info(f"🚀 GPU {gpu_id}: Optimal batch size = {optimal_batch_size:,} items")
            logger.info(f"   - Total memory: {total_memory_gb:.2f}GB")
            logger.info(f"   - Available for batching: {available_memory_gb:.2f}GB (80%)")
            logger.info(f"   - Estimated memory per item: {estimated_memory_per_item_mb:.2f}MB")
            logger.info(f"   - Expected memory usage: {optimal_batch_size * estimated_memory_per_item_mb / 1024:.2f}GB")
    
    def get_optimal_batch_size(self, gpu_id: int = 0) -> int:
        """Get the optimal batch size for a specific GPU."""
        if gpu_id in self.gpu_memory_info:
            return self.gpu_memory_info[gpu_id].get('optimal_batch_size', 10000)
        return 10000  # Default fallback
    
    def _distribute_data_evenly(self, data: List, num_gpus: int) -> List[List]:
        """Distribute data perfectly evenly across GPUs."""
        if num_gpus <= 1:
            return [data]
        
        # Calculate exact distribution
        total_items = len(data)
        base_size = total_items // num_gpus
        remainder = total_items % num_gpus
        
        distributed = []
        start_idx = 0
        
        for gpu_id in range(num_gpus):
            # First 'remainder' GPUs get one extra item
            current_size = base_size + (1 if gpu_id < remainder else 0)
            end_idx = start_idx + current_size
            
            distributed.append(data[start_idx:end_idx])
            start_idx = end_idx
        
        return distributed
    
    def process_embeddings_parallel(self, texts: List[str], batch_size: int = None, show_progress: bool = True) -> List[np.ndarray]:
        """Process embeddings in parallel across available GPUs."""
        if not self.gpu_models:
            logger.error("No GPU models available")
            return []
        
        if -1 in self.gpu_models:
            cpu_model = self.gpu_models[-1]
            cpu_batch_size = batch_size or 5000
            if show_progress:
                from tqdm import tqdm
                embeddings = cpu_model.encode(
                    texts,
                    convert_to_numpy=True,
                    batch_size=cpu_batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )
            else:
                embeddings = cpu_model.encode(
                    texts,
                    convert_to_numpy=True,
                    batch_size=cpu_batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
            return embeddings
        
        if batch_size is None:
            first_gpu_id = list(self.gpu_models.keys())[0]
            batch_size = self.get_optimal_batch_size(first_gpu_id)
        
        distributed_texts = self._distribute_data_evenly(texts, len(self.gpu_models))
        results = [None] * len(texts)
        threads = []
        
        # Create progress bars for each GPU if show_progress is True
        progress_bars = {}
        if show_progress:
            from tqdm import tqdm
            for i, gpu_texts in enumerate(distributed_texts):
                if gpu_texts:
                    actual_gpu_id = self.gpu_ids[i]
                    if actual_gpu_id in self.gpu_models:
                        gpu_batch_size = self.get_optimal_batch_size(actual_gpu_id)
                        total_batches = (len(gpu_texts) + gpu_batch_size - 1) // gpu_batch_size
                        progress_bars[actual_gpu_id] = tqdm(
                            total=total_batches,
                            desc=f"GPU {actual_gpu_id} Embeddings",
                            position=i,
                            unit="batch"
                        )
        
        for i, gpu_texts in enumerate(distributed_texts):
            if gpu_texts:  # Only process if there are texts for this GPU
                actual_gpu_id = self.gpu_ids[i]  # Get the actual GPU ID from the list
                if actual_gpu_id in self.gpu_models:
                    start_idx = sum(len(batch) for batch in distributed_texts[:i])
                    gpu_batch_size = self.get_optimal_batch_size(actual_gpu_id)
                    
                    thread = threading.Thread(
                        target=self._process_embeddings_on_gpu,
                        args=(actual_gpu_id, gpu_texts, start_idx, results, gpu_batch_size, progress_bars.get(actual_gpu_id))
                    )
                    threads.append(thread)
                    thread.start()
                else:
                    logger.error(f"❌ GPU {actual_gpu_id} not available in models: {list(self.gpu_models.keys())}")
                    # Set all results for this GPU to None
                    start_idx = sum(len(batch) for batch in distributed_texts[:i])
                    for j in range(len(gpu_texts)):
                        results[start_idx + j] = None
        
        for thread in threads:
            thread.join()
        
        # Close progress bars
        if show_progress:
            for pbar in progress_bars.values():
                pbar.close()
        
        if any(result is None for result in results):
            failed_count = sum(1 for result in results if result is None)
            total_count = len(results)
            logger.error(f"❌ Some embeddings failed to process: {failed_count}/{total_count} failed")
            logger.error(f"   - Failed indices: {[i for i, result in enumerate(results) if result is None]}")
            logger.error(f"   - Available GPUs: {list(self.gpu_models.keys())}")
            logger.error(f"   - GPU locks: {list(self.gpu_locks.keys())}")
            return []
        
        return results
    
    def _process_embeddings_on_gpu(self, gpu_id: int, texts: List[str], start_idx: int, results: List, batch_size: int, progress_bar=None):
        """Process embeddings on a specific GPU using threading."""
        try:
            with self.gpu_locks[gpu_id]:
                # After CUDA_VISIBLE_DEVICES is set, we need to use the remapped GPU ID
                remapped_gpu_id = self.gpu_ids.index(gpu_id)
                torch.cuda.set_device(remapped_gpu_id)
                logger.info(f"🔄 Processing {len(texts)} texts on GPU {gpu_id} with batch size {batch_size}")
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    logger.debug(f"GPU {gpu_id}: Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} with {len(batch_texts)} texts")
                    
                    batch_embeddings = self.gpu_models[gpu_id].encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    
                    for j, embedding in enumerate(batch_embeddings):
                        results[start_idx + i + j] = embedding
                    
                    # Update progress bar if provided
                    if progress_bar:
                        progress_bar.update(1)
                
                logger.info(f"✅ Successfully processed {len(texts)} texts on GPU {gpu_id}")
                
        except Exception as e:
            logger.error(f"❌ Error on GPU {gpu_id}: {e}")
            logger.error(f"   - GPU {gpu_id} available: {gpu_id < torch.cuda.device_count()}")
            logger.error(f"   - GPU {gpu_id} in models: {gpu_id in self.gpu_models}")
            logger.error(f"   - Number of texts to process: {len(texts)}")
            logger.error(f"   - Batch size: {batch_size}")
            logger.error(f"   - Start index: {start_idx}")
            
            # Set all results for this GPU to None
            for i in range(len(texts)):
                results[start_idx + i] = None
    
    def process_reranking_parallel(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Process reranking in parallel across all GPUs with equal load."""
        if not self.gpu_rerankers or self.num_gpus <= 1:
            # Fallback to CPU or single GPU
            if self.gpu_rerankers:
                gpu_id = list(self.gpu_rerankers.keys())[0]
                if gpu_id >= 0:
                    torch.cuda.set_device(gpu_id)
                reranker = self.gpu_rerankers[gpu_id]
            else:
                # No rerankers available, return default scores
                logger.warning("No rerankers available, returning default scores")
                return {chunk['chunk_id']: 0.0 for chunk in chunks}
        
        # Distribute chunks perfectly evenly
        distributed_chunks = self._distribute_data_evenly(chunks, self.num_gpus)
        all_scores = [None] * len(chunks)  # Pre-allocate result array
        
        logger.info(f"🚀 MAXIMUM GPU UTILIZATION: Reranking {len(chunks):,} chunks across {self.num_gpus} GPUs")
        
        # Create a standalone function to avoid closure issues in threading
        def process_reranking_batch_standalone(args):
            gpu_id, gpu_chunks, start_idx, query, gpu_locks, gpu_rerankers, get_optimal_batch_size_func, gpu_ids = args
            """Process reranking on a specific GPU with MAXIMUM memory utilization."""
            try:
                import torch
                with gpu_locks[gpu_id]:
                    # Set device context if GPU, skip for CPU reranker (-1)
                    if gpu_id >= 0:
                        # After CUDA_VISIBLE_DEVICES is set, we need to use the remapped GPU ID
                        remapped_gpu_id = gpu_ids.index(gpu_id) if gpu_id in gpu_ids else gpu_id
                        torch.cuda.set_device(remapped_gpu_id)
                    reranker = gpu_rerankers[gpu_id]
                    
                    # Calculate optimal batch size for this GPU (reranking can use larger batches)
                    optimal_batch_size = get_optimal_batch_size_func(gpu_id)
                    # For reranking, we can use 2x the embedding batch size since it's more memory efficient
                    rerank_batch_size = min(optimal_batch_size * 2, 20000)  # Use 2x the embedding batch size, up to 20k
                    
                    logger.info(f"🚀 GPU {gpu_id}: Reranking {len(gpu_chunks):,} chunks (indices {start_idx:,}-{start_idx + len(gpu_chunks) - 1:,}) with batch size {rerank_batch_size:,}")
                    
                    # Process in optimized batches for maximum memory utilization
                    all_scores = []
                    
                    for batch_start in range(0, len(gpu_chunks), rerank_batch_size):
                        batch_end = min(batch_start + rerank_batch_size, len(gpu_chunks))
                        batch_chunks = gpu_chunks[batch_start:batch_end]
                        
                        # Prepare query-chunk pairs for this batch
                        query_chunk_pairs = [[query, chunk['chunk_text']] for chunk in batch_chunks]
                        
                        # Get scores for this batch
                        batch_scores = reranker.compute_score(query_chunk_pairs, normalize=True)
                        # Handle both list and numpy array returns
                        if isinstance(batch_scores, list):
                            all_scores.extend(batch_scores)
                        else:
                            all_scores.extend(batch_scores.tolist())
                        
                        # Log progress for large chunk sets
                        if len(gpu_chunks) > rerank_batch_size:
                            logger.info(f"🚀 GPU {gpu_id}: Completed batch {batch_start//rerank_batch_size + 1}/{(len(gpu_chunks) + rerank_batch_size - 1)//rerank_batch_size}")
                    
                    logger.info(f"✅ GPU {gpu_id}: Completed reranking {len(gpu_chunks):,} chunks with batch size {rerank_batch_size:,}")
                    return start_idx, all_scores
                    
            except Exception as e:
                logger.error(f"Error on GPU {gpu_id}: {e}")
                return start_idx, [0.0] * len(gpu_chunks)
        
        # Process all GPUs in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            
            for i, gpu_chunks in enumerate(distributed_chunks):
                if gpu_chunks:
                    actual_gpu_id = self.gpu_ids[i]  # Get the actual GPU ID from the list
                    start_idx = sum(len(distributed_chunks[j]) for j in range(i))
                    # Prepare arguments for the standalone function
                    args = (actual_gpu_id, gpu_chunks, start_idx, query, self.gpu_locks, self.gpu_rerankers, self.get_optimal_batch_size, self.gpu_ids)
                    future = executor.submit(process_reranking_batch_standalone, args)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    start_idx, gpu_scores = future.result()
                    # Place results in correct positions
                    for i, score in enumerate(gpu_scores):
                        all_scores[start_idx + i] = score
                except Exception as e:
                    logger.error(f"Error collecting reranking results: {e}")
        
        # Map scores back to chunk IDs
        chunk_scores = {}
        for i, score in enumerate(all_scores):
            if score is not None:
                chunk_scores[chunks[i]['chunk_id']] = float(score)
            else:
                chunk_scores[chunks[i]['chunk_id']] = 0.0
        
        # MEMORY CLEANUP after reranking to prevent GPU memory accumulation
        logger.info("🧹 Cleaning up GPU memory after reranking")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("✅ Memory cleanup completed after reranking")
        except Exception as e:
            logger.warning(f"Memory cleanup after reranking failed: {e}")
        
        return chunk_scores
    
    def process_reranking_parallel_batch(self, all_pairs: List[Tuple[str, str]]) -> List[float]:
        """Process all claim-chunk pairs in parallel across all GPUs."""
        if not self.gpu_rerankers or self.num_gpus <= 1:
            if self.gpu_rerankers:
                gpu_id = list(self.gpu_rerankers.keys())[0]
                if gpu_id >= 0:
                    torch.cuda.set_device(gpu_id)
                reranker = self.gpu_rerankers[gpu_id]
                
                batch_size = 1000
                all_scores = []
                for i in range(0, len(all_pairs), batch_size):
                    batch_pairs = all_pairs[i:i + batch_size]
                    batch_scores = reranker.compute_score(batch_pairs, normalize=True)
                    # Handle both list and numpy array returns
                    if isinstance(batch_scores, list):
                        all_scores.extend(batch_scores)
                    else:
                        all_scores.extend(batch_scores.tolist())
                return all_scores
            else:
                return [0.0] * len(all_pairs)
        
        distributed_pairs = self._distribute_data_evenly(all_pairs, self.num_gpus)
        all_scores = [None] * len(all_pairs)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            futures = []
            
            for i, gpu_pairs in enumerate(distributed_pairs):
                if gpu_pairs:
                    actual_gpu_id = self.gpu_ids[i]  # Get the actual GPU ID from the list
                    start_idx = sum(len(distributed_pairs[j]) for j in range(i))
                    future = executor.submit(
                        self._process_reranking_batch_on_gpu,
                        actual_gpu_id, gpu_pairs, start_idx, all_scores
                    )
                    futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in reranking batch: {e}")
        
        final_scores = [score if score is not None else 0.0 for score in all_scores]
        return final_scores
    
    def _process_reranking_batch_on_gpu(self, gpu_id: int, pairs: List[Tuple[str, str]], start_idx: int, all_scores: List):
        """Process a batch of reranking pairs on a specific GPU."""
        try:
            with self.gpu_locks[gpu_id]:
                # Set device context if GPU; CPU reranker (-1) needs no cuda set
                if gpu_id >= 0:
                    remapped_gpu_id = self.gpu_ids.index(gpu_id)
                    torch.cuda.set_device(remapped_gpu_id)
                reranker = self.gpu_rerankers[gpu_id]
                
                optimal_batch_size = self.get_optimal_batch_size(gpu_id)
                rerank_batch_size = min(optimal_batch_size * 2, 20000)
                
                for batch_start in range(0, len(pairs), rerank_batch_size):
                    batch_end = min(batch_start + rerank_batch_size, len(pairs))
                    batch_pairs = pairs[batch_start:batch_end]
                    
                    batch_scores = reranker.compute_score(batch_pairs, normalize=True)
                    
                    # Handle both list and numpy array returns
                    scores_list = batch_scores if isinstance(batch_scores, list) else batch_scores.tolist()
                    for i, score in enumerate(scores_list):
                        all_scores[start_idx + batch_start + i] = score
                
        except Exception as e:
            logger.error(f"Error on GPU {gpu_id}: {e}")
            for i in range(len(pairs)):
                all_scores[start_idx + i] = None

class SimilarityFilter:
    """Class to handle similarity filtering between claims and web content chunks."""
    
    def __init__(self, similarity_threshold: float = 0.4, num_gpus: int = 4, gpu_ids: List[int] = None):
        """
        Initialize the similarity filter with GPU acceleration.
        
        Args:
            similarity_threshold: Minimum similarity score to consider a chunk relevant
            num_gpus: Number of GPUs to use for both similarity filtering and reranking (default: 4)
            gpu_ids: List of specific GPU IDs to use (e.g., [0, 1, 3]). If None, uses first num_gpus.
        """
        self.similarity_threshold = similarity_threshold
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids
        self.context_locator = OptimizedContextLocator()
        
        # Initialize multi-GPU manager with specific GPU IDs
        self.gpu_manager = MultiGPUManager(num_gpus, gpu_ids)
        
        # Show GPU information
        self._show_gpu_info()
    
    def _show_gpu_info(self):
        """Display GPU information and utilization."""
        if not torch or not torch.cuda.is_available():
            logger.info("No GPU available for processing")
            return
        
        logger.info(f"GPU Information:")
        logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"  Active GPUs: {self.num_gpus}")
        
        for gpu_id in range(min(self.num_gpus, torch.cuda.device_count())):
            try:
                torch.cuda.set_device(gpu_id)
                props = torch.cuda.get_device_properties(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = props.total_memory
                available = total - allocated
                
                logger.info(f"  GPU {gpu_id}: {props.name}")
                logger.info(f"    Memory: {allocated/1024**3:.2f}GB used, {available/1024**3:.2f}GB available, {total/1024**3:.2f}GB total")
            except Exception as e:
                logger.warning(f"  Could not get info for GPU {gpu_id}: {e}")
    
    def load_web_content(self, web_content_file: str) -> Dict[str, str]:
        """
        Load web content from the cache file.
        
        Args:
            web_content_file: Path to the web content cache JSON file
            
        Returns:
            Dictionary mapping URLs to their content
        """
        try:
            with open(web_content_file, 'r', encoding='utf-8') as f:
                web_content = json.load(f)
            logger.info(f"Loaded web content for {len(web_content)} URLs")
            return web_content
        except Exception as e:
            logger.error(f"Error loading web content: {e}")
            raise
    
    def extract_chunks_from_content(self, content: str, url: str) -> List[Dict[str, Any]]:
        """
        Extract chunks from web content using OptimizedContextLocator.
        
        Args:
            content: Raw web content text
            url: Source URL for the content
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            chunks = self.context_locator.extract_sentences(content)
            
            # Add URL information to each chunk
            for chunk in chunks:
                chunk['source_url'] = url
                
            logger.info(f"Extracted {len(chunks)} chunks from {url}")
            return chunks
        except Exception as e:
            logger.error(f"Error extracting chunks from {url}: {e}")
            return []
    
    def compute_similarity(self, claim_embedding: List[float], chunk_embedding: List[float]) -> float:
        """
        Compute cosine similarity between claim and chunk embeddings.
        
        Args:
            claim_embedding: Embedding of the claim
            chunk_embedding: Embedding of the chunk
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Reshape for sklearn cosine_similarity
            claim_reshaped = np.array(claim_embedding).reshape(1, -1)
            chunk_reshaped = np.array(chunk_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(claim_reshaped, chunk_reshaped)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def filter_chunks_by_similarity(self, claims: List[str], urls: List[str], web_content: Dict[str, str], apply_reranking: bool = True, top_k: int = 10) -> Dict[str, Any]:
        """
        Main method to filter chunks by similarity to claims.
        
        Args:
            claims: List of claim texts
            urls: List of URLs to process
            web_content_file: Path to web content cache file
            apply_reranking: Whether to apply reranking and select top chunks
            top_k: Number of top chunks to select after reranking
            
        Returns:
            Dictionary organized by claims with their relevant chunks
        """
        logger.info(f"Starting similarity filtering for {len(claims)} claims and {len(urls)} URLs")

        # Filter content to only include requested URLs
        filtered_content = {url: web_content.get(url, "") for url in urls if url in web_content}
        logger.info(f"Found content for {len(filtered_content)} out of {len(urls)} requested URLs")
        
        # Extract chunks from all content
        all_chunks = []
        for url, content in filtered_content.items():
            if content.strip():
                chunks = self.extract_chunks_from_content(content, url)
                all_chunks.extend(chunks)
        
        logger.info(f"Total chunks extracted: {len(all_chunks)}")
        
        # Get embeddings for all chunks using multi-GPU processing
        chunk_texts = [chunk['chunk_text'] for chunk in all_chunks]
        try:
            chunk_embeddings_array = self.gpu_manager.process_embeddings_parallel(chunk_texts, show_progress=False)
            chunk_embeddings = {chunk['chunk_id']: embedding 
                               for chunk, embedding in zip(all_chunks, chunk_embeddings_array)}
            
            logger.info(f"Successfully generated embeddings for {len(chunk_embeddings)} chunks using {self.num_gpus} GPUs")
        except Exception as e:
            logger.error(f"Failed to generate chunk embeddings: {e}")
            return {}
        
        # Process claims using multi-GPU processing
        try:
            claim_embeddings_array = self.gpu_manager.process_embeddings_parallel(claims, show_progress=False)
            logger.info(f"Successfully generated embeddings for {len(claims)} claims using {self.num_gpus} GPUs")
        except Exception as e:
            logger.error(f"Failed to generate claim embeddings: {e}")
            return {}
        
        results = {}
        
        for i, (claim, claim_embedding) in enumerate(zip(claims, claim_embeddings_array)):
            logger.info(f"Processing claim {i+1}/{len(claims)}: {claim[:100]}...")
            
            try:
                # Find relevant chunks
                relevant_chunks = []
                
                for chunk in all_chunks:
                    chunk_id = chunk['chunk_id']
                    if chunk_id in chunk_embeddings:
                        similarity = self.compute_similarity(claim_embedding, chunk_embeddings[chunk_id])
                        
                        if similarity >= self.similarity_threshold:
                            relevant_chunk = {
                                'chunk_id': chunk['chunk_id'],
                                'chunk_text': chunk['chunk_text'],
                                'source_url': chunk['source_url'],
                                'similarity_score': similarity,
                                'position': chunk['position'],
                                'length': chunk['length'],
                                'sentence_count': chunk['sentence_count']
                            }
                            relevant_chunks.append(relevant_chunk)
                
                # Sort by similarity score (highest first)
                relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                results[claim] = {
                    'relevant_chunks': relevant_chunks,
                    'total_chunks_found': len(relevant_chunks),
                    'similarity_threshold': self.similarity_threshold
                }
                
                logger.info(f"Found {len(relevant_chunks)} relevant chunks for claim {i+1}")
                
            except Exception as e:
                logger.error(f"Error processing claim {i+1}: {e}")
                results[claim] = {
                    'relevant_chunks': [],
                    'total_chunks_found': 0,
                    'similarity_threshold': self.similarity_threshold,
                    'error': str(e)
                }
        
        # Apply reranking if requested
        if apply_reranking and self.gpu_manager.gpu_rerankers:
            logger.info("Applying reranking to select top chunks...")
            results = self.apply_reranking_and_select_top_chunks(results, top_k)
        elif apply_reranking and not self.gpu_manager.gpu_rerankers:
            logger.warning("Reranking requested but reranker not available. Using similarity scores only.")
            # Select top chunks based on similarity scores
            for claim, claim_data in results.items():
                if 'relevant_chunks' in claim_data and claim_data['relevant_chunks']:
                    relevant_chunks = claim_data['relevant_chunks']
                    relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
                    claim_data['top_chunks'] = relevant_chunks[:top_k]
                    claim_data['reranking_applied'] = False
                    claim_data['top_k'] = top_k
        
        return results
    
    def filter_chunks_by_similarity_with_mapping(self, claim_chunk_mapping: Dict[str, List[Dict[str, Any]]], apply_reranking: bool = True, top_k: int = 10) -> Dict[str, Any]:
        """
        Process similarity filtering with proper claim-chunk mapping.
        Each claim only processes its own chunks (no cross-claim combinations).
        OPTIMIZED: Pre-compute all embeddings once, then compute similarities based on mapping.
        
        Args:
            claim_chunk_mapping: Dictionary mapping each claim to its own chunks
            apply_reranking: Whether to apply reranking and select top chunks
            top_k: Number of top chunks to select after reranking
            
        Returns:
            Dictionary organized by claims with their relevant chunks
        """
        claims = list(claim_chunk_mapping.keys())
        logger.info(f"🚀 Starting OPTIMIZED TWO-STAGE processing: {len(claims)} claims with proper claim-chunk mapping")
        logger.info("🎯 Each claim will ONLY process its own chunks (no cross-claim combinations)")
        logger.info("⚡ OPTIMIZATION: Pre-compute all embeddings once, then compute similarities based on mapping")
        
        # STAGE 1: Similarity Filtering for ALL claims
        logger.info("=" * 80)
        logger.info("STAGE 1: SIMILARITY FILTERING - Pre-compute embeddings then compute similarities")
        logger.info("=" * 80)
        
        similarity_results = self._stage1_similarity_filtering_optimized(claim_chunk_mapping)
        
        if not similarity_results:
            logger.error("Stage 1 failed, returning empty results")
            return {}
        
        # STAGE 2: Reranking for ALL claims (if requested)
        if apply_reranking and self.gpu_manager.gpu_rerankers:
            logger.info("=" * 80)
            logger.info("STAGE 2: RERANKING - Processing ALL claims simultaneously")
            logger.info("=" * 80)
            
            final_results = self._stage2_reranking_proper_mapping(similarity_results, top_k)
        else:
            logger.info("Skipping reranking, using similarity scores only")
            final_results = self._select_top_chunks_by_similarity(similarity_results, top_k)
        
        # MEMORY CLEANUP
        self._cleanup_gpu_memory()
        
        return final_results

    def filter_chunks_by_similarity_direct(self, claims: List[str], chunks: List[Dict[str, Any]], apply_reranking: bool = True, top_k: int = 10) -> Dict[str, Any]:
        """
        Two-stage processing: First similarity filtering for all claims, then reranking for all claims.
        Each claim only processes its own chunks (no cross-claim processing).
        
        Args:
            claims: List of claim texts
            chunks: List of chunk dictionaries (already extracted and prepared)
            apply_reranking: Whether to apply reranking and select top chunks
            top_k: Number of top chunks to select after reranking
            
        Returns:
            Dictionary organized by claims with their relevant chunks
        """
        logger.info(f"🚀 Starting TWO-STAGE processing: {len(claims)} claims and {len(chunks)} chunks")
        logger.info("🎯 Each claim will ONLY process its own chunks (no cross-claim combinations)")
        
        # STAGE 1: Similarity Filtering for ALL claims
        logger.info("=" * 80)
        logger.info("STAGE 1: SIMILARITY FILTERING - Processing ALL claims simultaneously")
        logger.info("=" * 80)
        
        similarity_results = self._stage1_similarity_filtering_proper_mapping(claims, chunks)
        
        if not similarity_results:
            logger.error("Stage 1 failed, returning empty results")
            return {}
        
        # STAGE 2: Reranking for ALL claims (if requested)
        if apply_reranking and self.gpu_manager.gpu_rerankers:
            logger.info("=" * 80)
            logger.info("STAGE 2: RERANKING - Processing ALL claims simultaneously")
            logger.info("=" * 80)
            
            final_results = self._stage2_reranking_proper_mapping(similarity_results, top_k)
        else:
            logger.info("Skipping reranking, using similarity scores only")
            final_results = self._select_top_chunks_by_similarity(similarity_results, top_k)
        
        # MEMORY CLEANUP
        self._cleanup_gpu_memory()
        
        return final_results
    
    def _stage1_similarity_filtering_with_mapping(self, claim_chunk_mapping: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Stage 1: Process similarity filtering with proper claim-chunk mapping."""
        from tqdm import tqdm
        
        claims = list(claim_chunk_mapping.keys())
        
        # Report detailed chunk counts per claim and the overall total
        total_chunks_to_process = 0
        logger.info("📦 Chunk counts per claim (pre-embedding):")
        for claim in claims:
            num_chunks = len(claim_chunk_mapping.get(claim, []))
            total_chunks_to_process += num_chunks
            logger.info(f"  - {num_chunks:5d} chunks | {claim[:100]}...")
        logger.info(f"🧮 Total chunks to process across all claims: {total_chunks_to_process}")
        
        # Progress 1: Generate claim embeddings (no progress bar for internal batch processing)
        logger.info(f"🔄 Generating embeddings for {len(claims)} claims...")
        claim_embeddings_array = self.gpu_manager.process_embeddings_parallel(claims)
        claim_embeddings = {claim: embedding for claim, embedding in zip(claims, claim_embeddings_array)}
        logger.info(f"✅ Claim embeddings generated successfully")
        
        # Progress 2: Process each claim with its own chunks - show claim-level progress
        results = {}
        logger.info(f"🔄 Processing {len(claims)} claims for similarity filtering...")
        with tqdm(total=len(claims), desc="Processing Claims", position=1, unit="claim", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} claims [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for claim in claims:
                claim_chunks = claim_chunk_mapping[claim]
                if not claim_chunks:
                    results[claim] = {
                        'relevant_chunks': [],
                        'total_chunks_found': 0,
                        'similarity_threshold': self.similarity_threshold
                    }
                    pbar.update(1)
                    continue
                
                # Generate embeddings for this claim's chunks
                chunk_texts = [chunk['chunk_text'] for chunk in claim_chunks]
                chunk_embeddings_array = self.gpu_manager.process_embeddings_parallel(chunk_texts, show_progress=False)
                chunk_embeddings = {chunk['chunk_id']: embedding 
                                   for chunk, embedding in zip(claim_chunks, chunk_embeddings_array)}
                
                # Find relevant chunks for this claim (vectorized cosine similarity)
                claim_embedding = np.asarray(claim_embeddings[claim], dtype=np.float32)
                # Build matrix of chunk embeddings aligned to claim_chunks order
                chunk_ids = []
                chunk_matrix_list = []
                chunk_meta_list = []
                for chunk in claim_chunks:
                    chunk_id = chunk['chunk_id']
                    if chunk_id in chunk_embeddings:
                        chunk_ids.append(chunk_id)
                        chunk_matrix_list.append(chunk_embeddings[chunk_id])
                        chunk_meta_list.append(chunk)

                if chunk_matrix_list:
                    chunk_matrix = np.asarray(chunk_matrix_list, dtype=np.float32)  # shape: (N, D)
                    # embeddings are normalized at encode-time; cosine = dot product
                    similarities = chunk_matrix @ claim_embedding  # shape: (N,)
                    # Filter by threshold and sort descending
                    above_threshold_idx = np.where(similarities >= self.similarity_threshold)[0]
                    if above_threshold_idx.size > 0:
                        # Sort selected indices by similarity desc
                        sorted_idx = above_threshold_idx[np.argsort(similarities[above_threshold_idx])[::-1]]
                        relevant_chunks = []
                        for idx in sorted_idx.tolist():
                            meta = chunk_meta_list[idx]
                            relevant_chunks.append({
                                'chunk_id': meta['chunk_id'],
                                'chunk_text': meta['chunk_text'],
                                'source_url': meta['source_url'],
                                'similarity_score': float(similarities[idx]),
                                'position': meta.get('position', 0),
                                'length': meta.get('length', len(meta['chunk_text'])),
                                'sentence_count': meta.get('sentence_count', 0)
                            })
                    else:
                        relevant_chunks = []
                else:
                    relevant_chunks = []
                
                results[claim] = {
                    'relevant_chunks': relevant_chunks,
                    'total_chunks_found': len(relevant_chunks),
                    'similarity_threshold': self.similarity_threshold
                }
                
                pbar.update(1)
        
        # Progress 3: Finalize similarity results
        logger.info(f"✅ Similarity filtering completed for {len(results)} claims")
        
        return results
    
    def _stage1_similarity_filtering_optimized(self, claim_chunk_mapping: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        OPTIMIZED Stage 1: Pre-compute all embeddings once, then compute similarities based on mapping.
        This avoids redundant embedding computations and maximizes efficiency.
        """
        from tqdm import tqdm
        
        claims = list(claim_chunk_mapping.keys())
        
        # Collect all unique chunks across all claims
        all_chunks = []
        chunk_id_to_chunk = {}
        for claim_chunks in claim_chunk_mapping.values():
            for chunk in claim_chunks:
                chunk_id = chunk['chunk_id']
                if chunk_id not in chunk_id_to_chunk:
                    chunk_id_to_chunk[chunk_id] = chunk
                    all_chunks.append(chunk)
        
        logger.info(f"📦 Total unique chunks across all claims: {len(all_chunks)}")
        logger.info(f"📦 Total claims to process: {len(claims)}")
        
        # Step 1: Pre-compute ALL claim embeddings in parallel
        logger.info(f"🔄 Step 1: Pre-computing embeddings for {len(claims)} claims...")
        with tqdm(total=1, desc="Pre-computing Claim Embeddings", position=0, unit="batch") as pbar:
            claim_embeddings_array = self.gpu_manager.process_embeddings_parallel(claims, show_progress=False)
            claim_embeddings = {claim: embedding for claim, embedding in zip(claims, claim_embeddings_array)}
            pbar.update(1)
        
        # Step 2: Pre-compute ALL chunk embeddings in parallel
        logger.info(f"🔄 Step 2: Pre-computing embeddings for {len(all_chunks)} unique chunks...")
        chunk_texts = [chunk['chunk_text'] for chunk in all_chunks]
        with tqdm(total=1, desc="Pre-computing Chunk Embeddings", position=1, unit="batch") as pbar:
            chunk_embeddings_array = self.gpu_manager.process_embeddings_parallel(chunk_texts, show_progress=False)
            chunk_embeddings = {chunk['chunk_id']: embedding for chunk, embedding in zip(all_chunks, chunk_embeddings_array)}
            pbar.update(1)
        
        logger.info(f"✅ All embeddings pre-computed successfully!")
        
        # Step 3: Compute similarities for each claim with its own chunks
        results = {}
        logger.info(f"🔄 Step 3: Computing similarities for {len(claims)} claims...")
        with tqdm(total=len(claims), desc="Computing Similarities", position=2, unit="claim",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} claims [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for claim in claims:
                claim_chunks = claim_chunk_mapping[claim]
                if not claim_chunks:
                    results[claim] = {
                        'relevant_chunks': [],
                        'total_chunks_found': 0,
                        'similarity_threshold': self.similarity_threshold
                    }
                    pbar.update(1)
                    continue
                
                # Get claim embedding
                claim_embedding = np.asarray(claim_embeddings[claim], dtype=np.float32)
                
                # Build matrix of chunk embeddings for this claim's chunks
                chunk_matrix_list = []
                chunk_meta_list = []
                for chunk in claim_chunks:
                    chunk_id = chunk['chunk_id']
                    if chunk_id in chunk_embeddings:
                        chunk_matrix_list.append(chunk_embeddings[chunk_id])
                        chunk_meta_list.append(chunk)
                
                if chunk_matrix_list:
                    # Vectorized similarity computation
                    chunk_matrix = np.asarray(chunk_matrix_list, dtype=np.float32)  # shape: (N, D)
                    # embeddings are normalized at encode-time; cosine = dot product
                    similarities = chunk_matrix @ claim_embedding  # shape: (N,)
                    
                    # Filter by threshold and sort descending
                    above_threshold_idx = np.where(similarities >= self.similarity_threshold)[0]
                    if above_threshold_idx.size > 0:
                        # Sort selected indices by similarity desc
                        sorted_idx = above_threshold_idx[np.argsort(similarities[above_threshold_idx])[::-1]]
                        relevant_chunks = []
                        for idx in sorted_idx.tolist():
                            meta = chunk_meta_list[idx]
                            relevant_chunks.append({
                                'chunk_id': meta['chunk_id'],
                                'chunk_text': meta['chunk_text'],
                                'source_url': meta['source_url'],
                                'similarity_score': float(similarities[idx]),
                                'position': meta.get('position', 0),
                                'length': meta.get('length', len(meta['chunk_text'])),
                                'sentence_count': meta.get('sentence_count', 0)
                            })
                    else:
                        relevant_chunks = []
                else:
                    relevant_chunks = []
                
                results[claim] = {
                    'relevant_chunks': relevant_chunks,
                    'total_chunks_found': len(relevant_chunks),
                    'similarity_threshold': self.similarity_threshold
                }
                
                pbar.update(1)
        
        logger.info(f"✅ Optimized similarity filtering completed for {len(results)} claims")
        return results
    
    def _stage1_similarity_filtering_proper_mapping(self, claims: List[str], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 1: Process similarity filtering for ALL claims simultaneously with proper claim-chunk mapping."""
        from tqdm import tqdm
        
        # First, we need to understand the claim-chunk mapping from the calling code
        # Since we don't have that information here, we'll process each claim against all chunks
        # but this should be called with the proper mapping from the calling code
        
        # Progress 1: Generate chunk embeddings
        chunk_texts = [chunk['chunk_text'] for chunk in chunks]
        with tqdm(total=1, desc="Similarity Filtering - Chunk Embeddings", position=0) as pbar:
            chunk_embeddings_array = self.gpu_manager.process_embeddings_parallel(chunk_texts, show_progress=False)
            chunk_embeddings = {chunk['chunk_id']: embedding 
                               for chunk, embedding in zip(chunks, chunk_embeddings_array)}
            pbar.update(1)
        
        # Progress 2: Generate claim embeddings
        with tqdm(total=1, desc="Similarity Filtering - Claim Embeddings", position=1) as pbar:
            claim_embeddings_array = self.gpu_manager.process_embeddings_parallel(claims, show_progress=False)
            pbar.update(1)
        
        # Progress 3: Process similarity filtering - each claim processes its own chunks
        import multiprocessing as mp
        num_cores = min(256, mp.cpu_count())
        
        similarity_args = []
        for i, (claim, claim_embedding) in enumerate(zip(claims, claim_embeddings_array)):
            similarity_args.append((claim, claim_embedding, chunks, chunk_embeddings, i, self.similarity_threshold))
        
        results = {}
        with tqdm(total=len(claims), desc="Similarity Filtering - Processing Claims", position=2) as pbar:
            try:
                with mp.Pool(processes=num_cores) as pool:
                    parallel_results = pool.map(self._process_single_claim_similarity_standalone, similarity_args)
                    
                    for claim, claim_result in zip(claims, parallel_results):
                        results[claim] = claim_result
                        pbar.update(1)
                        
            except Exception as e:
                logger.error(f"Error in parallel similarity processing: {e}")
                for args in similarity_args:
                    result = self._process_single_claim_similarity_standalone(args)
                    results[args[0]] = result
                    pbar.update(1)
        
        # Progress 4: Finalize similarity results
        with tqdm(total=1, desc="Similarity Filtering - Finalizing", position=3) as pbar:
            pbar.update(1)
        
        return results
    
    def _stage2_reranking_proper_mapping(self, similarity_results: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Stage 2: Process reranking for ALL claims simultaneously with proper claim-chunk mapping."""
        from tqdm import tqdm
        
        # Progress 1: Prepare reranking pairs for each claim
        all_rerank_pairs = []
        claim_to_pairs_mapping = {}
        
        logger.info(f"🔄 Preparing reranking pairs for {len(similarity_results)} claims...")
        for claim, claim_data in similarity_results.items():
            if 'relevant_chunks' in claim_data and claim_data['relevant_chunks']:
                relevant_chunks = claim_data['relevant_chunks']
                claim_pairs = []
                
                for chunk in relevant_chunks:
                    pair = (claim, chunk['chunk_text'])
                    all_rerank_pairs.append(pair)
                    claim_pairs.append(len(all_rerank_pairs) - 1)
                
                claim_to_pairs_mapping[claim] = claim_pairs
        
        if not all_rerank_pairs:
            logger.info("No reranking pairs found, using similarity scores only")
            return self._select_top_chunks_by_similarity(similarity_results, top_k)
        
        # Progress 2: Process reranking (no progress bar for internal batch processing)
        logger.info(f"🔄 Processing reranking for {len(all_rerank_pairs)} pairs across {self.gpu_manager.num_gpus} GPUs...")
        try:
            rerank_scores = self.gpu_manager.process_reranking_parallel_batch(all_rerank_pairs)
            logger.info(f"✅ Reranking completed successfully")
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._select_top_chunks_by_similarity(similarity_results, top_k)
        
        # Progress 3: Apply scores and select top chunks
        final_results = {}
        logger.info(f"🔄 Applying reranking scores to {len(similarity_results)} claims...")
        with tqdm(total=len(similarity_results), desc="Applying Reranking Scores", position=5, unit="claim",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} claims [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for claim, claim_data in similarity_results.items():
                if claim in claim_to_pairs_mapping:
                    pair_indices = claim_to_pairs_mapping[claim]
                    relevant_chunks = claim_data['relevant_chunks']
                    
                    for i, chunk in enumerate(relevant_chunks):
                        if i < len(pair_indices):
                            pair_idx = pair_indices[i]
                            if pair_idx < len(rerank_scores):
                                chunk['rerank_score'] = rerank_scores[pair_idx]
                            else:
                                chunk['rerank_score'] = 0.0
                        else:
                            chunk['rerank_score'] = 0.0
                    
                    relevant_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
                    top_chunks = relevant_chunks[:top_k]
                    
                    final_results[claim] = {
                        'top_chunks': top_chunks,
                        'reranking_applied': True,
                        'top_k': top_k,
                        'total_chunks_processed': len(relevant_chunks)
                    }
                else:
                    final_results[claim] = {
                        'top_chunks': [],
                        'reranking_applied': False,
                        'top_k': top_k,
                        'total_chunks_processed': 0
                    }
                pbar.update(1)
        
        # Progress 4: Finalize reranking results
        with tqdm(total=1, desc="Reranking - Finalizing", position=6) as pbar:
            pbar.update(1)
        
        return final_results
    
    def _stage2_reranking(self, similarity_results: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Stage 2: Process reranking for ALL claims simultaneously."""
        from tqdm import tqdm
        
        # Progress 1: Prepare reranking pairs
        all_rerank_pairs = []
        claim_to_pairs_mapping = {}
        
        with tqdm(total=1, desc="Reranking - Preparing Pairs", position=4) as pbar:
            for claim, claim_data in similarity_results.items():
                if 'relevant_chunks' in claim_data and claim_data['relevant_chunks']:
                    relevant_chunks = claim_data['relevant_chunks']
                    claim_pairs = []
                    
                    for chunk in relevant_chunks:
                        pair = (claim, chunk['chunk_text'])
                        all_rerank_pairs.append(pair)
                        claim_pairs.append(len(all_rerank_pairs) - 1)
                    
                    claim_to_pairs_mapping[claim] = claim_pairs
            pbar.update(1)
        
        if not all_rerank_pairs:
            return self._select_top_chunks_by_similarity(similarity_results, top_k)
        
        # Progress 2: Process reranking
        with tqdm(total=1, desc="Reranking - Processing Pairs", position=5) as pbar:
            try:
                rerank_scores = self.gpu_manager.process_reranking_parallel_batch(all_rerank_pairs)
                pbar.update(1)
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                return self._select_top_chunks_by_similarity(similarity_results, top_k)
        
        # Progress 3: Apply scores and select top chunks
        final_results = {}
        with tqdm(total=len(similarity_results), desc="Reranking - Applying Scores", position=6) as pbar:
            for claim, claim_data in similarity_results.items():
                if claim in claim_to_pairs_mapping:
                    pair_indices = claim_to_pairs_mapping[claim]
                    relevant_chunks = claim_data['relevant_chunks']
                    
                    for i, chunk in enumerate(relevant_chunks):
                        if i < len(pair_indices):
                            pair_idx = pair_indices[i]
                            if pair_idx < len(rerank_scores):
                                chunk['rerank_score'] = rerank_scores[pair_idx]
                            else:
                                chunk['rerank_score'] = 0.0
                        else:
                            chunk['rerank_score'] = 0.0
                    
                    relevant_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
                    top_chunks = relevant_chunks[:top_k]
                    
                    final_results[claim] = {
                        'top_chunks': top_chunks,
                        'reranking_applied': True,
                        'top_k': top_k,
                        'total_chunks_processed': len(relevant_chunks)
                    }
                else:
                    final_results[claim] = {
                        'top_chunks': [],
                        'reranking_applied': False,
                        'top_k': top_k,
                        'total_chunks_processed': 0
                    }
                pbar.update(1)
        
        # Progress 4: Finalize reranking results
        with tqdm(total=1, desc="Reranking - Finalizing", position=7) as pbar:
            pbar.update(1)
        
        return final_results
    
    def _select_top_chunks_by_similarity(self, similarity_results: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Select top chunks based on similarity scores only."""
        final_results = {}
        for claim, claim_data in similarity_results.items():
            if 'relevant_chunks' in claim_data and claim_data['relevant_chunks']:
                relevant_chunks = claim_data['relevant_chunks']
                relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
                top_chunks = relevant_chunks[:top_k]
                
                final_results[claim] = {
                    'top_chunks': top_chunks,
                    'reranking_applied': False,
                    'top_k': top_k,
                    'total_chunks_processed': len(relevant_chunks)
                }
            else:
                final_results[claim] = {
                    'top_chunks': [],
                    'reranking_applied': False,
                    'top_k': top_k,
                    'total_chunks_processed': 0
                }
        return final_results
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after processing."""
        logger.info("🧹 Cleaning up GPU memory")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("✅ Memory cleanup completed")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")


def _process_single_claim_similarity_standalone(args):
    """
    Standalone function for processing similarity filtering of a single claim.
    This function is designed to be called from multiprocessing.Pool.
    
    Args:
        args: Tuple of (claim, claim_embedding, chunks, chunk_embeddings, claim_index, similarity_threshold)
        
    Returns:
        Dictionary containing similarity filtering results for this claim
    """
    try:
        claim, claim_embedding, chunks, chunk_embeddings, claim_index, similarity_threshold = args
        
        # Find relevant chunks using vectorized operations
        relevant_chunks = []
        
        # Process chunks in batches for efficiency
        chunk_batch_size = 1000
        for batch_start in range(0, len(chunks), chunk_batch_size):
            batch_end = min(batch_start + chunk_batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            # Extract embeddings for this batch
            batch_embeddings = []
            valid_chunks = []
            
            for chunk in batch_chunks:
                chunk_id = chunk['chunk_id']
                if chunk_id in chunk_embeddings:
                    batch_embeddings.append(chunk_embeddings[chunk_id])
                    valid_chunks.append(chunk)
            
            if batch_embeddings:
                # Convert to numpy arrays for vectorized computation
                batch_embeddings_array = np.array(batch_embeddings)
                
                # Vectorized similarity computation
                claim_embedding_reshaped = claim_embedding.reshape(1, -1)
                similarities = np.dot(batch_embeddings_array, claim_embedding_reshaped.T).flatten()
                
                # Find chunks above threshold
                above_threshold_mask = similarities >= similarity_threshold
                above_threshold_indices = np.where(above_threshold_mask)[0]
                
                # Create relevant chunks for those above threshold
                for idx in above_threshold_indices:
                    chunk = valid_chunks[idx]
                    similarity = float(similarities[idx])
                    
                    relevant_chunk = {
                        'chunk_id': chunk['chunk_id'],
                        'chunk_text': chunk['chunk_text'],
                        'source_url': chunk['source_url'],
                        'similarity_score': similarity,
                        'position': chunk.get('position', 0),
                        'length': chunk.get('length', len(chunk['chunk_text'])),
                        'sentence_count': chunk.get('sentence_count', 0)
                    }
                    relevant_chunks.append(relevant_chunk)
        
        # Sort by similarity score (highest first)
        relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return {
            'relevant_chunks': relevant_chunks,
            'total_chunks_found': len(relevant_chunks),
            'similarity_threshold': similarity_threshold
        }
        
    except Exception as e:
        logger.error(f"Error processing claim {claim_index + 1}: {e}")
        return {
            'relevant_chunks': [],
            'total_chunks_found': 0,
            'similarity_threshold': similarity_threshold,
            'error': str(e)
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save results to a JSON file.
        
        Args:
            results: Results dictionary to save
            output_file: Path to output JSON file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

