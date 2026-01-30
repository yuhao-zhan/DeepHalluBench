#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import os
import json
import math
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
import torch
import warnings

from sentence_transformers import SentenceTransformer
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp



warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GPU locks to prevent multiple processes from using the same GPU
GPU_LOCKS = {}
GPU_LOCKS_LOCK = threading.Lock()

def get_gpu_lock(gpu_id: int) -> threading.Lock:
    """Get or create a lock for a specific GPU to prevent concurrent access."""
    with GPU_LOCKS_LOCK:
        if gpu_id not in GPU_LOCKS:
            GPU_LOCKS[gpu_id] = threading.Lock()
        return GPU_LOCKS[gpu_id]


# ===== MODIFIED reranker_scoring.py =====

class BGEScorer:
    """Handles BGE Reranker scoring operations with memory-efficient GPU processing."""
    
    def __init__(self, num_gpus: int = 4, gpu_ids: List[int] = None):
        """Initialize BGEScorer with memory-efficient GPU management."""
        # Import FlagReranker directly without CUDA_VISIBLE_DEVICES restriction
        from FlagEmbedding import FlagReranker
        
        self.available_gpus = 0
        self.rerankers = []
        self.reranker = None  # CPU fallback
        
        # OPTIMIZED batch sizes for maximum GPU utilization
        self.gpu_batch_size = 5000   # Increased for better GPU utilization
        self.query_batch_size = 2000 # Increased for better throughput
        
        if num_gpus > 0 and torch.cuda.is_available():
            # Determine which GPUs to use
            if gpu_ids is not None:
                # Use physical GPU IDs directly (not remapped by CUDA_VISIBLE_DEVICES)
                available_gpu_count = torch.cuda.device_count()
                print(f"🔍 Total available GPUs: {available_gpu_count}")
                print(f"🔍 Requested physical GPU IDs: {gpu_ids}")
                
                # Filter to only use valid GPU IDs
                target_gpus = [gpu_id for gpu_id in gpu_ids if 0 <= gpu_id < available_gpu_count]
                
                self.available_gpus = len(target_gpus)
                print(f"🎯 Using physical GPU IDs: {target_gpus}")
            else:
                # Use first num_gpus GPUs
                self.available_gpus = min(num_gpus, torch.cuda.device_count())
                target_gpus = list(range(self.available_gpus))
                print(f"🔧 Using first {self.available_gpus} GPUs with conservative memory management")
            
            # Initialize reranker on each target GPU with memory checks
            for gpu_id in target_gpus:
                try:
                    # Check available memory before initialization
                    torch.cuda.set_device(gpu_id)
                    props = torch.cuda.get_device_properties(gpu_id)
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    total = props.total_memory
                    available = total - allocated
                    
                    print(f"🔍 GPU {gpu_id}: {available/1024**3:.2f}GB available of {total/1024**3:.2f}GB total")
                    
                    # Only initialize if enough memory is available (at least 2GB)
                    if available < 2 * 1024**3:  # 2GB minimum
                        print(f"⚠️ Skipping GPU {gpu_id} due to insufficient memory ({available/1024**3:.2f}GB)")
                        continue
                    
                    # Initialize reranker on this specific GPU
                    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device=f'cuda:{gpu_id}')
                    
                    # Store (gpu_id, reranker) tuple
                    self.rerankers.append((gpu_id, reranker))
                    print(f"✅ Initialized reranker on GPU {gpu_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to initialize reranker on GPU {gpu_id}: {e}")
                    continue
            
            if self.rerankers:
                print(f"🎯 Successfully initialized {len(self.rerankers)} GPU rerankers")
            else:
                print("⚠️ Failed to initialize any GPU rerankers, falling back to CPU")
                self.available_gpus = 0
        
        # Initialize CPU fallback reranker
        if self.available_gpus == 0:
            print("🖥️ Initializing CPU reranker as fallback")
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False, device='cpu')
    
    def set_batch_sizes(self, gpu_batch_size: int = None, query_batch_size: int = None):
        """Set custom batch sizes for memory-efficient processing."""
        if gpu_batch_size is not None:
            # Enforce maximum limits to prevent OOM
            self.gpu_batch_size = min(gpu_batch_size, 3000)  # Hard limit of 3000
            print(f"🎯 GPU batch size set to {self.gpu_batch_size} (limited for memory safety)")
        
        if query_batch_size is not None:
            # Enforce maximum limits to prevent OOM
            self.query_batch_size = min(query_batch_size, 1500)  # Hard limit of 1500
            print(f"🎯 Query batch size set to {self.query_batch_size} (limited for memory safety)")
    
    def _distribute_work_with_memory_awareness(self, num_tasks: int) -> List[List[int]]:
        """Distribute work across GPUs with memory awareness to prevent OOM."""
        if not self.rerankers:
            return []
        
        num_gpus = len(self.rerankers)
        
        # Check current GPU memory usage and distribute accordingly
        gpu_memory_info = []
        for gpu_idx, (gpu_id, reranker) in enumerate(self.rerankers):
            try:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                available = total - allocated
                gpu_memory_info.append((gpu_idx, gpu_id, available, allocated))
            except Exception as e:
                print(f"⚠️ Could not check GPU {gpu_id} memory: {e}")
                gpu_memory_info.append((gpu_idx, gpu_id, 0, total))
        
        # Sort GPUs by available memory (descending)
        gpu_memory_info.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🎯 GPU Memory Distribution:")
        for gpu_idx, gpu_id, available, allocated in gpu_memory_info:
            print(f"  GPU {gpu_id}: {available/1024**3:.2f}GB available, {allocated/1024**3:.2f}GB used")
        
        # Distribute work based on available memory
        distribution = [[] for _ in range(num_gpus)]
        
        # Calculate work per GPU based on available memory
        total_available = sum(available for _, _, available, _ in gpu_memory_info)
        if total_available > 0:
            for gpu_idx, gpu_id, available, _ in gpu_memory_info:
                # Allocate tasks proportionally to available memory
                gpu_tasks = int((available / total_available) * num_tasks)
                if gpu_tasks > 0:
                    start_idx = len([task for tasks in distribution[:gpu_idx] for task in tasks])
                    end_idx = min(start_idx + gpu_tasks, num_tasks)
                    distribution[gpu_idx] = list(range(start_idx, end_idx))
        else:
            # Fallback: equal distribution
            tasks_per_gpu = num_tasks // num_gpus
            remainder = num_tasks % num_gpus
            
            for gpu_idx in range(num_gpus):
                start_idx = gpu_idx * tasks_per_gpu + min(gpu_idx, remainder)
                end_idx = start_idx + tasks_per_gpu + (1 if gpu_idx < remainder else 0)
                distribution[gpu_idx] = list(range(start_idx, end_idx))
        
        # Print distribution
        print(f"📊 Work Distribution:")
        for gpu_idx, tasks in enumerate(distribution):
            gpu_id = gpu_memory_info[gpu_idx][1] if gpu_idx < len(gpu_memory_info) else "unknown"
            print(f"  GPU {gpu_id}: {len(tasks)} tasks")
        
        return distribution
    
    def _score_with_memory_efficient_gpu_processing(self, query_chunk_pairs: List[List[str]]) -> List[float]:
        """Score query-chunk pairs using memory-efficient GPU processing."""
        if not self.rerankers:
            return [0.0] * len(query_chunk_pairs)
        
        # Import torch at the beginning to avoid UnboundLocalError
        import torch
        
        total_pairs = len(query_chunk_pairs)
        num_gpus = len(self.rerankers)
        
        print(f"🚀 MEMORY-EFFICIENT PROCESSING: {total_pairs} pairs using {num_gpus} GPUs")
        
        # Use smaller batch sizes to prevent OOM
        max_batch_size = min(self.gpu_batch_size, total_pairs // num_gpus + 1)
        print(f"🎯 Using batch size: {max_batch_size}")
        
        # Distribute work with memory awareness
        gpu_task_distribution = self._distribute_work_with_memory_awareness(total_pairs)
        
        all_scores = [0.0] * total_pairs
        
        def process_gpu_batch_with_chunking(gpu_idx: int, task_indices: List[int]) -> List[Tuple[int, float]]:
            """Process a GPU's batch with chunking to prevent OOM."""
            if not task_indices:
                return []
            
            gpu_id, reranker = self.rerankers[gpu_idx]
            
            try:
                    torch.cuda.set_device(gpu_id)
                    
                    # Process in optimized chunks for maximum GPU utilization
                    chunk_size = min(max_batch_size, 2000)  # Process max 2000 pairs at a time
                    results = []
                    
                    for start_idx in range(0, len(task_indices), chunk_size):
                        end_idx = min(start_idx + chunk_size, len(task_indices))
                        chunk_indices = task_indices[start_idx:end_idx]
                        
                        # Extract pairs for this chunk
                        chunk_pairs = [query_chunk_pairs[i] for i in chunk_indices]
                        
                        print(f"🎯 GPU {gpu_id} processing chunk {start_idx//chunk_size + 1} ({len(chunk_pairs)} pairs)")
                        
                        # Process this chunk
                        chunk_scores = reranker.compute_score(chunk_pairs, normalize=True)
                        
                        # Store results
                        for local_idx, score in enumerate(chunk_scores):
                            global_idx = chunk_indices[local_idx]
                            results.append((global_idx, float(score)))
                        
                        # Memory cleanup after each chunk
                        torch.cuda.empty_cache()
                    
                    print(f"✅ GPU {gpu_id} completed {len(task_indices)} pairs in {len(range(0, len(task_indices), chunk_size))} chunks")
                    return results
                    
            except Exception as e:
                print(f"❌ Error processing on GPU {gpu_id}: {e}")
                return [(idx, 0.0) for idx in task_indices]
        
        # Process all GPUs in parallel
        import concurrent.futures
        
        # Use all available GPUs for maximum parallelization
        max_workers = num_gpus  # Use all GPUs for maximum performance
        print(f"🚀 Using {max_workers} workers for FULL GPU utilization")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_gpu = {
                executor.submit(process_gpu_batch_with_chunking, gpu_idx, task_indices): gpu_idx
                for gpu_idx, task_indices in enumerate(gpu_task_distribution)
                if task_indices
            }
            
            for future in concurrent.futures.as_completed(future_to_gpu):
                gpu_idx = future_to_gpu[future]
                try:
                    results = future.result()
                    for global_idx, score in results:
                        if global_idx < len(all_scores):
                            all_scores[global_idx] = score
                except Exception as e:
                    print(f"❌ Error collecting results from GPU {gpu_idx}: {e}")
        
        print(f"🎉 Memory-efficient processing completed for {total_pairs} pairs")
        return all_scores
    
    def score_multiple_queries_parallel(self, queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Score multiple queries against chunks using memory-efficient streaming processing."""
        if not queries or not chunks:
            return []
        
        # Import torch at the beginning to avoid UnboundLocalError
        import torch
        
        total_queries = len(queries)
        total_chunks = len(chunks)
        total_pairs = total_queries * total_chunks
        
        print(f"🚀 MEMORY-EFFICIENT STREAMING MULTI-QUERY PROCESSING")
        print(f"📊 {total_queries} queries × {total_chunks} chunks = {total_pairs} total pairs")
        
        # Memory check before processing
        if torch.cuda.is_available() and total_pairs > 50000:
            print(f"⚠️ Large workload detected ({total_pairs} pairs). Using streaming processing.")
        
        if self.available_gpus == 0:
            # CPU fallback with streaming
            all_results = []
            for query in queries:
                query_scores = self.score_query_chunk_pairs(query, chunks)
                all_results.append(query_scores)
            return all_results
        
        # STREAMING APPROACH: Process queries in batches to prevent OOM
        batch_size = min(self.query_batch_size, max(1, total_queries // 8))  # Process max 25% of queries at once
        print(f"🎯 Using streaming batch size: {batch_size} queries per batch")
        
        all_results = []
        
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_queries = queries[batch_start:batch_end]
            
            print(f"🔄 Processing batch {batch_start//batch_size + 1}: queries {batch_start+1}-{batch_end}")
            
            # Process this batch of queries
            batch_results = self._process_query_batch_streaming(batch_queries, chunks)
            all_results.extend(batch_results)
            
            # Memory cleanup after each batch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"✅ Batch {batch_start//batch_size + 1} completed. Memory cleaned up.")
        
        print(f"🎉 Streaming multi-query processing completed for {total_queries} queries")
        return all_results
    
    def _process_query_batch_streaming(self, batch_queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Process a batch of queries using streaming to prevent OOM."""
        if not batch_queries or not chunks:
            return []
        
        batch_size = len(batch_queries)
        total_chunks = len(chunks)
        
        # Create query-chunk pairs for this batch only
        batch_query_chunk_pairs = []
        query_mapping = []
        
        for query_idx, query in enumerate(batch_queries):
            for chunk in chunks:
                batch_query_chunk_pairs.append([query, chunk['chunk_text']])
                query_mapping.append(query_idx)
        
        print(f"🎯 Processing {len(batch_query_chunk_pairs)} pairs for {batch_size} queries")
        
        # Process using memory-efficient GPU processing
        batch_scores = self._score_with_memory_efficient_gpu_processing(batch_query_chunk_pairs)
        
        # Reconstruct results by query for this batch
        batch_results = [{} for _ in range(batch_size)]
        
        for pair_idx, score in enumerate(batch_scores):
            query_idx = query_mapping[pair_idx]
            chunk_id = chunks[pair_idx % total_chunks]['chunk_id']
            batch_results[query_idx][chunk_id] = score
        
        return batch_results
    
    def score_query_chunk_pairs(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score a single query against multiple chunks."""
        if not query or not chunks:
            return {}
        
        # Create query-chunk pairs
        query_chunk_pairs = [[query, chunk['chunk_text']] for chunk in chunks]
        
        # Score using GPU if available, otherwise CPU
        if self.available_gpus > 0 and self.rerankers:
            scores = self._score_with_memory_efficient_gpu_processing(query_chunk_pairs)
        elif self.reranker is not None:
            # CPU fallback
            try:
                scores = self.reranker.compute_score(query_chunk_pairs, normalize=True)
                scores = [float(score) for score in scores]
            except Exception as e:
                print(f"❌ Error in CPU scoring: {e}")
                scores = [0.0] * len(query_chunk_pairs)
        else:
            print("❌ No reranker available (neither GPU nor CPU)")
            scores = [0.0] * len(query_chunk_pairs)
        
        # Return results as dict mapping chunk_id to score
        result = {}
        for i, chunk in enumerate(chunks):
            if i < len(scores):
                result[chunk['chunk_id']] = scores[i]
            else:
                result[chunk['chunk_id']] = 0.0
        
        return result