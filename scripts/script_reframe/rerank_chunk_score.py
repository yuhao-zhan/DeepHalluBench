#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any

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
from reranker_scoring import BGEScorer
from memory_config import get_memory_config, get_gpu_batch_size, get_query_batch_size, should_cleanup_memory


class IntegratedChunkScorer:
    """Top program that orchestrates relevance scoring using only NLI reranker with memory optimization."""

    def __init__(
        self,
        sbert_model: str = "all-MiniLM-L6-v2",
        ner_threshold: float = 0.5,
        c: float = 6.0,
        num_gpus: int = 4,
        processes_per_gpu: int = 1,
        gpu_ids: List[int] = None,
        bm25_k1: float = 1.5,  # Kept for interface compatibility but unused
        bm25_b: float = 0.75,   # Kept for interface compatibility but unused
        bm25_weight: float = 0.0,  # Set to 0 since we're not using BM25
        reranker_weight: float = 1.0,  # Set to 1.0 since it's the only scoring method
        gpu_batch_size: int = None,  # Will be set dynamically
        query_batch_size: int = None,  # Will be set dynamically
        reranker_instance: Any = None,  # Optional pre-initialized reranker instance
    ) -> None:
        self.sbert_model = sbert_model
        self.ner_threshold = ner_threshold
        self.c = c
        self.num_gpus = num_gpus
        self.processes_per_gpu = processes_per_gpu
        self.gpu_ids = gpu_ids
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.bm25_weight = bm25_weight
        self.reranker_weight = reranker_weight
        
        # Get memory configuration
        self.memory_config = get_memory_config()
        
        # Set batch sizes dynamically based on available memory
        if gpu_batch_size is None:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.gpu_batch_size = get_gpu_batch_size(gpu_memory_gb)
                else:
                    self.gpu_batch_size = self.memory_config['gpu_limits']['max_gpu_batch_size']
            except:
                self.gpu_batch_size = self.memory_config['gpu_limits']['max_gpu_batch_size']
        else:
            self.gpu_batch_size = gpu_batch_size
            
        if query_batch_size is None:
            try:
                import psutil
                system_memory_gb = psutil.virtual_memory().total / 1024**3
                self.query_batch_size = get_query_batch_size(system_memory_gb)
            except:
                self.query_batch_size = self.memory_config['gpu_limits']['max_query_batch_size']
        else:
            self.query_batch_size = query_batch_size

        self.chunk_locator = OptimizedContextLocator()
        
        # MEMORY OPTIMIZATION: Lazy initialization or use provided instance
        if reranker_instance is not None:
            print(f"🔧 Using provided reranker instance to avoid model reloading")
            self._reranker = reranker_instance
        else:
            self._reranker = None
        
        print(f"🔧 IntegratedChunkScorer initialized with dynamic batch sizes:")
        print(f"  - GPU batch size: {self.gpu_batch_size}")
        print(f"  - Query batch size: {self.query_batch_size}")
        print(f"  - Memory config: {self.memory_config['gpu_limits']}")
        print(f"  - Using only reranker model (BM25 disabled)")
    
    @property
    def reranker(self):
        """Lazy initialization of reranker to save memory."""
        if self._reranker is None:
            print("🔧 Lazy loading BGEScorer...")
            self._reranker = BGEScorer(self.num_gpus, gpu_ids=self.gpu_ids, processes_per_gpu=self.processes_per_gpu)
            # Set conservative batch sizes to prevent OOM
            self._reranker.set_batch_sizes(
                gpu_batch_size=self.gpu_batch_size,
                query_batch_size=self.query_batch_size
            )
        return self._reranker

    @staticmethod
    def _min_max_norm(values: List[float]) -> Tuple[List[float], float, float]:
        """Normalize values using min-max normalization."""
        if not values:
            return [], 0.0, 0.0
        vmin = min(values)
        vmax = max(values)
        if vmax - vmin < 1e-12:
            return [0.0 for _ in values], vmin, vmax
        return [float((v - vmin) / (vmax - vmin)) for v in values], vmin, vmax

    def score_chunks_sync(
        self,
        queries: List[str],
        urls: List[str],
        web_content_cache: Dict[str, str],
        cache_file: str = None,
        url_mapping: Dict[str, int] = None,
    ) -> Dict[str, Any]:
        """
        SYNCHRONOUS version of score_chunks for multiprocessing compatibility.
        """
        import asyncio
        return asyncio.run(self.score_chunks(queries, urls, web_content_cache, cache_file, url_mapping))
    
    async def score_chunks(
        self,
        queries: List[str],
        urls: List[str],
        web_content_cache: Dict[str, str],
        cache_file: str = None,
        url_mapping: Dict[str, int] = None,
    ) -> Dict[str, Any]:
        """
        Score chunks against queries using only NLI reranker approach.
        
        Args:
            queries: List of queries to score against
            urls: List of URLs to process
            web_content_cache: Cached web content
            cache_file: Optional cache file path for saving/loading scores
            url_mapping: Global URL-to-index mapping for consistent chunk IDs across runs
            
        Returns:
            Dictionary containing scoring results
        """
        # MEMORY CHECK before processing
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"🔍 GPU {i}: {allocated:.2f}GB / {total:.2f}GB used before processing")
        
        # CACHE CHECK: Load existing chunk scores if cache file provided
        cached_chunk_scores = {}
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if 'chunk_score' in cache_data:
                    cached_chunk_scores = cache_data['chunk_score']
                    print(f"📋 Loaded {len(cached_chunk_scores)} cached chunk scores from {cache_file}")
                else:
                    print(f"❌ No chunk scores found in cache file: {cache_file}")
                    cached_chunk_scores = {}
            except Exception as e:
                print(f"⚠️ Error loading cache file: {e}, proceeding without cache")
                cached_chunk_scores = {}
        
        # Select documents by provided urls if given; otherwise use all cache entries
        if urls:
            valid_pairs = [(u, web_content_cache.get(u, "")) for u in urls if u in web_content_cache]
        else:
            valid_pairs = list(web_content_cache.items())

        valid_urls: List[str] = []
        documents: List[str] = []
        for url, content in valid_pairs:
            if not content:
                continue
            valid_urls.append(url)
            documents.append(content)

        # MEMORY CHECK: Limit document processing if too many
        max_docs = self.memory_config['gpu_limits']['max_documents']
        # Removed max_docs limitation - process all documents without truncation

        # Extract 4-sentence chunks per document and make chunk ids globally unique
        all_chunks: List[Dict[str, Any]] = []
        chunks_to_score: List[Dict[str, Any]] = []
        cached_chunks: List[Dict[str, Any]] = []
        
        for url_idx, (url, document) in enumerate(zip(valid_urls, documents)):
            chunks = self.chunk_locator.extract_sentences(document)
            for chunk in chunks:
                # Use global URL mapping if available, otherwise fall back to local index
                if url_mapping and url in url_mapping:
                    global_url_idx = url_mapping[url]
                else:
                    global_url_idx = url_idx
                global_chunk_id = f"{global_url_idx}-{chunk['chunk_id']}"
                
                # Check if this chunk has already been scored
                if global_chunk_id in cached_chunk_scores:
                    print(f"✅ Using cached scores for chunk {global_chunk_id}")
                    cached_chunks.append({
                        "chunk_id": global_chunk_id,
                        "chunk_text": chunk["chunk_text"],
                        "position": chunk.get("position", 0),
                        "url": url,
                        "chunk_id_original": chunk["chunk_id"],  # Store original chunk ID
                        "sentence_count": chunk.get("sentence_count", 0),
                        "sentence_indices": chunk.get("sentence_indices", []),
                        "cached_scores": cached_chunk_scores[global_chunk_id]
                    })
                else:
                    chunks_to_score.append({
                        "chunk_id": global_chunk_id,
                        "chunk_text": chunk["chunk_text"],
                        "position": chunk.get("position", 0),
                        "url": url,
                        "chunk_id_original": chunk["chunk_id"],  # Store original chunk ID
                        "sentence_count": chunk.get("sentence_count", 0),
                        "sentence_indices": chunk.get("sentence_indices", []),
                    })
                
                all_chunks.append({
                    "chunk_id": global_chunk_id,
                    "chunk_text": chunk["chunk_text"],
                    "position": chunk.get("position", 0),
                    "url": url,
                    "chunk_id_original": chunk["chunk_id"],  # Store original chunk ID from extract_sentences
                    "sentence_count": chunk.get("sentence_count", 0),
                    "sentence_indices": chunk.get("sentence_indices", []),
                })

        print(f"📊 Chunk analysis: {len(cached_chunks)} cached, {len(chunks_to_score)} to score")
        
        # If all chunks are cached, return cached results
        if not chunks_to_score:
            print(f"🎉 All chunks already scored! Returning cached results.")
            return self._build_result_from_cache(queries, all_chunks, cached_chunk_scores)
        
        # MEMORY CHECK: Limit chunks if too many
        max_chunks = self.memory_config['gpu_limits']['max_chunks']
        if len(chunks_to_score) > max_chunks:
            print(f"⚠️ WARNING: Processing {len(chunks_to_score)} chunks (exceeds max_chunks={max_chunks}), but proceeding without truncation")
            # chunks_to_score = chunks_to_score[:max_chunks]  # DISABLED: Don't truncate chunks

        chunk_ids = [c["chunk_id"] for c in chunks_to_score]
        chunk_texts = [c["chunk_text"] for c in chunks_to_score]

        # MEMORY-EFFICIENT processing - only reranker, no BM25
        print(f"🚀 Processing {len(queries)} queries against {len(chunks_to_score)} chunks using only reranker")
        
        # Get reranker scores using memory-efficient GPU processing
        reranker_scores_per_query = self.reranker.score_multiple_queries_parallel(queries, chunks_to_score)
        
        # MEMORY CLEANUP after reranking
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Normalize per query and compute combined scores (reranker only)
        per_query_norms: List[Dict[str, Any]] = []
        for q_idx, (query, reranker_scores) in enumerate(zip(queries, reranker_scores_per_query)):
            # Convert reranker scores dict to list
            rer_scores = [float(reranker_scores.get(cid, 0.0)) for cid in chunk_ids]
            
            # Since we're not using BM25, set those values to 0
            bm25_norm = [0.0 for _ in rer_scores]
            rer_norm, rer_min, rer_max = self._min_max_norm(rer_scores)
            
            per_query_norms.append(
                {
                    "query": query,
                    "bm25_norm": bm25_norm,
                    "reranker_norm": rer_norm,
                    "bm25_min": 0.0,
                    "bm25_max": 0.0,
                    "rer_min": rer_min,
                    "rer_max": rer_max,
                }
            )

        # Build new chunk scores for chunks that were just scored
        new_chunk_scores = {}
        for idx, chunk in enumerate(chunks_to_score):
            chunk_id = chunk["chunk_id"]
            new_chunk_scores[chunk_id] = {
                "url": chunk["url"],
                "chunk_text": chunk["chunk_text"],
                "position": chunk.get("position", 0),
                "chunk_id_original": chunk.get("chunk_id_original", chunk["chunk_id"]),  # Store original chunk ID
                "sentence_count": chunk.get("sentence_count", 0),
                "sentence_indices": chunk.get("sentence_indices", []),
                "scores": {}
            }
            
            for q_idx, qnorm in enumerate(per_query_norms):
                bm25_n = 0.0  # BM25 is disabled
                rer_n = float(qnorm["reranker_norm"][idx]) if idx < len(qnorm["reranker_norm"]) else 0.0
                combined = self.reranker_weight * rer_n  # Only reranker score since bm25_weight = 0
                
                new_chunk_scores[chunk_id]["scores"][str(q_idx)] = {
                    "bm25_norm": bm25_n,
                    "reranker_norm": rer_n,
                    "combined": combined,
                    "query": qnorm["query"]
                }

        # Merge new scores with cached scores
        all_chunk_scores = {**cached_chunk_scores, **new_chunk_scores}
        
        # Save updated chunk scores to cache file if provided
        if cache_file:
            try:
                # Load existing cache data
                if os.path.exists(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                else:
                    cache_data = {}
                
                # Update chunk scores
                cache_data['chunk_score'] = all_chunk_scores
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                
                # Save updated cache
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Saved {len(new_chunk_scores)} new chunk scores to cache: {cache_file}")
                
            except Exception as e:
                print(f"⚠️ Error saving chunk scores to cache: {e}")

        # Build final result using all chunk scores (cached + new)
        return self._build_result_from_all_scores(queries, all_chunks, all_chunk_scores)

    def _build_result_from_cache(self, queries: List[str], all_chunks: List[Dict[str, Any]], cached_chunk_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Build result using only cached chunk scores."""
        print(f"🔄 Building result from {len(cached_chunk_scores)} cached chunk scores")
        
        # Per-query sorted chunks with cached scores
        per_query_output: List[Dict[str, Any]] = []
        for q_idx, query in enumerate(queries):
            q_chunks: List[Dict[str, Any]] = []
            for chunk in all_chunks:
                chunk_id = chunk["chunk_id"]
                if chunk_id in cached_chunk_scores:
                    chunk_data = cached_chunk_scores[chunk_id]
                    q_scores = chunk_data["scores"].get(str(q_idx), {})
                    
                    q_chunks.append({
                        "chunk_id": chunk_id,
                        "url": chunk["url"],
                        "chunk_text": chunk["chunk_text"],
                        "bm25_norm": q_scores.get("bm25_norm", 0.0),
                        "reranker_norm": q_scores.get("reranker_norm", 0.0),
                        "combined": q_scores.get("combined", 0.0),
                    })
            
            # Sort by combined score
            q_chunks.sort(key=lambda x: x["combined"], reverse=True)
            per_query_output.append({"query": query, "chunks": q_chunks})

        # Per-chunk summary across queries
        per_chunk_summary: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in cached_chunk_scores:
                chunk_data = cached_chunk_scores[chunk_id]
                
                combined_scores = {}
                bm25_scores = {}
                rer_scores = {}
                
                for q_idx in range(len(queries)):
                    q_scores = chunk_data["scores"].get(str(q_idx), {})
                    combined_scores[str(q_idx)] = q_scores.get("combined", 0.0)
                    bm25_scores[str(q_idx)] = q_scores.get("bm25_norm", 0.0)
                    rer_scores[str(q_idx)] = q_scores.get("reranker_norm", 0.0)
                
                per_chunk_summary.append({
                    "chunk_id": chunk_id,
                    "url": chunk["url"],
                    "chunk_text": chunk["chunk_text"],
                    "position": chunk.get("position", 0),
                    "chunk_id_original": chunk.get("chunk_id_original", chunk_id),
                    "sentence_count": chunk.get("sentence_count", 0),
                    "sentence_indices": chunk.get("sentence_indices", []),
                    "original_query_scores": combined_scores,
                    "bm25_norm_scores": bm25_scores,
                    "reranker_norm_scores": rer_scores,
                })

        return self._build_metadata_and_result(queries, all_chunks, per_query_output, per_chunk_summary, "cached_only")

    def _build_result_from_all_scores(self, queries: List[str], all_chunks: List[Dict[str, Any]], all_chunk_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Build result using both cached and newly scored chunks."""
        print(f"🔄 Building result from {len(all_chunk_scores)} total chunk scores")
        
        # Per-query sorted chunks with combined scores
        per_query_output: List[Dict[str, Any]] = []
        for q_idx, query in enumerate(queries):
            q_chunks: List[Dict[str, Any]] = []
            for chunk in all_chunks:
                chunk_id = chunk["chunk_id"]
                if chunk_id in all_chunk_scores:
                    chunk_data = all_chunk_scores[chunk_id]
                    q_scores = chunk_data["scores"].get(str(q_idx), {})
                    
                    q_chunks.append({
                        "chunk_id": chunk_id,
                        "url": chunk["url"],
                        "chunk_text": chunk["chunk_text"],
                        "bm25_norm": q_scores.get("bm25_norm", 0.0),
                        "reranker_norm": q_scores.get("reranker_norm", 0.0),
                        "combined": q_scores.get("combined", 0.0),
                    })
            
            # Sort by combined score
            q_chunks.sort(key=lambda x: x["combined"], reverse=True)
            per_query_output.append({"query": query, "chunks": q_chunks})

        # Per-chunk summary across queries
        per_chunk_summary: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in all_chunk_scores:
                chunk_data = all_chunk_scores[chunk_id]
                
                combined_scores = {}
                bm25_scores = {}
                rer_scores = {}
                
                for q_idx in range(len(queries)):
                    q_scores = chunk_data["scores"].get(str(q_idx), {})
                    combined_scores[str(q_idx)] = q_scores.get("combined", 0.0)
                    bm25_scores[str(q_idx)] = q_scores.get("bm25_norm", 0.0)
                    rer_scores[str(q_idx)] = q_scores.get("reranker_norm", 0.0)
                
                per_chunk_summary.append({
                    "chunk_id": chunk_id,
                    "url": chunk["url"],
                    "chunk_text": chunk["chunk_text"],
                    "position": chunk.get("position", 0),
                    "chunk_id_original": chunk.get("chunk_id_original", chunk_id),
                    "sentence_count": chunk.get("sentence_count", 0),
                    "sentence_indices": chunk.get("sentence_indices", []),
                    "original_query_scores": combined_scores,
                    "bm25_norm_scores": bm25_scores,
                    "reranker_norm_scores": rer_scores,
                })

        return self._build_metadata_and_result(queries, all_chunks, per_query_output, per_chunk_summary, "mixed")

    def _build_metadata_and_result(self, queries: List[str], all_chunks: List[Dict[str, Any]], per_query_output: List[Dict[str, Any]], per_chunk_summary: List[Dict[str, Any]], processing_mode: str) -> Dict[str, Any]:
        """Build the final result with metadata."""
        result: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_queries": len(queries),
                "num_chunks": len(all_chunks),
                "queries": queries,
                "urls": list(set(chunk["url"] for chunk in all_chunks)),
                "normalization": "min-max per query",
                "combine_weights": {"bm25": self.bm25_weight, "reranker": self.reranker_weight},
                "processing_mode": processing_mode,
                "model_info": {
                    "reranker": "BGEScorer with memory-efficient processing (BM25 disabled)",
                    "available_gpus": getattr(self.reranker, "available_gpus", 0),
                    "sbert_model": self.sbert_model,
                    "batch_sizes": {
                        "gpu_batch_size": self.gpu_batch_size,
                        "query_batch_size": self.query_batch_size
                    }
                },
            },
            "per_query": per_query_output,
            "detailed_chunk_scores": per_chunk_summary,
        }

        return result
