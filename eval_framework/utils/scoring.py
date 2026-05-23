"""
Scoring utilities for DeepHalluBench.

Provides:
1. ClaimQueryReranker — BGE reranker-based claim→sub-query relevance mapping (related_query)
2. ChunkScorer — BGE embedding + reranker two-stage chunk scoring (top_k_chunks)
"""

import re
import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Global reranker singleton to avoid multi-instance GPU conflicts
_RERANKER_INSTANCE = {"model": None, "model_name": None}


def _get_global_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Get or create a shared FlagReranker instance (single-GPU to avoid multiprocessing pool issues)."""
    if _RERANKER_INSTANCE["model"] is not None:
        return _RERANKER_INSTANCE["model"]
    try:
        from FlagEmbedding import FlagReranker
        import torch
        if torch.cuda.is_available():
            # Use single GPU (cuda:0) to avoid multi-process pool corruption
            reranker = FlagReranker(model_name, use_fp16=True, devices=["cuda:0"])
        else:
            reranker = FlagReranker(model_name, use_fp16=False)
        _RERANKER_INSTANCE["model"] = reranker
        _RERANKER_INSTANCE["model_name"] = model_name
        logger.info(f"Loaded shared reranker (single-GPU): {model_name}")
        return reranker
    except Exception as e:
        logger.warning(f"Failed to load shared reranker: {e}")
        return None


class ClaimQueryReranker:
    """
    BGE Reranker-based claim-to-query relevance mapping.

    Scores all (claim, sub_query) pairs and filters by fixed threshold.
    Used to compute `related_query` in the cache.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.model_name = model_name
        self._reranker = None
        self._device = device

    def _load(self):
        if self._reranker is None:
            self._reranker = _get_global_reranker(self.model_name)
        return self._reranker

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'=+\s*', ' ', text)
        text = re.sub(r'-+\s*', ' ', text)
        text = re.sub(r'_+\s*', ' ', text)
        text = re.sub(r'\*+\s*', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def compute_mapping(
        self, claims: List[str], sub_queries: List[str], fixed_threshold: float = 0.001
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map each claim to its relevant sub-queries via reranker scoring.

        Args:
            claims: List of claim texts.
            sub_queries: List of sub-query/constraint texts.
            fixed_threshold: Minimum reranker score to consider a (claim, query) pair relevant.

        Returns:
            Dict mapping claim_text -> {claim, relevant_queries: [{query_idx, query, score}, ...]}
        """
        reranker = self._load()
        if reranker is None or not claims or not sub_queries:
            return {}

        # Clean texts
        cleaned_claims = [self._clean_text(c) for c in claims]
        cleaned_queries = [self._clean_text(q) for q in sub_queries]

        # Build all pairs
        pairs = []
        pair_mapping = []  # (claim_idx, query_idx)
        for ci, claim in enumerate(cleaned_claims):
            for qi, query in enumerate(cleaned_queries):
                pairs.append([claim, query])
                pair_mapping.append((ci, qi))

        # Score in batches
        batch_size = 512
        all_scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            try:
                scores = reranker.compute_score(batch, normalize=True)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                all_scores.extend(scores)
            except Exception as e:
                logger.warning(f"Reranker batch scoring failed at {start}: {e}")
                all_scores.extend([0.0] * len(batch))

        # Build mapping
        result = {}
        # Initialize per-claim score dicts
        for ci, claim in enumerate(claims):
            result[claim] = {
                "claim": claim,
                "relevant_queries": [],
            }

        for (ci, qi), score in zip(pair_mapping, all_scores):
            score = round(float(score), 4)
            if score > fixed_threshold:
                result[claims[ci]]["relevant_queries"].append({
                    "query_idx": qi,
                    "query": sub_queries[qi],
                    "score": score,
                })

        # Sort by score descending
        for claim in claims:
            result[claim]["relevant_queries"].sort(key=lambda x: x["score"], reverse=True)

        return result


class ChunkScorer:
    """
    Two-stage chunk scoring: embedding similarity filter → reranker top-k selection.
    Matches the reference SimilarityFilter approach.
    """

    def __init__(
        self,
        embedder_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        similarity_threshold: float = 0.4,
        top_k: int = 5,
    ):
        self.embedder_model = embedder_model
        self.reranker_model = reranker_model
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self._embedder = None
        self._reranker = None

    def _load_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedder_model, device='cpu')
                logger.info(f"Loaded embedder: {self.embedder_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedder: {e}")
        return self._embedder

    def _load_reranker(self):
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                import torch
                if torch.cuda.is_available():
                    self._reranker = FlagReranker(self.reranker_model, use_fp16=True, devices=["cuda:0"])
                else:
                    self._reranker = FlagReranker(self.reranker_model, use_fp16=False)
                logger.info(f"Loaded reranker: {self.reranker_model}")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
        return self._reranker

    def compute_top_k_for_claim(
        self, claim: str, chunk_texts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Two-stage top-k chunk selection.

        Stage 1: Embedding similarity filter (cosine > threshold)
        Stage 2: Reranker re-scoring → top-k

        Args:
            claim: Claim text.
            chunk_texts: List of dicts with {"chunk_id": str, "text": str}.

        Returns:
            Top-k chunk dicts with added "similarity" and "rerank_score" fields.
        """
        if not chunk_texts:
            return []

        # Stage 1: Embedding similarity filter
        embedder = self._load_embedder()
        reranker = self._load_reranker()

        filtered = chunk_texts
        if embedder is not None:
            try:
                import numpy as np
                claim_emb = embedder.encode([claim], normalize_embeddings=True)
                chunk_embs = embedder.encode(
                    [c.get("text", "") for c in chunk_texts],
                    normalize_embeddings=True,
                )
                similarities = chunk_embs @ claim_emb.T
                similarities = similarities.flatten()

                # Filter by threshold
                filtered = []
                for i, c in enumerate(chunk_texts):
                    sim = float(similarities[i])
                    if sim >= self.similarity_threshold:
                        c = dict(c)
                        c["similarity"] = sim
                        filtered.append(c)

                if not filtered:
                    # Fallback: keep top-20 even if below threshold
                    idxs = np.argsort(-similarities.flatten())[:20]
                    filtered = []
                    for i in idxs:
                        c = dict(chunk_texts[i])
                        c["similarity"] = float(similarities[i])
                        filtered.append(c)
            except Exception as e:
                logger.warning(f"Embedding similarity filter failed: {e}")
                filtered = chunk_texts

        # Stage 2: Reranker scoring
        if reranker is not None and filtered:
            try:
                pairs = [[claim, c.get("text", "")[:512]] for c in filtered]
                scores = reranker.compute_score(pairs, normalize=True)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                for i, c in enumerate(filtered):
                    c["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

                filtered.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            except Exception as e:
                logger.warning(f"Reranker scoring failed: {e}")
                pass

        return filtered[:self.top_k]
