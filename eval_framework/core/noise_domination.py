"""
Noise Domination Detection Module for DeepHalluBench

Implements the noise-induced hallucination detection:

Noise-induced Hallucination (N): Failure to prioritize informative evidence

The module uses a cluster-based heuristic at two granularities:
1. Global-level: Assessing total information utilization
2. Local-level: Measuring utilization within each iterative loop

Components:
1. Semantic Clustering: Map retrieved chunks into semantic clusters
2. Value Estimation: Rank clusters by relevance to sub-queries
3. Penalty Quantification: Penalize neglected high-ranking clusters
4. Hallucination Quantification: Normalize against worst-case
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ClusterInfo:
    """Container for cluster information."""
    cluster_id: int
    chunks: List[Dict[str, Any]]
    score: float
    rank: int
    is_utilized: bool
    size: int


class NoiseDominationDetector:
    """
    Detects noise-induced hallucinations by analyzing chunk utilization.

    Identifies clusters of high-value information that were ignored
    in favor of lower-value content.
    """

    def __init__(self, config):
        """
        Initialize the noise domination detector.

        Args:
            config: EvaluationConfig object
        """
        self.config = config

        # Lazy-loaded components
        self._embedder = None
        self._clusterer = None

    @property
    def embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            self._embedder = self._load_embedder()
        return self._embedder

    @property
    def clusterer(self):
        """Lazy load clustering model."""
        if self._clusterer is None:
            self._clusterer = self._load_clusterer()
        return self._clusterer

    def _load_embedder(self):
        """Load embedding model (CPU to save GPU memory)."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self.config.embedding.model_name
            embedder = SentenceTransformer(model_name, device='cpu')
            return embedder
        except Exception as e:
            print(f"⚠️ Failed to load embedder: {e}")
            return None

    def _load_clusterer(self):
        """Load clustering model (UMAP + HDBSCAN)."""
        try:
            import umap
            import hdbscan
            return {"umap": umap, "hdbscan": hdbscan}
        except ImportError:
            print("⚠️ umap-learn or hdbscan not installed. Install with: pip install umap-learn hdbscan")
            return None

    def detect(
        self, decomposed: Dict[str, Any], claim_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect noise domination hallucinations.

        Args:
            decomposed: Decomposed trajectory data
            claim_results: Claim verification results

        Returns:
            Dictionary with noise domination analysis
        """
        chunks = decomposed.get("chunks", [])
        sub_queries = decomposed.get("sub_queries", [decomposed.get("query", "")])

        if not chunks:
            return {
                "score": 0.0,
                "mapped_clusters": [],
                "unmapped_clusters": [],
                "details": {}
            }

        # Generate cache file path
        chunks_hash = hash(tuple(sorted([c.get("chunk_id", str(c)) if isinstance(c, dict) else str(c) for c in chunks])))
        cache_file = os.path.join(self.config.cache_dir, f"noise_domination_{chunks_hash}.json")

        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"  📁 Loading noise domination from cache")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"  ✅ Loaded noise domination from cache")
                return cached_data
            except Exception as e:
                print(f"  ⚠️ Failed to load cache: {e}")

        print(f"  Analyzing noise domination for {len(chunks)} chunks...")

        # Identify utilized chunks (chunks that supported claims)
        utilized_chunks = self._get_utilized_chunks(claim_results)

        # Step 1: Semantic Clustering
        clusters = self._cluster_chunks(chunks, sub_queries)

        if not clusters:
            return {
                "score": 0.0,
                "mapped_clusters": [],
                "unmapped_clusters": [],
                "details": {"reason": "clustering_failed"}
            }

        # Step 2: Rank clusters by relevance
        clusters = self._rank_clusters(clusters, sub_queries)

        # Step 3: Mark utilized vs neglected clusters
        clusters = self._mark_utilized_clusters(clusters, utilized_chunks)

        # Step 4: Calculate hallucination score
        score = self._calculate_nd_score(clusters)

        # Separate mapped and unmapped clusters
        mapped = [c for c in clusters if c.is_utilized]
        unmapped = [c for c in clusters if not c.is_utilized]

        result = {
            "score": score,
            "mapped_clusters": [
                {"id": c.cluster_id, "rank": c.rank, "size": c.size, "score": c.score}
                for c in mapped
            ],
            "unmapped_clusters": [
                {"id": c.cluster_id, "rank": c.rank, "size": c.size, "score": c.score}
                for c in unmapped
            ],
            "details": {
                "total_clusters": len(clusters),
                "utilized_clusters": len(mapped),
                "neglected_clusters": len(unmapped),
                "total_chunks": len(chunks),
                "utilized_chunks": len(utilized_chunks),
            }
        }

        # Save to cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  💾 Saved noise domination to cache")
        except Exception:
            pass

        return result

    def _get_utilized_chunks(self, claim_results: Dict[str, Any]) -> set:
        """Get set of chunk IDs that were used to support claims (Chunk Memory).
        Reference: EntailedChunkMemory from overall_noise_domination.py."""
        utilized = set()
        for result in claim_results.get("results", []):
            if result.get("judgment") == "Support":
                for chunk in result.get("evidence_chunks", []):
                    cid = chunk.get("chunk_id", "")
                    if cid:
                        utilized.add(cid)
                for rc in result.get("relevant_chunks", []):
                    cid = rc.get("chunk_id", "")
                    if cid:
                        utilized.add(cid)
        return utilized

    def _cluster_chunks(
        self, chunks: List[Dict[str, Any]], queries: List[str]
    ) -> List[ClusterInfo]:
        """Cluster chunks using UMAP + HDBSCAN."""
        if not chunks or not self.embedder:
            return []

        try:
            # Extract texts
            texts = [c.get("text", "")[:512] for c in chunks]  # Truncate

            # Get embeddings
            embeddings = self.embedder.encode(texts, show_progress_bar=False)

            # Reduce dimensionality with UMAP
            if self.clusterer:
                reducer = self.clusterer["umap"].UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                embedding_2d = reducer.fit_transform(embeddings)

                # Cluster with HDBSCAN
                clusterer = self.clusterer["hdbscan"].HDBSCAN(
                    min_cluster_size=2,
                    min_samples=1,
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(embedding_2d)
            else:
                # Fallback: simple random clustering
                labels = np.random.randint(0, min(5, len(chunks)), size=len(chunks))

            # Build cluster info
            cluster_dict = defaultdict(list)
            for i, (chunk, label) in enumerate(zip(chunks, labels)):
                cluster_dict[label].append((i, chunk))

            clusters = []
            for label, items in cluster_dict.items():
                cluster_chunks = [c for _, c in items]
                # Keep noise points as singleton clusters (reference: preserves them)
                cluster_id = int(label) if label != -1 else max(
                    [int(l) for l in cluster_dict.keys() if l != -1] + [0]) + 1 + len(clusters)
                clusters.append(ClusterInfo(
                    cluster_id=cluster_id,
                    chunks=cluster_chunks,
                    score=0.0,
                    rank=0,
                    is_utilized=False,
                    size=len(cluster_chunks)
                ))

            return clusters

        except Exception as e:
            print(f"⚠️ Clustering failed: {e}")
            return []

    def _rank_clusters(
        self, clusters: List[ClusterInfo], queries: List[str]
    ) -> List[ClusterInfo]:
        """Rank clusters by relevance to queries."""
        if not clusters or not self.embedder:
            return clusters

        try:
            # Get query embeddings
            query_embeddings = self.embedder.encode(queries, show_progress_bar=False)
            query_embedding = np.mean(query_embeddings, axis=0)

            # Score each cluster
            for cluster in clusters:
                texts = [c.get("text", "")[:256] for c in cluster.chunks]
                if texts:
                    chunk_embeddings = self.embedder.encode(texts, show_progress_bar=False)
                    # Average similarity to query
                    similarities = np.dot(chunk_embeddings, query_embedding) / (
                        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
                    )
                    cluster.score = float(np.max(similarities))

            # Sort by score descending and assign ranks
            clusters.sort(key=lambda c: c.score, reverse=True)
            for rank, cluster in enumerate(clusters):
                cluster.rank = rank + 1  # 1-based rank

            return clusters

        except Exception as e:
            print(f"⚠️ Cluster ranking failed: {e}")
            # Fallback: assign ranks by size
            clusters.sort(key=lambda c: c.size, reverse=True)
            for rank, cluster in enumerate(clusters):
                cluster.rank = rank + 1
            return clusters

    def _mark_utilized_clusters(
        self, clusters: List[ClusterInfo], utilized_chunks: set
    ) -> List[ClusterInfo]:
        """Mark clusters as utilized if they contain entailed chunks."""
        for cluster in clusters:
            for chunk in cluster.chunks:
                chunk_id = chunk.get("chunk_id", "")
                if chunk_id in utilized_chunks:
                    cluster.is_utilized = True
                    break

        return clusters

    def _calculate_nd_score(self, clusters: List[ClusterInfo]) -> float:
        """
        Calculate noise domination hallucination score.

        Aligned with reference overall_noise_domination.py:
        NWIS = actual_wis / max_wis

        actual_wis = Σ(inversion_count / (rank+1)) for unmapped clusters
        max_wis = num_mapped * Σ(1/i) for i=1..num_unmapped
        """
        if not clusters:
            return 0.0

        # Get 0-indexed ranks of mapped and unmapped clusters (position in sorted list)
        mapped_ranks = sorted([i for i, c in enumerate(clusters) if c.is_utilized])
        unmapped_ranks = sorted([i for i, c in enumerate(clusters) if not c.is_utilized])
        num_mapped = len(mapped_ranks)
        num_unmapped = len(unmapped_ranks)

        if num_mapped == 0:
            # No evidence utilized → worst-case noise
            return 1.0
        if num_unmapped == 0:
            return 0.0

        # Actual WIS: for each unmapped cluster, penalty P_c = (S_c × N_inv) / R_c (Paper Eq.)
        # P_c = size × inversion_count / (rank + 1)
        actual_wis = 0.0
        for u_rank in unmapped_ranks:
            c = clusters[u_rank]
            inversion_count = sum(1 for m_rank in mapped_ranks if m_rank > u_rank)
            if inversion_count > 0:
                actual_wis += c.size * inversion_count / (u_rank + 1)

        if actual_wis == 0:
            return 0.0

        # Max WIS: worst case = all mapped clusters rank after all unmapped.
        # Unmapped clusters at best ranks (0, 1, ...), largest first.
        unmapped_sizes = sorted([clusters[r].size for r in unmapped_ranks], reverse=True)
        harmonic_weighted = sum(unmapped_sizes[i] / (i + 1) for i in range(len(unmapped_sizes)))
        max_wis = num_mapped * harmonic_weighted

        if max_wis == 0:
            return 0.0

        return min(actual_wis / max_wis, 1.0)
