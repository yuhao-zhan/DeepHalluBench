"""
Restriction Checking via elbow method on action-to-sub_query relevance scores.

Reference: action_checking/scripts/rerank_ato_query_and_action.py
Paper: "For each atomic action, we rank its relevance to all sub-queries and
apply the elbow method to isolate the restrictions it effectively satisfies."

Q_executed = union of first-cluster queries across all actions.
"""

import json
import os
import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ConstraintChecker:

    def __init__(self, config):
        self.config = config

    def check(self, decomposed: Dict[str, Any], query: str) -> Dict[str, Any]:
        actions = decomposed.get("actions", [])
        sub_queries = decomposed.get("sub_queries", [])

        if not sub_queries or not actions:
            return {"total_queries": len(sub_queries), "missed_count": len(sub_queries),
                    "coverage_rate": 0.0, "missed_queries": sub_queries[:],
                    "first_cluster_queries": {}}

        actions_hash = hash(tuple(sorted(
            a.get("action", str(a)) if isinstance(a, dict) else str(a) for a in actions
        )))
        query_hash = hash(query)
        cache_file = os.path.join(self.config.cache_dir,
                                   f"constraint_checking_{query_hash}_{actions_hash}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        action_texts = [a.get("action", str(a)) if isinstance(a, dict) else str(a)
                        for a in actions]
        print(f"  Computing action-sub_query relevance for "
              f"{len(actions)} actions x {len(sub_queries)} queries...")

        scores = _compute_pairwise_scores(action_texts, sub_queries,
                                          self.config.reranker.model_name)
        if not scores:
            result = {"total_queries": len(sub_queries),
                      "missed_count": len(sub_queries),
                      "coverage_rate": 0.0,
                      "missed_queries": sub_queries[:],
                      "first_cluster_queries": {}}
        else:
            result = _analyze_query_action_coverage(scores, sub_queries)

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return result


def _compute_pairwise_scores(action_texts: List[str], sub_queries: List[str],
                              model_name: str) -> Dict[str, Dict[str, float]]:
    """Score every (query, action) pair using BGE reranker.
    Reference: compute_reranking_scores() in rerank_ato_query_and_action.py."""
    from eval_framework.utils.scoring import _get_global_reranker

    reranker = _get_global_reranker(model_name)
    if reranker is None:
        return {}

    pairs = []
    for i, action in enumerate(action_texts):
        for query in sub_queries:
            pairs.append((query, action, i))

    pair_texts = [[q, a] for q, a, _ in pairs]
    try:
        raw = reranker.compute_score(pair_texts, normalize=True, batch_size=32)
        raw_scores = [float(s) for s in raw] if isinstance(raw, list) else [float(raw)]
    except Exception:
        return {}

    scores = {}
    for (query, action, i), score in zip(pairs, raw_scores):
        key = f"{action} ::iter_{i}"
        scores.setdefault(key, {})[query] = score
    return scores


def _find_optimal_clusters(scores: List[float], max_clusters: int = 5) -> int:
    """Elbow method (second derivative of inertias) + silhouette validation.
    Reference: find_optimal_clusters() in rerank_ato_query_and_action.py."""
    if len(scores) <= 1:
        return 1
    X = np.array(scores).reshape(-1, 1)
    inertias, sils = [], []
    k_range = range(1, min(max_clusters + 1, len(scores)))

    for k in k_range:
        if k == 1:
            inertias.append(0)
            sils.append(0)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, km.labels_))

    if len(inertias) <= 2:
        return 2

    second_derivs = [inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
                     for i in range(1, len(inertias) - 1)]
    elbow_k = np.argmax(second_derivs) + 2 if second_derivs else 2

    if len(sils) > 1 and max(sils[1:]) > 0.3:
        return min(elbow_k, np.argmax(sils[1:]) + 2)
    return elbow_k


def _cluster_queries_for_action(action_key: str,
                                 query_scores: Dict[str, float]) -> List[List[str]]:
    """KMeans cluster sub-queries by relevance score per action. First cluster = highest relevance.
    Reference: cluster_queries_for_action() in rerank_ato_query_and_action.py."""
    if not query_scores:
        return []
    queries = list(query_scores.keys())
    sc = list(query_scores.values())
    if len(queries) <= 1:
        return [queries]

    k = _find_optimal_clusters(sc)
    X = np.array(sc).reshape(-1, 1)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

    clusters = [[] for _ in range(k)]
    for q, lab in zip(queries, labels):
        clusters[lab].append(q)

    avg_scores = [(np.mean([query_scores[q] for q in c]), i, c)
                  for i, c in enumerate(clusters)]
    avg_scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in avg_scores]


def _analyze_query_action_coverage(scores: Dict[str, Dict[str, float]],
                                    all_queries: List[str]) -> Dict[str, Any]:
    """Union of first-cluster queries across all actions = Q_executed.
    Reference: analyze_query_action_coverage() in rerank_ato_query_and_action.py."""
    first_cluster_queries = {}
    for action_key, query_scores in scores.items():
        clusters = _cluster_queries_for_action(action_key, query_scores)
        first_cluster_queries[action_key] = clusters[0] if clusters else []

    all_in_first = set()
    for queries in first_cluster_queries.values():
        all_in_first.update(queries)

    all_set = set(all_queries)
    missed = list(all_set - all_in_first)
    total_q = len(all_set)
    covered = len(all_in_first)

    return {
        "total_queries": total_q,
        "missed_count": len(missed),
        "coverage_rate": covered / total_q if total_q > 0 else 0.0,
        "missed_queries": missed,
        "first_cluster_queries": first_cluster_queries,
    }
