#!/usr/bin/env python3
"""
Judge hallucinated claims against previously supported memory.

Pipeline:
1. Collect Support -> NotSupport claim pairs from chain-of-research iterations.
2. Use NLI to relabel entailed NotSupport claims as Support.
3. For remaining NotSupport claims, select top-k similar supported claims.
4. Call LLM in parallel to judge whether each NotSupport claim can be supported
   by the selected memory (including reflective reasoning).
5. Save updated results to an output directory, preserving all other content.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

# Ensure HuggingFace mirror is used
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Local imports (use absolute paths to avoid multiprocessing issues)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(os.path.join(PROJECT_ROOT, "claim_verification", "top_scripts", "models"))
sys.path.append(CURRENT_DIR)

from nli import initialize_nli_models_once, nli_score_batch_parallel  # type: ignore  # noqa: E402
from llm_as_judges import (  # type: ignore  # noqa: E402
    API_KEYS,
    BASE_URL,
    WEB_MODEL,
    APILoadBalancer,
    extract_json_from_content,
    _extract_fields_from_text,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DEFAULT_THRESHOLD = 0.99
DEFAULT_TOP_K = 10
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE = 2.0


def _iteration_order_value(iteration: Any) -> int:
    """Convert iteration labels to comparable integers."""
    if isinstance(iteration, int):
        return iteration
    if iteration == "report":
        # Ensure report is always the latest
        return 10_000
    return 20_000


@dataclass(unsafe_hash=True)
class ClaimEntry:
    """Metadata wrapper for a claim inside the result JSON."""

    text: str
    iteration: Any
    pointer: Dict[str, Any] = field(compare=False)
    section: str  # 'chain' or 'report'
    indices: Tuple[int, int]
    original_judgment: str
    judgment: str
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    cache_list_index: Optional[int] = field(default=None, compare=False)
    cache_position: Optional[int] = field(default=None, compare=False)

    @property
    def order_key(self) -> Tuple[int, int, int]:
        if isinstance(self.iteration, int):
            iter_val = self.iteration
        elif self.iteration == "report":
            iter_val = 10_000
        else:
            iter_val = 20_000
        list_idx = self.cache_list_index if self.cache_list_index is not None else -1
        pos = self.cache_position if self.cache_position is not None else -1
        return iter_val, list_idx, pos

    @property
    def order(self) -> int:
        return self.order_key[0]

    def to_support_record(self) -> Dict[str, Any]:
        record = {"text": self.text}
        record.update(self.metadata)
        record["iteration"] = self.iteration
        record["section"] = self.section
        return record


@dataclass(unsafe_hash=True)
class ActionEntry:
    """Metadata wrapper for an action from cache."""

    text: str
    iteration: int
    list_index: Optional[int] = None
    position: Optional[int] = None

    @property
    def order_key(self) -> Tuple[int, int, int]:
        list_idx = self.list_index if self.list_index is not None else -1
        pos = self.position if self.position is not None else -1
        return self.iteration, list_idx, pos


class SentenceEncoder:
    """Thin wrapper around a SentenceTransformer encoder with caching."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        logger.info("Loading sentence transformer model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 1024
        self.cache: Dict[str, np.ndarray] = {}

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        unique_texts = []
        mapping = []

        for text in texts:
            if text in self.cache:
                mapping.append(text)
            else:
                unique_texts.append(text)
                mapping.append(text)

        if unique_texts:
            new_vectors = self.model.encode(
                unique_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vec in zip(unique_texts, new_vectors):
                self.cache[text] = vec

        return np.array([self.cache[text] for text in mapping])


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _deep_merge(dest: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge src into dest without dropping existing keys.
    - For dict values: merge recursively
    - For lists and scalars: src overwrites dest
    """
    for key, src_val in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(src_val, dict):
            _deep_merge(dest[key], src_val)
        else:
            dest[key] = src_val
    return dest

def _call_llm_with_retry(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
    )
    backoff = 1.0
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=WEB_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            cleaned_content = extract_json_from_content(content)
            return MemoryJudge._parse_llm_json(cleaned_content)
        except Exception as exc:  # pragma: no cover - API failure path
            if attempt == LLM_MAX_RETRIES - 1:
                logger.error("LLM call failed after retries: %s", exc)
                raise exc
            wait_time = backoff + random.random()
            logger.warning("LLM call failed (%s). Retrying in %.2f s.", exc, wait_time)
            time.sleep(wait_time)
            backoff *= LLM_BACKOFF_BASE
    raise RuntimeError("LLM call failed after retries")  # pragma: no cover


def _memory_llm_worker(
    args: Tuple[str, List[Tuple[int, Dict[str, Any], str, str, str]]]
):
    api_key, tasks = args
    results = []
    for claim_id, candidate, support_context, action_context, query in tasks:
        system_prompt, user_prompt = MemoryJudge._build_llm_prompt(query, candidate, support_context, action_context)
        # print("--------------------------------")
        # print(f"User prompt: {user_prompt}")
        llm_result = _call_llm_with_retry(api_key, system_prompt, user_prompt)
        results.append((claim_id, support_context, action_context, llm_result))
    return results


class MemoryJudge:
    """Main orchestrator for memory-based claim relabeling."""

    def __init__(
        self,
        results_dir: str,
        cache_dir: str,
        output_dir: str,
        nli_threshold: float = DEFAULT_THRESHOLD,
        similarity_top_k: int = DEFAULT_TOP_K,
        llm_workers: Optional[int] = None,
        similarity_gpu_ids: Optional[List[int]] = None,
    ):
        self.results_dir = results_dir
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.nli_threshold = nli_threshold
        self.similarity_top_k = similarity_top_k
        self.llm_workers = llm_workers or max(1, min(mp.cpu_count(), len(API_KEYS)))
        self.similarity_gpu_ids = similarity_gpu_ids or []

        os.makedirs(self.output_dir, exist_ok=True)
        initialize_nli_models_once()
        self.encoder = SentenceEncoder()
        self.api_load_balancer = APILoadBalancer(API_KEYS)

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------
    def _collect_action_entries(
        self,
        file_id: str,
    ) -> List[ActionEntry]:
        """Collect all actions from cache file with iteration tracking."""
        cache_path = os.path.join(self.cache_dir, f"cache_{file_id}.json")
        cache_data = _load_json(cache_path) if os.path.exists(cache_path) else {}
        
        actions: List[ActionEntry] = []
        for iter_idx, iter_data in enumerate(cache_data.get("iterations", [])):
            for key, value in iter_data.items():
                if key.startswith("action_list_") and isinstance(value, list):
                    try:
                        list_idx = int(key.split("_")[-1])
                    except (TypeError, ValueError):
                        list_idx = None
                    for pos_idx, action_text in enumerate(value):
                        if isinstance(action_text, str) and action_text.strip():
                            actions.append(
                                ActionEntry(
                                    text=action_text.strip(),
                                    iteration=iter_idx,
                                    list_index=list_idx,
                                    position=pos_idx,
                                )
                            )
        return actions

    def _collect_claim_entries(
        self,
        file_id: str,
        result_data: Dict[str, Any],
    ) -> Dict[str, List[ClaimEntry]]:
        support_claims: List[ClaimEntry] = []
        not_support_claims: List[ClaimEntry] = []

        cache_path = os.path.join(self.cache_dir, f"cache_{file_id}.json")
        cache_data = _load_json(cache_path) if os.path.exists(cache_path) else {}
        cache_claim_map: Dict[str, List[Tuple[int, Optional[int], Optional[int]]]] = defaultdict(list)
        for iter_idx, iter_data in enumerate(cache_data.get("iterations", [])):
            for key, value in iter_data.items():
                if key.startswith("claim_list_") and isinstance(value, list):
                    try:
                        list_idx = int(key.split("_")[-1])
                    except (TypeError, ValueError):
                        list_idx = None
                    for pos_idx, claim_text in enumerate(value):
                        if isinstance(claim_text, str):
                            cache_claim_map[claim_text.strip()].append((iter_idx, list_idx, pos_idx))

        # Chain-of-research claims
        for iter_idx, iter_data in enumerate(result_data.get("chain_of_research_results", [])):
            claim_results = iter_data.get("claim_results", [])
            for claim_idx, claim_obj in enumerate(claim_results):
                final_judgment = claim_obj.get("final_judgment", "Unknown")
                claim_text = claim_obj.get("claim", "").strip()
                claim_obj.pop("nli_scores", None)
                claim_obj.pop("memory_update", None)
                cache_iteration = iter_idx
                cache_list_index: Optional[int] = None
                cache_position: Optional[int] = None
                if cache_claim_map.get(claim_text):
                    cache_iteration, cache_list_index, cache_position = cache_claim_map[claim_text].pop(0)
                entry = ClaimEntry(
                    text=claim_text,
                    iteration=cache_iteration,
                    pointer=claim_obj,
                    section="chain",
                    indices=(iter_idx, claim_idx),
                    original_judgment=final_judgment,
                    judgment=final_judgment,
                    cache_list_index=cache_list_index,
                    cache_position=cache_position,
                )
                if final_judgment == "Support":
                    support_claims.append(entry)
                elif final_judgment == "NotSupport":
                    not_support_claims.append(entry)

        # Report claims
        for report_idx, report_data in enumerate(result_data.get("report_results", [])):
            claim_results = report_data.get("claim_results", [])
            for claim_idx, claim_obj in enumerate(claim_results):
                final_judgment = claim_obj.get("final_judgment", "Unknown")
                claim_text = claim_obj.get("claim", "").strip()
                claim_obj.pop("nli_scores", None)
                claim_obj.pop("memory_update", None)
                entry = ClaimEntry(
                    text=claim_text,
                    iteration="report",
                    pointer=claim_obj,
                    section="report",
                    indices=(report_idx, claim_idx),
                    original_judgment=final_judgment,
                    judgment=final_judgment,
                    cache_list_index=None,
                    cache_position=None,
                )
                if final_judgment == "Support":
                    support_claims.append(entry)
                elif final_judgment == "NotSupport":
                    not_support_claims.append(entry)

        return {
            "support": support_claims,
            "not_support": not_support_claims,
        }

    # ------------------------------------------------------------------
    # Stage 1: NLI relabeling
    # ------------------------------------------------------------------
    def _collect_sc_hc_pairs(
        self, support_claims: List[ClaimEntry], not_support_claims: List[ClaimEntry]
    ) -> List[Dict[str, Any]]:
        pairs = []
        for target in not_support_claims:
            for source in support_claims:
                if not source.text or not target.text:
                    continue
                if tuple(source.order_key) >= tuple(target.order_key):
                    continue
                pairs.append({"source": source, "target": target})
                # print(f"Collected pair - source: {source.text}, position: {source.order_key}, target: {target.text}, position: {target.order_key}")
        return pairs

    def _run_nli_relabeling(
        self,
        pairs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not pairs:
            return []

        logger.info("Running NLI on %d SC->HC pairs", len(pairs))
        claim_chunk_pairs = [(pair["target"].text, pair["source"].text) for pair in pairs]
        nli_results = nli_score_batch_parallel(claim_chunk_pairs)

        entailed_pairs = []
        for pair, scores in zip(pairs, nli_results):
            entail_score = scores.get("entailment", 0.0)
            if entail_score > self.nli_threshold:
                entailed_pairs.append(
                    {
                        "source": pair["source"],
                        "target": pair["target"],
                        "entailment_score": entail_score,
                    }
                )

        logger.info("NLI relabeling identified %d entailed pairs", len(entailed_pairs))
        return entailed_pairs

    def _apply_nli_updates(
        self,
        entailed_pairs: List[Dict[str, Any]],
        support_claims: List[ClaimEntry],
        not_support_claims: List[ClaimEntry],
    ) -> Tuple[List[ClaimEntry], List[ClaimEntry], List[Dict[str, Any]]]:
        if not entailed_pairs:
            return support_claims, not_support_claims, []

        # Keep highest entailment per target
        best_pair_by_target: Dict[ClaimEntry, Dict[str, Any]] = {}
        for pair in entailed_pairs:
            target = pair["target"]
            existing = best_pair_by_target.get(target)
            if not existing or pair["entailment_score"] > existing["entailment_score"]:
                best_pair_by_target[target] = pair

        relabeled_records = []
        updated_supports = support_claims.copy()
        updated_not_supports = []

        for target in not_support_claims:
            best_pair = best_pair_by_target.get(target)
            if not best_pair:
                updated_not_supports.append(target)
                continue

            target.pointer["final_judgment"] = "Support"
            target.pointer["processing_source"] = "Memory_NLI"
            target.pointer["relevant_chunks"] = []
            target.pointer["all_judged_chunks"] = []
            target.pointer.pop("memory_update", None)
            target.pointer.pop("nli_scores", None)

            target.judgment = "Support"
            target.metadata["memory_method"] = "nli"
            target.metadata["memory_entailment_score"] = best_pair["entailment_score"]
            target.metadata["memory_source_claim"] = best_pair["source"].text
            target.metadata["memory_source_iteration"] = best_pair["source"].iteration

            updated_supports.append(target)
            relabeled_records.append(
                {
                    "claim": target.text,
                    "iteration": target.iteration,
                    "method": "nli",
                    "source_claim": best_pair["source"].text,
                    "source_iteration": best_pair["source"].iteration,
                    "entailment_score": best_pair["entailment_score"],
                    "support_claims": [
                        {
                            "text": best_pair["source"].text,
                            "iteration": best_pair["source"].iteration,
                            "entailment_score": best_pair["entailment_score"],
                        }
                    ],
                }
            )

        return updated_supports, updated_not_supports, relabeled_records

    # ------------------------------------------------------------------
    # Stage 2: Similarity selection
    # ------------------------------------------------------------------
    def _select_top_similar_actions(
        self,
        all_actions: List[ActionEntry],
        target_claim: ClaimEntry,
        max_actions: int = DEFAULT_TOP_K,
    ) -> List[Tuple[ActionEntry, float]]:
        """Select top k similar actions for a given target claim (the claim being judged)."""
        if not all_actions:
            return []

        # Filter actions that come before the target claim
        target_order = target_claim.order_key
        valid_actions = [
            action for action in all_actions
            if tuple(action.order_key) <= tuple(target_order)
        ]

        if not valid_actions:
            return []

        # Encode target claim and valid actions
        target_vector = self.encoder.encode([target_claim.text]).astype(np.float32, copy=False)[0]
        action_texts = [action.text for action in valid_actions]
        action_vectors = self.encoder.encode(action_texts).astype(np.float32, copy=False)

        # Compute similarities
        similarities = np.matmul(action_vectors, target_vector)

        # Select top k
        top_k = min(max_actions, len(valid_actions))
        if top_k <= 0:
            return []

        top_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        top_indices_sorted = top_indices[np.argsort(-similarities[top_indices])]

        top_pairs = []
        for idx in top_indices_sorted:
            action = valid_actions[idx]
            score = float(similarities[idx])
            top_pairs.append((action, score))

        return top_pairs

    def _select_top_similar_supports(
        self,
        support_claims: List[ClaimEntry],
        target_claims: List[ClaimEntry],
    ) -> Dict[ClaimEntry, List[Tuple[ClaimEntry, float]]]:
        if not support_claims or not target_claims:
            return {}

        logger.info(
            "Computing similarity for %d NotSupport claims against %d support claims",
            len(target_claims),
            len(support_claims),
        )

        support_by_iteration = [
            claim for claim in support_claims if claim.text.strip()
        ]

        support_texts = [claim.text for claim in support_by_iteration]
        if not support_texts:
            return {}

        support_vectors = self.encoder.encode(support_texts).astype(np.float32, copy=False)

        target_texts = [claim.text for claim in target_claims]
        if not target_texts:
            return {claim: [] for claim in target_claims}
        target_vectors = self.encoder.encode(target_texts).astype(np.float32, copy=False)

        similarity_matrix = self._compute_similarity_matrix(support_vectors, target_vectors)

        selected_supports: Dict[ClaimEntry, List[Tuple[ClaimEntry, float]]] = {}
        for target_idx, target in enumerate(target_claims):
            target_key = target.order_key
            valid_indices = [
                idx for idx, support in enumerate(support_by_iteration)
                if tuple(support.order_key) < tuple(target_key)
            ]

            if not valid_indices:
                selected_supports[target] = []
                continue

            sim_vector = similarity_matrix[valid_indices, target_idx]
            top_k = min(self.similarity_top_k, len(valid_indices))
            if top_k <= 0:
                selected_supports[target] = []
                continue

            top_local = np.argpartition(-sim_vector, top_k - 1)[:top_k]
            top_local_sorted = top_local[np.argsort(-sim_vector[top_local])]

            top_pairs = []
            for local_idx in top_local_sorted:
                support_idx = valid_indices[local_idx]
                source_claim = support_by_iteration[support_idx]
                score = float(sim_vector[local_idx])
                top_pairs.append((source_claim, score))

            selected_supports[target] = top_pairs

        return selected_supports

    def _compute_similarity_matrix(
        self,
        support_vectors: np.ndarray,
        target_vectors: np.ndarray,
    ) -> np.ndarray:
        num_support = support_vectors.shape[0]
        num_targets = target_vectors.shape[0]
        similarities = np.empty((num_support, num_targets), dtype=np.float32)

        available_gpus: List[int] = []
        if (
            torch is not None
            and torch.cuda.is_available()
            and self.similarity_gpu_ids
        ):
            available_gpus = [
                gid
                for gid in self.similarity_gpu_ids
                if 0 <= gid < torch.cuda.device_count()
            ]

        if available_gpus:
            logger.info(
                "Computing similarity on GPUs %s (supports=%d, targets=%d)",
                available_gpus,
                num_support,
                num_targets,
            )
            support_tensors = {}
            for gid in available_gpus:
                device = torch.device(f"cuda:{gid}")
                support_tensors[gid] = torch.from_numpy(support_vectors).to(
                    device=device, dtype=torch.float32
                )

            target_indices = np.arange(num_targets)
            chunks = np.array_split(target_indices, len(available_gpus))

            for chunk_indices, gid in zip(chunks, available_gpus):
                if chunk_indices.size == 0:
                    continue
                device = torch.device(f"cuda:{gid}")
                support_tensor = support_tensors[gid]
                target_tensor = torch.from_numpy(target_vectors[chunk_indices]).to(
                    device=device, dtype=torch.float32
                )
                sim_chunk = torch.matmul(support_tensor, target_tensor.T).cpu().numpy()
                similarities[:, chunk_indices] = sim_chunk
            return similarities

        logger.info(
            "Computing similarity on CPU (supports=%d, targets=%d)",
            num_support,
            num_targets,
        )
        cpu_result = np.matmul(support_vectors, target_vectors.T)
        return cpu_result.astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Stage 3: LLM evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_llm_prompt(
        query: str,
        candidate: Dict[str, Any],
        support_context: str,
        action_context: str = "",
    ) -> Tuple[str, str]:
        scenario_header = (
            "Scenario: You are validating reasoning steps produced during a deep research workflow. "
            "Each reasoning claim may be either a direct observation from browsing or a reflection "
            "that interprets previous observations. The overall research query describes the macro "
            "task that the agent is trying to solve; you must refer to this macro query as context when "
            "deciding whether a reflection is reasonable.\n"
        )
        instructions = (
            "You are given the overall research query, a candidate reasoning claim produced later in the workflow, "
            "a set of earlier supported claims that act as trusted memory, and relevant actions taken during the research process.\n\n"
            "Decision rules:\n"
            "- If ANY supported claim explicitly states or strongly implies the candidate claim, label it Support.\n"
            "- If the candidate is a reasonable reflection, synthesis, hypothesis, or next-step plan that naturally follows from the memory, actions, or the macro research query—even with probabilistic language—label it Support. This includes cases where it reasonably judges that some action, information, or tool is necessary/needed to complete the main task, even if no single memory sentence states that necessity explicitly.\n"
            "- Consider the context provided by relevant actions: if actions taken align with or support the reasoning in the candidate claim, this strengthens the case for Support.\n"
            "- If the candidate states universally accepted common knowledge (e.g., \"Sora is developed by OpenAI\" or \"Rosaceae is a flowering plant family, not an insect order\"), you may label it Support even when the memory omits it.\n"
            "- Only label NotSupport when the supplied memory and actions provide no justification or when they clearly conflict with the candidate claim.\n\n"
            "Examples:\n"
            "Example A:\n"
            "  Supported claims:\n"
            "    • The archive contains a 2015 blog post titled \"Mapping the Poet's Journey\".\n"
            "  Relevant actions:\n"
            "    • Search for blog posts published between 2015-2017\n"
            "  Candidate claim: The blog post title reveals the poet's research direction.\n"
            "  Output: {\"final_judgment\": \"Support\", \"confidence\": 0.8, \"explanation\": \"The title already highlights the focus, and the action context shows systematic search.\"}\n"
            "Example B (Reflection):\n"
            "  Supported claims:\n"
            "    • The team plans to review the poet's bibliography for clues.\n"
            "  Relevant actions:\n"
            "    • Investigate poet's published works\n"
            "    • Review bibliography entries\n"
            "  Candidate claim: Reviewing the bibliography will help uncover the poet's identity.\n"
            "  Output: {\"final_judgment\": \"Support\", \"confidence\": 0.9, \"explanation\": \"This reflection is a logical next step given the plan and actions taken.\"}\n"
            "Example C (Negative):\n"
            "  Supported claims:\n"
            "    • Searches returned no evidence that the blogger hosted events.\n"
            "  Relevant actions:\n"
            "    • Search for event listings\n"
            "    • Check blogger's event history\n"
            "  Candidate claim: The blogger organized a summer workshop.\n"
            "  Output: {\"final_judgment\": \"NotSupport\", \"confidence\": 0.9, \"explanation\": \"Memory and actions show no workshop evidence despite searches.\"}\n"
            "{\n"
            '  \"final_judgment\": \"Support\" | \"NotSupport\",\n'
            '  \"confidence\": float between 0 and 1,\n'
            '  \"explanation\": \"One sentence explaining your decision.\"\n'
            "}\n"
        )

        supports_block = support_context.strip() if support_context.strip() else "None available."
        actions_block = action_context.strip() if action_context.strip() else "None available."

        user_prompt = (
            f"{scenario_header}\n"
            f"Research Query:\n{query}\n\n"
            f"Candidate Claim (iteration={candidate.get('iteration')}):\n{candidate.get('text')}\n\n"
            f"Previously Supported Claims:\n{supports_block}\n\n"
        )
        
        if actions_block != "None available.":
            user_prompt += f"Relevant Actions Taken:\n{actions_block}\n\n"
        
        user_prompt += f"{instructions}"

        system_prompt = (
            "You are an expert fact-checking assistant specialized in multi-step research chains. "
            "Follow the user's instructions carefully and respond with valid JSON only."
        )
        return system_prompt, user_prompt

    @staticmethod
    def _parse_llm_json(content: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            fallback = _extract_fields_from_text(content)
            if fallback:
                judgment_raw = fallback.get("judgment", "neutral").lower()
                final_judgment = "Support" if judgment_raw == "entailed" else "NotSupport"
                confidence = fallback.get("confidence", 0.0)
                if not isinstance(confidence, (int, float)):
                    confidence = 0.0
                confidence = max(0.0, min(1.0, float(confidence)))
                explanation = fallback.get("explanation", "")
                explanation = explanation.strip() if isinstance(explanation, str) else ""
                return {
                    "final_judgment": final_judgment,
                    "confidence": confidence,
                    "explanation": explanation,
                }
            return {
                "final_judgment": "NotSupport",
                "confidence": 0.0,
                "explanation": "Failed to parse LLM response.",
            }

        if not isinstance(parsed, dict):
            return {
                "final_judgment": "NotSupport",
                "confidence": 0.0,
                "explanation": "LLM response was not a JSON object.",
            }

        final_judgment_raw = parsed.get("final_judgment")
        if isinstance(final_judgment_raw, str):
            normalized = final_judgment_raw.strip().lower()
            if normalized in {"support"}:
                final_judgment = "Support"
            elif normalized in {"not support", "not_support"}:
                final_judgment = "NotSupport"
            elif normalized in {"entailed"}:
                final_judgment = "Support"
            else:
                final_judgment = "NotSupport"
        else:
            final_judgment = "NotSupport"

        confidence_raw = parsed.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        explanation_raw = parsed.get("explanation", "")
        explanation = explanation_raw.strip() if isinstance(explanation_raw, str) else ""

        return {
            "final_judgment": final_judgment,
            "confidence": confidence,
            "explanation": explanation,
        }

    def _run_llm_stage(
        self,
        query: str,
        remaining_claims: List[ClaimEntry],
        support_mapping: Dict[ClaimEntry, List[Tuple[ClaimEntry, float]]],
        all_actions: List[ActionEntry],
    ) -> List[Dict[str, Any]]:
        if not remaining_claims:
            return []

        tasks: List[Tuple[int, Dict[str, Any], str, str, str]] = []
        claim_map: Dict[int, ClaimEntry] = {}
        support_payload_lookup: Dict[int, List[Dict[str, Any]]] = {}
        action_payload_lookup: Dict[int, List[Dict[str, Any]]] = {}
        for idx, claim in enumerate(remaining_claims):
            supports = support_mapping.get(claim, [])
            
            # Process even if no support claims found (can still use actions and query)
            supports_payload = [
                {
                    "text": support_claim.text,
                    "iteration": support_claim.iteration,
                    "similarity": score,
                }
                for support_claim, score in supports
            ]
            support_payload_lookup[idx] = supports_payload
            support_context_lines = []
            for line_idx, (support_claim, _) in enumerate(supports, start=1):
                support_context_lines.append(
                    f"[{line_idx}] {support_claim.text}"
                )
            support_context = "\n\n".join(support_context_lines) if support_context_lines else ""
            
            # Collect top k actions similar to the claim being judged
            action_pairs = self._select_top_similar_actions(all_actions, claim, self.similarity_top_k)
            unique_actions = [(action, score) for action, score in action_pairs]
            actions_payload = [
                {
                    "text": action.text,
                    "iteration": action.iteration,
                    "similarity": score,
                }
                for action, score in unique_actions
            ]
            action_payload_lookup[idx] = actions_payload
            
            action_context_lines = []
            for line_idx, (action, _) in enumerate(unique_actions, start=1):
                action_context_lines.append(
                    f"[{line_idx}] {action.text}"
                )
            action_context = "\n\n".join(action_context_lines) if action_context_lines else ""
            
            candidate_payload = {"text": claim.text, "iteration": claim.iteration}
            tasks.append((idx, candidate_payload, support_context, action_context, query))
            claim_map[idx] = claim

        distributed = self.api_load_balancer.distribute_work_evenly(tasks)
        worker_args: List[Tuple[str, List[Tuple[int, Dict[str, Any], str, str, str]]]] = []
        for api_id, chunk in enumerate(distributed):
            if not chunk:
                continue
            api_key = API_KEYS[api_id % len(API_KEYS)]
            worker_args.append((api_key, chunk))

        if not worker_args:
            return []

        workers = min(self.llm_workers, len(worker_args))
        logger.info("Running LLM memory judgment with %d workers", workers)

        with mp.Pool(processes=workers) as pool:
            worker_results = pool.map(_memory_llm_worker, worker_args)

        llm_records = []
        for worker_output in worker_results:
            for claim_id, support_context, action_context, llm_result in worker_output:
                claim_entry = claim_map.get(claim_id)
                if claim_entry is None:
                    continue

                claim_entry.pointer.pop("memory_update", None)
                claim_entry.pointer.pop("nli_scores", None)
                claim_entry.metadata["memory_llm"] = {
                    "method": "Memory_LLM",
                    "llm_result": llm_result,
                    "llm_supports_context": support_context,
                    "llm_supports_used": support_payload_lookup.get(claim_id, []),
                    "llm_actions_context": action_context,
                    "llm_actions_used": action_payload_lookup.get(claim_id, []),
                }
                claim_entry.metadata["memory_llm_result"] = llm_result

                if llm_result["final_judgment"] == "Support":
                    claim_entry.pointer["final_judgment"] = "Support"
                    claim_entry.pointer["processing_source"] = "Memory_LLM"
                    claim_entry.pointer["relevant_chunks"] = []
                    claim_entry.pointer["all_judged_chunks"] = []
                    claim_entry.judgment = "Support"
                    claim_entry.metadata["memory_method"] = "llm"
                    llm_records.append(
                        {
                            "claim": claim_entry.text,
                            "iteration": claim_entry.iteration,
                            "method": "llm",
                            "confidence": llm_result.get("confidence", 0.0),
                            "explanation": llm_result.get("explanation", ""),
                            "support_claims": support_payload_lookup.get(claim_id, [])[:10],
                            "actions_used": action_payload_lookup.get(claim_id, [])[:10],
                        }
                    )
        return llm_records

    # ------------------------------------------------------------------
    # Full pipeline per file
    # ------------------------------------------------------------------
    def process_single_file(self, file_id: str) -> Dict[str, Any]:
        # skip it if the file has already been processed
        result_path = os.path.join(self.results_dir, f"{file_id}_combined.json")
        # result_path = os.path.join(self.results_dir, f"results_{file_id}.json")
        cache_path = os.path.join(self.cache_dir, f"cache_{file_id}.json")

        if not os.path.exists(result_path):
            logger.warning("Result file missing: %s", result_path)
            return {}
        if not os.path.exists(cache_path):
            logger.warning("Cache file missing: %s", cache_path)

        result_data = _load_json(result_path)

        chain_of_research_results = result_data.get("chain_of_research_results", [])
        for iteration in chain_of_research_results:
            for claim_result in iteration.get("claim_results", []):
                # As long as there is one claim with "processing_source": "Memory_LLM" or "Memory_NLI", skip the memory judgment
                if claim_result.get("processing_source") in ["Memory_LLM", "Memory_NLI"]:
                    return {
                        "file_id": file_id,
                        "total_support": -1,
                        "total_not_support": 0,
                        "total_relabeled": 0,
                        "relabeled_records": [],
                    }

        for report_result in result_data.get("report_results", []):
            for claim_result in report_result.get("claim_results", []):
                if claim_result.get("processing_source") in ["Memory_LLM", "Memory_NLI"]:
                    return {
                        "file_id": file_id,
                        "total_support": -1,
                        "total_not_support": 0,
                        "total_relabeled": 0,
                        "relabeled_records": [],
                    }

        claims = self._collect_claim_entries(file_id, result_data)
        support_claims = claims["support"]
        not_support_claims = claims["not_support"]
        
        # Collect all actions from cache
        all_actions = self._collect_action_entries(file_id)

        logger.info(
            "[%s] Loaded %d support claims, %d NotSupport claims, %d actions",
            file_id,
            len(support_claims),
            len(not_support_claims),
            len(all_actions),
        )

        relabel_log = []

        # Stage 1: NLI relabeling
        sc_hc_pairs = self._collect_sc_hc_pairs(support_claims, not_support_claims)
        nli_entailed = self._run_nli_relabeling(sc_hc_pairs)
        support_claims, remaining_not_support, nli_records = self._apply_nli_updates(
            nli_entailed,
            support_claims,
            not_support_claims,
        )
        relabel_log.extend(nli_records)

        # Stage 2: Similarity + Stage 3 LLM
        similarity_mapping = self._select_top_similar_supports(support_claims, remaining_not_support)

        llm_records = self._run_llm_stage(
            result_data.get("query", ""),
            remaining_not_support,
            similarity_mapping,
            all_actions,
        )
        relabel_log.extend(llm_records)

        # Write updated result file in-place with deep-merge to preserve other attributes
        output_path = os.path.join(self.output_dir, f"{file_id}_combined.json")
        try:
            if os.path.exists(output_path):
                existing_on_disk = _load_json(output_path)
                merged = _deep_merge(existing_on_disk, result_data)
            else:
                merged = result_data
        except Exception:
            merged = result_data
        _save_json(merged, output_path)

        print(f"Completed memory judgment: {file_id} - {len(relabel_log)} claims relabeled")

        return {
            "file_id": file_id,
            "total_support": len(support_claims),
            "total_not_support": len(not_support_claims),
            "total_relabeled": len(relabel_log),
            "relabeled_records": relabel_log,
        }

    # ------------------------------------------------------------------
    def process_all(self) -> List[Dict[str, Any]]:
        files = [
            filename.replace("_combined.json", "")
            for filename in os.listdir(self.results_dir)
            if filename.endswith("_combined.json")
        ]
        files.sort()

        summary = []
        for file_id in files:
            # skip it if the file has already been processed
            output_path = os.path.join(self.output_dir, f"{file_id}_combined.json")
            if os.path.exists(output_path):
                logger.info(f"Skipping file: {file_id} - already processed")
                continue
            print(f"Processing file: {file_id}")
            result = self.process_single_file(file_id)
            if result:
                summary.append(result)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge NotSupport claims against memory.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/browsecomp/gemini/after_update",
        help="Directory containing *_combined.json result files.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/json_cache/browsecomp/gemini/after_update",
        help="Directory containing cache_*.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/browsecomp/gemini/after_memory_update",
        help="Directory to write updated result files.",
    )
    parser.add_argument(
        "--nli_threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Entailment threshold for NLI relabeling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top similar supports to select per claim.",
    )
    parser.add_argument(
        "--llm_workers",
        type=int,
        default=64,
        help="Number of parallel workers for LLM judgment.",
    )
    parser.add_argument(
        "--similarity_gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for similarity computation (e.g., '0,1').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    similarity_gpu_ids = None
    if args.similarity_gpus:
        similarity_gpu_ids = [
            int(item.strip())
            for item in args.similarity_gpus.split(",")
            if item.strip()
        ]
    judge = MemoryJudge(
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        nli_threshold=args.nli_threshold,
        similarity_top_k=args.top_k,
        llm_workers=args.llm_workers,
        similarity_gpu_ids=similarity_gpu_ids,
    )
    summary = judge.process_all()
    summary_path = os.path.join(args.output_dir, "memory_judgment_summary.json")
    _save_json({"files": summary}, summary_path)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()


