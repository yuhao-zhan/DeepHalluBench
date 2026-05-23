"""
DeepHalluBench: Hallucination Evaluation for Deep Research Agents

A process-aware evaluation framework for detecting and quantifying hallucinations
in full research trajectories using the PING Taxonomy.

PING Taxonomy:
- P: Propagation (Trajectory-Level) — cascading errors from earlier hallucinations
- I: Intent (Query-Level) — restriction neglect + deviated actions
- N: Noise-induced (Context-Level) — failure to prioritize informative evidence
- G: Grounding (Source-Level) — fabrication + misattribution

Example usage:
    from eval_framework import DeepHalluBench

    evaluator = DeepHalluBench.from_env()
    results = evaluator.evaluate("trajectory.json", "query")
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Enable offline mode for HuggingFace models (use locally cached models only)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Import configuration
from .config import EvaluationConfig, ConfigManager

# Import parser (add parent directory to path for absolute import)
import sys
from pathlib import Path
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from parsers import JSONTrajectoryParser
except ImportError:
    # Fallback for when package is installed
    from eval_framework.parsers import JSONTrajectoryParser

__version__ = "1.0.0"
__all__ = ["DeepHalluBench", "EvaluationConfig", "ConfigManager"]


class DeepHalluBench:
    """
    Main evaluator for DeepHalluBench.

    Evaluates research trajectories for hallucinations across the PING taxonomy:
    - Grounding (G): Fabrication + Misattribution
    - Noise-induced (N): Failure to prioritize informative evidence
    - Intent (I): Restriction neglect + Action deviation
    - Propagation (P): Cascading errors from earlier hallucinations

    Example:
        evaluator = DeepHalluBench.from_env()
        results = evaluator.evaluate("trajectory.json", "Your research query")
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.parser = JSONTrajectoryParser()

        # Initialize components lazily
        self._decomposition = None
        self._claim_verifier = None
        self._action_checker = None
        self._noise_detector = None
        self._constraint_checker = None

        # Ensure directories exist
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "DeepHalluBench":
        """Create evaluator from environment variables."""
        config = ConfigManager.from_env()
        return cls(config)

    @classmethod
    def from_file(cls, config_path: str) -> "DeepHalluBench":
        """Create evaluator from a configuration file."""
        config = ConfigManager.from_file(config_path)
        return cls(config)

    @property
    def decomposition(self):
        """Lazy load decomposition module."""
        if self._decomposition is None:
            from .core.decomposition import TrajectoryDecomposer
            self._decomposition = TrajectoryDecomposer(self.config)
        return self._decomposition

    @property
    def claim_verifier(self):
        """Lazy load claim verification module."""
        if self._claim_verifier is None:
            from .core.claim_verification import ClaimVerifier
            self._claim_verifier = ClaimVerifier(self.config)
        return self._claim_verifier

    @property
    def action_checker(self):
        """Lazy load action checking module."""
        if self._action_checker is None:
            from .core.action_checking import ActionChecker
            self._action_checker = ActionChecker(self.config)
        return self._action_checker

    @property
    def noise_detector(self):
        """Lazy load noise domination detection module."""
        if self._noise_detector is None:
            from .core.noise_domination import NoiseDominationDetector
            self._noise_detector = NoiseDominationDetector(self.config)
        return self._noise_detector

    @property
    def constraint_checker(self):
        """Lazy load constraint checking module."""
        if self._constraint_checker is None:
            from .core.constraint_checking import ConstraintChecker
            self._constraint_checker = ConstraintChecker(self.config)
        return self._constraint_checker

    def evaluate(
        self,
        trajectory: Union[str, Dict[str, Any]],
        query: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_claims: Optional[int] = None,
        max_actions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a research trajectory for hallucinations.

        Args:
            trajectory: Path to trajectory JSON file OR dict with trajectory data
            query: User query (optional if provided in trajectory or config)
            output_dir: Directory to save results
            max_claims: If set, sample only this many claims for verification (demo mode)
            max_actions: If set, sample only this many actions for verification (demo mode)

        Returns:
            Dictionary containing:
            - ping_scores: Dict with scores for each PING category
            - overall_score: Combined hallucination score
            - detailed_results: Full evaluation results
        """
        print("=" * 80)
        print("DeepHalluBench Evaluation")
        print("=" * 80)

        # Parse trajectory
        if isinstance(trajectory, str):
            print(f"\n📂 Loading trajectory from: {trajectory}")
            parsed = self.parser.parse(trajectory)
            trajectory_path = trajectory
        elif hasattr(trajectory, 'query') and hasattr(trajectory, 'final_report'):
            # Already a ParsedTrajectory object
            print(f"\n📂 Using provided ParsedTrajectory object")
            parsed = trajectory
            trajectory_path = "memory"
        else:
            print(f"\n📂 Using provided trajectory data")
            parsed = self._dict_to_parsed_trajectory(trajectory)
            trajectory_path = "memory"

        # Use query from trajectory if not provided
        if query is None:
            query = parsed.query
        if not query:
            raise ValueError("Query is required but not found in trajectory or arguments")

        print(f"📝 Query: {query[:100]}..." if len(query) > 100 else f"📝 Query: {query}")
        print(f"📊 Report length: {len(parsed.final_report)} characters")
        print(f"🔗 Source links: {len(parsed.all_source_links)}")
        print(f"📚 Citations: {len(parsed.summary_citations)}")

        # Determine output directory
        if output_dir is None:
            output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up caches: json_cache (decomposition only), web_content_cache, results
        basename = Path(trajectory_path).stem if trajectory_path != "memory" else "memory"
        json_cache_dir = os.path.join(self.config.cache_dir, "json_cache")
        web_cache_dir = os.path.join(self.config.cache_dir, "web_content_cache")
        os.makedirs(json_cache_dir, exist_ok=True)
        os.makedirs(web_cache_dir, exist_ok=True)
        json_cache_file = os.path.join(json_cache_dir, f"cache_{basename}.json")

        # Load existing json_cache (decomposition data only — 6 keys)
        json_cache = {}
        if os.path.exists(json_cache_file):
            try:
                with open(json_cache_file, 'r', encoding='utf-8') as f:
                    json_cache = json.load(f)
                print(f"  📁 Loaded json_cache: {json_cache_file}")
            except Exception:
                pass

        # Step 1: Decompose trajectory into atomic units
        print("\n[1/7] Decomposing trajectory into atomic units...")
        if json_cache and "query_list" in json_cache:
            print("  📁 Loading decomposition from json_cache")
            decomposed = json_cache
            # json_cache only has 6 keys; reconstruct convenience fields for verifiers
            if "chunks" not in decomposed:
                decomposed["chunks"] = self._extract_chunks_from_cache(decomposed)
            if "all_actions" not in decomposed:
                decomposed["all_actions"] = self._extract_actions_from_cache(decomposed)
            if "all_claims" not in decomposed:
                decomposed["all_claims"] = self._extract_claims_from_cache(decomposed)
        else:
            try:
                decomposed = self.decomposition.decompose(parsed, self.config.cache_dir)
            except Exception as e:
                print(f"⚠️ Decomposition error: {e}")
                decomposed = self._fallback_decomposition(parsed)
            # Save ONLY the 6 original keys to json_cache
            json_cache = {
                "query_list": decomposed.get("query_list", []),
                "iterations": decomposed.get("iterations", []),
                "report": decomposed.get("report", []),
                "chunk_score": decomposed.get("chunk_score", {}),
                "related_query": decomposed.get("related_query", {}),
                "top_k_chunks": decomposed.get("top_k_chunks", {}),
            }
            try:
                with open(json_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(json_cache, f, ensure_ascii=False, indent=2)
                print(f"  💾 Saved json_cache: {json_cache_file}")
            except Exception:
                pass

        # Step 1.5: Load/fetch web content for chunk-level scoring
        print("\n[1.5/7] Loading web content cache...")
        all_urls = parsed.all_source_links
        from .utils.web_fetcher import WebContentFetcher
        fetcher = WebContentFetcher(cache_dir=self.config.cache_dir)
        web_content = fetcher.fetch(all_urls, basename)

        # Replace bare-URL chunk texts with fetched web content
        for chunk in decomposed.get("chunks", []):
            text = chunk.get("text", "").strip()
            if text.startswith("http://") or text.startswith("https://"):
                if text in web_content and not web_content[text].startswith("[Error]"):
                    chunk["text"] = web_content[text]

        # Step 1.75: Compute related_query (claim→sub-query mapping) if not in cache
        print("\n[1.75/7] Computing claim→sub-query mapping (related_query)...")
        if not decomposed.get("related_query") or not any(decomposed["related_query"].values()):
            related_query = self._compute_related_query(decomposed)
            decomposed["related_query"] = related_query
            # Persist to json_cache
            if os.path.exists(json_cache_file):
                try:
                    with open(json_cache_file, 'r', encoding='utf-8') as f:
                        jc = json.load(f)
                    jc["related_query"] = related_query
                    with open(json_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(jc, f, ensure_ascii=False, indent=2)
                    print(f"  💾 Saved related_query ({len(related_query)} entries) to json_cache")
                except Exception:
                    pass
        else:
            print(f"  📁 Using cached related_query ({len(decomposed['related_query'])} entries)")

        # Normalize decomposed data for verifiers
        # Verifiers expect: actions, claims, sub_queries, chunks, report_claims
        verifier_data = self._normalize_for_verifiers(decomposed)

        # Steps 2-6: Verification — results saved only to results/, NOT to json_cache
        # Claim verification runs FIRST (reference order) so Claim Memory is available for action checking
        print("\n[2/7] Verifying claims (Grounding/Explicit Summarization)...")
        try:
            claim_results = self.claim_verifier.verify(verifier_data, query, max_claims=max_claims)
        except Exception as e:
            print(f"⚠️ Claim verification error: {e}")
            claim_results = self._empty_claim_results()

        # Persist top_k_chunks to json_cache only (not in output results)
        top_k_chunks = claim_results.pop("_top_k_chunks", claim_results.pop("top_k_chunks", {}))
        if top_k_chunks:
            decomposed["top_k_chunks"] = top_k_chunks
            if os.path.exists(json_cache_file):
                try:
                    with open(json_cache_file, 'r', encoding='utf-8') as f:
                        jc = json.load(f)
                    jc["top_k_chunks"] = top_k_chunks
                    with open(json_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(jc, f, ensure_ascii=False, indent=2)
                    print(f"  💾 Saved top_k_chunks ({len(top_k_chunks)} claims) to json_cache")
                except Exception:
                    pass

        # Pass Claim Memory (entailed claims) to action checker (reference: build_entailed_claim_memory)
        claim_memory = claim_results  # full claim results for Claim Memory context

        print("\n[3/7] Verifying actions (Intent/Explicit Planning)...")
        try:
            action_results = self.action_checker.verify(verifier_data, query, max_actions=max_actions, claim_memory=claim_memory)
        except Exception as e:
            print(f"⚠️ Action checking error: {e}")
            action_results = self._empty_action_results()

        print("\n[4/7] Checking constraints (Intent/Implicit Planning)...")
        try:
            constraint_results = self.constraint_checker.check(verifier_data, query)
        except Exception as e:
            print(f"⚠️ Constraint checking error: {e}")
            constraint_results = self._empty_constraint_results()

        print("\n[5/7] Detecting noise domination (Implicit Summarization)...")
        try:
            noise_results = self.noise_detector.detect(verifier_data, claim_results)
        except Exception as e:
            print(f"⚠️ Noise detection error: {e}")
            noise_results = self._empty_noise_results()

        print("\n[6/7] Detecting propagation hallucinations...")
        propagation_results = self._detect_propagation(verifier_data, claim_results)
        # Use action checker's propagation count (paper: derived from action verification)
        propagation_results["propagation_count"] = action_results.get("propagation", 0)
        total_acts = max(action_results.get("total", 0), 1)
        propagation_results["propagation_score"] = propagation_results["propagation_count"] / total_acts

        # Compute PING scores
        ping_scores = self._compute_ping_scores(
            action_results, constraint_results, claim_results, noise_results, propagation_results
        )

        overall_score = sum(ping_scores.values()) / len(ping_scores) if ping_scores else 0.0

        # Build action_results with iteration grouping (reference format)
        action_results_by_iter = {}
        all_actions_data = verifier_data.get("actions", [])
        for i, act in enumerate(action_results.get("results", [])):
            it = act.get("iteration", act.get("iteration_num", 0))
            if it not in action_results_by_iter:
                action_results_by_iter[it] = []
            action_results_by_iter[it].append(act)
        action_results_formatted = [
            {
                "iteration_num": it,
                "action_results": sorted(action_results_by_iter[it], key=lambda x: x.get("iteration", 0)),
            }
            for it in sorted(action_results_by_iter.keys())
        ]

        # Build claim→iteration and iteration→search_urls mappings (reference collect_claims_and_urls)
        claim_to_iteration = {}  # claim_text → {"iter": int, "source_type": str}
        iteration_search_urls = {}  # iteration_index → cumulative search URLs
        all_seen_urls = []
        iterations = decomposed.get("iterations", [])
        for i, it in enumerate(iterations):
            iter_idx = i + 1
            sl = it.get(f"search_list_{iter_idx}", [])
            # Cumulative: add previous iterations' search URLs (reference pattern)
            cumulative = list(sl)
            for prev_url in all_seen_urls:
                if prev_url not in cumulative:
                    cumulative.append(prev_url)
            iteration_search_urls[iter_idx] = cumulative
            for url in sl:
                if url not in all_seen_urls:
                    all_seen_urls.append(url)
            # Map claims in this iteration
            for claim_text in it.get(f"claim_list_{iter_idx}", []):
                claim_to_iteration[claim_text] = {"iter": iter_idx, "source_type": "iteration"}
        # Also map report claims with their paragraph's atomic_claims for find_target_url
        report_claim_to_para = {}  # claim_text → atomic_claims list from its paragraph
        for para in decomposed.get("report", []):
            acs = para.get("atomic_claims", [])
            for claim_text in acs:
                claim_to_iteration[claim_text] = {"iter": len(iterations) + 1, "source_type": "report"}
                report_claim_to_para[claim_text] = acs
        # Summary citations as fallback for report claims (reference: summary_citations)
        summary_citations = parsed.summary_citations if hasattr(parsed, 'summary_citations') else []

        # Build chain_of_research_results from claim results (reference format)
        chain_claim_results = []
        for i, r in enumerate(claim_results.get("results", [])):
            source = r.get("source", "nli")
            # Only exact nli/nli_concat = high confidence; nli_adaptive/llm/llm_adaptive = LLM
            if source in ("nli", "nli_concat"):
                processing_source = "NLI_HIGH_CONFIDENCE"
                high_confidence_skip = True
            else:
                processing_source = "LLM"
                high_confidence_skip = False

            # Look up claim metadata from decomposition
            claim_text = r["claim"]
            claim_meta = claim_to_iteration.get(claim_text, {})
            src_type = claim_meta.get("source_type", "iteration")
            iter_idx = claim_meta.get("iter", i + 1)
            if src_type == "iteration":
                target_urls = iteration_search_urls.get(iter_idx, [])
            else:
                # Report claim: use find_target_url (reference: looks for citation URLs nearby in paragraph)
                para_claims = report_claim_to_para.get(claim_text, [])
                target_urls = self._find_target_urls_for_report_claim(claim_text, para_claims)
                if not target_urls:
                    target_urls = summary_citations  # fallback: summary citations (reference)

            entry = {
                "claim": claim_text,
                "final_judgment": r["judgment"],
                "confidence": r["confidence"],
                "source_type": src_type,
                "iteration_index": iter_idx,
                "target_urls": target_urls,
                "total_tokens_used": 0,
                "processing_source": processing_source,
                "high_confidence_skip": high_confidence_skip,
                "relevant_chunks": r.get("relevant_chunks", []),
                "all_judged_chunks": r.get("all_judged_chunks", []),
                "explanation": r.get("explanation", ""),
            }

            # If LLM was the final decider, update relevant_chunks judgment to match
            if processing_source == "LLM":
                for rc in entry["relevant_chunks"]:
                    rc["judgment"] = entry["final_judgment"]
                    rc["confidence"] = entry["confidence"]

            # Add similarity_score and sentence_idx to relevant_chunks if missing
            for rc in entry["relevant_chunks"]:
                rc.setdefault("similarity_score", 0.0)
                rc.setdefault("sentence_idx", 0)

            # Add similarity_score and sentence_idx to all_judged_chunks if missing
            for ajc in entry["all_judged_chunks"]:
                ajc.setdefault("similarity_score", 0.0)
                ajc.setdefault("sentence_idx", 0)

            chain_claim_results.append(entry)

        chain_of_research_results = [{"claim_results": chain_claim_results}] if chain_claim_results else []

        # Build noise_analysis from noise_results
        noise_analysis = {
            "score": noise_results.get("score", 0.0),
            "mapped_clusters": noise_results.get("mapped_clusters", []),
            "unmapped_clusters": noise_results.get("unmapped_clusters", []),
            "details": noise_results.get("details", {}),
        }

        # Compile results with reference-aligned format
        final_results = {
            "query": query,
            "ping_scores": ping_scores,
            "overall_score": overall_score,
            "chain_of_research_results": chain_of_research_results,
            "action_results": action_results_formatted,
            "noise_analysis": noise_analysis,
            "detailed_results": {
                "action_verification": action_results,
                "constraint_checking": constraint_results,
                "claim_verification": claim_results,
                "noise_domination": noise_results,
                "propagation": propagation_results,
            },
            "metadata": {
                "trajectory_path": trajectory_path,
                "config": self.config.to_dict(),
            }
        }

        # Save results
        output_file = os.path.join(output_dir, f"evaluation_{Path(trajectory_path).stem}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved to: {output_file}")

        # Clean up per-module cache files (unified cache in json_cache/ is the source of truth)
        self._cleanup_module_caches()

        # Print summary
        self._print_summary(ping_scores, overall_score)

        return final_results

    def _dict_to_parsed_trajectory(self, data: Dict[str, Any]):
        """Convert a dict to ParsedTrajectory."""
        # Import ParsedTrajectory with proper path handling
        import sys
        from pathlib import Path
        _parent = Path(__file__).parent.parent
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        from parsers.base_parser import ParsedTrajectory
        return ParsedTrajectory(
            query=data.get("query", ""),
            final_report=data.get("final_report", ""),
            all_source_links=data.get("all_source_links", []),
            summary_citations=data.get("summary_citations", []),
            chain_of_research=data.get("chain_of_research", {}),
            metadata=data.get("metadata", {}),
        )

    def _normalize_for_verifiers(self, decomposed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new cache format to what verifiers expect."""
        # Extract all claims: from iterations + from report paragraphs
        all_claims = list(decomposed.get("all_claims", []))
        for para in decomposed.get("report", []):
            for claim_text in para.get("atomic_claims", []):
                all_claims.append({"claim": claim_text, "source": "report"})

        return {
            "actions": decomposed.get("all_actions", []),
            "claims": all_claims,
            "report_claims": [{"claim": c, "source": "report"}
                              for rp in decomposed.get("report", [])
                              for c in rp.get("atomic_claims", [])],
            "sub_queries": decomposed.get("query_list", []),
            "chunks": decomposed.get("chunks", []),
            "iterations": decomposed.get("iterations", []),
            "related_query": decomposed.get("related_query", {}),
            "chunk_score": decomposed.get("chunk_score", {}),
        }

    def _find_target_urls_for_report_claim(self, claim: str, atomic_claims: List[str]) -> List[str]:
        """Find target URLs for a report claim (reference: find_target_url in utils.py).

        Report claims may have inline citations (@https://...) or be adjacent to citation claims.
        Reference looks backward then forward for URL-formatted claims; also checks inline citations.
        """
        if not atomic_claims or claim not in atomic_claims:
            return []
        import re as _re
        target_urls = []
        claim_index = atomic_claims.index(claim)

        # Helper: extract URL from @https://... inline citation or markdown/pure URL
        def _extract(text: str) -> str:
            md = _re.search(r'\[.*?\]\((https?://[^\s\)]+)\)', text)
            if md: return md.group(1)
            pure = _re.match(r'^(https?://[^\s]+)$', text.strip())
            if pure: return pure.group(1).rstrip('.,;:!?\'\")')
            inline = _re.search(r'@?(https?://[^\s\)\'\"\]\[]+)', text)
            if inline: return inline.group(1).rstrip('.,;:')
            return ''

        # Look backwards for nearest URL neighbor
        for j in range(claim_index - 1, -1, -1):
            url = _extract(atomic_claims[j])
            if url:
                url = url.rstrip('/')
                if url not in target_urls:
                    target_urls.append(url)
                break
        # Look forwards for nearest URL neighbor
        for j in range(claim_index + 1, len(atomic_claims)):
            url = _extract(atomic_claims[j])
            if url:
                url = url.rstrip('/')
                if url not in target_urls:
                    target_urls.append(url)
                break
        # Also extract inline citation from the claim itself
        own_url = _extract(claim)
        if own_url:
            own_url = own_url.rstrip('/')
            if own_url not in target_urls:
                target_urls.append(own_url)
        return target_urls

    def _fallback_decomposition(self, parsed) -> Dict[str, Any]:
        """Fallback decomposition when main decomposition fails."""
        return {
            "query_list": [parsed.query],
            "iterations": [],
            "report": [],
            "all_actions": [],
            "all_claims": [],
            "chunks": [],
            "chunk_score": {},
            "related_query": {},
            "top_k_chunks": {},
        }

    def _empty_action_results(self) -> Dict[str, Any]:
        return {"total": 0, "deviation": 0, "propagation": 0, "results": []}

    def _empty_constraint_results(self) -> Dict[str, Any]:
        return {"total_queries": 0, "missed_count": 0, "coverage_rate": 1.0,
                "missed_queries": [], "first_cluster_queries": {}}

    def _empty_claim_results(self) -> Dict[str, Any]:
        return {"total": 0, "fabrication": 0, "misattribution": 0, "results": []}

    def _empty_noise_results(self) -> Dict[str, Any]:
        return {"score": 0.0, "mapped_clusters": [], "unmapped_clusters": [], "details": {}}

    def _extract_chunks_from_cache(self, json_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct chunks list from json_cache (which only has 6 keys)."""
        chunks = []
        chunk_score = json_cache.get("chunk_score", {})
        for chunk_id, cs_entry in chunk_score.items():
            chunks.append({
                "chunk_id": chunk_id,
                "text": cs_entry.get("chunk_text", ""),
                "url": cs_entry.get("url", ""),
            })
        return chunks

    def _extract_actions_from_cache(self, json_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct actions list from json_cache iterations."""
        actions = []
        for iteration in json_cache.get("iterations", []):
            for key, value in iteration.items():
                if key.startswith("action_list_"):
                    iter_num = int(key.split("_")[-1])
                    for action_text in value:
                        actions.append({"action": action_text, "iteration": iter_num})
        return actions

    def _extract_claims_from_cache(self, json_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct claims list from json_cache iterations."""
        claims = []
        for iteration in json_cache.get("iterations", []):
            for key, value in iteration.items():
                if key.startswith("claim_list_"):
                    iter_num = int(key.split("_")[-1])
                    for claim_text in value:
                        claims.append({"claim": claim_text, "iteration": iter_num})
        return claims


        decomposed["chunk_score"] = chunk_score
        return decomposed

    def _compute_related_query(self, decomposed: Dict[str, Any]) -> Dict[str, Any]:
        """Compute claim→sub-query mapping using BGE reranker."""
        # Collect all claims
        all_claims = set()
        for c in decomposed.get("all_claims", []):
            text = c.get("claim", str(c)) if isinstance(c, dict) else str(c)
            if text and len(text) > 10:
                all_claims.add(text)
        for para in decomposed.get("report", []):
            for c in para.get("atomic_claims", []):
                if c and len(c) > 10:
                    all_claims.add(c)

        query_list = decomposed.get("query_list", [])
        if not all_claims or not query_list:
            print("  ⚠️ No claims or sub-queries to map")
            return {}

        print(f"  Mapping {len(all_claims)} claims to {len(query_list)} sub-queries...")
        try:
            from .utils.scoring import ClaimQueryReranker
            reranker = ClaimQueryReranker()
            mapping = reranker.compute_mapping(list(all_claims), query_list)
            print(f"  ✅ Mapped {len(mapping)} claims to sub-queries")
            return mapping
        except Exception as e:
            print(f"  ⚠️ Failed to compute related_query: {e}")
            return {}

    def _cleanup_module_caches(self) -> None:
        """Remove per-module cache files in cache root; unified cache in json_cache/ is authoritative."""
        import glob as _glob
        for pat in ["decomposition_*.json", "action_verification_*.json",
                     "claim_verification_*.json", "constraint_checking_*.json",
                     "noise_domination_*.json"]:
            for f in _glob.glob(os.path.join(self.config.cache_dir, pat)):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _detect_propagation(self, decomposed: Dict, claim_results: Dict) -> Dict[str, Any]:
        """Detect propagation hallucinations — derived from action checking results (paper Eq. 4)."""
        # Propagation count comes from action checker's per-action propagation detection
        # which uses Claim Memory context (paper: "built on previously hallucinated claims")
        # This is a reporting wrapper; the actual count is in action_results
        total_actions = len(decomposed.get("actions", []))
        return {
            "total_actions": total_actions,
            "propagation_count": 0,  # Populated from action_results.propagation
            "propagation_score": 0.0,
        }

    def _compute_ping_scores(
        self,
        action_results: Dict,
        constraint_results: Dict,
        claim_results: Dict,
        noise_results: Dict,
        propagation_results: Dict,
    ) -> Dict[str, float]:
        """Compute PING scores from evaluation results."""

        # Grounding (G) = Fabrication + Misattribution
        total_claims = max(claim_results.get("total", 0), 1)
        g_fabrication = claim_results.get("fabrication", 0) / total_claims
        g_misattribution = claim_results.get("misattribution", 0) / total_claims
        grounding_score = min(g_fabrication + g_misattribution, 1.0)

        # Noise-induced (N)
        noise_score = noise_results.get("score", 0.0)

        # Intent (I) = Action Deviation + Restriction Neglect
        total_actions = max(action_results.get("total", 0), 1)
        total_constraints = max(constraint_results.get("total_queries", 0), 1)
        i_deviation = action_results.get("deviation", 0) / total_actions
        i_neglect = constraint_results.get("missed_count", 0) / total_constraints
        intent_score = min(i_deviation + i_neglect, 1.0)

        # Propagation (P) = A_propagation / A_total (Paper Eq. 4)
        total_acts = max(action_results.get("total", 0), 1)
        propagation_score = action_results.get("propagation", 0) / total_acts

        return {
            "grounding": grounding_score,
            "noise_induced": noise_score,
            "intent": intent_score,
            "propagation": propagation_score,
        }

    def _print_summary(self, ping_scores: Dict[str, float], overall_score: float):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("Evaluation Summary (PING Taxonomy)")
        print("=" * 80)
        print(f"Propagation (P):     {ping_scores.get('propagation', 0):.4f}")
        print(f"Intent (I):          {ping_scores.get('intent', 0):.4f}")
        print(f"Noise-induced (N):   {ping_scores.get('noise_induced', 0):.4f}")
        print(f"Grounding (G):       {ping_scores.get('grounding', 0):.4f}")
        print("-" * 40)
        print(f"Overall Score (H):   {overall_score:.4f}")
        print("=" * 80)
