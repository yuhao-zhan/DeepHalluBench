"""
Claim Verification Module for DeepHalluBench

Implements the grounding hallucination detection using a two-round verification pipeline:

Round 1: Initial Verification
- Verify claims against their specific evidence scope
- Use NLI-then-LLM cascade for cost-efficient verification

Round 2: Adaptive Re-Verification
- Misattribution check: expand evidence scope for cited claims
- Reflection check: verify against Claim Memory for intermediate claims

This module detects:
- Fabrication: Claims not supported by any evidence
- Misattribution: Claims cited but not supported by cited sources
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClaimJudgment:
    """Container for claim verification judgment."""
    claim: str
    judgment: str  # Support, NotSupport, Contradiction
    confidence: float
    evidence_chunks: List[Dict[str, Any]]
    source: str  # nli, llm, memory
    error_type: Optional[str] = None  # fabrication, misattribution, None
    all_judged_chunks: Optional[List[Dict[str, Any]]] = None  # per-chunk NLI scores
    relevant_chunks: Optional[List[Dict[str, Any]]] = None  # best supporting chunks
    explanation: str = ""  # LLM explanation for the judgment


class ClaimVerifier:
    """
    Verifies claims against retrieved evidence to detect grounding hallucinations.

    Uses a two-round verification pipeline:
    1. Initial verification against cited/found evidence
    2. Adaptive re-verification for unsupported claims
    """

    def __init__(self, config):
        """
        Initialize the claim verifier.

        Args:
            config: EvaluationConfig object
        """
        self.config = config
        self.nli_threshold = config.nli_threshold
        self.similarity_threshold = config.similarity_threshold
        self.top_k = config.top_k_chunks

        # Lazy-loaded components
        self._nli_model = None
        self._nli_load_attempted = False
        self._reranker = None
        self._embedder = None

        # Cache for top-k chunks collected during verification (keyed by claim_text)
        self._top_k_chunks_cache = {}

    @property
    def nli_model(self):
        """Lazy load NLI model (never retry after failure)."""
        if self._nli_model is None and not self._nli_load_attempted:
            self._nli_model = self._load_nli_model()
        return self._nli_model

    @property
    def reranker(self):
        """Lazy load reranker model."""
        if self._reranker is None:
            self._reranker = self._load_reranker()
        return self._reranker

    @property
    def embedder(self):
        """Lazy load embedding model."""
        if self._embedder is None:
            self._embedder = self._load_embedder()
        return self._embedder

    def _load_nli_model(self):
        """Load NLI model for claim verification."""
        try:
            import signal
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = self.config.nli.model_name
            device = self.config.nli.device

            # Set a timeout for model loading
            def timeout_handler(signum, frame):
                raise TimeoutError("NLI model loading timed out")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout for NLI model

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model = model.to(device)
                model.eval()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            def classify(premise: str, hypothesis: str) -> Tuple[str, float, Dict[str, float]]:
                """Run NLI classification, returning (label, confidence, scores_dict)."""
                inputs = tokenizer(
                    premise, hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                # Map to labels (0=entailment, 1=neutral, 2=contradiction)
                labels = ["entailment", "neutral", "contradiction"]
                pred_idx = probs.argmax().item()
                confidence = probs[0][pred_idx].item()
                scores_dict = {
                    labels[i]: float(probs[0][i].item())
                    for i in range(len(labels))
                }

                return labels[pred_idx], confidence, scores_dict

            self._nli_model_loaded = True
            return classify
        except TimeoutError:
            print(f"⚠️ NLI model loading timed out after 60s - using fallback")
            self._nli_model_loaded = False
            self._nli_load_attempted = True
            return None
        except Exception as e:
            print(f"⚠️ Failed to load NLI model: {e}")
            self._nli_model_loaded = False
            self._nli_load_attempted = True
            return None

    def _load_reranker(self):
        """Load reranker model for evidence ranking (shared singleton to avoid GPU conflicts)."""
        if getattr(self, '_reranker_failed', False):
            return None
        try:
            from ..utils.scoring import _get_global_reranker
            model_name = self.config.reranker.model_name
            reranker = _get_global_reranker(model_name)
            if reranker is None:
                self._reranker_failed = True
            return reranker
        except Exception as e:
            print(f"⚠️ Failed to load reranker: {e}")
            self._reranker_failed = True
            return None

    def _load_embedder(self):
        """Load embedding model for chunk retrieval (CPU to save GPU memory)."""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.config.embedding.model_name
            embedder = SentenceTransformer(model_name, device='cpu')
            return embedder
        except Exception as e:
            print(f"⚠️ Failed to load embedder: {e}")
            return None

    def verify(self, decomposed: Dict[str, Any], query: str, max_claims: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify claims against retrieved evidence with caching.

        Args:
            decomposed: Decomposed trajectory data
            query: User query
            max_claims: If set, sample only this many claims for verification

        Returns:
            Dictionary with claim verification results
        """
        claims = decomposed.get("claims", []) + decomposed.get("report_claims", [])
        chunks = decomposed.get("chunks", [])

        # Sample claims if max_claims is set
        sample_mode = False
        if max_claims is not None and max_claims > 0 and max_claims < len(claims):
            sample_mode = True
            print(f"  📋 Sampling {max_claims} claims out of {len(claims)} for demo verification")
            import random
            random.seed(42)
            claims = random.sample(claims, max_claims)

        if not claims:
            return {
                "total": 0,
                "fabrication": 0,
                "misattribution": 0,
                "support": 0,
                "results": []
            }

        # Generate cache file path based on claims and chunks hash
        claims_hash = hash(tuple(sorted([c.get("claim", str(c)) if isinstance(c, dict) else str(c) for c in claims])))
        chunks_hash = hash(tuple(sorted([c.get("chunk_id", str(c)) if isinstance(c, dict) else str(c) for c in chunks])))
        cache_file = os.path.join(self.config.cache_dir, f"claim_verification_{claims_hash}_{chunks_hash}.json")

        # Check if cache exists
        if os.path.exists(cache_file):
            print(f"  📁 Loading claim verification from cache")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"  ✅ Loaded {cached_data['total']} claims from cache")
                return cached_data
            except Exception as e:
                print(f"  ⚠️ Failed to load cache: {e}")

        print(f"  Verifying {len(claims)} claims against {len(chunks)} chunks...")

        results = []
        fabrication_count = 0
        misattribution_count = 0
        support_count = 0

        # Prepare claim texts
        claim_texts = []
        claim_sources = []
        for claim_data in claims:
            if isinstance(claim_data, dict):
                claim_texts.append(claim_data.get("claim", str(claim_data)))
                claim_sources.append(claim_data.get("source", "unknown"))
            else:
                claim_texts.append(str(claim_data))
                claim_sources.append("unknown")

        for i, claim_text in enumerate(claim_texts):
            source = claim_sources[i]
            judgment = self._verify_claim_initial(claim_text, chunks, query)

            if judgment.judgment == "Support":
                support_count += 1
            else:
                # Round 2: Adaptive re-verification with Claim Memory (paper: §Reflection Check Logic)
                claim_memory = results
                judgment = self._verify_claim_adaptive(
                    claim_text, chunks, judgment, source, claim_memory
                )
                # Re-check after adaptive: may have changed to Support via Reflection Check
                if judgment.judgment == "Support":
                    support_count += 1
                elif judgment.error_type == "fabrication":
                    fabrication_count += 1
                elif judgment.error_type == "misattribution":
                    misattribution_count += 1

            results.append({
                "claim": claim_text,
                "judgment": judgment.judgment,
                "confidence": judgment.confidence,
                "source": judgment.source,
                "error_type": judgment.error_type,
                "evidence_chunks": judgment.evidence_chunks,
                "all_judged_chunks": judgment.all_judged_chunks or [],
                "relevant_chunks": judgment.relevant_chunks or [],
                "explanation": judgment.explanation or "",
            })

        # Save results (top_k_chunks only stored in json_cache, not in output)
        result = {
            "total": len(claims),
            "support": support_count,
            "fabrication": fabrication_count,
            "misattribution": misattribution_count,
            "results": results,
        }
        # Store top_k_chunks on result for json_cache persistence (accessed via key, not in output)
        result["_top_k_chunks"] = self._top_k_chunks_cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  💾 Saved claim verification to cache: {cache_file}")
        except Exception:
            pass

        return result

    def _verify_claim_initial(
        self, claim: str, chunks: List[Dict[str, Any]], query: str
    ) -> ClaimJudgment:
        """
        Round 1: Initial verification against retrieved evidence.

        Uses NLI-centric approach aligned with reference determine_final_judgment_new():
        1. Run NLI on all top-k chunks individually
        2. If any chunk is non-neutral (argmax != neutral) with confidence > 0.99:
           pick the best, set judgment from NLI scores
        3. If ALL chunks are neutral/ambiguous: concatenate with [SEP] and re-run NLI
        4. LLM fallback only when NLI model is unavailable or no chunks
        """
        if not chunks:
            return ClaimJudgment(
                claim=claim,
                judgment="NotSupport",
                confidence=1.0,
                evidence_chunks=[],
                source="initial",
                error_type="fabrication",
            )

        # Get top-k chunks using reranking
        top_chunks = self._get_top_chunks(claim, chunks, query, k=self.top_k)

        if not top_chunks:
            return ClaimJudgment(
                claim=claim,
                judgment="NotSupport",
                confidence=1.0,
                evidence_chunks=[],
                source="initial",
                error_type="fabrication",
                all_judged_chunks=[],
                relevant_chunks=[],
            )

        # Build all_judged_chunks: per-chunk NLI scores (consistent truncation)
        CHUNK_TRUNC = 500
        nli_scores_list = []  # list of (chunk, scores_dict)
        all_judged_chunks = []
        for chunk in top_chunks:
            chunk_text = chunk.get("text", "")[:CHUNK_TRUNC]
            chunk_id = chunk.get("chunk_id", "")
            source_url = chunk.get("url", "")
            rerank_score = chunk.get("rerank_score", 0.0)

            if self.nli_model:
                label, confidence, scores_dict = self.nli_model(chunk_text, claim)
                nli_scores_list.append((chunk, scores_dict))
                all_judged_chunks.append({
                    "chunk_id": chunk_id,
                    "source_url": source_url,
                    "judgment": "Support" if label == "entailment" else ("Contradiction" if label == "contradiction" else "Neutral"),
                    "confidence": float(confidence),
                    "entailment_score": scores_dict["entailment"],
                    "neutral_score": scores_dict["neutral"],
                    "contradiction_score": scores_dict["contradiction"],
                    "rerank_score": float(rerank_score),
                    "chunk_text": chunk_text,
                    "similarity_score": 0.0,
                    "sentence_idx": 0,
                })
            else:
                scores_dict = {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}
                nli_scores_list.append((chunk, scores_dict))
                all_judged_chunks.append({
                    "chunk_id": chunk_id,
                    "source_url": source_url,
                    "judgment": "Neutral",
                    "confidence": 0.5,
                    "entailment_score": 0.0,
                    "neutral_score": 1.0,
                    "contradiction_score": 0.0,
                    "rerank_score": float(rerank_score),
                    "chunk_text": chunk_text,
                    "similarity_score": 0.0,
                    "sentence_idx": 0,
                })

        # Aligned with reference determine_final_judgment_new() + user's > 0.99 threshold
        #
        # 1. Find high-confidence non-neutral chunks (argmax != neutral AND max > 0.99)
        # 2. If found: pick best, use NLI judgment
        # 3. If ALL chunks neutral: concatenate with [SEP], re-run NLI, check > 0.99
        # 4. Otherwise → LLM
        if self.nli_model:
            HIGH_CONF = 0.99

            # Step 1: Find chunks with non-neutral argmax AND confidence > 0.99
            high_conf_non_neutral = []
            for chunk, scores_dict in nli_scores_list:
                max_score = max(scores_dict.values())
                if scores_dict["neutral"] != max_score and max_score > HIGH_CONF:
                    high_conf_non_neutral.append((chunk, scores_dict))

            if high_conf_non_neutral:
                # Step 2: Pick best high-confidence non-neutral chunk
                best_chunk, best_scores = max(
                    high_conf_non_neutral,
                    key=lambda x: max(x[1]["entailment"], x[1]["contradiction"])
                )

                if best_scores["entailment"] > best_scores["contradiction"]:
                    final_judgment = "Support"
                    confidence = best_scores["entailment"]
                    error_type = None
                else:
                    final_judgment = "NotSupport"
                    confidence = best_scores["contradiction"]
                    error_type = "fabrication"

                # relevant_chunks: point to matching entries from all_judged_chunks
                best_id = best_chunk.get("chunk_id", "")
                relevant_chunks = [aj for aj in all_judged_chunks if aj["chunk_id"] == best_id]
                if not relevant_chunks:
                    relevant_chunks = all_judged_chunks[:1]

                return ClaimJudgment(
                    claim=claim, judgment=final_judgment, confidence=confidence,
                    evidence_chunks=[best_chunk], source="nli",
                    error_type=error_type,
                    all_judged_chunks=all_judged_chunks, relevant_chunks=relevant_chunks,
                )

            # Step 3: Check if ALL chunks are neutral (argmax == neutral for all)
            all_neutral = all(
                scores_dict["neutral"] == max(scores_dict.values())
                for _, scores_dict in nli_scores_list
            )

            if all_neutral:
                # All chunks neutral → concatenate with [SEP], re-run NLI
                concatenated_text = " [SEP] ".join([c.get("text", "")[:CHUNK_TRUNC] for c in top_chunks])
                try:
                    concat_label, concat_confidence, concat_scores = self.nli_model(concatenated_text, claim)
                except Exception as e:
                    print(f"  ⚠️ Concatenated NLI failed: {e}")
                    return self._llm_verify(claim, top_chunks, query, all_judged_chunks)

                concat_max_score = max(concat_scores.values())
                concat_argmax = max(concat_scores, key=concat_scores.get)

                if concat_max_score > HIGH_CONF and concat_argmax in ("entailment", "contradiction"):
                    if concat_argmax == "entailment":
                        final_judgment = "Support"
                        confidence = concat_scores["entailment"]
                        error_type = None
                    else:
                        final_judgment = "NotSupport"
                        confidence = concat_scores["contradiction"]
                        error_type = "fabrication"

                    # relevant_chunks = all_judged_chunks (consistent per-chunk scores)
                    return ClaimJudgment(
                        claim=claim, judgment=final_judgment, confidence=confidence,
                        evidence_chunks=top_chunks[:3], source="nli_concat",
                        error_type=error_type,
                        all_judged_chunks=all_judged_chunks, relevant_chunks=all_judged_chunks[:3],
                    )

                # Concatenated NLI also low confidence → LLM
                return self._llm_verify(claim, top_chunks, query, all_judged_chunks)

            # Step 4: Mixed chunks (some non-neutral but all < 0.99) → LLM
            return self._llm_verify(claim, top_chunks, query, all_judged_chunks)

        # No NLI model → LLM
        return self._llm_verify(claim, top_chunks, query, all_judged_chunks)

    def _verify_claim_adaptive(
        self, claim: str, chunks: List[Dict[str, Any]], initial_judgment: ClaimJudgment,
        source: str, claim_memory: Optional[List[Dict[str, Any]]] = None
    ) -> ClaimJudgment:
        """
        Round 2: Adaptive re-verification for unsupported claims (paper Appendix §Reflection Check Logic).

        (1) Misattribution Check: expand evidence scope to all chunks
        (2) Reflection Check: verify intermediate claims against Claim Memory
        """
        if initial_judgment.judgment != "NotSupport":
            return initial_judgment

        # Paper: Round 2 depends on claim source type
        if source == "report":
            # (1) Misattribution Check: only for claims with explicit citations
            all_chunks = self._get_top_chunks(claim, chunks, None, k=len(chunks))
            for chunk in all_chunks:
                if self.nli_model:
                    chunk_text = chunk.get("text", "")[:500]
                    label, confidence, scores_dict = self.nli_model(chunk_text, claim)
                    if label == "entailment" and confidence > 0.99:
                        return ClaimJudgment(
                            claim=claim, judgment="NotSupport",
                            confidence=initial_judgment.confidence,
                            evidence_chunks=[chunk], source="nli_adaptive",
                            error_type="misattribution",
                            all_judged_chunks=initial_judgment.all_judged_chunks,
                            relevant_chunks=initial_judgment.relevant_chunks,
                        )
        else:
            # (2) Reflection Check: only for intermediate claims, against Claim Memory
            if claim_memory:
                result = self._reflection_check(claim, claim_memory)
                if result is not None:
                    return result

        # No support = fabrication
        adapt_source = initial_judgment.source
        if not adapt_source.endswith("_adaptive"):
            adapt_source = f"{adapt_source}_adaptive"
        return ClaimJudgment(
            claim=claim, judgment="NotSupport",
            confidence=initial_judgment.confidence,
            evidence_chunks=initial_judgment.evidence_chunks,
            source=adapt_source, error_type="fabrication",
            all_judged_chunks=initial_judgment.all_judged_chunks,
            relevant_chunks=initial_judgment.relevant_chunks,
        )

    def _reflection_check(self, claim: str, claim_memory: List[Dict[str, Any]]) -> Optional[ClaimJudgment]:
        """Paper: verify intermediate claim against Claim Memory (top-K similar claims)."""
        if not claim_memory:
            return None
        try:
            from ..utils import LLMClient
            client = None
            if hasattr(self.config, 'llm') and self.config.llm:
                try:
                    client = LLMClient.from_config(self.config.llm)
                except Exception:
                    pass
            if not client:
                client = LLMClient.from_env()
            if not client or not client._client:
                return None

            # Select top-K (K=10 per paper) most similar Support claims from Claim Memory
            support_claims = [r.get("claim", "") for r in claim_memory
                            if r.get("judgment") == "Support"]
            # Simple keyword overlap to select top-10
            claim_words = set(claim.lower().split())
            scored = []
            for sc in support_claims:
                sc_words = set(sc.lower().split())
                overlap = len(claim_words & sc_words)
                scored.append((overlap, sc))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_claims = [s[1] for s in scored[:10]]

            if not top_claims:
                return None

            context = "\n".join(f"- {c}" for c in top_claims)
            prompt = f"""Verify if the following claim is a valid internal reflection based on the agent's prior findings.

Claim: {claim}

Prior Findings (Claim Memory):
{context}

Is this claim a valid reflection, reasonable next-step inference, or universally accepted common knowledge based on the prior findings? Answer Support if yes, Unsupport if fabricated.

Output JSON:
```json
{{"judgment": "Support|Unsupport", "confidence": 0.0-1.0, "evidence": "explanation"}}
```"""
            response = client.generate(
                system_prompt="Expert verifier. Determine if the claim is a valid reflection based on prior findings.",
                user_prompt=prompt,
            )
            import json as _json, re as _re
            content = response.strip()
            if "```json" in content:
                s = content.find("```json") + 7
                e = content.find("```", s)
                if e != -1:
                    content = content[s:e].strip()
            try:
                result = _json.loads(content)
            except _json.JSONDecodeError:
                jm = _re.search(r'"judgment"\s*:\s*"([^"]+)"', response)
                result = {"judgment": jm.group(1) if jm else "Unsupport", "confidence": 0.5}
            if result.get("judgment", "") == "Support":
                return ClaimJudgment(
                    claim=claim, judgment="Support",
                    confidence=float(result.get("confidence", 0.7)),
                    evidence_chunks=[], source="llm_reflection",
                    error_type=None,
                    all_judged_chunks=[],
                    relevant_chunks=[],
                    explanation=result.get("evidence", result.get("explanation", "")),
                )
        except Exception:
            pass
        return None

    def _get_top_chunks(
        self, claim: str, chunks: List[Dict[str, Any]], query: Optional[str], k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top-k most relevant chunks for a claim and cache the result."""
        if not chunks:
            self._top_k_chunks_cache[claim] = []
            return []

        if not self.reranker:
            top = [dict(c) for c in chunks[:k]]
            self._top_k_chunks_cache[claim] = top
            return top

        try:
            # Prepare chunk-text pairs for FlagReranker
            chunk_infos = [(c.get("text", "")[:512], c) for c in chunks]
            pair_texts = [[claim, ct] for ct, _ in chunk_infos]

            # Get reranker scores via FlagReranker.compute_score
            scores = self.reranker.compute_score(pair_texts, normalize=True, batch_size=16)
            if not isinstance(scores, list):
                scores = [scores]

            # Sort by score descending — handle both float and numpy scalar
            scored = sorted(
                zip(scores, [c for _, c in chunk_infos]),
                key=lambda x: float(x[0]) if hasattr(x[0], '__float__') else 0.0,
                reverse=True,
            )
            # Attach rerank_score to each chunk dict
            top = []
            for score, chunk in scored[:k]:
                chunk = dict(chunk)  # copy to avoid mutating originals
                try:
                    chunk["rerank_score"] = score.item() if hasattr(score, 'item') else float(score)
                except (TypeError, ValueError):
                    chunk["rerank_score"] = 0.0
                top.append(chunk)

            # Cache result
            self._top_k_chunks_cache[claim] = top
            return top
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
            top = chunks[:k]
            self._top_k_chunks_cache[claim] = top
            return top

    def _llm_judge_single_chunk(self, claim: str, chunk_text: str, query: str) -> Dict[str, Any]:
        """Per-chunk LLM judgment using reference llm_judge_claim_with_retry prompt."""
        from ..utils import LLMClient
        client = None
        if hasattr(self.config, 'llm') and self.config.llm:
            try:
                client = LLMClient.from_config(self.config.llm)
            except Exception:
                pass
        if not client:
            client = LLMClient.from_env()
        if not client or not client._client:
            return {"judgment": "neutral", "confidence": 0.0, "evidence": ""}

        # Handle bare URL chunks: extract keywords
        if chunk_text.startswith("http://") or chunk_text.startswith("https://"):
            slug = chunk_text.split("/")[-1] if "/" in chunk_text else chunk_text
            chunk_text = f"Keywords from source URL: {slug.replace('-', ' ').replace('_', ' ')}"

        user_prompt = f"""Query: {query}

Claim to verify: {claim}

Document chunk content:
{chunk_text[:1500]}

Please analyze whether the claim is entailed, contradicted, or neutral with respect to the document chunk.

REMEMBER: Respond with ONLY a JSON code block using the exact format specified. Do not escape characters unnecessarily."""

        combined_prompt = f"""You are an expert claim verification system. Given a claim, query, and document content, classify the relationship as ENTAILED, CONTRADICTED, or NEUTRAL.

## Classification Criteria

**ENTAILED**: The claim is explicitly stated, reasonably inferred, or represents a valid summary/abstraction of the document content.

**CONTRADICTED**: The document contains information that directly refutes the claim. Requires: (1) all necessary claim elements present in document, (2) sufficient information for definitive judgment, (3) clear contradictory evidence.

**NEUTRAL**: Insufficient information to support or contradict the claim. Use when any necessary verification elements are missing, information is incomplete, or the claim cannot be properly evaluated.

## Examples

**Example 1 - ENTAILED**
Document: "The study found a 25% increase in productivity after implementing the new system."
Claim: "Productivity improved following system implementation."
-> ENTAILED: The claim summarizes the documented finding.

**Example 2 - CONTRADICTED**
Document: "The experiment was conducted with 100 participants aged 18-25."
Claim: "The study included elderly participants over 65."
-> CONTRADICTED: Document clearly states age range 18-25, directly contradicting the claim about elderly participants.

**Example 3 - NEUTRAL**
Document: "The company announced a new product launch."
Claim: "The product launch increased quarterly revenue by 15%."
-> NEUTRAL: Document mentions launch but lacks revenue information needed to verify the specific claim.

## Output Format
```json
{{
  "judgment": "entailed|contradicted|neutral",
  "evidence": "One-sentence explanation (required for entailed/contradicted only)",
  "confidence": 0.0-1.0
}}
```

{user_prompt}"""

        try:
            response = client.generate(system_prompt="", user_prompt=combined_prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            print(f"    LLM per-chunk failed: {e}")
            return {"judgment": "neutral", "confidence": 0.0, "evidence": ""}

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON from code blocks or raw text."""
        import json as _json, re as _re
        content = response.strip()
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif content.startswith("{") and "}" in content:
            brace_count = 0
            json_end = -1
            for i, ch in enumerate(content):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end != -1:
                content = content[:json_end]
        try:
            result = _json.loads(content)
        except _json.JSONDecodeError:
            jm = _re.search(r'"judgment"\s*:\s*"([^"]+)"', response)
            cm = _re.search(r'"confidence"\s*:\s*([0-9.]+)', response)
            em = _re.search(r'"(?:evidence|explanation)"\s*:\s*"([^"]*)"', response)
            result = {
                "judgment": jm.group(1).lower() if jm else "neutral",
                "confidence": float(cm.group(1)) if cm else 0.0,
                "evidence": em.group(1) if em else "",
            }
        if not isinstance(result, dict):
            return {"judgment": "neutral", "confidence": 0.0, "evidence": ""}
        return result

    def _llm_verify(
        self, claim: str, chunks: List[Dict[str, Any]], query: str,
        all_judged_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> ClaimJudgment:
        """Per-chunk LLM verification + 3-case aggregation.
        Reference: claim_checking_LLM.py process_claims_parallel lines 539-556."""
        if all_judged_chunks is None:
            all_judged_chunks = []

        if not chunks:
            return ClaimJudgment(claim=claim, judgment="NotSupport", confidence=0.5,
                evidence_chunks=[], source="llm", error_type="fabrication",
                all_judged_chunks=all_judged_chunks, relevant_chunks=[])

        # Per-chunk LLM evaluation — replace NLI judgments in all_judged_chunks
        chunk_judgments = []
        for i, c in enumerate(chunks[:5]):
            chunk_text = c.get("text", "")
            chunk_id = c.get("chunk_id", "")
            source_url = c.get("url", "")
            result = self._llm_judge_single_chunk(claim, chunk_text, query)
            chunk_judgments.append({
                "chunk_id": chunk_id,
                "source_url": source_url,
                "chunk_text": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                "judgment": result["judgment"],
                "confidence": result["confidence"],
                "evidence": result.get("evidence", ""),
            })
            # Update all_judged_chunks with LLM judgment
            if i < len(all_judged_chunks) and all_judged_chunks[i].get("chunk_id") == chunk_id:
                all_judged_chunks[i]["judgment"] = "Support" if result["judgment"] == "entailed" else (
                    "Contradiction" if result["judgment"] == "contradicted" else "Neutral")
                all_judged_chunks[i]["confidence"] = result["confidence"]

        # 3-case aggregation (reference: claim_checking_LLM.py lines 539-556)
        entails = [cj["judgment"] == "entailed" for cj in chunk_judgments]
        contradicts = [cj["judgment"] == "contradicted" for cj in chunk_judgments]
        final_j = None; final_conf = 0.0; final_explanation = ""; relevant = []

        if any(entails) and any(contradicts):
            # CASE 1: MIXED
            mixed_chunks = [cj for cj in chunk_judgments
                           if cj["judgment"] in ("entailed", "contradicted")]
            concat_text = " ".join([cj["chunk_text"] for cj in mixed_chunks])
            result = self._llm_judge_single_chunk(claim, concat_text[:3000], query)
            final_j = result["judgment"]
            final_conf = result["confidence"]
            final_explanation = result.get("evidence", "")
            relevant = [aj for aj in all_judged_chunks
                       if any(aj.get("chunk_id") == cj["chunk_id"] for cj in chunk_judgments
                              if cj["judgment"] == final_j)]
        elif all(cj["judgment"] == "neutral" for cj in chunk_judgments):
            # CASE 2: ALL NEUTRAL
            concat_text = " ".join([cj["chunk_text"] for cj in chunk_judgments])
            result = self._llm_judge_single_chunk(claim, concat_text[:3000], query)
            final_j = result["judgment"]
            final_conf = result["confidence"]
            final_explanation = result.get("evidence", "")
        else:
            # CASE 3: CONSENSUS
            best = max(chunk_judgments,
                       key=lambda x: x["judgment"] != "neutral")
            final_j = best["judgment"]
            final_conf = best["confidence"]
            final_explanation = best.get("evidence", "")
            relevant = [aj for aj in all_judged_chunks
                       if any(aj.get("chunk_id") == cj["chunk_id"] for cj in chunk_judgments
                              if cj["judgment"] == final_j)]

        if not relevant:
            relevant = all_judged_chunks[:1] if all_judged_chunks else []

        if final_j in ("entailed", "entailment"):
            judgment, error_type = "Support", None
        else:
            judgment, error_type = "NotSupport", "fabrication"

        return ClaimJudgment(claim=claim, judgment=judgment, confidence=final_conf,
            evidence_chunks=chunks[:1], source="llm",
            error_type=error_type,
            all_judged_chunks=all_judged_chunks,
            relevant_chunks=relevant[:1],
            explanation=final_explanation)


