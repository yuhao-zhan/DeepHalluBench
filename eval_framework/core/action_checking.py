"""
Action Checking Module for DeepHalluBench.

Implements planning-stage hallucination detection:
- Action Deviation: actions that deviate from or misinterpret user intent
- Action Propagation: actions built on previously hallucinated claims
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ActionJudgment:
    action: str
    iteration: int
    judgment: str
    confidence: float
    source: str  # nli, llm
    error_type: Optional[str] = None
    scores: Optional[Dict[str, float]] = None
    explanation: str = ""
    source_text: str = ""  # claim text the action references (for propagation detection)


# ---------------------------------------------------------------------------
# JSON extraction helpers (from reference llm_as_judges.py)
# ---------------------------------------------------------------------------

def _extract_json_from_content(content: str) -> str:
    content = content.strip()
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()
    if content.startswith("```") and content.endswith("```"):
        inner = content[3:-3].strip()
        if inner.startswith(("{", "[")):
            return inner
    if content.count("```") >= 2:
        first = content.find("```")
        start = first + 3
        if content[start:start + 4] == "json":
            start += 4
        end = content.find("```", start)
        if end != -1:
            extracted = content[start:end].strip()
            if extracted.startswith(("{", "[")):
                return extracted
    json_start = content.find('{')
    if json_start != -1:
        brace_count = 0
        json_end = -1
        for i, ch in enumerate(content[json_start:], json_start):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        if json_end != -1:
            return content[json_start:json_end]
        return _fix_incomplete_json(content[json_start:])
    return content


def _fix_incomplete_json(incomplete: str) -> str:
    open_b = incomplete.count('{') - incomplete.count('}')
    open_br = incomplete.count('[') - incomplete.count(']')
    fixed = incomplete.rstrip().rstrip(',')
    if open_b > 0:
        fixed += '}' * open_b
    if open_br > 0:
        fixed += ']' * open_br
    return fixed


def _extract_fields_from_text(content: str) -> Optional[Dict[str, Any]]:
    try:
        jm = re.search(r'"judgment"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
        if not jm:
            jm = re.search(r'judgment["\s]*:?\s*["\s]*([a-zA-Z]+)', content, re.IGNORECASE)
        if not jm:
            return None
        judgment = jm.group(1).lower()
        if "entail" in judgment:
            judgment = "entailed"
        elif "contradict" in judgment:
            judgment = "contradicted"
        else:
            judgment = "neutral"
        cm = re.search(r'"confidence"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
        if not cm:
            cm = re.search(r'confidence["\s]*:?\s*([0-9.]+)', content, re.IGNORECASE)
        confidence = float(cm.group(1)) if cm else 0.5
        confidence = max(0.0, min(1.0, confidence))
        em = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
        if not em:
            em = re.search(r'explanation["\s]*:?\s*["\s]*([^"]*)', content, re.IGNORECASE)
        explanation = em.group(1).strip() if em else ""
        return {"judgment": judgment, "confidence": confidence, "explanation": explanation}
    except Exception:
        return None


# ---------------------------------------------------------------------------

class ActionChecker:

    def __init__(self, config):
        self.config = config
        self._nli_model = None
        self._nli_load_attempted = False

    @property
    def nli_model(self):
        if self._nli_model is None and not self._nli_load_attempted:
            self._nli_model = self._load_nli_model()
        return self._nli_model

    def _load_nli_model(self):
        try:
            import signal
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_name = self.config.nli.model_name
            device = self.config.nli.device

            def timeout_handler(signum, frame):
                raise TimeoutError("NLI model loading timed out")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model = model.to(device)
                model.eval()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            def classify(premise: str, hypothesis: str) -> Tuple[str, float]:
                inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                                   truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                labels = ["entailment", "neutral", "contradiction"]
                pred_idx = probs.argmax().item()
                return labels[pred_idx], probs[0][pred_idx].item()

            return classify
        except TimeoutError:
            self._nli_load_attempted = True
            return None
        except Exception:
            self._nli_load_attempted = True
            return None

    # ---- Main entry point ----

    def verify(self, decomposed: Dict[str, Any], query: str,
               max_actions: Optional[int] = None,
               claim_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        actions = decomposed.get("actions", [])
        sub_queries = decomposed.get("sub_queries", [query])

        if max_actions is not None and 0 < max_actions < len(actions):
            import random
            random.seed(42)
            indices = sorted(random.sample(range(len(actions)), max_actions))
            actions = [actions[i] for i in indices]

        if not actions:
            return {"total": 0, "deviation": 0, "propagation": 0, "support": 0, "results": []}

        actions_hash = hash(tuple(sorted(
            a.get("action", str(a)) if isinstance(a, dict) else str(a) for a in actions)))
        cache_file = os.path.join(self.config.cache_dir,
                                   f"action_verification_{actions_hash}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        # Build Claim Memory and hallucinated claims map (reference: get_hallucinated_actions)
        entailed_claims_by_iteration = {}
        hallucinated_claim_texts = set()  # claim_text → hallucinated
        if claim_memory:
            for result in claim_memory.get("results", []):
                claim = result.get("claim", "")
                if result.get("judgment") == "Support":
                    iter_idx = self._get_claim_iteration(claim, decomposed)
                    entailed_claims_by_iteration.setdefault(iter_idx, []).append(claim)
                elif result.get("error_type") in ("fabrication", "misattribution"):
                    hallucinated_claim_texts.add(claim)

        results = []
        deviation_count = 0
        propagation_count = 0
        support_count = 0

        # Build action_texts list for Previous Actions context
        all_action_texts = [
            a.get("action", str(a)) if isinstance(a, dict) else str(a) for a in actions
        ]

        for i, action_data in enumerate(actions):
            if isinstance(action_data, dict):
                action_text = action_data.get("action", str(action_data))
                iteration = action_data.get("iteration", i)
            else:
                action_text = str(action_data)
                iteration = i

            observation_memory = self._build_observation_memory(
                entailed_claims_by_iteration, iteration, query)

            # Previous Actions (reference: get_previous_actions_for_action)
            previous_actions = all_action_texts[:i]

            judgment = self._check_action_deviation(action_text, query, sub_queries,
                                                     observation_memory, previous_actions)
            if judgment.judgment == "Support":
                # Check propagation: is the action's source claim hallucinated?
                source_text = getattr(judgment, 'source_text', None) or ""
                if source_text and source_text in hallucinated_claim_texts:
                    judgment = ActionJudgment(action=action_text, iteration=iteration,
                        judgment="Propagation", confidence=judgment.confidence,
                        source=judgment.source, error_type="propagation",
                        scores=judgment.scores, explanation=judgment.explanation)
                    propagation_count += 1
                else:
                    support_count += 1
            else:
                deviation_count += 1

            results.append({
                "action": action_text,
                "iteration": iteration,
                "judgment": judgment.judgment,
                "confidence": judgment.confidence,
                "source": judgment.source,
                "error_type": judgment.error_type,
                "scores": judgment.scores or {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0},
                "explanation": judgment.explanation or "",
                "source_text": judgment.source_text or "",
            })

        result = {"total": len(actions), "deviation": deviation_count,
                  "propagation": propagation_count, "support": support_count,
                  "results": results}
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return result

    # ---- Helpers ----

    def _call_llm_json(self, system_prompt: str, user_prompt: str,
                        user_only: bool = False) -> Optional[Dict[str, Any]]:
        try:
            from eval_framework.utils import LLMClient
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
            if user_only:
                response = client.generate(system_prompt="", user_prompt=user_prompt)
            else:
                response = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            cleaned = _extract_json_from_content(response)
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                result = _extract_fields_from_text(response)
            if isinstance(result, dict):
                return result
            return None
        except Exception:
            return None

    def _get_claim_iteration(self, claim: str, decomposed: Dict[str, Any]) -> int:
        for i, it in enumerate(decomposed.get("iterations", [])):
            if claim in it.get(f"claim_list_{i+1}", []):
                return i + 1
        return 0

    def _build_observation_memory(self, entailed_claims: Dict[int, List[str]],
                                   target_iteration: int, query: str) -> str:
        """Build Claim Memory from entailed claims of previous iterations.
        Reference: build_entailed_claim_memory() in action_checking.py."""
        memory_items = []
        for iter_num in range(1, target_iteration):
            if iter_num in entailed_claims:
                memory_items.extend(entailed_claims[iter_num])
        if not memory_items:
            return query
        return query + "\n\n" + "\n\n".join(memory_items)

    # ---- Action checks ----

    def _check_action_deviation(self, action: str, query: str, sub_queries: List[str],
                                 observation_memory: str = "",
                                 previous_actions: List[str] = None) -> ActionJudgment:
        """NLI first (>0.99), then LLM with Claim Memory + Previous Actions context.
        Reference: run_two_stage_judgment() in nli_against_query.py."""
        HIGH_NLI_CONF = 0.99
        if self.nli_model is not None:
            try:
                label, confidence = self.nli_model(query, action)
                if label == "entailment" and confidence > HIGH_NLI_CONF:
                    return ActionJudgment(action=action, iteration=0, judgment="Support",
                        confidence=confidence, source="nli", error_type=None,
                        scores={"entailment": 1.0, "contradiction": 0.0, "neutral": 0.0})
                elif label == "contradiction" and confidence > HIGH_NLI_CONF:
                    return ActionJudgment(action=action, iteration=0, judgment="Deviation",
                        confidence=confidence, source="nli", error_type="deviation",
                        scores={"entailment": 0.0, "contradiction": 1.0, "neutral": 0.0})
            except Exception:
                pass
        return self._llm_check_action(action, query, sub_queries, observation_memory,
                                       previous_actions or [])

    def _llm_check_action(self, action: str, query: str, sub_queries: List[str],
                           observation_memory: str = "",
                           previous_actions: List[str] = None) -> ActionJudgment:
        """LLM-based action evaluation with three-part input.
        Reference: _render_judgment_prompt() in nli_against_query.py."""

        obs_context = observation_memory[:2000] if observation_memory else query

        obs_lines = observation_memory.split("\n\n") if observation_memory else [query]
        claims_context = ""
        claim_map = {}
        claim_idx = 0
        for line in obs_lines:
            line = line.strip()
            if line and line != query and len(line) > 10:
                claims_context += f"[{claim_idx}] {line[:200]}\n"
                claim_map[claim_idx] = line
                claim_idx += 1
        if not claims_context:
            claims_context = "No previous claims available."

        if previous_actions:
            actions_context = "\n".join([f"[{i}] {a}" for i, a in enumerate(previous_actions[-10:])])
        else:
            actions_context = "No previous actions available."

        user_prompt = f"""Evaluate whether the action supports the user's query using two principles:

**1. Goal Coherence**: Does the action align with the user's stated objectives and help achieve their goals?
**2. Logical Continuity**: Is the action a reasonable next step that builds upon previous findings without contradicting them?

---
**User Query**: {query}

**Previous Claims** (top 10 most relevant prior claims):
{claims_context}

**Previous Actions** (in-progress steps; when evaluating the current action, **assume these will return ideal results**):
{actions_context}

**Action to Evaluate**: {action}

---
**CRITICAL INSTRUCTION**: When evaluating, treat previous actions as if they have already succeeded in obtaining their target data/information. An action is NOT "premature" if previous actions are pursuing its prerequisites.

---
**Evaluation Criteria**:

**Support**: Action makes reasonable progress toward the goal and maintains logical consistency.
- Prefer Support if the action could plausibly uncover new information, expand the search space, or advance the task -- even if it zooms in on only one facet of the query.
- **Lightweight actions** (Extract, Identify, Compile, Format, Aggregate, Summarize) that process already-surfaced information should almost always be Support -- they rarely affect future searches or tool executions.
- High-level navigation or setup steps that keep the investigation on-topic are Support even when phrased broadly.
- **Sequential Planning**: Multi-step plans are generated together. Evaluate each action assuming previous actions return ideal results. If previous actions are obtaining X, actions using X are valid next steps -- NOT premature, NOT deviations.

**NotSupport**: Only when clearly meeting one of these:

1. **Deviation** -- Action contradicts confirmed claims or pursues completely irrelevant tangent. **NOT deviation if**: previous actions are obtaining prerequisites this action needs (assume they succeed), OR action is an intermediate step toward the goal.

2. **Redundancy** -- **Critical distinction**: Suggestions ("would be helpful to...", "should try...") are NOT executions. Mark NotSupport (Redundancy) only when:
   - A previous claim **executed the exact same search/tool call** (same tool, same topic/query, same parameters) **and** documented concrete results
   - Different tools (e.g., google_search vs wikipedia_searcher), different query phrasings, or broader/narrower scopes are NOT redundant -- they may yield new information
   - If a previous claim states this exact search already FAILED or returned no ideal results, repeating it is redundant
   - Claims that merely reflect, suggest, or plan ("performed search for...", "attempted to...", "would be helpful to...") **without** showing actual results do NOT make later attempts redundant
   - **Actions that depend on prerequisites being fetched by previous actions are NOT redundant** -- they are part of a planned sequence assuming prior success
   - Redundancy is determined by **Previous Claims**, NOT by previous actions; actions merely show context

**source Rules**:
- If "Support": Use -1 if derived from the query, or claim index [i] if building upon that claim
- If "NotSupport": Use -1 if contradicts the query, or claim index [i] if contradicts/made redundant by that claim

**Examples**:

Example 1 (Support - search action without redundancy):
- Query: "Find information about Python's new features"
- Previous Claim [2]: "The official Python documentation does not contain details about version 3.12"
- Action: "Search for Python 3.12 features on GitHub and Stack Overflow"
- Output: {{"label": "Support", "source": 2, "type": null, "confidence": 0.9, "explanation": "Search action explores alternative sources after [2] showed documentation gap"}}

Example 2 (NotSupport - Redundancy for search action):
- Query: "What are the top-rated Italian restaurants in Boston?"
- Previous Claim [1]: "Found the highest-rated Italian restaurants in Boston: Mamma Maria (4.8 stars), Giulia (4.7 stars), and Sportello (4.6 stars)"
- Action: "Search for the best Italian restaurants in Boston"
- Output: {{"label": "NotSupport", "source": 1, "type": "redundancy", "confidence": 0.95, "explanation": "Claim [1] already provides the search results this action would retrieve"}}

Example 3 (Support - lightweight extraction action):
- Query: "Summarize the key findings from the quarterly sales report"
- Previous Claim [2]: "The Q3 sales report shows revenue increased 15% year-over-year, with strong growth in Asia Pacific and Europe"
- Action: "Extract all regional sales figures and create a summary table"
- Output: {{"label": "Support", "source": 2, "type": null, "confidence": 0.85, "explanation": "Extraction formats data for analysis; low-impact action doesn't affect future searches"}}

Example 4 (NotSupport - Deviation):
- Query: "Analyze the economic impact of the 2008 financial crisis on European banks"
- Previous Claim [1]: "Deutsche Bank reported significant losses in Q4 2008"
- Action: "Research the history of banking regulations in medieval Europe"
- Output: {{"label": "NotSupport", "source": -1, "type": "deviation", "confidence": 0.9, "explanation": "Action pursues irrelevant historical tangent unrelated to 2008 crisis analysis"}}

Example 5 (Support - NOT premature when previous actions handle prerequisites):
- Query: "Download the latest climate data and calculate the 10-year temperature trend"
- Previous Actions context: "[0] Fetch climate data from NOAA API", "[1] Download temperature records"
- Previous Claims: [0] "We haven't retrieved the climate data yet", [1] "The data is not available"
- Action: "Set up the linear regression model to compute temperature trends from the climate data"
- WRONG: {{"label": "NotSupport", "type": "deviation", "explanation": "This is premature since we haven't retrieved the data yet"}}
- CORRECT: {{"label": "Support", "source": -1, "type": null, "confidence": 0.88, "explanation": "Previous actions [0][1] are fetching the data; assuming they succeed, this is a valid next step in the planned sequence."}}

Return JSON:
{{
    "label": "Support" or "NotSupport",
    "source": -1 or integer index,
    "type": "deviation" or "redundancy" (required when label is "NotSupport"; omit or null for Support),
    "confidence": confidence score between 0 and 1,
    "explanation": "Brief justification for your decision. ONLY one sentence."
}}"""

        llm_result = self._call_llm_json(
            system_prompt="",
            user_prompt=user_prompt,
            user_only=True)

        if llm_result is None:
            return ActionJudgment(action=action, iteration=0, judgment="Support",
                confidence=0.5, source="llm_fallback", error_type=None,
                scores={"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0})

        label = llm_result.get("label", llm_result.get("judgment", "Support")).strip()
        conf = llm_result.get("confidence", 0.5)
        explanation = llm_result.get("explanation", "")
        source_idx = int(llm_result.get("source", -1) or -1)
        refs = sorted({int(m) for m in re.findall(r"\[(\d+)\]", explanation) if m.isdigit()})
        if refs and source_idx not in refs:
            source_idx = refs[0]
        source_text = claim_map.get(source_idx, "") if 0 <= source_idx else ""

        if label == "NotSupport":
            ns_type = llm_result.get("type", "deviation")
            return ActionJudgment(action=action, iteration=0, judgment="Deviation",
                confidence=conf, source="llm",
                error_type=ns_type if ns_type in ("deviation", "redundancy") else "deviation",
                scores={"entailment": 0.0, "contradiction": 1.0, "neutral": 0.0},
                explanation=explanation, source_text=source_text)
        else:
            return ActionJudgment(action=action, iteration=0, judgment="Support",
                confidence=conf, source="llm", error_type=None,
                scores={"entailment": 1.0, "contradiction": 0.0, "neutral": 0.0},
                explanation=explanation, source_text=source_text)

