"""
Trajectory Decomposition Module for DeepHalluBench

Decomposes research trajectories into atomic units for fine-grained evaluation:
- Atomic actions (from planning steps)
- Atomic claims (from observations and reports)
- Sub-queries (from the user query)

This module implements a two-stage LLM pipeline:
1. Initial decomposition into fragments
2. Double-check refinement for atomic units
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import LLM client
try:
    from ..utils import LLMClient
except ImportError:
    # Fallback if LLM client not available
    LLMClient = None


@dataclass
class DecomposedTrajectory:
    """Container for decomposed trajectory data matching original HalluBench format."""
    query: str
    query_list: List[str]           # Decomposed atomic queries/constraints
    iterations: List[Dict[str, Any]] # Iterations with action_list_N, search_list_N, claim_list_N
    report: List[Dict[str, Any]]     # Report paragraphs with paragraph_index, paragraph_text, atomic_claims
    all_actions: List[str]           # Flat list of all action texts (for verification convenience)
    all_claims: List[str]            # Flat list of all claim texts (for verification convenience)
    chunks: List[Dict[str, Any]]     # Extracted chunks from search results
    chunk_score: Dict[str, Any]      # Chunk scoring data (matches original format)
    related_query: Dict[str, Any]    # Claim-to-query mappings
    top_k_chunks: Dict[str, Any]     # Top-k chunks per query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_list": self.query_list,
            "iterations": self.iterations,
            "report": self.report,
            "chunk_score": self.chunk_score,
            "related_query": self.related_query,
            "top_k_chunks": self.top_k_chunks,
        }


class TrajectoryDecomposer:
    """
    Decomposes research trajectories into atomic units.

    This class handles the decomposition of full research trajectories
    into smaller, verifiable units for hallucination evaluation.
    """

    SYSTEM_PROMPT_DOUBLE_CHECK_CLAIM = """
You are a quality control system specialized in validating and refining atomic claims for maximum precision and context-independence.

## TASK
Review claims and break down any non-atomic elements while ensuring complete context independence.

## DECOMPOSITION RULES
- **No conjunctions:** and, or, but, while, though, however
- **No conditionals:** if, when, unless
- **No compound statements:** Each claim must contain a single, indivisible fact

## CONTEXT-INDEPENDENCE RULES
- Claims must be verifiable independently without preceding or following context
- **Resolve all pronouns:** Replace "the", "this", "that", "it", "they" with specific entities
- **Explicit references:** If referencing "the position" or "this role", specify the exact entity
- **Context integration:** Use broader claim list context to provide necessary specificity
- **Exclusion rule:** If context cannot be determined, exclude the claim entirely

## EXAMPLES

**Decomposition:**
- Input: "Role available in Menlo Park, CA and Seattle, WA"
- Output:
  - Role available in Menlo Park, CA
  - Role available in Seattle, WA

**Context Independence:**
- Input: "The position focuses on experimenting with neural network architectures."
- Context: DeepMind Research Engineer/Scientist position
- Output: "DeepMind Research Engineer/Scientist position focuses on experimenting with neural network architectures"

## OUTPUT FORMAT
Return each atomic, context-independent claim on a new line with "- " prefix.
"""

    SYSTEM_PROMPT_PURE_OBSERVATION_DECOMPOSITION = """
You are an expert fact extraction system specialized in identifying and extracting concrete, verifiable claims from text.

## TASK
Extract ONLY factual, concrete claims that can be independently verified.

## EXTRACTION CRITERIA

### INCLUDE
- Specific facts, data, discoveries
- Concrete information (entities, locations, numbers)
- Definitive results and findings
- Context-independent statements

### EXCLUDE
- Speculative language (may, might, could, possibly, likely, appears, seems, suggests)
- Subjective opinions (effective, ideal, best, good, useful)
- Process summaries ("Progress has been made...", "We plan to...")
- Vague statements without specific results
- URLs in claims

## CONTEXT-INDEPENDENCE REQUIREMENT
Each claim must be completely self-contained and verifiable without surrounding context:

**Pronoun Resolution:**
- Replace "this", "that", "these", "the following", "it", "they" with specific referents
- Preserve clarifying modifiers, purpose clauses (to...), and qualifying details

**Verification Test:** Can someone verify this claim's truthfulness without reading the original paragraph?

## EXAMPLES

**Context-Independence:**
- "This approach improved performance by 15%" (BAD - unresolved pronoun)
- "The neural network optimization approach improved performance by 15%" (GOOD - specific referent)
- "They offer remote positions" (BAD)
- "Google offers remote positions" (GOOD)

**Atomic Claims:**
- "Meta has roles in Menlo Park AND Seattle" (BAD - conjunction)
- "Meta has a role in Menlo Park" (GOOD - single fact)
- "Meta has a role in Seattle" (GOOD - single fact)

## OUTPUT FORMAT
Return each atomic claim on a new line with "- " prefix.
If no extractable content: "No extractable content - paragraph contains only vague descriptions."
"""

    SYSTEM_PROMPT_DOUBLE_CHECK_ACTION = """
You are a quality control system specialized in validating and refining atomic actions for maximum precision and context-independence.

## TASK
Review actions and break down any non-atomic elements while ensuring complete context independence.

## DECOMPOSITION RULES
- **No conjunctions:** and, or, but, while, though, however
- **No conditionals:** if, when, unless
- **No compound statements:** Each action must contain a single, indivisible task

## CONTEXT-INDEPENDENCE RULES
- Actions must be verifiable independently without preceding or following context
- **Resolve all pronouns:** Replace "the", "this", "that", "it", "they" with specific entities
- **Explicit references:** If referencing "the position" or "this role", specify the exact entity
- **Context integration:** Use broader action list context to provide necessary specificity
- **Exclusion rule:** If context cannot be determined, exclude the action entirely

## EXAMPLE
- Input: "We will then deploy the model to the test server"
- Output: "Deploy the model to the test server"

## OUTPUT FORMAT
Return each atomic, context-independent action on a new line with "- " prefix.
"""

    SYSTEM_PROMPT_DECOMPOSITION = """You are an expert text analysis system specialized in extracting structured information from unstructured text.

## TASK
Deconstruct paragraphs to isolate and extract concrete factual observations and specific actionable plans through systematic fragmentation, classification, and atomic extraction.

## LANGUAGE REQUIREMENT
**CRITICAL: All output MUST be in English.**
If the input text is in Chinese or any other language, you MUST translate it EXACTLY to English.

## METHODOLOGY

### Step 1: Fragmentation (Minimal Splitting)
- Produce the smallest set of fragments that faithfully reflect the paragraph's explicit sentences or numbered steps.
- If a sentence mixes observation and plan, split only along that boundary; otherwise keep the sentence intact.

### Step 2: Classification
- **observation**: Facts, discoveries, or statements quoted from sources inside the paragraph.
- **plan**: Actions the agent explicitly states it will take next.

### Step 3: Atomic Extraction
- Keep each item self-contained and independently verifiable.
- Prefer to keep clauses together; only split truly parallel elements.

## OUTPUT FORMAT
```
Fragment 1: [fragment text]
Classification: [observation/plan]
Atomic [Claims/Actions]:
- [atomic item 1]
- [atomic item 2]
```
"""

    SYSTEM_PROMPT_QUERY_DECOMPOSITION = """You are an expert query analysis system specialized in extracting atomic constraints from user queries.

## TASK
Extract concise, independent atomic constraints from user queries.

## ATOMIC CONSTRAINT DEFINITION
An atomic constraint is:
- Single, self-contained, and indivisible with clear meaning representation
- Objective conditions or criteria only
- Each constraint must be independently verifiable and unambiguous

## OUTPUT FORMAT
Output each constraint on a new line prefixed with "- ".
"""

    def __init__(self, config):
        """
        Initialize the decomposer.

        Args:
            config: EvaluationConfig object
        """
        self.config = config
        self.llm_client = None

        # Only create LLM client if API key is valid
        api_key = config.llm.api_key if config.llm else None
        if api_key and api_key not in ("YOUR_API_KEY_HERE", "", None):
            try:
                self.llm_client = LLMClient.from_config(config.llm)
            except Exception as e:
                print(f"⚠️ Failed to initialize LLM client: {e}")

    def decompose(self, parsed_trajectory, cache_dir: str) -> Dict[str, Any]:
        """
        Decompose a trajectory into atomic units.

        Args:
            parsed_trajectory: ParsedTrajectory object
            cache_dir: Directory for caching intermediate results

        Returns:
            Dictionary with decomposed trajectory data
        """
        # Step 1: Decompose query into query_list (atomic constraints)
        print("  [1/4] Decomposing query into atomic constraints...")
        query_list = self._decompose_query(parsed_trajectory.query)

        # Step 2: Decompose chain of research into iterations
        # Format: [{action_list_1: [...], search_list_1: [...], claim_list_1: [...]}, ...]
        print("  [2/4] Decomposing research chain into iterations...")
        iterations_data = self._decompose_chain_of_research(
            parsed_trajectory.chain_of_research,
            parsed_trajectory.query
        )

        # Step 3: Decompose final report into paragraphs with atomic_claims
        # Format: [{paragraph_index, paragraph_text, atomic_claims}, ...]
        print("  [3/4] Decomposing final report into atomic claims...")
        report = self._decompose_report(
            parsed_trajectory.final_report,
            parsed_trajectory.query
        )

        # Step 4: Extract chunks from search results
        print("  [4/4] Extracting chunks from search results...")
        chunks = self._extract_chunks(parsed_trajectory.chain_of_research)

        # Build chunk_score dict (matches original cache format)
        chunk_score = {}
        for c in chunks:
            cid = c.get("chunk_id", "unknown")
            chunk_score[cid] = {
                "url": c.get("url", ""),
                "chunk_text": c.get("text", ""),
            }

        result = {
            "query_list": query_list,
            "iterations": iterations_data["iterations"],
            "report": report,
            "all_actions": iterations_data["actions"],
            "all_claims": iterations_data["claims"],
            "chunks": chunks,
            "chunk_score": chunk_score,
            "related_query": {},
            "top_k_chunks": {},
        }

        return result

    def _save_cache_partial(self, cache_file: str, data: Dict[str, Any]) -> None:
        """Save partial cache data during decomposition."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # Silent fail for partial cache saves

    def _save_cache_complete(self, cache_file: str, data: Dict[str, Any]) -> None:
        """Save complete cache data after decomposition is done."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose user query into atomic sub-queries/constraints."""
        if not self.llm_client:
            # Rule-based fallback: extract constraints from numbered/bulleted lists
            constraints = []
            lines = query.split('\n')
            # Track parent numbered item for sub-bullets
            parent_numbered = None
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # Match numbered items: "1. ...", "2. ..."
                m_num = re.match(r'^(\d+)\.\s+(.*)', stripped)
                if m_num:
                    parent_numbered = m_num.group(2).rstrip(':').strip()
                    constraints.append(parent_numbered)
                    continue
                # Match sub-bullets: "- ..." under a numbered item
                if stripped.startswith('- ') and parent_numbered:
                    sub_item = stripped[2:].strip()
                    constraint = f"{parent_numbered}: {sub_item}"
                    constraints.append(constraint)
                    continue
                # Standalone bullet
                if stripped.startswith('- ') and not parent_numbered:
                    constraints.append(stripped[2:].strip())
                    continue
                # Non-numbered, non-bullet line -> reset parent context (section boundary)
                parent_numbered = None
            if constraints:
                return constraints
            return [query]

        prompt = f"""Extract all atomic constraint conditions from the following query.

Query: {query}

Output each constraint on a new line prefixed with "- ".
"""

        try:
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT_QUERY_DECOMPOSITION,
                user_prompt=prompt,
            )

            constraints = []
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    constraints.append(line[2:].strip())

            return constraints if constraints else [query]
        except Exception as e:
            print(f"⚠️ Query decomposition failed: {e}")
            return [query]

    def _decompose_chain_of_research(
        self, chain: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """
        Decompose the chain of research into iterations matching original format.

        Returns:
            Dictionary with:
            - iterations: list of dicts with action_list_N, search_list_N, claim_list_N
            - actions: flat list of action strings
            - claims: flat list of claim strings
        """
        # Find all iteration numbers
        iter_numbers = set()
        for key in chain.keys():
            match = re.match(r'^(plan|search|observation|reasoning)_(\d+)$', key)
            if match:
                iter_numbers.add(int(match.group(2)))
        iter_numbers = sorted(iter_numbers)

        # Determine pattern: plan+search+obs VS reasoning+search
        has_reasoning = any(k.startswith('reasoning_') for k in chain.keys())
        has_planning = any(k.startswith('plan_') for k in chain.keys())

        if has_reasoning and not has_planning:
            return self._decompose_reasoning_chain(chain, query, iter_numbers)
        else:
            return self._decompose_planning_chain(chain, query, iter_numbers)

    def _decompose_reasoning_chain(
        self, chain: Dict[str, Any], query: str, iter_numbers: List[int]
    ) -> Dict[str, Any]:
        """
        Decompose reasoning+search pattern (matches original decomposition.py logic).

        Iteration count = number of search keys (reference: decomposition.py:1236).
        reasoning_1: planning_items -> action_list_1
        reasoning_x (x>1): obs -> claim_list_{x-1}, planning_items -> action_list_x
        Extra reasoning past last search -> obs appended to final claim_list, planning skipped.
        """
        # Count iterations by search keys (reference line 1236)
        search_keys = sorted(
            [k for k in chain.keys() if k.startswith('search_')],
            key=lambda x: int(x.split('_')[1])
        )
        num_iters = len(search_keys)

        # Build empty iteration slots: action_list_i, search_list_i, claim_list_i
        iterations = []
        for i in range(1, num_iters + 1):
            iterations.append({
                f"action_list_{i}": [],
                f"search_list_{i}": chain.get(f"search_{i}", []),
                f"claim_list_{i}": [],
            })

        all_actions = []
        all_claims = []

        reasoning_steps = sorted(
            [k for k in chain.keys() if k.startswith('reasoning_')],
            key=lambda x: int(x.split('_')[1])
        )

        for reasoning_key in reasoning_steps:
            reasoning_num = int(reasoning_key.split('_')[1])
            reasoning_text = chain.get(reasoning_key, "")
            if not reasoning_text or not isinstance(reasoning_text, str):
                continue

            items = self._decompose_reasoning_text(reasoning_text, query)
            obs_items = [a["claim"] if isinstance(a, dict) else str(a) for a in items.get("claims", [])]
            plan_items = [a["action"] if isinstance(a, dict) else str(a) for a in items.get("actions", [])]

            if reasoning_num == 1:
                # For reasoning_1, planning_items -> action_list_1
                iterations[0][f"action_list_{1}"].extend(plan_items)
            else:
                # For reasoning_x (x>1): obs -> claim_list_{x-1}
                if reasoning_num - 2 < num_iters:
                    iterations[reasoning_num - 2][f"claim_list_{reasoning_num - 1}"].extend(obs_items)
                else:
                    # Extra reasoning after last search -> obs to final claim_list; skip planning
                    if num_iters > 0:
                        iterations[-1][f"claim_list_{num_iters}"].extend(obs_items)

                # plan -> action_list_x (only if within search bounds)
                if reasoning_num - 1 < num_iters:
                    iterations[reasoning_num - 1][f"action_list_{reasoning_num}"].extend(plan_items)

            all_actions.extend([{"action": p, "iteration": reasoning_num} for p in plan_items])
            all_claims.extend([{"claim": o, "iteration": reasoning_num} for o in obs_items])

        return {"iterations": iterations, "actions": all_actions, "claims": all_claims}

    def _decompose_planning_chain(
        self, chain: Dict[str, Any], query: str, iter_numbers: List[int]
    ) -> Dict[str, Any]:
        """Decompose plan+search+observation pattern."""
        iterations = []
        all_actions = []
        all_claims = []

        for iter_num in iter_numbers:
            iter_data = {
                f"action_list_{iter_num}": [],
                f"search_list_{iter_num}": chain.get(f"search_{iter_num}", []),
                f"claim_list_{iter_num}": [],
            }

            plan_text = chain.get(f"plan_{iter_num}", "")
            if plan_text and isinstance(plan_text, str):
                actions = self._decompose_planning_text(plan_text, query)
                action_texts = [a["action"] if isinstance(a, dict) else str(a) for a in actions]
                iter_data[f"action_list_{iter_num}"] = action_texts
                all_actions.extend(actions)

            obs_text = chain.get(f"observation_{iter_num}", "")
            if obs_text and isinstance(obs_text, str):
                claims = self._decompose_observation_text(obs_text, query)
                claim_texts = [c["claim"] if isinstance(c, dict) else str(c) for c in claims]
                iter_data[f"claim_list_{iter_num}"] = claim_texts
                all_claims.extend(claims)

            iterations.append(iter_data)

        return {"iterations": iterations, "actions": all_actions, "claims": all_claims}

    def _decompose_planning_text(self, text: str, query: str) -> List[Dict[str, Any]]:
        """Decompose planning text into atomic actions."""
        if not self.llm_client:
            return [{"action": text, "iteration": 0}]

        prompt = f"""Query: {query}

Paragraph to analyze: {text}

Classify this paragraph and decompose it into atomic actions (plans)."""

        try:
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT_DECOMPOSITION,
                user_prompt=prompt,
            )

            actions = []
            current_section = None

            for line in response.splitlines():
                line = line.strip()
                if line.startswith("Classification:"):
                    current_section = line.split(":")[1].strip().lower()
                elif line.startswith("- ") and current_section == "plan":
                    action_text = line[2:].strip()
                    if action_text:
                        actions.append({"action": action_text, "type": "plan"})

            if actions:
                action_texts = [a["action"] for a in actions]
                checked = self._double_check_actions(action_texts, query)
                if len(checked) != len(action_texts):
                    actions = [{"action": a, "type": "plan"} for a in checked]
            return actions if actions else [{"action": text, "type": "plan"}]
        except Exception as e:
            print(f"⚠️ Planning decomposition failed: {e}")
            return [{"action": text, "type": "plan"}]

    def _decompose_observation_text(self, text: str, query: str) -> List[Dict[str, Any]]:
        """Decompose observation text into atomic claims."""
        if not self.llm_client:
            return [{"claim": text, "iteration": 0}]

        prompt = f"""Query: {query}

Paragraph to analyze: {text}

Extract all factual claims from this paragraph."""

        try:
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT_PURE_OBSERVATION_DECOMPOSITION,
                user_prompt=prompt,
            )

            claims = []
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    claim_text = line[2:].strip()
                    if claim_text and claim_text != "No extractable content - paragraph contains only vague descriptions.":
                        claims.append({"claim": claim_text, "type": "observation"})

            if claims:
                claim_texts = [c["claim"] for c in claims]
                checked = self._double_check_claims(claim_texts, query)
                if len(checked) != len(claim_texts):
                    claims = [{"claim": c, "type": "observation"} for c in checked]
            return claims if claims else [{"claim": text, "type": "observation"}]
        except Exception as e:
            print(f"⚠️ Observation decomposition failed: {e}")
            return [{"claim": text, "type": "observation"}]

    def _decompose_reasoning_text(self, text: str, query: str) -> Dict[str, List]:
        """Decompose reasoning text (mixed format) into actions and claims."""
        if not self.llm_client:
            # Fallback: return raw text as single action (reference always uses LLM)
            return {"actions": [{"action": text, "type": "reasoning"}], "claims": []}

        prompt = f"""Query: {query}

Paragraph to analyze: {text}

Classify this paragraph and decompose it into atomic claims and actions."""

        try:
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT_DECOMPOSITION,
                user_prompt=prompt,
            )

            actions = []
            claims = []
            current_section = None

            for line in response.splitlines():
                line = line.strip()
                if line.startswith("Classification:"):
                    current_section = line.split(":")[1].strip().lower()
                elif line.startswith("- "):
                    item_text = line[2:].strip()
                    if item_text:
                        if current_section == "plan":
                            actions.append({"action": item_text})
                        elif current_section == "observation":
                            claims.append({"claim": item_text})

            return {"actions": actions, "claims": claims}
        except Exception as e:
            print(f"⚠️ Reasoning decomposition failed: {e}")
            return {"actions": [{"action": text}], "claims": []}

    @staticmethod
    def _split_report_into_paragraphs(report: str) -> List[str]:
        """
        Split report into paragraphs using the same 3-level fallback as the original HalluBench.
        Reference: HalluBench_backup_0828/HalluDetector/script_reframe/utils.py:342
        1. Split by '\\n\\n' if present
        2. Split by Markdown headers line-by-line
        3. Inline header pattern
        4. Fallback: return the whole report as one paragraph
        """
        # Level 1: Double newlines
        if '\n\n' in report:
            paragraphs = [p.strip() for p in report.split('\n\n') if p.strip()]
            if len(paragraphs) > 1:
                return paragraphs

        lines = report.split('\n')

        # Level 2: Markdown headers at start of line
        if len(lines) > 1:
            paragraphs = []
            current = []
            for line in lines:
                if re.match(r'^#{1,4}\s+.+', line.strip()):
                    if current:
                        para_text = '\n'.join(current).strip()
                        if para_text:
                            paragraphs.append(para_text)
                        current = []
                    current.append(line)
                else:
                    current.append(line)
            if current:
                para_text = '\n'.join(current).strip()
                if para_text:
                    paragraphs.append(para_text)
            if len(paragraphs) > 1:
                return paragraphs

        # Level 3: Inline Markdown headers (headers within a single line)
        header_pattern_inline = r'(?:^|\s)(#{1,4}\s+)'
        matches = list(re.finditer(header_pattern_inline, report))
        if len(matches) > 1:
            paragraphs = []
            for i, match in enumerate(matches):
                if i + 1 < len(matches):
                    content_end = matches[i + 1].start()
                else:
                    content_end = len(report)
                start_pos = match.start()
                if start_pos > 0 and report[start_pos] == ' ':
                    start_pos = match.start()
                else:
                    start_pos = match.start() if match.start() == 0 else match.start()
                para_text = report[start_pos:content_end].strip()
                if para_text:
                    paragraphs.append(para_text)
            if len(paragraphs) > 1:
                return paragraphs

        # Level 4: Fallback — return whole report as one paragraph
        return [report.strip()] if report.strip() else []

    def _decompose_report(self, report: str, query: str, max_claims: int = 200) -> List[Dict[str, Any]]:
        """Decompose the final report into paragraphs with atomic_claims (original format)."""
        if not report:
            return []

        paragraphs = self._split_report_into_paragraphs(report)
        report_data = []

        for i, para in enumerate(paragraphs, 1):
            entry = {
                "paragraph_index": i,
                "paragraph_text": para,  # Full text, not truncated
                "atomic_claims": [],
            }

            if not self.llm_client:
                # Fallback: sentence-based extraction
                sentences = re.split(r'(?<=[.!?])\s+', para)
                claims = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 300]
                entry["atomic_claims"] = claims[:max_claims]
                report_data.append(entry)
                continue

            # Only use LLM for first 5 paragraphs
            if i <= 5:
                prompt = f"""Query: {query}

Paragraph to analyze: {para[:1000]}

Extract all factual claims from this paragraph."""
                try:
                    response = self.llm_client.generate(
                        system_prompt=self.SYSTEM_PROMPT_DECOMPOSITION,
                        user_prompt=prompt,
                    )
                    claims = self._parse_decomposition_response(response, "observation")
                    if claims:
                        entry["atomic_claims"] = claims
                    else:
                        # Fallback: sentence-based
                        sentences = re.split(r'(?<=[.!?])\s+', para)
                        entry["atomic_claims"] = [s.strip() for s in sentences if 20 < len(s.strip()) < 300][:20]
                except Exception as e:
                    print(f"⚠️ Report decomposition failed for paragraph {i}: {e}")
                    entry["error"] = str(e)
            else:
                # Sentence-based for remaining paragraphs
                sentences = re.split(r'(?<=[.!?])\s+', para)
                entry["atomic_claims"] = [s.strip() for s in sentences if 20 < len(s.strip()) < 300][:20]

            if sum(len(r["atomic_claims"]) for r in report_data) >= max_claims:
                print(f"    Reached max_claims limit ({max_claims}), stopping report decomposition")
                break

            report_data.append(entry)

        return report_data

    def _double_check_claims(self, claims: List[str], query: str) -> List[str]:
        """Double-check atomic claims for non-atomic elements (reference: double_check_atomic_claims_with_client)."""
        if not claims or not self.llm_client:
            return claims
        try:
            prompt = f"""Query: {query}

Please review and double-check the following atomic claims for any non-atomic elements:

Claims to review:
{chr(10).join([f"- {c}" for c in claims])}

Return only valid atomic claims, breaking down any non-atomic ones into separate atomic parts."""
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT_DOUBLE_CHECK_CLAIM,
                user_prompt=prompt,
            )
            filtered = []
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    claim = line[2:].strip()
                    if claim and self._is_valid_claim(claim):
                        filtered.append(claim)
            return filtered if filtered else claims
        except Exception:
            return claims

    def _double_check_actions(self, actions: List[str], query: str) -> List[str]:
        """Double-check atomic actions (reference: SYSTEM_PROMPT_DOUBLE_CHECK_ACTION)."""
        if not actions or not self.llm_client:
            return actions
        try:
            prompt = f"""Query: {query}

Please review and double-check the following atomic actions for any non-atomic elements:

Actions to review:
{chr(10).join([f"- {a}" for a in actions])}

Return only valid atomic actions, breaking down any non-atomic ones into separate atomic parts."""
            response = self.llm_client.generate(
                system_prompt=self.SYSTEM_PROMPT_DOUBLE_CHECK_ACTION,
                user_prompt=prompt,
            )
            filtered = []
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    action = line[2:].strip()
                    if action:
                        filtered.append(action)
            return filtered if filtered else actions
        except Exception:
            return actions

    BANNED_USER_WORDS = (
        "the user", "the requester", "the client", "the customer",
        "i ", "i'm", "i'll", "i've", "i'd", "my ", "me ",
        "we ", "we're", "we'll", "we've", "we'd", "our ",
    )

    def _is_valid_claim(self, claim: str) -> bool:
        """Filter claims with banned user-reference words (reference: BANNED_USER_WORDS)."""
        txt = claim.lower()
        return not any(w in txt for w in self.BANNED_USER_WORDS)

    def _parse_decomposition_response(self, response: str, target_section: str) -> List[str]:
        """Parse LLM decomposition response into items list."""
        items = []
        current_section = None
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("Classification:"):
                current_section = line.split(":")[1].strip().lower()
            elif line.startswith("- ") and current_section == target_section:
                item = line[2:].strip()
                if item and len(item) > 10:
                    items.append(item)
        return items

    def _extract_chunks(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract chunks from search results."""
        chunks = []

        # Find all search keys
        search_keys = [k for k in chain.keys() if k.startswith("search_")]

        for search_key in sorted(search_keys, key=lambda x: int(x.split("_")[1])):
            search_data = chain[search_key]
            iter_num = int(search_key.split("_")[1])

            if isinstance(search_data, list):
                for i, chunk in enumerate(search_data):
                    if isinstance(chunk, dict):
                        chunks.append({
                            "chunk_id": f"iter{iter_num}_chunk{i}",
                            "text": chunk.get("text", ""),
                            "url": chunk.get("url", ""),
                            "iteration": iter_num,
                        })
                    elif isinstance(chunk, str):
                        chunks.append({
                            "chunk_id": f"iter{iter_num}_chunk{i}",
                            "text": chunk,
                            "url": chunk,  # URL string is the URL itself
                            "iteration": iter_num,
                        })
            elif isinstance(search_data, str):
                chunks.append({
                    "chunk_id": f"iter{iter_num}_chunk0",
                    "text": search_data,
                    "url": "",
                    "iteration": iter_num,
                })

        return chunks
