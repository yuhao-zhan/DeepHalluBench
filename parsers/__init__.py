"""
Model-specific HTML Trajectory Parsers for DeepHalluBench

Parsers to convert raw Web UI traces (HTML) from specific model providers
into structured JSON trajectories for the evaluation pipeline.

Supported models:
- OpenAI (ChatGPT deep research)
- Grok
- Perplexity
- Gemini (Mind2Web2)

All entry points share the same signature:

    parse_<model>_html(html_content: str) -> dict

    Returns dict with keys: query, final_report, all_source_links,
    summary_citations, chain_of_research
"""

import json
from .base_parser import ParsedTrajectory
from .openai import parse_openai_html
from .grok_new import parse_grok_html
from .perplexity import parse_perplexity_html
from .gemini import parse_gemini_html


import re as _re

class JSONTrajectoryParser:
    """Parse structured JSON trajectory files into ParsedTrajectory objects.

    Accepts two formats:
    1. Nested: chain_of_research = {reasoning_1, search_1, plan_1, observation_1, ...}
    2. Flat:   reasoning_1, search_1, ... directly at top level (auto-detected)
    """

    @staticmethod
    def _auto_nest_chain(data: dict) -> dict:
        """If reasoning_/search_/plan_/observation_ keys are at top level,
        nest them into chain_of_research."""
        chain_keys = {k for k in data if _re.match(
            r'^(reasoning|search|plan|observation)_\d+$', k)}
        if not chain_keys or data.get("chain_of_research"):
            return data.get("chain_of_research", {})
        return {k: data[k] for k in sorted(
            chain_keys, key=lambda x: int(x.split('_')[-1]))}

    def parse(self, path: str) -> ParsedTrajectory:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ParsedTrajectory(
            query=data.get("query", data.get("question", "")),
            final_report=data.get("final_report", ""),
            all_source_links=data.get("all_source_links", []),
            summary_citations=data.get("summary_citations", []),
            chain_of_research=self._auto_nest_chain(data),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "JSONTrajectoryParser",
    "parse_openai_html",
    "parse_grok_html",
    "parse_perplexity_html",
    "parse_gemini_html",
]
