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


class JSONTrajectoryParser:
    """Parse structured JSON trajectory files into ParsedTrajectory objects."""

    def parse(self, path: str) -> ParsedTrajectory:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ParsedTrajectory(
            query=data.get("query", ""),
            final_report=data.get("final_report", ""),
            all_source_links=data.get("all_source_links", []),
            summary_citations=data.get("summary_citations", []),
            chain_of_research=data.get("chain_of_research", {}),
            metadata=data.get("metadata", {}),
        )


__all__ = [
    "JSONTrajectoryParser",
    "parse_openai_html",
    "parse_grok_html",
    "parse_perplexity_html",
    "parse_gemini_html",
]
