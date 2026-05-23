from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class ParsedTrajectory:
    """Container for parsed trajectory data."""
    query: str
    final_report: str
    all_source_links: List[str]
    summary_citations: List[str]
    chain_of_research: Dict[str, Any]
    metadata: Dict[str, Any]
