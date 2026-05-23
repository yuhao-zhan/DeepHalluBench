# DeepHalluBench

**Hallucination Evaluation for Deep Research Agents**

A process-aware evaluation framework for detecting and quantifying hallucinations in full research trajectories using the **PING Taxonomy**.

## Overview

Deep Research Agents (DRAs) often produce hallucinations that accumulate throughout the research process. Existing benchmarks evaluate outcomes but miss intermediate failures. DeepHalluBench introduces **process-aware evaluation** by auditing the full research trajectory.

### PING Taxonomy


| Category          | Description                                  | Error Types                           |
| ----------------- | -------------------------------------------- | ------------------------------------- |
| **P**ropagation   | Cascading errors from earlier hallucinations | Trajectory-level propagation          |
| **I**ntent        | Planning-stage failures                      | Restriction neglect, Action deviation |
| **N**oise-induced | Failure to prioritize informative evidence   | Neglected important retrieval         |
| **G**rounding     | Claims unsupported by evidence               | Fabrication, Misattribution           |


## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for NLI/reranking models)

### Quick Install

```bash
pip install -r requirements.txt
playwright install chromium
```

### Environment Variables

Configure your API keys and preferences:

```bash
# LLM Configuration
export LLM_API_KEY="your-api-key"
export LLM_PROVIDER="openai"  # openai, anthropic, aliyun, deepseek
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL=""  # Auto-set based on provider

# NLI Model (for claim verification)
export NLI_MODEL="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
export NLI_THRESHOLD=0.95

# GPU Configuration
export GPU_IDS="0,1,2,3"
export NUM_GPUS=4

# Directories
export CACHE_DIR="./cache"
export OUTPUT_DIR="./results"
```

## Quick Start

### 1. Prepare Your Trajectory

The evaluation pipeline accepts **JSON trajectories** with the following format:

```json
{
    "query": "Your research question",
    "final_report": "The agent's final research report",
    "all_source_links": ["url1", "url2", ...],
    "summary_citations": ["url1", "url3", ...],
    "chain_of_research": {
        "plan_1": "Planning text for iteration 1",
        "search_1": [{"chunk_id": "c1", "text": "...", "url": "..."}],
        "observation_1": "Observation text after search 1",
        "plan_2": "Planning text for iteration 2",
        ...
    }
}
```

### 2. Parse HTML to JSON (Optional)

If you have raw HTML traces from supported model providers, use the model-specific parsers to convert them into structured JSON trajectories:

```python
from parsers import (
    parse_openai_html,         # ChatGPT Deep Research
    parse_grok_html,           # Grok
    parse_perplexity_html,     # Perplexity
    parse_gemini_html,         # Gemini (Mind2Web2)
)

# Example: parse an OpenAI HTML trace
with open("openai_trace.html") as f:
    result = parse_openai_html(f.read())
# result contains: query, final_report, all_source_links, summary_citations, chain_of_research
```

**Debugging parsers**: Since parser implementations are model-specific and HTML structures may change over time, you may encounter empty attributes in the parsed output. If this happens:

- Use vibe coding to quickly debug — the parsers are straightforward scripts that extract text from HTML elements, so adjusting selectors or regex patterns is usually all that's needed.
- Alternatively, file an issue on GitHub and the authors will update the parsers.

### 3. Run Evaluation

```bash
# Basic usage
python -m eval_framework.evaluate trajectory.json "Your query"

# With custom output
python -m eval_framework.evaluate trajectory.json "Your query" \
    --output-dir ./results \
    --cache-dir ./cache

# Batch evaluation
python -m eval_framework.evaluate ./trajectories/ "Optional query" \
    --output-dir ./results
```

### 4. Using the Python API

```python
from eval_framework import DeepHalluBench

# Initialize from environment
evaluator = DeepHalluBench.from_env()

# Or from config file
evaluator = DeepHalluBench.from_file("config.json")

# Run evaluation
results = evaluator.evaluate("trajectory.json", "Your query")

# Access PING scores
print(results["ping_scores"])
# {
#   "grounding": 0.12,
#   "noise_induced": 0.08,
#   "intent": 0.15,
#   "propagation": 0.05,
# }
print(f"Overall Score: {results['overall_score']:.4f}")
```

## Project Structure

```
DeepHalluBench/
├── README.md                    # This file
├── paper.pdf                    # Paper PDF
├── config_example.json          # Example configuration
│
├── eval_framework/             # Main evaluation package
│   ├── __init__.py              # Main evaluator
│   ├── evaluate.py              # CLI entry point
│   ├── config.py               # Configuration
│   ├── core/                   # Core evaluation modules
│   │   ├── decomposition.py     # Trajectory decomposition
│   │   ├── claim_verification.py  # Grounding detection
│   │   ├── action_checking.py  # Intent detection
│   │   ├── noise_domination.py # Noise-induced detection
│   │   └── constraint_checking.py  # Restriction neglect
│   └── utils/
│       ├── __init__.py          # LLM API client
│       ├── scoring.py           # Reranker utilities
│       └── web_fetcher.py       # Web content fetching
│
├── parsers/                    # Model-specific HTML trace parsers
│   ├── __init__.py
│   ├── openai.py               # ChatGPT Deep Research parser
│   ├── grok_new.py             # Grok parser
│   ├── perplexity.py           # Perplexity parser
│   └── gemini.py               # Gemini (Mind2Web2) parser
│
├── data/
│   └── DeepHalluBench.jsonl    # Benchmark queries (100 queries)
│
└── cache/                      # Cache directory (created at runtime)
```

## Configuration

### Configuration File

Create a `config.json`:

```json
{
    "llm": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1",
        "max_tokens": 4096,
        "temperature": 0.1
    },
    "nli": {
        "model_name": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "threshold": 0.95
    },
    "embedding": {
        "model_name": "BAAI/bge-m3"
    },
    "reranker": {
        "model_name": "BAAI/bge-reranker-v2-m3"
    },
    "nli_threshold": 0.95,
    "top_k_chunks": 5,
    "gpu_ids": [0, 1, 2, 3],
    "num_gpus": 4,
    "cache_dir": "./cache",
    "output_dir": "./results"
}
```

### Supported LLM Providers


| Provider  | Model Examples          | Notes                        |
| --------- | ----------------------- | ---------------------------- |
| OpenAI    | `gpt-4o`, `gpt-4-turbo` | Set `LLM_PROVIDER=openai`    |
| Anthropic | `claude-3-5-sonnet`     | Set `LLM_PROVIDER=anthropic` |
| Aliyun    | `deepseek-v3`, `qwen`   | Set `LLM_PROVIDER=aliyun`    |
| DeepSeek  | `deepseek-chat`         | Set `LLM_PROVIDER=deepseek`  |


## Evaluation Pipeline

The evaluation consists of five stages:

### 1. Trajectory Decomposition

Decomposes the research trajectory into atomic units:

- **Sub-queries**: Atomic constraints from the user query
- **Actions**: Atomic planning steps
- **Claims**: Atomic factual statements from observations and reports
- **Chunks**: Retrieved evidence chunks

### 2. Claim Verification (Grounding)

Two-round verification pipeline:

**Round 1**: Initial Verification

- Retrieve top-K chunks via embedding + reranking
- Verify with NLI-then-LLM cascade
- High confidence (>0.99 NLI) = finalize

**Round 2**: Adaptive Re-Verification

- Misattribution check: expand evidence scope
- Reflection check: verify against Claim Memory

### 3. Noise Domination Detection

Quantifies failure to prioritize informative evidence:

1. Semantic clustering of retrieved chunks
2. Rank clusters by relevance to sub-queries
3. Penalize neglected high-ranking clusters

### 4. Action Verification (Intent)

History-aware verification:

- Detect action deviation from user intent
- Detect action propagation from hallucinated claims

### 5. Constraint Checking (Restriction Neglect)

- Extract atomic constraints from query
- Rank action relevance to each constraint
- Apply elbow method to identify addressed restrictions

## Benchmark Data

The `data/DeepHalluBench.jsonl` file contains 100 benchmark queries spanning diverse domains. Each entry has `question`, `answer`, and `domain` fields.

Example query:

```json
{"question": "Research all currency information in all the 'The Legend of Heroes' series.", "answer": "", "domain": "Entertainment & Gaming"}
```

## Output Format

```json
{
  "query": "User's research query",
  "ping_scores": {
    "grounding": 0.12,
    "noise_induced": 0.08,
    "intent": 0.15,
    "propagation": 0.05
  },
  "overall_score": 0.10,
  "detailed_results": {
    "claim_verification": {
      "total": 45,
      "support": 38,
      "fabrication": 5,
      "misattribution": 2
    },
    "noise_domination": {
      "score": 0.08,
      "mapped_clusters": [...],
      "unmapped_clusters": [...]
    },
    "action_verification": {
      "total": 12,
      "deviation": 2,
      "propagation": 1
    },
    "constraint_checking": {
      "total_queries": 5,
      "missed_count": 1,
      "missed_queries": [...],
      "first_cluster_queries": {...}
    }
  }
}
```

## Troubleshooting

### Common Issues

**"LLM_API_KEY not set"**

```bash
export LLM_API_KEY="your-key"
```

**"NLI model not loading"**

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Try installing transformers directly: `pip install transformers[torch]`

**"Out of memory"**

- Reduce `top_k_chunks` in config
- Use fewer GPUs by setting `GPU_IDS="0"`

### Debug Mode

Enable verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use DeepHalluBench in your research, please cite:

```bibtex
@misc{zhan2026deepresearchagentfails,
      title={Why Your Deep Research Agent Fails? On Hallucination Evaluation in Full Research Trajectory}, 
      author={Yuhao Zhan and Tianyu Fan and Linxuan Huang and Zirui Guo and Chao Huang},
      year={2026},
      eprint={2601.22984},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.22984}, 
}
```

## License

MIT License

## Acknowledgments

This project builds on research in hallucination detection, NLI verification, and semantic clustering. See the paper for full references.