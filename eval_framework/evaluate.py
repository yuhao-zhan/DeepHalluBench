#!/usr/bin/env python3
"""
DeepHalluBench Evaluation Entry Point

Evaluates research trajectories for hallucinations using the PING taxonomy.

Usage:
    python -m eval_framework.evaluate <trajectory.json>
    python -m eval_framework.evaluate <directory/> --output-dir ./results

Environment Variables:
    LLM_API_KEY      Your API key for LLM calls
    LLM_PROVIDER     Provider: openai, anthropic, aliyun, deepseek
    LLM_MODEL        Model name (default: gpt-4o)
    LLM_BASE_URL     API base URL (auto-set based on provider)
    NLI_MODEL        NLI model name
    NLI_THRESHOLD    NLI confidence threshold (default: 0.95)
    GPU_IDS          Comma-separated GPU IDs (default: 0)
    CACHE_DIR        Cache directory (default: ./cache)
    OUTPUT_DIR       Output directory (default: ./results)
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="DeepHalluBench: Hallucination Evaluation for Deep Research Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval_framework.evaluate demos/apple_watch.json
  python -m eval_framework.evaluate demos/ --output-dir ./results
  python -m eval_framework.evaluate trajectory.json --skip-noise --sample-claims 10
        """
    )

    parser.add_argument(
        "input",
        help="Path to trajectory JSON file or directory containing JSON files"
    )
    parser.add_argument(
        "--output-dir", default="./results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--cache-dir", default="./cache",
        help="Cache directory for intermediate results"
    )
    parser.add_argument(
        "--config", help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--skip-noise", action="store_true",
        help="Skip noise domination detection"
    )
    parser.add_argument(
        "--skip-actions", action="store_true",
        help="Skip action verification"
    )
    parser.add_argument(
        "--skip-claims", action="store_true",
        help="Skip claim verification"
    )
    parser.add_argument(
        "--sample-claims", type=int, default=None,
        help="Sample only N claims for verification (demo mode)"
    )
    parser.add_argument(
        "--sample-actions", type=int, default=None,
        help="Sample only N actions for verification (demo mode)"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--gpu-ids", default="0",
        help="Comma-separated GPU IDs (default: 0)"
    )

    args = parser.parse_args()

    os.environ.setdefault("CACHE_DIR", args.cache_dir)
    os.environ.setdefault("NUM_GPUS", str(args.num_gpus))
    os.environ.setdefault("GPU_IDS", args.gpu_ids)

    # Import evaluator
    try:
        from eval_framework import DeepHalluBench, ConfigManager
    except ImportError as e:
        print(f"Failed to import DeepHalluBench: {e}")
        sys.exit(1)

    # Initialize
    try:
        if args.config:
            evaluator = DeepHalluBench.from_file(args.config)
        else:
            evaluator = DeepHalluBench.from_env()
    except Exception as e:
        print(f"Failed to initialize evaluator: {e}")
        sys.exit(1)

    input_path = Path(args.input)

    # Collect JSON files
    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in: {input_path}")
            sys.exit(1)
    elif input_path.is_file():
        json_files = [input_path]
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)

    print(f"Found {len(json_files)} file(s) to evaluate\n")

    for json_file in json_files:
        print(f"{'='*80}")
        print(f"Processing: {json_file.name}")
        print(f"{'='*80}")

        try:
            result = evaluator.evaluate(
                trajectory=str(json_file),
                output_dir=args.output_dir,
                max_claims=args.sample_claims,
                max_actions=args.sample_actions,
                skip_noise=args.skip_noise,
                skip_actions=args.skip_actions,
                skip_claims=args.skip_claims,
            )
            print(f"\nOverall Hallucination Score: {result['overall_score']:.4f}")
        except Exception as e:
            print(f"Evaluation failed for {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("All evaluations complete.")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
