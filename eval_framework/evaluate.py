#!/usr/bin/env python3
"""
DeepHalluBench Evaluation Entry Point

Evaluates research trajectories for hallucinations using the PING taxonomy.

Usage:
    python -m eval_framework.evaluate trajectory.json "query" --output-dir ./results

Environment Variables:
    LLM_API_KEY: API key for LLM calls
    LLM_PROVIDER: Provider name (openai, anthropic, aliyun, deepseek)
    LLM_MODEL: Model name (default: gpt-4o)
    NLI_MODEL: NLI model name (default: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
    NLI_THRESHOLD: NLI confidence threshold (default: 0.95)
    GPU_IDS: Comma-separated GPU IDs (default: 0)
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
  # Evaluate a trajectory
  python -m eval_framework.evaluate trajectory.json "Your research query"

  # Evaluate with custom output
  python -m eval_framework.evaluate trajectory.json "Query" --output-dir ./results

  # Skip specific evaluations
  python -m eval_framework.evaluate trajectory.json "Query" --skip-noise --skip-actions

Environment Variables:
  LLM_API_KEY         Your API key for LLM calls
  LLM_PROVIDER        Provider: openai, anthropic, aliyun, deepseek
  LLM_MODEL           Model name (default: gpt-4o)
  LLM_BASE_URL        API base URL (auto-set based on provider)
  NLI_MODEL           NLI model name
  NLI_THRESHOLD       NLI confidence threshold (default: 0.95)
  GPU_IDS             Comma-separated GPU IDs (default: 0)
  CACHE_DIR           Cache directory (default: ./cache)
  OUTPUT_DIR           Output directory (default: ./results)
        """
    )

    parser.add_argument(
        "trajectory",
        help="Path to trajectory JSON file or directory containing trajectory files"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="User query (optional if provided in trajectory file)"
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--cache-dir",
        default="./cache",
        help="Cache directory for intermediate results"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip noise domination detection"
    )
    parser.add_argument(
        "--skip-actions",
        action="store_true",
        help="Skip action verification"
    )
    parser.add_argument(
        "--skip-claims",
        action="store_true",
        help="Skip claim verification"
    )
    parser.add_argument(
        "--sample-claims",
        type=int,
        default=None,
        help="Sample only N claims for verification (demo mode)"
    )
    parser.add_argument(
        "--sample-actions",
        type=int,
        default=None,
        help="Sample only N actions for verification (demo mode)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Comma-separated GPU IDs (default: 0)"
    )

    args = parser.parse_args()

    # Set environment variables
    if args.cache_dir:
        os.environ["CACHE_DIR"] = args.cache_dir
    if args.num_gpus:
        os.environ["NUM_GPUS"] = str(args.num_gpus)
    if args.gpu_ids:
        os.environ["GPU_IDS"] = args.gpu_ids

    # Import DeepHalluBench
    try:
        from eval_framework import DeepHalluBench, EvaluationConfig, ConfigManager
    except ImportError as e:
        print(f"❌ Failed to import DeepHalluBench: {e}")
        print("   Make sure you're running from the correct directory or install the package")
        sys.exit(1)

    # Initialize evaluator
    try:
        if args.config:
            print(f"📋 Loading configuration from: {args.config}")
            evaluator = DeepHalluBench.from_file(args.config)
        else:
            print("📋 Using environment variable configuration")
            evaluator = DeepHalluBench.from_env()
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        sys.exit(1)

    # Check if trajectory is a file or directory
    trajectory_path = Path(args.trajectory)

    if trajectory_path.is_dir():
        # Process all JSON files in directory
        json_files = list(trajectory_path.glob("*.json")) + list(trajectory_path.glob("*.jsonl"))
        if not json_files:
            print(f"❌ No JSON files found in directory: {trajectory_path}")
            sys.exit(1)

        print(f"📂 Found {len(json_files)} trajectory files to process")

        results = []
        for json_file in json_files:
            print(f"\n{'='*80}")
            print(f"Processing: {json_file.name}")
            print(f"{'='*80}")

            try:
                result = evaluator.evaluate(
                    trajectory=str(json_file),
                    query=args.query,
                    output_dir=args.output_dir,
                    max_claims=args.sample_claims,
                    max_actions=args.sample_actions,
                )
                results.append(result)
            except Exception as e:
                print(f"❌ Evaluation failed for {json_file}: {e}")
                continue

        # Print aggregate summary
        if results:
            print(f"\n{'='*80}")
            print("Aggregate Summary")
            print(f"{'='*80}")

            avg_scores = {}
            for key in ["grounding", "noise_induced", "intent", "propagation"]:
                scores = [r["ping_scores"].get(key, 0) for r in results]
                avg_scores[key] = sum(scores) / len(scores) if scores else 0

            overall = sum(avg_scores.values()) / len(avg_scores)

            print(f"Grounding:        {avg_scores.get('grounding', 0):.4f}")
            print(f"Noise-induced:    {avg_scores.get('noise_induced', 0):.4f}")
            print(f"Intent:          {avg_scores.get('intent', 0):.4f}")
            print(f"Propagation:     {avg_scores.get('propagation', 0):.4f}")
            print("-" * 40)
            print(f"Overall Score:   {overall:.4f}")

    else:
        # Process single file
        if not trajectory_path.exists():
            print(f"❌ Trajectory file not found: {trajectory_path}")
            sys.exit(1)

        print(f"📂 Loading trajectory from: {trajectory_path}")

        try:
            result = evaluator.evaluate(
                trajectory=str(trajectory_path),
                query=args.query,
                output_dir=args.output_dir,
                max_claims=args.sample_claims,
                max_actions=args.sample_actions,
            )

            print("\n✅ Evaluation complete!")
            print(f"📊 Overall Hallucination Score: {result['overall_score']:.4f}")

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
