#!/usr/bin/env python3
"""
Sequential evaluation script for HalluBench
Processes all JSON files in the json_with_citation directory by calling evaluate.py for each file.
"""

import os
import sys
import subprocess
import time
import json
import logging
from typing import List, Dict, Any
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sequential_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# TARGET_JSON_FILES: List[str] = [
#     "GPT-4o_002_memory_b000_t00_e01-7ab08a46_CoR.json",
#     "GPT-4o_006_memory_b000_t00_e06-71f77595_CoR.json",
#     "GPT-4o_013_memory_b001_t00_e05-c9015705_CoR.json",
#     "GPT-4o_014_memory_b001_t00_e06-71f77595_CoR.json",
#     "Llama3.3-70B-Turbo_002_memory_b000_t00_e01-7ab08a46_CoR.json",
#     "Llama3.3-70B-Turbo_006_memory_b000_t00_e05-c9015705_CoR.json",
#     "Qwen3-8B_002_memory_b000_t00_e01-7ab08a46_CoR.json",
#     "Qwen3-8B_008_memory_b000_t00_e08-f572f2ad_CoR.json",
#     "Qwen3-8B_015_memory_b001_t00_e05-c9015705_CoR.json",
# ]

TARGET_JSON_FILES: List[str] = [
    *[
        os.path.basename(p)
        for p in sorted(
            glob.glob("/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/BrowseComp/json/Art_*.json")
        )
    ]
]


def get_json_files(json_dir: str, target_files: List[str]) -> List[str]:
    """Return only the whitelisted JSON files that exist in the specified directory."""
    if not os.path.exists(json_dir):
        logger.error(f"Directory does not exist: {json_dir}")
        return []

    existing_files = []
    missing_files = []

    for file_name in target_files:
        file_path = os.path.join(json_dir, file_name)
        if os.path.exists(file_path):
            existing_files.append(file_name)
        else:
            missing_files.append(file_name)

    if missing_files:
        logger.warning(
            "The following target JSON files were not found and will be skipped: %s",
            ", ".join(missing_files)
        )

    return existing_files

def check_existing_results(json_file: str, results_dir: str) -> bool:
    """Check if results already exist for a given JSON file."""
    base_name = os.path.splitext(json_file)[0]
    result_file = f"{base_name}_combined.json"
    result_path = os.path.join(results_dir, result_file)
    return os.path.exists(result_path)

def run_evaluation(json_file: str, script_dir: str, json_dir: str, 
                  num_gpus: int = 4, gpu_ids: str = "0,1,2,3") -> Dict[str, Any]:
    """Run evaluation for a single JSON file."""
    start_time = time.time()
    
    # Construct the full path to the JSON file
    json_path = os.path.join(json_dir, json_file)
    
    # Construct the command
    cmd = [
        sys.executable, "evaluate_for_GAIA_AgentDebug.py", json_path,
        "--num_gpus", str(num_gpus),
        "--gpu_ids", gpu_ids
    ]
    
    logger.info(f"🚀 Starting evaluation for: {json_file}")
    logger.info(f"📝 Command: {' '.join(cmd)}")
    
    result = {
        "json_file": json_file,
        "start_time": start_time,
        "end_time": None,
        "duration": None,
        "success": False,
        "error": None,
        "stdout": "",
        "stderr": ""
    }
    
    try:
        # Change to the script directory
        original_cwd = os.getcwd()
        os.chdir(script_dir)
        
        # Run the evaluation
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        result.update({
            "end_time": end_time,
            "duration": duration,
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr
        })
        
        if process.returncode == 0:
            logger.info(f"✅ Successfully completed evaluation for: {json_file} (took {duration/60:.2f} minutes)")
        else:
            logger.error(f"❌ Evaluation failed for: {json_file} (return code: {process.returncode})")
            logger.error(f"Error output: {process.stderr}")
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        result.update({
            "end_time": end_time,
            "duration": duration,
            "success": False,
            "error": "Timeout (2 hours)"
        })
        logger.error(f"⏰ Evaluation timed out for: {json_file} (took {duration/60:.2f} minutes)")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        result.update({
            "end_time": end_time,
            "duration": duration,
            "success": False,
            "error": str(e)
        })
        logger.error(f"❌ Unexpected error for: {json_file}: {str(e)}")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    return result

def main():
    """Main function to run sequential evaluation."""
    start_time = time.time()
    
    # Configuration
    script_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/script_reframe"
    # json_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/GAIA_from_AgentDebug/Converted_CoR_new"
    # results_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/GAIA_AgentDebug/before_update"
    json_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/train/close-source/gemini/BrowseComp/json"
    results_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/train_gemini/browsecomp/gemini/after_memory_and_action_update"
    num_gpus = 4
    gpu_ids = "0,1,2,3"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("🚀 Starting Sequential Evaluation for HalluBench")
    logger.info("="*80)
    logger.info(f"📁 Script directory: {script_dir}")
    logger.info(f"📁 JSON directory: {json_dir}")
    logger.info(f"📁 Results directory: {results_dir}")
    logger.info(f"🎮 GPU configuration: {num_gpus} GPUs, IDs: {gpu_ids}")
    
    # Get all JSON files
    json_files = get_json_files(json_dir, TARGET_JSON_FILES)
    total_files = len(json_files)
    
    logger.info(f"📊 Found {total_files} JSON files to process")
    
    if total_files == 0:
        logger.error("❌ No JSON files found. Exiting.")
        return
    
    # Track results
    results_summary = {
        "start_time": start_time,
        "total_files": total_files,
        "processed_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "skipped_files": 0,
        "file_results": []
    }
    
    # Process each JSON file
    for i, json_file in enumerate(json_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 Processing file {i}/{total_files}: {json_file}")
        logger.info(f"{'='*60}")
        
        # Check if results already exist
        # if check_existing_results(json_file, results_dir):
        #     logger.info(f"⏭️ Skipping {json_file} - results already exist")
        #     results_summary["skipped_files"] += 1
        #     continue
        
        # Run evaluation
        result = run_evaluation(json_file, script_dir, json_dir, num_gpus, gpu_ids)
        results_summary["file_results"].append(result)
        results_summary["processed_files"] += 1
        
        if result["success"]:
            results_summary["successful_files"] += 1
        else:
            results_summary["failed_files"] += 1
        
        # Log progress
        logger.info(f"📊 Progress: {i}/{total_files} files processed")
        logger.info(f"✅ Successful: {results_summary['successful_files']}")
        logger.info(f"❌ Failed: {results_summary['failed_files']}")
        logger.info(f"⏭️ Skipped: {results_summary['skipped_files']}")
        
        # Add a small delay between evaluations to avoid overwhelming the system
        if i < total_files:
            logger.info("⏳ Waiting 5 seconds before next evaluation...")
            time.sleep(5)
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"\n{'='*80}")
    logger.info("🎯 SEQUENTIAL EVALUATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"⏱️ Total time: {total_duration/60:.2f} minutes ({total_duration/3600:.2f} hours)")
    logger.info(f"📊 Total files: {results_summary['total_files']}")
    logger.info(f"✅ Successful: {results_summary['successful_files']}")
    logger.info(f"❌ Failed: {results_summary['failed_files']}")
    logger.info(f"⏭️ Skipped: {results_summary['skipped_files']}")
    logger.info(f"📝 Processed: {results_summary['processed_files']}")
    
    # Save detailed results
    results_summary["end_time"] = end_time
    results_summary["total_duration"] = total_duration
    
    results_file = os.path.join(script_dir, "sequential_eval_results.json")
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Detailed results saved to: {results_file}")
    except Exception as e:
        logger.error(f"❌ Error saving results file: {e}")
    
    # Show failed files if any
    failed_files = [r for r in results_summary["file_results"] if not r["success"]]
    if failed_files:
        logger.info(f"\n❌ FAILED FILES ({len(failed_files)}):")
        for result in failed_files:
            logger.info(f"  - {result['json_file']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n🎉 Sequential evaluation completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)
