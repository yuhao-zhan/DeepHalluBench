import re
import os
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
import time
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
from functools import partial
from pathlib import Path
from utils import split_report_into_paragraphs
try:
    from add_urls_for_report_claims import URLClaimMapper, process_single_file
    URL_FIXING_AVAILABLE = True
except ImportError:
    URL_FIXING_AVAILABLE = False
    print("⚠️ Warning: URL claim mapping module not available. URL fixing will be skipped.")

# Configuration
API_KEYS = [
    "sk-e3571f7b6c8f40d7bb22f3e24ed19fe5",
    "sk-462b0b0587f948c5ad814c6ef8d7b06d",
    "sk-bc332844ad9a433293231722a3bf7ee9",
    "sk-668c4d9427cc4d899c771656592e42c1",
    "sk-504f9bf2c9a1499699b3916ee3e18ff6",
    "sk-9fbe0e7ba4ba406591aedbdb556658ee",
    "sk-e7a77c525b3244719c0552a48a3d6ac4",
    "sk-fc1110e1047949218c21c6d49a5fee97",
    "sk-21ed35dcef0f47f38c1e48b0f5d7ec60",
    "sk-cf053eea342c425aa3ae312114b8a8f3",
    "sk-987aa24df5114e30bcf6c0e4993ddf82",
    "sk-b9bde29c16724157b7da326f58d51d73",
    "sk-df7855244c14491aa8a1bc656819fe50",
    "sk-81efa88acaf04a7b868d76e6811006af",
    "sk-5355487f104544c681c0aa9652418381",
    "sk-b90ea014cb204cb9804ff09e7d2bdf65",
    "sk-c4e08121bafb4abfa289b74c5088c6fe",
    "sk-c0fc17d4e8be4de6b2df939779baccab",
    "sk-b0d6018c1f604ca3a5fa911785d2c144",
    "sk-36402abb3f034d2eabadba38da507946"
]

# Legacy single API key for backward compatibility
API_KEY = API_KEYS[0]  # Keep the original key as first in list

BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
WEB_MODEL    = "deepseek-v3.2"

# Parallel processing configuration
CPU_CORES = mp.cpu_count()  # Automatically detect available CPU cores
USE_PARALLEL = True  # Set to False to disable parallel processing
MAX_CONCURRENT_API_CALLS = min(len(API_KEYS), CPU_CORES)  # Use all available API keys and CPU cores

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def create_client_with_key(api_key: str) -> OpenAI:
    """
    Create a new OpenAI client with a specific API key.
    
    Args:
        api_key: The API key to use for this client
        
    Returns:
        OpenAI client instance
    """
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )


def get_api_key_for_worker(worker_id: int) -> str:
    """
    Get an API key for a specific worker based on worker ID.
    This ensures even distribution of API keys across workers.
    
    Args:
        worker_id: The worker ID (0-based)
        
    Returns:
        API key string
    """
    return API_KEYS[worker_id % len(API_KEYS)]


def create_worker_client(worker_id: int) -> OpenAI:
    """
    Create a client for a specific worker with a dedicated API key.
    
    Args:
        worker_id: The worker ID (0-based)
        
    Returns:
        OpenAI client instance with worker-specific API key
    """
    api_key = get_api_key_for_worker(worker_id)
    return create_client_with_key(api_key)


def get_optimal_workers(workload_size: int) -> int:
    """
    Calculate optimal number of workers based on workload size and available resources.
    
    Args:
        workload_size: Number of items to process
        
    Returns:
        Optimal number of workers
    """
    # Use all available resources, but don't exceed the workload size
    optimal = min(MAX_CONCURRENT_API_CALLS, workload_size, CPU_CORES)
    
    # For very small workloads, use fewer workers to reduce overhead
    if workload_size <= 2:
        optimal = min(2, optimal)
    elif workload_size <= 5:
        optimal = min(4, optimal)
    
    return max(1, optimal)  # Always use at least 1 worker


def print_api_key_distribution(workload_size: int = None):
    """
    Print the distribution of API keys across workers for monitoring.
    
    Args:
        workload_size: Optional workload size to show optimal workers
    """
    optimal_workers = get_optimal_workers(workload_size) if workload_size else MAX_CONCURRENT_API_CALLS
    
    print(f"\n🔑 API Key Distribution for Parallel Processing:")
    print(f"   Total API Keys: {len(API_KEYS)}")
    print(f"   Max Concurrent Workers: {MAX_CONCURRENT_API_CALLS}")
    print(f"   CPU Cores Available: {CPU_CORES}")
    if workload_size:
        print(f"   Workload Size: {workload_size}")
        print(f"   Optimal Workers: {optimal_workers}")
    
    print(f"\n   Worker ID → API Key (last 8 chars):")
    for worker_id in range(optimal_workers):
        api_key = get_api_key_for_worker(worker_id)
        short_key = api_key[-8:] if len(api_key) >= 8 else api_key
        print(f"   Worker {worker_id:2d} → ...{short_key}")
    
    print(f"\n   Note: Workers beyond {len(API_KEYS)} will cycle through the available keys")


def log_parallel_performance(workload_size: int, workers_used: int, start_time: float, end_time: float):
    """
    Log performance metrics for parallel processing.
    
    Args:
        workload_size: Number of items processed
        workers_used: Number of workers used
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    total_time = end_time - start_time
    throughput = workload_size / total_time if total_time > 0 else 0
    efficiency = (workload_size / workers_used) / total_time if total_time > 0 and workers_used > 0 else 0
    
    print(f"\n📊 Parallel Processing Performance:")
    print(f"   Items Processed: {workload_size}")
    print(f"   Workers Used: {workers_used}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Throughput: {throughput:.2f} items/second")
    print(f"   Worker Efficiency: {efficiency:.2f} items/worker/second")
    print(f"   CPU Utilization: {(workers_used / CPU_CORES * 100):.1f}%")

# Prompt Template
SYSTEM_PROMPT_DECOUPLE = """
You are an expert text analysis system specialized in extracting structured information from unstructured text.

## TASK
Deconstruct paragraphs to isolate and extract concrete factual observations and specific actionable plans through systematic fragmentation, classification, and atomic extraction.

## LANGUAGE REQUIREMENT
**CRITICAL: All output MUST be in English.**
- If the input text is in Chinese or any other language, you MUST translate it EXACTLY to English.
- All fragments, classifications, and atomic items must be in English.
- Preserve the exact meaning and specificity of the original text during translation.
- Do not output any Chinese characters or non-English text.

## METHODOLOGY

### Source Fidelity
- Use the provided paragraph as the single source of truth. The query is context only; never add details that are not explicitly written in the paragraph.
- Do not infer missing steps, reasons, or entities from background knowledge or the query.

### Step 1: Fragmentation (Minimal Splitting)
- Produce the smallest set of fragments that faithfully reflect the paragraph's explicit sentences or numbered steps.
- If a sentence mixes observation and plan, split only along that boundary; otherwise keep the sentence intact.
- Resolve pronouns using paragraph context while preserving modifiers, purposes, and qualifiers with their clauses.
- **Translate all fragments to English if the original is in another language.**

### Step 2: Classification
**Context reminder:** The reasoning text may contain both discoveries and plans. Classify only what is explicitly written.

- **`observation`**: Facts, discoveries, or statements quoted from sources inside the paragraph.
- **`plan`**: Actions the agent explicitly states it will take next. Ignore implied or assumed steps.

### Step 3: Atomic Extraction
- Keep each item self-contained and independently verifiable while preserving necessary modifiers and conditions.
- Prefer to keep clauses together; only split truly parallel elements (e.g., clearly enumerated lists).
- Output only information explicitly present in the fragment.
- **For plans:** Start with imperative verbs and keep integral conditions attached (e.g., "Search for issues in the target module with the specified label").
- **All atomic items must be in English.**

## FILTERING CRITERIA
**EXCLUDE:**
- Speculative language (may, might, could, likely, seems)
- Subjective opinions (effective, best, good)
- Vague process descriptions

## EXAMPLES

**Fragmentation:**
- Input: "I found some roles, but I need to search more"
- Output: Two fragments:
  - "I found some roles" (observation)
  - "Search for more roles" (plan)

**Context-Independence:**
- Input: "This approach improved performance by 15%"
- Output: "The neural network optimization approach improved performance by 15%"

**Atomic Extraction:**
- Input: "Meta's careers page lists 'Research Scientist, Computer Vision' in Menlo Park, CA, and Seattle, WA"
- Output:
  - "Meta's careers page lists 'Research Scientist, Computer Vision' in Menlo Park, CA"
  - "Meta's careers page lists 'Research Scientist, Computer Vision' in Seattle, WA"

**Semantic Integrity - DO NOT Split:**
- Input: "search for issues within the target module that have the specified label"
- ❌ Wrong Output:
  - "Search for issues within the target module"
  - "Filter issues with the specified label"
- ✅ Correct Output:
  - "Search for issues within the target module that have the specified label"

## OUTPUT FORMAT
```
Fragment 1: [context-independent fragment text]
Classification: [observation/plan]
Atomic [Claims/Actions]:
- [atomic item 1]
- [atomic item 2]
```

If no extractable content: `No extractable content - paragraph contains only vague descriptions.`
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

### SEMANTIC INTEGRITY
- **CRITICAL:** Preserve modifiers, conditions, and qualifiers with their main clauses
- Do NOT split prepositional phrases, relative clauses, or qualifiers that are semantically integral
- Only split truly parallel/conjunctive elements (e.g., "X and Y" where X and Y are independent facts)

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
- ❌ "This approach improved performance by 15%"
- ✅ "The neural network optimization approach improved performance by 15%"
- ❌ "They offer remote positions"
- ✅ "Google offers remote positions"

**Atomic Claims (Parallel Elements - OK to Split):**
- ❌ "Meta has roles in Menlo Park and Seattle"
- ✅ "Meta has a role in Menlo Park"
- ✅ "Meta has a role in Seattle"

**Semantic Integrity - DO NOT Split:**
- Input: "xxx to find information about the oldest closed issue in the target module with the specified label"
- ❌ Wrong Output:
  - "xxx to find information about the oldest closed issue in the target module"
  - "The oldest closed issue in the target module has the specified label"
- ✅ Correct Output:
  - "xxx to find information about the oldest closed issue in the target module with the specified label"

**CRITICAL:** Only split when elements are truly parallel and independent. Preserve all modifiers, conditions, and qualifiers that are semantically integral to the main clause.

## OUTPUT FORMAT
```
- [factual claim 1]
- [factual claim 2]
```

If no extractable content: "No extractable content - paragraph contains only vague descriptions."
"""

SYSTEM_PROMPT_DOUBLE_CHECK_CLAIM = """
You are a quality control system specialized in validating and refining atomic claims for maximum precision and context-independence.

## TASK
Review claims and break down any non-atomic elements while ensuring complete context independence.

## LANGUAGE REQUIREMENT
**CRITICAL: All output MUST be in English.**
- If any input claim is in Chinese or any other language, you MUST translate it EXACTLY to English.
- All output claims must be in English.
- Preserve the exact meaning and specificity of the original text during translation.
- Do not output any Chinese characters or non-English text.

## DECOMPOSITION RULES
- **CRITICAL - Semantic Integrity:** Preserve modifiers, conditions, and qualifiers with their main clauses. Do NOT split:
  - Prepositional phrases that modify the main clause (e.g., "with the specified label", "within the target module")
  - Relative clauses that specify conditions (e.g., "that have the specified label", "which are closed")
  - Purpose clauses and qualifiers that are semantically integral
- **Split ONLY truly parallel/conjunctive elements:** Break statements with "and", "or", "but" ONLY when they represent independent, parallel facts that don't affect each other's meaning
- **No compound statements:** Each claim must contain a single, indivisible fact, but preserve all necessary modifiers and qualifiers

## CONTEXT-INDEPENDENCE RULES
- Claims must be verifiable independently without preceding or following context
- **Resolve all pronouns:** Replace "the", "this", "that", "it", "they" with specific entities
- **Explicit references:** If referencing "the position" or "this role", specify the exact entity
- **Context integration:** Use broader claim list context to provide necessary specificity
- **Exclusion rule:** If context cannot be determined, exclude the claim entirely

## EXAMPLES

**Decomposition (Parallel Elements - OK to Split):**
- Input: "Role available in Menlo Park, CA and Seattle, WA"
- Output:
  - Role available in Menlo Park, CA
  - Role available in Seattle, WA

**Semantic Integrity - DO NOT Split:**
- Input: "xxx to find information about the oldest closed issue in the target module with the specified label"
- ❌ Wrong Output:
  - "xxx to find information about the oldest closed issue in the target module"
  - "The oldest closed issue in the target module satisfies the specified label condition"
- ✅ Correct Output:
  - "xxx to find information about the oldest closed issue in the target module with the specified label"

**Context Independence:**
- Input: "The position focuses on experimenting with neural network architectures."
- Context: DeepMind Research Engineer/Scientist position
- Output: "DeepMind Research Engineer/Scientist position focuses on experimenting with neural network architectures"

## OUTPUT FORMAT
Return each atomic, context-independent claim on a new line with "- " prefix.
"""


SYSTEM_PROMPT_DOUBLE_CHECK_ACTION = """
You are a quality control system specialized in validating and refining atomic actions for maximum precision and context-independence.

## TASK
Review actions and break down any non-atomic elements while ensuring complete context independence.

## LANGUAGE REQUIREMENT
**CRITICAL: All output MUST be in English.**
- If any input action is in Chinese or any other language, you MUST translate it EXACTLY to English.
- All output actions must be in English.
- Preserve the exact meaning and specificity of the original text during translation.
- Do not output any Chinese characters or non-English text.

## DECOMPOSITION RULES
- **CRITICAL - Semantic Integrity:** Preserve modifiers, conditions, and qualifiers with their main clauses. Do NOT split:
  - Prepositional phrases that modify the action (e.g., "with the specified label", "within the target module")
  - Relative clauses that specify conditions for the action (e.g., "that have the specified label", "which are closed")
  - Purpose clauses and qualifiers that are semantically integral to the action
- **Split ONLY truly parallel/conjunctive elements:** Break statements with "and", "or", "but" ONLY when they represent independent, parallel actions that don't affect each other's meaning
- **No compound statements:** Each action must contain a single, indivisible task, but preserve all necessary modifiers and qualifiers
- **Imperative form:** Start with verbs. Remove "I/my/the agent/the user". Transform "I will search" → "Search"

## CONTEXT-INDEPENDENCE RULES
- Actions must be verifiable independently without preceding or following context
- **Resolve all pronouns:** Replace "the", "this", "that", "it", "they" with specific entities according to the context
- **Context integration:** Use broader action list context to provide necessary specificity
- **Exclusion rule:** If context cannot be determined, exclude the action entirely

## NOTE: if the item is NOT related to action or plan, remove it. E.g., "Ronnie Wood has four children.", which is a fact or claim, not an action.

## EXAMPLES

**Basic Action Transformation:**
- Input: "The agent will search for authors and identify the ones that have the specified label"
- Output: 
  - "Search for authors"
  - "Identify the ones that have the specified label"

**Semantic Integrity - DO NOT Split:**
- Input: "Search for issues within the target module that have the specified label"
- ❌ Wrong Output:
  - "Search for issues within the target module"
  - "Filter issues with the specified label"
- ✅ Correct Output:
  - "Search for issues within the target module that have the specified label"

**Context Independence:**
- Input: "Confirm this information"
- Context: "Check the population data for Tokyo first" \n "Confirm this information"
- Output: "Confirm the population data for Tokyo"

## OUTPUT FORMAT
Return each atomic, context-independent action on a new line with "- " prefix.
"""


SYSTEM_PROMPT_QUERY_DECOMPOSITION = """
You are an expert query analysis system specialized in extracting atomic constraints from user queries.

## TASK
Extract concise, independent atomic constraints from user queries.

## ATOMIC CONSTRAINT DEFINITION
An atomic constraint is:
- Single, self-contained, and indivisible with clear meaning representation
- Objective conditions or criteria only
- No personal references, background info, or descriptive statements
- Each constraint must be independently verifiable and unambiguous

## EXTRACTION REQUIREMENTS
- Break down complex queries into individual constraints
- If a sentence contains multiple claims (linked by 'and', 'with', 'while'), break them into separate atomic claims
- Maintain objective, neutral language—avoid pronouns like 'I', 'me', 'my', 'for me'
- Exclude background or descriptive facts
- List each constraint on its own line prefixed with '- '
- Focus on verifiable conditions and criteria
- Ensure each constraint has clear, self-contained meaning that can be understood without additional context
"""

USER_PROMPT_QUERY_DECOMPOSITION = """
Extract all atomic constraint conditions from the following sentences.

## EXAMPLE
- Input: I am specifically interested in roles such as Research Scientist, Machine Learning Engineer, and Research Engineer (or equivalent) working on AI.
- Output:
  - "Research Scientist (or equivalent) is a role of interest"
  - "Machine Learning Engineer (or equivalent) is a role of interest"
  - "Research Engineer (or equivalent) is a role of interest"
  - "AI is a field of interest"

## INPUT
{sentences}

## OUTPUT
Provide only the list of constraints.
"""



SYSTEM_PROMPT_QUESTION_GENERATION = """
You are an expert question generation system specialized in converting atomic constraints into verifiable yes/no questions.

## TASK
Convert atomic constraints into clear, focused yes/no questions designed for verification and investigation.

## REQUIREMENTS
1. **Format:** Use "Does/Do/Is/Are" structure for yes/no questions
2. **Atomicity:** Each question must be single, focused, and independent
3. **Verifiability:** Questions must be answerable with yes or no
4. **Fidelity:** Maintain core meaning and specificity of original constraints
5. **Precision:** Keep the same level of detail and precision

## CONVERSION RULES
- Convert statements into yes/no questions using "Does/Do/Is/Are"
- Preserve all specific details and measurements
- Maintain objective, neutral language
- Ensure each question tests one specific claim

## EXAMPLE
**Input:**
- The model achieves 95% accuracy on the test dataset

**Output:**
- Does the model achieve 95% accuracy on the test dataset?

## OUTPUT FORMAT
Output each question on a new line prefixed with "- ".
"""

BANNED_USER_WORDS = (
    "the user", "the requester", "the researcher", "the analyst", "the investigator"
)



def _is_valid_observation_item(text: str) -> bool:
    t = f" {text.strip().lower()} "
    if not text.strip():
        return False
    # filter hedges
    
    if any(f" {w} " in t for w in BANNED_USER_WORDS):
        return False
    # filter planning cues (should never appear in observation claims)
    
    return True


# --- URL helpers to prepend links to the corresponding claim_list ---
_URL_REGEX = re.compile(r'https?://[^\s)\]}>,]+')

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = _URL_REGEX.findall(text)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _prepend_urls(claim_list: List[str], urls: List[str]) -> None:
    if not urls:
        return
    # Insert at front preserving original order, avoiding duplicates already present
    for u in reversed(urls):
        if u not in claim_list:
            claim_list.insert(0, u)


def _deduplicate_items(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def fix_url_placement_in_cache(cache_file_path: str, original_report: str) -> bool:
    """
    Fix URL placement in a cached JSON file using the original report text.
    
    Args:
        cache_file_path: Path to the cache JSON file
        original_report: Original report text with URLs
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not URL_FIXING_AVAILABLE:
        print("⚠️ Skipping URL placement fix - URL fixing module not available")
        return False
        
    try:
        print(f"🔗 Fixing URL placement in cache file...")
        
        # Load cache file
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if report section exists
        if 'report' not in cache_data or not isinstance(cache_data['report'], list):
            print("⚠️ No report section found in cache file, skipping URL fix")
            return False
        
        mapper = URLClaimMapper()
        
        # Process only the 'report' section
        updated_count = 0
        for i, paragraph_data in enumerate(cache_data['report']):
            if isinstance(paragraph_data, dict) and 'atomic_claims' in paragraph_data:
                original_claims_count = len(paragraph_data.get('atomic_claims', []))
                cache_data['report'][i] = mapper.process_paragraph(paragraph_data, original_report)
                new_claims_count = len(cache_data['report'][i].get('atomic_claims', []))
                if new_claims_count != original_claims_count:
                    updated_count += 1
        
        # Write back to file
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ URL placement fixed for {updated_count} paragraphs in cache file")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing URL placement in cache: {str(e)}")
        return False


def _extract_text_from_planning(planning_data) -> str:
    """Extract text content from planning data which may be string or dict."""
    if isinstance(planning_data, str):
        return planning_data
    elif isinstance(planning_data, dict):
        # Extract from title and description fields
        parts = []
        if 'title' in planning_data:
            parts.append(planning_data['title'])
        if 'description' in planning_data:
            parts.append(planning_data['description'])
        return ' '.join(parts)
    else:
        return str(planning_data) if planning_data else ""

# -------------- 1. Decouple planning and observation from reasoning text, then decompose them into atomic items -------------- #
def analyze_paragraph_fragments(paragraph, query):
    """Original function using the global client."""
    return analyze_paragraph_fragments_with_client(paragraph, query, client)


def analyze_paragraph_fragments_with_client(paragraph, query, client_instance):
    """
    Analyze paragraph fragments using a specific client instance.
    
    Args:
        paragraph: The paragraph text to analyze
        query: The query for context
        client_instance: OpenAI client instance to use
        
    Returns:
        Tuple of (observation_items, planning_items)
    """
    system_prompt = SYSTEM_PROMPT_DECOUPLE

    user_prompt = f"""Query: {query}

Paragraph to analyze: {paragraph}

Please classify this paragraph and decompose it into atomic claims or actions."""
    
    try:
        completion = client_instance.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        response = completion.choices[0].message.content.strip()
        
        
            
    except Exception as e:
        print(f"Error processing paragraph: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return [], []

    # Parse the response, return two lists: observation_items and planning_items
    # Track fragments in order to filter observations that appear after plans
    fragments = []  # List of (classification, items) tuples
    
    current_fragment = None
    current_classification = None
    current_items = []
    
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Check for fragment start
        if line.startswith("Fragment"):
            # Save previous fragment if exists
            if current_fragment and current_classification and current_items:
                fragments.append((current_classification, current_items))
            
            # Start new fragment
            current_fragment = line
            current_classification = None
            current_items = []
    
            
        # Check for classification
        elif line.startswith("Classification:"):
            current_classification = line.split(":", 1)[1].strip()
            
        # Check for atomic items (handle variations in format)
        elif line.startswith("Atomic") and ":" in line:
            # This line indicates the start of atomic items
            continue
        elif line.startswith("Atomic Claims:") or line.startswith("Atomic Actions:"):
            # Alternative format for atomic items
            continue
            
        # Check for individual atomic items (lines starting with "-")
        elif line.startswith("- "):
            item = line[2:].strip()
            current_items.append(item)
        elif line.startswith("-") and len(line.strip()) > 1:
            # Alternative format for atomic items
            item = line[1:].strip()
            current_items.append(item)
    
    # Don't forget to save the last fragment
    if current_fragment and current_classification and current_items:
        fragments.append((current_classification, current_items))
    
    # Now filter: only include observations that appear BEFORE the first plan fragment
    observation_items = []
    planning_items = []
    first_plan_index = None
    
    # Find the index of the first plan fragment
    for i, (classification, items) in enumerate(fragments):
        if classification and classification.lower() in ["plan"]:
            first_plan_index = i
            print(f"Found first plan fragment at index {i}")
            break
    
    # Collect items based on fragment order
    for i, (classification, items) in enumerate(fragments):
        if classification and classification.lower() in ["observation"]:
            # Only include observations that come before the first plan
            if first_plan_index is None or i < first_plan_index:
                observation_items.extend(items)
                print(f"Added {len(items)} observation items from fragment {i} (before plans)")
            else:
                print(f"Skipped {len(items)} observation items from fragment {i} (after plans)")
        elif classification and classification.lower() in ["plan"]:
            planning_items.extend(items)
            print(f"Added {len(items)} planning items from fragment {i}")
        else:
            if classification:
                print(f"⚠️  Unknown classification: '{classification}' - skipping items")

    # Post-filter observation items strictly to observations without hedging/opinions/meta
    original_obs_count = len(observation_items)
    observation_items = [it for it in observation_items if _is_valid_observation_item(it)]
    filtered_obs_count = len(observation_items)
    if original_obs_count != filtered_obs_count:
        print(f"⚠️  Post-filtering: {original_obs_count - filtered_obs_count} observation items were filtered out")

    # Double-check observation items for non-atomic parts
    if observation_items:
        print(f"\n🔍 Double-checking {len(observation_items)} observation items for non-atomic parts...")
        observation_items = double_check_atomic_claims_with_client(observation_items, query, client_instance)
    if planning_items:
        print(f"\n🔍 Double-checking {len(planning_items)} planning items for non-atomic parts...")
        planning_items = double_check_atomic_actions_with_client(planning_items, query, client_instance)

    observation_items = _deduplicate_items(observation_items)
    planning_items = _deduplicate_items(planning_items)

    return observation_items, planning_items


# -------------- 2. Decompose the observation into atomic claims -------------- #
def decompose_observation(observation_paragraph, query):
    """Original function using the global client."""
    return decompose_observation_with_client(observation_paragraph, query, client)


def decompose_observation_with_client(observation_paragraph, query, client_instance):
    """
    Decompose observation using a specific client instance.
    
    Args:
        observation_paragraph: The observation paragraph to decompose
        query: The query for context
        client_instance: OpenAI client instance to use
        
    Returns:
        List of atomic claims
    """
    system_prompt = SYSTEM_PROMPT_PURE_OBSERVATION_DECOMPOSITION
    user_prompt = f"""Query: {query}

Paragraph to analyze: {observation_paragraph}

Please decompose this paragraph into atomic claims."""
    
    try:
        completion = client_instance.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        response = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error processing observation: {e}")
        return f"Error: {str(e)}"

    # Parse the response, return a list of atomic claims
    items = []
    for line in response.splitlines():
        if line.startswith("- "):
            txt = line[2:].strip()
            if _is_valid_observation_item(txt):
                items.append(txt)
    
    # Double-check items for non-atomic parts
    print(f"Before double-check: {items}")
    for item in items:
        print(f"🔍 Debug: Item: {item}")
    if items:
        print(f"\n🔍 Double-checking {len(items)} observation items for non-atomic parts...")
        items = double_check_atomic_claims_with_client(items, query, client_instance)
    print(f"After double-check: {items}")   
    for item in items:
        print(f"🔍 Debug: Item: {item}")

    print()

    
    return items



# -------------- 3. Decompose the query into atomic constraints -------------- #
def decompose_query(query, cache_file):
    if os.path.exists(cache_file):
        json_data = json.load(open(cache_file, 'r', encoding='utf-8'))
        if 'query_list' in json_data and len(json_data['query_list']) > 0:
            print(f"✅ Found {len(json_data['query_list'])} queries in cache file")
            sub_queries = json_data['query_list']
            return sub_queries
    else:
        system_prompt = SYSTEM_PROMPT_QUERY_DECOMPOSITION
        user_prompt = USER_PROMPT_QUERY_DECOMPOSITION.format(sentences=query)

        try:
            completion = client.chat.completions.create(
                model=WEB_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent classification
            )
            
            response = completion.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return f"Error: {str(e)}"

        items = []
        for line in response.splitlines():
            if line.startswith("- "):
                txt = line[2:].strip()
                # If the txt contains BANNED_USER_WORDS (insensitive to case), remove it
                if not any(w in txt.lower() for w in BANNED_USER_WORDS):
                    items.append(txt)

        # Turn the items into question format
        # questions = get_question_via_LLM(items)

        # Double-check questions for non-atomic parts
        # if questions:
        #     print(f"\n🔍 Double-checking {len(questions)} query items for non-atomic parts...")
        items = double_check_atomic_claims_with_client(items, query, client)

        # Load existing cache data or create new, then add query_list
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                json_data = {}
        else:
            json_data = {}
        
        json_data['query_list'] = items
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    
        return items


# -------------- 4. Convert atomic constraints into yes/no questions -------------- #
def get_question_via_LLM(constraints: List[str]) -> List[str]:
    """
    Convert atomic constraints into question-format atomic queries using LLM.
    
    Args:
        constraints: List of atomic constraints to convert
        
    Returns:
        List of question-format queries
    """
    if not constraints:
        return []
    
    system_prompt = SYSTEM_PROMPT_QUESTION_GENERATION
    user_prompt = f"""Convert the following atomic constraints into question-format queries:

Constraints:
{chr(10).join([f"- {constraint}" for constraint in constraints])}

Please generate one question for each constraint."""
    
    try:
        completion = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent output
        )
        
        response = completion.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []
    
    # Parse the response, return a list of questions
    questions = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("- "):
            question = line[2:].strip()
            if question:
                questions.append(question)
    
    return questions


# -------------- 5. Double-check atomic claims for non-atomic parts -------------- #
def double_check_atomic_claims(claims: List[str], query: str) -> List[str]:
    """Original function using the global client."""
    return double_check_atomic_claims_with_client(claims, query, client)

def double_check_atomic_actions(actions: List[str], query: str) -> List[str]:
    """Original function using the global client."""
    return double_check_atomic_actions_with_client(actions, query, client)

def double_check_atomic_actions_with_client(actions: List[str], query: str, client_instance: OpenAI) -> List[str]:
    """
    Double-check atomic actions to ensure they don't contain non-atomic parts like "and", "or".
    
    Args:
        actions: List of atomic actions to double-check
        query: The original query for context
        client_instance: OpenAI client instance to use
    """
    if not actions:
        return []

    system_prompt = SYSTEM_PROMPT_DOUBLE_CHECK_ACTION
    user_prompt = f"""Query: {query}
    

Please review and double-check the following atomic actions for any non-atomic elements:

Actions to review:
{chr(10).join([f"- {action}" for action in actions])}

Return only valid atomic actions, breaking down any non-atomic ones into separate atomic parts.
    """

    try:
        completion = client_instance.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent output
        )
        response = completion.choices[0].message.content.strip()
        # Parse the response, return a list of double-checked actions
        filtered_actions = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("- "):
                action = line[2:].strip()
                filtered_actions.append(action)
        
        # Remove any action containing 'url_text_extractor'
        filtered_actions = [a for a in filtered_actions if 'url_text_extractor' not in a]
        
        return filtered_actions

            
    except Exception as e:
        print(f"Error double-checking atomic actions: {e}")
        # If double-check fails, return original actions
        return actions


    
    # Parse the response, return a list of double-checked actions   
def double_check_atomic_claims_with_client(claims: List[str], query: str, client_instance: OpenAI) -> List[str]:
    """
    Double-check atomic claims to ensure they don't contain non-atomic parts like "and", "or".
    
    Args:
        claims: List of atomic claims to double-check
        query: The original query for context
        client_instance: OpenAI client instance to use
        
    Returns:
        List of filtered atomic claims that pass the double-check
    """
    if not claims:
        return []
    
    system_prompt = SYSTEM_PROMPT_DOUBLE_CHECK_CLAIM
    user_prompt = f"""Query: {query}

Please review and double-check the following atomic claims for any non-atomic elements:

Claims to review:
{chr(10).join([f"- {claim}" for claim in claims])}

Return only valid atomic claims, breaking down any non-atomic ones into separate atomic parts."""
    
    try:
        completion = client_instance.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent output
        )
        
        response = completion.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error double-checking atomic claims: {e}")
        # If double-check fails, return original claims
        return claims
    
    # Parse the response, return a list of double-checked claims
    filtered_claims = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("- "):
            claim = line[2:].strip()
            if claim and _is_valid_observation_item(claim):
                filtered_claims.append(claim)
    
    # If no claims were extracted from the response, return original claims
    if not filtered_claims:
        print(f"⚠️  Double-check returned no claims, using original {len(claims)} claims")
        return claims
    
    print(f"✅ Double-check completed: {len(claims)} → {len(filtered_claims)} claims")
    return filtered_claims


# -------------- 6. Parallel Processing Functions -------------- #

def process_paragraph_fragments_parallel(args_tuple: Tuple[int, str, str, int]) -> Tuple[int, List[str], List[str]]:
    """
    Process a single paragraph's fragments in parallel.
    
    Args:
        args_tuple: (paragraph_index, paragraph_text, query, worker_id)
        
    Returns:
        Tuple of (paragraph_index, observation_items, planning_items)
    """
    paragraph_index, paragraph_text, query, worker_id = args_tuple
    try:
        # Create worker-specific client with dedicated API key
        worker_client = create_worker_client(worker_id)
        api_key_short = get_api_key_for_worker(worker_id)[-8:]
        print(f"    Worker {worker_id} using API key ...{api_key_short} for paragraph {paragraph_index}")
        obs_items, plan_items = analyze_paragraph_fragments_with_client(paragraph_text, query, worker_client)
        return paragraph_index, obs_items, plan_items
    except Exception as e:
        print(f"❌ Error processing paragraph {paragraph_index}: {e}")
        return paragraph_index, [], []


def process_observation_parallel(args_tuple: Tuple[int, str, str, int]) -> Tuple[int, List[str]]:
    """
    Process a single observation paragraph in parallel.
    
    Args:
        args_tuple: (paragraph_index, paragraph_text, query, worker_id)
        
    Returns:
        Tuple of (paragraph_index, atomic_claims)
    """
    paragraph_index, paragraph_text, query, worker_id = args_tuple
    try:
        # Create worker-specific client with dedicated API key
        worker_client = create_worker_client(worker_id)
        api_key_short = get_api_key_for_worker(worker_id)[-8:]
        print(f"    Worker {worker_id} using API key ...{api_key_short} for paragraph {paragraph_index}")
        atomic_claims = decompose_observation_with_client(paragraph_text, query, worker_client)
        return paragraph_index, atomic_claims
    except Exception as e:
        print(f"❌ Error processing observation {paragraph_index}: {e}")
        return paragraph_index, []


def process_workflow_iteration_parallel(args_tuple: Tuple[int, Dict, str, str, int]) -> Tuple[int, Dict]:
    """
    Process a single workflow iteration in parallel.
    
    Args:
        args_tuple: (iteration_index, iteration_data, query, pattern_type, worker_id)
        
    Returns:
        Tuple of (iteration_index, processed_iteration_data)
    """
    iteration_index, iteration_data, query, pattern_type, worker_id = args_tuple
    
    try:
        # Create worker-specific client with dedicated API key
        worker_client = create_worker_client(worker_id)
        api_key_short = get_api_key_for_worker(worker_id)[-8:]
        print(f"    Worker {worker_id} using API key ...{api_key_short} for iteration {iteration_index}")
        
        if pattern_type == "planning":
            # Process planning and observation for this iteration
            planning_text = iteration_data.get('planning_text', '')
            observation_text = iteration_data.get('observation_text', '')
            
            action_list = []
            claim_list = []
            
            # Process planning text
            if planning_text:
                planning_text_str = _extract_text_from_planning(planning_text)
                # Split based on '\n\ntool: ', only keep the its front to decompose further
                planning_text_parts = planning_text_str.split('\n\ntool: ')
                planning_text_str_to_decompose = planning_text_parts[0]
                planning_obs_items, planning_plan_items = analyze_paragraph_fragments_with_client(planning_text_str_to_decompose, query, worker_client)
                action_list.extend(planning_plan_items)
                # Note: planning observations need to be handled by caller for previous iteration
                # Only process tool part if it exists
                if len(planning_text_parts) > 1:
                    planning_text_str_tool = planning_text_parts[1]
                    if not planning_text_str_tool.startswith('url_text_extractor'):
                        action_list.append(planning_text_str_tool)
            
            # Process observation text
            if observation_text:
                if "\n\n\n\n" in observation_text:
                    observation_parts = observation_text.split("\n\n\n\n")
                    for part in observation_parts:
                        if not part.strip():
                            continue
                        part_urls = _extract_urls(part)
                        obs_items, plan_items = analyze_paragraph_fragments_with_client(part.strip(), query, worker_client)
                        
                        # Add URLs and claims
                        if part_urls:
                            claim_list.extend(part_urls)
                        claim_list.extend(obs_items)
                        action_list.extend(plan_items)
                else:
                    _prepend_urls(claim_list, _extract_urls(observation_text))
                    obs_items, plan_items = analyze_paragraph_fragments_with_client(observation_text, query, worker_client)
                    claim_list.extend(obs_items)
                    action_list.extend(plan_items)
            
            return iteration_index, {
                'action_list': action_list,
                'claim_list': claim_list,
                'planning_obs_items': planning_obs_items if planning_text else []
            }
            
        elif pattern_type == "reasoning":
            # Process reasoning for this iteration
            reasoning_text = iteration_data.get('reasoning_text', '')
            
            if reasoning_text:
                obs_items, plan_items = analyze_paragraph_fragments_with_client(reasoning_text, query, worker_client)
                return iteration_index, {
                    'observation_items': obs_items,
                    'planning_items': plan_items,
                    'urls': _extract_urls(reasoning_text)
                }
            else:
                return iteration_index, {
                    'observation_items': [],
                    'planning_items': [],
                    'urls': []
                }
        
        return iteration_index, {}
        
    except Exception as e:
        print(f"❌ Error processing iteration {iteration_index}: {e}")
        return iteration_index, {}


def decompose_workflow_to_cache(input_json, cache_file):
    data = json.load(open(input_json, 'r', encoding='utf-8'))
    chain = data.get('chain_of_research', {})
    
    # If chain_of_research is empty, check if keys are at top level (alternative format)
    if not chain:
        # Check if reasoning_ or search_ keys exist at top level
        if any(k.startswith('reasoning_') or k.startswith('search_') or k.startswith('plan_') for k in data.keys()):
            chain = data  # Use the entire data as chain
            print("📋 Detected alternative JSON format: keys at top level instead of chain_of_research")
    
    # Determine the number of iterations by counting search steps
    num_iters = len([k for k in chain if k.startswith('search_')])
    query = data.get('query', '')
    
    # Initialize lists to store results
    iterations = []  # Will store ordered groups of action_list_N, search_list_N, claim_list_N
    
    print("-" * 60)

    # Decompose the query into sub-queries and always rebuild cache

    if os.path.exists(cache_file):
        json_data = json.load(open(cache_file, 'r', encoding='utf-8'))
        if 'iterations' in json_data and len(json_data['iterations']) > 0:
            print(f"✅ Found {len(json_data['iterations'])} iterations in cache file")
            return
       
        # Determine the pattern of the JSON file
        has_planning = any(k.startswith('plan_') for k in chain.keys())
        has_reasoning = any(k.startswith('reasoning_') for k in chain.keys())
        
        if has_planning and not has_reasoning:
            # Pattern 1: plan-search-observation
            print("Detected plan-search-observation pattern")
            
            if USE_PARALLEL and num_iters >= 1:
                optimal_workers = get_optimal_workers(num_iters)
                print(f"🚀 Using parallel processing with {optimal_workers} workers (out of {CPU_CORES} CPU cores) for {num_iters} iterations")
                print_api_key_distribution(num_iters)
                
                # Prepare iteration data for parallel processing
                iteration_data = []
                for i in range(1, num_iters + 1):
                    iteration_data.append({
                        'iteration_index': i,
                        'planning_text': chain.get(f'plan_{i}', ''),
                        'observation_text': chain.get(f'observation_{i}', ''),
                        'search_list': chain.get(f'search_{i}', [])
                    })
                
                # Process iterations in parallel
                processed_iterations = [None] * num_iters
                
                with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                    # Submit all iterations for parallel processing
                    # Use enumerate to assign worker IDs cyclically
                    future_to_iteration = {
                        executor.submit(process_workflow_iteration_parallel, 
                                     (data['iteration_index'], data, query, "planning", i % optimal_workers)): data['iteration_index']
                        for i, data in enumerate(iteration_data)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_iteration):
                        iteration_index = future_to_iteration[future]
                        try:
                            idx, result = future.result()
                            processed_iterations[idx-1] = result
                            print(f"✅ Iteration {idx}/{num_iters} processed in parallel")
                        except Exception as e:
                            print(f"❌ Error processing iteration {iteration_index} in parallel: {e}")
                            # Fall back to sequential processing for this iteration
                            processed_iterations[iteration_index-1] = {
                                'action_list': [],
                                'claim_list': [],
                                'planning_obs_items': []
                            }
                
                # Build iterations list from parallel results
                iterations = []
                for i in range(num_iters):
                    if processed_iterations[i] is not None:
                        result = processed_iterations[i]
                        iterations.append({
                            'action_list': result.get('action_list', []),
                            'search_list': iteration_data[i]['search_list'],
                            'claim_list': result.get('claim_list', [])
                        })
                        
                        # Handle planning observations for previous iteration
                        if i > 0 and result.get('planning_obs_items'):
                            iterations[i-1]['claim_list'].extend(result['planning_obs_items'])
                    else:
                        # Fallback for failed iterations
                        iterations.append({
                            'action_list': [],
                            'search_list': iteration_data[i]['search_list'],
                            'claim_list': []
                        })
                
                print(f"🎉 Parallel workflow decomposition completed using {CPU_CORES} CPU cores")
                
            else:
                # Sequential processing (original logic)
                print(f"Using sequential processing for {num_iters} iterations")
                
                # Preallocate iteration groups using the known searches
                iterations = [{
                    'action_list': [],
                    'search_list': chain.get(f'search_{idx}', []),
                    'claim_list': []
                } for idx in range(1, num_iters + 1)]
                
                for i in range(1, num_iters + 1):
                    print(f"\nProcessing Iteration {i}/{num_iters}...")
                    
                    # Get plan and observation for this iteration
                    planning_text = chain.get(f'plan_{i}', '')
                    observation_text = chain.get(f'observation_{i}', '')
                    
                    # Prepend any URLs from planning_text to the previous iteration's claim_list (if exists)
                    if planning_text and i > 1:
                        planning_text_str = _extract_text_from_planning(planning_text)
                        _prepend_urls(iterations[i-2]['claim_list'], _extract_urls(planning_text_str))
                    
                    # Process planning text
                    if planning_text:
                        print(f"\nProcessing planning text for iteration {i}:")
                        planning_text_str = _extract_text_from_planning(planning_text)
                        planning_obs_items, planning_plan_items = analyze_paragraph_fragments(planning_text_str, query)
                        # Add planning actions to current iteration action_list
                        iterations[i-1]['action_list'].extend(planning_plan_items)
                        # If observation items found in planning, add to previous claim_list
                        if planning_obs_items and i > 1:
                            iterations[i-2]['claim_list'].extend(planning_obs_items)
                    
                    # Process observation text
                    if observation_text:
                        print(f"\nProcessing observation text for iteration {i}:")
                        # Check if observation contains "\n\n\n\n" and split accordingly
                        if "\n\n\n\n" in observation_text:
                            observation_parts = observation_text.split("\n\n\n\n")
                            for part in observation_parts:
                                if not part.strip():
                                    continue
                                # Extract URLs from this specific part
                                part_urls = _extract_urls(part)
                                obs_items, plan_items = analyze_paragraph_fragments(part.strip(), query)
                                
                                # Insert URLs right before the claims from this part
                                if part_urls:
                                    for url in part_urls:
                                        iterations[i-1]['claim_list'].append(url)
                                # Add claims to current iteration claim_list
                                iterations[i-1]['claim_list'].extend(obs_items)
                                
                                # If planning items found in observation, add to next action_list
                                if plan_items and i < num_iters:
                                    iterations[i]['action_list'].extend(plan_items)
                        else:
                            # Prepend URLs from whole observation_text to current iteration's claim_list
                            _prepend_urls(iterations[i-1]['claim_list'], _extract_urls(observation_text))
                            obs_items, plan_items = analyze_paragraph_fragments(observation_text, query)
                            # Add claims to current iteration claim_list
                            iterations[i-1]['claim_list'].extend(obs_items)
                            # If planning items found in observation, add to next action_list
                            if plan_items and i < num_iters:
                                iterations[i]['action_list'].extend(plan_items)
                    
                    print(f"Finished Iteration {i}.\n" + "-" * 60)
                
        elif has_reasoning and not has_planning:
            # Pattern 2: reason-search
            print("Detected reason-search pattern")
            
            # Find all reasoning steps
            reasoning_steps = [k for k in chain.keys() if k.startswith('reasoning_')]
            reasoning_steps.sort(key=lambda x: int(x.split('_')[1]))
            
            if USE_PARALLEL and len(reasoning_steps) >= 1:
                optimal_workers = get_optimal_workers(len(reasoning_steps))
                print(f"🚀 Using parallel processing with {optimal_workers} workers (out of {CPU_CORES} CPU cores) for {len(reasoning_steps)} reasoning steps")
                print_api_key_distribution(len(reasoning_steps))
                
                # Prepare reasoning data for parallel processing
                reasoning_data = []
                for reasoning_key in reasoning_steps:
                    reasoning_num = int(reasoning_key.split('_')[1])
                    reasoning_data.append({
                        'iteration_index': reasoning_num,
                        'reasoning_text': chain.get(reasoning_key, ''),
                        'search_list': chain.get(f'search_{reasoning_num}', []) if reasoning_num <= num_iters else []
                    })
                
                # Process reasoning steps in parallel
                processed_reasoning = [None] * len(reasoning_steps)
                
                with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                    # Submit all reasoning steps for parallel processing
                    # Use enumerate to assign worker IDs cyclically
                    future_to_reasoning = {
                        executor.submit(process_workflow_iteration_parallel, 
                                     (data['iteration_index'], data, query, "reasoning", i % optimal_workers)): data['iteration_index']
                        for i, data in enumerate(reasoning_data)
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_reasoning):
                        reasoning_index = future_to_reasoning[future]
                        try:
                            idx, result = future.result()
                            # Find the index in the reasoning_data list
                            data_idx = next(i for i, data in enumerate(reasoning_data) if data['iteration_index'] == idx)
                            processed_reasoning[data_idx] = result
                            print(f"✅ Reasoning {idx} processed in parallel")
                        except Exception as e:
                            print(f"❌ Error processing reasoning {reasoning_index} in parallel: {e}")
                            # Fall back to sequential processing for this reasoning step
                            data_idx = next(i for i, data in enumerate(reasoning_data) if data['iteration_index'] == reasoning_index)
                            processed_reasoning[data_idx] = {
                                'observation_items': [],
                                'planning_items': [],
                                'urls': []
                            }
                
                # Initialize iterations list with empty groups for each search
                iterations = [{
                    'action_list': [],
                    'search_list': chain.get(f'search_{i}', []),
                    'claim_list': []
                } for i in range(1, num_iters + 1)]
                
                # Build iterations from parallel reasoning results
                for i, (reasoning_data_item, processed_result) in enumerate(zip(reasoning_data, processed_reasoning)):
                    if processed_result is not None:
                        reasoning_num = reasoning_data_item['iteration_index']
                        
                        # Handle URLs
                        if processed_result.get('urls'):
                            target_idx = None
                            if reasoning_num > 1:
                                if reasoning_num - 2 < len(iterations):
                                    target_idx = reasoning_num - 2
                                elif len(iterations) > 0:
                                    target_idx = len(iterations) - 1
                            if target_idx is not None:
                                _prepend_urls(iterations[target_idx]['claim_list'], processed_result['urls'])
                        
                        # Handle observations and planning
                        obs_items = processed_result.get('observation_items', [])
                        plan_items = processed_result.get('planning_items', [])
                        
                        if reasoning_num == 1:
                            # For reasoning_1, only include planning_items in action_list_1
                            iterations[0]['action_list'].extend(plan_items)
                        else:
                            # For reasoning_x (x > 1): obs -> claim_list_(x-1)
                            if reasoning_num - 2 < len(iterations):
                                iterations[reasoning_num - 2]['claim_list'].extend(obs_items)
                            else:
                                # Extra reasoning after last search → add obs to final claim_list; ignore planning
                                if len(iterations) > 0:
                                    iterations[-1]['claim_list'].extend(obs_items)
                            # plan -> action_list_x (if within search bounds)
                            if reasoning_num - 1 < len(iterations):
                                iterations[reasoning_num - 1]['action_list'].extend(plan_items)
                
                print(f"🎉 Parallel reasoning processing completed using {CPU_CORES} CPU cores")
                
            else:
                # Sequential processing (original logic)
                print(f"Using sequential processing for {len(reasoning_steps)} reasoning steps")
                
                # Initialize iterations list with empty groups for each search
                iterations = [{
                    'action_list': [],
                    'search_list': chain.get(f'search_{i}', []),
                    'claim_list': []
                } for i in range(1, num_iters + 1)]
                
                for reasoning_key in reasoning_steps:
                    reasoning_num = int(reasoning_key.split('_')[1])
                    print(f"\nProcessing Reasoning {reasoning_num}...")
                    
                    reasoning_text = chain.get(reasoning_key, '')
                    
                    if reasoning_text:
                        print(f"\nProcessing reasoning text {reasoning_num}:")
                        # Determine target claim_list index for observations from this reasoning
                        target_idx = None
                        if reasoning_num > 1:
                            if reasoning_num - 2 < len(iterations):
                                target_idx = reasoning_num - 2
                            elif len(iterations) > 0:
                                target_idx = len(iterations) - 1
                        # Prepend any URLs found in reasoning_text to the target claim_list (only if obs would be placed)
                        if target_idx is not None:
                            _prepend_urls(iterations[target_idx]['claim_list'], _extract_urls(reasoning_text))

                        obs_items, plan_items = analyze_paragraph_fragments(reasoning_text, query)
                        
                        if reasoning_num == 1:
                            # For reasoning_1, only include planning_items in action_list_1
                            iterations[0]['action_list'].extend(plan_items)
                        else:
                            # For reasoning_x (x > 1): obs -> claim_list_(x-1)
                            if reasoning_num - 2 < len(iterations):
                                iterations[reasoning_num - 2]['claim_list'].extend(obs_items)
                            else:
                                # Extra reasoning after last search → add obs to final claim_list; ignore planning
                                if len(iterations) > 0:
                                    iterations[-1]['claim_list'].extend(obs_items)
                            # plan -> action_list_x (if within search bounds)
                            if reasoning_num - 1 < len(iterations):
                                iterations[reasoning_num - 1]['action_list'].extend(plan_items)
                    
                    print(f"Finished Reasoning {reasoning_num}.\n" + "-" * 60)
        else:
            print("Warning: Could not determine JSON pattern. Both planning and reasoning keys found or neither found.")
            print("Falling back to simple format: converting search_x to search_list_x")
            
            # Fallback: Convert search_x to search_list_x format
            iterations = []
            search_keys = [k for k in chain.keys() if k.startswith('search_')]
            search_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
            
            for search_key in search_keys:
                search_num = search_key.split('_')[1]
                iterations.append({
                    'action_list': [],
                    'search_list': chain.get(search_key, []),
                    'claim_list': []
                })

        # Save the results to cache file
        # Load existing cache data or create new
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                cache_data = {}
        else:
            cache_data = {}
        
        # Update iterations field
        # Check if all iterations have empty action_list and claim_list (fallback mode)
        is_simple_format = iterations and all(
            not group.get('action_list') and not group.get('claim_list')
            for group in iterations
        )
        
        if is_simple_format:
            # Simple fallback format: only save search_list_x (each dict contains only search_list_x)
            cache_data['iterations'] = [
                {f'search_list_{i+1}': group['search_list']}
                for i, group in enumerate(iterations)
            ]
        else:
            # Full format: save all three lists
            cache_data['iterations'] = [
                {
                    f'action_list_{i+1}': group['action_list'],
                    f'search_list_{i+1}': group['search_list'],
                    f'claim_list_{i+1}': group['claim_list']
                }
                for i, group in enumerate(iterations)
            ]
        
        # Ensure output directory exists if any
        _dir = os.path.dirname(cache_file)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved to {cache_file} (overwritten if existed)")


def decompose_workflow_to_cache_auto(input_json, cache_file):
    """
    Automatically choose between parallel and sequential workflow decomposition.
    
    Args:
        input_json: Path to the input JSON file
        
    Returns:
        Nonedecompose_workflow_to_cache
    """
    if USE_PARALLEL:
        print(f"🚀 Auto-selecting parallel processing with {CPU_CORES} CPU cores")
        return decompose_workflow_to_cache(input_json, cache_file)
    else:
        print(f"📝 Auto-selecting sequential processing")
        return decompose_workflow_to_cache(input_json, cache_file)


def decompose_report_to_cache_auto(report: str, query: str, cache_file: str) -> bool:
    """
    Automatically choose between parallel and sequential report decomposition.
    
    Args:
        report: The full report text to decompose
        query: The original query for context
        cache_file: Path to the cache JSON file
        
    Returns:
        bool: True if decomposition was performed, False if skipped
    """
    if USE_PARALLEL:
        print(f"🚀 Auto-selecting parallel processing with {CPU_CORES} CPU cores")
        return decompose_report_to_cache_parallel(report, query, cache_file)
    else:
        print(f"📝 Auto-selecting sequential processing")
        return decompose_report_to_cache(report, query, cache_file)


def decompose_report_to_cache(report: str, query: str, cache_file: str) -> bool:
    """
    Decompose report paragraphs into atomic claims and save to cache file.
    
    Args:
        report: The full report text to decompose
        query: The original query for context
        cache_file: Path to the cache JSON file
        
    Returns:
        bool: True if decomposition was performed, False if skipped (already exists)
    """
    print(f"\n🔍 Checking if report decomposition already exists in cache...")
    
    # Check if cache file exists and has report attribute
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if report attribute already exists
            if 'report' in cache_data:
                print(f"✅ Report decomposition already exists in cache, skipping...")
                print(f"  Found {len(cache_data['report'])} paragraphs with atomic claims")
                return False
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ Error reading cache file, will proceed with decomposition...")
    
    print(f"📝 Report decomposition not found, decomposing paragraphs...")
    
    # Split report into paragraphs (using both \n\n and Markdown headers)
    paragraphs = split_report_into_paragraphs(report)
    print(f"  Found {len(paragraphs)} paragraphs to decompose")
    
    # Decompose each paragraph into atomic claims
    report_data = []
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"  Decomposing paragraph {i}/{len(paragraphs)}...")
        # print(f"    Paragraph: {paragraph}")
        try:
            # Decompose the paragraph into atomic claims
            atomic_claims = decompose_observation(paragraph, query)
            
            if not atomic_claims:
                print(f"    No extractable claims found in paragraph {i}")
                report_data.append({
                    'paragraph_index': i,
                    'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                    'atomic_claims': [],
                    'error': 'No extractable claims found'
                })
            else:
                print(f"    Extracted {len(atomic_claims)} claims from paragraph {i}")
                for claim in atomic_claims:
                    print(f"      - {claim}")
                report_data.append({
                    'paragraph_index': i,
                    'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                    'atomic_claims': atomic_claims
                })
                
        except Exception as e:
            print(f"    ❌ Error decomposing paragraph {i}: {str(e)}")
            report_data.append({
                'paragraph_index': i,
                'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                'atomic_claims': [],
                'error': str(e)
            })
    
    # Load existing cache data or create new
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cache_data = {}
    else:
        cache_data = {}
    
    # Add report data to cache
    cache_data['report'] = report_data
    
    # Save updated cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Report decomposition saved to cache file: {cache_file}")
        print(f"  Saved {len(report_data)} paragraphs with atomic claims")
        
        # Fix URL placement after decomposition is complete
        print(f"🔗 Post-processing: Fixing URL placement in claims...")
        fix_url_placement_in_cache(cache_file, report)
        
        return True
    except Exception as e:
        print(f"❌ Error saving report decomposition to cache: {str(e)}")
        return False


def decompose_report_to_cache_parallel(report: str, query: str, cache_file: str) -> bool:
    """
    Decompose report paragraphs into atomic claims in parallel and save to cache file.
    
    Args:
        report: The full report text to decompose
        query: The original query for context
        cache_file: Path to the cache JSON file
        
    Returns:
        bool: True if decomposition was performed, False if skipped (already exists)
    """
    print(f"\n🔍 Checking if report decomposition already exists in cache...")
    
    # Check if cache file exists and has report attribute
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if report attribute already exists
            if 'report' in cache_data:
                print(f"✅ Report decomposition already exists in cache, skipping...")
                print(f"  Found {len(cache_data['report'])} paragraphs with atomic claims")
                return False
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ Error reading cache file, will proceed with decomposition...")
    
    print(f"📝 Report decomposition not found, decomposing paragraphs...")
    
    # Split report into paragraphs (using both \n\n and Markdown headers)
    paragraphs = split_report_into_paragraphs(report)
    print(f"  Found {len(paragraphs)} paragraphs to decompose")
    
    if not USE_PARALLEL or len(paragraphs) <= 2:
        print(f"  Using sequential processing for {len(paragraphs)} paragraphs")
        return decompose_report_to_cache(report, query, cache_file)
    
    # Calculate optimal number of workers for this workload
    optimal_workers = get_optimal_workers(len(paragraphs))
    print(f"  Using parallel processing with {optimal_workers} workers (out of {CPU_CORES} CPU cores)")
    print_api_key_distribution(len(paragraphs))
    
    # Prepare data for parallel processing
    paragraph_data = [(i, paragraph, query, i % optimal_workers) for i, paragraph in enumerate(paragraphs, 1)]
    
    # Process paragraphs in parallel
    report_data = [None] * len(paragraphs)  # Pre-allocate result array
    
    # Start timing for performance monitoring
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        # Submit all paragraphs for parallel processing
        future_to_paragraph = {
            executor.submit(process_observation_parallel, data): data[0] 
            for data in paragraph_data
        }
        
        # Collect results as they complete with progress tracking
        completed_count = 0
        for future in as_completed(future_to_paragraph):
            paragraph_index = future_to_paragraph[future]
            completed_count += 1
            try:
                idx, atomic_claims = future.result()
                # Convert to 0-based index for array
                array_idx = idx - 1
                
                if not atomic_claims:
                    print(f"    [{completed_count}/{len(paragraphs)}] No extractable claims found in paragraph {idx}")
                    report_data[array_idx] = {
                        'paragraph_index': idx,
                        'paragraph_text': paragraphs[array_idx][:200] + "..." if len(paragraphs[array_idx]) > 200 else paragraphs[array_idx],
                        'atomic_claims': [],
                        'error': 'No extractable claims found'
                    }
                else:
                    print(f"    [{completed_count}/{len(paragraphs)}] Extracted {len(atomic_claims)} claims from paragraph {idx}")
                    for claim in atomic_claims:
                        print(f"      - {claim}")
                    report_data[array_idx] = {
                        'paragraph_index': idx,
                        'paragraph_text': paragraphs[array_idx][:200] + "..." if len(paragraphs[array_idx]) > 200 else paragraphs[array_idx],
                        'atomic_claims': atomic_claims
                    }
                    
            except Exception as e:
                print(f"    ❌ Error processing paragraph {paragraph_index}: {str(e)}")
                array_idx = paragraph_index - 1
                report_data[array_idx] = {
                    'paragraph_index': paragraph_index,
                    'paragraph_text': paragraphs[array_idx][:200] + "..." if len(paragraphs[array_idx]) > 200 else paragraphs[array_idx],
                    'atomic_claims': [],
                    'error': str(e)
                }
    
    # End timing and log performance
    end_time = time.time()
    log_parallel_performance(len(paragraphs), optimal_workers, start_time, end_time)
    
    # Load existing cache data or create new
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cache_data = {}
    else:
        cache_data = {}
    
    # Add report data to cache
    cache_data['report'] = report_data
    
    # Save updated cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Parallel report decomposition completed and saved to cache file: {cache_file}")
        print(f"  Saved {len(report_data)} paragraphs with atomic claims using {CPU_CORES} CPU cores")
        
        # Fix URL placement after decomposition is complete
        print(f"🔗 Post-processing: Fixing URL placement in claims...")
        fix_url_placement_in_cache(cache_file, report)
        
        return True
    except Exception as e:
        print(f"❌ Error saving report decomposition to cache: {str(e)}")
        return False


def main():
    # Trial for query decomposition
    query = "Real gas models beyond the Van der Waals model, detailed explanation of their historical background, derivation process, theoretical rationality, and limitations."
    items = decompose_query(query)
    print(items)

if __name__ == "__main__":
    main()