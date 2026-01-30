#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import time
import random
import multiprocessing as mp
from typing import List, Dict, Tuple, Any
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_json_from_content(content: str) -> str:
    """
    Extract JSON content from various markdown code block formats.
    Handles both complete and incomplete JSON responses.
    
    Args:
        content: Raw content from LLM response
        
    Returns:
        Extracted JSON string (potentially fixed if incomplete)
    """
    # Remove leading/trailing whitespace
    content = content.strip()
    
    # Handle various markdown code block formats
    if "```json" in content:
        # Find JSON code block
        start_idx = content.find("```json") + 7
        end_idx = content.find("```", start_idx)
        if end_idx != -1:
            return content[start_idx:end_idx].strip()
    
    # Handle generic code blocks that might contain JSON
    if content.startswith("```") and content.endswith("```"):
        # Remove the outer ``` markers
        inner_content = content[3:-3].strip()
        # Check if it looks like JSON (starts with { or [)
        if inner_content.startswith(("{", "[")):
            return inner_content
    
    # Handle cases where there are multiple ``` in the content
    if content.count("```") >= 2:
        # Find the first complete code block
        first_triple = content.find("```")
        if first_triple != -1:
            # Check if it's a json block
            start_idx = first_triple + 3
            if content[start_idx:start_idx+4] == "json":
                start_idx += 4
            end_idx = content.find("```", start_idx)
            if end_idx != -1:
                extracted = content[start_idx:end_idx].strip()
                # Check if it looks like JSON
                if extracted.startswith(("{", "[")):
                    return extracted
    
    # If no code blocks found, try to find JSON object in the content
    # Look for the first JSON object (complete or incomplete)
    json_start = content.find('{')
    if json_start != -1:
        # Find the matching closing brace by counting braces
        brace_count = 0
        json_end = -1
        for i, char in enumerate(content[json_start:], json_start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end != -1:
            # Complete JSON object found
            return content[json_start:json_end]
        else:
            # Incomplete JSON object - try to fix it
            incomplete_json = content[json_start:].strip()
            return _fix_incomplete_json(incomplete_json)
    
    # If no JSON object found, try to find JSON array
    json_start = content.find('[')
    if json_start != -1:
        # Find the matching closing bracket by counting brackets
        bracket_count = 0
        json_end = -1
        for i, char in enumerate(content[json_start:], json_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    json_end = i + 1
                    break
        
        if json_end != -1:
            return content[json_start:json_end]
        else:
            # Incomplete JSON array - try to fix it
            incomplete_json = content[json_start:].strip()
            return _fix_incomplete_json(incomplete_json)
    
    # If no JSON found, return the original content
    return content

def _fix_incomplete_json(incomplete_json: str) -> str:
    """
    Attempt to fix incomplete JSON by adding missing closing braces/brackets.
    
    Args:
        incomplete_json: Incomplete JSON string
        
    Returns:
        Potentially fixed JSON string
    """
    # Count opening and closing braces/brackets
    open_braces = incomplete_json.count('{')
    close_braces = incomplete_json.count('}')
    open_brackets = incomplete_json.count('[')
    close_brackets = incomplete_json.count(']')
    
    # Add missing closing braces
    missing_braces = open_braces - close_braces
    if missing_braces > 0:
        incomplete_json += '}' * missing_braces
    
    # Add missing closing brackets
    missing_brackets = open_brackets - close_brackets
    if missing_brackets > 0:
        incomplete_json += ']' * missing_brackets
    
    # Try to parse the fixed JSON to validate it
    try:
        json.loads(incomplete_json)
        return incomplete_json
    except json.JSONDecodeError:
        # If still invalid, try a more aggressive fix
        # Remove any trailing commas before adding closing braces
        fixed_json = incomplete_json.rstrip().rstrip(',')
        
        # Add missing closing braces
        missing_braces = open_braces - close_braces
        if missing_braces > 0:
            fixed_json += '}' * missing_braces
        
        # Add missing closing brackets
        missing_brackets = open_brackets - close_brackets
        if missing_brackets > 0:
            fixed_json += ']' * missing_brackets
        
        return fixed_json

def _extract_fields_from_text(content: str) -> Dict[str, Any]:
    """
    Extract JSON fields from text using regex patterns when JSON parsing fails.
    
    Args:
        content: Raw content from LLM response
        
    Returns:
        Dictionary with extracted fields or None if extraction fails
    """
    try:
        # Extract judgment
        judgment_match = re.search(r'"judgment"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
        if not judgment_match:
            judgment_match = re.search(r'judgment["\s]*:?\s*["\s]*([a-zA-Z]+)', content, re.IGNORECASE)
        
        if judgment_match:
            judgment = judgment_match.group(1).lower().strip()
            if judgment not in ["entailed", "contradicted", "neutral"]:
                # Try to map common variations
                if "entail" in judgment:
                    judgment = "entailed"
                elif "contradict" in judgment:
                    judgment = "contradicted"
                else:
                    judgment = "neutral"
        else:
            return None
        
        # Extract confidence
        confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
        if not confidence_match:
            confidence_match = re.search(r'confidence["\s]*:?\s*([0-9.]+)', content, re.IGNORECASE)
        
        confidence = 0.5  # Default confidence
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                pass
        
        # Extract explanation
        explanation_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', content, re.IGNORECASE)
        if not explanation_match:
            explanation_match = re.search(r'explanation["\s]*:?\s*["\s]*([^"]*)', content, re.IGNORECASE)
        
        explanation = ""
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            # Clean up escaped characters
            explanation = explanation.replace('\\"', '"').replace('\\/', '/').replace('\\\\', '\\')
            explanation = explanation.replace('\\n', ' ').replace('\\t', ' ')
            explanation = re.sub(r'\\(.)', r'\1', explanation)
        
        # Extract query field
        query_match = re.search(r'"query"\s*:\s*([-0-9]+)', content, re.IGNORECASE)
        if not query_match:
            query_match = re.search(r'query["\s]*:?\s*([-0-9]+)', content, re.IGNORECASE)
        
        query = -1  # Default value
        if query_match:
            try:
                query = int(query_match.group(1))
            except ValueError:
                pass
        
        # Extract observation field
        observation_match = re.search(r'"observation"\s*:\s*([-0-9]+)', content, re.IGNORECASE)
        if not observation_match:
            observation_match = re.search(r'observation["\s]*:?\s*([-0-9]+)', content, re.IGNORECASE)
        
        observation = -1  # Default value
        if observation_match:
            try:
                observation = int(observation_match.group(1))
            except ValueError:
                pass
        
        return {
            "judgment": judgment,
            "confidence": confidence,
            "explanation": explanation,
            "query": query,
            "observation": observation
        }
        
    except Exception as e:
        logger.debug(f"Error extracting fields from text: {e}")
        return None

# API Configuration
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

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
WEB_MODEL = "deepseek-v3.2"

# API_KEYS = [
#     "sk-bpcBx1WUjkvqvlGMX1VLBm06ecRfNi6aGEpjYoO67IHprokb",
#     "sk-xineuzXwhIDpGgUwzUcj8ALoMSJGbZgq1m4GDuewnXCFZ6bV",
#     "sk-SdH5r1cYTSdv5OhRqT7KtiCd13YIBNgyD0NOVJdb1PvUR5cW",
#     "sk-3zedPgtFzMt9YfUhyuUOfWX6d0scqTZReCouJgR75U6hMoH7",
#     "sk-rz6wlah3nXbqYLAGIMOywxpnm7DgUv7SaI1LlKSJzxaTpdvO",
#     "sk-pJ5j0gQGQxn8LOFj5atyweRZaFAUXKOeAowYeXTR2zsyfexs",
#     "sk-mv2icJFvVirLiQNVEVzS1VFDgg2WkJ4oNuYQ6hGky6mHPVhi",
#     "sk-9ffMFgWBjx69adLIo8o01G3o4zlbDVo3oKKUAm56l47n94nH",
#     "sk-qsBekFYoz0shmJcu3JXuLBPJDI458eQH9wYpVZUxLQ5yiVxI",
#     "sk-MVin7CI0QmKbQA4zgxPaOl4SZPpjsaVZesQTVOyUjluzvEqQ",
#     "sk-YMbK5f9BfxXPNvgWUpGMGcLKg5ucLGDKGmIuusK1g1fgOrqQ",
#     "sk-D3jLVTPACAuXjVWTj3QWm2rcNFosbZEbl0AH0tAYpZUr0jaS",
#     "sk-NAChG9qBi6JvBI4SMYy48S0nm94wwlgsjAUkB6MoOKGt9QHf",
#     "sk-h3QSBfDP4yt6Qk1mQg4X01QOJ0V75ptOMEyV4LOm81KBwDl5",
#     "sk-JMsHzw6RE0CVnODu9j9y3X9TayVqZq13dAT2NFTpALrYQW5h",
#     "sk-3zXBNXwUVGMyFI4lFovmcSxTEoDSgbNesGMc3WDPvRW4leK4",
#     "sk-gSDTNscbJ4T1m9NComJYE8aQHJ6Unyz6IvEiEWy0ifHDlXYJ",
#     "sk-A3fd0hBsbmPv3n3aaJNtFBXEq4duBSbDyiSzmQE1mLX3GVbo",
#     "sk-6lMDVQqG42tNpFBO7t4rUGx2OxF8HrkiMwmmjOOAyl8if7w8",
#     "sk-lJ8UkcpFHqDAGMSAXwbmyfEWy7kUOH1KV0ONGbB9R17SuMwf"
# ]
# BASE_URL = "https://api.nuwaapi.com/v1"
# WEB_MODEL = "gpt-4o"

class APILoadBalancer:
    """Load balancer for distributing API calls across multiple API keys."""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.num_apis = len(api_keys)
        self.api_clients = []
        
        # Initialize clients for each API key
        for api_key in api_keys:
            client = OpenAI(
                api_key=api_key,
                base_url=BASE_URL
            )
            self.api_clients.append(client)
        
        logger.info(f"🚀 Initialized {self.num_apis} API clients for load balancing")
    
    def distribute_work_evenly(self, tasks: List[Any]) -> List[List[Any]]:
        """Distribute tasks evenly across all API keys."""
        if not tasks:
            return []
        
        # Calculate distribution
        total_tasks = len(tasks)
        base_size = total_tasks // self.num_apis
        remainder = total_tasks % self.num_apis
        
        distributed = []
        start_idx = 0
        
        for api_id in range(self.num_apis):
            # First 'remainder' APIs get one extra task
            current_size = base_size + (1 if api_id < remainder else 0)
            end_idx = start_idx + current_size
            
            distributed.append(tasks[start_idx:end_idx])
            start_idx = end_idx
        
        # Log distribution
        for api_id, api_tasks in enumerate(distributed):
            logger.info(f"API {api_id}: {len(api_tasks)} tasks")
        
        return distributed

def llm_judge_claim_with_retry(api_client: OpenAI, claim: str, chunk_text: str, query: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Use LLM to judge whether a claim is entailed, contradicted, or neutral with respect to a chunk.
    Includes retry logic for rate limiting.
    """
    system_prompt = """You are an expert claim verification system. Given a claim, query, and document content, classify the relationship as ENTAILED, CONTRADICTED, or NEUTRAL.

## Classification Criteria

**ENTAILED**: The claim is explicitly stated, reasonably inferred, or represents a valid summary/abstraction of the document content.

**CONTRADICTED**: The document contains information that directly refutes the claim. Requires: (1) all necessary claim elements present in document, (2) sufficient information for definitive judgment, (3) clear contradictory evidence.

**NEUTRAL**: Insufficient information to support or contradict the claim. Use when any necessary verification elements are missing, information is incomplete, or the claim cannot be properly evaluated.

## Examples

**Example 1 - ENTAILED**
Document: "The study found a 25% increase in productivity after implementing the new system."
Claim: "Productivity improved following system implementation."
→ ENTAILED: The claim summarizes the documented finding.

**Example 2 - CONTRADICTED** 
Document: "The experiment was conducted with 100 participants aged 18-25."
Claim: "The study included elderly participants over 65."
→ CONTRADICTED: Document clearly states age range 18-25, directly contradicting the claim about elderly participants.

**Example 3 - NEUTRAL**
Document: "The company announced a new product launch."
Claim: "The product launch increased quarterly revenue by 15%."
→ NEUTRAL: Document mentions launch but lacks revenue information needed to verify the specific claim.

## Output Format
```json
{
  "judgment": "entailed|contradicted|neutral",
  "evidence": "One-sentence explanation (required for entailed/contradicted only)",
  "confidence": 0.0-1.0 (the confidence of your judgment)
}
```
"""
    user_prompt = f"""
    Query: {query}

Claim to verify: {claim}

Document chunk content:
{chunk_text}

Please analyze whether the claim is entailed, contradicted, or neutral with respect to the document chunk.

REMEMBER: Respond with ONLY a JSON code block using the exact format specified. Do not escape characters unnecessarily."""

    for attempt in range(max_retries):
        try:
            response = api_client.chat.completions.create(
                model=WEB_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content
            
            # Extract JSON from markdown code blocks if present
            cleaned_content = extract_json_from_content(content)
            
            # Parse the JSON content
            try:
                result = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                # Enhanced regex extraction as fallback
                judgment_match = re.search(r'"judgment"\s*:\s*"([^"]+)"', cleaned_content)
                confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned_content)
                evidence_match = re.search(r'"evidence"\s*:\s*"([^"]*)"', cleaned_content)
                
                if judgment_match and confidence_match:
                    judgment = judgment_match.group(1).lower()
                    confidence = float(confidence_match.group(1))
                    evidence = evidence_match.group(1) if evidence_match else ""
                    
                    # Clean evidence field - handle escaped characters
                    if evidence:
                        evidence = evidence.replace('\\"', '"').replace('\\/', '/').replace('\\\\', '\\')
                        evidence = evidence.replace('\\n', ' ').replace('\\t', ' ')
                        # Remove any remaining escape sequences
                        evidence = re.sub(r'\\(.)', r'\1', evidence)
                    
                    # Validate judgment value
                    if judgment not in ["entailed", "contradicted", "neutral"]:
                        judgment = "neutral"
                    
                    result = {
                        "judgment": judgment,
                        "evidence": evidence,
                        "confidence": confidence
                    }
                else:
                    result = {"judgment": "neutral", "evidence": "", "confidence": 0.0}
            
            # Validate and ensure result structure
            if not isinstance(result, dict):
                result = {"judgment": "neutral", "evidence": "", "confidence": 0.0}
            
            # Ensure required fields exist with valid values
            if "judgment" not in result or result["judgment"] not in ["entailed", "contradicted", "neutral"]:
                result["judgment"] = "neutral"
            
            if "confidence" not in result or not isinstance(result["confidence"], (int, float)) or result["confidence"] < 0 or result["confidence"] > 1:
                result["confidence"] = 0.0
            
            if "evidence" not in result:
                result["evidence"] = ""
            
            # Clean and validate evidence field
            if isinstance(result["evidence"], str):
                # Remove any HTML-like tags or escape sequences that might have been introduced
                evidence = result["evidence"]
                # Remove common HTML entities and tags
                evidence = re.sub(r'<[^>]+>', '', evidence)  # Remove HTML tags
                evidence = re.sub(r'&[a-zA-Z]+;', '', evidence)  # Remove HTML entities
                # Clean up any double-escaped characters
                evidence = evidence.replace('\\"', '"').replace('\\/', '/').replace('\\\\', '\\')
                result["evidence"] = evidence.strip()
            
            # Add token usage information
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
            result["tokens_used"] = tokens_used
            
            return result
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                # Rate limit hit, wait and retry
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Error in LLM judgment (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return {"judgment": "neutral", "evidence": "", "confidence": 0.0, "tokens_used": 0}
                time.sleep(1)
    
    return {"judgment": "neutral", "evidence": "", "confidence": 0.0, "tokens_used": 0}

def llm_judge_claim_worker_parallel(args: Tuple[int, List[Tuple[str, str, str, dict]]]) -> List[Dict[str, Any]]:
    """
    Worker function for parallel LLM claim judging using a specific API key.
    
    Args:
        args: Tuple of (api_id, list of (claim, chunk_text, query, chunk_meta))
        
    Returns:
        List of dictionaries containing judgment results and metadata
    """
    api_id, tasks = args
    
    # Get the API client for this worker
    api_client = OpenAI(
        api_key=API_KEYS[api_id],
        base_url=BASE_URL
    )
    
    results = []
    
    for task in tasks:
        try:
            claim, chunk_text, query, chunk_meta = task
            
            # Get LLM result with retry logic
            llm_result = llm_judge_claim_with_retry(api_client, claim, chunk_text, query)
            
            # Return result with metadata
            result = {
                'chunk_id': chunk_meta['chunk_id'],
                'source_url': chunk_meta['source_url'],
                'chunk_index': chunk_meta['chunk_index'],
                'judgment': llm_result['judgment'],
                'evidence': llm_result['evidence'],
                'confidence': llm_result['confidence'],
                'chunk_text': chunk_text,
                'tokens_used': llm_result.get('tokens_used', 0),
                'api_id': api_id,
                'chunk_meta': chunk_meta  # 添加完整的chunk_meta
            }
            results.append(result)
            
        except Exception as e:
            # Return error result if something goes wrong
            logger.error(f"⚠️ Error in API {api_id} worker: {e}")
            result = {
                'chunk_id': 'error',
                'source_url': '',
                'chunk_index': 0,
                'judgment': 'neutral',
                'evidence': '',
                'confidence': 0.0,
                'chunk_text': 'Error processing chunk',
                'tokens_used': 0,
                'api_id': api_id,
                'error': str(e)
            }
            results.append(result)
    
    return results

def llm_judge_claim_worker_single_api(args: Tuple[int, List[Tuple[str, str, str, dict]]]) -> List[Dict[str, Any]]:
    """
    Worker function for parallel LLM claim judging using single API key.
    
    Args:
        args: Tuple of (core_id, list of (claim, chunk_text, query, chunk_meta))
        
    Returns:
        List of dictionaries containing judgment results and metadata
    """
    core_id, tasks = args
    
    # Get the API client using single API key
    api_client = OpenAI(
        api_key=API_KEYS[0],
        base_url=BASE_URL
    )
    
    results = []
    
    for task in tasks:
        try:
            claim, chunk_text, query, chunk_meta = task
            
            # Get LLM result with retry logic
            llm_result = llm_judge_claim_with_retry(api_client, claim, chunk_text, query)
            
            # Return result with metadata
            result = {
                'chunk_id': chunk_meta['chunk_id'],
                'source_url': chunk_meta['source_url'],
                'chunk_index': chunk_meta['chunk_index'],
                'judgment': llm_result['judgment'],
                'evidence': llm_result['evidence'],
                'confidence': llm_result['confidence'],
                'chunk_text': chunk_text,
                'tokens_used': llm_result.get('tokens_used', 0),
                'core_id': core_id,
                'chunk_meta': chunk_meta  # 添加完整的chunk_meta
            }
            results.append(result)
            
        except Exception as e:
            # Return error result if something goes wrong
            logger.error(f"⚠️ Error in Core {core_id} worker: {e}")
            result = {
                'chunk_id': 'error',
                'source_url': '',
                'chunk_index': 0,
                'judgment': 'neutral',
                'evidence': '',
                'confidence': 0.0,
                'chunk_text': 'Error processing chunk',
                'tokens_used': 0,
                'core_id': core_id,
                'error': str(e)
            }
            results.append(result)
    
    return results

def process_llm_judgments_parallel(all_chunk_args: List[Tuple[str, str, str, dict]], num_cores: int = 255) -> List[Dict[str, Any]]:
    """
    Process all LLM judgments in parallel using single API key with multiple CPU cores.
    
    Args:
        all_chunk_args: List of (claim, chunk_text, query, chunk_meta) tuples
        num_cores: Number of CPU cores to use (default: 255)
        
    Returns:
        List of judgment results
    """
    # Use single API key
    api_key = API_KEYS[0]
    
    logger.info(f"🚀 Starting parallel LLM processing with {num_cores} CPU cores")
    logger.info(f"📊 Total tasks: {len(all_chunk_args)}")
    logger.info(f"🔑 Using single API key: {api_key[:20]}...")
    
    # Distribute work evenly across CPU cores
    total_tasks = len(all_chunk_args)
    base_size = total_tasks // num_cores
    remainder = total_tasks % num_cores
    
    distributed_tasks = []
    start_idx = 0
    
    for core_id in range(num_cores):
        # First 'remainder' cores get one extra task
        current_size = base_size + (1 if core_id < remainder else 0)
        end_idx = start_idx + current_size
        
        if current_size > 0:  # Only add cores with tasks
            distributed_tasks.append((core_id, all_chunk_args[start_idx:end_idx]))
            start_idx = end_idx
    
    logger.info(f"🔄 Processing with {len(distributed_tasks)} CPU cores")
    
    # Process in parallel using multiprocessing
    all_results = []
    
    try:
        with mp.Pool(processes=len(distributed_tasks)) as pool:
            parallel_results = pool.map(llm_judge_claim_worker_single_api, distributed_tasks)
            
            # Flatten results
            for core_results in parallel_results:
                all_results.extend(core_results)
    
    except Exception as e:
        logger.error(f"❌ Error in parallel processing: {e}")
        # Fallback to sequential processing
        logger.info("🔄 Falling back to sequential processing...")
        for core_id, core_tasks in distributed_tasks:
            if core_tasks:
                core_results = llm_judge_claim_worker_single_api((core_id, core_tasks))
                all_results.extend(core_results)
    
    logger.info(f"✅ Parallel LLM processing completed: {len(all_results)} results")
    
    # Log core usage statistics
    core_usage = {}
    for result in all_results:
        core_id = result.get('core_id', 'unknown')
        core_usage[core_id] = core_usage.get(core_id, 0) + 1
    
    logger.info(f"📈 Core usage statistics:")
    for core_id, count in sorted(core_usage.items()):
        logger.info(f"   Core {core_id}: {count} tasks")
    
    return all_results

# =============================================================================
# ACTION CHECKING LLM FUNCTIONS
# =============================================================================

def create_action_evaluation_prompt(query_list: List[str], observation_memory: str, action: str) -> str:
    """Create prompt for action evaluation."""
    
    # Format query list with indices
    query_list_formatted = ""
    for i, query_item in enumerate(query_list):
        query_list_formatted += f"{i}: {query_item}\n"
    
    prompt = f"""You are an expert action evaluation system specialized in assessing research actions against user intent and context.

## TASK
Determine whether a given research action is ENTAILED, CONTRADICTED, or NEUTRAL with respect to the user query list and previous observations.

## JUDGMENT CRITERIA

### ENTAILED
The action is:
- **Directly aligned** with one or more of the user's stated intents in the query list
- **Reasonably inferred** as a logical next step from the query list and context
- **Appropriate extension** based on previous observations and research progress

### CONTRADICTED
The action:
- **Actively opposes** one or more of the user's stated intents or query requirements
- **Conflicts with** established context from previous observations
- **Contradicts** explicit instructions or constraints in the query list

### NEUTRAL
The action:
- **Lacks clear connection** to the user query list or research objectives
- **Is unsupported** by previous observations and context
- **Is irrelevant** to the current research focus

## EVIDENCE REQUIREMENTS
- **ENTAILED/CONTRADICTED**: Provide a one-sentence explanation of why the judgment was made
- **NEUTRAL**: No evidence needed

## OUTPUT FORMAT
Respond with ONLY a JSON code block:

```json
{{
    "judgment": "entailed|contradicted|neutral",
    "confidence": 0.0-1.0,
    "explanation": "One-sentence explanation for entailed/contradicted judgments only",
    "query": -1,
    "observation": -1
}}
```

## ADDITIONAL OUTPUT REQUIREMENTS
- **If ENTAILED**: Set "query" to the index of the query that the action is entailed by (from the query list), set "observation" to -1
- **If CONTRADICTED**: Set "query" to the index of the contradicted query (if contradicted by query), OR set "observation" to 1 (if contradicted by context). At least one should not be -1
- **If NEUTRAL**: Set both "query" and "observation" to -1

## EXAMPLES

**Example 1 - Entailed Action:**
- Query: "Find AI research positions at tech companies"
- Context: "Found Google AI research page with job listings"
- Action: "Search for specific AI research roles at Google"
- Judgment: "entailed" (action directly follows from query and builds on previous findings)

**Example 2 - Contradicted Action:**
- Query: "Find remote AI positions only"
- Context: "User specifically requested remote work"
- Action: "Apply for on-site position in San Francisco"
- Judgment: "contradicted" (action directly contradicts user's remote work requirement)

**Example 3 - Neutral Action:**
- Query: "Research AI job opportunities"
- Context: "Found several relevant AI positions"
- Action: "Check stock market prices for tech companies"
- Judgment: "neutral" (action is unrelated to job research objectives)

## INPUT CONTEXT

**User Query**: 
{query_list_formatted}

**Previous Observations (Context Memory)**:
{observation_memory}

**Action to Evaluate**:
{action}

Provide your evaluation:"""
    
    return prompt

def evaluate_action_with_llm_retry(api_client: OpenAI, query_list: List[str], observation_memory: str, action: str, 
                                  model: str = WEB_MODEL, max_retries: int = 3) -> Dict[str, Any]:
    """Evaluate action with LLM using retry logic for rate limiting."""
    
    prompt = create_action_evaluation_prompt(query_list, observation_memory, action)
    
    for attempt in range(max_retries):
        try:
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from markdown code blocks if present
            json_content = extract_json_from_content(content)
            
            # Try to parse JSON response
            try:
                result = json.loads(json_content)
                
                # Validate required fields
                if all(key in result for key in ["judgment", "confidence", "explanation"]):
                    # Validate judgment value
                    if result["judgment"] in ["entailed", "contradicted", "neutral"]:
                        # Ensure confidence is a float between 0 and 1
                        result["confidence"] = float(result["confidence"])
                        if not (0.0 <= result["confidence"] <= 1.0):
                            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                        
                        return result
                    else:
                        logger.warning(f"Invalid judgment value: {result['judgment']}")
                else:
                    logger.warning(f"Missing required fields in response: {result}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response (attempt {attempt + 1}): {json_content}")
                logger.debug(f"Original content: {content}")
                
                # Try to extract judgment and other fields using regex patterns
                result = _extract_fields_from_text(content)
                if result:
                    return result
                
                # Fallback: extract judgment from text if JSON parsing fails
                content_lower = content.lower()
                if "entailed" in content_lower:
                    judgment = "entailed"
                elif "contradicted" in content_lower:
                    judgment = "contradicted"
                else:
                    judgment = "neutral"
                
                return {
                    "judgment": judgment,
                    "confidence": 0.5,
                    "explanation": "Extracted from non-JSON response",
                    "query": -1,
                    "observation": -1
                }
                
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                # Rate limit hit, wait and retry
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.2f}s...")
                time.sleep(wait_time)
                continue
            else:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {
                        "judgment": "neutral",
                        "confidence": 0.0,
                        "explanation": f"API call failed after {max_retries} attempts",
                        "query": -1,
                        "observation": -1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return {
        "judgment": "neutral",
        "confidence": 0.0,
        "explanation": "Failed to get valid response from LLM",
        "query": -1,
        "observation": -1
    }

def llm_judge_action_worker_parallel(args: Tuple[int, List[Tuple[str, str, str, int, int]]]) -> List[Dict[str, Any]]:
    """
    Worker function for parallel LLM action judging using a specific API key.
    
    Args:
        args: Tuple of (api_id, list of (query, observation_memory, action, iteration_num, action_idx))
        
    Returns:
        List of dictionaries containing action judgment results and metadata
    """
    api_id, tasks = args
    
    # Get the API client for this worker
    api_client = OpenAI(
        api_key=API_KEYS[api_id],
        base_url=BASE_URL
    )
    
    results = []
    
    for task in tasks:
        try:
            query_list, observation_memory, action, iteration_num, action_idx = task
            
            # Evaluate the action using LLM with retry logic
            llm_result = evaluate_action_with_llm_retry(api_client, query_list, observation_memory, action)
            
            # Create the result structure matching the original format
            result = {
                "iteration_num": iteration_num,
                "action_idx": action_idx,
                "action": action,
                "observation_memory": observation_memory[:500] + "..." if len(observation_memory) > 500 else observation_memory,
                "full_observation_memory": observation_memory,
                "memory_length": len(observation_memory),
                "judgment": llm_result["judgment"],
                "confidence": llm_result["confidence"],
                "explanation": llm_result["explanation"],
                "query": llm_result.get("query", -1),
                "observation": llm_result.get("observation", -1),
                "scores": {
                    "entailment": 1.0 if llm_result["judgment"] == "entailed" else 0.0,
                    "contradiction": 1.0 if llm_result["judgment"] == "contradicted" else 0.0,
                    "neutral": 1.0 if llm_result["judgment"] == "neutral" else 0.0
                },
                "judgment_score": llm_result["confidence"],
                "api_id": api_id
            }
            
            results.append(result)
            
        except Exception as e:
            # Return error result if something goes wrong
            logger.error(f"⚠️ Error in API {api_id} action worker: {e}")
            query, observation_memory, action, iteration_num, action_idx = task
            result = {
                "iteration_num": iteration_num,
                "action_idx": action_idx,
                "action": action,
                "observation_memory": observation_memory[:500] + "..." if len(observation_memory) > 500 else observation_memory,
                "full_observation_memory": observation_memory,
                "memory_length": len(observation_memory),
                "judgment": "neutral",
                "confidence": 0.0,
                "explanation": f"Processing failed: {str(e)}",
                "query": -1,
                "observation": -1,
                "scores": {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
                "judgment_score": 0.0,
                "api_id": api_id,
                "error": str(e)
            }
            results.append(result)
    
    return results

def llm_judge_action_worker_single_api(args: Tuple[int, List[Tuple[str, str, str, int, int]]]) -> List[Dict[str, Any]]:
    """
    Worker function for parallel LLM action judging using single API key.
    
    Args:
        args: Tuple of (core_id, list of (query, observation_memory, action, iteration_num, action_idx))
        
    Returns:
        List of dictionaries containing action judgment results and metadata
    """
    core_id, tasks = args
    
    # Get the API client using single API key
    api_client = OpenAI(
        api_key=API_KEYS[0],
        base_url=BASE_URL
    )
    
    results = []
    
    for task in tasks:
        try:
            query_list, observation_memory, action, iteration_num, action_idx = task
            
            # Evaluate the action using LLM with retry logic
            llm_result = evaluate_action_with_llm_retry(api_client, query_list, observation_memory, action)
            
            # Create the result structure matching the original format
            result = {
                "iteration_num": iteration_num,
                "action_idx": action_idx,
                "action": action,
                "observation_memory": observation_memory[:500] + "..." if len(observation_memory) > 500 else observation_memory,
                "full_observation_memory": observation_memory,
                "memory_length": len(observation_memory),
                "judgment": llm_result["judgment"],
                "confidence": llm_result["confidence"],
                "explanation": llm_result["explanation"],
                "query": llm_result.get("query", -1),
                "observation": llm_result.get("observation", -1),
                "scores": {
                    "entailment": 1.0 if llm_result["judgment"] == "entailed" else 0.0,
                    "contradiction": 1.0 if llm_result["judgment"] == "contradicted" else 0.0,
                    "neutral": 1.0 if llm_result["judgment"] == "neutral" else 0.0
                },
                "judgment_score": llm_result["confidence"],
                "core_id": core_id
            }
            
            results.append(result)
            
        except Exception as e:
            # Return error result if something goes wrong
            logger.error(f"⚠️ Error in Core {core_id} action worker: {e}")
            query, observation_memory, action, iteration_num, action_idx = task
            result = {
                "iteration_num": iteration_num,
                "action_idx": action_idx,
                "action": action,
                "observation_memory": observation_memory[:500] + "..." if len(observation_memory) > 500 else observation_memory,
                "full_observation_memory": observation_memory,
                "memory_length": len(observation_memory),
                "judgment": "neutral",
                "confidence": 0.0,
                "explanation": f"Processing failed: {str(e)}",
                "query": -1,
                "observation": -1,
                "scores": {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
                "judgment_score": 0.0,
                "core_id": core_id,
                "error": str(e)
            }
            results.append(result)
    
    return results

def process_llm_action_judgments_parallel(all_action_args: List[Tuple[str, str, str, int, int]], num_cores: int = 255) -> List[Dict[str, Any]]:
    """
    Process all LLM action judgments in parallel using single API key with multiple CPU cores.
    
    Args:
        all_action_args: List of (query, observation_memory, action, iteration_num, action_idx) tuples
        num_cores: Number of CPU cores to use (default: 255)
        
    Returns:
        List of action judgment results
    """
    # Use single API key
    api_key = API_KEYS[0]
    
    logger.info(f"🚀 Starting parallel LLM action processing with {num_cores} CPU cores")
    logger.info(f"📊 Total action tasks: {len(all_action_args)}")
    logger.info(f"🔑 Using single API key: {api_key[:20]}...")
    
    # Distribute work evenly across CPU cores
    total_tasks = len(all_action_args)
    base_size = total_tasks // num_cores
    remainder = total_tasks % num_cores
    
    distributed_tasks = []
    start_idx = 0
    
    for core_id in range(num_cores):
        # First 'remainder' cores get one extra task
        current_size = base_size + (1 if core_id < remainder else 0)
        end_idx = start_idx + current_size
        
        if current_size > 0:  # Only add cores with tasks
            distributed_tasks.append((core_id, all_action_args[start_idx:end_idx]))
            start_idx = end_idx
    
    logger.info(f"🔄 Processing with {len(distributed_tasks)} CPU cores")
    
    # Process in parallel using multiprocessing
    all_results = []
    
    try:
        with mp.Pool(processes=len(distributed_tasks)) as pool:
            parallel_results = pool.map(llm_judge_action_worker_single_api, distributed_tasks)
            
            # Flatten results
            for core_results in parallel_results:
                all_results.extend(core_results)
    
    except Exception as e:
        logger.error(f"❌ Error in parallel action processing: {e}")
        # Fallback to sequential processing
        logger.info("🔄 Falling back to sequential action processing...")
        for core_id, core_tasks in distributed_tasks:
            if core_tasks:
                core_results = llm_judge_action_worker_single_api((core_id, core_tasks))
                all_results.extend(core_results)
    
    logger.info(f"✅ Parallel LLM action processing completed: {len(all_results)} results")
    
    # Log core usage statistics
    core_usage = {}
    for result in all_results:
        core_id = result.get('core_id', 'unknown')
        core_usage[core_id] = core_usage.get(core_id, 0) + 1
    
    logger.info(f"📈 Core usage statistics for actions:")
    for core_id, count in sorted(core_usage.items()):
        logger.info(f"   Core {core_id}: {count} action tasks")
    
    return all_results

# =============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

def llm_judge_claim(claim: str, chunk_text: str, query: str = "") -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    api_client = OpenAI(
        api_key=API_KEYS[0],
        base_url=BASE_URL
    )
    return llm_judge_claim_with_retry(api_client, claim, chunk_text, query)

def evaluate_action_with_llm(query_list: List[str], observation_memory: str, action: str, 
                            model: str = WEB_MODEL, max_retries: int = 3) -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    api_client = OpenAI(
        api_key=API_KEYS[0],
        base_url=BASE_URL
    )
    return evaluate_action_with_llm_retry(api_client, query_list, observation_memory, action, model, max_retries)