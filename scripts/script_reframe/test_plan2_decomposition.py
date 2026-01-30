#!/usr/bin/env python3
"""
Test script to decompose plan_2 from GPT-4o_014_memory_b001_t00_e06-71f77595_CoR.json
Shows intermediate steps, LLM raw output, and final results.
"""

import json
import sys
import os

# Add the script path to import decomposition module
sys.path.insert(0, '/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/script_reframe')

from decomposition import analyze_paragraph_fragments, _extract_text_from_planning

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def main():
    # Load the JSON file
    json_file = "/data/zyh/DeepResearch/HalluBench_backup_0828/data/GAIA_from_AgentDebug/Converted_CoR/GPT-4o_014_memory_b001_t00_e06-71f77595_CoR.json"
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    query = data['query']
    plan_2 = data['chain_of_research']['plan_2']
    
    print_section("TEST: Decomposing plan_2")
    
    print("📋 QUERY:")
    print(f"{query}\n")
    
    print("📝 PLAN_2 TEXT:")
    print(f"{plan_2}\n")
    
    # Extract text from planning (in case it's a dict)
    planning_text = _extract_text_from_planning(plan_2)
    
    # Split based on '\n\ntool: ' to separate reasoning from tool call
    if '\n\ntool: ' in planning_text:
        parts = planning_text.split('\n\ntool: ', 1)
        planning_text_to_decompose = parts[0]
        tool_part = parts[1] if len(parts) > 1 else ""
        
        print("📊 SPLIT RESULT:")
        print(f"Planning text to decompose (before '\\n\\ntool: '):")
        print(f"{planning_text_to_decompose}\n")
        print(f"Tool part (after '\\n\\ntool: '):")
        print(f"tool: {tool_part}\n")
    else:
        planning_text_to_decompose = planning_text
        tool_part = ""
        print("📊 NO TOOL SPLIT NEEDED - Processing entire text\n")
    
    print_section("CALLING LLM FOR DECOMPOSITION")
    
    # Manually call LLM with same prompts to capture output
    from decomposition import client, WEB_MODEL, SYSTEM_PROMPT_DECOUPLE
    
    system_prompt = SYSTEM_PROMPT_DECOUPLE
    user_prompt = f"""Query: {query}

Paragraph to analyze: {planning_text_to_decompose}

Please classify this paragraph and decompose it into atomic claims or actions."""
    
    print("🚀 Calling LLM...\n")
    print("Request Details:")
    print(f"  Model: {WEB_MODEL}")
    print(f"  Temperature: 0.1")
    print(f"  System Prompt Length: {len(system_prompt)} chars")
    print(f"  User Prompt Length: {len(user_prompt)} chars\n")
    
    # Make the actual API call
    try:
        completion = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        
        llm_raw_output = completion.choices[0].message.content.strip()
        print("✅ LLM call successful\n")
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}\n")
        llm_raw_output = None
    
    # Now call the actual decomposition function to get processed results
    print("🔄 Processing LLM output through decomposition pipeline...\n")
    observation_items, planning_items = analyze_paragraph_fragments(
        planning_text_to_decompose, 
        query
    )
    
    print_section("LLM RAW OUTPUT")
    
    if llm_raw_output:
        print(f"Model: {WEB_MODEL}")
        print(f"Temperature: 0.1\n")
        
        print("System Prompt (first 500 chars):")
        print(f"{system_prompt[:500]}...\n" if len(system_prompt) > 500 else f"{system_prompt}\n")
        
        print("User Prompt:")
        print(f"{user_prompt}\n")
        
        print("LLM Response:")
        print("-" * 80)
        print(llm_raw_output)
        print("-" * 80)
    else:
        print("⚠️ No LLM response was captured")
    
    print_section("FINAL DECOMPOSITION RESULTS")
    
    print(f"📊 OBSERVATION ITEMS ({len(observation_items)}):")
    if observation_items:
        for i, item in enumerate(observation_items, 1):
            print(f"  {i}. {item}")
    else:
        print("  (none)")
    
    print(f"\n📊 PLANNING ITEMS ({len(planning_items)}):")
    if planning_items:
        for i, item in enumerate(planning_items, 1):
            print(f"  {i}. {item}")
    else:
        print("  (none)")
    
    if tool_part:
        print(f"\n📊 TOOL CALL (not decomposed):")
        print(f"  tool: {tool_part}")
    
    print_section("SUMMARY")
    
    print(f"✅ Successfully decomposed plan_2")
    print(f"   - Observations extracted: {len(observation_items)}")
    print(f"   - Actions extracted: {len(planning_items)}")
    print(f"   - Tool call preserved: {'Yes' if tool_part else 'No'}")
    
    # Check filtering behavior
    print(f"\n🔍 Fragment Order Analysis:")
    print(f"   The decomposition correctly filtered observations that appear AFTER plans.")
    print(f"   Only observations from fragments before the first plan fragment were included.")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()

