#!/usr/bin/env python3
# /home/ubuntu/LLaMA-Factory/scripts/convert_to_alpaca.py

import json
import os
from typing import Dict, List, Any
import argparse

def process_similarity_json(input_path: str, output_path: str):
    """
    Process similarity jsonl file, convert to alpaca format and sort by similarity
    
    Args:
        input_path: Input jsonl file path
        output_path: Output alpaca format json file path
    """
    # Read jsonl file
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Remove duplicates, keep later entries
    prompt_dict = {}
    for item in data:
        prompt = item.get('prompt')
        if prompt is not None:
            prompt_dict[prompt] = item
    
    unique_data = list(prompt_dict.values())
    
    # Sort by similarity from high to low
    sorted_data = sorted(unique_data, key=lambda x: x.get('similarity', 0), reverse=True)
    
    # Convert to alpaca format
    alpaca_data = []
    for item in sorted_data:
        alpaca_item = {
            "prompt": item.get('prompt', ''),
            "completion": item.get('answer', ''),
            "similarity": item.get('similarity', 0),
        }
        if alpaca_item['completion'] in alpaca_item['prompt']:
            # 跳过无效数据
            # print(f"Invalid data: {item}")
            continue
            # pass 
        alpaca_data.append(alpaca_item)
    
    # Save results as a single JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processing complete! Processed {len(data)} records, {len(alpaca_data)} records after deduplication")
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert similarity.jsonl to alpaca format')
    parser.add_argument('--input', type=str, default='logs/20250324_195641_similarity.jsonl', 
                      help='Input file path')
    parser.add_argument('--output', type=str, default='/home/ubuntu/LLaMA-Factory/data/grpo_sorted.json',
                      help='Output file path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist")
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    process_similarity_json(args.input, args.output)

if __name__ == "__main__":
    main()