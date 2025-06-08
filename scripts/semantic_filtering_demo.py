#!/usr/bin/env python3
"""
Demo script for semantic filtering based on MetaLIMEN intentions:
- Loads concepts JSONL
- Applies data filter criteria and progression stages
- Prints or saves the filtered output
"""
import os
import sys
import argparse
import json

# ensure project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.meta_limen.embedder import DeepSeekEmbedder
from modules.meta_limen.meta_limen import SimpleCurriculumGenerator


def main():
    parser = argparse.ArgumentParser(description="Semantic filtering demo for MetaLIMEN concepts")
    parser.add_argument('--concepts', type=str, required=True, help='Path to concepts.jsonl')
    parser.add_argument('--output', type=str, required=False, help='Optional output JSONL for filtering demo')
    args = parser.parse_args()

    generator = SimpleCurriculumGenerator()
    demos = []
    with open(args.concepts, 'r') as fin:
        for line in fin:
            intent = json.loads(line)
            criteria = generator.create_data_filter_criteria(intent)
            stages = generator.create_progression_stages(intent)
            demos.append({
                'domain': intent['domain'],
                'criteria': criteria,
                'stages': stages
            })

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as fout:
            for item in demos:
                fout.write(json.dumps(item) + '\n')
        print(f"Saved semantic filtering demo to {args.output}")
    else:
        for item in demos:
            print(json.dumps(item, indent=2))

if __name__ == '__main__':
    main() 