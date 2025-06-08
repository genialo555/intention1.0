#!/usr/bin/env python3
"""
Script to map domain embeddings to normalized intention vectors:
- Loads config and meta-intentions
- Applies DomainIntentionMapper.map() to each vector
- Writes out JSONL file with normalized vectors
"""
import os
import sys
import argparse
import yaml
import json

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.meta_limen.meta_limen import MetaLIMEN
from modules.meta_limen.meta_limen import DomainIntentionMapper


def main():
    parser = argparse.ArgumentParser(description="Map domain embeddings to normalized intention vectors")
    parser.add_argument('--config', type=str, required=True, help='Path to meta_limen config YAML')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file for mapped intents')
    args = parser.parse_args()

    ml = MetaLIMEN(args.config)
    intents = ml.define_learning_intentions()
    mapper = DomainIntentionMapper()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as fout:
        for intent in intents:
            # normalize vector
            mapped = mapper.map(intent['vector'])
            intent['vector_normalized'] = mapped
            fout.write(json.dumps(intent) + '\n')
    print(f"Mapped {len(intents)} domain vectors into {args.output}")

if __name__ == '__main__':
    main() 