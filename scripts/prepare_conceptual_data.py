#!/usr/bin/env python3
"""
Script to prepare conceptual corpus for MetaLIMEN:
- Loads domain configs
- Embeds domain descriptions
- Writes out JSONL files into a specified output directory
"""
import os
import sys
import argparse
import yaml
import json

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.meta_limen.meta_limen import MetaLIMEN


def main():
    parser = argparse.ArgumentParser(description="Prepare conceptual corpus for MetaLIMEN")
    parser.add_argument('--config', type=str, required=True, help='Path to meta_limen config YAML')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to write conceptual corpus JSONL files')
    args = parser.parse_args()

    # instantiate MetaLIMEN to use its embedder
    ml = MetaLIMEN(args.config)
    intents = ml.define_learning_intentions()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'concepts.jsonl')
    with open(out_path, 'w') as fout:
        for intent in intents:
            # write one JSON object per line
            fout.write(json.dumps(intent) + '\n')
    print(f"Prepared {len(intents)} concept entries in {out_path}")


if __name__ == '__main__':
    main() 