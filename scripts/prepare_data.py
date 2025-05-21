#!/usr/bin/env python3
# AI-Assisted (YYYY-MM-DD): Script to prepare data for Supervised Fine-Tuning (SFT).
# Reads raw JSONL data, formats it using the model's chat template, and saves it.

import os
import sys
import argparse
import json
from tqdm import tqdm

# Ensure project root is in PYTHONPATH for local model paths if necessary
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers library not found. Please install it: pip install transformers")
    sys.exit(1)

def format_data(raw_data_path: str, output_path: str, model_name_or_path: str, prompt_key: str, completion_key: str):
    """
    Reads raw data from a JSONL file, formats each entry using the tokenizer's chat template,
    and writes the formatted data to a new JSONL file.

    Args:
        raw_data_path (str): Path to the input JSONL file.
        output_path (str): Path to save the formatted JSONL file.
        model_name_or_path (str): Name or local path of the Hugging Face model for tokenizer.
        prompt_key (str): The key in the JSONL for the user's prompt/instruction.
        completion_key (str): The key in the JSONL for the model's completion/response.
    """
    print(f"Loading tokenizer for: {model_name_or_path}")
    try:
        # trust_remote_code=True is often needed for Qwen-based models like DeepSeek distillations
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Please ensure the model path '{model_name_or_path}' is correct and accessible.")
        sys.exit(1)

    # Qwen models (which DeepSeek-R1-Distill-Qwen-7B is based on) typically don't use a pad token
    # or expect padding on the left. SFT script handles padding if tokenizer.pad_token is None.
    # If tokenizer.chat_template is not explicitly set, apply_chat_template might raise error
    # or use a default one. For Qwen, it should be well-defined.

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Formatting data from {raw_data_path} into {output_path}...")

    formatted_entries = 0
    with open(raw_data_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc="Processing lines"):
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line: {line.strip()}", file=sys.stderr)
                continue

            prompt_text = record.get(prompt_key)
            completion_text = record.get(completion_key)

            if prompt_text is None or completion_text is None:
                print(f"Skipping record due to missing '{prompt_key}' or '{completion_key}': {record}", file=sys.stderr)
                continue

            # Format for Qwen-based chat models
            # The conversation should be a list of dicts, e.g., [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": completion_text}
            ]

            try:
                # `tokenize=False` returns a string. `add_generation_prompt=False` ensures assistant turn is not treated as a new prompt.
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False # Important for SFT on completed dialogues
                )
            except Exception as e:
                print(f"Error applying chat template to record: {record}. Error: {e}", file=sys.stderr)
                print("Ensure your tokenizer supports apply_chat_template or adjust formatting.", file=sys.stderr)
                continue
            
            # The SFT script expects a single JSON object per line with a "text" field
            output_record = {"text": formatted_text}
            outfile.write(json.dumps(output_record) + '\n')
            formatted_entries += 1
            
    print(f"Formatted {formatted_entries} entries and saved to {output_path}")
    if formatted_entries == 0:
        print("Warning: No entries were processed. Check your input file and keys.")

def main():
    parser = argparse.ArgumentParser(description="Prepare raw JSONL data for SFT.")
    parser.add_argument('--raw_data_path', type=str, required=True,
                        help='Path to the raw input JSONL file (e.g., data/prompts.jsonl).')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the formatted JSONL file (e.g., data/processed/sft_prepared_data.jsonl).')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Name or local path of the Hugging Face model to use for tokenization and chat templating.')
    parser.add_argument('--prompt_key', type=str, default='prompt',
                        help="The key in the input JSONL for the user's prompt/instruction (default: prompt).")
    parser.add_argument('--completion_key', type=str, default='completion',
                        help="The key in the input JSONL for the model's completion/response (default: completion).")
    
    args = parser.parse_args()

    format_data(args.raw_data_path, args.output_path, args.model_name_or_path, args.prompt_key, args.completion_key)

if __name__ == '__main__':
    main() 