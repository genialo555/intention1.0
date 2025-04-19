#!/usr/bin/env python3
"""
AI-Assisted (2025-04-17): Script for Supervised Fine-Tuning (SFT) baseline.
Trains a causal LM on JSONL examples and records model path in Blackboard.
"""
import os
import sys
# Ensure project root is in PYTHONPATH for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import yaml
import glob
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from orchestration.controller import Blackboard


def main():
    parser = argparse.ArgumentParser(description="Train the SFT baseline model.")
    parser.add_argument('--config', type=str, required=True, help='Path to SFT config YAML')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model and metrics')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Extract and cast config parameters to correct types
    data_pattern = str(config.get('data_pattern', 'data/sft_examples/*.jsonl'))
    field = str(config.get('field', 'text'))
    model_name = str(config.get('model_name', 'gpt2'))
    batch_size = int(config.get('batch_size', 8))
    gradient_accumulation_steps = int(config.get('gradient_accumulation_steps', 1))
    epochs = int(config.get('epochs', 1))
    lr = float(config.get('learning_rate', 5e-5))
    max_length = int(config.get('max_length', 512))
    logging_steps = int(config.get('logging_steps', 10))
    evaluation_strategy = str(config.get('evaluation_strategy', 'no'))
    eval_steps = config.get('eval_steps', None)
    eval_steps = int(eval_steps) if eval_steps is not None else None
    save_steps = config.get('save_steps', None)
    save_steps = int(save_steps) if save_steps is not None else None
    warmup_steps = int(config.get('warmup_steps', 0))
    weight_decay = float(config.get('weight_decay', 0.0))
    seed = config.get('seed', None)
    seed = int(seed) if seed is not None else None
    fp16 = bool(config.get('fp16', False))
    gradient_checkpointing = bool(config.get('gradient_checkpointing', False))

    # Detect CUDA availability
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("CUDA not available: training will run on CPU. Disabling fp16.")
        fp16 = False
    no_cuda = not use_cuda

    # Prepare dataset by loading JSONL examples manually due to parsing issues
    filepaths = glob.glob(data_pattern)
    examples = []
    for fp in filepaths:
        with open(fp, 'r') as fh:
            for ln in fh:
                examples.append(json.loads(ln))
    dataset = Dataset.from_list(examples)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenization function
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    # Split dataset for evaluation if enabled
    if evaluation_strategy != 'no':
        split = tokenized.train_test_split(test_size=0.1, seed=seed or 42)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        train_dataset = tokenized
        eval_dataset = None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_steps=save_steps or logging_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=fp16,
        no_cuda=no_cuda,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed or 42,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # Save model and metrics
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w') as mf:
        json.dump(metrics, mf, indent=4)

    # Record to blackboard
    bb = Blackboard()
    bb.write('model_sft', args.output)
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main() 