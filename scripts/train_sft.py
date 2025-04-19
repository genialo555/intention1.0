#!/usr/bin/env python3
"""
AI-Assisted (2025-04-17): Script for Supervised Fine-Tuning (SFT) baseline.
Trains a causal LM on JSONL examples and records model path in Blackboard.
"""
import os
import argparse
import json
import yaml
from datasets import load_dataset
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

    model_name = config.get('model_name', 'gpt2')
    batch_size = config.get('batch_size', 8)
    epochs = config.get('epochs', 1)
    lr = config.get('learning_rate', 5e-5)
    max_length = config.get('max_length', 512)
    logging_steps = config.get('logging_steps', 10)

    # Prepare dataset
    data_pattern = os.path.join('data', 'sft_examples', '*.jsonl')
    dataset = load_dataset('json', data_files={'train': data_pattern}, field='text')

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenization function
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=logging_steps,
        save_steps=logging_steps,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
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