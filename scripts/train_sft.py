#!/usr/bin/env python3
import os
# --- [MODIFIED] Assurer l'ordre PCI_BUS et Forcer CUDA_VISIBLE_DEVICES --- 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(f"[INFO] Explicitly set os.environ['CUDA_DEVICE_ORDER'] = {os.environ.get('CUDA_DEVICE_ORDER')}")
print(f"[INFO] Explicitly set os.environ['CUDA_VISIBLE_DEVICES'] = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
# --- Fin Modifications --- 

"""
AI-Assisted (2025-04-17): Script for Supervised Fine-Tuning (SFT) baseline.
Trains a causal LM on JSONL examples and records model path in Blackboard.
"""
import sys
# Ensure project root is in PYTHONPATH for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import yaml
import glob
import random
import numpy as np
import torch

# --- [MODIFIED] Code de débogage pour vérifier la visibilité CUDA (niveau supérieur) ---
print("[INFO][TOP_LEVEL_DEBUG] Checking CUDA availability immediately after torch import.")
if torch.cuda.is_available():
    print(f"[DEBUG][TOP_LEVEL_DEBUG] torch.cuda.is_available(): True")
    print(f"[DEBUG][TOP_LEVEL_DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[DEBUG][TOP_LEVEL_DEBUG] GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"[DEBUG][TOP_LEVEL_DEBUG] Current CUDA device after import torch: {torch.cuda.current_device()}")
elif not torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES'):
    print(f"[DEBUG][TOP_LEVEL_DEBUG] CUDA not available, but CUDA_VISIBLE_DEVICES is set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}. This might indicate an issue with CUDA setup or driver for the selected device.")
else:
    print("[DEBUG][TOP_LEVEL_DEBUG] torch.cuda.is_available(): False")
# --- Fin code de débogage (niveau supérieur) ---

from datasets import Dataset 
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig 
from orchestration.controller import Blackboard
from datasets import load_dataset

# --- [MODIFIED] Importer PEFT ---
try:
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
    print("[INFO] PEFT library loaded successfully.")
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] PEFT library not found. LoRA training will be disabled.")
# --- Fin Import PEFT ---


def main():
    parser = argparse.ArgumentParser(description="Train the SFT baseline model.")
    parser.add_argument('--config', type=str, required=True, help='Path to SFT config YAML')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model and metrics')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
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
    fp16_config = bool(config.get('fp16', False))
    gradient_checkpointing = bool(config.get('gradient_checkpointing', False))

    # --- [MODIFIED] PEFT/LoRA parameters ---
    peft_enable = bool(config.get('peft_enable', False))
    lora_r = int(config.get('lora_r', 8))
    lora_alpha = int(config.get('lora_alpha', 16))
    lora_dropout = float(config.get('lora_dropout', 0.05))
    lora_target_modules_str = config.get('lora_target_modules', None)
    lora_target_modules = lora_target_modules_str.split(',') if isinstance(lora_target_modules_str, str) else lora_target_modules_str


    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    print("[INFO][MAIN_DEBUG] Checking CUDA availability inside main() before Trainer setup.")
    if torch.cuda.is_available():
        print(f"[DEBUG][MAIN_DEBUG] torch.cuda.is_available(): True")
        print(f"[DEBUG][MAIN_DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"[DEBUG][MAIN_DEBUG] GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"[DEBUG][MAIN_DEBUG] Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("[DEBUG][MAIN_DEBUG] torch.cuda.is_available(): False")

    use_cuda_trainer = torch.cuda.is_available()
    
    load_4bit = bool(config.get('load_in_4bit', False))
    load_8bit = bool(config.get('load_in_8bit', True))
    
    if (load_4bit or load_8bit) and peft_enable:
        actual_fp16_trainer = False 
        print("[INFO] Quantization (4/8bit) and PEFT enabled, Trainer fp16 will be False.")
    elif use_cuda_trainer:
        actual_fp16_trainer = fp16_config
        print(f"[INFO] PEFT not enabled or no quantization. Trainer fp16 set to: {actual_fp16_trainer}")
    else: 
        actual_fp16_trainer = False
        if fp16_config: print("[WARN] CUDA not available, disabling fp16 for Trainer.")


    dataset = load_dataset('json', data_files={config.get('dataset_split', 'train'): data_pattern}, split=config.get('dataset_split', 'train'))
    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Using field '{field}' for text content.")
    if field not in dataset.column_names:
         raise ValueError(f"Field '{field}' not found in dataset. Available fields: {dataset.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[INFO] tokenizer.pad_token set to tokenizer.eos_token: {tokenizer.eos_token}")


    quantization_config_bnb = None
    if load_4bit:
        compute_dtype_str = config.get("bnb_4bit_compute_dtype", "float16")
        compute_dtype = getattr(torch, compute_dtype_str, torch.float16)
        quantization_config_bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", False)
        )
        print(f"Loading model '{model_name}' with 4-bit quantization (compute_dtype: {compute_dtype_str})...")
    elif load_8bit:
        quantization_config_bnb = BitsAndBytesConfig(load_in_8bit=True)
        print(f"Loading model '{model_name}' with 8-bit quantization...")
    else:
        print(f"Loading model '{model_name}' in full precision...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config_bnb, 
        trust_remote_code=True,
    )
    
    if quantization_config_bnb: print(f"Model loaded with {'4-bit' if load_4bit else '8-bit'} quantization.")
    else: print("Model loaded in full precision.")

    if peft_enable and PEFT_AVAILABLE:
        print("[INFO] PEFT enabled. Preparing model for k-bit training and applying LoRA config...")
        if load_4bit or load_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
            print("[INFO] Model prepared for k-bit training.")

        if not lora_target_modules and ("qwen" in model_name.lower() or "deepseek" in model_name.lower()):
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            print(f"[INFO] No lora_target_modules specified, defaulting for Qwen/DeepSeek: {lora_target_modules}")
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none", 
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_target_modules
        )
        model = get_peft_model(model, peft_config)
        print("[INFO] LoRA model created.")
        model.print_trainable_parameters()
    elif peft_enable and not PEFT_AVAILABLE:
        print("[WARN] peft_enable is true in config, but PEFT library is not available. Proceeding without LoRA.")


    def tokenize_fn(examples): return tokenizer(examples[field], truncation=True, max_length=max_length, padding='max_length')
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=[field])
    tokenized = tokenized.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    
    train_dataset, eval_dataset = tokenized, None
    if evaluation_strategy != 'no' and float(config.get('test_size', 0.1)) > 0 and len(tokenized) > 1:
        split_data = tokenized.train_test_split(test_size=float(config.get('test_size', 0.1)), seed=seed or 42)
        train_dataset, eval_dataset = split_data['train'], split_data['test']
    elif float(config.get('test_size', 0.1)) > 0 and len(tokenized) <=1:
        print(f"[WARN] Not enough data for test split. Using all data for training.")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, 
        logging_steps=logging_steps,
        eval_strategy=evaluation_strategy if eval_dataset is not None else "no",
        eval_steps=eval_steps or logging_steps,
        save_strategy="steps",
        save_steps=save_steps or logging_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=actual_fp16_trainer,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed or 42,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    print(f"[INFO] Trainer initialized. Model device: {model.device}. Trainer device: {trainer.args.device}")
    if torch.cuda.is_available(): print(f"[INFO] PyTorch current device before train: {torch.cuda.current_device()}")

    trainer.train()
    
    if eval_dataset is not None: metrics = trainer.evaluate(eval_dataset=eval_dataset)
    else: metrics = {}

    if peft_enable and PEFT_AVAILABLE:
        print("[INFO] Saving PEFT adapters...")
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)

    else: 
        print("[INFO] Saving full model...")
        trainer.save_model(args.output)
        
    with open(os.path.join(args.output, 'metrics.json'), 'w') as mf: json.dump(metrics, mf, indent=4)

    Blackboard().write('model_sft', args.output)
    print(f"Model (or adapters) saved to {args.output}")
    print(f"Metrics saved to {os.path.join(args.output, 'metrics.json')}")

if __name__ == '__main__':
    main()