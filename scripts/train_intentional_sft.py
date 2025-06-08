#!/usr/bin/env python3
"""
Train a Supervised Fine-Tuning (SFT) model guided by MetaLIMEN intentions.
"""
import os
import sys
import argparse
import yaml
import logging
import torch

# HF & datasets
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    IntervalStrategy,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    Trainer,
)  # bitsandbytes integration for 8-bit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.meta_limen.meta_limen import MetaLIMEN

try:
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] PEFT library not found. LoRA training disabled.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Train SFT model with MetaLIMEN guidance")
    parser.add_argument('--config', type=str, required=True, help='Path to meta_limen config YAML')
    parser.add_argument('--prompts', type=str, required=True, help='Path to JSONL prompts file')
    parser.add_argument('--model_name_or_path', type=str, default=None, help='HF model identifier or local path (defaults to config pretrained_embeddings)')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--peft', action='store_true', help='Enable PEFT/LoRA')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default=None, help='Comma-separated target modules for LoRA')
    parser.add_argument('--sft_config', type=str, default='configs/sft_config.yaml', help='Path to SFT training config YAML')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Determine training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    # Load SFT-specific training config
    with open(args.sft_config, 'r') as f:
        sft_cfg = yaml.safe_load(f)

    # Helper to cast numeric strings to proper numeric types
    def _as_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return v

    # Load MetaLIMEN and compute intentions & curriculum
    meta = MetaLIMEN(args.config)
    meta_intentions = meta.define_learning_intentions()
    curriculum = meta.generate_sft_curriculum(meta_intentions)

    # Load raw prompts JSONL
    logging.info(f"Loading prompts from {args.prompts}")
    raw_ds = load_dataset('json', data_files={'train': args.prompts}, split='train')

    # Filter examples by domain keywords
    filtered = []
    if meta.config.get('curriculum_generation', {}).get('data_filtering', True):
        logging.info("Applying keyword-based filtering per domain")
        for intent in meta_intentions:
            domain = intent['domain']
            keywords = curriculum['data_filtering_criteria'].get(domain, {}).get('keywords', [])
            if not keywords:
                continue
            subset = raw_ds.filter(
                lambda x: any(kw.lower() in x['prompt'].lower() for kw in keywords)
            )
            for ex in subset:
                filtered.append(ex)
        if not filtered:
            logging.warning("No examples matched filtering, using full dataset")
            filtered = list(raw_ds)
    else:
        logging.info("Skipping filtering, using full dataset")
        filtered = list(raw_ds)

    # Determine model source: CLI or SFT config
    model_source = args.model_name_or_path or sft_cfg.get('model_name') or meta.config.get('pretrained_embeddings')
    logging.info(f"Using model and tokenizer: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load model: 8-bit quantization if enabled in SFT config
    bnb_config = None
    if sft_cfg.get('load_in_8bit', False):
        try:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                quantization_config=bnb_config,
                trust_remote_code=sft_cfg.get('trust_remote_code', True),
                device_map='auto',
                offload_folder='offload',
                offload_state_dict=True,
            )
            logging.info("Model loaded with 8-bit quantization and offload.")
        except Exception as e:
            logging.warning(f"8-bit quantization failed ({e}), loading full precision.")
            model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=sft_cfg.get('trust_remote_code', True),
                device_map='auto'
            )
    else:
        # Full precision load (use fp16 if enabled & CUDA available to save VRAM)
        load_kwargs = dict(
            trust_remote_code=sft_cfg.get('trust_remote_code', True),
            device_map='auto',
        )
        if torch.cuda.is_available() and sft_cfg.get('fp16', False):
            load_kwargs['torch_dtype'] = torch.float16
        model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    # Enable gradient checkpointing if configured
    if sft_cfg.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Ensure compatibility
    # Move model to selected device unless we are using 8-bit quantization (bitsandbytes)
    if not (bnb_config and getattr(bnb_config, "load_in_8bit", False)):
        try:
            model.to(device)
        except (RuntimeError, ValueError) as e:
            # When using `device_map='auto'` or offloaded sub-modules, .to() can fail; just log and proceed
            logging.warning(f"Skipping model.to(device) : {e}")

    # Prepare LoRA if requested
    if args.peft and PEFT_AVAILABLE:
        # convert comma-separated modules to list
        targets = args.lora_target_modules.split(',') if isinstance(args.lora_target_modules, str) else args.lora_target_modules
        # prepare model for k-bit training if quantized
        if bnb_config and bnb_config.load_in_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets
        )
        model = get_peft_model(model, peft_config)
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()

    # Prepare Dataset for HF
    logging.info(f"Total examples for SFT: {len(filtered)}")
    # Combine prompt and completion for causal LM using tokenizer.eos_token
    combined = [{'text': ex['prompt'] + tokenizer.eos_token + ex['completion']} for ex in filtered]
    train_ds = Dataset.from_list(combined)

    # Tokenization respecting max_length from SFT config (default 512 if absent)
    max_len_cfg = int(sft_cfg.get('max_length', 512))
    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=max_len_cfg, padding='max_length')
    tokenized = train_ds.map(tokenize_fn, batched=False)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Optionally split off evaluation set
    evaluation_strategy_cfg = str(sft_cfg.get('evaluation_strategy', 'no'))
    val_split = float(sft_cfg.get('validation_split', 0.1))
    train_dataset, eval_dataset = tokenized, None
    if evaluation_strategy_cfg != 'no' and val_split > 0 and len(tokenized) > 1:
        split_data = tokenized.train_test_split(test_size=val_split, seed=sft_cfg.get('seed', 42))
        train_dataset, eval_dataset = split_data['train'], split_data['test']

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments from SFT config, disabling evaluation if no eval_dataset
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=sft_cfg.get('overwrite_output_dir', True),
        num_train_epochs=sft_cfg.get('epochs', args.epochs),
        per_device_train_batch_size=sft_cfg.get('per_device_train_batch_size', args.batch_size),
        learning_rate=_as_float(sft_cfg.get('learning_rate', args.lr)),
        logging_dir=os.path.join(args.output_dir, sft_cfg.get('logging_dir', 'logs')),
        logging_steps=sft_cfg.get('logging_steps', 50),
        save_steps=sft_cfg.get('save_steps', 500),
        save_total_limit=sft_cfg.get('save_total_limit', 2),
        eval_strategy=IntervalStrategy(evaluation_strategy_cfg) if eval_dataset is not None else IntervalStrategy("no"),
        eval_steps=sft_cfg.get('eval_steps', None),
        warmup_steps=sft_cfg.get('warmup_steps', 0),
        weight_decay=_as_float(sft_cfg.get('weight_decay', 0.0)),
        fp16=sft_cfg.get('fp16', False),
        gradient_checkpointing=sft_cfg.get('gradient_checkpointing', False),
        seed=sft_cfg.get('seed', None),
        report_to=sft_cfg.get('report_to', None),
    )

    # -----------------------------------------------------------------------------
    # Workaround: ensure DataLoader uses CPU-based RNG to avoid the runtime error
    # "Expected a 'cpu' device type for generator but found 'cuda'" in RandomSampler
    # when default torch device is CUDA.
    # -----------------------------------------------------------------------------

    class CpuGeneratorTrainer(Trainer):
        """Trainer subclass that constructs the train dataloader with a CPU
        torch.Generator so the RandomSampler's internal `torch.randperm` call runs
        on CPU regardless of the model/device placement.
        """

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            if getattr(self, "_train_dataloader", None) is not None:
                return self._train_dataloader

            cpu_gen = torch.Generator(device="cpu")

            shuffle = True  # default
            if hasattr(self.args, "dataloader_shuffle"):
                shuffle = bool(self.args.dataloader_shuffle)

            sampler = RandomSampler(self.train_dataset, generator=cpu_gen) if shuffle else SequentialSampler(self.train_dataset)

            self._train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            return self._train_dataloader

    # Trainer (with fallback for meta-tensor .to() errors)
    try:
        trainer = CpuGeneratorTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    except NotImplementedError as e:
        logging.warning(f"Trainer .to() error: {e}, falling back to CPU-only training.")
        # Force CPU training
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        training_args.no_cuda = True
        training_args.fp16 = False
        # Move model to CPU
        model = model.to('cpu')
        trainer = CpuGeneratorTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    # Train
    logging.info("Starting SFT training...")
    trainer.train()
    # Save LoRA adapters or full model
    if args.peft and PEFT_AVAILABLE:
        logging.info("Saving LoRA adapters...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    logging.info(f"Model saved to {args.output_dir}")


if __name__ == '__main__':
    main() 