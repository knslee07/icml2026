#!/usr/bin/env python3
"""
Domain-Specific LoRA Fine-tuning for Information Extraction

This script demonstrates the training process for domain-specific IE alignment
using QLoRA (4-bit quantization + LoRA) on Gemma 3 27B.

Paper: "Domain-Specific Alignment for Information Extraction:
        A Framework for Social Science Data Collection"

HARDWARE REQUIREMENTS:
    - A100 80GB or H100 80GB for 27B model
    - A40 48GB for 12B model (batch_size=1)

TRAINING CONFIGURATION:
    Model:              Gemma 3 27B-IT
    Quantization:       4-bit NF4 (QLoRA)
    LoRA rank:          8
    LoRA alpha:         16
    Learning rate:      2e-4 (cosine decay)
    Batch size:         4 (27B) / 2 (12B)
    Gradient accum:     4
    Effective batch:    16
    Max sequence:       5120 tokens
    Epochs:             3
    Validation split:   20%

TRAINING DATA FORMAT (JSONL):
    {
        "instruction": "You are a meticulous reader of proxy-statement excerpts...",
        "input": "TEMPORAL CONTEXT: This document was filed in 2023...\nTEXT TO ANALYZE:\n...",
        "output": "{RET: 'Pearl Meyer & Partners, LLC'}, {SURV: 'Mercer', 'Radford'}"
    }

USAGE:
    python train.py --model 27b --data training_data.jsonl
"""

import os
import gc
import torch
import warnings
import argparse
import logging
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIGS = {
    "12b": {
        "name": "google/gemma-3-12b-it",
        "batch_size": 2,
        "gradient_accumulation": 4,
    },
    "27b": {
        "name": "google/gemma-3-27b-it",
        "batch_size": 4,
        "gradient_accumulation": 4,
    }
}

LORA_CONFIG = {
    "r": 8,                    # LoRA rank
    "alpha": 16,               # LoRA alpha (scaling factor)
    "dropout": 0.05,           # LoRA dropout
    "target_modules": [        # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
}

TRAINING_CONFIG = {
    "max_seq_length": 5120,    # Context window (matches inference)
    "validation_split": 0.2,   # 20% for validation
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "max_grad_norm": 0.3,
    "lr_scheduler": "cosine",
}


# =============================================================================
# Prompt Formatting
# =============================================================================

def make_gemma_prompt(instruction: str, input_text: str) -> str:
    """
    Format input for Gemma 3 instruction-tuned model.

    Uses the standard Gemma chat format:
        <start_of_turn>user
        {instruction}

        Input: {input}<end_of_turn>
        <start_of_turn>model
        {output}<end_of_turn>
    """
    instruction = instruction.replace('\\n', '\n').strip()
    return (
        "<start_of_turn>user\n"
        f"{instruction}\n\n"
        f"Input: {input_text.strip()}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


# =============================================================================
# Model Setup
# =============================================================================

def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """
    Load model with 4-bit quantization for memory-efficient training.

    QLoRA approach:
        1. Load base model in 4-bit precision (NF4 quantization)
        2. Freeze base weights
        3. Add trainable LoRA adapters
    """
    logger.info(f"Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,      # Nested quantization
    )

    # Load model
    try:
        from transformers import Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    logger.info(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return model, tokenizer


def setup_lora(model):
    """
    Apply LoRA adapters to the model.

    LoRA (Low-Rank Adaptation):
        - Adds small trainable matrices to attention/MLP layers
        - Only ~0.5% of parameters are trainable
        - Preserves base model capabilities (reduces alignment tax)
    """
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["alpha"],
        lora_dropout=LORA_CONFIG["dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# =============================================================================
# Data Preprocessing
# =============================================================================

def preprocess_dataset(examples, tokenizer, max_length):
    """
    Tokenize training examples with label masking.

    Key insight: We mask the prompt tokens (set to -100) so the model
    only learns to predict the output, not memorize the instruction.
    """
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        prompt = make_gemma_prompt(inst, inp)
        full_text = prompt + out + "<end_of_turn>"
        texts.append(full_text)

    # Tokenize
    model_inputs = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # Create labels with prompt masking
    labels = []
    for i, (inst, inp) in enumerate(zip(examples["instruction"], examples["input"])):
        prompt = make_gemma_prompt(inst, inp)
        prompt_tokens = tokenizer(prompt, padding=False, truncation=True, max_length=max_length)
        prompt_length = len(prompt_tokens["input_ids"])

        # Copy input_ids and mask prompt portion
        label_ids = model_inputs["input_ids"][i].copy()
        label_ids[:prompt_length] = [-100] * prompt_length

        # Also mask padding tokens
        for j in range(len(label_ids)):
            if model_inputs["input_ids"][i][j] == tokenizer.pad_token_id:
                label_ids[j] = -100

        labels.append(label_ids)

    model_inputs["labels"] = labels
    return model_inputs


def filter_long_samples(dataset, tokenizer, max_length):
    """
    Remove samples exceeding max context length.

    Important: We filter rather than truncate to avoid training on
    incomplete examples where output might reference truncated text.
    """
    def is_valid_length(example):
        full_text = make_gemma_prompt(example["instruction"], example["input"]) + example["output"] + "<end_of_turn>"
        tokens = tokenizer(full_text, return_length=True, add_special_tokens=True)
        return tokens['length'][0] <= max_length

    filtered = dataset.filter(is_valid_length)
    logger.info(f"Filtered: {len(dataset)} -> {len(filtered)} samples")
    return filtered


# =============================================================================
# Training
# =============================================================================

def train(args):
    """Main training loop."""

    # Get model config
    config = MODEL_CONFIGS[args.model]

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load dataset
    logger.info(f"Loading dataset from {args.data}")
    dataset = load_dataset("json", data_files=args.data)["train"]
    logger.info(f"Loaded {len(dataset)} samples")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config["name"])
    model = setup_lora(model)

    # Filter long samples
    dataset = filter_long_samples(dataset, tokenizer, TRAINING_CONFIG["max_seq_length"])

    # Tokenize
    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        lambda x: preprocess_dataset(x, tokenizer, TRAINING_CONFIG["max_seq_length"]),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Train/validation split
    split = tokenized.train_test_split(
        test_size=TRAINING_CONFIG["validation_split"],
        seed=42
    )
    logger.info(f"Train: {len(split['train'])}, Validation: {len(split['test'])}")

    # Output directory
    output_dir = args.output or f"./checkpoints/gemma3_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_drop_last=True,
        group_by_length=True,
        report_to="none",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final evaluation
    metrics = trainer.evaluate()
    logger.info(f"Final eval_loss: {metrics['eval_loss']:.4f}")

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Training complete! Model saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain-specific LoRA fine-tuning")
    parser.add_argument("--model", choices=["12b", "27b"], default="27b",
                        help="Model size (12b or 27b)")
    parser.add_argument("--data", required=True,
                        help="Path to training data (JSONL format)")
    parser.add_argument("--output", default=None,
                        help="Output directory for checkpoints")

    args = parser.parse_args()
    train(args)
