#!/usr/bin/env python3
"""
train.py — LoRA fine-tune Nemotron 3 Nano 30B on the KD corpus.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# Configuration
# ==============================================================================

STUDENT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

CORPUS_DIR = Path.home() / "pmm-demos/knowledge-distillation/data/final"
OUTPUT_DIR = Path.home() / "pmm-demos/knowledge-distillation-training/lora_adapter"
LOGS_DIR = Path.home() / "pmm-demos/knowledge-distillation-training/logs"

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "up_proj", "down_proj",                    # MoE FFN
    "in_proj",                      # Mamba
]

LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 4096
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01


def main():
    print(f"Loading tokenizer: {STUDENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {STUDENT_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"Loading dataset from {CORPUS_DIR}")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(CORPUS_DIR / "train.jsonl"),
            "validation": str(CORPUS_DIR / "validation.jsonl"),
        },
    )

    def rename_conversations(example):
        # SFTTrainer expects "messages", we have "conversations"
        return {"messages": example["conversations"]}

    dataset = dataset.map(rename_conversations, remove_columns=["conversations", "metadata"])

    print(f"Train:      {len(dataset['train'])} examples")
    print(f"Validation: {len(dataset['validation'])} examples")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOGS_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        max_seq_length=MAX_SEQ_LENGTH,
        logging_steps=3,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Training complete.")


if __name__ == "__main__":
    main()