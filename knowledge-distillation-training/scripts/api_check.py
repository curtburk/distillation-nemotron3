#!/usr/bin/env python3
"""API validation — verify all training-related objects construct correctly."""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")

print("=== API COMPATIBILITY CHECK ===")

try:
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    print("✓ All imports OK")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# SFTConfig — with the fixes
try:
    cfg = SFTConfig(
        output_dir="/tmp/api_test",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        max_seq_length=4096,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
    )
    print("✓ SFTConfig constructs OK")
except Exception as e:
    print(f"✗ SFTConfig failed: {e}")
    exit(1)

# LoraConfig
try:
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print("✓ LoraConfig constructs OK")
except Exception as e:
    print(f"✗ LoraConfig failed: {e}")
    exit(1)

# Dataset load
try:
    ds = load_dataset(
        "json",
        data_files={"train": "/home/gb300/pmm-demos/knowledge-distillation/data/final/train.jsonl"},
    )
    print(f"✓ Dataset loads OK: {len(ds['train'])} examples")
    print(f"  Sample keys: {list(ds['train'][0].keys())}")
    print(f"  Sample messages: {len(ds['train'][0]['conversations'])}")
except Exception as e:
    print(f"✗ Dataset load failed: {e}")
    exit(1)

print("=== ALL CHECKS PASSED ===")