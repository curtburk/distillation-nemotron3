#!/usr/bin/env python3
"""Train on 10 examples for 1 epoch — proves the setup works before the full run."""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

STUDENT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CORPUS_DIR = Path.home() / "pmm-demos/knowledge-distillation/data/final"
OUTPUT_DIR = Path.home() / "pmm-demos/knowledge-distillation-training/smoke_output"


def main():
    print("=== SMOKE TEST — 10 examples, 1 epoch ===")

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (this may take a while first time due to download)...")
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
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(CORPUS_DIR / "train.jsonl")},
    )
    dataset["train"] = dataset["train"].select(range(10))
    dataset = dataset.map(
        lambda ex: {"messages": ex["conversations"]},
        remove_columns=["conversations", "metadata"],
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=1e-5,
        bf16=True,
        max_seq_length=2048,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Running smoke test training...")
    trainer.train()
    print("=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()