#!/usr/bin/env python3
"""Finetune Nemotron 3 Nano on the KD corpus via Megatron-Bridge."""

import argparse
import logging
import sys
from typing import Any, Optional, Tuple

import torch
from omegaconf import OmegaConf

from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig, ProcessExampleOutput
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_finetune_config as finetune_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)

logger = logging.getLogger(__name__)

DATA_DIR = "/workspace/knowledge-distillation/data/final"
CACHE_DIR = "/workspace/mbridge_run/data_cache"


def process_kd_example(
    example: dict[str, Any], tokenizer: Optional[Any] = None
) -> ProcessExampleOutput:
    """Turn a KD corpus record into (input, output) for SFT.

    Loss is computed on `output` only.
    """
    msgs = example["conversations"]
    system = ""
    user = ""
    assistant = ""
    for m in msgs:
        role = m.get("role")
        if role == "system":
            system = m.get("content", "")
        elif role == "user":
            user = m.get("content", "")
        elif role == "assistant":
            assistant = m.get("content", "")

    _input = f"{system}\n\n{user}" if system else user
    return ProcessExampleOutput(
        input=_input,
        output=assistant,
        original_answers=[assistant],
    )


def kd_dataset_config(seq_length: int, packed_sequence: bool) -> HFDatasetConfig:
    if packed_sequence:
        dataset_kwargs = {"pad_to_max_length": True}
        packed_sequence_specs = PackedSequenceSpecs(packed_sequence_size=seq_length)
    else:
        dataset_kwargs = {}
        packed_sequence_specs = None

    return HFDatasetConfig(
        dataset_name="json",
        process_example_fn=process_kd_example,
        seq_length=seq_length,
        seed=5678,
        dataloader_type="batch",
        num_workers=1,
        do_validation=True,
        do_test=False,
        split_val_from_train=False,
        val_proportion=None,
        dataset_root=CACHE_DIR,
        dataset_kwargs=dataset_kwargs,
        packed_sequence_specs=packed_sequence_specs,
        rewrite=True,
        hf_kwargs={
            "data_files": {
                "train": f"{DATA_DIR}/train.jsonl",
                "validation": f"{DATA_DIR}/validation.jsonl",
            }
        },
    )


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Finetune Nemotron 3 Nano on KD corpus")
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--peft", type=str, default="lora")
    parser.add_argument("--packed-sequence", action="store_true")
    parser.add_argument("--seq-length", type=int, default=4096)
    return parser.parse_known_args()


def main() -> None:
    args, cli_overrides = parse_cli_args()

    cfg: ConfigContainer = finetune_config(
        seq_length=args.seq_length,
        peft=args.peft,
        packed_sequence=args.packed_sequence,
    )
    cfg.model.seq_length = args.seq_length

    # Swap SQuAD for our KD corpus
    cfg.dataset = kd_dataset_config(args.seq_length, args.packed_sequence)

    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    if args.config_file:
        yaml_overrides = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides)

    if cli_overrides:
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)

    final_overrides = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides, excluded_fields)

    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
