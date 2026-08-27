#!/usr/bin/env python3
"""
inspect_grouped.py — Dump the structure of grouped-linear LoRA adapters.

The in-place merge crashed on:
    AttributeError: 'TEColumnParallelGroupedLinear' object has no attribute 'weight'.
                    Did you mean: 'weight0'?

MoE expert linears are fused: one module holds weight0..weightN, one per expert.
A single ParallelLinearAdapter wraps that module, so we need to know how its
lora_in/lora_out map onto the N expert weights before merging.

This script only prints shapes — it does not modify or export anything.

Run:
  docker run --rm --gpus '"device=1"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e HF_HOME=/hfcache \
      --shm-size=16g --net=host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /mnt/bigdata/kd-export/hf_cache:/hfcache \
      -v ~/pmm-demos:/workspace \
      -w /opt/Megatron-Bridge \
      nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
      torchrun --nproc-per-node=1 /workspace/mbridge_run/inspect_grouped.py \
      2>&1 | tee /mnt/bigdata/kd-export/inspect_grouped.log
"""

import sys

sys.path.insert(0, "/workspace/mbridge_run")

import torch

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_finetune_config as finetune_config,
)
from megatron.bridge.training.setup import setup
from megatron.bridge.training.state import GlobalState

from finetune_kd import kd_dataset_config

BASE_CKPT = "/workspace/megatron_ckpt"
ADAPTER_CKPT = "/workspace/mbridge_run/checkpoints"
SEQ_LENGTH = 4096


def build_config():
    cfg = finetune_config(seq_length=SEQ_LENGTH, peft="lora", packed_sequence=False)
    cfg.model.seq_length = SEQ_LENGTH
    cfg.dataset = kd_dataset_config(SEQ_LENGTH, False)
    cfg.checkpoint.pretrained_checkpoint = BASE_CKPT
    cfg.checkpoint.load = ADAPTER_CKPT
    cfg.checkpoint.save = None
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.model.expert_model_parallel_size = 1
    cfg.model.moe_enable_deepep = False
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.train_iters = 1500
    cfg.train.skip_train = True
    cfg.train.eval_iters = 0
    return cfg


def main() -> None:
    cfg = build_config()
    try:
        from megatron.bridge.training.config import runtime_config_update

        runtime_config_update(cfg)
    except (ImportError, AttributeError):
        pass

    state = GlobalState()
    state.cfg = cfg

    dataset_provider = get_dataset_provider(cfg.dataset)
    setup_output = setup(state, dataset_provider)
    model = setup_output.model
    m0 = model[0]

    print("\n" + "=" * 70, flush=True)
    print("GROUPED ADAPTER INSPECTION", flush=True)
    print("=" * 70, flush=True)

    shown = 0
    for name, mod in m0.named_modules():
        if not (hasattr(mod, "to_wrap") and hasattr(mod, "adapter")):
            continue
        inner = mod.to_wrap
        if hasattr(inner, "weight"):
            continue  # normal case, already handled

        print(f"\n--- {name} ---", flush=True)
        print(f"wrapper class : {type(mod).__name__}", flush=True)
        print(f"to_wrap class : {type(inner).__name__}", flush=True)

        # the fused expert weights
        inner_params = [(n, tuple(p.shape)) for n, p in inner.named_parameters()]
        print(f"to_wrap param count: {len(inner_params)}", flush=True)
        for n, s in inner_params[:4]:
            print(f"    {n}  {s}", flush=True)
        if len(inner_params) > 4:
            print(f"    ... ({len(inner_params) - 4} more)", flush=True)

        # anything describing expert count / partitioning
        for attr in (
            "num_gemms",
            "num_experts",
            "expert_parallel",
            "in_features",
            "out_features",
            "input_size",
            "output_size",
        ):
            if hasattr(inner, attr):
                print(f"    inner.{attr} = {getattr(inner, attr)}", flush=True)

        ad = mod.adapter
        print(f"adapter class : {type(ad).__name__}", flush=True)
        for n, p in ad.named_parameters():
            print(f"    adapter.{n}  {tuple(p.shape)}  "
                  f"max|w|={p.abs().max().item():.3e}", flush=True)
        for attr in ("dim", "alpha", "scaling", "num_experts", "input_is_parallel"):
            if hasattr(ad, attr):
                print(f"    adapter.{attr} = {getattr(ad, attr)}", flush=True)

        shown += 1
        if shown >= 3:
            break

    if shown == 0:
        print("No grouped wrapper adapters found.", flush=True)

    # Also show how ParallelLinearAdapter computes its output, so we can mirror it.
    print("\n" + "=" * 70, flush=True)
    print("ParallelLinearAdapter.forward source", flush=True)
    print("=" * 70, flush=True)
    try:
        import inspect as _inspect
        from megatron.bridge.peft.utils import ParallelLinearAdapter

        print(_inspect.getsource(ParallelLinearAdapter.forward), flush=True)
    except Exception as e:
        print(f"could not read source: {e}", flush=True)

    # And the wrapper's forward, which shows how adapter output combines with base.
    print("=" * 70, flush=True)
    print("wrapper forward source", flush=True)
    print("=" * 70, flush=True)
    for name, mod in m0.named_modules():
        if hasattr(mod, "to_wrap") and hasattr(mod, "adapter"):
            try:
                import inspect as _inspect

                print(f"class {type(mod).__name__}", flush=True)
                print(_inspect.getsource(type(mod).forward), flush=True)
            except Exception as e:
                print(f"could not read source: {e}", flush=True)
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()