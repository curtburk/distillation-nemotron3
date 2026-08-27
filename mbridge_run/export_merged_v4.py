#!/usr/bin/env python3
"""
export_merged_v4.py — Merge LoRA adapters into base weights in-place, then export to HF.

Background
----------
Megatron-Bridge's own merge (_merge_lora_adapter_weights, model_bridge.py:668) fires
only for conversion tasks whose param_name ends in ".adapter.linear_out.weight" and
derives the base as ".to_wrap.weight". That naming belongs to ParallelLinearAdapter,
a true wrapper module.

Nemotron 3 Nano's LoRA is mostly SUBCLASS adapters:
    TELinearAdapter(te.Linear)      726
    LinearAdapter(nn.Linear)        780
    ParallelLinearAdapter(Module)    54
Subclass adapters keep the base weight as self.weight and hold factors as
self.lora_a / self.lora_b with self.scaling. No ".to_wrap" exists, so the built-in
merge never matches and the export writes base weights unchanged. Verified: the first
export was bit-identical to base (max|delta| == 0 across every module family), which
is why stock and "distilled" produced 28/30 identical completions.

This script folds adapters in directly:
    W += scaling * (B @ A)
mirroring TELinearAdapter.forward:
    lora_res = self.lora_b(self.lora_a(x)) * self.scaling
    return te.Linear.forward(self, x) + lora_res

The 54 ParallelLinearAdapter instances wrap TEColumnParallelGroupedLinear (fused MoE
experts: weight0..weight128, no single .weight). Inspection showed their
adapter.linear_out.weight is all zeros — LoRA inits B to zero and these never trained,
so B @ A == 0 and merging them would be a no-op anyway. They are skipped, and the
grouped guard prevents the AttributeError that killed v3:
    'TEColumnParallelGroupedLinear' object has no attribute 'weight'

Run
---
  docker run --rm --gpus '"device=1"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e HF_HOME=/hfcache \
      --shm-size=16g --net=host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /mnt/bigdata/kd-export/hf_cache:/hfcache \
      -v /mnt/bigdata/kd-export/output:/output \
      -v ~/pmm-demos:/workspace \
      -w /opt/Megatron-Bridge \
      nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
      torchrun --nproc-per-node=1 /workspace/mbridge_run/export_merged_v4.py \
      2>&1 | tee /mnt/bigdata/kd-export/export_v4.log
"""

import sys

sys.path.insert(0, "/workspace/mbridge_run")

from collections import Counter

import torch

from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_finetune_config as finetune_config,
)
from megatron.bridge.training.setup import setup
from megatron.bridge.training.state import GlobalState

from finetune_kd import kd_dataset_config


HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
BASE_CKPT = "/workspace/megatron_ckpt"
ADAPTER_CKPT = "/workspace/mbridge_run/checkpoints"
OUTPUT_DIR = "/output/nemotron3-nano-kd-merged"

SEQ_LENGTH = 4096
PACKED_SEQUENCE = False

DO_EXPORT = True


def build_config():
    cfg = finetune_config(
        seq_length=SEQ_LENGTH,
        peft="lora",
        packed_sequence=PACKED_SEQUENCE,
    )
    cfg.model.seq_length = SEQ_LENGTH
    cfg.dataset = kd_dataset_config(SEQ_LENGTH, PACKED_SEQUENCE)

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


def _rel_delta(before: torch.Tensor, after: torch.Tensor) -> float:
    return (after - before).abs().mean().item() / max(before.abs().mean().item(), 1e-12)


def merge_adapters_inplace(model) -> dict:
    """Fold every trained LoRA adapter into its base weight."""
    stats = {
        "subclass_merged": 0,
        "wrapper_merged": 0,
        "skipped_zero_b": 0,
        "skipped_grouped": 0,
        "skipped_shape": 0,
        "max_rel_delta": 0.0,
        "sum_rel_delta": 0.0,
        "examples": [],
        "grouped_examples": [],
    }

    for chunk in model:
        for name, mod in chunk.named_modules():
            cls = type(mod).__name__

            # ---- subclass adapters: TELinearAdapter, LinearAdapter ----
            if hasattr(mod, "lora_a") and hasattr(mod, "lora_b") and hasattr(mod, "weight"):
                with torch.no_grad():
                    A = mod.lora_a.weight          # [dim, in_features]
                    B = mod.lora_b.weight          # [out_features, dim]
                    scaling = float(getattr(mod, "scaling", 1.0))

                    if B.abs().max().item() == 0.0:
                        stats["skipped_zero_b"] += 1
                        continue

                    delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
                    if delta.shape != mod.weight.shape:
                        stats["skipped_shape"] += 1
                        continue

                    before = mod.weight.detach().float().clone()
                    mod.weight.add_(delta.to(mod.weight.dtype))
                    rel = _rel_delta(before, mod.weight.detach().float())

                    stats["subclass_merged"] += 1
                    stats["sum_rel_delta"] += rel
                    stats["max_rel_delta"] = max(stats["max_rel_delta"], rel)
                    if len(stats["examples"]) < 6:
                        stats["examples"].append(f"{name} ({cls}) rel={rel:.4%}")
                continue

            # ---- wrapper adapters: ParallelLinearAdapter ----
            if hasattr(mod, "to_wrap") and hasattr(mod, "adapter"):
                inner = mod.to_wrap

                # Grouped linear (fused MoE experts: weight0..weightN, no .weight).
                # Their adapter.linear_out is all zeros, so nothing to merge anyway.
                if not hasattr(inner, "weight"):
                    stats["skipped_grouped"] += 1
                    if len(stats["grouped_examples"]) < 3:
                        n_w = sum(1 for n, _ in inner.named_parameters())
                        b = getattr(getattr(mod.adapter, "linear_out", None), "weight", None)
                        bmax = b.abs().max().item() if b is not None else float("nan")
                        stats["grouped_examples"].append(
                            f"{name} -> {type(inner).__name__} "
                            f"({n_w} fused params, linear_out max|w|={bmax:.3e})"
                        )
                    continue

                with torch.no_grad():
                    ad = mod.adapter
                    lin_in = getattr(ad, "linear_in", None)
                    lin_out = getattr(ad, "linear_out", None)
                    if lin_in is None or lin_out is None:
                        stats["skipped_shape"] += 1
                        continue

                    A = lin_in.weight
                    B = lin_out.weight
                    alpha = getattr(ad, "alpha", None)
                    dim = getattr(ad, "dim", None)
                    if alpha and dim:
                        scaling = float(alpha) / float(dim)
                    else:
                        scaling = float(getattr(ad, "scaling", 1.0))

                    if B.abs().max().item() == 0.0:
                        stats["skipped_zero_b"] += 1
                        continue

                    delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
                    base = inner.weight
                    if delta.shape != base.shape:
                        stats["skipped_shape"] += 1
                        continue

                    before = base.detach().float().clone()
                    base.add_(delta.to(base.dtype))
                    rel = _rel_delta(before, base.detach().float())

                    stats["wrapper_merged"] += 1
                    stats["sum_rel_delta"] += rel
                    stats["max_rel_delta"] = max(stats["max_rel_delta"], rel)
                    if len(stats["examples"]) < 6:
                        stats["examples"].append(f"{name} ({cls}) rel={rel:.4%}")

    return stats


def main() -> None:
    print("Building config...", flush=True)
    cfg = build_config()

    try:
        from megatron.bridge.training.config import runtime_config_update

        runtime_config_update(cfg)
        print("Applied runtime_config_update", flush=True)
    except (ImportError, AttributeError):
        print("runtime_config_update not available, continuing", flush=True)

    state = GlobalState()
    state.cfg = cfg

    print(f"Base weights:    {BASE_CKPT}", flush=True)
    print(f"Adapter weights: {ADAPTER_CKPT}", flush=True)

    dataset_provider = get_dataset_provider(cfg.dataset)
    setup_output = setup(state, dataset_provider)
    model = setup_output.model
    print(f"Model built: {len(model)} chunk(s)", flush=True)

    classes = Counter(type(m).__name__ for _, m in model[0].named_modules())
    adapter_classes = {k: v for k, v in classes.items() if "Adapter" in k}
    print(f"adapter modules: {adapter_classes}", flush=True)

    if not adapter_classes:
        print("ERROR: no adapter modules found — nothing to merge.", flush=True)
        sys.exit(1)

    print("\nMerging adapters in-place...", flush=True)
    stats = merge_adapters_inplace(model)

    merged = stats["subclass_merged"] + stats["wrapper_merged"]
    print(f"  subclass adapters merged  : {stats['subclass_merged']}", flush=True)
    print(f"  wrapper adapters merged   : {stats['wrapper_merged']}", flush=True)
    print(f"  skipped (lora_b all zero) : {stats['skipped_zero_b']}", flush=True)
    print(f"  skipped (grouped MoE)     : {stats['skipped_grouped']}", flush=True)
    print(f"  skipped (shape mismatch)  : {stats['skipped_shape']}", flush=True)
    print(f"  max relative delta        : {stats['max_rel_delta']:.4%}", flush=True)
    if merged:
        print(f"  mean relative delta       : {stats['sum_rel_delta'] / merged:.4%}", flush=True)

    if stats["examples"]:
        print("  merged examples:", flush=True)
        for e in stats["examples"]:
            print(f"    {e}", flush=True)
    if stats["grouped_examples"]:
        print("  grouped (skipped) examples:", flush=True)
        for e in stats["grouped_examples"]:
            print(f"    {e}", flush=True)

    if merged == 0:
        print("\nERROR: nothing merged — not exporting.", flush=True)
        sys.exit(1)
    if stats["max_rel_delta"] == 0.0:
        print("\nERROR: all deltas zero — adapter appears untrained. Not exporting.", flush=True)
        sys.exit(1)

    if not DO_EXPORT:
        print("\nDO_EXPORT is False — stopping after merge.", flush=True)
        return

    print(f"\nConstructing bridge from {HF_MODEL}", flush=True)
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)

    print(f"Exporting to {OUTPUT_DIR}", flush=True)
    bridge.save_hf_pretrained(model, OUTPUT_DIR)

    print(f"\nExport complete: {OUTPUT_DIR}", flush=True)
    print("Verify with a tensor diff against the base model before evaluating.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()