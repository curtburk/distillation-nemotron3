#!/usr/bin/env python3
"""
export_merged_v3.py — Merge LoRA adapters into base weights IN-PLACE, then export to HF.

Why the built-in merge doesn't work:
  Megatron-Bridge's _merge_lora_adapter_weights (model_bridge.py:668) only fires for
  conversion tasks whose param_name ends in ".adapter.linear_out.weight", and derives
  the base weight as ".to_wrap.weight". That naming comes from ParallelLinearAdapter,
  a true wrapper module.

  But Nemotron 3 Nano's LoRA uses mostly SUBCLASS adapters:
      TELinearAdapter(te.Linear)   -- 726 instances
      LinearAdapter(nn.Linear)     -- 780 instances
      ParallelLinearAdapter(Module) --  54 instances
  The subclass adapters keep the base weight as self.weight and hold the low-rank
  factors as self.lora_a / self.lora_b with self.scaling. There is no ".to_wrap"
  anywhere, so no conversion task ever matches and 1506 of 1560 adapters merge to
  nothing. Verified: exported tensors were bit-identical to base (max|delta| == 0).

What this does instead:
  Walk the live model, and for every adapter module fold the adapter into the base
  weight directly:
      W += scaling * (B @ A)
  matching TELinearAdapter.forward:
      lora_res = self.lora_b(self.lora_a(x)) * self.scaling
      return te.Linear.forward(self, x) + lora_res
  Then export normally. The bridge sees already-merged base weights, so whether its
  own merge path fires is irrelevant.

Run:
  docker run --rm --gpus '"device=1"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e HF_HOME=/hfcache \
      --shm-size=16g --net=host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /mnt/bigdata/kd-export/hf_cache:/hfcache \
      -v /mnt/bigdata/kd-export/output:/output \
      -v ~/pmm-demos:/workspace \
      -w /opt/Megatron-Bridge \
      nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
      torchrun --nproc-per-node=1 /workspace/mbridge_run/export_merged_v3.py \
      2>&1 | tee /mnt/bigdata/kd-export/export_v3_merge.log
"""

import sys

sys.path.insert(0, "/workspace/mbridge_run")

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


def merge_adapters_inplace(model) -> dict:
    """Fold every LoRA adapter into its base weight. Returns a stats dict."""
    stats = {
        "subclass_merged": 0,
        "wrapper_merged": 0,
        "skipped_zero_b": 0,
        "skipped_unknown": 0,
        "max_rel_delta": 0.0,
        "examples": [],
    }

    for chunk in model:
        for name, mod in chunk.named_modules():
            cls = type(mod).__name__

            # Subclass adapters: TELinearAdapter(te.Linear), LinearAdapter(nn.Linear)
            if hasattr(mod, "lora_a") and hasattr(mod, "lora_b") and hasattr(mod, "weight"):
                with torch.no_grad():
                    A = mod.lora_a.weight  # [dim, in_features]
                    B = mod.lora_b.weight  # [out_features, dim]
                    scaling = getattr(mod, "scaling", 1.0)

                    if B.abs().max().item() == 0.0:
                        stats["skipped_zero_b"] += 1
                        continue

                    delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling

                    if delta.shape != mod.weight.shape:
                        stats["skipped_unknown"] += 1
                        continue

                    before = mod.weight.detach().float()
                    mod.weight.add_(delta.to(mod.weight.dtype))
                    after = mod.weight.detach().float()

                    rel = (after - before).abs().mean().item() / max(
                        before.abs().mean().item(), 1e-12
                    )
                    stats["max_rel_delta"] = max(stats["max_rel_delta"], rel)
                    stats["subclass_merged"] += 1
                    if len(stats["examples"]) < 5:
                        stats["examples"].append(f"{name} ({cls}) rel={rel:.4%}")
                continue

            # Wrapper adapter: ParallelLinearAdapter, exposes to_wrap + adapter
            if hasattr(mod, "to_wrap") and hasattr(mod, "adapter"):
                with torch.no_grad():
                    ad = mod.adapter
                    A = getattr(ad, "linear_in", None)
                    B = getattr(ad, "linear_out", None)
                    if A is None or B is None:
                        stats["skipped_unknown"] += 1
                        continue
                    Aw = A.weight
                    Bw = B.weight
                    alpha = getattr(ad, "alpha", None)
                    dim = getattr(ad, "dim", None)
                    scaling = (alpha / dim) if (alpha and dim) else getattr(ad, "scaling", 1.0)

                    if Bw.abs().max().item() == 0.0:
                        stats["skipped_zero_b"] += 1
                        continue

                    delta = (Bw.to(torch.float32) @ Aw.to(torch.float32)) * scaling
                    base = mod.to_wrap.weight
                    if delta.shape != base.shape:
                        stats["skipped_unknown"] += 1
                        continue

                    before = base.detach().float()
                    base.add_(delta.to(base.dtype))
                    after = base.detach().float()

                    rel = (after - before).abs().mean().item() / max(
                        before.abs().mean().item(), 1e-12
                    )
                    stats["max_rel_delta"] = max(stats["max_rel_delta"], rel)
                    stats["wrapper_merged"] += 1
                    if len(stats["examples"]) < 5:
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

    # Confirm adapters are present before merging
    m0 = model[0]
    from collections import Counter

    classes = Counter(type(m).__name__ for _, m in m0.named_modules())
    adapter_classes = {k: v for k, v in classes.items() if "Adapter" in k}
    print(f"adapter modules: {adapter_classes}", flush=True)

    if not adapter_classes:
        print("ERROR: no adapter modules found — nothing to merge.", flush=True)
        return

    print("\nMerging adapters in-place...", flush=True)
    stats = merge_adapters_inplace(model)

    print(f"  subclass adapters merged : {stats['subclass_merged']}", flush=True)
    print(f"  wrapper adapters merged  : {stats['wrapper_merged']}", flush=True)
    print(f"  skipped (lora_b all zero): {stats['skipped_zero_b']}", flush=True)
    print(f"  skipped (shape/unknown)  : {stats['skipped_unknown']}", flush=True)
    print(f"  max relative delta       : {stats['max_rel_delta']:.4%}", flush=True)
    for e in stats["examples"]:
        print(f"    {e}", flush=True)

    total_merged = stats["subclass_merged"] + stats["wrapper_merged"]
    if total_merged == 0:
        print("\nERROR: nothing merged. Not exporting.", flush=True)
        return
    if stats["max_rel_delta"] == 0.0:
        print("\nWARNING: all deltas were zero — adapter may be untrained.", flush=True)

    if not DO_EXPORT:
        print("DO_EXPORT is False — stopping after merge.", flush=True)
        return

    print(f"\nConstructing bridge from {HF_MODEL}", flush=True)
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)

    print(f"Exporting to {OUTPUT_DIR}", flush=True)
    bridge.save_hf_pretrained(model, OUTPUT_DIR)

    print(f"Export complete: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()