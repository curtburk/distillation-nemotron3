#!/usr/bin/env python3
"""
export_merged_v5.py — Full LoRA merge (including grouped MoE experts) + HF export.

Background
----------
Megatron-Bridge's built-in merge (_merge_lora_adapter_weights, model_bridge.py:668)
only fires for conversion tasks whose param_name ends in ".adapter.linear_out.weight",
deriving the base as ".to_wrap.weight". Nemotron 3 Nano uses several adapter shapes and
most never match, so save_hf_pretrained() writes base weights unchanged with no warning.
Verified: the first export was bit-identical to base (max|delta| == 0 everywhere).

This script merges adapters into the base weights in-place, then exports.

Three adapter shapes are handled:

  1. Subclass adapters — TELinearAdapter(te.Linear), LinearAdapter(nn.Linear)
     Base weight is self.weight; factors are self.lora_a / self.lora_b; self.scaling.
     forward: te.Linear.forward(self, x) + lora_b(lora_a(x)) * scaling
     merge:   W += scaling * (B @ A)

  2. Wrapper adapters — LoRALinear wrapping a plain linear
     LoRALinear.forward (peft/lora_layers.py:39):
         linear_output, bias, layernorm_output = self.to_wrap(x)
         adapter_output = self.adapter(layernorm_output.contiguous())
         return linear_output + adapter_output, bias
     ParallelLinearAdapter.forward: linear_out(activation(linear_in(x)))
     No scaling at the wrapper; alpha/dim inside the adapter (both 32 here => 1.0).
     merge:   W += (alpha/dim) * (B @ A)

  3. Grouped MoE — LoRALinear wrapping TEColumnParallelGroupedLinear /
     TERowParallelGroupedLinear, which fuse 128 experts into weight0..weight127 and
     expose no .weight attribute (v3 crashed here with AttributeError).
     The adapter has no expert dimension — it sees the module's input and its output is
     added to the module's output as a whole. So the weight-space equivalent is the SAME
     delta added to every expert weight.
     merge:   for i in range(num_gemms): weight_i += (alpha/dim) * (B @ A)

Assumptions are validated PER MODULE rather than globally, because sampling a few
adapters and generalizing already produced one wrong conclusion this session (three
sampled grouped adapters had zero linear_out; the population does not).

A module is skipped, with a counted reason, if:
  - lora_b / linear_out is all zeros (LoRA inits B to zero; untrained => no-op)
  - the adapter activation is not Identity (then it isn't linear and can't be folded)
  - the computed delta shape doesn't match the target weight

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
      torchrun --nproc-per-node=1 /workspace/mbridge_run/export_merged_v5.py \
      2>&1 | tee /mnt/bigdata/kd-export/export_v5.log
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
OUTPUT_DIR = "/output/nemotron3-nano-kd-full"

SEQ_LENGTH = 4096
PACKED_SEQUENCE = False

# Set False to validate the merge without spending ~10 min on the export
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


def _adapter_scaling(ad) -> float:
    """ParallelLinearAdapter sets self.alpha = dim when alpha is None."""
    alpha = getattr(ad, "alpha", None)
    dim = getattr(ad, "dim", None)
    if alpha is not None and dim:
        return float(alpha) / float(dim)
    return float(getattr(ad, "scaling", 1.0))


def _is_identity_activation(ad) -> bool:
    act = getattr(ad, "activation", None)
    if act is None:
        return True
    return type(act).__name__.lower() in ("identity", "nn.identity")


def _grouped_weight_names(inner) -> list[str]:
    """weight0, weight1, ... on a fused grouped linear, in index order."""
    names = []
    for n, _ in inner.named_parameters(recurse=False):
        if n.startswith("weight") and n[len("weight"):].isdigit():
            names.append(n)
    names.sort(key=lambda s: int(s[len("weight"):]))
    return names


def merge_adapters_inplace(model) -> dict:
    st = {
        "subclass_merged": 0,
        "wrapper_merged": 0,
        "grouped_modules_merged": 0,
        "grouped_expert_weights_merged": 0,
        "skip_zero_b": 0,
        "skip_activation": 0,
        "skip_shape": 0,
        "skip_no_factors": 0,
        "max_rel": 0.0,
        "sum_rel": 0.0,
        "n_rel": 0,
        "examples": [],
        "grouped_examples": [],
        "skip_examples": [],
    }

    def record(name, cls, rel):
        st["max_rel"] = max(st["max_rel"], rel)
        st["sum_rel"] += rel
        st["n_rel"] += 1
        if len(st["examples"]) < 6:
            st["examples"].append(f"{name} ({cls}) rel={rel:.4%}")

    for chunk in model:
        for name, mod in chunk.named_modules():
            cls = type(mod).__name__

            # ---------- 1. subclass adapters ----------
            if hasattr(mod, "lora_a") and hasattr(mod, "lora_b") and hasattr(mod, "weight"):
                with torch.no_grad():
                    A = mod.lora_a.weight
                    B = mod.lora_b.weight
                    scaling = float(getattr(mod, "scaling", 1.0))

                    if B.abs().max().item() == 0.0:
                        st["skip_zero_b"] += 1
                        continue

                    delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling
                    if delta.shape != mod.weight.shape:
                        st["skip_shape"] += 1
                        continue

                    before = mod.weight.detach().float().clone()
                    mod.weight.add_(delta.to(mod.weight.dtype))
                    record(name, cls, _rel_delta(before, mod.weight.detach().float()))
                    st["subclass_merged"] += 1
                continue

            # ---------- wrapper adapters (plain and grouped) ----------
            if hasattr(mod, "to_wrap") and hasattr(mod, "adapter"):
                inner = mod.to_wrap
                ad = mod.adapter

                lin_in = getattr(ad, "linear_in", None)
                lin_out = getattr(ad, "linear_out", None)
                if lin_in is None or lin_out is None:
                    st["skip_no_factors"] += 1
                    if len(st["skip_examples"]) < 4:
                        st["skip_examples"].append(f"{name}: adapter has no linear_in/out")
                    continue

                if not _is_identity_activation(ad):
                    # non-linear adapter path cannot be folded into a weight
                    st["skip_activation"] += 1
                    if len(st["skip_examples"]) < 4:
                        st["skip_examples"].append(
                            f"{name}: activation={type(ad.activation).__name__}"
                        )
                    continue

                with torch.no_grad():
                    A = lin_in.weight
                    B = lin_out.weight
                    if B.abs().max().item() == 0.0:
                        st["skip_zero_b"] += 1
                        continue

                    scaling = _adapter_scaling(ad)
                    delta = (B.to(torch.float32) @ A.to(torch.float32)) * scaling

                    # ---------- 2. plain wrapper ----------
                    if hasattr(inner, "weight"):
                        base = inner.weight
                        if delta.shape != base.shape:
                            st["skip_shape"] += 1
                            continue
                        before = base.detach().float().clone()
                        base.add_(delta.to(base.dtype))
                        record(name, cls, _rel_delta(before, base.detach().float()))
                        st["wrapper_merged"] += 1
                        continue

                    # ---------- 3. grouped MoE ----------
                    wnames = _grouped_weight_names(inner)
                    if not wnames:
                        st["skip_no_factors"] += 1
                        if len(st["skip_examples"]) < 4:
                            st["skip_examples"].append(
                                f"{name}: {type(inner).__name__} has no weightN params"
                            )
                        continue

                    first = getattr(inner, wnames[0])
                    if delta.shape != first.shape:
                        st["skip_shape"] += 1
                        if len(st["skip_examples"]) < 4:
                            st["skip_examples"].append(
                                f"{name}: delta {tuple(delta.shape)} vs "
                                f"{wnames[0]} {tuple(first.shape)}"
                            )
                        continue

                    n_done = 0
                    rel_sum = 0.0
                    for wn in wnames:
                        w = getattr(inner, wn)
                        if w.shape != delta.shape:
                            continue
                        before = w.detach().float().clone()
                        w.add_(delta.to(w.dtype))
                        rel_sum += _rel_delta(before, w.detach().float())
                        n_done += 1

                    if n_done == 0:
                        st["skip_shape"] += 1
                        continue

                    mean_rel = rel_sum / n_done
                    record(f"{name} [{n_done} experts]", type(inner).__name__, mean_rel)
                    st["grouped_modules_merged"] += 1
                    st["grouped_expert_weights_merged"] += n_done
                    if len(st["grouped_examples"]) < 4:
                        st["grouped_examples"].append(
                            f"{name} -> {type(inner).__name__}: "
                            f"{n_done}/{len(wnames)} experts, mean rel={mean_rel:.4%}"
                        )

    return st


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
    adapter_classes = {k: v for k, v in classes.items() if "Adapter" in k or "LoRA" in k}
    print(f"adapter/wrapper modules: {adapter_classes}", flush=True)

    if not adapter_classes:
        print("ERROR: no adapter modules found — nothing to merge.", flush=True)
        sys.exit(1)

    print("\nMerging adapters in-place...", flush=True)
    st = merge_adapters_inplace(model)

    merged_modules = (
        st["subclass_merged"] + st["wrapper_merged"] + st["grouped_modules_merged"]
    )

    print(f"  subclass adapters merged      : {st['subclass_merged']}", flush=True)
    print(f"  wrapper adapters merged       : {st['wrapper_merged']}", flush=True)
    print(f"  grouped MoE modules merged    : {st['grouped_modules_merged']}", flush=True)
    print(f"  grouped expert weights updated: {st['grouped_expert_weights_merged']}", flush=True)
    print(f"  skipped (lora_b all zero)     : {st['skip_zero_b']}", flush=True)
    print(f"  skipped (non-identity act)    : {st['skip_activation']}", flush=True)
    print(f"  skipped (shape mismatch)      : {st['skip_shape']}", flush=True)
    print(f"  skipped (no factors)          : {st['skip_no_factors']}", flush=True)
    print(f"  max relative delta            : {st['max_rel']:.4%}", flush=True)
    if st["n_rel"]:
        print(f"  mean relative delta           : {st['sum_rel'] / st['n_rel']:.4%}", flush=True)

    if st["examples"]:
        print("  merged examples:", flush=True)
        for e in st["examples"]:
            print(f"    {e}", flush=True)
    if st["grouped_examples"]:
        print("  grouped examples:", flush=True)
        for e in st["grouped_examples"]:
            print(f"    {e}", flush=True)
    if st["skip_examples"]:
        print("  skip examples:", flush=True)
        for e in st["skip_examples"]:
            print(f"    {e}", flush=True)

    if merged_modules == 0:
        print("\nERROR: nothing merged — not exporting.", flush=True)
        sys.exit(1)
    if st["max_rel"] == 0.0:
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
    print("Verify routed-expert tensors changed before evaluating:", flush=True)
    print("  experts.0.up_proj.weight should now differ from base.", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()