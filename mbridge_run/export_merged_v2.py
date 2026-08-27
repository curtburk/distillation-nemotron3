#!/usr/bin/env python3
"""
export_merged.py — Load base + LoRA adapter via Megatron-Bridge, merge, export to HF.

DIAGNOSTIC VERSION. The previous export produced a byte-identical copy of the base
model (verified: max|delta| == 0 across attention, MoE expert, shared expert, and
Mamba in_proj tensors). So _merge_lora_adapter_weights never fired.

That merge only runs for params whose name ends in ".to_wrap.weight"
(model_bridge.py:673). This version prints what the model built by setup() actually
contains, so we can tell which of these is true:

  A) no "to_wrap" params, but adapter params present
       -> adapters attached under different naming; merge predicate never matches
  B) neither present
       -> setup() returned a plain model; PEFT transform or adapter load didn't
          land on the returned object
  C) both present and non-zero
       -> naming is fine, failure is inside the merge (e.g. megatron_module is None)

It also reports adapter weight magnitudes — if lora_B is still all zeros
(lora_B_init_method: zero) then training never moved it and the merge would be a
no-op even if wired correctly.
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
OUTPUT_DIR = "/output/nemotron3-nano-kd-distilled-v2"

SEQ_LENGTH = 4096
PACKED_SEQUENCE = False

# Set False to run diagnostics only and skip the ~10 min export
DO_EXPORT = False


def build_config():
    cfg = finetune_config(
        seq_length=SEQ_LENGTH,
        peft_scheme="lora",
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


def diagnose(model) -> None:
    """Report whether LoRA wrappers are present and whether adapters are non-zero."""
    print("\n" + "=" * 64, flush=True)
    print("MODEL STRUCTURE DIAGNOSTIC", flush=True)
    print("=" * 64, flush=True)

    n_wrap = 0
    n_adapter = 0
    wrap_names: list[str] = []
    adapter_names: list[str] = []

    for chunk in model:
        for name, param in chunk.named_parameters():
            lowered = name.lower()
            if "to_wrap" in lowered:
                n_wrap += 1
                if len(wrap_names) < 6:
                    wrap_names.append(name)
            if ".adapter." in lowered or "lora" in lowered:
                n_adapter += 1
                if len(adapter_names) < 6:
                    adapter_names.append(name)

    print(f"params containing 'to_wrap'      : {n_wrap}", flush=True)
    print(f"params containing 'adapter'/'lora': {n_adapter}", flush=True)

    if wrap_names:
        print("\nsample 'to_wrap' names:", flush=True)
        for n in wrap_names:
            print(f"   {n}", flush=True)

    if adapter_names:
        print("\nsample adapter names:", flush=True)
        for n in adapter_names:
            print(f"   {n}", flush=True)

    # Are the adapter weights actually non-zero? lora_B starts at zero by default,
    # so an untrained/unloaded adapter merges to nothing even if wired correctly.
    print("\nadapter weight magnitudes (first 8):", flush=True)
    shown = 0
    for chunk in model:
        for name, param in chunk.named_parameters():
            lowered = name.lower()
            if ".adapter." in lowered or "lora" in lowered:
                with torch.no_grad():
                    absmax = param.abs().max().item()
                    absmean = param.abs().mean().item()
                print(
                    f"   {name}  shape={tuple(param.shape)}  "
                    f"max|w|={absmax:.3e}  mean|w|={absmean:.3e}",
                    flush=True,
                )
                shown += 1
                if shown >= 8:
                    break
        if shown >= 8:
            break

    if n_adapter == 0:
        print("\nNo adapter params found. Dumping a sample of all param names:", flush=True)
        count = 0
        for chunk in model:
            for name, _ in chunk.named_parameters():
                print(f"   {name}", flush=True)
                count += 1
                if count >= 25:
                    break
            if count >= 25:
                break

    # Module-level view: the merge checks task.megatron_module, so knowing the
    # wrapper class names matters too.
    print("\nmodule classes containing 'lora'/'adapter' (first 8):", flush=True)
    shown = 0
    for chunk in model:
        for name, mod in chunk.named_modules():
            cls = type(mod).__name__.lower()
            if "lora" in cls or "adapter" in cls:
                print(f"   {name}  -> {type(mod).__name__}", flush=True)
                shown += 1
                if shown >= 8:
                    break
        if shown >= 8:
            break
    if shown == 0:
        print("   (none found)", flush=True)

    print("=" * 64 + "\n", flush=True)


def main() -> None:
    print("Building config...", flush=True)
    cfg = build_config()

    for mod_path in (
        "megatron.bridge.training.config",
        "megatron.bridge.training.pretrain",
    ):
        try:
            mod = __import__(mod_path, fromlist=["runtime_config_update"])
            mod.runtime_config_update(cfg)
            print(f"Applied runtime_config_update from {mod_path}", flush=True)
            break
        except (ImportError, AttributeError):
            continue
    else:
        print("runtime_config_update not found, continuing without it", flush=True)

    state = GlobalState()
    state.cfg = cfg

    print(f"Base weights:    {BASE_CKPT}", flush=True)
    print(f"Adapter weights: {ADAPTER_CKPT}", flush=True)

    dataset_provider = get_dataset_provider(cfg.dataset)
    setup_output = setup(state, dataset_provider)
    model = setup_output.model

    print(f"Model built: {len(model)} chunk(s)", flush=True)

    print("type(model):", type(model), flush=True)
    print("len:", len(model) if hasattr(model,'__len__') else 'n/a', flush=True)
    m0 = model[0]
    print("type(model[0]):", type(m0), flush=True)
    names = [n for n,_ in m0.named_parameters()]
    print("params:", len(names), "| to_wrap:", sum('to_wrap' in n for n in names), flush=True)
    mods = [type(mm).__name__ for _,mm in m0.named_modules()]
    from collections import Counter
    print("module classes:", Counter(mods).most_common(15), flush=True)

    diagnose(model)

    if not DO_EXPORT:
        print("DO_EXPORT is False — stopping after diagnostics.", flush=True)
        return

    print(f"Constructing bridge from {HF_MODEL}", flush=True)
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
