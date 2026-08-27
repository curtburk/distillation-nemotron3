#!/usr/bin/env python3
"""
export_merged.py — Load Nemotron 3 Nano base + LoRA adapter through Megatron-Bridge,
merge the adapter into the base weights, and export to HuggingFace format.

Why this exists:
  `convert_checkpoints.py export` calls load_megatron_model() on a single checkpoint
  directory. A PEFT run only writes adapter tensors (peft/base.py filters saves to
  params_to_save + ".adapter." keys), so that path fails with:
      KeyError: "decoder.layers.0.mixer.dt_bias from model not in state dict"

  Megatron-Bridge's LoRA merge (_merge_lora_adapter_weights in model_bridge.py) runs
  during weight conversion on a *live* model that has adapters attached. So we rebuild
  the model the same way training does — setup() loads base weights from
  pretrained_checkpoint and adapter weights from checkpoint.load — then hand that model
  to bridge.save_hf_pretrained().

Run inside nvcr.io/nvidia/nemo:25.11.nemotron_3_nano:

  docker run --rm --gpus '"device=1"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
      -e HF_HOME=/hfcache \
      --shm-size=16g --net=host --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /mnt/bigdata/kd-export/hf_cache:/hfcache \
      -v /mnt/bigdata/kd-export/output:/output \
      -v ~/pmm-demos:/workspace \
      -w /opt/Megatron-Bridge \
      nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
      torchrun --nproc-per-node=1 /workspace/mbridge_run/export_merged.py \
      2>&1 | tee /mnt/bigdata/kd-export/export_merged.log
"""

import sys

# finetune_kd.py lives here and provides kd_dataset_config / process_kd_example
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


# ==============================================================================
# Configuration — must match the training run
# ==============================================================================

HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
BASE_CKPT = "/workspace/megatron_ckpt"
ADAPTER_CKPT = "/workspace/mbridge_run/checkpoints"
OUTPUT_DIR = "/output/nemotron3-nano-kd-distilled"

SEQ_LENGTH = 4096
PACKED_SEQUENCE = False


def build_config():
    """Rebuild the exact training config, but with training disabled."""
    cfg = finetune_config(
        seq_length=SEQ_LENGTH,
        peft="lora",
        packed_sequence=PACKED_SEQUENCE,
    )
    cfg.model.seq_length = SEQ_LENGTH

    # Same dataset wiring as training — setup() builds iterators from this
    cfg.dataset = kd_dataset_config(SEQ_LENGTH, PACKED_SEQUENCE)

    # Base weights from the imported HF checkpoint, adapter from the training run
    cfg.checkpoint.pretrained_checkpoint = BASE_CKPT
    cfg.checkpoint.load = ADAPTER_CKPT
    cfg.checkpoint.save = None
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False

    # Single-GPU overrides (recipe defaults target 8x GB200)
    cfg.model.expert_model_parallel_size = 1
    cfg.model.moe_enable_deepep = False
    cfg.model.moe_token_dispatcher_type = "alltoall"

    # Match training batch shape so config validation passes; skip the train loop
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.train_iters = 1500
    cfg.train.skip_train = True
    cfg.train.eval_iters = 0

    return cfg


def main():
    print("Building config...", flush=True)
    cfg = build_config()

    # pretrain() applies this before creating GlobalState; it sets derived fields.
    # Skip quietly if the helper isn't exposed in this version.
    try:
        from megatron.bridge.training.config import runtime_config_update

        runtime_config_update(cfg)
        print("Applied runtime_config_update", flush=True)
    except ImportError:
        try:
            from megatron.bridge.training.pretrain import runtime_config_update

            runtime_config_update(cfg)
            print("Applied runtime_config_update", flush=True)
        except ImportError:
            print("runtime_config_update not found, continuing without it", flush=True)

    state = GlobalState()
    state.cfg = cfg

    print(f"Loading base weights from {BASE_CKPT}", flush=True)
    print(f"Loading adapter weights from {ADAPTER_CKPT}", flush=True)

    dataset_provider = get_dataset_provider(cfg.dataset)
    setup_output = setup(state, dataset_provider)
    model = setup_output.model

    print(f"Model built: {len(model)} chunk(s)", flush=True)

    print(f"Constructing bridge from {HF_MODEL}", flush=True)
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL, trust_remote_code=True)

    print(f"Exporting merged model to {OUTPUT_DIR}", flush=True)
    bridge.save_hf_pretrained(model, OUTPUT_DIR)

    print(f"Export complete: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()