"""
Configuration for ZGX Fury Knowledge Distillation — Corpus Construction Only.

Teacher:  DeepSeek-V4-Flash-DSpark (FP4+FP8 mixed, ~165 GB, HBM-resident)
Student:  NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (post-distillation training target)

This config covers ONLY corpus construction (steps 01-06). It produces an
SFT-ready dataset. Training and deployment are separate concerns handled by
Torchtune/NeMo downstream.
"""

import os
from pathlib import Path

# ==============================================================================
# Paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                    # Downloaded benchmark datasets
SEEDS_DIR = DATA_DIR / "seeds"                # Generated seed problems
TRACES_DIR = DATA_DIR / "traces"              # Teacher-generated reasoning traces
FILTERED_DIR = DATA_DIR / "filtered"          # Execution-filtered traces
SCORED_DIR = DATA_DIR / "scored"              # Reasoning-quality-scored traces
FINAL_DIR = DATA_DIR / "final"                # Assembled SFT dataset
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

for d in [DATA_DIR, RAW_DIR, SEEDS_DIR, TRACES_DIR, FILTERED_DIR,
          SCORED_DIR, FINAL_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Teacher — DeepSeek V4 Flash
# ==============================================================================

TEACHER_MODEL = "nvidia/DeepSeek-V4-Flash-NVFP4"
TEACHER_PORT = 8091
VLLM_BASE_URL = f"http://localhost:{TEACHER_PORT}/v1"

# vLLM launch env vars for FP4 MoE kernels
TEACHER_ENV = {
    # DSpark speculative decoding (7 tokens, greedy draft) — see model card
    # Only enable once base model is verified stable at TP=1 on Blackwell Ultra.
}

# Extra vLLM args beyond the base command
TEACHER_VLLM_ARGS = [
    "--trust-remote-code",
    "--kv-cache-dtype", "fp8",
    "--block-size", "256",
    "--tensor-parallel-size", "1",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.85",
    # DSpark spec decode — comment out for initial smoke test
    # "--speculative-config", '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}',
]

# ==============================================================================
# Sampling
# ==============================================================================

# Per DeepSeek V4 model card
DEFAULT_SAMPLING = {"temperature": 1.0, "top_p": 1.0}

# For seed problem generation — slightly lower temp for structural consistency
SEED_GENERATION = {**DEFAULT_SAMPLING, "max_tokens": 2048, "temperature": 0.8}

# For reasoning trace generation — full recommended params
TRACE_GENERATION = {**DEFAULT_SAMPLING, "max_tokens": 6144}

# For quality scoring — low temp for consistency (override recommended params)
QUALITY_SCORING = {"temperature": 0.1, "top_p": 0.9, "max_tokens": 256}

# Number of candidate traces per problem
TRACES_PER_PROBLEM = 3

# ==============================================================================
# Execution Sandbox
# ==============================================================================

SANDBOX_IMAGE = "fury-kd-sandbox:latest"
SANDBOX_TIMEOUT_SECONDS = 30
SANDBOX_MEMORY_LIMIT = "512m"
SANDBOX_CPU_LIMIT = "2.0"

# ==============================================================================
# Quality Filtering
# ==============================================================================

QUALITY_WEIGHTS = {
    "analysis": 0.25,
    "approach_justification": 0.30,
    "implementation_clarity": 0.25,
    "verification_thoroughness": 0.20,
}
QUALITY_THRESHOLD_KEEP = 3.5
QUALITY_THRESHOLD_REVISE = 3.0

# ==============================================================================
# Dataset Assembly
# ==============================================================================

# Splits must sum to 1.0. Rounding remainder goes to train.
DATASET_SPLITS = {
    "train": 0.90,
    "validation": 0.05,
    "test": 0.05,
}

# System prompt baked into every SFT example. Matches Nemotron 3 Nano's
# native reasoning behavior — student already knows how to emit <think> blocks,
# so the SFT data should preserve that structure from the DeepSeek teacher.
SYSTEM_PROMPT = (
    "You are an expert software engineer. Think step by step before writing code. "
    "For each problem: analyze the requirements, choose an approach with justification, "
    "implement a clean solution, and verify against the test cases."
)

# ==============================================================================
# Concurrency
# ==============================================================================

MAX_CONCURRENT_REQUESTS = 12      # vLLM to teacher
MAX_CONCURRENT_SANDBOXES = 8     # Docker execution

# ==============================================================================
# HTTP / Logging
# ==============================================================================

HTTPX_TIMEOUT = 600              # 10 min — long reasoning traces can be slow
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# ==============================================================================
# Memory Budget (for reference / start.sh validation)
# ==============================================================================

FURY_HBM_GB = 252
FURY_TOTAL_GB = 748
TEACHER_MEMORY_GB = 165          # DeepSeek V4 Flash FP4+FP8
KV_CACHE_ESTIMATE_GB = 25        # At max_model_len=8192
FRAMEWORK_OVERHEAD_GB = 15

def hbm_budget() -> dict:
    used = TEACHER_MEMORY_GB + KV_CACHE_ESTIMATE_GB + FRAMEWORK_OVERHEAD_GB
    return {
        "teacher_gb": TEACHER_MEMORY_GB,
        "kv_cache_gb": KV_CACHE_ESTIMATE_GB,
        "overhead_gb": FRAMEWORK_OVERHEAD_GB,
        "total_used_gb": used,
        "hbm_gb": FURY_HBM_GB,
        "hbm_headroom_gb": FURY_HBM_GB - used,
        "fits_hbm": used <= FURY_HBM_GB,
    }
