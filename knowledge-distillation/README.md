# ZGX Fury — Knowledge Distillation Corpus Construction

Builds an SFT-ready dataset by having DeepSeek V4 Flash generate reasoning traces for coding problems, filtering by execution against test suites, and scoring reasoning quality.

**Scope**: corpus construction only. Training (Torchtune/NeMo) and any downstream use are separate concerns.

## Setup

**Teacher**: DeepSeek-V4-Flash-DSpark (284B params / 13B active, FP4+FP8 mixed, ~165 GB)
**Deployment**: single vLLM instance on the ZGX Fury (fits entirely in HBM3e)
**Downstream student** (not covered here): NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

## HBM Budget

| Component | Memory |
|-----------|-------:|
| DeepSeek V4 Flash (FP4+FP8) | ~165 GB |
| KV cache (max_model_len=8192) | ~25 GB |
| Framework overhead | ~15 GB |
| **Total** | **~205 GB** |
| **HBM available** | **252 GB** |
| **Headroom** | **~47 GB** |

Everything HBM-resident. No LPDDR spillover.

## Usage

```bash
./start.sh sandbox            # Build execution sandbox (one-time)
./start.sh teacher            # Launch vLLM (~20 min first time for model download)
./start.sh status             # Verify health + budget
./start.sh pipeline           # Run corpus construction (~5-7 days)
./start.sh stop               # Shut down teacher when done
```

## Pipeline

| Step | Script | Purpose | Est. Time |
|------|--------|---------|-----------|
| 01 | `01_download_datasets.py` | Fetch APPS + MBPP + HumanEval+ | ~30 min |
| 02 | `02_generate_seed_problems.py` | Generate enterprise-realistic problems via teacher | ~12 hrs |
| 03 | `03_generate_traces.py` | Generate 3 reasoning traces per problem | ~72-96 hrs |
| 04 | `04_execute_and_filter.py` | Sandbox execution, keep passing solutions | ~6-8 hrs |
| 05 | `05_score_reasoning.py` | Score reasoning quality of passing traces | ~8-10 hrs |
| 06 | `06_assemble_dataset.py` | Assemble train/validation/test splits | ~2 min |

Step 03 supports resume — if interrupted, rerun to continue from the last completed trace.

## Output

`data/final/`:
- `train.jsonl` — SFT training data (~10K examples in conversation format)
- `validation.jsonl` — for loss monitoring
- `test.jsonl` — held-out for evaluation
- `dataset_stats.json` — split breakdowns by source, difficulty, category
- `torchtune_dataset_config.yaml` — snippet for training config

Each example is a `{"conversations": [system, user, assistant]}` object compatible with Torchtune, LlamaFactory, and Axolotl.

## Project Structure

```
fury-kd/
├── config.py                    # All configuration
├── start.sh                     # Preflight + command dispatcher
├── Dockerfile.sandbox           # Code execution sandbox
├── requirements.txt
├── scripts/                     # Numbered pipeline scripts (run in order)
├── sandbox/run_tests.py         # Runs inside the sandbox container
└── prompts/                     # Teacher system prompts (trace gen, seed gen, scoring)
```
