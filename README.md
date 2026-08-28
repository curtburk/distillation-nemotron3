# distillation-nemotron3

Knowledge distillation of NVIDIA Nemotron 3 Nano 30B-A3B from DeepSeek V4 Flash 284B, run end to end on a single HP ZGX Fury (NVIDIA GB300 Grace Blackwell Ultra, sm_103).

Everything here runs on-premises. The teacher, the student, the corpus, and the training all fit on one machine, and no data leaves the box.

## Result

Held-out test split of 213 problems (182 APPS, 8 HumanEval+), none of them seen during student training. Greedy decoding.

| | stock | **distilled** | teacher |
|---|---:|---:|---:|
| pass@1 | 23.0% | **63.4%** | 68.5% |
| APPS (n=182) | 22.5% | 69.8% | 75.8% |
| hard problems (n=26) | 3.8% | 65.4% | 73.1% |
| easy problems (n=66) | 27.3% | 56.1% | 56.1% |
| responses that terminated | 34.3% | 93.0% | 100.0% |
| mean completion tokens | 9,736 | 3,125 | 875 |
| responses with no code block | 138 | 11 | 8 |

Two ways to read this. Against its own baseline the student gained 40 points. Against the teacher it retained **92.5% of a model 9.5 times its size** - and on easy problems it matched the teacher exactly.

The gain comes from finishing, not from better algorithms. Stock Nemotron 3 Nano failed 138 of 213 problems by never producing code at all: it reasoned past the token budget and got cut off mid-thought, usually with a correct analysis and no solution. The teacher terminates every time, in 875 tokens. Distillation moved the student most of the way there — 93% termination at a third of stock's length - without fully closing the gap. The effect is largest on hard problems, which is where the base model spirals longest.

Caveats worth stating plainly. This is one training run with one seed and one checkpoint. The teacher generated traces for these same problems during corpus construction, so the split is held out from the student but not from the teacher, and its score should be read as a contaminated ceiling; it also ran at a smaller token budget and on a different vLLM build. Evaluation uses greedy decoding while the corpus was generated at temperature 1.0.

MBPP is excluded from the table above. It scores 1/23 for every model tested including the 284B teacher, which is a harness bug rather than a model property - running a known-strong reference model through the same pipeline is what surfaced it.

## Layout

```
knowledge-distillation/            corpus construction pipeline
  scripts/                         six numbered stages, 01 through 06
  prompts/                         teacher system prompts
  sandbox/                         isolated test execution
  data/final/                      3,872 train / 215 val / 215 test
  Dockerfile.sandbox
  config.py
  start.sh

knowledge-distillation-training/   training and evaluation
  scripts/                         TRL attempt, kept for reference
  eval/                            generation and scoring harness
  lessons-learned.md               full writeup of what worked and what didn't

mbridge_run/                       Megatron-Bridge training and export
  finetune_kd.py                   custom launcher, swaps SQuAD for the KD corpus
  export_merged_v5.py              in-place LoRA merge plus HF export
  inspect_grouped.py               structural inspection of grouped MoE adapters
```

Model weights, checkpoints, and intermediate corpus stages are excluded from the repo. All of them regenerate from the pipeline.

## Corpus construction

DeepSeek V4 Flash NVFP4 serves as teacher on port 8091 through vLLM. The pipeline pulls problems from HumanEval+, MBPP, and APPS, generates three candidate reasoning traces per problem, executes the extracted code against the real test suites in a network-isolated sandbox, scores the surviving traces for reasoning quality, and keeps the best trace per problem.

```bash
cd knowledge-distillation
./start.sh start          # bring up the teacher
./start.sh pipeline       # run all six stages
./start.sh status         # progress across every stage
```

The full run took about 15 hours and produced 17,973 traces, of which 4,302 survived execution filtering and quality scoring. Average quality score 4.57 out of 5. Execution pass rate on teacher-generated code was 92.7%.

Use `CANARY=100` to cap the run at 100 problems for a twelve minute end-to-end validation before committing to the full thing.

## Training

Megatron-Bridge inside `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano`. LoRA at rank 32, alpha 32, learning rate 1e-4, 1,500 iterations at 4,096 sequence length. About two hours at 5.2 seconds per step. Loss fell from 0.73 to 0.46 with no NaN iterations.

Convert the base model first:

```bash
python scripts/import_hf_ckpt.py \
    --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --output-path /workspace/megatron_ckpt \
    --trust-remote-code
```

Then train, with three overrides that single-GPU runs require:

```bash
torchrun --nproc-per-node=1 /workspace/mbridge_run/finetune_kd.py \
    --peft lora --seq-length 4096 \
    checkpoint.pretrained_checkpoint=/workspace/megatron_ckpt \
    checkpoint.save=/workspace/mbridge_run/checkpoints \
    model.expert_model_parallel_size=1 \
    model.moe_enable_deepep=false \
    model.moe_token_dispatcher_type=alltoall \
    train.global_batch_size=8 train.micro_batch_size=1 \
    train.train_iters=1500 scheduler.lr_warmup_iters=75
```

The recipe defaults assume eight GB200s. Without those overrides you get an assertion about world size, then a DeepEP dispatch buffer error, since DeepEP has no configuration for a single expert-parallel rank.

Set `checkpoint.save` to a mounted path. The default writes inside the container, and with `--rm` the checkpoints vanish the moment training ends.

## Exporting

`save_hf_pretrained` claims to merge LoRA adapters automatically. For this model it does not, and it fails silently. The merge only fires for parameter names ending in `.adapter.linear_out.weight`, which is the naming used by wrapper-style adapters. Nemotron 3 Nano mostly gets subclass adapters that inherit from the linear layer and keep the base weight as `self.weight`, so nothing matches and the export writes the base model back out unchanged. It still prints a success message.

The first export was byte-for-byte identical to the base model. Every evaluation run against it was comparing stock to stock.

`export_merged_v5.py` folds the adapters in directly before exporting, handling all three adapter shapes including the grouped MoE experts, and refuses to export if nothing merged or if every delta came out zero.

```bash
torchrun --nproc-per-node=1 /workspace/mbridge_run/export_merged_v5.py
```

**Always diff the exported weights against the base model before you evaluate anything.** A broken export produces a null result that looks completely reasonable.

```python
from safetensors import safe_open
with safe_open(exported_shard, "pt") as f: a = f.get_tensor(key).float()
with safe_open(base_shard, "pt") as f:     b = f.get_tensor(key).float()
print((a - b).abs().max())   # zero means the merge did nothing
```

## Evaluation

Serve one model at a time, generate, then score. Never run two vLLM instances on the same GPU.

```bash
docker run -d --rm --name vllm-eval --gpus '"device=1"' \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --shm-size=16g --ipc=host -p 8091:8000 \
    -v /path/to/exports:/models \
    vllm/vllm-openai:v0.24.0-ubuntu2404 \
    --model /models/nemotron3-nano-kd-merged --served-model-name merged \
    --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.85

until curl -s http://localhost:8091/v1/models >/dev/null 2>&1; do sleep 5; done

cd knowledge-distillation-training/eval
python3 eval_holdout_generate.py --model-name merged213 --port 8091 --served-model merged
docker stop vllm-eval

python3 eval_compare3.py stock213 merged213 full213
```

Wait for the server before generating. `docker run -d` returns immediately and the port opens well before the model has loaded.

Two things the holdout harness has to handle that a HumanEval-only harness does not. Test cases live in `data/raw/*.jsonl` and have to be joined by `problem_id`, because the final split carries only conversations and metadata. And APPS problems use stdin/stdout pairs rather than assertions, so the sandbox switches modes on the test file extension.

Report termination rate alongside pass@1. For this model they measure nearly the same thing, and pass@1 on its own hides why the number moved.

Running the teacher through the same harness is worth the extra hour. It turns a delta into a retention figure, and it is what caught the MBPP bug — a 284B model scoring 4.3% is a harness problem, not a model problem.

```bash
docker run -d --rm --name vllm-teacher --gpus '"device=1"' \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
    --shm-size=16g --ipc=host -p 8091:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:nightly \
    --model nvidia/DeepSeek-V4-Flash-NVFP4 \
    --trust-remote-code --kv-cache-dtype fp8 --block-size 256 \
    --tensor-parallel-size 1 --max-model-len 12288 \
    --gpu-memory-utilization 0.95 --reasoning-parser deepseek_v3
```

Three details this needs. Utilization has to be 0.95, not the 0.85 used for the student — 157 GB of weights against a 215 GB budget leaves negative KV cache. The image has to be `nightly`; `v0.24.0` loads the model and then dies in kernel warmup on a `tilelang`/`flashinfer` symbol conflict that does not fire for Nemotron 3 Nano. And `MAX_TOKENS` in the generation script has to drop to 10,240, since a request asking for the full 12,288 leaves no room for the prompt.

## Notes

`lessons-learned.md` in `knowledge-distillation-training/` has the full account, including the TRL path that ran 25 times slower, the silent merge failure and how it was caught, and a secondary result where merging the routed MoE expert adapters raised format adherence tenfold and cost eight points of pass@1.

## Environment

- HP ZGX Fury, NVIDIA GB300 Grace Blackwell Ultra, 268 GB HBM3e, sm_103
- Driver 595.71.05, CUDA 13.2, Ubuntu 24, Python 3.12
- `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` for training and export
- `vllm/vllm-openai:v0.24.0-ubuntu2404` for serving
- Teacher: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Student: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
