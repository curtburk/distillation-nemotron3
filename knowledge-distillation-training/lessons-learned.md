# Lessons Learned: Distilling Nemotron 3 Nano 30B on ZGX Fury (GB300)

Session: 2026-08-25 → 2026-08-27
Hardware: HP ZGX Fury, NVIDIA GB300 Blackwell Ultra, sm_103, 268 GB HBM3e
Goal: LoRA fine-tune Nemotron 3 Nano 30B-A3B on a 4,302-example knowledge distillation
corpus built from DeepSeek V4 Flash NVFP4, then measure the effect.

---

## Headline Result

Held-out test split, **213 problems** (182 APPS, 23 MBPP, 8 HumanEval+), never seen
during student training. Greedy decoding. Four models: stock Nemotron 3 Nano, partial
merge (104 of 150 adapters), full merge (all 150 including routed MoE experts), and the
DeepSeek V4 Flash 284B teacher.

| metric | stock | **distilled** | full merge | teacher |
|---|---:|---:|---:|---:|
| **pass@1** | 23.0% | **63.4%** | 55.4% | 68.5% |
| APPS (n=182) | 22.5% | **69.8%** | 59.9% | 75.8% |
| HumanEval+ (n=8) | 87.5% | 87.5% | 100.0% | 87.5% |
| MBPP (n=23) | 4.3% | 4.3% | 4.3% | 4.3% |
| easy (n=66) | 27.3% | **56.1%** | 47.0% | 56.1% |
| medium (n=121) | 24.8% | **66.9%** | 63.6% | 74.4% |
| hard (n=26) | 3.8% | **65.4%** | 38.5% | 73.1% |
| terminated | 34.3% | **93.0%** | 90.6% | 100.0% |
| mean tokens | 9,736 | 3,125 | 2,778 | 875 |
| no code block | 138 | **11** | 29 | 8 |
| full format rate | 0.0% | 2.3% | 22.5% | 0.0% |
| avg sections /5 | 0.37 | 0.37 | 1.59 | 0.11 |

**Two ways to state the result:**

- **23.0% → 63.4%**, +40.4 points over the base model.
- **92.5% retention of the teacher** (63.4 / 68.5) at **9.5× fewer parameters**.

Retention by slice: APPS 92%, hard problems 89%, easy problems **100%** (56.1% on both).

Problem-level movement:
```
stock -> distilled  : +90 fixed,  -4 broken,  net +86
stock -> full merge : +75 fixed,  -6 broken,  net +69
stock -> teacher    : +102 fixed, -5 broken,  net +97
distilled -> full   : +21 fixed, -38 broken,  net -17
distilled -> teacher: +32 fixed, -21 broken,  net +11
```

The last line is the tightest evidence for retention: student vs teacher is net +11 out
of 213, while stock vs student is net +86. The student behaves like the teacher; stock
does not.

**Teacher caveats — this is a ceiling, not a controlled comparison.** The teacher
generated traces for all 8,164 corpus problems during construction, including these 213.
The split is held out from the *student*, not the teacher, so its number is partly a
memorization check and flatters the teacher. It also ran at `max_tokens=10240` (its
context window was 12,288) against the students' 12,288, and on `vllm/vllm-openai:nightly`
rather than `v0.24.0`, because v0.24.0 crashed during kernel warmup on this model (see
Part 4). Report retention as approximate.

### The mechanism is termination, not better algorithms

Stock produces **no code at all on 138 of 213 problems** — it reasons past the token cap
and never gets to a solution, which is an automatic fail. Only 34.3% of its responses
terminate normally, at a mean of 9,736 tokens.

The teacher terminates **100%** of the time at **875 mean tokens**. Stock runs 11× longer
and finishes a third as often. The distilled student lands between them: 93.0%
termination at 3,125 tokens.

```
                terminated    mean tokens
stock              34.3%         9,736
distilled          93.0%         3,125
teacher           100.0%           875
```

That is a clean quantitative statement of what transferred. The student moved most of the
way from stock toward the teacher on termination behaviour without matching it — it still
uses 3.6× the teacher's tokens.

The gain is overwhelmingly "learned to stop analyzing and commit to an answer," not
"learned better algorithms." Say it that way — it is more defensible and more interesting
than a vague "improved reasoning" claim.

Biggest single slice: **hard problems, 3.8% → 65.4%** (teacher 73.1%). Stock essentially
never finishes a hard problem.

### Merging the routed MoE experts made it worse

The full merge triples format adherence (0.37 → 1.59 sections, 2.3% → 22.5% full format)
and **costs 8 points of pass@1**. Not noise: 21 fixed vs 38 broken.

The failure mode is specific. Of the full merge's 29 no-code responses, only 13 hit the
token cap — the other **15 terminate normally, having written analysis and approach and
no solution**. Example (APPS_2619, 592 tokens): correct modular-arithmetic insight,
prefix-sum approach, complexity bounds, edge cases, implementation notes, ending with
"Let's write the solution." Then EOS.

The partial merge has no such cases — all 10 of its no-code failures are cap truncation.

**Unverified:** the training corpus contains **0 of 3,872** examples without a code
fence, so the model never saw this pattern. That points at the merge rather than the
data. The grouped-MoE merge assumes adding one low-rank delta to all 128 expert weights
is equivalent to the adapter's runtime behaviour — the adapter sits outside routing and
adds to the module output, so in principle it holds, but experts are selected top-6 per
token and the merged version perturbs experts that were never routed to. This was not
verified. The definitive test is a logit comparison between the un-merged model
(adapters attached, straight out of `setup()`) and the merged export on identical input.

So: **do not present the format-vs-completion tradeoff as settled.** State it as
"merging routed experts increased format adherence ~10× and cost 8 points of pass@1 via
15 problems that stop after planning; this may be an artifact of how a single delta is
distributed across 128 experts rather than a property of the training."

### n=30 gave the opposite answer

At n=30 the full merge looked **better** (+3.3%, 53.3% vs 50.0%). At n=213 it is
**8 points worse**. The n=30 sample also showed the full merge's format gain without
enough problems to expose the completion cost.

Thirty problems was enough to establish the 40-point stock-vs-distilled gap and not
nearly enough to rank two similar models. Sample size the comparison to the effect
you are trying to measure.

### MBPP at 4.3% is a harness bug, confirmed

MBPP scores **1/23 for every model tested, including the 284B teacher.** A frontier model
scoring 4.3% on MBPP while scoring 87.5% on HumanEval+ is not a model property. Something
in the MBPP path — test format, code extraction, or the sandbox invocation — is broken.

This is exactly the value of running a known-strong reference model through the same
harness: it separates "the model is bad at this" from "the harness is bad at this."
Without the teacher run, the 4.3% would have looked like a real limitation.

**Fix it or drop MBPP from any published numbers.** Do not report it as-is.

### Other caveats

- Evaluation is greedy (`temperature=0.0`); the corpus was generated at
  `temperature=1.0, top_p=1.0`. Different regime from the one the training data came
  from.
- Single training run, single seed, one checkpoint (iter 1400 of 1500).
- Teacher ran at a smaller token budget and on a different vLLM build (see above).

---

## Part 0: The Bug That Nearly Produced a False Null Result

**The first two exports were byte-identical to the base model.** Every measurement taken
against them was stock-vs-stock.

Symptoms that looked like a real (null) result:
- HumanEval+ pass@1 87.2% vs 87.8% — "no effect, within noise"
- Mean completion length within 1%
- Format adherence unchanged
- 7 of 164 problems flipped outcome — "decision-boundary noise"

The tell was **identical mean token counts to the digit (10,391 both)** across 30 long
generations. Two different models do not produce that. Checking directly:

```python
d = (exported_tensor - base_tensor).abs()
print(d.max())   # 0.0 across attention, MoE, shared expert, and Mamba in_proj
```

**Rule: diff exported weights against the base model before running any evaluation.**
A broken export produces a completely plausible null result, and every downstream number
is void. This cost several hours of analysis on meaningless data.

---

## Part 1: Why the Built-In LoRA Merge Silently Does Nothing

`AutoBridge.save_hf_pretrained()` claims "If the model contains LoRA adapters, they will
be automatically merged." For Nemotron 3 Nano, it doesn't.

`_merge_lora_adapter_weights` (`model_bridge.py:668`):

```python
if not task.param_name.endswith(".adapter.linear_out.weight"):
    return converted_weights_dict
base_param_name = task.param_name.replace(".adapter.linear_out.weight", ".to_wrap.weight")
```

That naming belongs to `ParallelLinearAdapter`, a true wrapper module. But Megatron-Bridge
applies **three different adapter shapes** depending on the layer:

| class | file | pattern | base weight |
|---|---|---|---|
| `TELinearAdapter(te.Linear)` | `peft/lora_layers.py:61` | subclass | `self.weight` |
| `LinearAdapter(nn.Linear)` | `peft/lora_layers.py:194` | subclass | `self.weight` |
| `ParallelLinearAdapter(nn.Module)` | `peft/utils.py:345` | wrapper | `self.to_wrap.weight` |

Subclass adapters have **no `.to_wrap` anywhere** — they inherit from the linear layer and
keep the base weight as `self.weight`, with factors in `self.lora_a` / `self.lora_b` and a
`self.scaling` attribute. No conversion task ever matches the predicate, so those adapters
merge to nothing and the export writes base weights unchanged.

There is no error, no warning. The export prints
`Success: All tensors from the original checkpoint were written.`

### The workaround: merge in-place before export

Mirror `TELinearAdapter.forward`:

```python
lora_res = self.lora_b(self.lora_a(x)) * self.scaling
return te.Linear.forward(self, x) + lora_res
```

so the merged weight is `W += scaling * (B @ A)`:

```python
for chunk in model:
    for name, mod in chunk.named_modules():
        # subclass adapters
        if hasattr(mod, "lora_a") and hasattr(mod, "lora_b") and hasattr(mod, "weight"):
            A = mod.lora_a.weight            # [dim, in_features]
            B = mod.lora_b.weight            # [out_features, dim]
            scaling = float(getattr(mod, "scaling", 1.0))
            if B.abs().max().item() == 0.0:  # lora_B inits to zero; untrained == no-op
                continue
            delta = (B.float() @ A.float()) * scaling
            mod.weight.add_(delta.to(mod.weight.dtype))

        # wrapper adapters
        elif hasattr(mod, "to_wrap") and hasattr(mod, "adapter"):
            inner = mod.to_wrap
            if not hasattr(inner, "weight"):
                continue                      # grouped MoE — see below
            ad = mod.adapter
            A, B = ad.linear_in.weight, ad.linear_out.weight
            scaling = float(ad.alpha) / float(ad.dim)
            if B.abs().max().item() == 0.0:
                continue
            inner.weight.add_(((B.float() @ A.float()) * scaling).to(inner.weight.dtype))
```

Then call `bridge.save_hf_pretrained(model, out_dir)` normally — it sees already-merged
base weights and the broken merge path is irrelevant.

Observed on this model: **104 adapters merged**, relative weight deltas 1.6–3.6%
(mean 2.3%), across attention (`q/k/v/o_proj`), Mamba (`in_proj`, `out_proj`), and MoE
shared experts (`linear_fc1`, `linear_fc2`).

### The third case: grouped MoE experts

46 adapters wrap `TEColumnParallelGroupedLinear` / `TERowParallelGroupedLinear` — 128
experts fused into one module holding `weight0`..`weight127`, with **no `.weight`
attribute**. Naively reading `mod.to_wrap.weight` raises:

```
AttributeError: 'TEColumnParallelGroupedLinear' object has no attribute 'weight'.
                Did you mean: 'weight0'?
```

These adapters **are trained** (`linear_out` max|w| ≈ 1.3e-02). Shapes: `linear_in`
(32, 2048), `linear_out` (1024, 32), `dim=32`, `alpha=32`, against `weight0` of shape
(1024, 2048).

`ParallelLinearAdapter.forward` is `linear_out(activation(linear_in(x)))` with
`activation="identity"` by default — no expert dimension, no reshape, no split.
`LoRALinear.forward` (`peft/lora_layers.py:39`) is:

```python
linear_output, bias, layernorm_output = self.to_wrap(x)
adapter_output = self.adapter(layernorm_output.contiguous())
return linear_output + adapter_output, bias
```

Plain addition, no scaling at the wrapper; `alpha` defaults to `dim`, so `alpha/dim`
is 1.0 here. The adapter has no knowledge that experts exist, so the weight-space
equivalent is the **same delta added to every expert weight**:

```python
delta = (B.float() @ A.float()) * (alpha / dim)
for i in range(num_gemms):
    getattr(inner, f"weight{i}").add_(delta.to(...))
```

Merging this way updated 46 modules / 5,888 expert weights, 128/128 experts each,
relative deltas 2.9–3.9%, zero skips.

**But the resulting model scored 8 points worse than the partial merge** (see Headline
Result). The equivalence assumption above is unverified — experts are routed top-6 per
token, and the merge perturbs experts that were never selected. Verify with a logit
comparison against the un-merged model before trusting a full merge.

### The exported `config.json` is not valid JSON

Megatron-Bridge exports (via transformers 4.57) write a `time_step_limit` key whose value
contains **`Infinity`**. Python's `json.load` accepts that as an extension, so local
validation passes — but strict JSON parsers reject it. HuggingFace shows:

```
Configuration Parsing Warning: Invalid JSON for config file config.json
```

and falls back to a generic usage snippet **without `trust_remote_code=True`**, which
fails for anyone who copies it. Inference Provider support is also disabled.

The base model has no such key. Removing it fixes the warning:

```python
c = json.load(open(path))
c.pop("time_step_limit", None)
json.dump(c, open(path, "w"), indent=2)
```

Plain `json.load` will not catch this. Validate strictly:

```python
json.load(f, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
```

Also worth checking on export: transformers 4.57 writes **`dtype`** where 4.55 wrote
**`torch_dtype`**. Diff your exported config against the base model's before publishing:

```python
a, b = json.load(open(mine)), json.load(open(base))
print("only in mine:", sorted(set(a) - set(b)))
print("only in base:", sorted(set(b) - set(a)))
```

---

## Part 2: The Working Megatron-Bridge Pipeline

Container: `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` (56.6 GB)
Contains NeMo 2.5.3, Megatron-Bridge at `/opt/Megatron-Bridge`, Megatron-Core at
`/opt/megatron-lm`, TE 2.7.0, CUDA 13.0, Python 3.12.

### Step 1 — HF → Megatron

```bash
docker run --rm --gpus '"device=1"' -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --shm-size=16g --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/pmm-demos:/workspace \
    -w /opt/Megatron-Bridge \
    nvcr.io/nvidia/nemo:25.11.nemotron_3_nano \
    python scripts/import_hf_ckpt.py \
        --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --output-path /workspace/megatron_ckpt \
        --trust-remote-code
```

Flags are `--model-id` / `--output-path` / `--trust-remote-code`, not the
`--hf-model` / `--output-dir` shown in some docs. Runs on CPU, ~2 min, writes 59 GB.

**Exits with a non-fatal traceback** after the checkpoint is fully written:
```
AttributeError: 'NoneType' object has no attribute 'save_hf_tokenizer_assets'
```
The tokenizer is already saved under `iter_0000000/tokenizer/`. Ignore it.

### Step 2 — Custom launcher for a non-SQuAD dataset

The stock `examples/recipes/nemotron_3/finetune_nemotron_3_nano.py` hardcodes SQuAD
(`dataset=default_squad_config(...)`) and has **no `--per-split-data-args-path` flag** in
this container version. Passing it fails with:

```
hydra.errors.OverrideParseException: LexerNoViableAltException: --per-split-data-args-path
```

`HFDatasetConfig.process_example_fn` is a Python callable, so it cannot come from Hydra
overrides or YAML. **A custom launcher is required.** Build the recipe config, then
replace `cfg.dataset`:

```python
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig, ProcessExampleOutput
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_finetune_config as finetune_config)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

def process_kd_example(example, tokenizer=None) -> ProcessExampleOutput:
    msgs = example["conversations"]
    system = user = assistant = ""
    for m in msgs:
        if m["role"] == "system":      system = m["content"]
        elif m["role"] == "user":      user = m["content"]
        elif m["role"] == "assistant": assistant = m["content"]
    return ProcessExampleOutput(
        input=f"{system}\n\n{user}" if system else user,
        output=assistant,
        original_answers=[assistant],
    )

cfg.dataset = HFDatasetConfig(
    dataset_name="json",                      # HF json loader
    process_example_fn=process_kd_example,
    seq_length=4096, seed=5678,
    dataloader_type="batch", num_workers=1,
    do_validation=True, do_test=False,
    split_val_from_train=False, val_proportion=None,
    dataset_root=CACHE_DIR,                   # somewhere with disk space
    dataset_kwargs={}, packed_sequence_specs=None, rewrite=True,
    hf_kwargs={"data_files": {                # forwarded to load_dataset()
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/validation.jsonl"}},
)
```

Corpus format that works with no conversion and no Parquet step:
```json
{"conversations": [{"role":"system","content":"..."},
                   {"role":"user","content":"..."},
                   {"role":"assistant","content":"..."}],
 "metadata": {...}}
```

### Step 3 — Two overrides required for single GPU

The recipe defaults target 8× GB200. On one GPU:

```
AssertionError: world_size (1) is not divisible by total_model_size (8)
```
→ `model.expert_model_parallel_size=1`

```
AssertionError: Unsupported number of EP ranks: 1   (deep_ep/buffer.py:249)
```
DeepEP's `config_map` only has entries for EP ∈ {2, 4, 8, 16, ...}.
→ `model.moe_enable_deepep=false` **and** `model.moe_token_dispatcher_type=alltoall`

The dispatcher must be switched too — disabling DeepEP alone leaves the `flex` dispatcher
routing through the DeepEP backend.

### Step 4 — Launch

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

**Set `checkpoint.save` to a mounted path.** The default is
`/opt/Megatron-Bridge/nemo_experiments/default/checkpoints` — inside the container. With
`--rm`, everything is deleted the moment training finishes.

Rescue for a run already in flight (`docker cp` does not disturb the process):
```bash
nohup bash -c 'while docker ps -q --filter id=<CID> | grep -q .; do
    docker cp <CID>:/opt/Megatron-Bridge/nemo_experiments/default/checkpoints \
        ~/pmm-demos/mbridge_run/ 2>/dev/null
    sleep 300
done' > ckpt_copy.log 2>&1 &
```
This is what saved the run — but the loop died before the final `iter_1500` save, so only
`iter_1400` survived. Loss at 1400 was 0.481 vs 0.465 at 1500, and LR had decayed to
~1e-6, so the difference is negligible.

### Observed training performance

```
iteration   10/1500 | elapsed 18654.3 ms | lm loss 7.266050E-01 | grad norm 0.401
iteration   20/1500 | elapsed  5199.5 ms | lm loss 6.750866E-01 | grad norm 0.283
iteration 1500/1500 |                    | lm loss 4.65E-01     | grad norm 0.18
```

- **5.2 sec/step** steady state, **~2 hours** total for 1,500 iterations
- Loss 0.73 → 0.46, zero NaN iterations
- 67 GB allocated / 91 GB reserved of 256 GB
- LoRA targets: `[linear_qkv, linear_proj, linear_fc1, linear_fc2, in_proj, out_proj]`,
  dim 32, alpha 32, LR 1e-4

---

## Part 3: The TRL Path That Didn't Work

Before Megatron-Bridge, the same job was attempted with HuggingFace TRL + PEFT.
**~112–135 sec/step (25–47 hour projections) — roughly 25× slower on identical hardware.**

### Attention is the bottleneck

NemotronH supports neither SDPA nor FlashAttention in transformers:
```
ValueError: NemotronHForCausalLM does not support an attention implementation
through torch.nn.functional.scaled_dot_product_attention yet.
```
Forced to `attn_implementation="eager"`. `pip install flash-attn` fails to compile on
sm_103. Unsloth's own Nemotron 3 Nano notebooks also use `eager`, so this is not a local
misconfiguration.

nsys profile, 3-step run:

| layer type | forward + backward |
|---|---:|
| attention (eager, "PageAttention") | 4.7 sec/layer |
| Mamba-2 | 0.7 sec/layer |

Only 6 of 52 layers are attention, but O(n²) at 4096 tokens dominates. ~40% of attention
time was `.copy_` moving materialized 4096×4096 matrices. Throughput ~243 tok/s vs
1,500–3,000 for a healthy H100 LoRA run.

### Sequence-length tuning made it worse

| max_seq_length | steps (packed) | sec/step | projected |
|---|---:|---:|---:|
| 4096 packed | 798 | ~130 | ~28 h |
| 3072 packed | 1,062 | ~130 | ~38 h |
| 4096 unpacked | 1,452 | ~112 | ~45 h |

Lowering sequence length raised step *count* faster than it lowered step *time*. Mamba
and MoE costs scale with length too, not just attention.

### The loss=5.8 mystery, never isolated

Smoke test (2048, unpacked, 10 examples): loss ~1.7. Full runs: 5.8 → 4.9, and 6.5 after
adding LoRA targets. Three candidate causes, all plausible, none isolated before the
pivot:

1. **`assistant_only_loss=True` missing from SFTConfig.** TRL 0.15 does *not* auto-mask
   user/system tokens for conversational data. NVIDIA's SFT docs specify role-based
   masking (system=0, user=0, assistant=1) explicitly.
2. **transformers gradient-accumulation loss reporting.** With
   `gradient_accumulation_steps=8`, reported loss inflates several-fold while training
   proceeds normally.
3. **`tokenizer.padding_side` defaulting to "right".** Right-padding injects pad tokens
   into Mamba's recurrent stream.

Megatron-Bridge handles all three internally and produced loss 0.73 on the same data.

### PEFT target modules for NemotronH (HF naming)

Determined by walking `model.named_modules()`:

```
Linear leaf names present: down_proj, in_proj, k_proj, lm_head,
                           o_proj, out_proj, q_proj, up_proj, v_proj
```

- `q/k/v/o_proj`: 6 each (the 6 attention layers)
- `up_proj`, `down_proj`: 2,967 each (128+1 experts × 23 MoE layers)
- **`gate_proj`: 0 matches** — does not exist, despite appearing in common target lists
  (including Unsloth's, where it is a silent no-op)
- `out_proj` triggers a hard PEFT refusal:
  ```
  ValueError: Module 'out_proj' is incompatible with Mamba-based models
  (model_type='nemotron_h'). Incompatible modules: {'out_proj', 'conv1d'}
  ```

Working list: `["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","in_proj"]`

Megatron-Bridge targets `out_proj` successfully — it uses Megatron-native names and
doesn't go through PEFT's compatibility check.

### Environment that does work on sm_103

```bash
pip install --break-system-packages torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130      # torch 2.13.0+cu130

export TORCH_CUDA_ARCH_LIST="10.3"
pip install --break-system-packages --no-build-isolation causal-conv1d
pip install --break-system-packages --no-build-isolation mamba-ssm
```

`--no-build-isolation` is mandatory — both `setup.py` files import torch at build time and
pip's isolated env doesn't have it.

`torch.cuda.get_arch_list()` shows `['sm_80','sm_90','sm_100','sm_110','sm_120']` — no
sm_103. Everything still runs via PTX JIT: `rmsnorm_fn`, `selective_scan_fn`,
`causal_conv1d_fn`, TE `Linear`, TE `DotProductAttention` all verified working.

### TRL API drift (transformers 5.13.1 / TRL 0.15.2)

- `SFTConfig(max_length=)` → **`max_seq_length=`**
- `completion_only_loss` → removed; use `assistant_only_loss`
- `SFTTrainer(tokenizer=)` → **`processing_class=`**
- `SFTTrainer` accepts `peft_config=` directly
- `from_pretrained(torch_dtype=)` → **`dtype=`**

Validating with `inspect.signature()` and `dataclasses.fields()` costs 30 seconds and
saves a 20-minute model-load cycle per mistake.

---

## Part 4: Evaluation Methodology

### Serving

vLLM `v0.24.0-ubuntu2404` registers `NemotronHForCausalLM` and serves it on sm_103
without issue. One model at a time — never two vLLM instances on the same GPU.

```bash
docker run -d --rm --name vllm-eval --gpus '"device=1"' \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID --shm-size=16g --ipc=host -p 8091:8000 \
    -v /mnt/bigdata/kd-export/output:/models \
    vllm/vllm-openai:v0.24.0-ubuntu2404 \
    --model /models/nemotron3-nano-kd-merged --served-model-name merged \
    --trust-remote-code --max-model-len 16384 --gpu-memory-utilization 0.85

until curl -s http://localhost:8091/v1/models >/dev/null 2>&1; do sleep 5; done
```

**Wait for the server before generating.** `docker run -d` returns immediately; the port
opens before the model finishes loading.

### HumanEval+ was the wrong instrument

Baseline pass@1 was already 87.2% with only 21 failures out of 164, and the teacher
itself only reached 92.7% on the same problems — roughly 5.5 points of headroom. The
corpus is 85% APPS (competition problems), while HumanEval+ is short function
completions. Measuring on a distribution you didn't train for, against a near-saturated
baseline, buys nothing.

The held-out split (`data/final/test.jsonl`, 215 examples, 182 APPS, never trained on) is
the right target.

### Held-out eval needs two things HumanEval+ didn't

**Tests must be joined from `data/raw/*.jsonl` by `problem_id`.** `data/final/*.jsonl`
carries only `conversations` + `metadata` — no `test_code`, no `io_tests`.

**APPS uses stdin/stdout, not assertions.** The sandbox runner switches on file
extension: `.json` second argument → I/O mode, anything else → assertion mode.

```python
if io_tests:
    (tmp/"io_tests.json").write_text(json.dumps(io_tests)); test_name = "io_tests.json"
else:
    (tmp/"tests.py").write_text(test_code); test_name = "tests.py"
```

### max_tokens matters enormously here

At 6,144 tokens (matching `--max-model-len 8192`), **5/5 stock completions hit the cap
with no code block**. Raising to 12,288 (`--max-model-len 16384`) still left stock
truncating on 23/30.

Corpus calibration — APPS assistant turns in the training data:
```
median 4,584 chars (~1,146 tokens) | p90 11,856 | max 25,389
```
So 12,288 is generous relative to what the teacher produced, and not so generous that it
masks a real verbosity difference. Stock's 10,391-token mean is ~9× the teacher's median.

### Report termination separately from pass@1

For this model the two are nearly the same measurement — the gain is almost entirely
"produced a code block at all." Reporting pass@1 alone would obscure the mechanism:

```python
stopped = sum(1 for r in recs if r["finish_reason"] == "stop")
mean_tok = sum(r["completion_tokens"] for r in recs) / len(recs)
```

### Sampling regime differs from corpus construction

The corpus was generated at `temperature=1.0, top_p=1.0` (a DeepSeek V4 requirement).
Evaluation uses greedy (`temperature=0.0`) so the A/B is deterministic. Worth stating in
any writeup — it's not the regime the training data came from.

### Run the teacher through the same harness

Worth the extra hour. It converts a delta into a retention figure, and it caught the MBPP
harness bug — a 284B model scoring 4.3% is a harness problem, not a model problem, and
nothing else in the run would have revealed that.

Serving DeepSeek V4 Flash NVFP4 (157 GB, 46 shards) on one GB300:

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

Three things this took to get right:

**`--gpu-memory-utilization 0.95`, not 0.85.** At 0.85 with `--max-model-len 16384`:
```
Model loading took 156.8203 GiB
Available KV cache memory: -22.05 GiB
ValueError: No available memory for the cache blocks.
```
157 GB of weights against a 215 GB budget leaves nothing. At 0.95 and 12,288 context the
same model gets 79.09 GiB of KV cache and 27× concurrency.

**`vllm/vllm-openai:nightly`, not `v0.24.0`.** v0.24.0 loads the model and allocates KV
cache fine, then dies in kernel warmup:
```
kernel_warmup -> minimax_m3_msa_warmup -> fused_allreduce_gemma_rms_norm
  -> flashinfer.comm -> CudaRTLibrary()
AttributeError: tilelang/lib/libcudart_stub.so: undefined symbol: cudaDeviceReset
```
A `tilelang`/`flashinfer` packaging conflict in that image, unrelated to the model. It
does not fire for Nemotron 3 Nano on the same image. Nightly loads cleanly — expect a few
minutes of TileLang JIT compilation for the MHC kernels.

**`max_tokens` must be below `max_model_len`.** With both at 12,288 every request 400s:
zero prompt headroom. Same failure as corpus construction. Set `MAX_TOKENS = 10240`.

The `CUDACachingAllocator ... OOM ... free: 624427008` warnings during load are normal —
the allocator tries, fails, frees cached blocks, retries. They stop once loading finishes.

---

## Part 5: Operational Notes

**`nohup` without `-u` hides all loss output.** Python buffers stdout when it isn't a
TTY; loss dicts sit in a 64 KB buffer and never appear. A full hour of TRL training ran
with zero visible loss because of this.

**tqdm doesn't survive redirection.** Carriage returns mean `grep "it/s"` matches almost
nothing. Use `cat -v logs/train.log | grep -oE "[0-9]+/[0-9]+ \[[^]]+\]" | tail -3`.

**Stale PID files cause phantom diagnoses.** `train.pid` held a PID from a dead launch
while a *different* PID was the live run. Several `kill $(cat train.pid)` calls silently
did nothing. This produced a false "zombie process" theory and nearly discarded 4.5 hours
of valid progress. Always verify with `ps aux | grep -E "python3.*train" | grep -v grep`
and `nvidia-smi` before and after.

**Don't chain a generation command after `docker stop`.** Ctrl+C on a chained sequence
killed the server, and the subsequent generation wrote a zero-row file that scored as
0.0% across the board — which looked like a catastrophic regression rather than an empty
input. Run server start, generation, and scoring as separate commands.

**CUDA device ordering.** `nvidia-smi` uses PCI bus order (RTX 4000 = 0, GB300 = 1);
PyTorch defaults to "fastest first" (GB300 = 0). Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
everywhere, including `docker run -e`.

**Disk.** Root filesystem hit 100% three times. Consumers: ~703 GB HF cache (DeepSeek V4
Flash 165 GB, Nemotron 3 Super 121 GB, Qwen3-VL 91 GB, Qwen3.6-27B 51 GB, Nemotron 3 Nano
58 GB), ~300 GB Docker images, 59 GB per Megatron checkpoint, 59 GB per HF export.
`/mnt/bigdata` (3.7 TB, shared) is the right home for exports and a redirected
`HF_HOME` — but confirm which caches belong to other people before deleting anything.

---

## Part 6: Process Lessons

**Ask what already worked before designing something new.** Prior successful SFT on a
ZGX Nano GB10 used a NeMo-RL container and finished in 14 hours. That reference never
surfaced until hours into debugging a from-scratch TRL script. When a first-party
framework exists for a first-party model on first-party hardware, start there — "TRL is
simpler for a brand-new architecture" was exactly backwards.

**Read the actual output.** Repeated wrong turns came from skimming rather than reading:
a `TE Attention on GB300: OK` line read as a failure; a `cuTENSOR` error attributed to
the wrong code path and chased for an hour; a `silu2` activation error reported that a
later `grep -rn "silu2" /opt/` proved existed nowhere; file paths asserted
(`nemotronh_provider.py:395`, `finetune_sft_dataset.py:61`) that returned zero `find`
results; a merge predicate quoted as `.to_wrap.weight` when the source says
`.adapter.linear_out.weight`, which sent four subsequent theories in the wrong direction.

**Use an interactive shell instead of chained one-shot inspections.**
```bash
docker run -it --rm --gpus '"device=1"' ... nvcr.io/nvidia/nemo:25.11.nemotron_3_nano bash
```
Then `inspect.getsourcefile()` finds real module paths in one shot — resolving in seconds
what a dozen guessed `find` and `grep` invocations failed to find.

**Smoke test at the production configuration.** The TRL smoke test ran at
`max_seq_length=2048` unpacked while the real run used 4096 packed. It passed and told us
nothing about the configuration actually being run.

**Change one variable at a time.** Multiple rounds applied sequence length, LoRA targets,
and packing changes simultaneously, making attribution impossible.

**Manufactured urgency produces bad decisions.** A self-imposed "LinkedIn deadline" was
repeatedly used to justify staying on the fast-looking-but-wrong path. There was no
deadline. Dropping it led directly to the pivot that worked.

---

## Quick Reference

| Component | Value |
|---|---|
| Container | `nvcr.io/nvidia/nemo:25.11.nemotron_3_nano` |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Architecture | 52 layers: 23 Mamba-2, 23 MoE (128+1 experts, top-6), 6 attention |
| Megatron ckpt | 59 GB, torch_dist format |
| Corpus | 3,872 train / 215 val / 215 test, `conversations` JSONL |
| PEFT | LoRA dim 32, alpha 32, LR 1e-4, cosine |
| Parallelism | TP=1, PP=1, EP=1, CP=1 |
| Sequence length | 4096, unpacked |
| Batch | micro 1, global 8 |
| Iterations | 1,500 (checkpoint recovered at 1,400) |
| Throughput | 5.2 sec/step, ~2 h |
| Serving | vLLM `v0.24.0-ubuntu2404`, `--max-model-len 16384` |

### Single-GPU overrides
```
model.expert_model_parallel_size=1
model.moe_enable_deepep=false
model.moe_token_dispatcher_type=alltoall
checkpoint.save=/workspace/mbridge_run/checkpoints
```

---

## Open Items

- [x] ~~Merge the 46 routed-expert adapters.~~ Done — same delta to all 128 experts per
      module. Result scored 8 points *worse* than the partial merge.
- [x] ~~Run the full 213-problem held-out eval.~~ Done, three-way.
- [ ] **Verify the grouped-MoE merge is mathematically equivalent** to the adapter at
      runtime. Compare logits from the un-merged model (`setup()` output, adapters
      attached) against the merged export on identical input. If they diverge, the
      format-vs-completion story is an artifact and the full merge is simply wrong.
- [ ] **Fix the MBPP path in the eval harness.** Confirmed a harness bug, not a model
      property — the 284B teacher also scores 1/23. Check test format handling and code
      extraction. Until fixed, exclude MBPP from any reported numbers.
- [x] ~~Run the teacher through the same harness.~~ Done: 68.5% pass@1, 92.5% student
      retention. Contaminated (teacher saw these problems during corpus construction),
      smaller token budget, different vLLM build.
- [ ] **Rerun HumanEval+ against the merged model.** The 87.2% vs 87.8% comparison was
      run against a base-weights copy and is void.
- [ ] File Megatron-Bridge issues: (a) `save_hf_pretrained` silently skips subclass LoRA
      adapters, (b) `import_hf_ckpt.py` tokenizer-assets traceback, (c) single-GPU DeepEP
      defaults.
- [ ] Determine which of the three TRL loss theories was actually responsible — useful
      even though TRL is no longer the path.
- [ ] Enterprise synthetic corpus yield was poor (327 seeds → 40 final examples, ~12%);
      the ≥2-assert validator in corpus step 02 is likely too strict.
- [ ] Move HF cache off the root filesystem permanently.