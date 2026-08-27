#!/usr/bin/env python3
"""
eval_score.py — Score HumanEval+ completions: pass@1 plus format adherence.

Reuses the sandbox execution approach from corpus construction step 04
(fury-kd-sandbox image, network-disabled, memory/CPU limited, timeout enforced).

Two metrics, reported separately because they can move independently:
  pass@1            — did the extracted code pass the test suite
  format adherence  — did the model emit the taught structure
                      (## Analysis / ## Approach / ## Implementation Plan /
                       ## Solution / ## Verification)

Usage:
    python3 eval_score.py --model-name stock
    python3 eval_score.py --model-name distilled
    python3 eval_score.py --compare stock distilled
"""

import argparse
import asyncio
import json
import re
import tempfile
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SANDBOX_IMAGE = "fury-kd-sandbox:latest"
SANDBOX_MEMORY = "2g"
SANDBOX_CPUS = 1.0
SANDBOX_TIMEOUT = 30
MAX_CONCURRENT_SANDBOXES = 8

REQUIRED_SECTIONS = [
    "## Analysis",
    "## Approach",
    "## Implementation Plan",
    "## Solution",
    "## Verification",
]


def strip_think_blocks(text: str) -> str:
    """Nemotron 3 emits <think>...</think>; code inside is exploratory."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_python_code(text: str) -> str | None:
    """Same extraction logic as corpus construction step 04."""
    text = strip_think_blocks(text)
    patterns = [
        r"## Solution\s*\n```python\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    return None


def score_format(text: str) -> dict:
    """How much of the taught structure did the model actually produce?"""
    body = strip_think_blocks(text)
    present = [s for s in REQUIRED_SECTIONS if s in body]
    return {
        "sections_present": present,
        "section_count": len(present),
        "full_format": len(present) == len(REQUIRED_SECTIONS),
        "emitted_think_block": "<think>" in text,
    }


async def run_in_sandbox(
    solution_code: str, test_code: str, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "solution.py").write_text(solution_code)
            (Path(tmpdir) / "tests.py").write_text(test_code)

            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", SANDBOX_MEMORY,
                "--cpus", str(SANDBOX_CPUS),
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "-v", f"{tmpdir}:/workspace:ro",
                SANDBOX_IMAGE,
                "/workspace/solution.py",
                "/workspace/tests.py",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=SANDBOX_TIMEOUT + 10
                )
                out = stdout.decode("utf-8", errors="replace").strip()
                try:
                    result = json.loads(out)
                except json.JSONDecodeError:
                    result = {
                        "passed": False,
                        "errors": [{"type": "ParseError", "message": out[:300]}],
                    }
                return result
            except asyncio.TimeoutError:
                proc.kill()
                return {"passed": False, "errors": [{"type": "TimeoutError"}]}
            except Exception as e:
                return {"passed": False, "errors": [{"type": type(e).__name__, "message": str(e)}]}


async def score_model(model_name: str) -> dict:
    path = RESULTS_DIR / f"completions_{model_name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Not found: {path} — run eval_generate.py first")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n=== {model_name} ===", flush=True)
    print(f"Completions: {len(records)}", flush=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SANDBOXES)
    tasks = []
    meta = []

    no_code = 0
    for rec in records:
        fmt = score_format(rec["completion"])
        code = extract_python_code(rec["completion"])
        if code is None:
            no_code += 1
            meta.append({"problem_id": rec["problem_id"], "format": fmt, "no_code": True})
            tasks.append(None)
            continue
        meta.append({"problem_id": rec["problem_id"], "format": fmt, "no_code": False})
        tasks.append(run_in_sandbox(code, rec["test_code"], semaphore))

    real_tasks = [t for t in tasks if t is not None]
    print(f"Executable: {len(real_tasks)} (no code block: {no_code})", flush=True)
    exec_results = await asyncio.gather(*real_tasks)

    it = iter(exec_results)
    detailed = []
    passed = 0
    for m, t in zip(meta, tasks):
        if t is None:
            m["passed"] = False
        else:
            r = next(it)
            m["passed"] = bool(r.get("passed", False))
            if m["passed"]:
                passed += 1
        detailed.append(m)

    n = len(records)
    full_format = sum(1 for m in detailed if m["format"]["full_format"])
    avg_sections = sum(m["format"]["section_count"] for m in detailed) / n if n else 0
    think_blocks = sum(1 for m in detailed if m["format"]["emitted_think_block"])

    summary = {
        "model": model_name,
        "total": n,
        "passed": passed,
        "pass_at_1": passed / n if n else 0.0,
        "no_code_block": no_code,
        "full_format": full_format,
        "full_format_rate": full_format / n if n else 0.0,
        "avg_sections": avg_sections,
        "think_blocks": think_blocks,
    }

    print(f"pass@1:            {summary['pass_at_1']:.1%}  ({passed}/{n})", flush=True)
    print(f"full format:       {summary['full_format_rate']:.1%}  ({full_format}/{n})", flush=True)
    print(f"avg sections /5:   {avg_sections:.2f}", flush=True)
    print(f"emitted <think>:   {think_blocks}/{n}", flush=True)
    print(f"no code block:     {no_code}/{n}", flush=True)

    with open(RESULTS_DIR / f"scored_{model_name}.json", "w") as f:
        json.dump({"summary": summary, "detail": detailed}, f, indent=2)

    return summary


def compare(a: dict, b: dict) -> None:
    print("\n" + "=" * 56, flush=True)
    print(f"{'metric':<22}{a['model']:>15}{b['model']:>15}", flush=True)
    print("-" * 56, flush=True)
    rows = [
        ("pass@1", "pass_at_1", "pct"),
        ("full format rate", "full_format_rate", "pct"),
        ("avg sections /5", "avg_sections", "num"),
        ("no code block", "no_code_block", "int"),
        ("emitted <think>", "think_blocks", "int"),
    ]
    for label, key, kind in rows:
        av, bv = a[key], b[key]
        if kind == "pct":
            print(f"{label:<22}{av:>14.1%}{bv:>15.1%}", flush=True)
        elif kind == "num":
            print(f"{label:<22}{av:>15.2f}{bv:>15.2f}", flush=True)
        else:
            print(f"{label:<22}{av:>15}{bv:>15}", flush=True)
    print("-" * 56, flush=True)
    delta = b["pass_at_1"] - a["pass_at_1"]
    print(f"pass@1 delta: {delta:+.1%}", flush=True)
    print("=" * 56, flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    args = ap.parse_args()

    if args.compare:
        a = await score_model(args.compare[0])
        b = await score_model(args.compare[1])
        compare(a, b)
    elif args.model_name:
        await score_model(args.model_name)
    else:
        ap.error("pass --model-name or --compare BASELINE CANDIDATE")


if __name__ == "__main__":
    asyncio.run(main())