#!/usr/bin/env python3
"""
eval_holdout_score.py — Score held-out completions, assertion + I/O modes.

APPS problems use stdin/stdout test cases; HumanEval+/MBPP use assertions.
The sandbox runner switches on the test file extension (.json => I/O mode),
so this script writes the right file type per problem.

Breaks results down by source and difficulty, since the held-out set is
mostly APPS and a change may only show up on the harder slices.

Usage:
    python3 eval_holdout_score.py --model-name stock
    python3 eval_holdout_score.py --compare stock distilled
"""

import argparse
import asyncio
import json
import re
import tempfile
from collections import defaultdict
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
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_python_code(text: str) -> str | None:
    text = strip_think_blocks(text)
    for pattern in (
        r"## Solution\s*\n```python\s*\n(.*?)```",
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ):
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()
    return None


def score_format(text: str) -> dict:
    body = strip_think_blocks(text)
    present = [s for s in REQUIRED_SECTIONS if s in body]
    return {
        "section_count": len(present),
        "full_format": len(present) == len(REQUIRED_SECTIONS),
        "emitted_think_block": "<think>" in text,
    }


async def run_in_sandbox(
    solution_code: str,
    test_code: str,
    io_tests: list,
    semaphore: asyncio.Semaphore,
) -> dict:
    """I/O mode when io_tests present, assertion mode otherwise."""
    async with semaphore:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "solution.py").write_text(solution_code)

            if io_tests:
                test_name = "io_tests.json"
                (Path(tmpdir) / test_name).write_text(json.dumps(io_tests))
            else:
                test_name = "tests.py"
                (Path(tmpdir) / test_name).write_text(test_code)

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
                f"/workspace/{test_name}",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=SANDBOX_TIMEOUT + 20
                )
                out = stdout.decode("utf-8", errors="replace").strip()
                try:
                    return json.loads(out)
                except json.JSONDecodeError:
                    return {"passed": False, "errors": [{"type": "ParseError", "message": out[:300]}]}
            except asyncio.TimeoutError:
                proc.kill()
                return {"passed": False, "errors": [{"type": "TimeoutError"}]}
            except Exception as e:
                return {"passed": False, "errors": [{"type": type(e).__name__, "message": str(e)}]}


async def score_model(model_name: str) -> dict:
    path = RESULTS_DIR / f"holdout_{model_name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Not found: {path} — run eval_holdout_generate.py first")

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
        entry = {
            "problem_id": rec["problem_id"],
            "source": rec["source"],
            "difficulty": rec["difficulty"],
            "format": fmt,
            "io_mode": bool(rec.get("io_tests")),
        }
        if code is None:
            no_code += 1
            entry["no_code"] = True
            meta.append(entry)
            tasks.append(None)
            continue
        entry["no_code"] = False
        meta.append(entry)
        tasks.append(
            run_in_sandbox(code, rec.get("test_code", ""), rec.get("io_tests") or [], semaphore)
        )

    real = [t for t in tasks if t is not None]
    print(f"Executable: {len(real)} (no code block: {no_code})", flush=True)
    exec_results = await asyncio.gather(*real)

    it = iter(exec_results)
    detail = []
    passed = 0
    for m, t in zip(meta, tasks):
        if t is None:
            m["passed"] = False
        else:
            r = next(it)
            m["passed"] = bool(r.get("passed", False))
            if m["passed"]:
                passed += 1
        detail.append(m)

    n = len(records)

    by_source = defaultdict(lambda: [0, 0])
    by_difficulty = defaultdict(lambda: [0, 0])
    for m in detail:
        by_source[m["source"]][1] += 1
        by_difficulty[m["difficulty"]][1] += 1
        if m["passed"]:
            by_source[m["source"]][0] += 1
            by_difficulty[m["difficulty"]][0] += 1

    summary = {
        "model": model_name,
        "total": n,
        "passed": passed,
        "pass_at_1": passed / n if n else 0.0,
        "no_code_block": no_code,
        "full_format_rate": sum(1 for m in detail if m["format"]["full_format"]) / n if n else 0.0,
        "avg_sections": sum(m["format"]["section_count"] for m in detail) / n if n else 0.0,
        "by_source": {k: {"passed": v[0], "total": v[1], "rate": v[0] / v[1]} for k, v in by_source.items()},
        "by_difficulty": {k: {"passed": v[0], "total": v[1], "rate": v[0] / v[1]} for k, v in by_difficulty.items()},
    }

    print(f"pass@1:          {summary['pass_at_1']:.1%}  ({passed}/{n})", flush=True)
    print(f"full format:     {summary['full_format_rate']:.1%}", flush=True)
    print(f"avg sections /5: {summary['avg_sections']:.2f}", flush=True)
    print("by source:", flush=True)
    for k, v in sorted(summary["by_source"].items()):
        print(f"   {k:<22} {v['rate']:>6.1%}  ({v['passed']}/{v['total']})", flush=True)
    print("by difficulty:", flush=True)
    for k, v in sorted(summary["by_difficulty"].items()):
        print(f"   {k:<22} {v['rate']:>6.1%}  ({v['passed']}/{v['total']})", flush=True)

    with open(RESULTS_DIR / f"holdout_scored_{model_name}.json", "w") as f:
        json.dump({"summary": summary, "detail": detail}, f, indent=2)

    return summary


def compare(a: dict, b: dict) -> None:
    print("\n" + "=" * 60, flush=True)
    print(f"{'metric':<26}{a['model']:>16}{b['model']:>16}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'pass@1 (overall)':<26}{a['pass_at_1']:>15.1%}{b['pass_at_1']:>16.1%}", flush=True)
    for key in sorted(set(a["by_source"]) | set(b["by_source"])):
        av = a["by_source"].get(key, {}).get("rate", 0.0)
        bv = b["by_source"].get(key, {}).get("rate", 0.0)
        n = a["by_source"].get(key, {}).get("total", 0)
        print(f"{'  ' + key + f' (n={n})':<26}{av:>15.1%}{bv:>16.1%}", flush=True)
    for key in sorted(set(a["by_difficulty"]) | set(b["by_difficulty"])):
        av = a["by_difficulty"].get(key, {}).get("rate", 0.0)
        bv = b["by_difficulty"].get(key, {}).get("rate", 0.0)
        n = a["by_difficulty"].get(key, {}).get("total", 0)
        print(f"{'  ' + key + f' (n={n})':<26}{av:>15.1%}{bv:>16.1%}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'full format rate':<26}{a['full_format_rate']:>15.1%}{b['full_format_rate']:>16.1%}", flush=True)
    print(f"{'avg sections /5':<26}{a['avg_sections']:>16.2f}{b['avg_sections']:>16.2f}", flush=True)
    print(f"{'no code block':<26}{a['no_code_block']:>16}{b['no_code_block']:>16}", flush=True)
    print("-" * 60, flush=True)
    print(f"pass@1 delta: {b['pass_at_1'] - a['pass_at_1']:+.1%}", flush=True)
    print("=" * 60, flush=True)


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