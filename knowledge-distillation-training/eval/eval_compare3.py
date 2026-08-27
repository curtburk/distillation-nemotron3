#!/usr/bin/env python3
"""
eval_compare3.py — Three-way comparison across held-out eval runs.

Scores each model's completions in the sandbox (assertion or stdin/stdout mode,
selected per problem), then prints a single table with pass@1, termination rate,
verbosity, format adherence, and per-slice breakdowns side by side.

Also reports problem-level movement between runs, since totals can stay flat while
the underlying outcomes shift.

Usage:
    python3 eval_compare3.py stock213 merged213 full213
    python3 eval_compare3.py stock30 merged30 full30
    python3 eval_compare3.py stock213 merged213 full213 --no-rescore   # reuse cached scores
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
    }


async def run_in_sandbox(code: str, test_code: str, io_tests: list,
                         sem: asyncio.Semaphore) -> dict:
    async with sem:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "solution.py").write_text(code)
            if io_tests:
                name = "io_tests.json"
                (Path(tmp) / name).write_text(json.dumps(io_tests))
            else:
                name = "tests.py"
                (Path(tmp) / name).write_text(test_code)

            cmd = [
                "docker", "run", "--rm", "--network", "none",
                "--memory", SANDBOX_MEMORY, "--cpus", str(SANDBOX_CPUS),
                "--read-only", "--tmpfs", "/tmp:size=64m",
                "-v", f"{tmp}:/workspace:ro",
                SANDBOX_IMAGE,
                "/workspace/solution.py", f"/workspace/{name}",
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=SANDBOX_TIMEOUT + 20
                )
                try:
                    return json.loads(out.decode("utf-8", errors="replace").strip())
                except json.JSONDecodeError:
                    return {"passed": False}
            except asyncio.TimeoutError:
                proc.kill()
                return {"passed": False}
            except Exception:
                return {"passed": False}


async def score_run(name: str, rescore: bool = True) -> dict:
    cache = RESULTS_DIR / f"compare3_scored_{name}.json"
    if not rescore and cache.exists():
        print(f"  {name}: using cached scores", flush=True)
        return json.load(open(cache))

    path = RESULTS_DIR / f"holdout_{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    records = [json.loads(l) for l in open(path) if l.strip()]
    if not records:
        raise SystemExit(f"{path} is empty — the generation run wrote nothing.")

    print(f"  {name}: scoring {len(records)} completions...", flush=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT_SANDBOXES)
    tasks, meta = [], []
    for r in records:
        fmt = score_format(r["completion"])
        code = extract_python_code(r["completion"])
        entry = {
            "problem_id": r["problem_id"],
            "source": r["source"],
            "difficulty": r["difficulty"],
            "sections": fmt["section_count"],
            "full_format": fmt["full_format"],
            "no_code": code is None,
            "terminated": r.get("finish_reason") == "stop",
            "tokens": r.get("completion_tokens", 0),
        }
        meta.append(entry)
        tasks.append(
            None if code is None
            else run_in_sandbox(code, r.get("test_code", ""), r.get("io_tests") or [], sem)
        )

    results = await asyncio.gather(*[t for t in tasks if t is not None])
    it = iter(results)
    for m, t in zip(meta, tasks):
        m["passed"] = False if t is None else bool(next(it).get("passed", False))

    n = len(meta)
    by_source = defaultdict(lambda: [0, 0])
    by_diff = defaultdict(lambda: [0, 0])
    for m in meta:
        by_source[m["source"]][1] += 1
        by_diff[m["difficulty"]][1] += 1
        if m["passed"]:
            by_source[m["source"]][0] += 1
            by_diff[m["difficulty"]][0] += 1

    out = {
        "model": name,
        "total": n,
        "passed": sum(m["passed"] for m in meta),
        "pass_at_1": sum(m["passed"] for m in meta) / n,
        "terminated": sum(m["terminated"] for m in meta),
        "term_rate": sum(m["terminated"] for m in meta) / n,
        "mean_tokens": sum(m["tokens"] for m in meta) / n,
        "no_code": sum(m["no_code"] for m in meta),
        "full_format_rate": sum(m["full_format"] for m in meta) / n,
        "avg_sections": sum(m["sections"] for m in meta) / n,
        "by_source": {k: v[0] / v[1] for k, v in by_source.items()},
        "by_source_n": {k: v[1] for k, v in by_source.items()},
        "by_difficulty": {k: v[0] / v[1] for k, v in by_diff.items()},
        "by_difficulty_n": {k: v[1] for k, v in by_diff.items()},
        "detail": {m["problem_id"]: m["passed"] for m in meta},
    }

    json.dump(out, open(cache, "w"), indent=2)
    return out


def table(runs: list[dict]) -> None:
    names = [r["model"] for r in runs]
    w = 16
    line = "-" * (26 + w * len(runs))

    print("\n" + "=" * (26 + w * len(runs)), flush=True)
    print(f"{'metric':<26}" + "".join(f"{n:>{w}}" for n in names), flush=True)
    print(line, flush=True)

    def row(label, key, kind="pct"):
        vals = []
        for r in runs:
            v = r[key]
            if kind == "pct":
                vals.append(f"{v:>{w}.1%}")
            elif kind == "num":
                vals.append(f"{v:>{w}.2f}")
            elif kind == "tok":
                vals.append(f"{v:>{w},.0f}")
            else:
                vals.append(f"{v:>{w}}")
        print(f"{label:<26}" + "".join(vals), flush=True)

    row("pass@1", "pass_at_1")
    print(line, flush=True)

    for src in sorted(runs[0]["by_source"]):
        n = runs[0]["by_source_n"][src]
        vals = "".join(f"{r['by_source'].get(src, 0):>{w}.1%}" for r in runs)
        print(f"{'  ' + src + f' (n={n})':<26}{vals}", flush=True)
    for d in sorted(runs[0]["by_difficulty"]):
        n = runs[0]["by_difficulty_n"][d]
        vals = "".join(f"{r['by_difficulty'].get(d, 0):>{w}.1%}" for r in runs)
        print(f"{'  ' + d + f' (n={n})':<26}{vals}", flush=True)

    print(line, flush=True)
    row("terminated", "term_rate")
    row("mean tokens", "mean_tokens", "tok")
    row("no code block", "no_code", "int")
    print(line, flush=True)
    row("full format rate", "full_format_rate")
    row("avg sections /5", "avg_sections", "num")
    print("=" * (26 + w * len(runs)), flush=True)


def movement(a: dict, b: dict) -> None:
    """Problem-level churn between two runs — totals can hide real changes."""
    ad, bd = a["detail"], b["detail"]
    shared = set(ad) & set(bd)
    fixed = [k for k in shared if not ad[k] and bd[k]]
    broke = [k for k in shared if ad[k] and not bd[k]]
    print(f"\n{a['model']} -> {b['model']}: "
          f"+{len(fixed)} fixed, -{len(broke)} broken, "
          f"net {len(fixed) - len(broke):+d}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=3, metavar=("BASELINE", "MID", "CANDIDATE"))
    ap.add_argument("--no-rescore", action="store_true",
                    help="reuse cached per-run scores if present")
    args = ap.parse_args()

    print("Scoring runs...", flush=True)
    runs = [await score_run(n, rescore=not args.no_rescore) for n in args.runs]

    table(runs)

    print("\nProblem-level movement:", flush=True)
    movement(runs[0], runs[1])
    movement(runs[0], runs[2])
    movement(runs[1], runs[2])


if __name__ == "__main__":
    asyncio.run(main())
    