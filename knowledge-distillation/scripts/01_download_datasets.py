#!/usr/bin/env python3
"""
01_download_datasets.py — Download and normalize benchmark coding datasets.

Downloads APPS, MBPP, and HumanEval+ from HuggingFace, normalizes them into
a common JSON format for the trace generation pipeline.

Output: data/raw/{dataset_name}.jsonl
Each line: {"problem_id", "source", "difficulty", "problem", "test_code", "function_signature"}
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("download_datasets")


def download_humaneval_plus():
    """Download HumanEval+ (EvalPlus enhanced test suites)."""
    from datasets import load_dataset

    logger.info("Downloading HumanEval+ (evalplus/humanevalplus)...")
    ds = load_dataset("evalplus/humanevalplus", split="test")

    output_path = config.RAW_DIR / "humaneval_plus.jsonl"
    count = 0

    with open(output_path, "w") as f:
        for item in ds:
            task_id = item.get("task_id", f"HumanEval/{count}")
            prompt = item.get("prompt", "")
            canonical = item.get("canonical_solution", "")
            test = item.get("test", "")
            entry_point = item.get("entry_point", "")

            # Build the problem statement from the docstring in the prompt
            problem_text = prompt.strip()

            # Build executable test code.
            # CRITICAL: never include the canonical solution or the prompt stub
            # in test_code — they would redefine the function in the shared
            # namespace and make every teacher solution "pass". The teacher's
            # extracted code must be the ONLY implementation present.
            test_code = f"{test}\ncheck({entry_point})"

            record = {
                "problem_id": task_id.replace("/", "_"),
                "source": "humaneval_plus",
                "difficulty": "medium",
                "problem": problem_text,
                "test_code": test_code,
                "function_signature": prompt.strip().split("\n")[0] if prompt else "",
                "entry_point": entry_point,
                "canonical_solution": canonical,
            }

            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info(f"HumanEval+: {count} problems → {output_path}")
    return count


def download_mbpp():
    """Download MBPP (Mostly Basic Python Problems)."""
    from datasets import load_dataset

    logger.info("Downloading MBPP (google-research-datasets/mbpp)...")
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")

    output_path = config.RAW_DIR / "mbpp.jsonl"
    count = 0

    with open(output_path, "w") as f:
        for item in ds:
            task_id = item.get("task_id", count)
            prompt = item.get("prompt", item.get("text", ""))
            test_list = item.get("test_list", [])
            code = item.get("code", "")

            # Build executable test code from test_list.
            # CRITICAL: do NOT prepend the reference `code` — it would redefine
            # the function after the teacher's solution and mask real failures.
            test_code = "\n".join(test_list) if test_list else ""

            record = {
                "problem_id": f"MBPP_{task_id}",
                "source": "mbpp",
                "difficulty": "easy",
                "problem": prompt,
                "test_code": test_code,
                "function_signature": "",
                "canonical_solution": code,   # reference only, never executed
            }

            f.write(json.dumps(record) + "\n")
            count += 1

    logger.info(f"MBPP: {count} problems → {output_path}")
    return count


def download_apps():
    """Download APPS dataset (coding competition problems)."""
    from datasets import load_dataset

    logger.info("Downloading APPS (codeparrot/apps)...")

    output_path = config.RAW_DIR / "apps.jsonl"
    count = 0

    difficulty_map = {"introductory": "easy", "interview": "medium", "competition": "hard"}

    with open(output_path, "w") as f:
        for split in ["test"]:
            try:
                ds = load_dataset("codeparrot/apps", split=split, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Could not load APPS split '{split}': {e}")
                continue

            for item in ds:
                problem_id = item.get("problem_id", count)
                question = item.get("question", "")
                difficulty = difficulty_map.get(item.get("difficulty", ""), "medium")
                solutions = item.get("solutions", "")
                input_output = item.get("input_output", "")

                # APPS problems are stdin/stdout style — assertion-based test
                # code does not apply. Store I/O pairs as structured data;
                # the execution filter (04) runs the solution as a script,
                # feeds each input on stdin, and compares stripped stdout.
                # NOTE: comments are NOT tests — a previous version emitted
                # "# Input:/# Expected:" comments which pass trivially.
                io_tests = []
                if input_output:
                    try:
                        io_data = json.loads(input_output)
                        inputs = io_data.get("inputs", [])
                        outputs = io_data.get("outputs", [])
                        for inp, out in zip(inputs[:8], outputs[:8]):
                            if isinstance(inp, list):
                                inp = "\n".join(str(x) for x in inp)
                            if isinstance(out, list):
                                out = "\n".join(str(x) for x in out)
                            io_tests.append({"input": str(inp), "output": str(out)})
                    except (json.JSONDecodeError, TypeError):
                        io_tests = []

                # Parse first solution if available
                canonical = ""
                if solutions:
                    try:
                        sol_list = json.loads(solutions)
                        if sol_list:
                            canonical = sol_list[0]
                    except (json.JSONDecodeError, TypeError):
                        canonical = ""

                # Skip APPS problems without executable I/O tests entirely —
                # an unverifiable trace is worse than no trace.
                if not io_tests:
                    continue

                record = {
                    "problem_id": f"APPS_{problem_id}",
                    "source": "apps",
                    "difficulty": difficulty,
                    "problem": question,
                    "test_code": "",              # not assertion-style
                    "io_tests": io_tests,          # stdin/stdout pairs
                    "function_signature": "",
                    "canonical_solution": canonical,
                }

                f.write(json.dumps(record) + "\n")
                count += 1

    logger.info(f"APPS: {count} problems → {output_path}")
    return count


def verify_downloads():
    """Verify all datasets were downloaded and report statistics."""
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)

    total = 0
    for path in sorted(config.RAW_DIR.glob("*.jsonl")):
        with open(path) as f:
            count = sum(1 for _ in f)
        logger.info(f"  {path.name}: {count:,} problems")
        total += count

    logger.info(f"  TOTAL: {total:,} problems")
    logger.info("=" * 60)
    return total


def main():
    logger.info("Starting dataset download pipeline...")

    counts = {}
    counts["humaneval_plus"] = download_humaneval_plus()
    counts["mbpp"] = download_mbpp()
    counts["apps"] = download_apps()

    total = verify_downloads()

    logger.info(f"Download complete. {total:,} problems ready for trace generation.")
    logger.info(f"Output directory: {config.RAW_DIR}")


if __name__ == "__main__":
    main()
