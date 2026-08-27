#!/usr/bin/env python3
"""
04_execute_and_filter.py — Execution-based quality filtering.

Extracts Python code from reasoning traces, runs it in a Docker sandbox
against the problem's test suite, and keeps only traces whose code passes
all tests.

This is the coding demo's killer advantage: automated ground truth.

Requires: Docker with the sandbox image built.
    docker build -f Dockerfile.sandbox -t fury-kd-sandbox:latest .

Input:  data/traces/traces.jsonl
Output: data/filtered/passing_traces.jsonl
        data/filtered/failing_traces.jsonl  (for analysis)
"""

import asyncio
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("execute_filter")


def strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks (Nemotron 3 emits these).
    Code inside thinking is exploratory/partial — never extract from it.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_python_code(trace: str) -> str | None:
    """Extract the Python code block from a reasoning trace."""

    trace = strip_think_blocks(trace)

    # Try to find ```python ... ``` blocks
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
        r"## Solution\s*\n```python\s*\n(.*?)```",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, trace, re.DOTALL)
        if matches:
            # Take the longest match (likely the main solution, not a snippet)
            return max(matches, key=len).strip()

    return None


async def run_in_sandbox(
    solution_code: str,
    test_code: str,
    semaphore: asyncio.Semaphore,
    io_tests: list | None = None,
) -> dict:
    """
    Execute solution + tests in a Docker sandbox container.

    Two modes:
    - Assertion mode (default): test_code executes in shared namespace
    - I/O mode (APPS): io_tests is a list of {"input","output"} pairs;
      the solution runs as a script, stdin fed per case, stdout compared
    """

    async with semaphore:
        with tempfile.TemporaryDirectory() as tmpdir:
            solution_path = Path(tmpdir) / "solution.py"
            solution_path.write_text(solution_code)

            if io_tests:
                test_path = Path(tmpdir) / "io_tests.json"
                test_path.write_text(json.dumps(io_tests))
            else:
                test_path = Path(tmpdir) / "tests.py"
                test_path.write_text(test_code)

            # Run in Docker sandbox
            cmd = [
                "docker", "run",
                "--rm",
                "--network", "none",                          # No network
                "--memory", config.SANDBOX_MEMORY_LIMIT,      # Memory limit
                "--cpus", str(config.SANDBOX_CPU_LIMIT),       # CPU limit
                "--read-only",                                 # Read-only filesystem
                "--tmpfs", "/tmp:size=64m",                    # Writable /tmp
                "-v", f"{tmpdir}:/workspace:ro",               # Mount code read-only
                config.SANDBOX_IMAGE,
                "/workspace/solution.py",
                f"/workspace/{test_path.name}",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=config.SANDBOX_TIMEOUT_SECONDS + 10,  # Extra margin for Docker overhead
                )

                stdout_str = stdout.decode("utf-8", errors="replace").strip()

                try:
                    result = json.loads(stdout_str)
                except json.JSONDecodeError:
                    result = {
                        "passed": False,
                        "errors": [{"type": "ParseError", "message": f"Could not parse sandbox output: {stdout_str[:500]}"}],
                        "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                    }

                result["exit_code"] = proc.returncode
                return result

            except asyncio.TimeoutError:
                # Kill the container if it hangs
                proc.kill()
                return {
                    "passed": False,
                    "errors": [{"type": "TimeoutError", "message": "Docker container timed out"}],
                    "exit_code": -1,
                }
            except Exception as e:
                return {
                    "passed": False,
                    "errors": [{"type": type(e).__name__, "message": str(e)}],
                    "exit_code": -1,
                }


async def main():
    logger.info("=" * 60)
    logger.info("Execution-Based Quality Filtering")
    logger.info("=" * 60)

    # Verify Docker and sandbox image
    proc = await asyncio.create_subprocess_exec(
        "docker", "image", "inspect", config.SANDBOX_IMAGE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"Sandbox image '{config.SANDBOX_IMAGE}' not found.")
        logger.error("Build it: docker build -f Dockerfile.sandbox -t fury-kd-sandbox:latest .")
        sys.exit(1)

    logger.info(f"Sandbox image: {config.SANDBOX_IMAGE}")

    # Load traces
    traces_path = config.TRACES_DIR / "traces.jsonl"
    if not traces_path.exists():
        logger.error("No traces found. Run 03_generate_traces.py first.")
        sys.exit(1)

    traces = []
    with open(traces_path) as f:
        for line in f:
            traces.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(traces):,} traces")

    # Filter: only traces with extractable code AND test code
    executable_traces = []
    no_code = 0
    no_tests = 0

    for trace in traces:
        code = extract_python_code(trace["reasoning_trace"])
        if not code:
            no_code += 1
            continue

        test_code = trace.get("test_code", "")
        io_tests = trace.get("io_tests") or []
        has_assertion_tests = test_code and len(test_code.strip()) >= 10
        if not has_assertion_tests and not io_tests:
            no_tests += 1
            continue

        trace["extracted_code"] = code
        executable_traces.append(trace)

    logger.info(f"Executable traces: {len(executable_traces):,}")
    logger.info(f"  Skipped (no code block): {no_code}")
    logger.info(f"  Skipped (no test code): {no_tests}")

    # Execute in sandboxes
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_SANDBOXES)
    passing = []
    failing = []
    errors = 0

    start_time = time.time()

    # Process in batches
    batch_size = 100
    for batch_start in range(0, len(executable_traces), batch_size):
        batch = executable_traces[batch_start : batch_start + batch_size]

        tasks = [
            run_in_sandbox(
                t["extracted_code"],
                t.get("test_code", ""),
                semaphore,
                io_tests=t.get("io_tests"),
            )
            for t in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for trace, result in zip(batch, results):
            if isinstance(result, Exception):
                errors += 1
                continue

            trace["execution_result"] = result

            if result.get("passed", False):
                passing.append(trace)
            else:
                failing.append(trace)

        # Progress
        total_done = batch_start + len(batch)
        elapsed = time.time() - start_time
        rate = total_done / elapsed if elapsed > 0 else 0
        pass_rate = len(passing) / total_done * 100 if total_done > 0 else 0
        logger.info(
            f"  Progress: {total_done}/{len(executable_traces)} | "
            f"Pass rate: {pass_rate:.1f}% | "
            f"Rate: {rate:.1f}/s"
        )

    # Write results
    passing_path = config.FILTERED_DIR / "passing_traces.jsonl"
    failing_path = config.FILTERED_DIR / "failing_traces.jsonl"

    with open(passing_path, "w") as f:
        for t in passing:
            f.write(json.dumps(t) + "\n")

    with open(failing_path, "w") as f:
        for t in failing:
            f.write(json.dumps(t) + "\n")

    # Select best trace per problem (from passing)
    best_per_problem = {}
    for t in passing:
        pid = t["problem_id"]
        if pid not in best_per_problem:
            best_per_problem[pid] = t
        # If multiple traces pass, keep the first (could add quality scoring here)

    best_path = config.FILTERED_DIR / "best_passing_traces.jsonl"
    with open(best_path, "w") as f:
        for t in best_per_problem.values():
            f.write(json.dumps(t) + "\n")

    # Report
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Execution Filtering Complete")
    logger.info(f"  Total traces tested: {len(executable_traces):,}")
    logger.info(f"  Passing: {len(passing):,} ({len(passing)/len(executable_traces)*100:.1f}%)")
    logger.info(f"  Failing: {len(failing):,}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Unique problems with passing traces: {len(best_per_problem):,}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  Output: {passing_path}")
    logger.info(f"  Best-of-N: {best_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
