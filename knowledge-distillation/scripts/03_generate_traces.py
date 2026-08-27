#!/usr/bin/env python3
"""
03_generate_traces.py — Generate reasoning traces from the 405B teacher.

For each seed problem (benchmark + enterprise), generates TRACES_PER_PROBLEM
candidate reasoning traces using the teacher model. Each trace includes
full chain-of-thought: analysis, approach selection, implementation, verification.

Requires: vLLM running the teacher model on TEACHER_PORT.

Input:  data/raw/*.jsonl + data/seeds/enterprise_seeds.jsonl
Output: data/traces/traces.jsonl
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("generate_traces")


# httpx client factory — fresh connection per request, no keepalive reuse.
# vLLM 0.23 drops persistent connections under async load, causing
# "Response ended prematurely" errors on every request.
def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(config.HTTPX_TIMEOUT, connect=30.0),
        limits=httpx.Limits(
            max_keepalive_connections=0,
            max_connections=config.MAX_CONCURRENT_REQUESTS,
        ),
        http2=False,
    )


def load_all_problems() -> list[dict]:
    """Load all seed problems from raw benchmarks and enterprise seeds."""
    problems = []

    # Load benchmark datasets
    for path in sorted(config.RAW_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        logger.info(f"  Loaded {path.name}")

    # Load enterprise seeds
    enterprise_path = config.SEEDS_DIR / "enterprise_seeds.jsonl"
    if enterprise_path.exists():
        with open(enterprise_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        logger.info(f"  Loaded enterprise_seeds.jsonl")
    else:
        logger.warning("No enterprise seeds found — run 02_generate_seed_problems.py first")

    logger.info(f"Total problems loaded: {len(problems):,}")

    # Canary mode: cap at N problems for end-to-end validation
    canary_limit = int(os.environ.get("CANARY", "0"))
    if canary_limit > 0:
        problems = problems[:canary_limit]
        logger.info(f"[CANARY] Capped to {len(problems):,} problems")

    logger.info(f"[DEBUG] problems returned from load: {len(problems)}, TRACES_PER_PROBLEM: {config.TRACES_PER_PROBLEM}")

    return problems


def build_trace_prompt(problem: dict) -> list[dict]:
    """Build the chat messages for trace generation."""

    system_prompt = (config.PROMPTS_DIR / "trace_generation.txt").read_text()

    # Build user message with problem + test cases if available
    user_parts = [problem["problem"]]

    test_code = problem.get("test_code", "")
    if test_code and len(test_code) < 2000:  # Don't overwhelm with huge test suites
        user_parts.append(
            f"\nTest cases for reference (your solution must pass these):\n```python\n{test_code}\n```"
        )

    func_sig = problem.get("function_signature", "")
    if func_sig:
        user_parts.append(f"\nExpected function signature: `{func_sig}`")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


async def generate_trace(
    client: httpx.AsyncClient,
    problem: dict,
    trace_num: int,
) -> dict | None:
    """Generate a single reasoning trace for a problem."""

    messages = build_trace_prompt(problem)

    try:
        response = await client.post(
            f"{config.VLLM_BASE_URL}/chat/completions",
            json={
                "model": config.TEACHER_MODEL,
                "messages": messages,
                **config.TRACE_GENERATION,
            },
            timeout=config.HTTPX_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        # Strip <think>...</think> blocks (Nemotron 3 reasoning traces).
        # The structured Analysis/Approach/Solution/Verification sections are
        # the training signal; raw think-stream would pollute the student's
        # output format and blow up sequence lengths past max_seq_length.
        import re as _re
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

        return {
            "problem_id": problem["problem_id"],
            "source": problem.get("source", "unknown"),
            "category": problem.get("category", "benchmark"),
            "difficulty": problem.get("difficulty", "medium"),
            "problem": problem["problem"],
            "test_code": problem.get("test_code", ""),
            "io_tests": problem.get("io_tests", []),
            "function_signature": problem.get("function_signature", ""),
            "trace_num": trace_num,
            "trace_id": f"{problem['problem_id']}_t{trace_num}",
            "reasoning_trace": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "timestamp": time.time(),
        }

    except httpx.TimeoutException:
        logger.warning(f"  Timeout: {problem['problem_id']} trace {trace_num}")
        return None
    except Exception as e:
        logger.error(f"  Error: {problem['problem_id']} trace {trace_num}: {e}")
        return None


async def generate_traces_for_problem(
    client: httpx.AsyncClient,
    problem: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Generate all candidate traces for a single problem."""

    traces = []
    for t in range(config.TRACES_PER_PROBLEM):
        async with semaphore:
            trace = await generate_trace(client, problem, t)
            if trace:
                traces.append(trace)
    return traces


async def main():
    logger.info("=" * 60)
    logger.info("Reasoning Trace Generation Pipeline")
    logger.info("=" * 60)

    # Load problems
    problems = load_all_problems()

    if not problems:
        logger.error("No problems found. Run 01_download_datasets.py first.")
        sys.exit(1)

    # Check for existing progress (resume support)
    output_path = config.TRACES_DIR / "traces.jsonl"
    completed_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line.strip())
                completed_ids.add(obj["trace_id"])
        logger.info(f"Resuming: {len(completed_ids)} traces already generated")

    # Filter to remaining work
    remaining = []
    for p in problems:
        for t in range(config.TRACES_PER_PROBLEM):
            trace_id = f"{p['problem_id']}_t{t}"
            if trace_id not in completed_ids:
                remaining.append((p, t))

    total_calls = len(remaining)
    logger.info(f"Remaining inference calls: {total_calls:,}")

    if total_calls == 0:
        logger.info("All traces already generated. Nothing to do.")
        return

    # Estimate time
    est_seconds_per_call = 45  # Conservative for 405B FP8
    est_total_hours = (total_calls * est_seconds_per_call) / 3600
    logger.info(f"Estimated time: {est_total_hours:.1f} hours")
    logger.info(f"Concurrency: {config.MAX_CONCURRENT_REQUESTS} parallel requests")

    # Verify teacher is available
    async with make_client() as client:
        try:
            url = f"{config.VLLM_BASE_URL}/models"
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            logger.info(f"Teacher model available on port {config.TEACHER_PORT}")
        except Exception as e:
            logger.error(f"Teacher not available: {e}")
            sys.exit(1)

    # Generate traces with progress tracking
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    completed = 0
    failed = 0
    start_time = time.time()

    async with make_client() as client:
        # Open output file in append mode for resume support
        with open(output_path, "a") as f:

            # Process in batches for memory efficiency
            batch_size = 50
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start : batch_start + batch_size]

                async def process_one(problem, trace_num):
                    async with semaphore:
                        return await generate_trace(client, problem, trace_num)

                tasks = [process_one(p, t) for p, t in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, dict) and result is not None:
                        f.write(json.dumps(result) + "\n")
                        completed += 1
                    else:
                        failed += 1

                # Flush periodically
                f.flush()

                # Progress report
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining_time = (total_calls - completed) / rate if rate > 0 else 0
                logger.info(
                    f"  Progress: {completed}/{total_calls} "
                    f"({completed/total_calls*100:.1f}%) | "
                    f"Failed: {failed} | "
                    f"Rate: {rate:.1f}/s | "
                    f"ETA: {remaining_time/3600:.1f}h"
                )

    # Final report
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Trace Generation Complete")
    logger.info(f"  Completed: {completed:,}")
    logger.info(f"  Failed: {failed:,}")
    logger.info(f"  Total time: {elapsed/3600:.1f} hours")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())