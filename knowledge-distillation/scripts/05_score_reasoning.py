#!/usr/bin/env python3
"""
05_score_reasoning.py — Score reasoning quality of passing traces.

All traces at this stage already pass their test suites. This step scores
the quality of the REASONING PROCESS (not the code correctness) to select
the best training data.

Requires: vLLM running the teacher model on TEACHER_PORT.

Input:  data/filtered/passing_traces.jsonl
Output: data/scored/scored_traces.jsonl
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("score_reasoning")


def compute_weighted_score(scores: dict) -> float:
    """Compute weighted average quality score."""
    total = 0.0
    for axis, weight in config.QUALITY_WEIGHTS.items():
        total += scores.get(axis, 1) * weight
    return round(total, 3)


async def score_trace(
    client: httpx.AsyncClient,
    trace: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Score a single trace's reasoning quality."""

    scoring_prompt = (config.PROMPTS_DIR / "quality_scoring.txt").read_text()

    user_message = (
        f"Problem:\n{trace['problem']}\n\n"
        f"Solution with reasoning:\n{trace['reasoning_trace']}"
    )

    async with semaphore:
        try:
            response = await client.post(
                f"{config.VLLM_BASE_URL}/chat/completions",
                json={
                    "model": config.TEACHER_MODEL,
                    "messages": [
                        {"role": "system", "content": scoring_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    **config.QUALITY_SCORING,
                    # NOTE: deliberately NOT applying the teacher sampling
                    # override here — scoring needs low temperature for
                    # consistency regardless of teacher.
                },
                timeout=config.HTTPX_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Robust JSON extraction: strip <think> blocks (Nemotron),
            # strip markdown fences, then pull the first {...} object.
            import re as _re
            content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL)
            match = _re.search(r"\{[^{}]*\}", content)
            if not match:
                raise json.JSONDecodeError("no JSON object found", content, 0)
            scores = json.loads(match.group(0))

            # Validate score ranges
            for axis in config.QUALITY_WEIGHTS:
                if axis not in scores:
                    scores[axis] = 1
                scores[axis] = max(1, min(5, int(scores[axis])))

            scores["weighted_score"] = compute_weighted_score(scores)

            trace["quality_scores"] = scores
            return trace

        except json.JSONDecodeError:
            logger.warning(f"  Could not parse scores for {trace['trace_id']}: {content[:100]}")
            return None
        except Exception as e:
            logger.error(f"  Scoring error for {trace['trace_id']}: {e}")
            return None


async def main():
    logger.info("=" * 60)
    logger.info("Reasoning Quality Scoring")
    logger.info("=" * 60)

    # Load passing traces
    input_path = config.FILTERED_DIR / "passing_traces.jsonl"
    if not input_path.exists():
        logger.error("No passing traces found. Run 04_execute_and_filter.py first.")
        sys.exit(1)

    traces = []
    with open(input_path) as f:
        for line in f:
            traces.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(traces):,} passing traces to score")

    # Verify teacher
    async with httpx.AsyncClient() as client:
        try:
            url = f"{config.VLLM_BASE_URL}/models"
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Teacher not available: {e}")
            sys.exit(1)

    # Score all traces
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    scored = []
    failed = 0
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        batch_size = 50
        for batch_start in range(0, len(traces), batch_size):
            batch = traces[batch_start : batch_start + batch_size]

            tasks = [score_trace(client, t, semaphore) for t in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, dict) and result is not None:
                    scored.append(result)
                else:
                    failed += 1

            total_done = batch_start + len(batch)
            elapsed = time.time() - start_time
            rate = total_done / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Progress: {total_done}/{len(traces)} | "
                f"Scored: {len(scored)} | Failed: {failed} | "
                f"Rate: {rate:.1f}/s"
            )

    # Apply quality threshold
    kept = [t for t in scored if t["quality_scores"]["weighted_score"] >= config.QUALITY_THRESHOLD_KEEP]
    borderline = [
        t for t in scored
        if config.QUALITY_THRESHOLD_REVISE <= t["quality_scores"]["weighted_score"] < config.QUALITY_THRESHOLD_KEEP
    ]
    discarded = [t for t in scored if t["quality_scores"]["weighted_score"] < config.QUALITY_THRESHOLD_REVISE]

    # Select best trace per problem
    best_per_problem = {}
    for t in kept:
        pid = t["problem_id"]
        current_best = best_per_problem.get(pid)
        if current_best is None or t["quality_scores"]["weighted_score"] > current_best["quality_scores"]["weighted_score"]:
            best_per_problem[pid] = t

    # Write outputs
    output_path = config.SCORED_DIR / "scored_traces.jsonl"
    with open(output_path, "w") as f:
        for t in scored:
            f.write(json.dumps(t) + "\n")

    best_path = config.SCORED_DIR / "best_scored_traces.jsonl"
    with open(best_path, "w") as f:
        for t in best_per_problem.values():
            f.write(json.dumps(t) + "\n")

    # Statistics
    if scored:
        all_scores = [t["quality_scores"]["weighted_score"] for t in scored]
        avg_score = sum(all_scores) / len(all_scores)
        min_score = min(all_scores)
        max_score = max(all_scores)
    else:
        avg_score = min_score = max_score = 0

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Reasoning Quality Scoring Complete")
    logger.info(f"  Total scored: {len(scored):,}")
    logger.info(f"  Score stats: avg={avg_score:.2f}, min={min_score:.2f}, max={max_score:.2f}")
    logger.info(f"  Kept (≥{config.QUALITY_THRESHOLD_KEEP}): {len(kept):,}")
    logger.info(f"  Borderline: {len(borderline):,}")
    logger.info(f"  Discarded (<{config.QUALITY_THRESHOLD_REVISE}): {len(discarded):,}")
    logger.info(f"  Best per problem: {len(best_per_problem):,}")
    logger.info(f"  Failed to score: {failed}")
    logger.info(f"  Time: {elapsed/60:.1f} minutes")
    logger.info(f"  Output: {best_path}")

    # Per-axis breakdown
    if scored:
        for axis in config.QUALITY_WEIGHTS:
            axis_scores = [t["quality_scores"][axis] for t in scored]
            axis_avg = sum(axis_scores) / len(axis_scores)
            logger.info(f"    {axis}: avg={axis_avg:.2f}")

    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
