#!/usr/bin/env python3
"""
02_generate_seed_problems.py — Generate enterprise-realistic coding problems.

Uses the 405B teacher to generate coding problems in categories that
benchmarks don't cover: API integration, data pipelines, debugging,
refactoring, security-aware code, etc.

Requires: vLLM running the teacher model on TEACHER_PORT.

Output: data/seeds/enterprise_seeds.jsonl
"""
import os 
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Canary mode: skip enterprise seed generation entirely
if os.environ.get("CANARY"):
    print("[CANARY] Skipping enterprise seed generation")
    sys.exit(0)

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("generate_seeds")

# Category definitions with target counts
CATEGORIES = {
    "api_integration": {
        "count": 500,
        "description": (
            "REST API client wrappers with pagination, authentication, "
            "retry logic, rate limiting, and error handling. Include "
            "realistic API response formats and edge cases like token "
            "expiry, 429 responses, and partial failures."
        ),
    },
    "data_pipeline": {
        "count": 500,
        "description": (
            "ETL pipelines that validate, transform, and load data. "
            "Include streaming processing, error recovery, schema "
            "validation, deduplication, and logging. Use CSV, JSON, "
            "and structured data formats."
        ),
    },
    "security_aware": {
        "count": 500,
        "description": (
            "Code that handles security concerns: input validation, "
            "parameterized queries, secrets management, audit logging, "
            "safe deserialization, CSRF protection, and secure file handling."
        ),
    },
    "refactoring": {
        "count": 300,
        "description": (
            "Provide working but poorly structured code (150-300 lines) "
            "and ask for refactoring into clean, modular, testable code. "
            "Include the original code in the problem statement. Preserve "
            "all behavior. Add type hints, docstrings, and tests."
        ),
    },
    "debugging": {
        "count": 300,
        "description": (
            "Provide code with subtle bugs (off-by-one, race conditions, "
            "incorrect state transitions, type confusion, boundary errors). "
            "Include the buggy code and failing test cases. Ask for root "
            "cause analysis, minimal fix, and explanation."
        ),
    },
    "code_review": {
        "count": 200,
        "description": (
            "Provide a code snippet or pull request and ask for review. "
            "Include security vulnerabilities, performance issues, style "
            "violations, missing error handling, or logic errors to find."
        ),
    },
    "system_design": {
        "count": 200,
        "description": (
            "Design and implement non-trivial systems: task queues with "
            "priorities, LRU caches, connection pools, rate limiters, "
            "state machines, pub/sub systems, circuit breakers, and "
            "graceful shutdown handlers."
        ),
    },
}


async def generate_batch(
    client: httpx.AsyncClient,
    category: str,
    batch_size: int,
    batch_num: int,
) -> list[dict]:
    """Generate a batch of seed problems for a category."""

    prompt_template = (config.PROMPTS_DIR / "seed_generation.txt").read_text()
    category_info = CATEGORIES[category]

    prompt = prompt_template.format(
        category=category,
        count=batch_size,
    )

    # Add category-specific guidance
    prompt += f"\n\nFocus area for this category:\n{category_info['description']}"

    try:
        response = await client.post(
            f"{config.VLLM_BASE_URL}/chat/completions",
            json={
                "model": config.TEACHER_MODEL,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                **config.SEED_GENERATION,
            },
            timeout=config.HTTPX_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse JSONL output from teacher
        problems = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Ensure required fields
                if "problem" not in obj or "test_code" not in obj:
                    continue

                # Validate the generated test code:
                # 1. must compile (broken tests discard good solutions)
                # 2. must contain >= 2 assert statements (0 asserts = trivially
                #    passing tests that let bad solutions through the filter)
                tc = obj["test_code"]
                try:
                    compile(tc, "<generated_test>", "exec")
                except SyntaxError:
                    continue
                if tc.count("assert") < 2:
                    continue

                obj["source"] = "enterprise_synthetic"
                obj["category"] = category
                obj.setdefault("difficulty", "medium")
                obj.setdefault("problem_id", f"{category}_{batch_num:03d}_{len(problems):03d}")
                problems.append(obj)
            except json.JSONDecodeError:
                continue

        logger.info(
            f"  [{category}] batch {batch_num}: generated {len(problems)}/{batch_size} problems"
        )
        return problems

    except Exception as e:
        logger.error(f"  [{category}] batch {batch_num} failed: {e}")
        return []


async def generate_category(category: str) -> list[dict]:
    """Generate all seed problems for a single category."""

    target = CATEGORIES[category]["count"]
    batch_size = 10  # Problems per teacher call
    num_batches = (target + batch_size - 1) // batch_size

    logger.info(f"Generating {target} problems for category: {category} ({num_batches} batches)")

    all_problems = []
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)

    async with httpx.AsyncClient() as client:

        async def bounded_generate(batch_num):
            async with semaphore:
                return await generate_batch(client, category, batch_size, batch_num)

        tasks = [bounded_generate(i) for i in range(num_batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_problems.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"  [{category}] batch exception: {result}")

    # Trim to target count
    all_problems = all_problems[:target]
    logger.info(f"  [{category}] total: {len(all_problems)} problems")
    return all_problems


def deduplicate(problems: list[dict], threshold: float = 0.92) -> list[dict]:
    """
    Remove near-duplicate problems based on text similarity.

    Uses a simple Jaccard similarity on word sets as a fast approximation.
    For production, use sentence embeddings with cosine similarity.
    """
    logger.info(f"Deduplicating {len(problems)} problems (threshold={threshold})...")

    def word_set(text: str) -> set:
        return set(text.lower().split())

    unique = []
    seen_sets = []

    for p in problems:
        ws = word_set(p["problem"])
        is_dup = False
        for seen in seen_sets:
            if not ws or not seen:
                continue
            jaccard = len(ws & seen) / len(ws | seen)
            if jaccard > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
            seen_sets.append(ws)

    removed = len(problems) - len(unique)
    logger.info(f"  Removed {removed} duplicates, {len(unique)} remain")
    return unique


async def main():
    logger.info("=" * 60)
    logger.info("Seed Problem Generation Pipeline")
    logger.info("=" * 60)

    # Verify teacher is available
    async with httpx.AsyncClient() as client:
        try:
            url = f"{config.VLLM_BASE_URL}/models"
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            logger.info(f"Teacher model available on port {config.TEACHER_PORT}")
        except Exception as e:
            logger.error(f"Teacher model not available on port {config.TEACHER_PORT}: {e}")
            logger.error("Start vLLM with the teacher model before running this script.")
            sys.exit(1)

    all_problems = []

    for category in CATEGORIES:
        problems = await generate_category(category)
        all_problems.extend(problems)

    # Deduplicate
    all_problems = deduplicate(all_problems)

    # Re-assign unique IDs after dedup
    for i, p in enumerate(all_problems):
        p["problem_id"] = f"{p.get('category', 'unknown')}_{i:05d}"

    # Write output
    output_path = config.SEEDS_DIR / "enterprise_seeds.jsonl"
    with open(output_path, "w") as f:
        for p in all_problems:
            f.write(json.dumps(p) + "\n")

    logger.info("=" * 60)
    logger.info(f"Generated {len(all_problems)} unique enterprise seed problems")
    logger.info(f"Output: {output_path}")

    # Per-category breakdown
    from collections import Counter
    cats = Counter(p.get("category") for p in all_problems)
    for cat, count in sorted(cats.items()):
        logger.info(f"  {cat}: {count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
