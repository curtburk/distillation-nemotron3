#!/usr/bin/env python3
"""
eval_holdout_generate.py — Generate completions on the held-out test split.

This is the distribution the model was actually trained on: 215 examples,
~85% APPS, never seen during training. HumanEval+ was the wrong instrument
for measuring reasoning distillation; this is the right one.

Problem text comes from data/final/test.jsonl (the user turn of each
conversation). Test cases are joined by problem_id from data/raw/*.jsonl,
since final/ carries only conversations + metadata.

Assumes a vLLM server is already running. Serve ONE model at a time.

Usage:
    python3 eval_holdout_generate.py --model-name stock --port 8091 \
        --served-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Output: results/holdout_<model-name>.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

KD_ROOT = Path("/home/gb300/pmm-demos/knowledge-distillation")
TEST_PATH = KD_ROOT / "data/final/test.jsonl"
RAW_DIR = KD_ROOT / "data/raw"
RESULTS_DIR = Path(__file__).parent / "results"

MAX_CONCURRENT = 8
MAX_TOKENS = 10240
TIMEOUT_SECONDS = 900


def load_test_lookup() -> dict[str, dict]:
    """problem_id -> {test_code, io_tests, entry_point} from the raw datasets."""
    lookup: dict[str, dict] = {}
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                lookup[r["problem_id"]] = {
                    "test_code": r.get("test_code", "") or "",
                    "io_tests": r.get("io_tests") or [],
                    "entry_point": r.get("entry_point", ""),
                }
    return lookup


def load_holdout(lookup: dict[str, dict]) -> list[dict]:
    """Pull system+user turns out of the conversation format, attach tests."""
    items = []
    skipped_no_tests = 0

    with open(TEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["metadata"]["problem_id"]

            system = ""
            user = ""
            for m in rec["conversations"]:
                if m["role"] == "system":
                    system = m["content"]
                elif m["role"] == "user":
                    user = m["content"]

            tests = lookup.get(pid)
            if tests is None:
                skipped_no_tests += 1
                continue
            if not tests["test_code"] and not tests["io_tests"]:
                skipped_no_tests += 1
                continue

            items.append({
                "problem_id": pid,
                "source": rec["metadata"]["source"],
                "difficulty": rec["metadata"]["difficulty"],
                "system": system,
                "user": user,
                "test_code": tests["test_code"],
                "io_tests": tests["io_tests"],
            })

    if skipped_no_tests:
        print(f"Skipped {skipped_no_tests} examples with no usable tests", flush=True)
    return items


def make_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=30.0),
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=MAX_CONCURRENT),
        http2=False,
    )


async def generate_one(
    client: httpx.AsyncClient,
    served_model: str,
    item: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": served_model,
                    "messages": [
                        {"role": "system", "content": item["system"]},
                        {"role": "user", "content": item["user"]},
                    ],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": MAX_TOKENS,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            return {
                "problem_id": item["problem_id"],
                "source": item["source"],
                "difficulty": item["difficulty"],
                "test_code": item["test_code"],
                "io_tests": item["io_tests"],
                "completion": msg.get("content") or "",
                "finish_reason": data["choices"][0].get("finish_reason"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
        except Exception as e:
            print(f"  ERROR {item['problem_id']}: {e}", flush=True)
            return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    lookup = load_test_lookup()
    items = load_holdout(lookup)
    if args.limit:
        items = items[: args.limit]

    by_source: dict[str, int] = {}
    for it in items:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1

    print(f"Held-out problems: {len(items)}", flush=True)
    print(f"By source: {by_source}", flush=True)
    print(f"Model: {args.served_model}", flush=True)

    base_url = f"http://localhost:{args.port}"
    async with make_client(base_url) as client:
        resp = await client.get("/v1/models", timeout=15)
        resp.raise_for_status()
        print(f"Server up: {[m['id'] for m in resp.json()['data']]}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"holdout_{args.model_name}.jsonl"

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    done = failed = 0

    async with make_client(base_url) as client:
        with open(out_path, "w") as out:
            batch_size = 20
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                tasks = [generate_one(client, args.served_model, it, semaphore) for it in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, dict):
                        out.write(json.dumps(r) + "\n")
                        done += 1
                    else:
                        failed += 1
                out.flush()
                print(f"  {done + failed}/{len(items)} | ok {done} | failed {failed}", flush=True)

    print(f"Wrote {done} completions to {out_path}", flush=True)
    if failed:
        print(f"WARNING: {failed} generations failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())