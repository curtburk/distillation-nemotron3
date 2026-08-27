#!/usr/bin/env python3
"""
eval_generate.py — Generate HumanEval+ completions from a vLLM-served model.

Assumes a vLLM server is already running on --port serving --model.
Serve ONE model at a time (never two vLLM instances on the same GPU).

Usage:
    python3 eval_generate.py --model-name stock  --port 8091 \
        --served-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

    python3 eval_generate.py --model-name distilled --port 8091 \
        --served-model /models/nemotron3-nano-kd-distilled

Output: results/completions_<model-name>.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

# Same system prompt the distillation corpus was built with.
SYSTEM_PROMPT = Path(
    "/home/gb300/pmm-demos/knowledge-distillation/prompts/trace_generation.txt"
).read_text()

HUMANEVAL_PATH = Path(
    "/home/gb300/pmm-demos/knowledge-distillation/data/raw/humaneval_plus.jsonl"
)
RESULTS_DIR = Path(__file__).parent / "results"

MAX_CONCURRENT = 8
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 900


def make_client(base_url: str) -> httpx.AsyncClient:
    """Fresh connection per request — vLLM drops keepalive under async load."""
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=30.0),
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=MAX_CONCURRENT),
        http2=False,
    )


def build_messages(problem: dict) -> list[dict]:
    user = problem["problem"]
    sig = problem.get("function_signature", "")
    if sig:
        user += f"\n\nExpected function signature: `{sig}`"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def generate_one(
    client: httpx.AsyncClient,
    served_model: str,
    problem: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": served_model,
                    "messages": build_messages(problem),
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": MAX_TOKENS,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            return {
                "problem_id": problem["problem_id"],
                "entry_point": problem.get("entry_point", ""),
                "test_code": problem.get("test_code", ""),
                "completion": msg.get("content") or "",
                "reasoning": msg.get("reasoning") or "",
                "finish_reason": data["choices"][0].get("finish_reason"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            }
        except Exception as e:
            print(f"  ERROR {problem['problem_id']}: {e}", flush=True)
            return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True, help="Label for output file, e.g. stock")
    ap.add_argument("--served-model", required=True, help="Model id/path as served by vLLM")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--limit", type=int, default=None, help="Cap problems (for smoke tests)")
    args = ap.parse_args()

    problems = []
    with open(HUMANEVAL_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if args.limit:
        problems = problems[: args.limit]

    print(f"Problems: {len(problems)}", flush=True)
    print(f"Model:    {args.served_model}", flush=True)

    base_url = f"http://localhost:{args.port}"

    async with make_client(base_url) as client:
        resp = await client.get("/v1/models", timeout=15)
        resp.raise_for_status()
        print(f"Server up: {[m['id'] for m in resp.json()['data']]}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"completions_{args.model_name}.jsonl"

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    done = 0
    failed = 0

    async with make_client(base_url) as client:
        with open(out_path, "w") as out:
            batch_size = 20
            for start in range(0, len(problems), batch_size):
                batch = problems[start : start + batch_size]
                tasks = [
                    generate_one(client, args.served_model, p, semaphore) for p in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, dict):
                        out.write(json.dumps(r) + "\n")
                        done += 1
                    else:
                        failed += 1
                out.flush()
                print(
                    f"  {done + failed}/{len(problems)} | ok {done} | failed {failed}",
                    flush=True,
                )

    print(f"Wrote {done} completions to {out_path}", flush=True)
    if failed:
        print(f"WARNING: {failed} generations failed", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())