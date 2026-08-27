#!/usr/bin/env python3
"""
06_assemble_dataset.py — Assemble the final SFT dataset.

Takes the best-scored traces and assembles them into the conversation format
required by Torchtune/LlamaFactory, with train/val/test/demo splits.

Input:  data/scored/best_scored_traces.jsonl
Output: data/final/train.jsonl
        data/final/validation.jsonl
        data/final/test.jsonl
        data/final/demo.jsonl
        data/final/dataset_stats.json
"""

import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("assemble_dataset")


def format_as_conversation(trace: dict) -> dict:
    """Convert a trace into the SFT conversation format."""
    return {
        "conversations": [
            {
                "role": "system",
                "content": config.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": trace["problem"],
            },
            {
                "role": "assistant",
                "content": trace["reasoning_trace"],
            },
        ],
        # Metadata (not used during training, useful for analysis)
        "metadata": {
            "problem_id": trace["problem_id"],
            "source": trace.get("source", "unknown"),
            "category": trace.get("category", "unknown"),
            "difficulty": trace.get("difficulty", "medium"),
            "quality_score": trace.get("quality_scores", {}).get("weighted_score", 0),
            "trace_id": trace.get("trace_id", ""),
        },
    }


def stratified_split(
    data: list[dict],
    splits: dict[str, float],
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Split data into subsets, stratified by source and difficulty.

    Ensures each split has proportional representation of categories.
    """
    random.seed(seed)

    # Group by stratification key
    groups = {}
    for item in data:
        meta = item.get("metadata", {})
        key = f"{meta.get('source', 'unknown')}_{meta.get('difficulty', 'medium')}"
        groups.setdefault(key, []).append(item)

    # Shuffle within each group
    for group in groups.values():
        random.shuffle(group)

    # Allocate from each group proportionally.
    # Rounding remainder goes to train — never silently dropped.
    result = {name: [] for name in splits}

    for group_items in groups.values():
        n = len(group_items)
        idx = 0
        for split_name, split_ratio in splits.items():
            if split_name == "train":
                continue  # train gets everything left over at the end
            count = round(n * split_ratio)
            result[split_name].extend(group_items[idx : idx + count])
            idx += count
        # Everything remaining (including rounding remainder) → train
        result["train"].extend(group_items[idx:])

    # Final shuffle within each split
    for split in result.values():
        random.shuffle(split)

    return result


def compute_stats(data: list[dict], split_name: str) -> dict:
    """Compute statistics for a dataset split."""
    sources = Counter()
    difficulties = Counter()
    categories = Counter()
    scores = []
    total_tokens = 0

    for item in data:
        meta = item.get("metadata", {})
        sources[meta.get("source", "unknown")] += 1
        difficulties[meta.get("difficulty", "unknown")] += 1
        categories[meta.get("category", "unknown")] += 1
        score = meta.get("quality_score", 0)
        if score:
            scores.append(score)

        # Rough token estimate (chars / 4)
        for msg in item.get("conversations", []):
            total_tokens += len(msg.get("content", "")) // 4

    return {
        "split": split_name,
        "count": len(data),
        "sources": dict(sources.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "categories": dict(categories.most_common()),
        "quality_score_avg": round(sum(scores) / len(scores), 3) if scores else 0,
        "quality_score_min": round(min(scores), 3) if scores else 0,
        "quality_score_max": round(max(scores), 3) if scores else 0,
        "est_total_tokens": total_tokens,
    }


def main():
    logger.info("=" * 60)
    logger.info("SFT Dataset Assembly")
    logger.info("=" * 60)

    # Load best scored traces
    input_path = config.SCORED_DIR / "best_scored_traces.jsonl"
    if not input_path.exists():
        logger.error("No scored traces found. Run 05_score_reasoning.py first.")
        sys.exit(1)

    raw_traces = []
    with open(input_path) as f:
        for line in f:
            raw_traces.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(raw_traces):,} best-scored traces")

    # Convert to conversation format
    conversations = [format_as_conversation(t) for t in raw_traces]
    logger.info(f"Formatted {len(conversations):,} conversations")

    # Split
    splits = stratified_split(conversations, config.DATASET_SPLITS)

    # Write splits
    all_stats = {}
    for split_name, split_data in splits.items():
        output_path = config.FINAL_DIR / f"{split_name}.jsonl"
        with open(output_path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

        stats = compute_stats(split_data, split_name)
        all_stats[split_name] = stats

        logger.info(f"  {split_name}: {len(split_data):,} examples → {output_path}")

    # Write combined stats
    stats_path = config.FINAL_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    # Also write a Torchtune-compatible config snippet
    torchtune_config = {
        "dataset": {
            "source": str(config.FINAL_DIR / "train.jsonl"),
            "message_transform": {
                "train_on_input": False,
                "column_map": {
                    "messages": "conversations",
                },
            },
            "split": "train",
        },
    }

    torchtune_path = config.FINAL_DIR / "torchtune_dataset_config.yaml"
    # Write as simple text (no yaml dependency needed)
    with open(torchtune_path, "w") as f:
        f.write("# Torchtune dataset configuration\n")
        f.write(f"# Generated from {len(conversations)} examples\n\n")
        f.write(f"dataset:\n")
        f.write(f"  source: {config.FINAL_DIR / 'train.jsonl'}\n")
        f.write(f"  message_transform:\n")
        f.write(f"    train_on_input: false\n")
        f.write(f"    column_map:\n")
        f.write(f"      messages: conversations\n")
        f.write(f"  split: train\n")

    # Summary report
    logger.info("=" * 60)
    logger.info("Dataset Assembly Complete")
    logger.info("=" * 60)

    total = sum(len(s) for s in splits.values())
    logger.info(f"  Total examples: {total:,}")
    for split_name, stats in all_stats.items():
        logger.info(
            f"  {split_name}: {stats['count']:,} examples, "
            f"avg quality={stats['quality_score_avg']:.2f}, "
            f"~{stats['est_total_tokens']:,} tokens"
        )

    logger.info(f"\n  Stats: {stats_path}")
    logger.info(f"  Torchtune config: {torchtune_path}")
    logger.info(f"  Output directory: {config.FINAL_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
