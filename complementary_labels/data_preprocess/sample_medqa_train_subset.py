#!/usr/bin/env python3
"""Randomly sample a reproducible MedQA train subset for verl RL training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("~/data/complementary_labels/medqa_full_options/train.parquet")
DEFAULT_OUTPUT_DIR = Path("~/data/complementary_labels/medqa_full_options_train_5k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a MedQA train subset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input train parquet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if args.num_samples > len(df):
        raise ValueError(f"Requested {args.num_samples} samples from only {len(df)} rows.")

    subset = df.sample(n=args.num_samples, random_state=args.seed, replace=False).reset_index(drop=True)

    # Keep the original extra_info.index from the full train split and add a
    # deterministic subset index for easier debugging of the sampled dataset.
    records = subset.to_dict(orient="records")
    for subset_idx, record in enumerate(records):
        extra_info = record.get("extra_info")
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info["subset"] = f"train_{args.num_samples}"
        extra_info["subset_index"] = subset_idx
        extra_info["sample_seed"] = args.seed
        record["extra_info"] = extra_info

    subset = pd.DataFrame(records, columns=df.columns)
    parquet_path = output_dir / "train.parquet"
    jsonl_path = output_dir / "train.jsonl"
    subset.to_parquet(parquet_path, index=False)
    write_jsonl(records, jsonl_path)

    if records:
        with (output_dir / "train_example.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(records[0]), f, ensure_ascii=False, indent=2)

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_input_rows": len(df),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "train_files": str(parquet_path),
        "prompt_key": "prompt",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
