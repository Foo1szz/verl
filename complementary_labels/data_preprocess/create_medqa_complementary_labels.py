#!/usr/bin/env python3
"""Create a MedQA complementary-label train set.

Each row keeps the original question, options, and correct answer, then adds
one randomly sampled incorrect option as the complementary label. The output
keeps the verl dataset schema so it can be used by later training code with a
custom reward function.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("~/data/complementary_labels/medqa_full_options/train.parquet")
DEFAULT_OUTPUT_DIR = Path("~/data/complementary_labels/medqa_full_options_complementary_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MedQA complementary-label data from the full train set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input full train parquet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for complementary-label sampling.")
    parser.add_argument("--data-source", default="medqa_full_options_complementary_labels")
    return parser.parse_args()


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


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def get_options(record: dict[str, Any], row_idx: int) -> dict[str, str]:
    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError(f"Row {row_idx} has no extra_info dict.")

    options = extra_info.get("options")
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError(f"Row {row_idx} needs at least two options, got {options!r}.")

    return {str(key).strip(): str(value).strip() for key, value in options.items()}


def get_answer_idx(record: dict[str, Any], row_idx: int) -> str:
    reward_model = record.get("reward_model")
    extra_info = record.get("extra_info")
    answer_idx = None

    if isinstance(reward_model, dict):
        answer_idx = reward_model.get("ground_truth")
    if answer_idx is None and isinstance(extra_info, dict):
        answer_idx = extra_info.get("answer_idx")
    if answer_idx is None:
        raise ValueError(f"Row {row_idx} has no answer index in reward_model or extra_info.")

    return str(answer_idx).strip()


def add_complementary_label(
    record: dict[str, Any],
    row_idx: int,
    rng: np.random.Generator,
    seed: int,
    data_source: str,
) -> tuple[dict[str, Any], str]:
    record = to_jsonable(record)
    options = get_options(record, row_idx)
    answer_idx = get_answer_idx(record, row_idx)
    if answer_idx not in options:
        raise ValueError(f"Row {row_idx} answer_idx={answer_idx!r} is not in options {sorted(options)}.")

    incorrect_labels = sorted(label for label in options if label != answer_idx)
    complementary_label = str(rng.choice(np.array(incorrect_labels, dtype=object)))
    complementary_label_text = options[complementary_label]

    original_data_source = record.get("data_source")
    record["data_source"] = data_source

    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict):
        reward_model = {}
    reward_model["complementary_label"] = complementary_label
    reward_model["complementary_label_text"] = complementary_label_text
    reward_model["complementary_label_is_wrong"] = True
    record["reward_model"] = reward_model

    extra_info = record.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}
    extra_info["original_data_source"] = original_data_source
    extra_info["complementary_label"] = complementary_label
    extra_info["complementary_label_text"] = complementary_label_text
    extra_info["complementary_label_is_wrong"] = True
    extra_info["complementary_label_seed"] = seed
    extra_info["incorrect_options"] = incorrect_labels
    record["extra_info"] = extra_info

    return record, complementary_label


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    rng = np.random.default_rng(args.seed)

    records: list[dict[str, Any]] = []
    complementary_counts: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    for row_idx, raw_record in enumerate(df.to_dict(orient="records")):
        record, complementary_label = add_complementary_label(
            raw_record,
            row_idx=row_idx,
            rng=rng,
            seed=args.seed,
            data_source=args.data_source,
        )
        records.append(record)
        complementary_counts[complementary_label] += 1
        answer_counts[get_answer_idx(record, row_idx)] += 1

    out_df = pd.DataFrame(records, columns=df.columns)
    parquet_path = output_dir / "train.parquet"
    jsonl_path = output_dir / "train.jsonl"
    out_df.to_parquet(parquet_path, index=False)
    write_jsonl(records, jsonl_path)

    if records:
        with (output_dir / "train_example.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(records[0]), f, ensure_ascii=False, indent=2)

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "data_source": args.data_source,
        "num_rows": len(records),
        "train_files": str(parquet_path),
        "jsonl": str(jsonl_path),
        "complementary_label_counts": dict(sorted(complementary_counts.items())),
        "answer_counts": dict(sorted(answer_counts.items())),
        "schema_note": (
            "reward_model.ground_truth keeps the correct answer; "
            "reward_model.complementary_label stores one sampled incorrect option."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
