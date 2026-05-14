#!/usr/bin/env python3
"""Prepare full-option MedQA data for verl RL training/evaluation.

The raw dataset is expected to contain JSONL files with these fields:
question, options, answer, answer_idx, and meta_info. The output follows the
standard verl RL dataset schema used by RLHFDataset:

  data_source, prompt, ability, reward_model, extra_info

Parquet files are the training/evaluation inputs for verl. JSONL files and
example JSON files are also written for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RAW_DIR = Path("/data/common/datasets/MedQA_Oringin")
DEFAULT_OUTPUT_DIR = Path("~/data/complementary_labels/medqa_full_options")

PROMPT_PREFIX = (
    "A conversation between User and Assistant. The user asks a question, and the assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>.\n"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return records


def format_options(options: dict[str, str]) -> str:
    if not isinstance(options, dict) or not options:
        raise ValueError(f"Expected non-empty options dict, got: {options!r}")

    lines = []
    for key in sorted(options):
        value = str(options[key]).strip()
        lines.append(f"{key}. {value}")
    return "\n".join(lines)


def build_question_with_options(example: dict[str, Any]) -> str:
    question = str(example["question"]).strip()
    options = format_options(example["options"])
    return f"{question}\n\nOptions:\n{options}"


def build_prompt_text(question_with_options: str) -> str:
    return f"{PROMPT_PREFIX}User: {question_with_options} Assistant:"


def validate_example(example: dict[str, Any], source_path: Path, index: int) -> None:
    required = ["question", "options", "answer", "answer_idx"]
    missing = [key for key in required if key not in example]
    if missing:
        raise ValueError(f"{source_path} row {index} missing fields: {missing}")

    answer_idx = str(example["answer_idx"]).strip()
    options = example["options"]
    if answer_idx not in options:
        raise ValueError(
            f"{source_path} row {index} has answer_idx={answer_idx!r}, "
            f"but options keys are {sorted(options)}"
        )


def convert_example(example: dict[str, Any], split: str, index: int, data_source: str) -> dict[str, Any]:
    question_with_options = build_question_with_options(example)
    prompt_text = build_prompt_text(question_with_options)
    answer_idx = str(example["answer_idx"]).strip()
    answer_text = str(example["answer"]).strip()

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "medical_qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer_idx,
            "ground_truth_text": answer_text,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "question": str(example["question"]).strip(),
            "question_with_options": question_with_options,
            "prompt_text": prompt_text,
            "options": {str(k): str(v) for k, v in example["options"].items()},
            "answer_idx": answer_idx,
            "answer": answer_text,
            "meta_info": example.get("meta_info"),
        },
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_split(raw_dir: Path, output_dir: Path, split: str, data_source: str) -> int:
    source_path = raw_dir / f"{split}.jsonl"
    raw_records = load_jsonl(source_path)
    processed = []

    for idx, example in enumerate(raw_records):
        validate_example(example, source_path, idx)
        processed.append(convert_example(example, split=split, index=idx, data_source=data_source))

    df = pd.DataFrame(processed)
    df.to_parquet(output_dir / f"{split}.parquet", index=False)
    write_jsonl(processed, output_dir / f"{split}.jsonl")

    if processed:
        with (output_dir / f"{split}_example.json").open("w", encoding="utf-8") as f:
            json.dump(processed[0], f, ensure_ascii=False, indent=2)

    return len(processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare full-option MedQA JSONL data for verl.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing train/dev/test.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for processed outputs.")
    parser.add_argument("--data-source", default="medqa_full_options", help="Value for the verl data_source column.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Dataset splits to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in args.splits:
        counts[split] = process_split(raw_dir, output_dir, split, args.data_source)

    summary = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "data_source": args.data_source,
        "splits": counts,
        "format": {
            "train_files": str(output_dir / "train.parquet"),
            "val_files": str(output_dir / "test.parquet"),
            "dev_files": str(output_dir / "dev.parquet"),
            "prompt_key": "prompt",
            "ground_truth": "reward_model.ground_truth stores the answer option letter.",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
