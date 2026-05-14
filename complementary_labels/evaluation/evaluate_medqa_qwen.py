#!/usr/bin/env python3
"""Evaluate a local Qwen model on the processed full-option MedQA test set."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_DATA_FILE = Path("~/data/complementary_labels/medqa_full_options/test.parquet")
DEFAULT_MODEL_PATH = Path("/data/common/LLMs/Qwen2.5-3B")
DEFAULT_OUTPUT_DIR = Path("complementary_labels/results/qwen2_5_3b_medqa_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy-evaluate Qwen2.5-3B on processed MedQA test data.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--truncation-side",
        choices=["left", "right"],
        default="left",
        help="Use left truncation by default so the final options and Assistant suffix are preserved.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = args.truncation_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = args.dtype
    if dtype == "auto":
        torch_dtype: str | torch.dtype = "auto"
    else:
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()
    return model, tokenizer


def get_nested_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
    return value if isinstance(value, dict) else {}


def get_prompt(row: dict[str, Any]) -> str:
    extra_info = get_nested_dict(row.get("extra_info"))
    if isinstance(extra_info.get("prompt_text"), str):
        return extra_info["prompt_text"]

    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and isinstance(first.get("content"), str):
            return first["content"]

    raise ValueError("Could not find prompt text in row.")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_answer_region(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_prediction_letter(generation: str, options: dict[str, str]) -> str | None:
    answer_region = extract_answer_region(generation)
    search_spaces = [answer_region, generation]

    patterns = [
        r"(?:^|\b)(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-E])\)?\b",
        r"(?:^|\b)([A-E])\s*[\.\)]",
        r"^\s*([A-E])\s*$",
    ]
    for text in search_spaces:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()

    normalized_answer = normalize_text(answer_region)
    for key, option_text in sorted(options.items()):
        normalized_option = normalize_text(str(option_text))
        if normalized_option and normalized_option in normalized_answer:
            return str(key).upper()

    return None


def get_ground_truth(row: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    reward_model = get_nested_dict(row.get("reward_model"))
    extra_info = get_nested_dict(row.get("extra_info"))

    gold = str(reward_model.get("ground_truth") or extra_info.get("answer_idx")).strip().upper()
    answer_text = str(reward_model.get("ground_truth_text") or extra_info.get("answer") or "").strip()
    options = extra_info.get("options") or {}
    options = {str(k): str(v) for k, v in options.items()}
    return gold, answer_text, options


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.data_file = args.data_file.expanduser()
    args.model_path = args.model_path.expanduser()
    args.output_dir = args.output_dir.expanduser()
    df = pd.read_parquet(args.data_file)
    if args.limit is not None:
        df = df.head(args.limit)
    rows = df.to_dict(orient="records")

    run_config = {
        "model_path": str(args.model_path.resolve()),
        "data_file": str(args.data_file.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": args.device,
        "decoding": "greedy",
        "do_sample": False,
        "num_beams": 1,
        "truncation_side": args.truncation_side,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "run_config.json", run_config)

    model, tokenizer = load_model_and_tokenizer(args)

    predictions_path = args.output_dir / "predictions.jsonl"
    correct = 0
    total = 0

    with predictions_path.open("w", encoding="utf-8") as out:
        for start, batch_rows in tqdm(list(batched(rows, args.batch_size)), desc="Evaluating"):
            prompts = [get_prompt(row) for row in batch_rows]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            ).to(args.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            for offset, (row, prompt, generation, input_length) in enumerate(
                zip(batch_rows, prompts, generations, input_lengths, strict=True)
            ):
                gold, answer_text, options = get_ground_truth(row)
                pred = extract_prediction_letter(generation, options)
                is_correct = pred == gold
                correct += int(is_correct)
                total += 1

                extra_info = get_nested_dict(row.get("extra_info"))
                record = {
                    "index": int(extra_info.get("index", start + offset)),
                    "split": extra_info.get("split", "test"),
                    "prompt": prompt,
                    "generation": generation,
                    "prediction": pred,
                    "gold": gold,
                    "gold_text": answer_text,
                    "is_correct": is_correct,
                    "input_tokens": int(input_length),
                    "generated_tokens": int(len(generated_ids[offset])),
                    "options": options,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "predictions_file": str(predictions_path.resolve()),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
