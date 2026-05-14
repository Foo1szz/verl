#!/usr/bin/env python3
"""Rule-based MedQA reward for verl GRPO training."""

from __future__ import annotations

import re
from typing import Any


OPTION_LETTER_RE = re.compile(r"^[A-E]$")
STRICT_FORMAT_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*<answer>\s*(?P<answer>.*?)\s*</answer>\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
ANSWER_REGION_RE = re.compile(r"<answer>\s*(?P<answer>.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strict_format_match(solution_str: str) -> re.Match[str] | None:
    match = STRICT_FORMAT_RE.search(solution_str or "")
    if not match:
        return None
    if not match.group("think").strip() or not match.group("answer").strip():
        return None
    return match


def extract_answer_region(solution_str: str) -> str | None:
    match = ANSWER_REGION_RE.search(solution_str or "")
    if match:
        return match.group("answer").strip()
    return None


def extract_prediction_letter_from_generation(generation: str, options: dict[str, str] | None = None) -> str | None:
    answer_region = extract_answer_region(generation) or generation
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
    for key, option_text in sorted((options or {}).items()):
        normalized_option = normalize_text(str(option_text))
        if normalized_option and normalized_option in normalized_answer:
            return str(key).upper()

    return None


def extract_prediction_letter(answer_region: str, options: dict[str, str] | None = None) -> str | None:
    patterns = [
        r"(?:^|\b)(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-E])\)?\b",
        r"(?:^|\b)([A-E])\s*[\.\)]",
        r"^\s*([A-E])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer_region, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    normalized_answer = normalize_text(answer_region)
    for key, option_text in sorted((options or {}).items()):
        normalized_option = normalize_text(str(option_text))
        if normalized_option and normalized_option in normalized_answer:
            return str(key).upper()

    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return 1 only when both strict format and answer are correct."""
    del data_source

    extra_info = extra_info or {}
    gold = str(ground_truth or extra_info.get("answer_idx", "")).strip().upper()
    options = extra_info.get("options") or {}
    options = {str(k).upper(): str(v) for k, v in options.items()}

    pred = extract_prediction_letter_from_generation(solution_str, options)
    is_correct = bool(OPTION_LETTER_RE.match(gold)) and pred == gold

    format_match = strict_format_match(solution_str)
    is_valid_format = format_match is not None
    if not is_valid_format:
        return {
            "score": 0.0,
            "acc": 0.0,
            "format": 0.0,
            "answer_acc": float(is_correct),
        }

    score = 1.0 if is_correct else 0.0

    return {
        "score": score,
        "acc": score,
        "format": 1.0,
        "answer_acc": float(is_correct),
    }
