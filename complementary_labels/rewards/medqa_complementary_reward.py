#!/usr/bin/env python3
"""Complementary-label reward for MedQA GRPO training."""

from __future__ import annotations

from typing import Any

try:
    from complementary_labels.rewards.medqa_reward import (
        OPTION_LETTER_RE,
        extract_prediction_letter_from_generation,
        strict_format_match,
    )
except ModuleNotFoundError:
    from medqa_reward import OPTION_LETTER_RE, extract_prediction_letter_from_generation, strict_format_match


def get_complementary_label(ground_truth: Any, extra_info: dict[str, Any]) -> str:
    if isinstance(ground_truth, dict):
        value = ground_truth.get("complementary_label")
        if value is not None:
            return str(value).strip().upper()
    value = extra_info.get("complementary_label")
    if value is not None:
        return str(value).strip().upper()
    return ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return -1 when the response chooses the complementary label, otherwise 0."""
    del data_source

    extra_info = extra_info or {}
    options = extra_info.get("options") or {}
    options = {str(key).upper(): str(value) for key, value in options.items()}

    complementary_label = get_complementary_label(ground_truth, extra_info)
    pred = extract_prediction_letter_from_generation(solution_str, options)
    is_valid_format = strict_format_match(solution_str) is not None
    complementary_hit = bool(OPTION_LETTER_RE.match(complementary_label)) and pred == complementary_label

    if isinstance(ground_truth, dict):
        gold = str(ground_truth.get("ground_truth") or extra_info.get("answer_idx") or "").strip().upper()
    else:
        gold = str(ground_truth or extra_info.get("answer_idx") or "").strip().upper()
    answer_matches_gold = bool(OPTION_LETTER_RE.match(gold)) and pred == gold
    strict_answer_matches_gold = answer_matches_gold and is_valid_format

    score = -1.0 if complementary_hit else 0.0
    return {
        "score": score,
        "acc": float(strict_answer_matches_gold),
        "format": float(is_valid_format),
        "answer_acc": float(answer_matches_gold),
        "complementary_hit": float(complementary_hit),
        "answer_valid": float(pred is not None),
    }
