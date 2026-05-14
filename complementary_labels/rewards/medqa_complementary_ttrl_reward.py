#!/usr/bin/env python3
"""TTRL reward with complementary-label fallback for MedQA."""

from __future__ import annotations

import asyncio
import hashlib
import random
from collections import Counter, defaultdict
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

try:
    from complementary_labels.rewards.medqa_complementary_reward import get_complementary_label
    from complementary_labels.rewards.medqa_reward import (
        OPTION_LETTER_RE,
        extract_prediction_letter_from_generation,
        strict_format_match,
    )
except ModuleNotFoundError:
    from medqa_complementary_reward import get_complementary_label
    from medqa_reward import OPTION_LETTER_RE, extract_prediction_letter_from_generation, strict_format_match


def select_pseudo_label(
    preds: list[str | None],
    min_vote_count: int,
    tie_break_key: str = "",
) -> tuple[str | None, int]:
    vote_counts = Counter(pred for pred in preds if pred is not None)
    if not vote_counts:
        return None, 0

    top_count = max(vote_counts.values())
    if top_count < min_vote_count:
        return None, top_count

    top_labels = sorted(label for label, count in vote_counts.items() if count == top_count)
    if len(top_labels) == 1:
        return top_labels[0], top_count

    seed = int(hashlib.sha256(tie_break_key.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed).choice(top_labels), top_count


class MedQAComplementaryTTRLRewardManager(RewardManagerBase):
    """Batch reward manager for TTRL with a negative complementary-label reward."""

    requires_batch_context = True

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score or default_compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        reward_kwargs = config.reward.get("custom_reward_function", {}).get("reward_kwargs", {})
        self.min_vote_count = int(reward_kwargs.get("min_vote_count", 2))
        self.tie_break_seed = int(reward_kwargs.get("tie_break_seed", 42))
        self.require_format_for_positive = bool(reward_kwargs.get("require_format_for_positive", True))

    async def run_single(self, data: DataProto) -> dict:
        outputs = await self.run_batch(data[-1:])
        return outputs[0]

    async def run_batch(self, data: DataProto) -> list[dict]:
        response_strs = await self._decode_responses(data)
        preds = []
        format_ok = []
        golds = []
        complementary_labels = []

        for idx, response_str in enumerate(response_strs):
            data_item = data[idx]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            options = extra_info.get("options") or {}
            options = {str(key).upper(): str(value) for key, value in options.items()}
            reward_model = data_item.non_tensor_batch.get("reward_model", {})

            pred = extract_prediction_letter_from_generation(response_str, options)
            preds.append(pred)
            format_ok.append(strict_format_match(response_str) is not None)

            gold = str(reward_model.get("ground_truth") or extra_info.get("answer_idx") or "").strip().upper()
            golds.append(gold if OPTION_LETTER_RE.match(gold) else None)
            complementary_labels.append(get_complementary_label(reward_model, extra_info))

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        uids = data.non_tensor_batch.get("uid", [str(i) for i in range(len(data))])
        for idx, uid in enumerate(uids):
            uid_to_indices[str(uid)].append(idx)

        pseudo_labels: dict[str, str | None] = {}
        pseudo_counts: dict[str, int] = {}
        pseudo_is_complementary: dict[str, bool] = {}
        for uid, indices in uid_to_indices.items():
            complementary_label = complementary_labels[indices[0]]
            pseudo_label, pseudo_count = select_pseudo_label(
                [preds[idx] for idx in indices],
                min_vote_count=self.min_vote_count,
                tie_break_key=f"{self.tie_break_seed}:{uid}",
            )
            is_complementary_pseudo = pseudo_label == complementary_label
            if is_complementary_pseudo:
                pseudo_label = None
            pseudo_labels[uid] = pseudo_label
            pseudo_counts[uid] = pseudo_count
            pseudo_is_complementary[uid] = is_complementary_pseudo

        outputs = []
        for idx, pred in enumerate(preds):
            uid = str(uids[idx])
            pseudo_label = pseudo_labels[uid]
            has_pseudo_label = pseudo_label is not None
            is_valid_format = format_ok[idx]

            complementary_hit = pred == complementary_labels[idx]
            answer_matches_pseudo = has_pseudo_label and pred == pseudo_label
            positive = answer_matches_pseudo and (is_valid_format or not self.require_format_for_positive)

            if complementary_hit:
                score = -1.0
            elif positive:
                score = 1.0
            else:
                score = 0.0

            gold = golds[idx]
            answer_matches_gold = gold is not None and pred == gold
            strict_answer_matches_gold = answer_matches_gold and is_valid_format
            pseudo_matches_gold = has_pseudo_label and gold is not None and pseudo_label == gold

            outputs.append(
                {
                    "reward_score": score,
                    "reward_extra_info": {
                        "score": score,
                        "acc": float(strict_answer_matches_gold),
                        "format": float(is_valid_format),
                        "answer_acc": float(answer_matches_gold),
                        "complementary_hit": float(complementary_hit),
                        "pseudo_label_available": float(has_pseudo_label),
                        "pseudo_label_is_complementary": float(pseudo_is_complementary[uid]),
                        "pseudo_label_matches_gold": float(pseudo_matches_gold),
                        "answer_valid": float(pred is not None),
                    },
                }
            )

        return outputs

    async def _decode_responses(self, data: DataProto) -> list[str]:
        prompt_length = data.batch["prompts"].size(1)
        response_ids = data.batch["responses"]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)

        def decode_all() -> list[str]:
            decoded = []
            for idx in range(len(data)):
                valid_len = int(valid_response_lengths[idx].item())
                valid_response_ids = response_ids[idx][:valid_len]
                decoded.append(self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))
            return decoded

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, decode_all)


def compute_score(**kwargs):
    del kwargs
    raise RuntimeError("Use MedQAComplementaryTTRLRewardManager.run_batch for complementary-label TTRL rewards.")
