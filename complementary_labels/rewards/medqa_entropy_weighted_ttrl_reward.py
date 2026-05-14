#!/usr/bin/env python3
"""Entropy-weighted TTRL reward for MedQA.

For each prompt UID, this reward extracts answer letters from all rollouts,
chooses a majority-vote pseudo label when the top count reaches
``min_vote_count``, and rewards responses that match that pseudo label.

The positive pseudo-label reward is weighted by
``lambda = sigmoid(beta * (1 - H / (H_max + eps)))`` where ``H`` is the
token-level entropy mean of responses that vote for the pseudo label and
``H_max`` is the maximum response token-level entropy mean in the rollout
group.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from collections import Counter, defaultdict
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

try:
    from complementary_labels.rewards.medqa_reward import (
        OPTION_LETTER_RE,
        extract_prediction_letter_from_generation,
        strict_format_match,
    )
except ModuleNotFoundError:
    from medqa_reward import OPTION_LETTER_RE, extract_prediction_letter_from_generation, strict_format_match


def select_pseudo_label(
    preds: list[str | None],
    min_vote_count: int,
    tie_break_key: str = "",
) -> tuple[str | None, int]:
    """Return a majority pseudo label, randomly breaking ties reproducibly."""
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


def token_entropy_weight(
    pred_entropies: list[float],
    pred_token_counts: list[float],
    group_entropies: list[float],
    beta: float,
    eps: float,
) -> tuple[float, float, float]:
    """Return pseudo-label token entropy, group max entropy, and confidence weight."""
    token_count_sum = sum(max(count, 0.0) for count in pred_token_counts)
    if pred_entropies and token_count_sum > 0:
        entropy = sum(ent * max(count, 0.0) for ent, count in zip(pred_entropies, pred_token_counts, strict=True))
        entropy /= token_count_sum
    elif pred_entropies:
        entropy = sum(pred_entropies) / len(pred_entropies)
    else:
        entropy = 0.0

    h_max = max(group_entropies) if group_entropies else 0.0
    confidence = 1.0 - entropy / (h_max + eps)
    confidence = max(0.0, min(1.0, confidence))
    weight = 1.0 / (1.0 + math.exp(-beta * confidence))
    return entropy, h_max, weight


class MedQAEntropyWeightedTTRLRewardManager(RewardManagerBase):
    """Batch reward manager for entropy-weighted pseudo-label rewards."""

    requires_batch_context = True

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score or default_compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        reward_kwargs = config.reward.get("custom_reward_function", {}).get("reward_kwargs", {})
        self.min_vote_count = int(reward_kwargs.get("min_vote_count", 2))
        self.require_format = bool(reward_kwargs.get("require_format", True))
        self.tie_break_seed = int(reward_kwargs.get("tie_break_seed", 42))
        self.beta = float(reward_kwargs.get("beta", 2.0))
        self.eps = float(reward_kwargs.get("eps", 1e-8))

    async def run_single(self, data: DataProto) -> dict:
        outputs = await self.run_batch(data[-1:])
        return outputs[0]

    async def run_batch(self, data: DataProto) -> list[dict]:
        response_strs = await self._decode_responses(data)
        preds = []
        format_ok = []
        golds = []
        response_entropies = self._get_float_non_tensor(data, "response_token_entropy")
        response_token_counts = self._get_float_non_tensor(data, "response_token_count", default=1.0)

        for idx, response_str in enumerate(response_strs):
            data_item = data[idx]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            options = extra_info.get("options") or {}
            options = {str(key).upper(): str(value) for key, value in options.items()}

            pred = extract_prediction_letter_from_generation(response_str, options)
            preds.append(pred)
            format_ok.append(strict_format_match(response_str) is not None)

            reward_model = data_item.non_tensor_batch.get("reward_model", {})
            gold = str(reward_model.get("ground_truth") or extra_info.get("answer_idx") or "").strip().upper()
            golds.append(gold if OPTION_LETTER_RE.match(gold) else None)

        uid_to_indices: dict[str, list[int]] = defaultdict(list)
        uids = data.non_tensor_batch.get("uid", [str(i) for i in range(len(data))])
        for idx, uid in enumerate(uids):
            uid_to_indices[str(uid)].append(idx)

        pseudo_labels: dict[str, str | None] = {}
        pseudo_counts: dict[str, int] = {}
        token_entropies: dict[str, float] = {}
        token_h_maxes: dict[str, float] = {}
        pseudo_weights: dict[str, float] = {}
        for uid, indices in uid_to_indices.items():
            pseudo_label, pseudo_count = select_pseudo_label(
                [preds[idx] for idx in indices],
                min_vote_count=self.min_vote_count,
                tie_break_key=f"{self.tie_break_seed}:{uid}",
            )
            pseudo_indices = [idx for idx in indices if pseudo_label is not None and preds[idx] == pseudo_label]
            entropy, h_max, weight = token_entropy_weight(
                [response_entropies[idx] for idx in pseudo_indices],
                [response_token_counts[idx] for idx in pseudo_indices],
                [response_entropies[idx] for idx in indices],
                beta=self.beta,
                eps=self.eps,
            )
            pseudo_labels[uid] = pseudo_label
            pseudo_counts[uid] = pseudo_count
            token_entropies[uid] = entropy
            token_h_maxes[uid] = h_max
            pseudo_weights[uid] = weight

        outputs = []
        for idx, pred in enumerate(preds):
            uid = str(uids[idx])
            pseudo_label = pseudo_labels[uid]
            has_pseudo_label = pseudo_label is not None
            is_format_ok = format_ok[idx]
            answer_matches_pseudo = has_pseudo_label and pred == pseudo_label
            positive = answer_matches_pseudo and (is_format_ok or not self.require_format)
            score = pseudo_weights[uid] if positive else 0.0

            gold = golds[idx]
            answer_matches_gold = gold is not None and pred == gold
            strict_answer_matches_gold = answer_matches_gold and is_format_ok
            pseudo_matches_gold = has_pseudo_label and gold is not None and pseudo_label == gold

            outputs.append(
                {
                    "reward_score": score,
                    "reward_extra_info": {
                        "score": score,
                        "acc": float(strict_answer_matches_gold),
                        "format": float(is_format_ok),
                        "answer_acc": float(answer_matches_gold),
                        "pseudo_answer_acc": float(answer_matches_pseudo),
                        "pseudo_label_available": float(has_pseudo_label),
                        "pseudo_label_count": float(pseudo_counts[uid]),
                        "pseudo_label_matches_gold": float(pseudo_matches_gold),
                        "answer_matches_gold": float(answer_matches_gold),
                        "answer_valid": float(pred is not None),
                        "pseudo_weight": float(pseudo_weights[uid]),
                        "pseudo_token_entropy": float(token_entropies[uid]),
                        "token_entropy_max": float(token_h_maxes[uid]),
                    },
                }
            )

        return outputs

    def _get_float_non_tensor(self, data: DataProto, key: str, default: float = 0.0) -> list[float]:
        values = data.non_tensor_batch.get(key)
        if values is None:
            return [default] * len(data)
        raw_values = values.tolist() if hasattr(values, "tolist") else values
        return [float(value) for value in raw_values]

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
    raise RuntimeError("Use MedQAEntropyWeightedTTRLRewardManager.run_batch for entropy-weighted TTRL rewards.")
