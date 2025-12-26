#!/usr/bin/env python3
"""
Label a generated Parquet dataset with in-context critic predictions.

For each prompt_idx group and each labeled rollout in that group:
  - Build a packed in-context sequence with the target rollout LAST.
  - Context consists of as many other rollouts as fit: shuffle siblings, then append
    (assistant rollout + user feedback) blocks until the next would exceed max_length,
    while reserving space for the target rollout.
  - Run the critic once per labeled rollout.
  - Extract:
      - critic_p_correct per output token (marginalize over length bins)
      - critic_expected_tokens_remaining per output token (marginalize over reward bins)

Output Parquet preserves all input columns and appends:
  - critic_model_id
  - critic_p_correct
  - critic_expected_tokens_remaining

Unlabeled rows are kept with null critic columns.

Data-parallel mode splits by parquet row-group ranges with minimal overlap.
This assumes prompt_idx groups are contiguous in the parquet.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Qwen control tokens (must match training) ---
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
SYSTEM_TOKEN_ID = 8948
NEWLINE_TOKEN_ID = 198

NEW_COLUMNS = (
    ("critic_model_id", pa.string()),
    ("critic_p_correct", pa.list_(pa.float32())),
    ("critic_expected_tokens_remaining", pa.list_(pa.float32())),
)

# Additional columns for candidate Q-value evaluation
CANDIDATE_Q_COLUMNS = (
    ("candidate_q_values", pa.list_(pa.list_(pa.float32()))),  # [num_positions][num_candidates]
)


# ---------------------------------------------------------------------------
# KV Cache Helpers
# ---------------------------------------------------------------------------

def _slice_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    end: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Slice KV cache to [0:end] along the sequence dimension.

    Args:
        past_key_values: Tuple of (key, value) pairs, one per layer.
            Each key/value has shape [batch, num_heads, seq_len, head_dim].
        end: End index for slicing (exclusive).

    Returns:
        Sliced KV cache with seq_len = end.
    """
    if end <= 0:
        # Return empty cache structure
        return tuple(
            (k[:, :, :0, :], v[:, :, :0, :])
            for k, v in past_key_values
        )
    return tuple(
        (k[:, :, :end, :], v[:, :, :end, :])
        for k, v in past_key_values
    )


def _expand_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    batch_size: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Expand KV cache batch dimension from 1 to batch_size via broadcasting.

    This uses expand() which creates a view without copying memory.

    Args:
        past_key_values: Tuple of (key, value) pairs with batch_size=1.
        batch_size: Target batch size.

    Returns:
        Expanded KV cache with the new batch size.
    """
    return tuple(
        (
            k.expand(batch_size, -1, -1, -1),
            v.expand(batch_size, -1, -1, -1),
        )
        for k, v in past_key_values
    )


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


@dataclass(frozen=True)
class Config:
    critic_model_id: str
    in_parquet: str
    out_parquet: str
    dp_size: int
    arrow_batch_size: int
    dtype: torch.dtype
    attn_implementation: str
    allow_partial_merge: bool
    max_groups: int
    max_length: int
    distribution_token_id: int
    length_bins: List[int]
    reward_values: List[float]
    shuffle_seed: int
    label_rollouts_per_prompt: int
    label_seed: int
    label_column: str
    ablation_type: str


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--critic-model-id", required=True, help="HF id/path for the trained in-context critic model.")
    p.add_argument("--in-parquet", required=True, help="Input parquet (generated rollouts).")
    p.add_argument("--out-parquet", required=True, help="Output parquet with critic predictions added.")
    p.add_argument("--dp-size", type=int, default=0, help="Number of GPU workers (default: all visible GPUs).")
    p.add_argument("--arrow-batch-size", type=int, default=1024, help="Rows per parquet read batch (per worker).")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for inference.",
    )
    p.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation to use when loading the model.",
    )
    p.add_argument(
        "--allow-partial-merge",
        action="store_true",
        help="If set, merge whatever shard outputs exist; otherwise require all dp shards.",
    )
    p.add_argument(
        "--max-groups",
        type=int,
        default=-1,
        help="If >0, stop after processing this many prompt_idx groups per worker (debug).",
    )
    p.add_argument("--max-length", type=int, default=131_072, help="Max packed sequence length.")
    p.add_argument(
        "--distribution-token-id",
        type=int,
        default=151669,
        help="Starting token id for the joint head (must match training).",
    )
    p.add_argument(
        "--length-bins",
        type=int,
        nargs="+",
        default=[0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Length bin edges used for tokens-remaining prediction (must match training).",
    )
    p.add_argument(
        "--reward-values",
        type=float,
        nargs="+",
        default=[0.0, 1.0],
        help="Reward state midpoints (must match training; for correctness: 0 1).",
    )
    p.add_argument("--shuffle-seed", type=int, default=0, help="Seed for deterministic per-rollout context shuffles.")
    p.add_argument(
        "--label-rollouts-per-prompt",
        type=int,
        default=-1,
        help=(
            "If >0, label only this many rollouts per prompt_idx group (sampled without replacement). "
            "If 0, label none. If <0, label all rollouts (default)."
        ),
    )
    p.add_argument("--label-seed", type=int, default=0, help="Seed for selecting which rollouts to label per prompt.")
    p.add_argument(
        "--label-column",
        choices=["correct"],
        default="correct",
        help="Which column provides ground-truth reward for in-context feedback (currently only 'correct').",
    )
    p.add_argument(
        "--ablation-type",
        choices=["no_ans"],
        default="no_ans",
        help="In-context packing configuration (only 'no_ans' supported).",
    )
    args = p.parse_args()

    in_abs = os.path.abspath(args.in_parquet)
    out_abs = os.path.abspath(args.out_parquet)
    if in_abs == out_abs:
        raise ValueError("Refusing to overwrite input parquet; choose a different --out-parquet.")

    dtype = _dtype_from_str(args.dtype)

    dp_size = int(args.dp_size)
    if dp_size < 0:
        raise ValueError("--dp-size must be >= 0")

    arrow_batch_size = int(args.arrow_batch_size)
    if arrow_batch_size <= 0:
        raise ValueError("--arrow-batch-size must be >= 1")

    max_groups = int(args.max_groups)

    length_bins = [int(x) for x in args.length_bins]
    if len(length_bins) < 2:
        raise ValueError("--length-bins must contain at least two edges.")
    if any(b < 0 for b in length_bins):
        raise ValueError("--length-bins must be non-negative.")
    if any(length_bins[i] >= length_bins[i + 1] for i in range(len(length_bins) - 1)):
        raise ValueError("--length-bins must be strictly increasing.")

    reward_values = [float(x) for x in args.reward_values]
    if len(reward_values) != 2:
        raise ValueError("--reward-values must be exactly 2 values for p_correct labeling (expected: 0 1).")
    if not (min(reward_values) <= 0.0 <= max(reward_values) and min(reward_values) <= 1.0 <= max(reward_values)):
        raise ValueError("--reward-values must span [0, 1] for correctness labeling.")

    label_rollouts_per_prompt = int(args.label_rollouts_per_prompt)
    if label_rollouts_per_prompt < -1:
        raise ValueError("--label-rollouts-per-prompt must be >= -1")

    return Config(
        critic_model_id=str(args.critic_model_id),
        in_parquet=str(args.in_parquet),
        out_parquet=str(args.out_parquet),
        dp_size=dp_size,
        arrow_batch_size=arrow_batch_size,
        dtype=dtype,
        attn_implementation=str(args.attn_implementation),
        allow_partial_merge=bool(args.allow_partial_merge),
        max_groups=max_groups,
        max_length=int(args.max_length),
        distribution_token_id=int(args.distribution_token_id),
        length_bins=length_bins,
        reward_values=reward_values,
        shuffle_seed=int(args.shuffle_seed),
        label_rollouts_per_prompt=label_rollouts_per_prompt,
        label_seed=int(args.label_seed),
        label_column=str(args.label_column),
        ablation_type=str(args.ablation_type),
    )


def _ensure_required_columns(schema: pa.Schema, *, label_column: str, require_candidates: bool = False) -> None:
    names = set(schema.names)
    required = {"prompt_idx", "prompt", "output_token_ids", label_column}
    if require_candidates:
        required.add("candidate_ids")
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")


def _has_candidate_columns(schema: pa.Schema) -> bool:
    """Check if the schema has candidate_ids column for Q-value evaluation."""
    return "candidate_ids" in set(schema.names)


def _add_new_columns_to_schema(schema: pa.Schema, *, include_candidate_q: bool = False) -> pa.Schema:
    names = set(schema.names)
    all_columns = list(NEW_COLUMNS)
    if include_candidate_q:
        all_columns.extend(CANDIDATE_Q_COLUMNS)
    for col, _typ in all_columns:
        if col in names:
            raise ValueError(
                f"Input parquet already has column '{col}'. "
                "Write to a new --out-parquet (or remove the existing columns first)."
            )
    out = schema
    for col, typ in all_columns:
        out = out.append(pa.field(col, typ))
    return out


def _load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot run safely.")
    return tokenizer


def _load_model(model_id: str, *, dtype: torch.dtype, attn_implementation: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model


def _unwrap_lm_head(model: torch.nn.Module):
    tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    lm_head = tgt.lm_head if hasattr(tgt, "lm_head") else tgt.get_output_embeddings()
    return lm_head


def _get_base_model(model: torch.nn.Module):
    tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    for attr in ("model", "transformer", "base_model"):
        if hasattr(tgt, attr):
            return getattr(tgt, attr)
    return None


def _create_tokenized_message(role: str, tokenized_content: Sequence[int]) -> List[int]:
    role_token_id = {
        "system": SYSTEM_TOKEN_ID,
        "user": USER_TOKEN_ID,
        "assistant": ASSISTANT_TOKEN_ID,
    }[role]

    content = list(tokenized_content)
    if content and content[-1] == IM_END_TOKEN_ID:
        content = content[:-1]

    return [IM_START_TOKEN_ID, role_token_id, NEWLINE_TOKEN_ID] + content + [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]


def _length_bin_midpoints(length_bins: Sequence[int]) -> List[float]:
    mids: List[float] = []
    for i in range(len(length_bins) - 1):
        lo = float(length_bins[i])
        hi = float(length_bins[i + 1])
        mids.append(0.5 * (lo + hi))
    return mids


@dataclass
class _Trajectory:
    row_idx: int
    response_ids: List[int]
    reward: float


@dataclass
class _CtxInfo:
    traj_block: List[int]
    feedback_block: List[int]
    traj_content_len: int


def _build_ctx_infos(trajectories: Sequence[_Trajectory], tokenizer) -> List[_CtxInfo]:
    infos: List[_CtxInfo] = []
    header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
    for traj in trajectories:
        traj_content = list(traj.response_ids)
        has_eos = bool(traj_content) and traj_content[-1] == IM_END_TOKEN_ID
        footer = [] if has_eos else [IM_END_TOKEN_ID]
        traj_block = header + traj_content + footer

        content_plus_footer_len = len(traj_content) + len(footer)
        full_feedback_str = f"Reward: {float(traj.reward)}\nLength: {content_plus_footer_len} tokens"
        reward_tokens_full = tokenizer.encode(full_feedback_str, add_special_tokens=False)
        feedback_block = _create_tokenized_message("user", reward_tokens_full)

        infos.append(
            _CtxInfo(
                traj_block=traj_block,
                feedback_block=feedback_block,
                traj_content_len=len(traj_content),
            )
        )
    return infos


def _truncate_target_content_to_fit(prefix_len: int, target_content: List[int], max_length: int) -> Tuple[List[int], List[int], bool, bool]:
    """
    Returns (trunc_content, footer, truncated, can_fit_min_block).

    We always include assistant header; footer is IM_END if trunc_content does not end with IM_END.
    Positions are computed only for trunc_content (aligned to stored output_token_ids).
    """
    header_len = 3  # [IM_START, ASSISTANT, NEWLINE]
    budget_for_assistant = int(max_length) - int(prefix_len) - int(header_len)
    if budget_for_assistant <= 0:
        return ([], [IM_END_TOKEN_ID], True, False)

    L = len(target_content)

    trunc_len = min(L, budget_for_assistant)
    while trunc_len > 0:
        last_is_end = (target_content[trunc_len - 1] == IM_END_TOKEN_ID)
        footer_len = 0 if last_is_end else 1
        if trunc_len + footer_len <= budget_for_assistant:
            break
        trunc_len -= 1

    if trunc_len == 0:
        if 1 <= budget_for_assistant:
            return ([], [IM_END_TOKEN_ID], True, True)
        return ([], [IM_END_TOKEN_ID], True, False)

    trunc_content = list(target_content[:trunc_len])
    last_is_end = trunc_content[-1] == IM_END_TOKEN_ID
    footer = [] if last_is_end else [IM_END_TOKEN_ID]
    truncated = trunc_len < L
    return (trunc_content, footer, truncated, True)


def _pack_target_last(
    tokenizer,
    *,
    prompt: str,
    trajectories: Sequence[_Trajectory],
    ctx_infos: Sequence[_CtxInfo],
    target_idx: int,
    max_length: int,
    shuffle_seed: int,
    prompt_idx: int,
) -> Tuple[List[int], List[int], int, int]:
    """
    Returns (full_input_ids, target_positions, trunc_len, orig_len).
    target_positions correspond to trunc_content tokens only (exclude appended footer).
    """
    prompt_tokens = tokenizer.encode(prompt or "", add_special_tokens=False)
    prefix = _create_tokenized_message("user", prompt_tokens)
    prefix_len = len(prefix)

    header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]

    target = trajectories[target_idx]
    target_content_orig = list(target.response_ids)
    orig_len = len(target_content_orig)

    trunc_content, footer, _truncated, can_fit = _truncate_target_content_to_fit(prefix_len, target_content_orig, max_length)
    if not can_fit:
        return (prefix[:max_length], [], 0, orig_len)

    target_block = header + trunc_content + footer
    fixed_len = prefix_len + len(target_block)
    if fixed_len > max_length:
        return (prefix[:max_length], [], 0, orig_len)

    other = [i for i in range(len(trajectories)) if i != target_idx]
    seed = (int(shuffle_seed) * 1000003 + int(prompt_idx) * 1009 + int(target_idx) * 9176) & 0xFFFFFFFF
    rng = random.Random(seed)
    rng.shuffle(other)

    total_len = fixed_len
    ctx_keep: List[int] = []
    for j in other:
        add_len = len(ctx_infos[j].traj_block) + len(ctx_infos[j].feedback_block)
        if total_len + add_len > max_length:
            break
        ctx_keep.append(j)
        total_len += add_len

    full_input_ids: List[int] = list(prefix)
    for j in ctx_keep:
        full_input_ids.extend(ctx_infos[j].traj_block)
        full_input_ids.extend(ctx_infos[j].feedback_block)

    target_start = len(full_input_ids)
    full_input_ids.extend(target_block)

    if len(full_input_ids) > max_length:
        raise RuntimeError(f"Packed length {len(full_input_ids)} > max_length {max_length}")

    target_positions = [target_start + len(header) + i for i in range(len(trunc_content))]
    return (full_input_ids, target_positions, len(trunc_content), orig_len)


@torch.inference_mode()
def _predict_for_positions(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,        # [1, S]
    attention_mask: torch.Tensor,   # [1, S]
    positions: Sequence[int],
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    length_midpoints: torch.Tensor,  # [num_length_bins]
) -> Tuple[List[float], List[float]]:
    if not positions:
        return ([], [])

    base = _get_base_model(model)
    if base is None:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]
    else:
        out = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        else:
            raise RuntimeError("Cannot obtain last_hidden_state")

    device = hidden.device
    pos_t = torch.tensor(list(positions), device=device, dtype=torch.long)
    h_sel = hidden[0].index_select(0, pos_t)  # [T, E]

    lm_head = _unwrap_lm_head(model)
    w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
    b = (
        lm_head.bias[distribution_token_id : distribution_token_id + num_bins]
        if hasattr(lm_head, "bias") and lm_head.bias is not None
        else None
    )

    logits = F.linear(h_sel, w, b).float()  # [T, num_bins]

    num_reward_states = num_bins // num_length_bins
    logits_rs = logits.view(-1, num_reward_states, num_length_bins)

    logits_reward = torch.logsumexp(logits_rs, dim=2)  # [T, num_reward_states]
    probs_reward = torch.softmax(logits_reward, dim=1)
    p_correct = probs_reward[:, int(correct_reward_index)]

    logits_length = torch.logsumexp(logits_rs, dim=1)  # [T, num_length_bins]
    probs_length = torch.softmax(logits_length, dim=1)
    expected_remaining = (probs_length * length_midpoints.unsqueeze(0)).sum(dim=1)

    return (p_correct.detach().float().cpu().tolist(), expected_remaining.detach().float().cpu().tolist())


@torch.inference_mode()
def _predict_with_kv_cache(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,        # [1, S]
    attention_mask: torch.Tensor,   # [1, S]
    positions: Sequence[int],
    candidate_ids: Sequence[Sequence[int]],  # [num_positions][num_candidates]
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    length_midpoints: torch.Tensor,  # [num_length_bins]
) -> Tuple[List[float], List[float], List[List[float]]]:
    """
    Predict P(correct) for actual tokens AND evaluate Q-values for candidates.

    Uses KV caching for efficiency:
    1. Run full sequence once to get KV cache + hidden states for actual tokens
    2. For each position, slice KV cache and evaluate all candidates in a batch

    Returns:
        p_correct: List[float] - P(correct) for actual tokens at each position
        expected_remaining: List[float] - Expected tokens remaining at each position
        candidate_q_values: List[List[float]] - Q-values for each candidate at each position
    """
    if not positions:
        return ([], [], [])

    device = input_ids.device
    seq_len = input_ids.shape[1]

    # Step 1: Run full sequence with KV cache enabled
    base = _get_base_model(model)
    if base is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        past_key_values = out.past_key_values
    else:
        out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        else:
            raise RuntimeError("Cannot obtain last_hidden_state")
        past_key_values = out.past_key_values

    # Step 2: Extract predictions for actual tokens (same as before)
    pos_t = torch.tensor(list(positions), device=device, dtype=torch.long)
    h_sel = hidden[0].index_select(0, pos_t)  # [T, E]

    lm_head = _unwrap_lm_head(model)
    w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
    b = (
        lm_head.bias[distribution_token_id : distribution_token_id + num_bins]
        if hasattr(lm_head, "bias") and lm_head.bias is not None
        else None
    )

    logits = F.linear(h_sel, w, b).float()  # [T, num_bins]

    num_reward_states = num_bins // num_length_bins
    logits_rs = logits.view(-1, num_reward_states, num_length_bins)

    logits_reward = torch.logsumexp(logits_rs, dim=2)  # [T, num_reward_states]
    probs_reward = torch.softmax(logits_reward, dim=1)
    p_correct = probs_reward[:, int(correct_reward_index)]

    logits_length = torch.logsumexp(logits_rs, dim=1)  # [T, num_length_bins]
    probs_length = torch.softmax(logits_length, dim=1)
    expected_remaining = (probs_length * length_midpoints.unsqueeze(0)).sum(dim=1)

    p_correct_list = p_correct.detach().float().cpu().tolist()
    expected_remaining_list = expected_remaining.detach().float().cpu().tolist()

    # Step 3: Evaluate candidates at each position using KV cache
    candidate_q_values: List[List[float]] = []

    for pos_idx, (pos, cands) in enumerate(zip(positions, candidate_ids)):
        if not cands:
            candidate_q_values.append([])
            continue

        K = len(cands)

        # Slice KV cache to prefix [0:pos] (everything before this position)
        # pos is the absolute position in the full sequence
        sliced_kv = _slice_kv_cache(past_key_values, end=pos)

        # Expand for batch of K candidates (broadcast, no memory copy)
        expanded_kv = _expand_kv_cache(sliced_kv, batch_size=K)

        # Create input for candidates: [K, 1]
        cand_input = torch.tensor(cands, dtype=torch.long, device=device).unsqueeze(1)

        # Forward pass with candidates
        if base is None:
            cand_out = model(
                input_ids=cand_input,
                past_key_values=expanded_kv,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            cand_hidden = cand_out.hidden_states[-1]  # [K, 1, E]
        else:
            cand_out = base(
                input_ids=cand_input,
                past_key_values=expanded_kv,
                use_cache=False,
                return_dict=True,
            )
            if hasattr(cand_out, "last_hidden_state") and cand_out.last_hidden_state is not None:
                cand_hidden = cand_out.last_hidden_state
            elif hasattr(cand_out, "hidden_states") and cand_out.hidden_states is not None:
                cand_hidden = cand_out.hidden_states[-1]
            else:
                raise RuntimeError("Cannot obtain last_hidden_state for candidates")

        # Extract Q-values (P(correct)) for each candidate
        cand_h = cand_hidden.squeeze(1)  # [K, E]
        cand_logits = F.linear(cand_h, w, b).float()  # [K, num_bins]

        cand_logits_rs = cand_logits.view(-1, num_reward_states, num_length_bins)
        cand_logits_reward = torch.logsumexp(cand_logits_rs, dim=2)  # [K, num_reward_states]
        cand_probs_reward = torch.softmax(cand_logits_reward, dim=1)
        cand_q = cand_probs_reward[:, int(correct_reward_index)]  # [K]

        candidate_q_values.append(cand_q.detach().float().cpu().tolist())

    return (p_correct_list, expected_remaining_list, candidate_q_values)


def _row_group_range(num_row_groups: int, *, rank: int, world: int) -> Tuple[int, int]:
    start = (num_row_groups * rank) // world
    end = (num_row_groups * (rank + 1)) // world
    return start, end


def _iter_row_groups(pf: pq.ParquetFile, *, start_rg: int, batch_size: int) -> Iterable[Tuple[int, pa.RecordBatch]]:
    for rg in range(start_rg, pf.num_row_groups):
        for rb in pf.iter_batches(row_groups=[rg], batch_size=batch_size):
            yield rg, rb


def _merge_shards(out_parquet: str, *, dp_size: int, allow_partial: bool) -> Tuple[bool, int, int]:
    part_paths = [f"{out_parquet}.part{r}" for r in range(dp_size) if os.path.exists(f"{out_parquet}.part{r}")]
    found = len(part_paths)
    expected = dp_size

    if found == 0:
        return (False, 0, expected)
    if found != expected and not allow_partial:
        return (False, found, expected)

    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    schema = pq.ParquetFile(part_paths[0]).schema_arrow
    writer = pq.ParquetWriter(out_parquet, schema)
    try:
        for part in part_paths:
            pf = pq.ParquetFile(part)
            for rb in pf.iter_batches():
                writer.write_table(pa.Table.from_batches([rb], schema=schema))
        writer.close()
    except Exception:
        writer.close()
        raise

    for part in part_paths:
        try:
            os.remove(part)
        except OSError:
            pass

    return (True, found, expected)


def _worker(local_rank: int, world: int, cfg: Config) -> None:
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = _load_tokenizer(cfg.critic_model_id)
    attn_impl = cfg.attn_implementation
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"
    model = _load_model(cfg.critic_model_id, dtype=cfg.dtype, attn_implementation=attn_impl).to(device)

    if cfg.ablation_type != "no_ans":
        raise ValueError("Only --ablation-type no_ans is supported for labeling.")

    num_length_bins = len(cfg.length_bins) - 1
    num_reward_states = len(cfg.reward_values)
    num_bins = num_length_bins * num_reward_states
    correct_reward_index = int(max(range(len(cfg.reward_values)), key=lambda i: cfg.reward_values[i]))

    length_mids = torch.tensor(_length_bin_midpoints(cfg.length_bins), dtype=torch.float32, device=device)

    pf = pq.ParquetFile(cfg.in_parquet)
    _ensure_required_columns(pf.schema_arrow, label_column=cfg.label_column)
    has_candidates = _has_candidate_columns(pf.schema_arrow)
    out_schema = _add_new_columns_to_schema(pf.schema_arrow, include_candidate_q=has_candidates)

    start_rg, end_rg = _row_group_range(pf.num_row_groups, rank=local_rank, world=world)
    read_start_rg = max(0, start_rg - 1)

    os.makedirs(os.path.dirname(cfg.out_parquet) or ".", exist_ok=True)
    shard_path = f"{cfg.out_parquet}.part{local_rank}"
    writer = pq.ParquetWriter(shard_path, out_schema)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=cfg.dtype)
        if device.type == "cuda" and cfg.dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )

    start_t = time.perf_counter()
    processed_groups = 0

    current_prompt_idx = None
    current_group_start_rg = None
    current_batches: List[pa.RecordBatch] = []

    def flush_group():
        nonlocal processed_groups, current_prompt_idx, current_group_start_rg, current_batches
        if current_prompt_idx is None or not current_batches:
            current_prompt_idx = None
            current_group_start_rg = None
            current_batches = []
            return

        should_process = (current_group_start_rg is not None) and (start_rg <= current_group_start_rg < end_rg)

        if should_process:
            table = pa.Table.from_batches(current_batches, schema=pf.schema_arrow)

            prompt_vals = table.column(table.schema.get_field_index("prompt")).to_pylist()
            prompt = ""
            for x in prompt_vals:
                if x is not None:
                    prompt = str(x)
                    break

            output_token_ids = table.column(table.schema.get_field_index("output_token_ids")).to_pylist()
            reward_vals = table.column(table.schema.get_field_index(cfg.label_column)).to_pylist()

            # Load candidate_ids if available
            row_candidate_ids: Optional[List[List[List[int]]]] = None
            if has_candidates:
                cand_col = table.column(table.schema.get_field_index("candidate_ids")).to_pylist()
                row_candidate_ids = []
                for cands in cand_col:
                    if cands is None:
                        row_candidate_ids.append([])
                    else:
                        # cands is List[List[int]] - candidates per position
                        row_candidate_ids.append([list(c) if c else [] for c in cands])

            trajectories: List[_Trajectory] = []
            for i_row, (toks, r) in enumerate(zip(output_token_ids, reward_vals)):
                toks_list = list(toks) if toks is not None else []
                reward = float(r) if r is not None else 0.0
                reward = max(0.0, min(1.0, reward))
                trajectories.append(_Trajectory(row_idx=i_row, response_ids=toks_list, reward=reward))

            n_rows = len(trajectories)
            ctx_infos = _build_ctx_infos(trajectories, tokenizer)

            rows_to_label: Optional[Set[int]]
            if cfg.label_rollouts_per_prompt < 0:
                rows_to_label = None
            else:
                k = int(cfg.label_rollouts_per_prompt)
                if k <= 0:
                    rows_to_label = set()
                elif k >= n_rows:
                    rows_to_label = None
                else:
                    rng = random.Random(int(cfg.label_seed) + int(current_prompt_idx))
                    rows_to_label = set(rng.sample(range(n_rows), k))

            out_p_correct: List[Optional[List[float]]] = [None for _ in range(n_rows)]
            out_expected: List[Optional[List[float]]] = [None for _ in range(n_rows)]
            out_q_values: List[Optional[List[List[float]]]] = [None for _ in range(n_rows)] if has_candidates else []

            label_all = rows_to_label is None

            for target_idx in range(n_rows):
                if (not label_all) and (rows_to_label is not None) and (target_idx not in rows_to_label):
                    continue

                orig_len = len(trajectories[target_idx].response_ids)
                if orig_len == 0:
                    out_p_correct[target_idx] = []
                    out_expected[target_idx] = []
                    if has_candidates:
                        out_q_values[target_idx] = []
                    continue

                full_input_ids, target_positions, trunc_len, _orig_len_check = _pack_target_last(
                    tokenizer,
                    prompt=prompt,
                    trajectories=trajectories,
                    ctx_infos=ctx_infos,
                    target_idx=target_idx,
                    max_length=cfg.max_length,
                    shuffle_seed=cfg.shuffle_seed,
                    prompt_idx=int(current_prompt_idx),
                )

                if trunc_len <= 0 or not target_positions:
                    out_p_correct[target_idx] = [float("nan")] * orig_len
                    out_expected[target_idx] = [float("nan")] * orig_len
                    if has_candidates:
                        # Fill with empty lists for each position
                        out_q_values[target_idx] = [[] for _ in range(orig_len)]
                    continue

                input_ids_t = torch.tensor(full_input_ids, dtype=torch.long, device=device).unsqueeze(0)
                attention_mask_t = torch.ones_like(input_ids_t, dtype=torch.long, device=device)

                with autocast_ctx:
                    if has_candidates and row_candidate_ids is not None:
                        # Get candidates for this trajectory, truncated to match target_positions
                        traj_candidates = row_candidate_ids[target_idx]
                        # Truncate candidates to match the number of positions we're evaluating
                        traj_candidates = traj_candidates[:len(target_positions)]
                        # Pad with empty lists if needed
                        while len(traj_candidates) < len(target_positions):
                            traj_candidates.append([])

                        p_list, e_list, q_list = _predict_with_kv_cache(
                            model,
                            input_ids=input_ids_t,
                            attention_mask=attention_mask_t,
                            positions=target_positions,
                            candidate_ids=traj_candidates,
                            distribution_token_id=cfg.distribution_token_id,
                            num_bins=num_bins,
                            num_length_bins=num_length_bins,
                            correct_reward_index=correct_reward_index,
                            length_midpoints=length_mids,
                        )
                    else:
                        p_list, e_list = _predict_for_positions(
                            model,
                            input_ids=input_ids_t,
                            attention_mask=attention_mask_t,
                            positions=target_positions,
                            distribution_token_id=cfg.distribution_token_id,
                            num_bins=num_bins,
                            num_length_bins=num_length_bins,
                            correct_reward_index=correct_reward_index,
                            length_midpoints=length_mids,
                        )
                        q_list = None

                if len(p_list) < orig_len:
                    pad = [float("nan")] * (orig_len - len(p_list))
                    p_list = p_list + pad
                    e_list = e_list + pad
                    if has_candidates and q_list is not None:
                        # Pad q_list with empty lists for truncated positions
                        q_list = q_list + [[] for _ in range(orig_len - len(q_list))]

                out_p_correct[target_idx] = p_list
                out_expected[target_idx] = e_list
                if has_candidates:
                    out_q_values[target_idx] = q_list if q_list is not None else [[] for _ in range(orig_len)]

            if rows_to_label is None:
                model_ids = [cfg.critic_model_id] * n_rows
            else:
                model_ids = [cfg.critic_model_id if (out_p_correct[i] is not None) else None for i in range(n_rows)]

            table = table.append_column("critic_model_id", pa.array(model_ids, type=pa.string()))
            table = table.append_column("critic_p_correct", pa.array(out_p_correct, type=pa.list_(pa.float32())))
            table = table.append_column(
                "critic_expected_tokens_remaining", pa.array(out_expected, type=pa.list_(pa.float32()))
            )
            if has_candidates:
                table = table.append_column(
                    "candidate_q_values", pa.array(out_q_values, type=pa.list_(pa.list_(pa.float32())))
                )

            writer.write_table(table, row_group_size=None)
            processed_groups += 1

            if cfg.max_groups > 0 and processed_groups >= cfg.max_groups:
                current_prompt_idx = None
                current_group_start_rg = None
                current_batches = []
                raise StopIteration

        current_prompt_idx = None
        current_group_start_rg = None
        current_batches = []

    past_primary = False
    try:
        for rg, rb in _iter_row_groups(pf, start_rg=read_start_rg, batch_size=cfg.arrow_batch_size):
            if rg >= end_rg:
                past_primary = True

            prompt_idx_col = rb.column(rb.schema.get_field_index("prompt_idx")).to_pylist()
            n = len(prompt_idx_col)
            i = 0
            while i < n:
                pid = prompt_idx_col[i]
                j = i + 1
                while j < n and prompt_idx_col[j] == pid:
                    j += 1
                chunk = rb.slice(i, j - i)

                if current_prompt_idx is None:
                    current_prompt_idx = pid
                    current_group_start_rg = rg
                    current_batches = [chunk]
                elif pid == current_prompt_idx:
                    current_batches.append(chunk)
                else:
                    flush_group()
                    current_prompt_idx = pid
                    current_group_start_rg = rg
                    current_batches = [chunk]

                i = j

            if past_primary and current_prompt_idx is None:
                break

        flush_group()
    except StopIteration:
        pass
    finally:
        writer.close()

    elapsed = time.perf_counter() - start_t
    print(
        f"[rank {local_rank}] wrote {shard_path} ({processed_groups} prompt groups) in {elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    cfg = parse_args()

    ngpus = torch.cuda.device_count()
    if cfg.dp_size == 0:
        dp = max(1, ngpus)
    else:
        dp = cfg.dp_size

    if ngpus > 0 and dp > ngpus:
        raise RuntimeError(f"Requested dp_size={dp}, but only {ngpus} CUDA devices are visible.")
    if ngpus == 0 and dp != 1:
        raise RuntimeError("No CUDA devices visible; run with --dp-size 1 for CPU scoring.")

    # Check if input has candidate columns for Q-value evaluation
    pf_check = pq.ParquetFile(cfg.in_parquet)
    has_candidates = _has_candidate_columns(pf_check.schema_arrow)
    candidate_info = "  candidate_q:     enabled (using KV cache)\n" if has_candidates else ""

    print(
        f"Scoring parquet with in-context critic:\n"
        f"  critic_model_id: {cfg.critic_model_id}\n"
        f"  in:              {cfg.in_parquet}\n"
        f"  out:             {cfg.out_parquet}\n"
        f"  dp_size:         {dp}\n"
        f"  max_length:      {cfg.max_length}\n"
        f"  length_bins:     {cfg.length_bins}\n"
        f"  reward_values:   {cfg.reward_values}\n"
        f"  shuffle_seed:    {cfg.shuffle_seed}\n"
        f"  label_per_prompt: {cfg.label_rollouts_per_prompt}\n"
        f"  label_seed:      {cfg.label_seed}\n"
        f"{candidate_info}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs = []
    for r in range(dp):
        p = ctx.Process(target=_worker, args=(r, dp, cfg), daemon=False)
        p.start()
        procs.append(p)

    any_fail = False
    for r, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            any_fail = True
            print(f"[main] worker rank {r} (pid={p.pid}) exited with code {p.exitcode}", flush=True)

    merged, found, expected = _merge_shards(cfg.out_parquet, dp_size=dp, allow_partial=cfg.allow_partial_merge)
    if not merged:
        if found == 0:
            print("No shards produced.", flush=True)
        else:
            print(f"Shards missing ({found}/{expected}).", flush=True)
        if any_fail or found != expected:
            sys.exit(1)
        return

    suffix = "" if found == expected else f" (PARTIAL: {found}/{expected} shards)"
    print(f"Wrote {cfg.out_parquet}{suffix}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
