#!/usr/bin/env python3
"""
Label a generated Parquet dataset with in-context critic predictions.

For each prompt_idx group, we deterministically shuffle sibling trajectories, build
the packed in-context sequence exactly like `src/train_in_context_critic.py`
(currently only supports the `no_ans` ablation), run a forward pass, and extract:

  - p_correct per output token (marginalizing over length bins)
  - expected remaining tokens per output token (marginalizing over reward bins)

The output Parquet preserves all input columns and appends:
  - critic_model_id
  - critic_p_correct
  - critic_expected_tokens_remaining

Optionally, you can label only a subset of rollouts per prompt group (selected
uniformly without replacement with a per-prompt seed). Unlabeled rows are kept
in the output with null critic columns.

This script supports simple data-parallel execution across GPUs by splitting the
input parquet by contiguous row-group ranges (with minimal overlap so prompt groups
crossing boundaries are still processed exactly once).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Qwen Specific Control Tokens ---
# Must match `src/train_in_context_critic.py`.
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
    p.add_argument("--shuffle-seed", type=int, default=0, help="Seed for deterministic per-prompt shuffles.")
    p.add_argument(
        "--label-rollouts-per-prompt",
        type=int,
        default=-1,
        help=(
            "If >0, label only this many rollouts per prompt_idx group (selected uniformly without replacement). "
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
        help="In-context packing configuration (currently only 'no_ans' supported).",
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


def _ensure_required_columns(schema: pa.Schema, *, label_column: str) -> None:
    names = set(schema.names)
    required = {"prompt_idx", "prompt", "output_token_ids", label_column}
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")


def _add_new_columns_to_schema(schema: pa.Schema) -> pa.Schema:
    names = set(schema.names)
    for col, _typ in NEW_COLUMNS:
        if col in names:
            raise ValueError(
                f"Input parquet already has column '{col}'. "
                "Write to a new --out-parquet (or remove the existing columns first)."
            )
    out = schema
    for col, typ in NEW_COLUMNS:
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


def _length_bin_midpoints(length_bins: Sequence[int]) -> List[float]:
    mids: List[float] = []
    for i in range(len(length_bins) - 1):
        lo = float(length_bins[i])
        hi = float(length_bins[i + 1])
        mids.append(0.5 * (lo + hi))
    return mids


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


@dataclass
class _Trajectory:
    row_idx: int  # row index within the current prompt group table
    response_ids: List[int]
    reward: float


@dataclass
class _TrajInfo:
    traj: _Trajectory
    traj_content: List[int]
    header: List[int]
    footer: List[int]
    traj_block: List[int]
    traj_block_len: int
    content_plus_footer_len: int
    reward_tokens_full: List[int]
    reward_block_len_full: int


def _build_infos(trajectories: Sequence[_Trajectory], tokenizer) -> List[_TrajInfo]:
    infos: List[_TrajInfo] = []
    for traj in trajectories:
        traj_content = list(traj.response_ids)
        header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
        has_eos = bool(traj_content) and traj_content[-1] == IM_END_TOKEN_ID
        footer = [] if has_eos else [IM_END_TOKEN_ID]
        traj_block = header + traj_content + footer

        content_plus_footer_len = len(traj_content) + len(footer)
        full_feedback_str = f"Reward: {float(traj.reward)}\nLength: {content_plus_footer_len} tokens"
        reward_tokens_full = tokenizer.encode(full_feedback_str, add_special_tokens=False)

        # create_tokenized_message adds 5 tokens around content: IM_START, role, NL, IM_END, NL
        reward_block_len_full = len(reward_tokens_full) + 5

        infos.append(
            _TrajInfo(
                traj=traj,
                traj_content=traj_content,
                header=header,
                footer=footer,
                traj_block=traj_block,
                traj_block_len=len(traj_block),
                content_plus_footer_len=content_plus_footer_len,
                reward_tokens_full=reward_tokens_full,
                reward_block_len_full=reward_block_len_full,
            )
        )
    return infos


def _choose_suffix_that_fits(
    infos: Sequence[_TrajInfo],
    *,
    prefix_len: int,
    max_length: int,
    tokenizer,
) -> Tuple[List[_TrajInfo], bool]:
    """
    Return (included_infos, truncated_last).

    Mirrors the suffix-fit logic from `JointDistributionDataset.__getitem__` for
    the `no_ans` ablation.
    """
    n = len(infos)
    if n == 0:
        return ([], False)

    assistant_suffix = [0] * (n + 1)
    full_fb_suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        assistant_suffix[i] = assistant_suffix[i + 1] + infos[i].traj_block_len
        full_fb_suffix[i] = full_fb_suffix[i + 1]
        if i != n - 1:
            full_fb_suffix[i] += infos[i].reward_block_len_full

    start_idx = n - 1
    for s in range(n):
        total = prefix_len + assistant_suffix[s] + full_fb_suffix[s]
        if total <= max_length:
            start_idx = s
            break

    included = list(infos[start_idx:])
    truncated_last = False

    # If even the last trajectory alone cannot fit, truncate its content to fit.
    if start_idx == n - 1:
        last = included[0]
        budget = max_length - prefix_len
        min_block = len(last.header) + len(last.footer)
        if budget < min_block:
            return ([], False)

        max_content_len = budget - min_block
        if max_content_len < 0:
            max_content_len = 0

        trunc_content = list(last.traj_content[:max_content_len])
        has_eos = bool(trunc_content) and trunc_content[-1] == IM_END_TOKEN_ID
        footer = [] if has_eos else [IM_END_TOKEN_ID]
        traj_block = last.header + trunc_content + footer

        included = [
            _TrajInfo(
                traj=last.traj,
                traj_content=trunc_content,
                header=last.header,
                footer=footer,
                traj_block=traj_block,
                traj_block_len=len(traj_block),
                content_plus_footer_len=len(trunc_content) + len(footer),
                reward_tokens_full=last.reward_tokens_full,
                reward_block_len_full=last.reward_block_len_full,
            )
        ]
        truncated_last = True

    return (included, truncated_last)


@torch.inference_mode()
def _score_packed_sequence(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    length_midpoints: torch.Tensor,
) -> torch.Tensor:
    """
    Return packed per-position predictions for the joint head.

    Output: logits_all of shape [S, num_bins] (batch size 1).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states[-1]  # [1, S, E]

    lm_head = _unwrap_lm_head(model)
    w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
    b = (
        lm_head.bias[distribution_token_id : distribution_token_id + num_bins]
        if hasattr(lm_head, "bias") and lm_head.bias is not None
        else None
    )

    logits_all = F.linear(hidden_states, w, b)[0]  # [S, num_bins]
    return logits_all


def _extract_token_lists(
    logits_all: torch.Tensor,
    *,
    positions: Sequence[int],
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    length_midpoints: torch.Tensor,
) -> Tuple[List[float], List[float]]:
    """
    From [S, num_bins] logits, extract per-token p_correct and expected remaining tokens.
    """
    if not positions:
        return ([], [])

    pos_t = torch.tensor(list(positions), device=logits_all.device, dtype=torch.long)
    logits = logits_all.index_select(0, pos_t)  # [T, num_bins]

    num_reward_states = num_bins // num_length_bins
    logits_rs = logits.view(-1, num_reward_states, num_length_bins)

    logits_reward = torch.logsumexp(logits_rs, dim=2)  # [T, num_reward_states]
    probs_reward = torch.softmax(logits_reward, dim=1)
    p_correct = probs_reward[:, int(correct_reward_index)]

    logits_length = torch.logsumexp(logits_rs, dim=1)  # [T, num_length_bins]
    probs_length = torch.softmax(logits_length, dim=1)
    expected_remaining = (probs_length * length_midpoints.unsqueeze(0)).sum(dim=1)

    return (p_correct.detach().float().cpu().tolist(), expected_remaining.detach().float().cpu().tolist())


def _pack_and_score_prompt_group(
    model: torch.nn.Module,
    tokenizer,
    *,
    prompt: str,
    trajectories: Sequence[_Trajectory],
    max_length: int,
    shuffle_seed: int,
    prompt_idx: int,
    rows_to_label: set[int] | None,
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    length_midpoints: torch.Tensor,
    autocast_ctx,
) -> Tuple[List[List[float] | None], List[List[float] | None]]:
    """
    Return (p_correct_lists, expected_remaining_lists) aligned to each row's output_token_ids.
    """
    # Deterministic shuffle per prompt.
    import random

    order = list(trajectories)
    random.Random(int(shuffle_seed) + int(prompt_idx)).shuffle(order)

    # Allocate outputs (per row, aligned to its original output_token_ids length).
    out_p_correct: List[List[float] | None] = [None for _ in range(len(trajectories))]
    out_expected: List[List[float] | None] = [None for _ in range(len(trajectories))]

    label_all = rows_to_label is None
    remaining_to_label: set[int] | None = None if label_all else set(rows_to_label)

    prompt_tokens = tokenizer.encode(prompt or "", add_special_tokens=False)
    prefix = _create_tokenized_message("user", prompt_tokens)
    prefix_len = len(prefix)

    # Label all trajectories by scoring one or more suffix packs (minimal number of packs).
    remaining = list(order)
    while remaining and (label_all or remaining_to_label):
        infos = _build_infos(remaining, tokenizer)
        included_infos, _truncated_last = _choose_suffix_that_fits(
            infos, prefix_len=prefix_len, max_length=max_length, tokenizer=tokenizer
        )
        if not included_infos:
            # Degenerate: cannot fit even a single assistant header/footer.
            # Fill NaNs for any trajectories we intended to label to keep alignment.
            for traj in remaining:
                if not label_all and remaining_to_label is not None and traj.row_idx not in remaining_to_label:
                    continue
                L = len(traj.response_ids)
                out_p_correct[traj.row_idx] = [float("nan")] * L
                out_expected[traj.row_idx] = [float("nan")] * L
            break

        included_row_ids = {info.traj.row_idx for info in included_infos}
        should_score = label_all or (
            remaining_to_label is not None and bool(included_row_ids.intersection(remaining_to_label))
        )

        # Build packed input_ids and capture per-trajectory token positions (aligned to output_token_ids).
        if should_score:
            full_input_ids: List[int] = list(prefix)
            positions_by_row: Dict[int, List[int]] = {}

            for local_idx, info in enumerate(included_infos):
                current_start = len(full_input_ids)
                full_input_ids.extend(info.traj_block)

                # Positions correspond to the tokens in traj_content (excluding any appended footer token).
                tok_positions = [current_start + len(info.header) + i for i in range(len(info.traj_content))]
                positions_by_row[info.traj.row_idx] = tok_positions

                # Add reward/length feedback only if there is a next trajectory in this pack.
                if local_idx != (len(included_infos) - 1):
                    full_input_ids.extend(_create_tokenized_message("user", info.reward_tokens_full))

            if len(full_input_ids) > max_length:
                raise RuntimeError(
                    f"Internal error: packed sequence length {len(full_input_ids)} exceeds max_length={max_length}."
                )

            device = next(model.parameters()).device
            input_ids_t = torch.tensor(full_input_ids, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask_t = torch.ones_like(input_ids_t, dtype=torch.long, device=device)

            with autocast_ctx:
                logits_all = _score_packed_sequence(
                    model,
                    input_ids=input_ids_t,
                    attention_mask=attention_mask_t,
                    distribution_token_id=distribution_token_id,
                    num_bins=num_bins,
                    num_length_bins=num_length_bins,
                    correct_reward_index=correct_reward_index,
                    length_midpoints=length_midpoints,
                )

            # Extract and store outputs for trajectories in this pack.
            for info in included_infos:
                row_idx = info.traj.row_idx
                if not label_all and rows_to_label is not None and row_idx not in rows_to_label:
                    continue

                L_orig = len(info.traj.response_ids)
                tok_positions = positions_by_row.get(row_idx, [])
                p_list, e_list = _extract_token_lists(
                    logits_all,
                    positions=tok_positions,
                    num_bins=num_bins,
                    num_length_bins=num_length_bins,
                    correct_reward_index=correct_reward_index,
                    length_midpoints=length_midpoints,
                )

                # If we truncated content to fit, positions cover only the prefix; pad the remainder with NaNs.
                if len(p_list) < L_orig:
                    pad = [float("nan")] * (L_orig - len(p_list))
                    p_list = p_list + pad
                    e_list = e_list + pad
                out_p_correct[row_idx] = p_list
                out_expected[row_idx] = e_list

        # Remove processed suffix trajectories from remaining.
        remaining = [t for t in remaining if t.row_idx not in included_row_ids]
        if remaining_to_label is not None:
            remaining_to_label.difference_update(included_row_ids)

    return out_p_correct, out_expected


def _row_group_range(num_row_groups: int, *, rank: int, world: int) -> Tuple[int, int]:
    start = (num_row_groups * rank) // world
    end = (num_row_groups * (rank + 1)) // world
    return start, end


def _iter_row_groups(
    pf: pq.ParquetFile, *, start_rg: int, batch_size: int
) -> Iterable[Tuple[int, pa.RecordBatch]]:
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
    out_schema = _add_new_columns_to_schema(pf.schema_arrow)

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

    # State for building contiguous prompt groups.
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

        # Only process groups whose *first seen* row group lies in this rank's primary range.
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

            trajectories: List[_Trajectory] = []
            for i_row, (toks, r) in enumerate(zip(output_token_ids, reward_vals)):
                if toks is None:
                    toks_list: List[int] = []
                else:
                    toks_list = list(toks)
                reward = float(r) if r is not None else 0.0
                reward = max(0.0, min(1.0, reward))
                trajectories.append(_Trajectory(row_idx=i_row, response_ids=toks_list, reward=reward))

            rows_to_label: set[int] | None
            if cfg.label_rollouts_per_prompt < 0:
                rows_to_label = None
            else:
                k = int(cfg.label_rollouts_per_prompt)
                n_rows = len(trajectories)
                if k <= 0:
                    rows_to_label = set()
                elif k >= n_rows:
                    rows_to_label = None
                else:
                    import random

                    rng = random.Random(int(cfg.label_seed) + int(current_prompt_idx))
                    rows_to_label = set(rng.sample(range(n_rows), k))

            p_correct_lists, expected_lists = _pack_and_score_prompt_group(
                model,
                tokenizer,
                prompt=prompt,
                trajectories=trajectories,
                max_length=cfg.max_length,
                shuffle_seed=cfg.shuffle_seed,
                prompt_idx=int(current_prompt_idx),
                rows_to_label=rows_to_label,
                distribution_token_id=cfg.distribution_token_id,
                num_bins=num_bins,
                num_length_bins=num_length_bins,
                correct_reward_index=correct_reward_index,
                length_midpoints=length_mids,
                autocast_ctx=autocast_ctx,
            )

            n_rows = table.num_rows
            if rows_to_label is None:
                model_ids = [cfg.critic_model_id] * n_rows
            else:
                model_ids = [cfg.critic_model_id if (p_correct_lists[i] is not None) else None for i in range(n_rows)]
            table = table.append_column("critic_model_id", pa.array(model_ids, type=pa.string()))
            table = table.append_column("critic_p_correct", pa.array(p_correct_lists, type=pa.list_(pa.float32())))
            table = table.append_column(
                "critic_expected_tokens_remaining", pa.array(expected_lists, type=pa.list_(pa.float32()))
            )

            writer.write_table(table, row_group_size=None)
            processed_groups += 1

            if cfg.max_groups > 0 and processed_groups >= cfg.max_groups:
                # Signal early-stop by clearing state and raising a sentinel.
                current_prompt_idx = None
                current_group_start_rg = None
                current_batches = []
                raise StopIteration

        # Reset state.
        current_prompt_idx = None
        current_group_start_rg = None
        current_batches = []

    past_primary = False
    try:
        for rg, rb in _iter_row_groups(pf, start_rg=read_start_rg, batch_size=cfg.arrow_batch_size):
            if rg >= end_rg:
                past_primary = True

            # Split record batch into runs of constant prompt_idx.
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

            # If we've passed the primary row-group range and are not inside a group that started in-range,
            # we can stop (any subsequent groups belong to the next rank).
            if past_primary and current_prompt_idx is None:
                break

        # Flush trailing group if any.
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
        f"  label_seed:      {cfg.label_seed}\n",
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
            print("✗ No shards were produced; check worker errors above.", flush=True)
        else:
            print(f"✗ Shards missing ({found}/{expected}).", flush=True)
        if any_fail or found != expected:
            sys.exit(1)
        return

    suffix = "" if found == expected else f" (PARTIAL: {found}/{expected} shards)"
    print(f"✓ Wrote {cfg.out_parquet}{suffix}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
