#!/usr/bin/env python3
"""
Label trajectories with Q-values for all candidate tokens at each position.

This script computes Q-values that can be used to train a student policy with
KL divergence loss against the extracted policy: π_ext ∝ π_ref * exp(Q/τ)

For each trajectory:
1. Build packed in-context sequence (matching training format exactly)
2. Use reference model to get min-p filtered candidates at each position
3. Compute Q-values for all candidates using the critic model
4. Store candidate_ids, candidate_q_values, and candidate_ref_logprobs

Output: A new parquet file with all original columns preserved, plus:
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate
- candidate_ref_logprobs: List[List[float]] - reference model log probs for each candidate
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


# --- Qwen control tokens (must match training) ---
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
SYSTEM_TOKEN_ID = 8948
NEWLINE_TOKEN_ID = 198


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    critic_path: str
    ref_path: str
    in_parquet: str
    out_parquet: str

    dp_size: int = 0
    max_length: int = 131072
    min_p: float = 0.01

    distribution_token_id: int = 151669
    reward_col: str = "correct"

    seed: int = 42
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    max_groups: int = -1  # For debugging: limit number of prompt groups
    max_rollouts_per_prompt: int = -1  # Limit rollouts per prompt (-1 = all)

    min_correct_rate: float = 0.25  # Min fraction of correct samples per prompt
    min_incorrect_rate: float = 0.25  # Min fraction of incorrect samples per prompt
    only_label_correct: bool = True  # Only label correct samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_dtype(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"bad dtype: {s}")


def _unwrap_model(m: Any) -> Any:
    m = m._orig_mod if hasattr(m, "_orig_mod") else m
    m = m.module if hasattr(m, "module") else m
    return m


def _get_lm_head(m: Any) -> Any:
    if hasattr(m, "lm_head"):
        return m.lm_head
    emb = m.get_output_embeddings() if hasattr(m, "get_output_embeddings") else None
    if emb is None:
        raise RuntimeError("no lm_head")
    return emb


def _get_base_model(m: Any) -> Any:
    for attr in ("model", "transformer", "base_model"):
        if hasattr(m, attr):
            return getattr(m, attr)
    return None


# ---------------------------------------------------------------------------
# KV Cache Helpers (matching extract_policy_generate.py exactly)
# ---------------------------------------------------------------------------

class ReadOnlyCache(DynamicCache):
    """A cache wrapper that ignores update() calls, making it read-only."""

    def __init__(self, key_cache: List[torch.Tensor], value_cache: List[torch.Tensor]):
        super().__init__()
        self.key_cache = key_cache
        self.value_cache = value_cache
        if key_cache:
            self._seen_tokens = key_cache[0].shape[2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.key_cache):
            full_k = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            full_v = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            return full_k, full_v
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[2]
        return 0


def _slice_kv_cache(past_key_values: DynamicCache, end: int) -> DynamicCache:
    """Slice KV cache to [0:end] along the sequence dimension."""
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        if end <= 0:
            new_cache.update(k[:, :, :0, :], v[:, :, :0, :], layer_idx)
        else:
            new_cache.update(k[:, :, :end, :], v[:, :, :end, :], layer_idx)
    return new_cache


def _expand_kv_cache(past_key_values: DynamicCache, batch_size: int) -> ReadOnlyCache:
    """Expand KV cache batch dimension from 1 to batch_size via broadcasting."""
    expanded_keys = []
    expanded_values = []
    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        expanded_keys.append(k.expand(batch_size, -1, -1, -1))
        expanded_values.append(v.expand(batch_size, -1, -1, -1))
    return ReadOnlyCache(expanded_keys, expanded_values)


# ---------------------------------------------------------------------------
# Context Construction (matching extract_policy_generate.py exactly)
# ---------------------------------------------------------------------------

def create_tokenized_message(role: str, tokenized_content: List[int]) -> List[int]:
    role_token_id = {"system": SYSTEM_TOKEN_ID, "user": USER_TOKEN_ID, "assistant": ASSISTANT_TOKEN_ID}[role]
    content = list(tokenized_content)
    if content and content[-1] == IM_END_TOKEN_ID:
        content = content[:-1]
    return [IM_START_TOKEN_ID, role_token_id, NEWLINE_TOKEN_ID] + content + [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]


def _make_traj_block(traj_content: List[int]) -> Tuple[List[int], List[int], List[int]]:
    header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
    has_eos = bool(traj_content) and (traj_content[-1] == IM_END_TOKEN_ID)
    footer = [] if has_eos else [IM_END_TOKEN_ID]
    block = header + list(traj_content) + footer
    return header, footer, block


def _feedback_tokens(tokenizer: Any, reward: float, content_plus_footer_len: int) -> List[int]:
    s = f"Reward: {float(reward)}\nLength: {int(content_plus_footer_len)} tokens"
    return tokenizer.encode(s, add_special_tokens=False)


@dataclass
class Rollout:
    response_ids: List[int]
    reward: float
    prompt_token_ids: List[int] = None


def _pack_context_for_critic(
    tokenizer: Any,
    prompt_text: str,
    context_rollouts: List[Rollout],
    target_rollout: Rollout,
    max_length: int,
    rng: random.Random,
) -> Tuple[List[int], int]:
    """
    Build a packed context sequence for the critic, matching training format exactly.

    Returns:
        prefix_ids: The packed context ending with target header
        target_start_pos: Position where target response tokens start
    """
    # 1. User prompt
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prefix_ids = create_tokenized_message("user", prompt_tokens)

    # 2. Context rollouts with feedback (excluding target)
    ctx_rollouts = [r for r in context_rollouts if r is not target_rollout]
    context_indices = list(range(len(ctx_rollouts)))
    rng.shuffle(context_indices)

    # Target header
    target_header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]

    for ridx in context_indices:
        r = ctx_rollouts[ridx]
        content = list(r.response_ids)

        header, footer_r, block = _make_traj_block(content)
        content_plus_footer_len = len(content) + len(footer_r)

        fb_tokens = _feedback_tokens(tokenizer, r.reward, content_plus_footer_len)
        fb_msg = create_tokenized_message("user", fb_tokens)

        add_len = len(block) + len(fb_msg)
        if len(prefix_ids) + add_len + len(target_header) > max_length // 2:
            break

        prefix_ids.extend(block)
        prefix_ids.extend(fb_msg)

    # 3. Target header
    prefix_ids.extend(target_header)
    target_start_pos = len(prefix_ids)

    return prefix_ids, target_start_pos


# ---------------------------------------------------------------------------
# Q-Value Computation (matching extract_policy_generate.py exactly)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_candidate_q_values(
    ref_model: Any,
    critic_model: Any,
    tokenizer: Any,
    rollout: Rollout,
    context_rollouts: List[Rollout],
    prompt_text: str,
    cfg: Config,
    device: torch.device,
    amp_dtype: torch.dtype,
    rng: random.Random,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
    """
    Compute Q-values and reference log probs for all min-p filtered candidates at each position.

    Returns:
        candidate_ids: List of candidate token IDs for each position
        candidate_q_values: List of Q-values for each candidate at each position
        candidate_ref_logprobs: List of reference log probs for each candidate at each position
    """
    response_ids = list(rollout.response_ids)
    prompt_ids = list(rollout.prompt_token_ids) if rollout.prompt_token_ids else []

    if not response_ids:
        return [], [], []

    # Build context for critic
    critic_prefix, _ = _pack_context_for_critic(
        tokenizer, prompt_text, context_rollouts, rollout, cfg.max_length, rng
    )

    # Reference model: get logits for all positions
    ref_seq = prompt_ids + response_ids
    ref_input = torch.tensor([ref_seq], dtype=torch.long, device=device)

    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
        ref_out = ref_model(input_ids=ref_input, use_cache=False, return_dict=True)
        ref_logits = ref_out.logits  # [1, S, V]

    # Critic model: build KV cache for the prefix
    critic_prefix_input = torch.tensor([critic_prefix], dtype=torch.long, device=device)
    base = _get_base_model(_unwrap_model(critic_model))

    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
        if base is None:
            critic_prefix_out = critic_model(
                input_ids=critic_prefix_input,
                use_cache=True,
                return_dict=True,
            )
            critic_kv = critic_prefix_out.past_key_values
        else:
            critic_prefix_out = base(
                input_ids=critic_prefix_input,
                use_cache=True,
                return_dict=True,
            )
            critic_kv = critic_prefix_out.past_key_values

    del critic_prefix_input, critic_prefix_out

    # Prepare for Q-value extraction
    m = _unwrap_model(critic_model)
    lm_head = _get_lm_head(m)
    w = lm_head.weight[cfg.distribution_token_id : cfg.distribution_token_id + num_bins]
    b = lm_head.bias[cfg.distribution_token_id : cfg.distribution_token_id + num_bins] if hasattr(lm_head, "bias") and lm_head.bias is not None else None
    num_reward_states = num_bins // num_length_bins

    all_candidate_ids: List[List[int]] = []
    all_candidate_q_values: List[List[float]] = []
    all_candidate_ref_logprobs: List[List[float]] = []

    # Batch size for candidate processing
    CAND_BATCH_SIZE = 32

    for i, tid in enumerate(response_ids):
        # Reference position: position before this token
        ref_pos = len(prompt_ids) + i - 1
        if ref_pos < 0:
            ref_pos = 0

        # Get reference probs
        pos_logits = ref_logits[0, ref_pos, :]
        ref_log_probs_all = F.log_softmax(pos_logits.float(), dim=-1)
        ref_probs_all = ref_log_probs_all.exp()

        # Min-p filtering
        max_prob = ref_probs_all.max()
        threshold = cfg.min_p * max_prob
        mask = ref_probs_all >= threshold
        cand_indices = mask.nonzero(as_tuple=True)[0].tolist()

        # Always include actual token
        if tid not in cand_indices:
            cand_indices.append(tid)

        num_cands = len(cand_indices)

        # Get reference log probs for candidates
        cand_ref_logprobs: List[float] = [ref_log_probs_all[c].item() for c in cand_indices]

        # Compute Q-values for ALL candidates
        cand_q_values: List[float] = []

        # Slice KV cache to current position
        current_kv = _slice_kv_cache(critic_kv, end=critic_kv.get_seq_length())

        for batch_start in range(0, num_cands, CAND_BATCH_SIZE):
            batch_end = min(batch_start + CAND_BATCH_SIZE, num_cands)
            batch_cand_indices = cand_indices[batch_start:batch_end]
            batch_size = len(batch_cand_indices)

            expanded_kv = _expand_kv_cache(current_kv, batch_size=batch_size)
            cand_input = torch.tensor(batch_cand_indices, dtype=torch.long, device=device).unsqueeze(1)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                if base is None:
                    cand_out = critic_model(
                        input_ids=cand_input,
                        past_key_values=expanded_kv,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    cand_hidden = cand_out.hidden_states[-1]
                else:
                    cand_out = base(
                        input_ids=cand_input,
                        past_key_values=expanded_kv,
                        use_cache=False,
                        return_dict=True,
                    )
                    cand_hidden = cand_out.last_hidden_state if hasattr(cand_out, "last_hidden_state") else cand_out.hidden_states[-1]

            # Extract Q-values
            for j in range(batch_size):
                h = cand_hidden[j, 0, :]
                logits_h = F.linear(h, w, b).float()
                logits_rs = logits_h.view(num_reward_states, num_length_bins)
                logits_reward = torch.logsumexp(logits_rs, dim=1)
                probs_reward = torch.softmax(logits_reward, dim=0)
                q = probs_reward[correct_reward_index].item()
                cand_q_values.append(q)

            del expanded_kv, cand_input, cand_out, cand_hidden

        del current_kv

        all_candidate_ids.append(cand_indices)
        all_candidate_q_values.append(cand_q_values)
        all_candidate_ref_logprobs.append(cand_ref_logprobs)

        # Update KV cache with actual token for next position
        next_token = torch.tensor([[tid]], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
            if base is None:
                update_out = critic_model(
                    input_ids=next_token,
                    past_key_values=critic_kv,
                    use_cache=True,
                    return_dict=True,
                )
                critic_kv = update_out.past_key_values
            else:
                update_out = base(
                    input_ids=next_token,
                    past_key_values=critic_kv,
                    use_cache=True,
                    return_dict=True,
                )
                critic_kv = update_out.past_key_values
        del next_token, update_out

    del ref_input, ref_out, ref_logits, critic_kv
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_candidate_ids, all_candidate_q_values, all_candidate_ref_logprobs


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _load_data(
    data_path: str,
    reward_col: str,
    min_correct_rate: float = 0.0,
    min_incorrect_rate: float = 0.0,
    only_label_correct: bool = False,
) -> Tuple[pa.Table, Dict[int, Dict[str, Any]]]:
    """
    Load the full parquet table and build prompt groups for processing.

    Args:
        data_path: Path to the parquet file
        reward_col: Column name for reward values
        min_correct_rate: Minimum fraction of correct samples per prompt (0.0-1.0)
        min_incorrect_rate: Minimum fraction of incorrect samples per prompt (0.0-1.0)
        only_label_correct: If True, only include correct samples for labeling

    Returns:
        table: The full PyArrow table with all columns
        prompt_groups: Dict mapping prompt_idx to prompt text and rollouts
    """
    # Load full table to preserve all columns
    table = pq.read_table(data_path)
    df = table.to_pandas()

    if reward_col == "correct" and reward_col in df.columns:
        df["correct"] = df["correct"].astype(float)

    prompt_groups: Dict[int, Dict[str, Any]] = {}
    filtered_prompts = 0
    filtered_incorrect_samples = 0

    grouped = df.groupby("prompt_idx", sort=True)
    for pid, group in grouped:
        rows = group.to_dict("records")
        if not rows:
            continue

        # Calculate correct/incorrect rates for this prompt
        rewards = [float(r[reward_col]) for r in rows]
        num_correct = sum(1 for rw in rewards if rw > 0.5)
        num_incorrect = len(rewards) - num_correct
        correct_rate = num_correct / len(rewards)
        incorrect_rate = num_incorrect / len(rewards)

        # Filter prompts based on correct/incorrect rate thresholds
        if correct_rate < min_correct_rate or incorrect_rate < min_incorrect_rate:
            filtered_prompts += 1
            continue

        prompt_text = rows[0]["prompt"]
        rollouts: List[Rollout] = []
        row_indices: List[int] = []  # Track original row indices
        all_rollouts: List[Rollout] = []  # Keep all rollouts for context

        for idx, r in zip(group.index.tolist(), rows):
            toks = r["output_token_ids"]
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            prompt_toks = r["prompt_token_ids"]
            if hasattr(prompt_toks, "tolist"):
                prompt_toks = prompt_toks.tolist()
            rollout = Rollout(
                response_ids=list(toks),
                reward=float(r[reward_col]),
                prompt_token_ids=list(prompt_toks),
            )
            all_rollouts.append(rollout)

            # Only include correct samples for labeling if flag is set
            if only_label_correct and rollout.reward <= 0.5:
                filtered_incorrect_samples += 1
                continue

            rollouts.append(rollout)
            row_indices.append(idx)

        if rollouts:  # Only include if there are samples to label
            prompt_groups[int(pid)] = {
                "prompt": prompt_text,
                "rollouts": rollouts,
                "row_indices": row_indices,
                "all_rollouts": all_rollouts,  # Keep all rollouts for context
            }

    if filtered_prompts > 0 or filtered_incorrect_samples > 0:
        print(f"[Data] Filtered {filtered_prompts} prompts (correct rate not in [{min_correct_rate:.0%}, {1-min_incorrect_rate:.0%}])")
        print(f"[Data] Filtered {filtered_incorrect_samples} incorrect samples (only_label_correct={only_label_correct})")
        print(f"[Data] Kept {len(prompt_groups)} prompts for labeling")

    return table, prompt_groups


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _print_progress(msg: str, end: str = "\n") -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", end=end, flush=True)


def _worker(rank: int, world: int, cfg: Config) -> None:
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(cfg.seed + rank)
    amp_dtype = _torch_dtype(cfg.dtype)

    _print_progress(f"[Rank {rank}] Loading models...")

    model_kwargs: Dict[str, Any] = {"torch_dtype": amp_dtype, "trust_remote_code": True}
    attn_impl = cfg.attn_implementation
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"
    if attn_impl and attn_impl != "auto":
        model_kwargs["attn_implementation"] = attn_impl

    critic_model = AutoModelForCausalLM.from_pretrained(cfg.critic_path, **model_kwargs)
    critic_model.to(device)
    critic_model.eval()

    ref_model = AutoModelForCausalLM.from_pretrained(cfg.ref_path, **model_kwargs)
    ref_model.to(device)
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.ref_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _print_progress(f"[Rank {rank}] Loading data...")

    # Load data (full table + prompt groups)
    _full_table, prompt_groups = _load_data(
        cfg.in_parquet,
        cfg.reward_col,
        min_correct_rate=cfg.min_correct_rate,
        min_incorrect_rate=cfg.min_incorrect_rate,
        only_label_correct=cfg.only_label_correct,
    )
    prompt_ids_sorted = sorted(prompt_groups.keys())

    # Distribute prompts across workers
    my_prompt_ids = [pid for i, pid in enumerate(prompt_ids_sorted) if i % world == rank]

    if not my_prompt_ids:
        _print_progress(f"[Rank {rank}] No prompts assigned.")
        return

    _print_progress(f"[Rank {rank}] Processing {len(my_prompt_ids)} prompts...")

    # Critic config
    reward_values = [0.0, 1.0]
    length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_length_bins = len(length_bins) - 1
    num_reward_states = len(reward_values)
    num_bins = num_length_bins * num_reward_states
    correct_reward_index = int(max(range(len(reward_values)), key=lambda i: reward_values[i]))

    # Collect results: map row_index -> (candidate_ids, candidate_q_values)
    results: List[Dict[str, Any]] = []

    for local_i, pid in enumerate(my_prompt_ids):
        if cfg.max_groups > 0 and local_i >= cfg.max_groups:
            break

        grp = prompt_groups.get(int(pid), None)
        if grp is None:
            continue

        prompt_text = str(grp["prompt"])
        rollouts_to_label: List[Rollout] = list(grp["rollouts"])  # Only correct samples if only_label_correct
        row_indices: List[int] = list(grp["row_indices"])
        context_rollouts: List[Rollout] = list(grp["all_rollouts"])  # All rollouts for context

        # Limit rollouts per prompt if specified
        if cfg.max_rollouts_per_prompt > 0:
            rollouts_to_label = rollouts_to_label[:cfg.max_rollouts_per_prompt]
            row_indices = row_indices[:cfg.max_rollouts_per_prompt]

        _print_progress(f"[Rank {rank}] Prompt {local_i+1}/{len(my_prompt_ids)} (idx={pid}), {len(rollouts_to_label)} rollouts to label, {len(context_rollouts)} context")

        for rollout_i, (rollout, row_idx) in enumerate(zip(rollouts_to_label, row_indices)):
            seed_i = (cfg.seed * 1000003 + pid * 1009 + rollout_i * 7) & 0xFFFFFFFF
            rng = random.Random(seed_i)

            print(f"\r[Rank {rank}] Rollout {rollout_i+1}/{len(rollouts_to_label)}: {len(rollout.response_ids)} tokens...", end="", flush=True)
            start_time = time.time()

            candidate_ids, candidate_q_values, candidate_ref_logprobs = _compute_candidate_q_values(
                ref_model=ref_model,
                critic_model=critic_model,
                tokenizer=tokenizer,
                rollout=rollout,
                context_rollouts=context_rollouts,
                prompt_text=prompt_text,
                cfg=cfg,
                device=device,
                amp_dtype=amp_dtype,
                rng=rng,
                num_bins=num_bins,
                num_length_bins=num_length_bins,
                correct_reward_index=correct_reward_index,
            )

            elapsed = time.time() - start_time
            print(f" done in {elapsed:.1f}s", flush=True)

            results.append({
                "row_index": row_idx,
                "candidate_ids": candidate_ids,
                "candidate_q_values": candidate_q_values,
                "candidate_ref_logprobs": candidate_ref_logprobs,
            })

    # Write results to parquet shard (just the new columns with row indices)
    if results:
        shard_path = f"{cfg.out_parquet}.part{rank}"

        # Convert to pyarrow table
        table = pa.table({
            "row_index": pa.array([r["row_index"] for r in results], type=pa.int64()),
            "candidate_ids": pa.array([r["candidate_ids"] for r in results], type=pa.list_(pa.list_(pa.int64()))),
            "candidate_q_values": pa.array([r["candidate_q_values"] for r in results], type=pa.list_(pa.list_(pa.float32()))),
            "candidate_ref_logprobs": pa.array([r["candidate_ref_logprobs"] for r in results], type=pa.list_(pa.list_(pa.float32()))),
        })

        pq.write_table(table, shard_path)
        _print_progress(f"[Rank {rank}] Wrote {shard_path} ({len(results)} rollouts)")
    else:
        _print_progress(f"[Rank {rank}] No results to write")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label trajectories with candidate Q-values")
    p.add_argument("--critic_path", required=True, help="Path to critic model")
    p.add_argument("--ref_path", default="Qwen/Qwen3-1.7B", help="Path to reference model")
    p.add_argument("--in_parquet", required=True, help="Input parquet with trajectories")
    p.add_argument("--out_parquet", required=True, help="Output parquet with Q-value labels")

    p.add_argument("--dp_size", type=int, default=0, help="Data parallel size (0=all GPUs)")
    p.add_argument("--max_length", type=int, default=131072, help="Max context length")
    p.add_argument("--min_p", type=float, default=0.01, help="Min-p threshold for candidate filtering")

    p.add_argument("--distribution_token_id", type=int, default=151669)
    p.add_argument("--reward_col", type=str, default="correct")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    p.add_argument("--max_groups", type=int, default=-1, help="Max prompt groups to process (-1 = all)")
    p.add_argument("--max_rollouts_per_prompt", type=int, default=-1, help="Max rollouts per prompt (-1 = all)")

    p.add_argument("--min_correct_rate", type=float, default=0.25, help="Min fraction of correct samples per prompt (0.0-1.0)")
    p.add_argument("--min_incorrect_rate", type=float, default=0.25, help="Min fraction of incorrect samples per prompt (0.0-1.0)")
    p.add_argument("--only_label_correct", action="store_true", help="Only label correct samples")

    return p.parse_args()


def _merge_shards(in_parquet: str, out_parquet: str, dp_size: int) -> bool:
    """
    Merge Q-value label shards with the original input parquet.

    The shards contain (row_index, candidate_ids, candidate_q_values).
    We join these with the original data by row index.
    Only rows that were labeled are included in the output.
    """
    part_paths = [f"{out_parquet}.part{r}" for r in range(dp_size) if os.path.exists(f"{out_parquet}.part{r}")]

    if not part_paths:
        return False

    # Read original table
    original_table = pq.read_table(in_parquet)
    original_df = original_table.to_pandas()

    # Read and concatenate all shards
    shard_tables = [pq.read_table(p) for p in part_paths]
    labels_table = pa.concat_tables(shard_tables)
    labels_df = labels_table.to_pandas()

    # Get the set of labeled row indices
    labeled_indices = set(labels_df["row_index"].tolist())

    # Filter original dataframe to only include labeled rows
    original_df = original_df.loc[list(labeled_indices)].copy()

    # Create mapping from row_index to new columns
    row_to_labels = {}
    for _, row in labels_df.iterrows():
        row_to_labels[row["row_index"]] = {
            "candidate_ids": row["candidate_ids"],
            "candidate_q_values": row["candidate_q_values"],
            "candidate_ref_logprobs": row["candidate_ref_logprobs"],
        }

    # Add new columns
    original_df["candidate_ids"] = original_df.index.map(lambda idx: row_to_labels[idx]["candidate_ids"])
    original_df["candidate_q_values"] = original_df.index.map(lambda idx: row_to_labels[idx]["candidate_q_values"])
    original_df["candidate_ref_logprobs"] = original_df.index.map(lambda idx: row_to_labels[idx]["candidate_ref_logprobs"])

    # Reset index and convert back to pyarrow
    original_df = original_df.reset_index(drop=True)
    output_table = pa.Table.from_pandas(original_df, preserve_index=False)
    pq.write_table(output_table, out_parquet)

    # Clean up shards
    for p in part_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return True


def main() -> None:
    args = parse_args()
    script_start = time.time()

    ngpus = torch.cuda.device_count()
    if args.dp_size == 0:
        dp = max(1, ngpus)
    else:
        dp = args.dp_size

    if ngpus > 0 and dp > ngpus:
        raise RuntimeError(f"Requested dp_size={dp}, but only {ngpus} GPUs visible.")
    if ngpus == 0 and dp != 1:
        raise RuntimeError("No CUDA devices visible; use --dp_size 1 for CPU.")

    cfg = Config(
        critic_path=args.critic_path,
        ref_path=args.ref_path,
        in_parquet=args.in_parquet,
        out_parquet=args.out_parquet,
        dp_size=dp,
        max_length=args.max_length,
        min_p=args.min_p,
        distribution_token_id=args.distribution_token_id,
        reward_col=args.reward_col,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_groups=args.max_groups,
        max_rollouts_per_prompt=args.max_rollouts_per_prompt,
        min_correct_rate=args.min_correct_rate,
        min_incorrect_rate=args.min_incorrect_rate,
        only_label_correct=args.only_label_correct,
    )

    _print_progress(
        f"Labeling trajectories with Q-values:\n"
        f"  critic_path: {cfg.critic_path}\n"
        f"  ref_path: {cfg.ref_path}\n"
        f"  in_parquet: {cfg.in_parquet}\n"
        f"  out_parquet: {cfg.out_parquet}\n"
        f"  dp_size: {dp}\n"
        f"  min_p: {cfg.min_p}\n"
        f"  min_correct_rate: {cfg.min_correct_rate}\n"
        f"  min_incorrect_rate: {cfg.min_incorrect_rate}\n"
        f"  only_label_correct: {cfg.only_label_correct}\n"
    )

    if dp == 1:
        _worker(0, 1, cfg)
    else:
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
                print(f"[main] Worker {r} failed with code {p.exitcode}", flush=True)

        if any_fail:
            raise RuntimeError("One or more workers failed")

    # Merge shards with original data
    _print_progress("Merging shards with original data...")
    if _merge_shards(args.in_parquet, args.out_parquet, dp):
        total_time = time.time() - script_start
        _print_progress(f"Done! Wrote {args.out_parquet} in {total_time:.1f}s")
    else:
        _print_progress("No output produced")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
