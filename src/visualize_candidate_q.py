#!/usr/bin/env python3
"""
Visualize Q-values for candidate tokens at each position.

For each position in a trajectory:
  - Use reference model to determine candidate tokens (min_p filtering)
  - Use critic model to evaluate Q-values for each candidate
  - Display tokens colored by Q-value with hover showing all candidates
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import random
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass
class Config:
    """Configuration passed to worker processes."""
    critic_path: str
    ref_path: str
    data_path: str
    out_html: str
    num_prompts: int
    prompt_idx: Optional[List[int]]
    min_p: float
    max_candidates_show: int
    distribution_token_id: int
    label_column: str
    max_length: int
    seed: int
    dtype: str
    attn_implementation: str
    tokenizer_path: Optional[str]
    samples_per_prompt: int
    max_tokens_per_rollout: int
    require_mixed_outcomes: bool
    min_correct_pct: float
    min_incorrect_pct: float
    dp_size: int
    # Computed fields (set by main before spawning)
    selected_prompt_ids: List[int] = None
    reward_col: str = "correct"

# Qwen chat token ids
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
SYSTEM_TOKEN_ID = 8948
NEWLINE_TOKEN_ID = 198


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
# KV Cache Helpers (using DynamicCache API)
# ---------------------------------------------------------------------------

class ReadOnlyCache(DynamicCache):
    """A cache wrapper that ignores update() calls, making it read-only.

    This allows using .expand() views without the model corrupting them
    when it calls cache.update() during forward pass.
    """

    def __init__(self, key_cache: List[torch.Tensor], value_cache: List[torch.Tensor]):
        super().__init__()
        self.key_cache = key_cache
        self.value_cache = value_cache
        # Set _seen_tokens to match the sequence length
        if key_cache:
            self._seen_tokens = key_cache[0].shape[2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated k/v but don't actually store the update."""
        # Return what the model expects (full key/value for attention)
        # but don't modify our stored cache
        if layer_idx < len(self.key_cache):
            full_k = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            full_v = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            return full_k, full_v
        # Fallback for unexpected layer
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
    """Expand KV cache batch dimension from 1 to batch_size via broadcasting.

    Returns a ReadOnlyCache that ignores update() calls, allowing fast
    .expand() views without memory copying.
    """
    expanded_keys = []
    expanded_values = []
    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        # expand() creates a view - no memory copy, very fast
        expanded_keys.append(k.expand(batch_size, -1, -1, -1))
        expanded_values.append(v.expand(batch_size, -1, -1, -1))
    return ReadOnlyCache(expanded_keys, expanded_values)


# ---------------------------------------------------------------------------
# Reference Model: Candidate Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_candidates_at_positions(
    model: Any,
    input_ids: torch.Tensor,  # [1, S]
    positions: List[int],
    actual_token_ids: List[int],  # The actual tokens at each position (must be included)
    min_p: float,
    device: torch.device,
) -> List[List[Tuple[int, float]]]:
    """
    Get candidate tokens at each position using min_p filtering.

    Always includes the actual token in the candidate set, even if it
    doesn't pass the min_p threshold.

    Returns: List of [(token_id, log_prob), ...] for each position.
    """
    if not positions:
        return []

    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = outputs.logits  # [1, S, V]

    # Get log probs at positions before each target (predicting the target)
    # Position p in the sequence predicts token at p+1
    # So to get candidates for position p, we look at logits at p-1
    results: List[List[Tuple[int, float]]] = []

    for pos_idx, pos in enumerate(positions):
        if pos <= 0:
            results.append([])
            continue

        actual_tid = actual_token_ids[pos_idx]

        # Logits at pos-1 predict token at pos
        pos_logits = logits[0, pos - 1, :]  # [V]
        log_probs = F.log_softmax(pos_logits.float(), dim=-1)
        probs = log_probs.exp()

        # Apply min_p filtering
        max_prob = probs.max()
        threshold = min_p * max_prob
        mask = probs >= threshold

        indices = mask.nonzero(as_tuple=True)[0]
        cand_ids_set = set(indices.tolist())

        # Always include the actual token
        was_in_set = actual_tid in cand_ids_set
        cand_ids_set.add(actual_tid)

        cand_ids = list(cand_ids_set)
        cand_lps = log_probs[cand_ids].tolist()

        # Sort by probability (descending)
        candidates = sorted(zip(cand_ids, cand_lps), key=lambda x: -x[1])

        # Debug: check if actual token is in final candidates (first position only)
        if pos_idx == 0:
            actual_in_candidates = any(c[0] == actual_tid for c in candidates)
            top_cand_ids = [c[0] for c in candidates[:5]]
            print(f"[DEBUG ref-model] pos_idx=0: actual_tid={actual_tid}, was_in_set={was_in_set}, "
                  f"actual_in_candidates={actual_in_candidates}, num_candidates={len(candidates)}, "
                  f"top_cand_ids={top_cand_ids}", flush=True)

        results.append(candidates)

    return results


# ---------------------------------------------------------------------------
# Critic Model: Q-Value Evaluation with KV Cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate_q_values_with_kv_cache(
    model: Any,
    input_ids: torch.Tensor,  # [1, S]
    positions: List[int],
    actual_token_ids: List[int],  # The actual tokens at each position
    candidates_per_position: List[List[Tuple[int, float]]],  # [(token_id, ref_logprob), ...]
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    device: torch.device,
) -> Tuple[List[float], List[List[Tuple[int, float, float]]], int]:
    """
    Evaluate Q-values for candidates at each position using KV cache.

    Optimization: We already have Q-values for actual tokens from the initial
    forward pass, so we only need to evaluate non-actual candidates.

    Returns:
        actual_q_values: Q-value for the actual token at each position
        candidate_results: [[(token_id, ref_logprob, q_value), ...], ...] per position
        num_extra_forward_passes: Count of positions requiring additional forward passes
    """
    if not positions:
        return ([], [], 0)

    # Step 1: Run full sequence with KV cache
    base = _get_base_model(_unwrap_model(model))
    if base is None:
        out = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        past_key_values = out.past_key_values
    else:
        out = base(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
        past_key_values = out.past_key_values

    # Step 2: Get actual token Q-values from hidden states
    pos_t = torch.tensor(positions, device=device, dtype=torch.long)
    h_sel = hidden[0].index_select(0, pos_t)  # [T, E]

    m = _unwrap_model(model)
    lm_head = _get_lm_head(m)
    w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
    b = lm_head.bias[distribution_token_id : distribution_token_id + num_bins] if hasattr(lm_head, "bias") and lm_head.bias is not None else None

    logits = F.linear(h_sel, w, b).float()  # [T, num_bins]
    num_reward_states = num_bins // num_length_bins
    logits_rs = logits.view(-1, num_reward_states, num_length_bins)
    logits_reward = torch.logsumexp(logits_rs, dim=2)
    probs_reward = torch.softmax(logits_reward, dim=1)
    actual_q = probs_reward[:, correct_reward_index].cpu().tolist()

    # Step 3: Evaluate only non-actual candidates at each position
    candidate_results: List[List[Tuple[int, float, float]]] = []
    num_extra_forward_passes = 0

    for pos_idx, (pos, candidates) in enumerate(zip(positions, candidates_per_position)):
        if not candidates:
            candidate_results.append([])
            continue

        actual_tid = actual_token_ids[pos_idx]

        # Separate actual token from other candidates
        actual_entry = None
        other_candidates = []
        for cand_id, cand_lp in candidates:
            if cand_id == actual_tid:
                # Use pre-computed Q-value for actual token
                actual_entry = (cand_id, cand_lp, actual_q[pos_idx])
            else:
                other_candidates.append((cand_id, cand_lp))

        # Debug for first position
        if pos_idx == 0:
            cand_ids_in_input = [c[0] for c in candidates]
            print(f"[DEBUG Q-eval] pos_idx=0: actual_tid={actual_tid}, actual_entry={actual_entry is not None}, "
                  f"candidates_count={len(candidates)}, cand_ids={cand_ids_in_input[:10]}...", flush=True)

        # If no other candidates, just use the actual token's pre-computed Q
        if not other_candidates:
            if actual_entry is not None:
                candidate_results.append([actual_entry])
            else:
                # Edge case: actual token not in candidates (shouldn't happen with proper min_p)
                # Fall back to just the actual Q
                candidate_results.append([(actual_tid, 0.0, actual_q[pos_idx])])
            continue

        # We have other candidates - need to evaluate them
        num_extra_forward_passes += 1

        other_ids = [c[0] for c in other_candidates]
        other_lps = [c[1] for c in other_candidates]
        K = len(other_ids)

        # Slice KV cache to prefix [0:pos]
        sliced_kv = _slice_kv_cache(past_key_values, end=pos)
        expanded_kv = _expand_kv_cache(sliced_kv, batch_size=K)

        # Create input for other candidates: [K, 1]
        cand_input = torch.tensor(other_ids, dtype=torch.long, device=device).unsqueeze(1)

        # Forward pass with other candidates only
        if base is None:
            cand_out = model(
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

        # Extract Q-values for other candidates
        cand_h = cand_hidden.squeeze(1)  # [K, E]
        cand_logits = F.linear(cand_h, w, b).float()
        cand_logits_rs = cand_logits.view(-1, num_reward_states, num_length_bins)
        cand_logits_reward = torch.logsumexp(cand_logits_rs, dim=2)
        cand_probs_reward = torch.softmax(cand_logits_reward, dim=1)
        other_q = cand_probs_reward[:, correct_reward_index].cpu().tolist()

        # Combine: actual token (with pre-computed Q) + other candidates
        pos_results = [(other_ids[i], other_lps[i], other_q[i]) for i in range(K)]
        if actual_entry is not None:
            pos_results.append(actual_entry)

        # Sort by Q-value descending
        pos_results.sort(key=lambda x: -x[2])
        candidate_results.append(pos_results)

        # Cleanup intermediate caches to free memory
        del sliced_kv, expanded_kv, cand_input, cand_out, cand_hidden, cand_h
        del cand_logits, cand_logits_rs, cand_logits_reward, cand_probs_reward

    # Cleanup main cache
    del past_key_values, hidden, out

    return (actual_q, candidate_results, num_extra_forward_passes)


# ---------------------------------------------------------------------------
# Sequence Packing (same as visualize_critic.py)
# ---------------------------------------------------------------------------

def create_tokenized_message(role: str, tokenized_content: List[int]) -> List[int]:
    role_token_id = {"system": SYSTEM_TOKEN_ID, "user": USER_TOKEN_ID, "assistant": ASSISTANT_TOKEN_ID}[role]
    content = list(tokenized_content)
    if content and content[-1] == IM_END_TOKEN_ID:
        content = content[:-1]
    return [IM_START_TOKEN_ID, role_token_id, NEWLINE_TOKEN_ID] + content + [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]


@dataclass
class Rollout:
    response_ids: List[int]
    reward: float
    prompt_token_ids: List[int] = None  # Original prompt tokens for on-policy reference eval


@dataclass
class PackedExample:
    input_ids: torch.Tensor
    target_token_ids: List[int]
    target_positions: List[int]
    context_rollout_indices: List[int]
    target_truncated: bool


def _make_traj_block(traj_content: List[int]) -> Tuple[List[int], List[int], List[int]]:
    header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
    has_eos = bool(traj_content) and (traj_content[-1] == IM_END_TOKEN_ID)
    footer = [] if has_eos else [IM_END_TOKEN_ID]
    block = header + list(traj_content) + footer
    return header, footer, block


def _feedback_tokens(tokenizer: Any, reward: float, content_plus_footer_len: int) -> List[int]:
    s = f"Reward: {float(reward)}\nLength: {int(content_plus_footer_len)} tokens"
    return tokenizer.encode(s, add_special_tokens=False)


def _pack_for_target(
    tokenizer: Any,
    prompt_text: str,
    rollouts: List[Rollout],
    target_idx: int,
    max_length: int,
    rng: random.Random,
) -> PackedExample:
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prefix_ids = create_tokenized_message("user", prompt_tokens)
    prefix_len = len(prefix_ids)

    tgt = rollouts[target_idx]
    tgt_content = list(tgt.response_ids)
    tgt_header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]

    def build_target_block(content: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
        has_eos = bool(content) and (content[-1] == IM_END_TOKEN_ID)
        footer = [] if has_eos else [IM_END_TOKEN_ID]
        token_ids = list(content) + list(footer)
        block = tgt_header + list(content) + list(footer)
        return footer, token_ids, block, content

    footer, tgt_token_ids, tgt_block, tgt_content_used = build_target_block(tgt_content)
    fixed_len = prefix_len + len(tgt_block)

    target_truncated = False
    if fixed_len > int(max_length):
        budget = int(max_length) - prefix_len
        min_block = len(tgt_header) + 1
        if budget < min_block:
            return PackedExample(
                input_ids=torch.tensor(prefix_ids[: int(max_length)], dtype=torch.long),
                target_token_ids=[],
                target_positions=[],
                context_rollout_indices=[],
                target_truncated=True,
            )
        max_content_len = budget - len(tgt_header) - 1
        if max_content_len < 0:
            max_content_len = 0
        trunc_content = list(tgt_content[:max_content_len])
        footer, tgt_token_ids, tgt_block, tgt_content_used = build_target_block(trunc_content)
        target_truncated = True
        fixed_len = prefix_len + len(tgt_block)

    other = [i for i in range(len(rollouts)) if i != target_idx]
    rng.shuffle(other)

    context_indices: List[int] = []
    context_blocks: List[Tuple[int, List[int], List[int]]] = []

    total_len = fixed_len
    for ridx in other:
        r = rollouts[ridx]
        content = list(r.response_ids)
        header, footer_r, block = _make_traj_block(content)
        content_plus_footer_len = len(content) + len(footer_r)

        fb_tokens = _feedback_tokens(tokenizer, r.reward, content_plus_footer_len)
        fb_msg = create_tokenized_message("user", fb_tokens)

        add_len = len(block) + len(fb_msg)
        if total_len + add_len > int(max_length):
            break

        context_indices.append(ridx)
        context_blocks.append((ridx, block, fb_msg))
        total_len += add_len

    full_ids: List[int] = []
    full_ids.extend(prefix_ids)

    for _, block, fb_msg in context_blocks:
        full_ids.extend(block)
        full_ids.extend(fb_msg)

    target_start = len(full_ids)
    full_ids.extend(tgt_block)

    target_positions = [target_start + len(tgt_header) + i for i in range(len(tgt_token_ids))]

    return PackedExample(
        input_ids=torch.tensor(full_ids, dtype=torch.long),
        target_token_ids=tgt_token_ids,
        target_positions=target_positions,
        context_rollout_indices=context_indices,
        target_truncated=target_truncated,
    )


# ---------------------------------------------------------------------------
# HTML Generation
# ---------------------------------------------------------------------------

def _reward_rgb(r: float) -> str:
    x = max(0.0, min(1.0, float(r)))
    rr = int((1.0 - x) * 255.0)
    gg = int(x * 255.0)
    return f"rgb({rr},{gg},0)"


def _decode_token(tokenizer: Any, tid: int, cache: Dict[int, str]) -> str:
    if tid in cache:
        return cache[tid]
    s = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    if s == "":
        s = f"<{int(tid)}>"
    cache[tid] = s
    return s


def _html_header(title: str) -> str:
    t = html_lib.escape(title)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }}
h1,h2,h3 {{ margin: 12px 0 8px; }}
pre {{ background: #f6f6f6; padding: 10px; border: 1px solid #ddd; overflow-x: auto; white-space: pre-wrap; }}
.meta {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; background: #fbfbfb; border: 1px solid #ddd; padding: 10px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; align-items: start; }}
.card {{ border: 1px solid #ddd; padding: 10px; }}
.card .hdr {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; margin-bottom: 6px; }}
.toks {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; line-height: 1.4; }}
.tok {{ padding: 0 1px; border-radius: 2px; cursor: pointer; position: relative; }}
.tok:hover {{ outline: 2px solid #333; }}
.legend {{ display:flex; align-items:center; gap:10px; margin: 10px 0; }}
.bar {{ width: 220px; height: 14px; border: 1px solid #bbb; background: linear-gradient(to right, rgb(255,0,0), rgb(0,255,0)); }}
.small {{ font-size: 12px; color: #333; }}
hr {{ border: 0; border-top: 1px solid #ddd; margin: 16px 0; }}

/* Tooltip styles */
.tooltip {{
  display: none;
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  background: #222;
  color: #fff;
  padding: 8px;
  border-radius: 4px;
  font-size: 11px;
  white-space: nowrap;
  z-index: 1000;
  max-height: 300px;
  overflow-y: auto;
}}
.tok:hover .tooltip {{ display: block; }}
.tooltip table {{ border-collapse: collapse; }}
.tooltip th, .tooltip td {{ padding: 2px 6px; text-align: left; border-bottom: 1px solid #444; }}
.tooltip th {{ color: #aaa; }}
.tooltip .actual {{ background: #363; }}
</style>
</head>
<body>
<h1>{t}</h1>
<div class="legend">
  <div class="bar"></div>
  <div class="small">Q-value: 0 (red) → 1 (green)</div>
</div>
"""


def _html_footer() -> str:
    return "</body>\n</html>\n"


def _make_tooltip_html(
    tokenizer: Any,
    actual_tid: int,
    candidates: List[Tuple[int, float, float]],  # [(token_id, ref_logprob, q_value), ...]
    decode_cache: Dict[int, str],
    max_show: int = 15,
) -> str:
    """Generate HTML for tooltip showing candidates."""
    lines = ['<div class="tooltip"><table>']
    lines.append("<tr><th>Token</th><th>Ref LP</th><th>Q</th></tr>")

    shown = 0
    for tid, ref_lp, q_val in candidates[:max_show]:
        tok_str = html_lib.escape(_decode_token(tokenizer, tid, decode_cache))
        is_actual = (tid == actual_tid)
        row_class = ' class="actual"' if is_actual else ""
        marker = " *" if is_actual else ""
        lines.append(
            f"<tr{row_class}><td>{tok_str}{marker}</td>"
            f"<td>{ref_lp:.2f}</td><td>{q_val:.3f}</td></tr>"
        )
        shown += 1

    if len(candidates) > max_show:
        lines.append(f"<tr><td colspan='3'>... +{len(candidates) - max_show} more</td></tr>")

    lines.append("</table></div>")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--critic_path", required=True, help="Path to critic model")
    p.add_argument("--ref_path", default="Qwen/Qwen3-1.7B", help="Path to reference model (for candidate extraction)")
    p.add_argument("--data_path", required=True, help="Path to parquet data")
    p.add_argument("--out_html", required=True, help="Output HTML file")

    p.add_argument("--num_prompts", type=int, default=10)
    p.add_argument("--prompt_idx", type=int, nargs="*", default=None)
    p.add_argument("--examples_per_prompt", type=int, default=1)

    p.add_argument("--min_p", type=float, default=0.05, help="Min-p threshold for candidate filtering")
    p.add_argument("--max_candidates_show", type=int, default=15, help="Max candidates to show in tooltip")

    p.add_argument("--distribution_token_id", type=int, default=151669)
    p.add_argument("--label_column", choices=["correct", "value"], default="correct")
    p.add_argument("--max_length", type=int, default=131072)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--samples_per_prompt", type=int, default=0, help="Max rollouts to process per prompt (0=all)")
    p.add_argument("--max_tokens_per_rollout", type=int, default=0)
    p.add_argument("--require_mixed_outcomes", action="store_true",
                   help="Only include prompts with both correct and incorrect responses")
    p.add_argument("--min_correct_pct", type=float, default=10.0,
                   help="Minimum percent correct required (0-100, used with --require_mixed_outcomes)")
    p.add_argument("--min_incorrect_pct", type=float, default=10.0,
                   help="Minimum percent incorrect required (0-100, used with --require_mixed_outcomes)")
    p.add_argument("--dp_size", type=int, default=0, help="Data parallel size (0=use all GPUs)")

    return p.parse_args()


def _print_progress(msg: str, end: str = "\n") -> None:
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", end=end, flush=True)


def _has_mixed_outcomes(
    rollouts: List[Rollout],
    threshold: float = 0.5,
    min_correct_pct: float = 0.0,
    min_incorrect_pct: float = 0.0,
) -> bool:
    """Check if rollouts have sufficient correct and incorrect responses.

    Args:
        rollouts: List of rollouts to check
        threshold: Reward threshold for correct (>= threshold) vs incorrect (< threshold)
        min_correct_pct: Minimum percentage of correct responses required (0-100)
        min_incorrect_pct: Minimum percentage of incorrect responses required (0-100)
    """
    if len(rollouts) < 2:
        return False

    rewards = [r.reward for r in rollouts]
    num_correct = sum(1 for r in rewards if r >= threshold)
    num_incorrect = len(rewards) - num_correct

    correct_pct = 100.0 * num_correct / len(rewards)
    incorrect_pct = 100.0 * num_incorrect / len(rewards)

    return correct_pct >= min_correct_pct and incorrect_pct >= min_incorrect_pct


def _load_prompt_groups(cfg: Config) -> Dict[int, Dict[str, Any]]:
    """Load and parse prompt groups from parquet file."""
    table = pq.read_table(cfg.data_path, columns=["prompt_idx", "prompt", "prompt_token_ids", "output_token_ids", cfg.reward_col])
    df = table.to_pandas()

    if cfg.reward_col == "correct":
        df["correct"] = df["correct"].astype(float)

    prompt_groups: Dict[int, Dict[str, Any]] = {}
    grouped = df.groupby("prompt_idx", sort=True)
    for pid, group in grouped:
        rows = group.to_dict("records")
        if not rows:
            continue
        prompt_text = rows[0]["prompt"]
        ro: List[Rollout] = []
        for r in rows:
            toks = r["output_token_ids"]
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            prompt_toks = r["prompt_token_ids"]
            if hasattr(prompt_toks, "tolist"):
                prompt_toks = prompt_toks.tolist()
            ro.append(Rollout(
                response_ids=list(toks),
                reward=float(r[cfg.reward_col]),
                prompt_token_ids=list(prompt_toks),
            ))
        prompt_groups[int(pid)] = {"prompt": prompt_text, "rollouts": ro}

    return prompt_groups


def _worker(rank: int, world: int, cfg: Config, fragment_dir: str) -> None:
    """Worker process that handles a shard of prompts."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    torch.manual_seed(int(cfg.seed) + rank)
    amp_dtype = _torch_dtype(cfg.dtype)

    _print_progress(f"[Rank {rank}] Loading models on {device}...")

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

    tok_src = cfg.tokenizer_path if cfg.tokenizer_path else cfg.ref_path
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    _print_progress(f"[Rank {rank}] Models loaded, loading data...")

    # Load data
    prompt_groups = _load_prompt_groups(cfg)

    # Get assigned prompts for this worker
    all_prompt_ids = cfg.selected_prompt_ids
    my_prompt_ids = [pid for i, pid in enumerate(all_prompt_ids) if i % world == rank]

    if not my_prompt_ids:
        _print_progress(f"[Rank {rank}] No prompts assigned, exiting.")
        # Write empty fragment
        fragment_path = os.path.join(fragment_dir, f"fragment_{rank:04d}.html")
        with open(fragment_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    _print_progress(f"[Rank {rank}] Processing {len(my_prompt_ids)} prompts...")

    # Critic config
    if cfg.reward_col == "correct":
        reward_values = [0.0, 1.0]
    else:
        reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]

    length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_length_bins = len(length_bins) - 1
    num_reward_states = len(reward_values)
    num_bins = num_length_bins * num_reward_states
    correct_reward_index = int(max(range(len(reward_values)), key=lambda i: reward_values[i]))

    decode_cache: Dict[int, str] = {}
    parts: List[str] = []
    processed_rollouts = 0

    for local_i, pid in enumerate(my_prompt_ids):
        # Find global index for display
        global_i = all_prompt_ids.index(pid) + 1

        grp = prompt_groups.get(int(pid), None)
        if grp is None:
            continue

        prompt_text = str(grp["prompt"])
        all_rollouts: List[Rollout] = list(grp["rollouts"])  # Keep all for context

        # Determine which rollouts to visualize (but use all for context)
        if int(cfg.samples_per_prompt) > 0:
            num_to_visualize = min(len(all_rollouts), int(cfg.samples_per_prompt))
        else:
            num_to_visualize = len(all_rollouts)

        _print_progress(f"[Rank {rank}] Prompt {local_i+1}/{len(my_prompt_ids)} (global {global_i}, idx={pid}): {num_to_visualize} rollouts to visualize, {len(all_rollouts)} total for context")

        parts.append("<hr>")
        parts.append(f"<h2>prompt {global_i} | prompt_idx {int(pid)} | visualizing {num_to_visualize}/{len(all_rollouts)} rollouts</h2>")
        parts.append("<h3>prompt</h3>")
        parts.append(f"<pre>{html_lib.escape(prompt_text)}</pre>")

        parts.append("<h3>rollouts</h3>")
        parts.append('<div class="grid">')

        for r_i in range(num_to_visualize):
            rollout_start = time.time()
            processed_rollouts += 1

            seed_i = (int(cfg.seed) * 1000003 + int(pid) * 1009 + int(r_i) * 9176) & 0xFFFFFFFF
            rng = random.Random(int(seed_i))

            packed = _pack_for_target(
                tokenizer=tok,
                prompt_text=prompt_text,
                rollouts=all_rollouts,  # Use ALL rollouts for context
                target_idx=int(r_i),
                max_length=int(cfg.max_length),
                rng=rng,
            )

            tgt_token_ids = packed.target_token_ids
            tgt_positions = packed.target_positions

            if int(cfg.max_tokens_per_rollout) > 0 and len(tgt_token_ids) > int(cfg.max_tokens_per_rollout):
                tgt_token_ids = tgt_token_ids[: int(cfg.max_tokens_per_rollout)]
                tgt_positions = tgt_positions[: int(cfg.max_tokens_per_rollout)]

            if not tgt_positions:
                parts.append('<div class="card">')
                parts.append(f'<div class="hdr">rollout {r_i} | gt {all_rollouts[r_i].reward:.4f} | (empty)</div>')
                parts.append('<div class="toks">(no tokens)</div>')
                parts.append("</div>")
                continue

            # Build on-policy reference sequence (original prompt + response)
            # This ensures the reference model sees the same context as during inference
            rollout = all_rollouts[r_i]
            ref_prompt_ids = list(rollout.prompt_token_ids)
            ref_response_ids = list(rollout.response_ids)
            ref_input_ids = ref_prompt_ids + ref_response_ids
            ref_input_t = torch.tensor(ref_input_ids, dtype=torch.long, device=device).unsqueeze(0)

            # Positions in the reference sequence for each response token
            ref_positions = [len(ref_prompt_ids) + i for i in range(len(ref_response_ids))]

            # Align lengths: tgt_token_ids might include an extra EOS footer
            # that's not in the original response_ids
            num_tgt_tokens = len(tgt_token_ids)
            num_ref_tokens = len(ref_response_ids)

            if num_tgt_tokens < num_ref_tokens:
                # Truncation was applied to tgt_token_ids (max_tokens_per_rollout)
                ref_positions = ref_positions[:num_tgt_tokens]
                ref_response_ids = ref_response_ids[:num_tgt_tokens]
            elif num_tgt_tokens > num_ref_tokens:
                # Footer (EOS) was added to tgt_token_ids
                # We'll handle this by padding candidates later
                pass

            # Debug: show sequence info
            print(f"[DEBUG data] rollout {r_i}: ref_seq_len={len(ref_input_ids)}, "
                  f"ref_prompt_len={len(ref_prompt_ids)}, ref_response_len={len(ref_response_ids)}, "
                  f"tgt_tokens={num_tgt_tokens}", flush=True)

            # Step 1: Get candidates from reference model (on-policy sequence)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                candidates_per_pos = _get_candidates_at_positions(
                    ref_model,
                    input_ids=ref_input_t,
                    positions=ref_positions,
                    actual_token_ids=ref_response_ids,
                    min_p=cfg.min_p,
                    device=device,
                )

            # Free reference input tensor
            del ref_input_t

            # Pad candidates for any footer tokens not in original response
            while len(candidates_per_pos) < num_tgt_tokens:
                # For footer tokens (like EOS), just include the token itself
                footer_idx = len(candidates_per_pos)
                footer_tid = tgt_token_ids[footer_idx]
                candidates_per_pos.append([(footer_tid, 0.0)])

            # Step 2: Evaluate Q-values with critic (packed sequence with context)
            input_ids_t = packed.input_ids.unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                actual_q_values, candidate_results, num_extra_fwd = _evaluate_q_values_with_kv_cache(
                    critic_model,
                    input_ids=input_ids_t,
                    positions=tgt_positions,
                    actual_token_ids=tgt_token_ids,
                    candidates_per_position=candidates_per_pos,
                    distribution_token_id=cfg.distribution_token_id,
                    num_bins=num_bins,
                    num_length_bins=num_length_bins,
                    correct_reward_index=correct_reward_index,
                    device=device,
                )

            rollout_time = time.time() - rollout_start
            avg_cands = np.mean([len(c) for c in candidates_per_pos]) if candidates_per_pos else 0
            pct_extra = 100.0 * num_extra_fwd / len(tgt_positions) if tgt_positions else 0
            ctx_count = len(packed.context_rollout_indices)
            seq_len = len(packed.input_ids)
            _print_progress(
                f"[Rank {rank}]   Rollout {r_i+1}/{num_to_visualize}: {len(tgt_positions)} tgt toks, "
                f"seq_len={seq_len}, ctx={ctx_count}/{len(all_rollouts)-1}, "
                f"{num_extra_fwd} extra fwd ({pct_extra:.0f}%), {rollout_time:.2f}s"
            )

            mean_q = float(np.mean(actual_q_values)) if actual_q_values else 0.0

            # Generate token spans with tooltips
            spans: List[str] = []
            for idx, (tid, pos) in enumerate(zip(tgt_token_ids, tgt_positions)):
                q_val = actual_q_values[idx] if idx < len(actual_q_values) else 0.0
                candidates = candidate_results[idx] if idx < len(candidate_results) else []

                # Debug for first token
                if idx == 0:
                    cand_token_ids = [c[0] for c in candidates]
                    actual_in_final = tid in cand_token_ids
                    print(f"[DEBUG tooltip] idx=0: tid={tid} ({_decode_token(tok, int(tid), decode_cache)!r}), "
                          f"actual_in_final={actual_in_final}, cand_ids={cand_token_ids}", flush=True)

                tok_str = html_lib.escape(_decode_token(tok, int(tid), decode_cache))
                tooltip_html = _make_tooltip_html(tok, tid, candidates, decode_cache, cfg.max_candidates_show)

                spans.append(
                    f'<span class="tok" style="background-color:{_reward_rgb(q_val)}">'
                    f'{tok_str}{tooltip_html}</span>'
                )

            ctx = packed.context_rollout_indices
            avg_candidates = np.mean([len(c) for c in candidate_results]) if candidate_results else 0

            parts.append('<div class="card">')
            parts.append('<div class="hdr">')
            parts.append(
                html_lib.escape(
                    f"rollout {r_i} | gt {all_rollouts[r_i].reward:.4f} | mean_q {mean_q:.4f} | "
                    f"ctx_n {len(ctx)} | avg_cand {avg_candidates:.1f} | tok {len(tgt_token_ids)}"
                )
            )
            parts.append("</div>")
            parts.append(f'<div class="toks">{"".join(spans)}</div>')
            parts.append("</div>")

            # Free GPU memory after each rollout
            del input_ids_t, candidates_per_pos, actual_q_values, candidate_results
            if device.type == "cuda":
                torch.cuda.empty_cache()

        parts.append("</div>")

    # Write fragment to file
    fragment_path = os.path.join(fragment_dir, f"fragment_{rank:04d}.html")
    with open(fragment_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    _print_progress(f"[Rank {rank}] Done! Processed {processed_rollouts} rollouts, wrote fragment.")


def main() -> None:
    args = parse_args()
    script_start = time.time()

    # Determine dp_size
    ngpus = torch.cuda.device_count()
    if args.dp_size == 0:
        dp = max(1, ngpus)
    else:
        dp = args.dp_size

    if ngpus > 0 and dp > ngpus:
        raise RuntimeError(f"Requested dp_size={dp}, but only {ngpus} CUDA devices are visible.")
    if ngpus == 0 and dp != 1:
        raise RuntimeError("No CUDA devices visible; run with --dp-size 1 for CPU mode.")

    # Load data to determine prompts (only in main process)
    _print_progress(f"Loading data from {args.data_path}...")
    pf = pq.ParquetFile(args.data_path)
    names = set(pf.schema_arrow.names)

    reward_col = args.label_column
    if reward_col not in names:
        raise ValueError(f"Missing column: {reward_col}")

    # Create config
    cfg = Config(
        critic_path=args.critic_path,
        ref_path=args.ref_path,
        data_path=args.data_path,
        out_html=args.out_html,
        num_prompts=args.num_prompts,
        prompt_idx=args.prompt_idx,
        min_p=args.min_p,
        max_candidates_show=args.max_candidates_show,
        distribution_token_id=args.distribution_token_id,
        label_column=args.label_column,
        max_length=args.max_length,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        tokenizer_path=args.tokenizer_path,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens_per_rollout=args.max_tokens_per_rollout,
        require_mixed_outcomes=args.require_mixed_outcomes,
        min_correct_pct=args.min_correct_pct,
        min_incorrect_pct=args.min_incorrect_pct,
        dp_size=dp,
        reward_col=reward_col,
    )

    # Load prompt groups to determine selection
    prompt_groups = _load_prompt_groups(cfg)
    prompt_ids_sorted = sorted(prompt_groups.keys())
    _print_progress(f"  Loaded {len(prompt_groups)} prompts")

    # Filter for mixed outcomes if requested
    if args.require_mixed_outcomes:
        filtered_ids = [
            pid for pid in prompt_ids_sorted
            if _has_mixed_outcomes(
                prompt_groups[pid]["rollouts"],
                min_correct_pct=args.min_correct_pct,
                min_incorrect_pct=args.min_incorrect_pct,
            )
        ]
        _print_progress(
            f"  After mixed outcomes filter (>={args.min_correct_pct}% correct, "
            f">={args.min_incorrect_pct}% incorrect): {len(filtered_ids)} prompts"
        )
    else:
        filtered_ids = prompt_ids_sorted

    # Select prompts
    if args.prompt_idx and len(args.prompt_idx) > 0:
        selected_prompt_ids = [int(pid) for pid in args.prompt_idx if int(pid) in prompt_groups]
        if args.require_mixed_outcomes:
            selected_prompt_ids = [pid for pid in selected_prompt_ids if pid in filtered_ids]
    else:
        selected_prompt_ids = filtered_ids[: args.num_prompts]

    cfg.selected_prompt_ids = selected_prompt_ids

    # Count total rollouts to process
    total_rollouts = sum(
        min(len(prompt_groups[pid]["rollouts"]), args.samples_per_prompt) if args.samples_per_prompt > 0
        else len(prompt_groups[pid]["rollouts"])
        for pid in selected_prompt_ids if pid in prompt_groups
    )

    _print_progress(
        f"Will process {len(selected_prompt_ids)} prompts, {total_rollouts} rollouts "
        f"across {dp} GPU(s)"
    )

    # Create temp directory for fragments
    fragment_dir = tempfile.mkdtemp(prefix="visualize_q_")
    _print_progress(f"Fragment directory: {fragment_dir}")

    try:
        # Spawn workers
        if dp == 1:
            # Single GPU: run directly without spawning
            _worker(0, 1, cfg, fragment_dir)
        else:
            ctx = mp.get_context("spawn")
            procs = []
            for r in range(dp):
                p = ctx.Process(target=_worker, args=(r, dp, cfg, fragment_dir), daemon=False)
                p.start()
                procs.append(p)

            # Wait for all workers
            any_fail = False
            for r, p in enumerate(procs):
                p.join()
                if p.exitcode != 0:
                    any_fail = True
                    print(f"[main] Worker rank {r} (pid={p.pid}) exited with code {p.exitcode}", flush=True)

            if any_fail:
                raise RuntimeError("One or more workers failed")

        # Merge fragments into final HTML
        _print_progress("Merging fragments...")

        title = f"Candidate Q-Values: {os.path.basename(args.critic_path.rstrip('/'))}"
        final_parts: List[str] = [_html_header(title)]

        final_parts.append('<div class="meta">')
        final_parts.append(f"critic_path: {html_lib.escape(args.critic_path)}<br>")
        final_parts.append(f"ref_path: {html_lib.escape(args.ref_path)}<br>")
        final_parts.append(f"data_path: {html_lib.escape(args.data_path)}<br>")
        final_parts.append(f"min_p: {args.min_p}<br>")
        final_parts.append(f"reward_column: {html_lib.escape(reward_col)}<br>")
        final_parts.append(f"dp_size: {dp}<br>")
        final_parts.append(f"selected_prompts: {html_lib.escape(str(selected_prompt_ids))}<br>")
        final_parts.append("</div>")

        # Read and merge fragments in order
        # Fragments are ordered by rank, but prompts are interleaved across ranks
        # We need to reorder by global prompt index
        prompt_fragments: Dict[int, str] = {}

        for r in range(dp):
            fragment_path = os.path.join(fragment_dir, f"fragment_{r:04d}.html")
            if os.path.exists(fragment_path):
                with open(fragment_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Associate with the prompts this rank processed
                rank_prompts = [pid for i, pid in enumerate(selected_prompt_ids) if i % dp == r]
                if rank_prompts and content:
                    # Split content by prompt boundaries (each prompt starts with <hr>)
                    # Filter out empty sections and associate with prompt IDs
                    prompt_sections = [s for s in content.split("<hr>") if s.strip()]
                    for i, section in enumerate(prompt_sections):
                        if i < len(rank_prompts):
                            prompt_fragments[rank_prompts[i]] = "<hr>" + section

        # Output in original prompt order
        for pid in selected_prompt_ids:
            if pid in prompt_fragments:
                final_parts.append(prompt_fragments[pid])

        final_parts.append(_html_footer())

        out_dir = os.path.dirname(os.path.abspath(args.out_html))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(args.out_html, "w", encoding="utf-8") as f:
            f.write("".join(final_parts))

        total_time = time.time() - script_start
        _print_progress(
            f"Done! Wrote {args.out_html}\n"
            f"  Processed {total_rollouts} rollouts in {total_time:.1f}s "
            f"({total_time/max(total_rollouts,1):.2f}s/rollout)"
        )

    finally:
        # Cleanup fragment directory
        import shutil
        try:
            shutil.rmtree(fragment_dir)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
