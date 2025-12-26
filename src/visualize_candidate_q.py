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
import gc
import html as html_lib
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

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


def _expand_kv_cache(past_key_values: DynamicCache, batch_size: int) -> DynamicCache:
    """Expand KV cache batch dimension from 1 to batch_size via cloning."""
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values.key_cache)):
        k = past_key_values.key_cache[layer_idx]
        v = past_key_values.value_cache[layer_idx]
        # Use expand + contiguous to create actual copies (not views)
        # This prevents the model from modifying the original cache
        new_cache.update(
            k.expand(batch_size, -1, -1, -1).contiguous(),
            v.expand(batch_size, -1, -1, -1).contiguous(),
            layer_idx,
        )
    return new_cache


# ---------------------------------------------------------------------------
# Reference Model: Candidate Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_candidates_at_positions(
    model: Any,
    input_ids: torch.Tensor,  # [1, S]
    positions: List[int],
    min_p: float,
    device: torch.device,
) -> List[List[Tuple[int, float]]]:
    """
    Get candidate tokens at each position using min_p filtering.

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

    for pos in positions:
        if pos <= 0:
            results.append([])
            continue

        # Logits at pos-1 predict token at pos
        pos_logits = logits[0, pos - 1, :]  # [V]
        log_probs = F.log_softmax(pos_logits.float(), dim=-1)
        probs = log_probs.exp()

        # Apply min_p filtering
        max_prob = probs.max()
        threshold = min_p * max_prob
        mask = probs >= threshold

        indices = mask.nonzero(as_tuple=True)[0]
        cand_ids = indices.tolist()
        cand_lps = log_probs[indices].tolist()

        # Sort by probability (descending)
        candidates = sorted(zip(cand_ids, cand_lps), key=lambda x: -x[1])
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
    candidates_per_position: List[List[Tuple[int, float]]],  # [(token_id, ref_logprob), ...]
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    device: torch.device,
) -> Tuple[List[float], List[List[Tuple[int, float, float]]]]:
    """
    Evaluate Q-values for candidates at each position using KV cache.

    Returns:
        actual_q_values: Q-value for the actual token at each position
        candidate_results: [[(token_id, ref_logprob, q_value), ...], ...] per position
    """
    if not positions:
        return ([], [])

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

    # Step 3: Evaluate candidates at each position
    candidate_results: List[List[Tuple[int, float, float]]] = []

    for pos_idx, (pos, candidates) in enumerate(zip(positions, candidates_per_position)):
        if not candidates:
            candidate_results.append([])
            continue

        K = len(candidates)
        cand_ids = [c[0] for c in candidates]
        cand_ref_lps = [c[1] for c in candidates]

        # Slice KV cache to prefix [0:pos]
        sliced_kv = _slice_kv_cache(past_key_values, end=pos)
        expanded_kv = _expand_kv_cache(sliced_kv, batch_size=K)

        # Create input for candidates: [K, 1]
        cand_input = torch.tensor(cand_ids, dtype=torch.long, device=device).unsqueeze(1)

        # Forward pass with candidates
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

        # Extract Q-values
        cand_h = cand_hidden.squeeze(1)  # [K, E]
        cand_logits = F.linear(cand_h, w, b).float()
        cand_logits_rs = cand_logits.view(-1, num_reward_states, num_length_bins)
        cand_logits_reward = torch.logsumexp(cand_logits_rs, dim=2)
        cand_probs_reward = torch.softmax(cand_logits_reward, dim=1)
        cand_q = cand_probs_reward[:, correct_reward_index].cpu().tolist()

        # Combine: (token_id, ref_logprob, q_value)
        pos_results = [(cand_ids[i], cand_ref_lps[i], cand_q[i]) for i in range(K)]
        # Sort by Q-value descending
        pos_results.sort(key=lambda x: -x[2])
        candidate_results.append(pos_results)

        # Cleanup intermediate caches to free memory
        del sliced_kv, expanded_kv, cand_input, cand_out, cand_hidden, cand_h
        del cand_logits, cand_logits_rs, cand_logits_reward, cand_probs_reward

        # Periodic GPU memory cleanup (every 100 positions)
        if pos_idx % 100 == 99:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Cleanup main cache
    del past_key_values, hidden, out
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (actual_q, candidate_results)


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
    p.add_argument("--max_rollouts_per_prompt", type=int, default=0)
    p.add_argument("--max_tokens_per_rollout", type=int, default=0)

    return p.parse_args()


def _print_progress(msg: str, end: str = "\n") -> None:
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", end=end, flush=True)


def main() -> None:
    args = parse_args()
    script_start = time.time()

    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = _torch_dtype(args.dtype)

    _print_progress(f"Loading critic model from {args.critic_path}...")
    model_load_start = time.time()
    model_kwargs: Dict[str, Any] = {"torch_dtype": amp_dtype, "trust_remote_code": True}
    if args.attn_implementation and args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    critic_model = AutoModelForCausalLM.from_pretrained(args.critic_path, **model_kwargs)
    critic_model.to(device)
    critic_model.eval()
    _print_progress(f"  Critic model loaded in {time.time() - model_load_start:.1f}s")

    _print_progress(f"Loading reference model from {args.ref_path}...")
    model_load_start = time.time()
    ref_model = AutoModelForCausalLM.from_pretrained(args.ref_path, **model_kwargs)
    ref_model.to(device)
    ref_model.eval()
    _print_progress(f"  Reference model loaded in {time.time() - model_load_start:.1f}s")

    tok_src = args.tokenizer_path if args.tokenizer_path else args.ref_path
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Load data
    _print_progress(f"Loading data from {args.data_path}...")
    pf = pq.ParquetFile(args.data_path)
    names = set(pf.schema_arrow.names)

    reward_col = args.label_column
    if reward_col not in names:
        raise ValueError(f"Missing column: {reward_col}")

    table = pq.read_table(args.data_path, columns=["prompt_idx", "prompt", "output_token_ids", reward_col])
    df = table.to_pandas()

    if reward_col == "correct":
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
            ro.append(Rollout(response_ids=list(toks), reward=float(r[reward_col])))
        prompt_groups[int(pid)] = {"prompt": prompt_text, "rollouts": ro}

    prompt_ids_sorted = sorted(prompt_groups.keys())
    _print_progress(f"  Loaded {len(prompt_groups)} prompts, {len(df)} total rollouts")

    # Select prompts
    if args.prompt_idx and len(args.prompt_idx) > 0:
        selected_prompt_ids = [int(pid) for pid in args.prompt_idx if int(pid) in prompt_groups]
    else:
        selected_prompt_ids = prompt_ids_sorted[: args.num_prompts]

    # Count total rollouts to process
    total_rollouts = sum(
        min(len(prompt_groups[pid]["rollouts"]), args.max_rollouts_per_prompt) if args.max_rollouts_per_prompt > 0
        else len(prompt_groups[pid]["rollouts"])
        for pid in selected_prompt_ids if pid in prompt_groups
    )
    _print_progress(f"Will process {len(selected_prompt_ids)} prompts, {total_rollouts} rollouts")

    # Critic config
    if reward_col == "correct":
        length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        reward_values = [0.0, 1.0]
    else:
        length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]

    num_length_bins = len(length_bins) - 1
    num_reward_states = len(reward_values)
    num_bins = num_length_bins * num_reward_states
    correct_reward_index = int(max(range(len(reward_values)), key=lambda i: reward_values[i]))

    # Generate HTML
    title = f"Candidate Q-Values: {os.path.basename(args.critic_path.rstrip('/'))}"
    parts: List[str] = [_html_header(title)]

    parts.append('<div class="meta">')
    parts.append(f"critic_path: {html_lib.escape(args.critic_path)}<br>")
    parts.append(f"ref_path: {html_lib.escape(args.ref_path)}<br>")
    parts.append(f"data_path: {html_lib.escape(args.data_path)}<br>")
    parts.append(f"min_p: {args.min_p}<br>")
    parts.append(f"reward_column: {html_lib.escape(reward_col)}<br>")
    parts.append(f"selected_prompts: {html_lib.escape(str(selected_prompt_ids))}<br>")
    parts.append("</div>")

    decode_cache: Dict[int, str] = {}
    processed_rollouts = 0
    processing_start = time.time()

    _print_progress("Starting processing...")

    for p_i, pid in enumerate(selected_prompt_ids, start=1):
        grp = prompt_groups.get(int(pid), None)
        if grp is None:
            continue

        prompt_text = str(grp["prompt"])
        rollouts: List[Rollout] = list(grp["rollouts"])

        if int(args.max_rollouts_per_prompt) > 0:
            rollouts = rollouts[: int(args.max_rollouts_per_prompt)]

        _print_progress(f"Prompt {p_i}/{len(selected_prompt_ids)} (idx={pid}): {len(rollouts)} rollouts")

        parts.append("<hr>")
        parts.append(f"<h2>prompt {p_i} | prompt_idx {int(pid)} | rollouts {len(rollouts)}</h2>")
        parts.append("<h3>prompt</h3>")
        parts.append(f"<pre>{html_lib.escape(prompt_text)}</pre>")

        parts.append("<h3>rollouts</h3>")
        parts.append('<div class="grid">')

        for r_i in range(len(rollouts)):
            rollout_start = time.time()
            processed_rollouts += 1

            seed_i = (int(args.seed) * 1000003 + int(pid) * 1009 + int(r_i) * 9176) & 0xFFFFFFFF
            rng = random.Random(int(seed_i))

            packed = _pack_for_target(
                tokenizer=tok,
                prompt_text=prompt_text,
                rollouts=rollouts,
                target_idx=int(r_i),
                max_length=int(args.max_length),
                rng=rng,
            )

            tgt_token_ids = packed.target_token_ids
            tgt_positions = packed.target_positions

            if int(args.max_tokens_per_rollout) > 0 and len(tgt_token_ids) > int(args.max_tokens_per_rollout):
                tgt_token_ids = tgt_token_ids[: int(args.max_tokens_per_rollout)]
                tgt_positions = tgt_positions[: int(args.max_tokens_per_rollout)]

            if not tgt_positions:
                parts.append('<div class="card">')
                parts.append(f'<div class="hdr">rollout {r_i} | gt {rollouts[r_i].reward:.4f} | (empty)</div>')
                parts.append('<div class="toks">(no tokens)</div>')
                parts.append("</div>")
                continue

            input_ids_t = packed.input_ids.unsqueeze(0).to(device)

            # Step 1: Get candidates from reference model
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                candidates_per_pos = _get_candidates_at_positions(
                    ref_model,
                    input_ids=input_ids_t,
                    positions=tgt_positions,
                    min_p=args.min_p,
                    device=device,
                )

            # Step 2: Evaluate Q-values with critic
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype):
                actual_q_values, candidate_results = _evaluate_q_values_with_kv_cache(
                    critic_model,
                    input_ids=input_ids_t,
                    positions=tgt_positions,
                    candidates_per_position=candidates_per_pos,
                    distribution_token_id=args.distribution_token_id,
                    num_bins=num_bins,
                    num_length_bins=num_length_bins,
                    correct_reward_index=correct_reward_index,
                    device=device,
                )

            rollout_time = time.time() - rollout_start
            avg_cands = np.mean([len(c) for c in candidates_per_pos]) if candidates_per_pos else 0
            _print_progress(
                f"  Rollout {r_i+1}/{len(rollouts)}: {len(tgt_positions)} tokens, "
                f"{avg_cands:.1f} avg candidates, {rollout_time:.2f}s "
                f"[{processed_rollouts}/{total_rollouts}]"
            )

            mean_q = float(np.mean(actual_q_values)) if actual_q_values else 0.0

            # Generate token spans with tooltips
            spans: List[str] = []
            for idx, (tid, pos) in enumerate(zip(tgt_token_ids, tgt_positions)):
                q_val = actual_q_values[idx] if idx < len(actual_q_values) else 0.0
                candidates = candidate_results[idx] if idx < len(candidate_results) else []

                tok_str = html_lib.escape(_decode_token(tok, int(tid), decode_cache))
                tooltip_html = _make_tooltip_html(tok, tid, candidates, decode_cache, args.max_candidates_show)

                spans.append(
                    f'<span class="tok" style="background-color:{_reward_rgb(q_val)}">'
                    f'{tok_str}{tooltip_html}</span>'
                )

            ctx = packed.context_rollout_indices
            ctx_str = ",".join(str(int(x)) for x in ctx)
            avg_candidates = np.mean([len(c) for c in candidate_results]) if candidate_results else 0

            parts.append('<div class="card">')
            parts.append('<div class="hdr">')
            parts.append(
                html_lib.escape(
                    f"rollout {r_i} | gt {rollouts[r_i].reward:.4f} | mean_q {mean_q:.4f} | "
                    f"ctx_n {len(ctx)} | avg_cand {avg_candidates:.1f} | tok {len(tgt_token_ids)}"
                )
            )
            parts.append("</div>")
            parts.append(f'<div class="toks">{"".join(spans)}</div>')
            parts.append("</div>")

            # Free GPU memory after each rollout
            del input_ids_t, candidates_per_pos, actual_q_values, candidate_results
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        parts.append("</div>")

    parts.append(_html_footer())

    out_dir = os.path.dirname(os.path.abspath(args.out_html))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    total_time = time.time() - script_start
    _print_progress(
        f"Done! Wrote {args.out_html}\n"
        f"  Processed {processed_rollouts} rollouts in {total_time:.1f}s "
        f"({total_time/max(processed_rollouts,1):.2f}s/rollout)"
    )


if __name__ == "__main__":
    main()
