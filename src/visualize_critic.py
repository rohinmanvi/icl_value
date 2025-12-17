#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _last_hidden_state(model: Any, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    m = _unwrap_model(model)
    base = _get_base_model(m)
    if base is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("no hidden_states")
        return hs[-1]
    out = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    raise RuntimeError("no last hidden state")


def create_tokenized_message(role: str, tokenized_content: List[int]) -> List[int]:
    role_token_id = {
        "system": SYSTEM_TOKEN_ID,
        "user": USER_TOKEN_ID,
        "assistant": ASSISTANT_TOKEN_ID,
    }[role]
    content = list(tokenized_content)
    if content and content[-1] == IM_END_TOKEN_ID:
        content = content[:-1]
    return [IM_START_TOKEN_ID, role_token_id, NEWLINE_TOKEN_ID] + content + [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]


def _value_edges(reward_values: List[float]) -> List[float]:
    n = len(reward_values)
    if n >= 2:
        edges = [0.0] * (n + 1)
        for i in range(1, n):
            edges[i] = 0.5 * (reward_values[i - 1] + reward_values[i])
        first_step = reward_values[1] - reward_values[0]
        last_step = reward_values[-1] - reward_values[-2]
        edges[0] = reward_values[0] - 0.5 * first_step
        edges[-1] = reward_values[-1] + 0.5 * last_step
    else:
        edges = [reward_values[0] - 0.5, reward_values[0] + 0.5]
    edges[0] = max(0.0, edges[0])
    edges[-1] = min(1.0, edges[-1])
    return edges


def _lr_schedule(step: int, total_steps: int, lr: float, min_lr: float, warmup_ratio: float) -> float:
    if total_steps <= 0:
        return lr
    warmup_steps = int(float(warmup_ratio) * float(total_steps))
    if warmup_steps < 1:
        warmup_steps = 0
    if warmup_steps > 0 and step <= warmup_steps:
        return lr * float(step) / float(warmup_steps)
    if step <= warmup_steps:
        return lr
    if step >= total_steps:
        return min_lr
    denom = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(denom)
    return float(min_lr) + 0.5 * (1.0 + math.cos(math.pi * progress)) * (float(lr) - float(min_lr))


def _simulate_optimizer_steps(
    dataset_len: int,
    num_epochs: int,
    batch_size: int,
    grad_accum: int,
    seed: int,
    max_steps: int,
) -> List[List[int]]:
    g = torch.Generator()
    g.manual_seed(int(seed))

    order: List[int] = []
    for _ in range(int(num_epochs)):
        perm = torch.randperm(int(dataset_len), generator=g).tolist()
        order.extend(perm)

    batches: List[List[int]] = []
    bs = int(batch_size)
    for i in range(0, len(order), bs):
        batches.append(order[i : i + bs])

    steps: List[List[int]] = []
    cur: List[List[int]] = []
    ga = int(grad_accum)
    for it, b in enumerate(batches):
        cur.append(b)
        if (it + 1) % ga == 0:
            flat = [j for bb in cur for j in bb]
            steps.append(flat)
            cur = []
            if max_steps > 0 and len(steps) >= int(max_steps):
                break
    return steps


def _read_trace(trace_path: str) -> List[Dict[str, Any]]:
    if trace_path is None:
        return []
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(trace_path)

    if trace_path.endswith(".jsonl"):
        recs: List[Dict[str, Any]] = []
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                step = obj.get("global_step") or obj.get("step") or obj.get("optimizer_step")
                if step is None:
                    step = len(recs) + 1
                if "sample_indices" in obj:
                    inds = obj["sample_indices"]
                elif "dataset_indices" in obj:
                    inds = obj["dataset_indices"]
                elif "sample_idx" in obj:
                    inds = [obj["sample_idx"]]
                elif "dataset_idx" in obj:
                    inds = [obj["dataset_idx"]]
                elif "idx" in obj:
                    inds = [obj["idx"]]
                else:
                    inds = []
                prompt_idx = obj.get("prompt_idx")
                recs.append(
                    {
                        "step": int(step),
                        "sample_indices": [int(i) for i in inds],
                        "prompt_idx": int(prompt_idx) if prompt_idx is not None else None,
                    }
                )
        return recs

    recs = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            inds = [int(p) for p in parts]
            recs.append({"step": len(recs) + 1, "sample_indices": inds, "prompt_idx": None})
    return recs


def _reward_rgb(r: float) -> str:
    x = 0.0 if r < 0.0 else 1.0 if r > 1.0 else float(r)
    rr = int((1.0 - x) * 255.0)
    gg = int(x * 255.0)
    bb = 0
    return f"rgb({rr},{gg},{bb})"


def _decode_token(tokenizer: Any, tid: int, cache: Dict[int, str]) -> str:
    if tid in cache:
        return cache[tid]
    s = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    if s == "":
        s = f"<{int(tid)}>"
    cache[tid] = s
    return s


@torch.no_grad()
def _expected_reward_per_pos(
    model: Any,
    input_ids_1d: torch.Tensor,
    positions: List[int],
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    reward_values: List[float],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> Dict[int, float]:
    if not positions:
        return {}

    input_ids = input_ids_1d.unsqueeze(0).to(device)
    pos = torch.tensor(positions, device=device, dtype=torch.long)

    with torch.autocast(
        device_type="cuda" if device.type == "cuda" else "cpu",
        dtype=amp_dtype,
        enabled=(device.type == "cuda" and amp_dtype != torch.float32),
    ):
        h = _last_hidden_state(model, input_ids=input_ids)  # [1, S, E]
        h_sel = h[0, pos]  # [N, E]

        m = _unwrap_model(model)
        lm_head = _get_lm_head(m)
        w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
        b = None
        if hasattr(lm_head, "bias") and lm_head.bias is not None:
            b = lm_head.bias[distribution_token_id : distribution_token_id + num_bins]

        logits = F.linear(h_sel, w, b).float()  # [N, num_bins]
        probs = torch.softmax(logits, dim=-1)  # [N, num_bins]

        num_reward = int(num_bins // num_length_bins)
        probs3 = probs.view(-1, num_reward, num_length_bins)
        p_reward = probs3.sum(dim=2)  # [N, num_reward]
        rv = torch.tensor(reward_values, device=device, dtype=torch.float32).view(1, -1)
        exp_r = (p_reward * rv).sum(dim=1)  # [N]

    exp_r_cpu = exp_r.detach().cpu().tolist()
    return {int(p): float(v) for p, v in zip(positions, exp_r_cpu)}


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
.tok {{ padding: 0 1px; border-radius: 2px; }}
.legend {{ display:flex; align-items:center; gap:10px; margin: 10px 0; }}
.bar {{ width: 220px; height: 14px; border: 1px solid #bbb; background: linear-gradient(to right, rgb(255,0,0), rgb(0,255,0)); }}
.small {{ font-size: 12px; color: #333; }}
hr {{ border: 0; border-top: 1px solid #ddd; margin: 16px 0; }}
</style>
</head>
<body>
<h1>{t}</h1>
<div class="legend">
  <div class="bar"></div>
  <div class="small">0 → 1</div>
</div>
"""


def _html_footer() -> str:
    return "</body>\n</html>\n"


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


def _feedback_tokens(tokenizer: Any, reward: float, content_plus_footer_len: int, full: bool) -> List[int]:
    if full:
        s = f"Reward: {float(reward)}\nLength: {int(content_plus_footer_len)} tokens"
    else:
        s = f"Length: {int(content_plus_footer_len)} tokens"
    return tokenizer.encode(s, add_special_tokens=False)


def _pack_for_target(
    tokenizer: Any,
    prompt_text: str,
    rollouts: List[Rollout],
    target_idx: int,
    max_length: int,
    ablation_type: str,
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
        token_ids = list(content) + list(footer)  # content + footer
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
    # each entry: (rollout_idx, traj_block, feedback_user_message)

    total_len = fixed_len
    local_idx = 0
    for ridx in other:
        r = rollouts[ridx]
        content = list(r.response_ids)
        header, footer_r, block = _make_traj_block(content)
        content_plus_footer_len = len(content) + len(footer_r)

        if ablation_type in ("full", "no_ans"):
            full_fb = True
        elif ablation_type == "no_ans_no_rewards":
            full_fb = False
        elif ablation_type == "no_ans_first_reward_only":
            full_fb = (local_idx == 0)
        else:
            full_fb = True

        fb_tokens = _feedback_tokens(tokenizer, r.reward, content_plus_footer_len, full=full_fb)
        fb_msg = create_tokenized_message("user", fb_tokens)

        add_len = len(block) + len(fb_msg)
        if total_len + add_len > int(max_length):
            break

        context_indices.append(ridx)
        context_blocks.append((ridx, block, fb_msg))
        total_len += add_len
        local_idx += 1

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_html", required=True)

    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--tail_steps", type=int, default=50)
    p.add_argument("--trace_path", type=str, default=None)

    p.add_argument("--prompt_idx", type=int, nargs="*", default=None)

    p.add_argument("--distribution_token_id", type=int, default=151669)
    p.add_argument("--label_column", choices=["auto", "correct", "value"], default="correct")
    p.add_argument("--ablation_type", choices=["full", "no_ans", "no_ans_no_rewards", "no_ans_first_reward_only"], default="no_ans")
    p.add_argument("--examples_per_prompt", type=int, default=1)
    p.add_argument("--correctness_only", action="store_true")
    p.add_argument("--max_length", type=int, default=131072)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)

    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--min_learning_rate", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", type=str, default="auto")
    p.add_argument("--trust_remote_code", action="store_true")

    p.add_argument("--tokenizer_path", type=str, default=None)

    p.add_argument("--max_rollouts_per_prompt", type=int, default=0)
    p.add_argument("--max_tokens_per_rollout", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = _torch_dtype(args.dtype)

    model_kwargs: Dict[str, Any] = {"torch_dtype": amp_dtype, "trust_remote_code": bool(args.trust_remote_code)}
    if args.attn_implementation and args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.weights_path, **model_kwargs)
    model.to(device)
    model.eval()

    tok_src = args.tokenizer_path if args.tokenizer_path else args.weights_path
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=bool(args.trust_remote_code))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    pf = pq.ParquetFile(args.data_path)
    names = set(pf.schema_arrow.names)

    choice = (args.label_column or "correct").lower()
    if choice == "auto":
        if "correct" in names:
            reward_col = "correct"
        elif "value" in names:
            reward_col = "value"
        else:
            raise ValueError("need correct or value")
    elif choice == "correct":
        if "correct" not in names:
            raise ValueError("need correct")
        reward_col = "correct"
    elif choice == "value":
        if "value" not in names:
            raise ValueError("need value")
        reward_col = "value"
    else:
        raise ValueError(f"bad label_column: {args.label_column}")

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

    if args.correctness_only:
        length_bins = [0, int(args.max_length) + 1]
    else:
        length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_length_bins = len(length_bins) - 1

    if reward_col == "correct":
        reward_values = [0.0, 1.0]
    else:
        reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]
    num_reward_states = len(reward_values)
    num_bins = num_length_bins * num_reward_states

    sample_prompt_ids: List[int] = []
    for pid in prompt_ids_sorted:
        for _ in range(int(args.examples_per_prompt)):
            sample_prompt_ids.append(int(pid))

    selected_prompt_ids: List[int] = []
    if args.prompt_idx and len(args.prompt_idx) > 0:
        for pid in args.prompt_idx:
            if int(pid) in prompt_groups and int(pid) not in selected_prompt_ids:
                selected_prompt_ids.append(int(pid))
    else:
        if args.trace_path:
            recs = _read_trace(args.trace_path)
            tail = recs[-int(args.tail_steps) :] if int(args.tail_steps) > 0 else recs
            seen = set()
            for r in reversed(tail):
                pid = r.get("prompt_idx", None)
                if pid is None:
                    for si in r.get("sample_indices", []):
                        if 0 <= int(si) < len(sample_prompt_ids):
                            pid = sample_prompt_ids[int(si)]
                            break
                if pid is None:
                    continue
                pid = int(pid)
                if pid in seen or pid not in prompt_groups:
                    continue
                seen.add(pid)
                selected_prompt_ids.append(pid)
                if len(selected_prompt_ids) >= int(args.num_prompts):
                    break
            selected_prompt_ids = list(reversed(selected_prompt_ids))
        else:
            steps = _simulate_optimizer_steps(
                dataset_len=len(sample_prompt_ids),
                num_epochs=int(args.num_epochs),
                batch_size=int(args.batch_size),
                grad_accum=int(args.gradient_accumulation_steps),
                seed=int(args.seed),
                max_steps=int(args.max_steps),
            )
            total_steps = len(steps)
            tail_steps = min(int(args.tail_steps), total_steps) if int(args.tail_steps) > 0 else total_steps
            tail = steps[-tail_steps:] if tail_steps > 0 else []

            seen = set()
            for step_i, batch_inds in zip(range(total_steps - tail_steps + 1, total_steps + 1), tail):
                _ = _lr_schedule(
                    step=int(step_i),
                    total_steps=int(total_steps),
                    lr=float(args.learning_rate),
                    min_lr=float(args.min_learning_rate),
                    warmup_ratio=float(args.warmup_ratio),
                )
                for si in batch_inds:
                    pid = sample_prompt_ids[int(si)]
                    if pid in seen:
                        continue
                    seen.add(pid)
                    selected_prompt_ids.append(int(pid))
                    if len(selected_prompt_ids) >= int(args.num_prompts):
                        break
                if len(selected_prompt_ids) >= int(args.num_prompts):
                    break

    title = f"critic report: {os.path.basename(args.weights_path.rstrip('/'))}"
    parts: List[str] = [_html_header(title)]

    parts.append('<div class="meta">')
    parts.append(f"weights_path: {html_lib.escape(args.weights_path)}<br>")
    parts.append(f"tokenizer_path: {html_lib.escape(tok_src)}<br>")
    parts.append(f"data_path: {html_lib.escape(args.data_path)}<br>")
    parts.append(f"reward_column: {html_lib.escape(reward_col)}<br>")
    parts.append(f"ablation_type: {html_lib.escape(args.ablation_type)}<br>")
    parts.append(f"examples_per_prompt: {int(args.examples_per_prompt)}<br>")
    parts.append(f"max_length: {int(args.max_length)}<br>")
    parts.append(f"distribution_token_id: {int(args.distribution_token_id)}<br>")
    parts.append(f"num_bins: {int(num_bins)}<br>")
    parts.append(f"num_reward_states: {int(num_reward_states)}<br>")
    parts.append(f"num_length_bins: {int(num_length_bins)}<br>")
    parts.append(f"reward_values: {html_lib.escape(str(reward_values))}<br>")
    parts.append(f"selected_prompts: {html_lib.escape(str(selected_prompt_ids))}<br>")
    parts.append("</div>")

    decode_cache: Dict[int, str] = {}

    for p_i, pid in enumerate(selected_prompt_ids, start=1):
        grp = prompt_groups.get(int(pid), None)
        if grp is None:
            continue

        prompt_text = str(grp["prompt"])
        rollouts: List[Rollout] = list(grp["rollouts"])

        if int(args.max_rollouts_per_prompt) > 0:
            rollouts = rollouts[: int(args.max_rollouts_per_prompt)]

        parts.append("<hr>")
        parts.append(f"<h2>prompt {p_i} | prompt_idx {int(pid)} | rollouts {len(rollouts)}</h2>")
        parts.append("<h3>prompt</h3>")
        parts.append(f"<pre>{html_lib.escape(prompt_text)}</pre>")

        parts.append("<h3>rollouts</h3>")
        parts.append('<div class="grid">')

        for r_i in range(len(rollouts)):
            seed_i = (int(args.seed) * 1000003 + int(pid) * 1009 + int(r_i) * 9176) & 0xFFFFFFFF
            rng = random.Random(int(seed_i))

            packed = _pack_for_target(
                tokenizer=tok,
                prompt_text=prompt_text,
                rollouts=rollouts,
                target_idx=int(r_i),
                max_length=int(args.max_length),
                ablation_type=str(args.ablation_type),
                rng=rng,
            )

            tgt_token_ids = packed.target_token_ids
            tgt_positions = packed.target_positions

            if int(args.max_tokens_per_rollout) > 0 and len(tgt_token_ids) > int(args.max_tokens_per_rollout):
                tgt_token_ids = tgt_token_ids[: int(args.max_tokens_per_rollout)]
                tgt_positions = tgt_positions[: int(args.max_tokens_per_rollout)]

            pos_to_er = _expected_reward_per_pos(
                model=model,
                input_ids_1d=packed.input_ids,
                positions=list(tgt_positions),
                distribution_token_id=int(args.distribution_token_id),
                num_bins=int(num_bins),
                num_length_bins=int(num_length_bins),
                reward_values=list(reward_values),
                device=device,
                amp_dtype=amp_dtype,
            )

            er_vals = [float(pos_to_er.get(int(p), 0.0)) for p in tgt_positions]
            mean_er = float(np.mean(er_vals)) if er_vals else 0.0

            spans: List[str] = []
            for tid, ap in zip(tgt_token_ids, tgt_positions):
                er = float(pos_to_er.get(int(ap), 0.0))
                s = _decode_token(tok, int(tid), decode_cache)
                spans.append(
                    f'<span class="tok" style="background-color:{_reward_rgb(er)}" '
                    f'title="pos {int(ap)} er {er:.4f} tid {int(tid)}">{html_lib.escape(s)}</span>'
                )

            ctx = packed.context_rollout_indices
            ctx_str = ",".join(str(int(x)) for x in ctx)

            parts.append('<div class="card">')
            parts.append('<div class="hdr">')
            parts.append(
                html_lib.escape(
                    f"rollout {r_i} | gt {rollouts[r_i].reward:.4f} | mean {mean_er:.4f} | "
                    f"ctx_n {len(ctx)} | ctx [{ctx_str}] | trunc {1 if packed.target_truncated else 0} | tok {len(tgt_token_ids)}"
                )
            )
            parts.append("</div>")
            parts.append(f'<div class="toks">{"".join(spans)}</div>')
            parts.append("</div>")

        parts.append("</div>")

    parts.append(_html_footer())

    out_dir = os.path.dirname(os.path.abspath(args.out_html))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print(args.out_html)


if __name__ == "__main__":
    main()
