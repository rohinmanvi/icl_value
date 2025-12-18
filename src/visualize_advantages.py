#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
    s = (s or "").lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"bad dtype: {s}")


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


def _read_trace(trace_path: str) -> List[Dict[str, Any]]:
    if not trace_path:
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


def _unwrap_model(m: Any) -> Any:
    m = m._orig_mod if hasattr(m, "_orig_mod") else m
    m = m.module if hasattr(m, "module") else m
    return m


def _unwrap_lm_head(model: torch.nn.Module):
    tgt = _unwrap_model(model)
    return tgt.lm_head if hasattr(tgt, "lm_head") else tgt.get_output_embeddings()


def _get_base_model(model: torch.nn.Module):
    tgt = _unwrap_model(model)
    for attr in ("model", "transformer", "base_model"):
        if hasattr(tgt, attr):
            return getattr(tgt, attr)
    return None


def _decode_token(tokenizer: Any, tid: int, cache: Dict[int, str]) -> str:
    if tid in cache:
        return cache[tid]
    s = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    if s == "":
        s = f"<{int(tid)}>"
    cache[tid] = s
    return s


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _compute_advantages(p_correct: Sequence[float], *, tau: float, gae_lambda: float, adv_clip: float) -> Tuple[List[float], List[float]]:
    values: List[float] = []
    for p in p_correct:
        try:
            p_f = float(p)
        except Exception:
            values.append(float("nan"))
            continue
        if not math.isfinite(p_f):
            values.append(float("nan"))
            continue
        if p_f < 0.0:
            p_f = 0.0
        elif p_f > 1.0:
            p_f = 1.0
        values.append(math.log((p_f + float(tau)) / (1.0 + float(tau))))

    n = len(values)
    if n == 0:
        return ([], [])

    adv: List[float] = [float("nan")] * n
    adv[0] = 0.0
    gae = 0.0
    lam = float(gae_lambda)

    for t in range(n - 1, 0, -1):
        v_t = values[t]
        v_prev = values[t - 1]
        if not (math.isfinite(v_t) and math.isfinite(v_prev)):
            adv[t] = float("nan")
            gae = 0.0
            continue
        delta = v_t - v_prev
        gae = delta + lam * gae
        adv[t] = float(gae)

    if adv_clip and adv_clip > 0.0:
        c = float(adv_clip)
        for i, a in enumerate(adv):
            if not math.isfinite(a):
                continue
            if a > c:
                adv[i] = c
            elif a < -c:
                adv[i] = -c

    return adv, values


def _adv_color(a: float, scale: float) -> str:
    if not math.isfinite(float(a)):
        return "rgb(220,220,220)"
    x = float(a)
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    if x >= 0.0:
        v = int(255.0 * (1.0 - x))
        return f"rgb({v},255,{v})"
    v = int(255.0 * (1.0 + x))
    return f"rgb(255,{v},{v})"


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
.bar {{ width: 260px; height: 14px; border: 1px solid #bbb; background: linear-gradient(to right, rgb(255,0,0), rgb(255,255,255), rgb(0,255,0)); }}
.small {{ font-size: 12px; color: #333; }}
hr {{ border: 0; border-top: 1px solid #ddd; margin: 16px 0; }}
</style>
</head>
<body>
<h1>{t}</h1>
<div class="legend">
  <div class="bar"></div>
  <div class="small">adv clamped to [-1,1]</div>
</div>
"""


def _html_footer() -> str:
    return "</body>\n</html>\n"


@dataclass
class RolloutRow:
    row_idx: int
    token_ids: List[int]
    reward: float
    p_correct: Optional[List[float]]


@dataclass
class CtxInfo:
    traj_block: List[int]
    feedback_block: List[int]


def _build_ctx_infos(rollouts: Sequence[RolloutRow], tokenizer) -> List[CtxInfo]:
    infos: List[CtxInfo] = []
    header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
    for r in rollouts:
        traj_content = list(r.token_ids)
        has_eos = bool(traj_content) and traj_content[-1] == IM_END_TOKEN_ID
        footer = [] if has_eos else [IM_END_TOKEN_ID]
        traj_block = header + traj_content + footer

        content_plus_footer_len = len(traj_content) + len(footer)
        fb = f"Reward: {float(r.reward)}\nLength: {content_plus_footer_len} tokens"
        fb_tokens = tokenizer.encode(fb, add_special_tokens=False)
        feedback_block = _create_tokenized_message("user", fb_tokens)

        infos.append(CtxInfo(traj_block=traj_block, feedback_block=feedback_block))
    return infos


def _truncate_target(prefix_len: int, target_tokens: List[int], max_length: int) -> Tuple[List[int], List[int], bool]:
    header_len = 3
    budget = int(max_length) - int(prefix_len) - int(header_len)
    if budget <= 0:
        return ([], [IM_END_TOKEN_ID], True)

    L = len(target_tokens)
    trunc_len = min(L, budget)
    while trunc_len > 0:
        last_is_end = (target_tokens[trunc_len - 1] == IM_END_TOKEN_ID)
        footer_len = 0 if last_is_end else 1
        if trunc_len + footer_len <= budget:
            break
        trunc_len -= 1

    if trunc_len == 0:
        return ([], [IM_END_TOKEN_ID], True)

    trunc = list(target_tokens[:trunc_len])
    footer = [] if (trunc and trunc[-1] == IM_END_TOKEN_ID) else [IM_END_TOKEN_ID]
    truncated = trunc_len < L
    return (trunc, footer, truncated)


def _pack_target_last(
    tokenizer,
    *,
    prompt: str,
    rollouts: Sequence[RolloutRow],
    ctx_infos: Sequence[CtxInfo],
    target_idx: int,
    max_length: int,
    shuffle_seed: int,
    prompt_idx: int,
) -> Tuple[List[int], List[int], bool, List[int]]:
    prompt_tokens = tokenizer.encode(prompt or "", add_special_tokens=False)
    prefix = _create_tokenized_message("user", prompt_tokens)
    prefix_len = len(prefix)

    header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
    target_tokens = list(rollouts[target_idx].token_ids)
    trunc_content, footer, truncated = _truncate_target(prefix_len, target_tokens, max_length)
    target_block = header + trunc_content + footer
    fixed_len = prefix_len + len(target_block)
    if fixed_len > int(max_length):
        return (prefix[: int(max_length)], [], True, [])

    other = [i for i in range(len(rollouts)) if i != target_idx]
    seed = (int(shuffle_seed) * 1000003 + int(prompt_idx) * 1009 + int(target_idx) * 9176) & 0xFFFFFFFF
    rng = random.Random(seed)
    rng.shuffle(other)

    total_len = fixed_len
    ctx_keep: List[int] = []
    for j in other:
        add_len = len(ctx_infos[j].traj_block) + len(ctx_infos[j].feedback_block)
        if total_len + add_len > int(max_length):
            break
        ctx_keep.append(j)
        total_len += add_len

    full_ids: List[int] = list(prefix)
    for j in ctx_keep:
        full_ids.extend(ctx_infos[j].traj_block)
        full_ids.extend(ctx_infos[j].feedback_block)

    target_start = len(full_ids)
    full_ids.extend(target_block)

    positions = [target_start + len(header) + i for i in range(len(trunc_content))]
    return (full_ids, positions, truncated, ctx_keep)


@torch.no_grad()
def _predict_p_correct_for_positions(
    model: torch.nn.Module,
    *,
    input_ids_1d: torch.Tensor,
    positions: Sequence[int],
    distribution_token_id: int,
    num_bins: int,
    num_length_bins: int,
    correct_reward_index: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> List[float]:
    if not positions:
        return []

    input_ids = input_ids_1d.unsqueeze(0).to(device)
    attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

    use_amp = (device.type == "cuda") and (amp_dtype in (torch.float16, torch.bfloat16))
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)

    with ctx:
        base = _get_base_model(model)
        if base is None:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
        else:
            out = base(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                hidden = out.last_hidden_state
            elif hasattr(out, "hidden_states") and out.hidden_states is not None:
                hidden = out.hidden_states[-1]
            else:
                raise RuntimeError("no last_hidden_state")

        h = hidden[0]
        pos_t = torch.tensor(list(positions), device=device, dtype=torch.long)
        h_sel = h.index_select(0, pos_t)

        lm_head = _unwrap_lm_head(model)
        w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
        b = None
        if hasattr(lm_head, "bias") and lm_head.bias is not None:
            b = lm_head.bias[distribution_token_id : distribution_token_id + num_bins]

        logits = F.linear(h_sel, w, b).float()
        num_reward = int(num_bins // num_length_bins)
        logits_rs = logits.view(-1, num_reward, num_length_bins)
        logits_reward = torch.logsumexp(logits_rs, dim=2)
        probs_reward = torch.softmax(logits_reward, dim=1)
        p_corr = probs_reward[:, int(correct_reward_index)]

    return p_corr.detach().cpu().tolist()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--critic-model-id", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--out-html", required=True)

    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--tail-steps", type=int, default=50)
    p.add_argument("--trace-path", type=str, default=None)
    p.add_argument("--prompt-idx", type=int, nargs="*", default=None)

    p.add_argument("--examples-per-prompt", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--min-learning-rate", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)

    p.add_argument("--max-length", type=int, default=131072)
    p.add_argument("--shuffle-seed", type=int, default=0)

    p.add_argument("--distribution-token-id", type=int, default=151669)
    p.add_argument("--length-bins", type=int, nargs="+", default=[0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    p.add_argument("--reward-values", type=float, nargs="+", default=[0.0, 1.0])

    p.add_argument("--adv-smoothing-tau", type=float, default=0.1)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--adv-clip", type=float, default=0.0)

    p.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--col-prompt-idx", type=str, default="prompt_idx")
    p.add_argument("--col-prompt", type=str, default="prompt")
    p.add_argument("--col-output-token-ids", type=str, default="output_token_ids")
    p.add_argument("--col-reward", type=str, default="correct")
    p.add_argument("--col-p-correct", type=str, default="critic_p_correct")
    p.add_argument("--use-parquet-p-correct", action="store_true")

    p.add_argument("--max-rollouts-per-prompt", type=int, default=0)
    p.add_argument("--max-tokens-per-rollout", type=int, default=0)

    return p.parse_args()


def _collect_prompt_ids(parquet_path: str, col_prompt_idx: str) -> List[int]:
    pf = pq.ParquetFile(parquet_path)
    s: Set[int] = set()
    for rb in pf.iter_batches(columns=[col_prompt_idx], batch_size=65536):
        vals = rb.column(0).to_pylist()
        for v in vals:
            if v is None:
                continue
            s.add(int(v))
    return sorted(s)


def _select_prompts(args: argparse.Namespace, prompt_ids_sorted: List[int]) -> List[int]:
    if args.prompt_idx and len(args.prompt_idx) > 0:
        out: List[int] = []
        have = set(prompt_ids_sorted)
        for pid in args.prompt_idx:
            if int(pid) in have and int(pid) not in out:
                out.append(int(pid))
        return out

    sample_prompt_ids: List[int] = []
    for pid in prompt_ids_sorted:
        for _ in range(int(args.examples_per_prompt)):
            sample_prompt_ids.append(int(pid))

    if args.trace_path:
        recs = _read_trace(args.trace_path)
        tail = recs[-int(args.tail_steps) :] if int(args.tail_steps) > 0 else recs
        seen: Set[int] = set()
        sel: List[int] = []
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
            if pid in seen:
                continue
            seen.add(pid)
            sel.append(pid)
            if len(sel) >= int(args.num_prompts):
                break
        return list(reversed(sel))

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

    seen: Set[int] = set()
    sel: List[int] = []
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
            sel.append(int(pid))
            if len(sel) >= int(args.num_prompts):
                break
        if len(sel) >= int(args.num_prompts):
            break
    return sel


def _as_list(x) -> List[Any]:
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _collect_rows_for_prompts(
    parquet_path: str,
    *,
    prompt_ids: Sequence[int],
    col_prompt_idx: str,
    col_prompt: str,
    col_output_token_ids: str,
    col_reward: str,
    col_p_correct: str,
) -> Tuple[Dict[int, str], Dict[int, List[RolloutRow]], bool]:
    want: Set[int] = set(int(x) for x in prompt_ids)
    prompt_text: Dict[int, str] = {}
    groups: Dict[int, List[RolloutRow]] = {int(pid): [] for pid in prompt_ids}

    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    names = set(schema.names)
    have_p = col_p_correct in names
    cols = [col_prompt_idx, col_prompt, col_output_token_ids, col_reward]
    if have_p:
        cols.append(col_p_correct)

    for rb in pf.iter_batches(columns=cols, batch_size=4096):
        c_pid = rb.column(0).to_pylist()
        c_prompt = rb.column(1).to_pylist()
        c_out = rb.column(2).to_pylist()
        c_r = rb.column(3).to_pylist()
        c_p = rb.column(4).to_pylist() if have_p else None

        for i in range(rb.num_rows):
            pid = c_pid[i]
            if pid is None:
                continue
            pid_i = int(pid)
            if pid_i not in want:
                continue

            if pid_i not in prompt_text:
                pr = c_prompt[i]
                if pr is not None:
                    prompt_text[pid_i] = str(pr)

            toks = c_out[i]
            tok_list = _as_list(toks)

            rv = c_r[i]
            try:
                reward = float(rv) if rv is not None else 0.0
            except Exception:
                reward = 0.0
            if reward < 0.0:
                reward = 0.0
            if reward > 1.0:
                reward = 1.0

            p_list: Optional[List[float]] = None
            if have_p:
                pv = c_p[i]
                if pv is None:
                    p_list = None
                else:
                    p_raw = _as_list(pv)
                    p_list = []
                    for x in p_raw:
                        try:
                            p_list.append(float(x))
                        except Exception:
                            p_list.append(float("nan"))

            groups[pid_i].append(
                RolloutRow(
                    row_idx=len(groups[pid_i]),
                    token_ids=[int(x) for x in tok_list],
                    reward=reward,
                    p_correct=p_list,
                )
            )

    for pid in prompt_ids:
        if int(pid) not in prompt_text:
            prompt_text[int(pid)] = ""

    return prompt_text, groups, have_p


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = _torch_dtype(args.dtype)

    tok = AutoTokenizer.from_pretrained(args.critic_model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    attn_impl = str(args.attn_implementation)
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.critic_model_id,
        torch_dtype=amp_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model.to(device)

    length_bins = [int(x) for x in args.length_bins]
    num_length_bins = len(length_bins) - 1
    reward_values = [float(x) for x in args.reward_values]
    num_reward_states = len(reward_values)
    num_bins = int(num_length_bins * num_reward_states)
    correct_reward_index = int(max(range(len(reward_values)), key=lambda i: reward_values[i]))

    prompt_ids_sorted = _collect_prompt_ids(args.data_path, args.col_prompt_idx)
    selected_prompt_ids = _select_prompts(args, prompt_ids_sorted)

    prompt_texts, groups, have_p_col = _collect_rows_for_prompts(
        args.data_path,
        prompt_ids=selected_prompt_ids,
        col_prompt_idx=args.col_prompt_idx,
        col_prompt=args.col_prompt,
        col_output_token_ids=args.col_output_token_ids,
        col_reward=args.col_reward,
        col_p_correct=args.col_p_correct,
    )

    title = f"adv report: {os.path.basename(str(args.data_path))}"
    parts: List[str] = [_html_header(title)]

    parts.append('<div class="meta">')
    parts.append(f"critic_model_id: {html_lib.escape(str(args.critic_model_id))}<br>")
    parts.append(f"data_path: {html_lib.escape(str(args.data_path))}<br>")
    parts.append(f"max_length: {int(args.max_length)}<br>")
    parts.append(f"distribution_token_id: {int(args.distribution_token_id)}<br>")
    parts.append(f"length_bins: {html_lib.escape(str(length_bins))}<br>")
    parts.append(f"reward_values: {html_lib.escape(str(reward_values))}<br>")
    parts.append(f"adv_smoothing_tau: {float(args.adv_smoothing_tau)}<br>")
    parts.append(f"gae_lambda: {float(args.gae_lambda)}<br>")
    parts.append(f"adv_clip: {float(args.adv_clip)}<br>")
    parts.append(f"use_parquet_p_correct: {1 if bool(args.use_parquet_p_correct) else 0}<br>")
    parts.append(f"have_p_correct_column: {1 if have_p_col else 0}<br>")
    parts.append(f"selected_prompts: {html_lib.escape(str(selected_prompt_ids))}<br>")
    parts.append("</div>")

    decode_cache: Dict[int, str] = {}
    scale = 1.0

    for p_i, pid in enumerate(selected_prompt_ids, start=1):
        rollouts = list(groups.get(int(pid), []))
        if int(args.max_rollouts_per_prompt) > 0:
            rollouts = rollouts[: int(args.max_rollouts_per_prompt)]

        prompt = prompt_texts.get(int(pid), "")

        parts.append("<hr>")
        parts.append(f"<h2>prompt {p_i} | prompt_idx {int(pid)} | rollouts {len(rollouts)}</h2>")
        parts.append("<h3>prompt</h3>")
        parts.append(f"<pre>{html_lib.escape(str(prompt))}</pre>")
        parts.append("<h3>rollouts</h3>")
        parts.append('<div class="grid">')

        ctx_infos = _build_ctx_infos(rollouts, tok)

        for r_i, r in enumerate(rollouts):
            token_ids = list(r.token_ids)
            if int(args.max_tokens_per_rollout) > 0:
                token_ids = token_ids[: int(args.max_tokens_per_rollout)]

            use_p = bool(args.use_parquet_p_correct) and (r.p_correct is not None) and (len(r.p_correct) > 0)
            src_flag = "parquet" if use_p else "model"

            if use_p:
                p_corr = list(r.p_correct or [])
                if len(p_corr) < len(r.token_ids):
                    p_corr = p_corr + [float("nan")] * (len(r.token_ids) - len(p_corr))
                p_corr = p_corr[: len(r.token_ids)]
            else:
                full_ids, positions, _truncated, _ctx_keep = _pack_target_last(
                    tok,
                    prompt=prompt,
                    rollouts=rollouts,
                    ctx_infos=ctx_infos,
                    target_idx=r_i,
                    max_length=int(args.max_length),
                    shuffle_seed=int(args.shuffle_seed),
                    prompt_idx=int(pid),
                )
                orig_len = len(r.token_ids)
                p_prefix = _predict_p_correct_for_positions(
                    model,
                    input_ids_1d=torch.tensor(full_ids, dtype=torch.long),
                    positions=positions,
                    distribution_token_id=int(args.distribution_token_id),
                    num_bins=int(num_bins),
                    num_length_bins=int(num_length_bins),
                    correct_reward_index=int(correct_reward_index),
                    device=device,
                    amp_dtype=amp_dtype,
                )
                if len(p_prefix) < orig_len:
                    p_corr = p_prefix + [float("nan")] * (orig_len - len(p_prefix))
                else:
                    p_corr = p_prefix[:orig_len]

            adv, vals = _compute_advantages(
                p_corr,
                tau=float(args.adv_smoothing_tau),
                gae_lambda=float(args.gae_lambda),
                adv_clip=float(args.adv_clip),
            )

            adv_disp = adv[: len(token_ids)]
            p_disp = p_corr[: len(token_ids)]
            v_disp = vals[: len(token_ids)]

            finite_p = [float(x) for x in p_disp if _is_finite(float(x))]
            mean_p = float(np.mean(finite_p)) if finite_p else float("nan")
            finite_a = [float(x) for x in adv_disp if _is_finite(float(x))]
            mean_a = float(np.mean(finite_a)) if finite_a else float("nan")

            spans: List[str] = []
            for t, tid in enumerate(token_ids):
                a = float(adv_disp[t]) if t < len(adv_disp) else float("nan")
                p_t = float(p_disp[t]) if t < len(p_disp) else float("nan")
                v_t = float(v_disp[t]) if t < len(v_disp) else float("nan")
                s = _decode_token(tok, int(tid), decode_cache)
                bg = _adv_color(a, scale)
                tip = f"t {t} adv {a:.6f} p {p_t:.6f} v {v_t:.6f} tid {int(tid)}"
                spans.append(
                    f'<span class="tok" style="background-color:{bg}" title="{html_lib.escape(tip)}">{html_lib.escape(s)}</span>'
                )

            parts.append('<div class="card">')
            hdr = f"rollout {r_i} gt {float(r.reward):.4f} mean_p {mean_p:.6f} mean_adv {mean_a:.6f} src {src_flag}"
            parts.append(f'<div class="hdr">{html_lib.escape(hdr)}</div>')
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
