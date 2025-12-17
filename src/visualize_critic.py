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

# Qwen role/control token ids (set for Qwen3-style chat)
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
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("no hidden_states")
        return hs[-1]
    out = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=False, return_dict=True)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    raise RuntimeError("no last hidden state")


@dataclass
class TrajectoryMeta:
    local_idx: int
    reward_gt: float
    supervised: bool
    token_ids: List[int]          # content + footer (no header)
    abs_positions: List[int]      # positions for token_ids in packed input_ids
    truncated: bool


class JointDistributionPackedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int,
        label_column: str,
        ablation_type: str,
        examples_per_prompt: int,
        supervise_from_trajectory: int,
        correctness_only: bool,
        shuffle_seed: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.ablation_type = ablation_type
        self.supervise_from_trajectory = int(supervise_from_trajectory)
        self.correctness_only = bool(correctness_only)

        if shuffle_seed is not None:
            random.seed(int(shuffle_seed))

        label_choice = (label_column or "correct").lower()
        cols = ["prompt_idx", "prompt", "output_token_ids"]
        if label_choice == "auto":
            cols += ["correct", "value"]
        elif label_choice == "correct":
            cols += ["correct"]
        elif label_choice == "value":
            cols += ["value"]
        else:
            raise ValueError(f"bad label_column: {label_column}")

        table = pq.read_table(data_path, columns=list(dict.fromkeys(cols)))
        df = table.to_pandas()

        need = ["prompt_idx", "prompt", "output_token_ids"]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"missing col: {c}")

        if label_choice == "auto":
            if "correct" in df.columns:
                self.reward_column = "correct"
            elif "value" in df.columns:
                self.reward_column = "value"
            else:
                raise ValueError("need correct or value")
        elif label_choice == "correct":
            if "correct" not in df.columns:
                raise ValueError("need correct")
            self.reward_column = "correct"
        else:
            if "value" not in df.columns:
                raise ValueError("need value")
            self.reward_column = "value"

        if self.reward_column == "correct":
            df["correct"] = df["correct"].astype(float)

        self.samples: List[Dict[str, Any]] = []
        grouped = df.groupby("prompt_idx", sort=True)
        for prompt_idx, group in grouped:
            rows = group.to_dict("records")
            if not rows:
                continue
            prompt_text = rows[0]["prompt"]
            trajectories = []
            for r in rows:
                toks = r["output_token_ids"]
                if hasattr(toks, "tolist"):
                    toks = toks.tolist()
                trajectories.append(
                    {
                        "response_ids": list(toks),
                        "reward": float(r[self.reward_column]),
                    }
                )
            for _ in range(int(examples_per_prompt)):
                trajs = list(trajectories)
                random.shuffle(trajs)
                self.samples.append(
                    {
                        "prompt_idx": int(prompt_idx),
                        "prompt": prompt_text,
                        "trajectories": trajs,
                    }
                )

        if self.correctness_only:
            self.length_bins = [0, self.max_length + 1]
        else:
            self.length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        self.num_length_bins = len(self.length_bins) - 1

        if self.reward_column == "correct":
            self.reward_values = [0.0, 1.0]
        else:
            self.reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]

        self.num_reward_states = len(self.reward_values)
        self.num_bins = self.num_length_bins * self.num_reward_states

        self.value_bin_edges = self._value_edges()

    def _value_edges(self) -> List[float]:
        if self.num_reward_states >= 2:
            edges = [0.0] * (self.num_reward_states + 1)
            for i in range(1, self.num_reward_states):
                edges[i] = 0.5 * (self.reward_values[i - 1] + self.reward_values[i])
            first_step = self.reward_values[1] - self.reward_values[0]
            last_step = self.reward_values[-1] - self.reward_values[-2]
            edges[0] = self.reward_values[0] - 0.5 * first_step
            edges[-1] = self.reward_values[-1] + 0.5 * last_step
        else:
            edges = [self.reward_values[0] - 0.5, self.reward_values[0] + 0.5]
        edges[0] = max(0.0, edges[0])
        edges[-1] = min(1.0, edges[-1])
        return edges

    def __len__(self) -> int:
        return len(self.samples)

    def _get_bin_idx(self, tokens_to_completion: int, reward: float) -> int:
        length_bin = 0
        for i in range(len(self.length_bins) - 1):
            if tokens_to_completion >= self.length_bins[i] and tokens_to_completion < self.length_bins[i + 1]:
                length_bin = i
                break
        if tokens_to_completion >= self.length_bins[-1]:
            length_bin = self.num_length_bins - 1

        reward_state = 0
        for i in range(len(self.value_bin_edges) - 1):
            if reward >= self.value_bin_edges[i] and reward < self.value_bin_edges[i + 1]:
                reward_state = i
                break
        if reward >= self.value_bin_edges[-1]:
            reward_state = self.num_reward_states - 1

        return length_bin + reward_state * self.num_length_bins

    @staticmethod
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

    def get_packed(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        full_input_ids: List[int] = []
        full_label_positions: List[int] = []
        full_bin_labels: List[int] = []
        traj_meta: List[TrajectoryMeta] = []

        prompt_tokens = self.tokenizer.encode(sample["prompt"], add_special_tokens=False)
        full_input_ids.extend(self.create_tokenized_message("user", prompt_tokens))
        prefix_len = len(full_input_ids)

        trajectories = sample["trajectories"]
        if not trajectories:
            input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
            return {
                "prompt_idx": sample["prompt_idx"],
                "prompt": sample["prompt"],
                "input_ids": input_tensor,
                "label_positions": full_label_positions,
                "bin_labels": full_bin_labels,
                "num_bins": self.num_bins,
                "traj_meta": traj_meta,
            }

        infos: List[Dict[str, Any]] = []
        for traj in trajectories:
            traj_content = list(traj["response_ids"])
            header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
            has_eos = bool(traj_content) and (traj_content[-1] == IM_END_TOKEN_ID)
            footer = [] if has_eos else [IM_END_TOKEN_ID]
            traj_block = header + traj_content + footer

            content_plus_footer_len = len(traj_content) + len(footer)
            full_feedback_str = f"Reward: {float(traj['reward'])}\nLength: {content_plus_footer_len} tokens"
            length_only_str = f"Length: {content_plus_footer_len} tokens"

            reward_tokens_full = self.tokenizer.encode(full_feedback_str, add_special_tokens=False)
            reward_tokens_len = self.tokenizer.encode(length_only_str, add_special_tokens=False)

            reward_block_len_full = len(reward_tokens_full) + 5
            reward_block_len_len = len(reward_tokens_len) + 5

            infos.append(
                {
                    "traj": traj,
                    "traj_content": traj_content,
                    "header": header,
                    "footer": footer,
                    "traj_block": traj_block,
                    "traj_block_len": len(traj_block),
                    "content_plus_footer_len": content_plus_footer_len,
                    "reward_tokens_full": reward_tokens_full,
                    "reward_tokens_len": reward_tokens_len,
                    "reward_block_len_full": reward_block_len_full,
                    "reward_block_len_len": reward_block_len_len,
                }
            )

        n = len(infos)
        assistant_suffix = [0] * (n + 1)
        full_fb_suffix = [0] * (n + 1)
        len_fb_suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            assistant_suffix[i] = assistant_suffix[i + 1] + infos[i]["traj_block_len"]
            full_fb_suffix[i] = full_fb_suffix[i + 1]
            len_fb_suffix[i] = len_fb_suffix[i + 1]
            if i != n - 1:
                full_fb_suffix[i] += infos[i]["reward_block_len_full"]
                len_fb_suffix[i] += infos[i]["reward_block_len_len"]

        start_idx = n - 1
        for s in range(n):
            base = prefix_len + assistant_suffix[s]
            if self.ablation_type in ("full", "no_ans"):
                total = base + full_fb_suffix[s]
            elif self.ablation_type == "no_ans_no_rewards":
                total = base + len_fb_suffix[s]
            elif self.ablation_type == "no_ans_first_reward_only":
                if s == n - 1:
                    total = base
                else:
                    total = base + infos[s]["reward_block_len_full"] + len_fb_suffix[s + 1]
            else:
                total = base + full_fb_suffix[s]
            if total <= self.max_length:
                start_idx = s
                break

        included = infos[start_idx:]

        if start_idx == n - 1:
            last = included[0]
            budget = self.max_length - prefix_len
            min_block = len(last["header"]) + len(last["footer"])
            if budget < min_block:
                input_tensor = torch.tensor(full_input_ids[: self.max_length], dtype=torch.long)
                return {
                    "prompt_idx": sample["prompt_idx"],
                    "prompt": sample["prompt"],
                    "input_ids": input_tensor,
                    "label_positions": [],
                    "bin_labels": [],
                    "num_bins": self.num_bins,
                    "traj_meta": [],
                }

            max_content_len = budget - min_block
            if max_content_len < 0:
                max_content_len = 0

            trunc_content = list(last["traj_content"][:max_content_len])
            has_eos = bool(trunc_content) and trunc_content[-1] == IM_END_TOKEN_ID
            footer = [] if has_eos else [IM_END_TOKEN_ID]
            traj_block = last["header"] + trunc_content + footer

            included = [
                {
                    **last,
                    "traj_content": trunc_content,
                    "footer": footer,
                    "traj_block": traj_block,
                    "traj_block_len": len(traj_block),
                    "content_plus_footer_len": len(trunc_content) + len(footer),
                }
            ]

        supervise_start_idx = max(0, self.supervise_from_trajectory - 1)
        if len(included) <= supervise_start_idx:
            supervise_start_idx = max(0, len(included) - 1)

        for local_idx, info in enumerate(included):
            is_supervised = local_idx >= supervise_start_idx

            current_start_idx = len(full_input_ids)
            traj_block = info["traj_block"]
            header_len = len(info["header"])
            token_ids = list(info["traj_content"]) + list(info["footer"])
            abs_positions = [current_start_idx + header_len + i for i in range(len(token_ids))]

            reward_val = float(info["traj"]["reward"])
            reward_val = 0.0 if reward_val < 0.0 else 1.0 if reward_val > 1.0 else reward_val

            if is_supervised:
                block_len = info["traj_block_len"]
                for i in range(info["content_plus_footer_len"]):
                    abs_pos = current_start_idx + header_len + i
                    tokens_to_completion = (current_start_idx + block_len) - abs_pos - 1
                    bin_idx = self._get_bin_idx(tokens_to_completion, reward_val)
                    full_label_positions.append(abs_pos)
                    full_bin_labels.append(bin_idx)

            full_input_ids.extend(traj_block)

            if local_idx != (len(included) - 1):
                if self.ablation_type in ("full", "no_ans"):
                    reward_tokens = info["reward_tokens_full"]
                elif self.ablation_type == "no_ans_no_rewards":
                    reward_tokens = info["reward_tokens_len"]
                elif self.ablation_type == "no_ans_first_reward_only":
                    reward_tokens = info["reward_tokens_full"] if local_idx == 0 else info["reward_tokens_len"]
                else:
                    reward_tokens = info["reward_tokens_full"]
                full_input_ids.extend(self.create_tokenized_message("user", reward_tokens))

            traj_meta.append(
                TrajectoryMeta(
                    local_idx=local_idx,
                    reward_gt=reward_val,
                    supervised=is_supervised,
                    token_ids=token_ids,
                    abs_positions=abs_positions,
                    truncated=(start_idx == n - 1 and local_idx == 0 and len(info["traj_content"]) < len(info["traj"]["response_ids"])),
                )
            )

        input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
        return {
            "prompt_idx": sample["prompt_idx"],
            "prompt": sample["prompt"],
            "input_ids": input_tensor,
            "label_positions": full_label_positions,
            "bin_labels": full_bin_labels,
            "num_bins": self.num_bins,
            "traj_meta": traj_meta,
        }


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

    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=(device.type == "cuda")):
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
    out = {int(p): float(v) for p, v in zip(positions, exp_r_cpu)}
    return out


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
.prompt {{ margin-top: 8px; }}
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--out_html", required=True)

    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--tail_steps", type=int, default=50)
    p.add_argument("--trace_path", type=str, default=None)

    p.add_argument("--distribution_token_id", type=int, default=151669)
    p.add_argument("--label_column", choices=["auto", "correct", "value"], default="correct")
    p.add_argument("--ablation_type", choices=["full", "no_ans", "no_ans_no_rewards", "no_ans_first_reward_only"], default="no_ans")
    p.add_argument("--examples_per_prompt", type=int, default=1)
    p.add_argument("--supervise_from_trajectory", type=int, default=4)
    p.add_argument("--correctness_only", action="store_true")
    p.add_argument("--max_length", type=int, default=131072)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle_seed", type=int, default=0)

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

    p.add_argument("--max_trajectories", type=int, default=0)
    p.add_argument("--max_tokens_per_traj", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    amp_dtype = _torch_dtype(args.dtype)

    tok = AutoTokenizer.from_pretrained(args.weights_path, trust_remote_code=bool(args.trust_remote_code))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model_kwargs: Dict[str, Any] = {"torch_dtype": amp_dtype, "trust_remote_code": bool(args.trust_remote_code)}
    if args.attn_implementation and args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.weights_path, **model_kwargs)
    model.to(device)
    model.eval()

    ds = JointDistributionPackedDataset(
        data_path=args.data_path,
        tokenizer=tok,
        max_length=args.max_length,
        label_column=args.label_column,
        ablation_type=args.ablation_type,
        examples_per_prompt=args.examples_per_prompt,
        supervise_from_trajectory=args.supervise_from_trajectory,
        correctness_only=bool(args.correctness_only),
        shuffle_seed=(None if args.shuffle_seed is None else int(args.shuffle_seed)),
    )

    prompt_to_indices: Dict[int, List[int]] = {}
    for i, s in enumerate(ds.samples):
        pid = int(s["prompt_idx"])
        prompt_to_indices.setdefault(pid, []).append(i)

    selected: List[Dict[str, Any]] = []
    total_steps = 0

    if args.trace_path:
        recs = _read_trace(args.trace_path)
        if not recs:
            raise RuntimeError("trace empty")
        tail = recs[-int(args.tail_steps) :] if int(args.tail_steps) > 0 else recs

        seen = set()
        for r in reversed(tail):
            step = int(r.get("step", 0))
            pid = r.get("prompt_idx", None)
            sample_indices = r.get("sample_indices", [])
            if pid is None:
                for si in sample_indices:
                    if 0 <= int(si) < len(ds.samples):
                        pid = int(ds.samples[int(si)]["prompt_idx"])
                        break
            if pid is None:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            use_si = sample_indices[0] if sample_indices else prompt_to_indices.get(pid, [None])[0]
            selected.append({"prompt_idx": pid, "sample_idx": use_si, "step": step, "lr": None})
            if len(selected) >= int(args.num_prompts):
                break
        selected = list(selected)
    else:
        steps = _simulate_optimizer_steps(
            dataset_len=len(ds),
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
        for step_idx, batch_inds in zip(range(total_steps - tail_steps + 1, total_steps + 1), tail):
            lr = _lr_schedule(
                step=int(step_idx),
                total_steps=int(total_steps),
                lr=float(args.learning_rate),
                min_lr=float(args.min_learning_rate),
                warmup_ratio=float(args.warmup_ratio),
            )
            for si in batch_inds:
                pid = int(ds.samples[int(si)]["prompt_idx"])
                if pid in seen:
                    continue
                seen.add(pid)
                selected.append({"prompt_idx": pid, "sample_idx": int(si), "step": int(step_idx), "lr": float(lr)})
                if len(selected) >= int(args.num_prompts):
                    break
            if len(selected) >= int(args.num_prompts):
                break
        selected = list(reversed(selected))

    title = f"critic report: {os.path.basename(args.weights_path.rstrip('/'))}"
    out_parts: List[str] = [_html_header(title)]

    out_parts.append('<div class="meta">')
    out_parts.append(f"weights_path: {html_lib.escape(args.weights_path)}<br>")
    out_parts.append(f"data_path: {html_lib.escape(args.data_path)}<br>")
    out_parts.append(f"label_column: {html_lib.escape(args.label_column)}<br>")
    out_parts.append(f"reward_column: {html_lib.escape(ds.reward_column)}<br>")
    out_parts.append(f"ablation_type: {html_lib.escape(args.ablation_type)}<br>")
    out_parts.append(f"supervise_from_trajectory: {int(args.supervise_from_trajectory)}<br>")
    out_parts.append(f"examples_per_prompt: {int(args.examples_per_prompt)}<br>")
    out_parts.append(f"max_length: {int(args.max_length)}<br>")
    out_parts.append(f"distribution_token_id: {int(args.distribution_token_id)}<br>")
    out_parts.append(f"num_bins: {int(ds.num_bins)}<br>")
    out_parts.append(f"num_reward_states: {int(ds.num_reward_states)}<br>")
    out_parts.append(f"num_length_bins: {int(ds.num_length_bins)}<br>")
    out_parts.append(f"reward_values: {html_lib.escape(str(ds.reward_values))}<br>")
    if total_steps:
        out_parts.append(f"sim_total_steps: {int(total_steps)}<br>")
    out_parts.append("</div>")

    decode_cache: Dict[int, str] = {}

    for item_i, item in enumerate(selected, start=1):
        sample_idx = item.get("sample_idx", None)
        pid = int(item["prompt_idx"])
        step = item.get("step", None)
        lr = item.get("lr", None)

        if sample_idx is None or not (0 <= int(sample_idx) < len(ds.samples)):
            si_list = prompt_to_indices.get(pid, [])
            if not si_list:
                continue
            sample_idx = si_list[0]

        pack = ds.get_packed(int(sample_idx))
        input_ids_1d: torch.Tensor = pack["input_ids"]
        traj_meta: List[TrajectoryMeta] = pack["traj_meta"]

        max_traj = int(args.max_trajectories)
        if max_traj > 0 and len(traj_meta) > max_traj:
            traj_meta = traj_meta[:max_traj]

        max_tok = int(args.max_tokens_per_traj)
        if max_tok > 0:
            new_meta: List[TrajectoryMeta] = []
            for tm in traj_meta:
                if len(tm.token_ids) > max_tok:
                    new_meta.append(
                        TrajectoryMeta(
                            local_idx=tm.local_idx,
                            reward_gt=tm.reward_gt,
                            supervised=tm.supervised,
                            token_ids=tm.token_ids[:max_tok],
                            abs_positions=tm.abs_positions[:max_tok],
                            truncated=True,
                        )
                    )
                else:
                    new_meta.append(tm)
            traj_meta = new_meta

        pos_set = set()
        for tm in traj_meta:
            for p in tm.abs_positions:
                pos_set.add(int(p))
        positions = sorted(pos_set)

        pos_to_er = _expected_reward_per_pos(
            model=model,
            input_ids_1d=input_ids_1d,
            positions=positions,
            distribution_token_id=int(args.distribution_token_id),
            num_bins=int(ds.num_bins),
            num_length_bins=int(ds.num_length_bins),
            reward_values=list(ds.reward_values),
            device=device,
            amp_dtype=amp_dtype,
        )

        out_parts.append("<hr>")
        out_parts.append(f"<h2>item {item_i}</h2>")
        out_parts.append('<div class="meta">')
        out_parts.append(f"prompt_idx: {pid}<br>")
        out_parts.append(f"sample_idx: {int(sample_idx)}<br>")
        if step is not None:
            out_parts.append(f"step: {int(step)}<br>")
        if lr is not None:
            out_parts.append(f"lr: {float(lr):.8f}<br>")
        out_parts.append("</div>")

        out_parts.append("<h3>prompt</h3>")
        out_parts.append(f'<pre class="prompt">{html_lib.escape(str(pack["prompt"]))}</pre>')

        out_parts.append("<h3>trajectories</h3>")
        out_parts.append('<div class="grid">')

        for tm in traj_meta:
            er_vals: List[float] = []
            spans: List[str] = []
            for tid, ap in zip(tm.token_ids, tm.abs_positions):
                er = float(pos_to_er.get(int(ap), 0.0))
                er_vals.append(er)
                s = _decode_token(tok, int(tid), decode_cache)
                spans.append(
                    f'<span class="tok" style="background-color:{_reward_rgb(er)}" '
                    f'title="pos {int(ap)} er {er:.4f}">{html_lib.escape(s)}</span>'
                )
            er_mean = float(np.mean(er_vals)) if er_vals else 0.0

            hdr = [
                f"traj {tm.local_idx}",
                f"gt {tm.reward_gt:.4f}",
                f"mean {er_mean:.4f}",
                f"sup {1 if tm.supervised else 0}",
                f"tok {len(tm.token_ids)}",
                f"trunc {1 if tm.truncated else 0}",
            ]
            out_parts.append('<div class="card">')
            out_parts.append(f'<div class="hdr">{html_lib.escape(" | ".join(hdr))}</div>')
            out_parts.append(f'<div class="toks">{"".join(spans)}</div>')
            out_parts.append("</div>")

        out_parts.append("</div>")

    out_parts.append(_html_footer())

    out_dir = os.path.dirname(os.path.abspath(args.out_html))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write("".join(out_parts))

    print(args.out_html)


if __name__ == "__main__":
    main()
