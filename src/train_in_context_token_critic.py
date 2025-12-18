#!/usr/bin/env python3
"""
Fine-tune an "In-Context Token Critic" that predicts correctness for each possible next token.

This is a Q-style critic over tokens: for a given state (prefix) s, the model produces a logit
for every next-token action a. We interpret each action logit independently via a sigmoid as
P(correct | s, a).

Training supervision matches `train_in_context_critic.py` (same packed multi-trajectory prompts
with in-context feedback), except:
  - Only correctness is predicted (no length/cost target).
  - Loss is BCE-with-logits on the logit of the actually-taken next token at each supervised state.
  - Correctness label smoothing defaults to 0.1 with targets {0 -> 0.1, 1 -> 0.9}.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from typing import Any, Dict, List

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Qwen Specific Control Tokens ---
# Verify these match your specific model version if not using Qwen 2.5/3
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
SYSTEM_TOKEN_ID = 8948
NEWLINE_TOKEN_ID = 198


class TokenCorrectnessDataset(Dataset):
    def __init__(
        self,
        table: str,
        tokenizer,
        max_length: int = 131_072,
        thinking_only: bool = False,
        thinking_token_id: int = 151667,
        label_column: str = "correct",
        ablation_type: str = "no_ans",
        examples_per_prompt: int = 1,
        supervise_from_trajectory: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ablation_type = ablation_type
        self.supervise_from_trajectory = int(supervise_from_trajectory)

        df = pq.read_table(table).to_pandas()

        required_cols = ["prompt_idx", "prompt", "output_token_ids"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        choice = (label_column or "correct").lower()
        if choice == "auto":
            if "correct" in df.columns:
                self.reward_column = "correct"
            elif "value" in df.columns:
                self.reward_column = "value"
            else:
                raise ValueError("Data must contain 'correct' or 'value' column.")
        elif choice == "correct":
            if "correct" not in df.columns:
                raise ValueError("Requested label_column='correct' but column 'correct' is not present.")
            self.reward_column = "correct"
        elif choice == "value":
            if "value" not in df.columns:
                raise ValueError("Requested label_column='value' but 'value' not found.")
            self.reward_column = "value"
        else:
            raise ValueError(f"Unknown label_column: {label_column}")

        if self.reward_column == "correct":
            df["correct"] = df["correct"].astype(float)

        if thinking_only:
            if "reasoning_enabled" in df.columns:
                df = df[df["reasoning_enabled"] == True].copy()
            else:
                print(
                    "Warning: --thinking_only requested but 'reasoning_enabled' column not found. Skipping filter."
                )

        print("Grouping dataset by prompt for Packed In-Context generation...")
        self.samples = []
        grouped = df.groupby("prompt_idx")
        for _, group in grouped:
            rows = group.to_dict("records")
            if not rows:
                continue

            prompt_text = rows[0]["prompt"]
            trajectories = []
            for r in rows:
                toks = r["output_token_ids"]
                if hasattr(toks, "tolist"):
                    toks = toks.tolist()
                trajectories.append({"response_ids": toks, "reward": float(r[self.reward_column])})

            for _ in range(int(examples_per_prompt)):
                trajs = list(trajectories)
                random.shuffle(trajs)
                self.samples.append({"prompt": prompt_text, "trajectories": trajs})

        print(f"Constructed {len(self.samples)} packed samples ({examples_per_prompt} per prompt).")

    def __len__(self) -> int:
        return len(self.samples)

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        full_input_ids: List[int] = []
        full_label_positions: List[int] = []
        full_action_token_ids: List[int] = []
        full_correct_labels: List[float] = []

        prompt_tokens = self.tokenizer.encode(sample["prompt"], add_special_tokens=False)
        full_input_ids.extend(self.create_tokenized_message("user", prompt_tokens))
        prefix_len = len(full_input_ids)

        trajectories = sample["trajectories"]
        if not trajectories:
            input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
            return {
                "input_ids": input_tensor,
                "label_positions": full_label_positions,
                "action_token_ids": full_action_token_ids,
                "correct_labels": full_correct_labels,
            }

        infos: List[Dict[str, Any]] = []
        for traj in trajectories:
            traj_content = list(traj["response_ids"])

            header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
            has_eos = bool(traj_content and traj_content[-1] == IM_END_TOKEN_ID)
            footer = [] if has_eos else [IM_END_TOKEN_ID]
            traj_block = header + traj_content + footer

            content_plus_footer_len = len(traj_content) + len(footer)

            full_feedback_str = f"Reward: {traj['reward']}\nLength: {content_plus_footer_len} tokens"
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
            if self.ablation_type in {"full", "no_ans"}:
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
                    "input_ids": input_tensor,
                    "label_positions": [],
                    "action_token_ids": [],
                    "correct_labels": [],
                }

            max_content_len = max(0, budget - min_block)
            trunc_content = list(last["traj_content"][:max_content_len])
            has_eos = bool(trunc_content and trunc_content[-1] == IM_END_TOKEN_ID)
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

            if is_supervised:
                reward_val = float(info["traj"]["reward"])
                reward_val = min(1.0, max(0.0, reward_val))

                content_start_rel_idx = len(info["header"])
                for i in range(info["content_plus_footer_len"]):
                    action_abs_pos = current_start_idx + content_start_rel_idx + i
                    state_abs_pos = action_abs_pos - 1
                    if state_abs_pos < 0:
                        continue

                    full_label_positions.append(state_abs_pos)
                    full_action_token_ids.append(traj_block[content_start_rel_idx + i])
                    full_correct_labels.append(reward_val)

            full_input_ids.extend(traj_block)

            if local_idx != (len(included) - 1):
                if self.ablation_type in {"full", "no_ans"}:
                    reward_tokens = info["reward_tokens_full"]
                elif self.ablation_type == "no_ans_no_rewards":
                    reward_tokens = info["reward_tokens_len"]
                elif self.ablation_type == "no_ans_first_reward_only":
                    reward_tokens = info["reward_tokens_full"] if local_idx == 0 else info["reward_tokens_len"]
                else:
                    reward_tokens = info["reward_tokens_full"]
                full_input_ids.extend(self.create_tokenized_message("user", reward_tokens))

        input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "label_positions": full_label_positions,
            "action_token_ids": full_action_token_ids,
            "correct_labels": full_correct_labels,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(s["input_ids"].size(0) for s in batch)
        pad_val = 0
        return {
            "input_ids": torch.stack(
                [F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0)), value=pad_val) for s in batch]
            ),
            "label_positions": [s["label_positions"] for s in batch],
            "action_token_ids": [s["action_token_ids"] for s in batch],
            "correct_labels": [s["correct_labels"] for s in batch],
        }


def _unwrap_model(model):
    tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    return tgt


def _last_hidden_state(tgt, input_ids: torch.Tensor) -> torch.Tensor:
    base = None
    if hasattr(tgt, "model"):
        base = tgt.model
    elif hasattr(tgt, "transformer"):
        base = tgt.transformer

    if base is None:
        out = tgt(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1]

    out = base(input_ids=input_ids, use_cache=False, return_dict=True)
    return out.last_hidden_state


def compute_loss(model, batch, correctness_label_smoothing: float):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    tgt = _unwrap_model(model)
    hidden_states = _last_hidden_state(tgt, input_ids)  # [B, S, E]

    flat_b: List[int] = []
    flat_s: List[int] = []
    flat_a: List[int] = []
    flat_y: List[float] = []
    for i, (pos, actions, ys) in enumerate(
        zip(batch["label_positions"], batch["action_token_ids"], batch["correct_labels"])
    ):
        if len(pos) != len(actions) or len(pos) != len(ys):
            raise ValueError("Mismatched supervision lengths in batch item.")
        flat_b.extend([i] * len(pos))
        flat_s.extend(pos)
        flat_a.extend(actions)
        flat_y.extend(ys)

    if not flat_y:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {
            "acc": zero.detach(),
            "p_pos": zero.detach(),
            "p_neg": zero.detach(),
        }
        return zero, metrics

    b_idx = torch.tensor(flat_b, device=device)
    s_idx = torch.tensor(flat_s, device=device)
    action_ids = torch.tensor(flat_a, device=device)

    y_raw = torch.tensor(flat_y, device=device, dtype=torch.float32)
    if correctness_label_smoothing:
        if correctness_label_smoothing < 0.0 or correctness_label_smoothing > 0.5:
            raise ValueError("--correctness_label_smoothing must be in [0, 0.5].")
        y = y_raw * (1.0 - 2.0 * correctness_label_smoothing) + correctness_label_smoothing
    else:
        y = y_raw

    lm_head = tgt.lm_head if hasattr(tgt, "lm_head") else tgt.get_output_embeddings()
    h = hidden_states[b_idx, s_idx, :]  # [N, E]
    w = lm_head.weight[action_ids]  # [N, E]
    logits = (h * w).sum(dim=-1)
    if hasattr(lm_head, "bias") and lm_head.bias is not None:
        logits = logits + lm_head.bias[action_ids]

    loss = F.binary_cross_entropy_with_logits(logits.float(), y)

    with torch.no_grad():
        probs = torch.sigmoid(logits.float())
        pred = (probs >= 0.5).to(torch.float32)
        gold = (y_raw >= 0.5).to(torch.float32)
        acc = (pred == gold).to(torch.float32).mean()

        pos_mask = gold > 0.5
        neg_mask = ~pos_mask
        p_pos = probs[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0, device=device)
        p_neg = probs[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=device)

    return loss, {"acc": acc, "p_pos": p_pos, "p_neg": p_neg}


def print_trainable(model):
    print("\n=== Trainable parameters ===")
    total = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
    print(f"TOTAL trainable elements: {total}\n")


def train(
    model,
    dataset,
    weights_path: str,
    collate_fn,
    shuffle_data: bool,
    seed: int,
    dtype: str,
    compile_mode: str,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    beta_1: float,
    beta_2: float,
    grad_clip: float,
    wandb_project: str,
    dist_backend: str,
    correctness_label_smoothing: float,
    max_steps: int = -1,
):
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, master = 1, True

    torch.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=shuffle_data) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle_data and sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
    )

    if master and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"run_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "dist_backend": dist_backend,
                "max_steps": max_steps,
                "ablation_type": dataset.ablation_type,
                "supervise_from_trajectory": dataset.supervise_from_trajectory,
                "correctness_label_smoothing": correctness_label_smoothing,
            },
        )

    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[device])
    if master:
        print_trainable(model)

    if compile_mode in {"default", "reduce-overhead", "max-autotune"}:
        model = torch.compile(model, mode=compile_mode)

    model.train()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, betas=(beta_1, beta_2))
    scaler = torch.GradScaler(enabled=(dtype == "float16"))

    total_iters = (num_epochs * len(loader)) // gradient_accumulation_steps
    warmup_iters = int(warmup_ratio * total_iters)

    def lr_schedule(i: int) -> float:
        if warmup_iters > 0 and i < warmup_iters:
            return learning_rate * i / warmup_iters
        if total_iters <= warmup_iters:
            return min_learning_rate
        progress = (i - warmup_iters) / (total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    global_step = 0
    accum_loss = 0.0
    accum_acc = 0.0
    accum_p_pos = 0.0
    accum_p_neg = 0.0

    for epoch in range(num_epochs):
        if distributed:
            sampler.set_epoch(epoch)

        for it, batch in enumerate(loader):
            update = (it + 1) % gradient_accumulation_steps == 0
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = update

            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=getattr(torch, dtype)):
                loss, metrics = compute_loss(model, batch, correctness_label_smoothing)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            stats = torch.stack([loss.detach(), metrics["acc"].detach(), metrics["p_pos"].detach(), metrics["p_neg"].detach()])
            if distributed:
                dist.all_reduce(stats)
                stats /= world

            accum_loss += stats[0].item()
            accum_acc += stats[1].item()
            accum_p_pos += stats[2].item()
            accum_p_neg += stats[3].item()

            if not update:
                continue

            scaler.unscale_(optimizer)
            if grad_clip:
                params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                clip_grad_norm_(params, grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            lr = lr_schedule(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if master and wandb_project:
                wandb.log(
                    {
                        "train/loss": accum_loss,
                        "train/acc": accum_acc,
                        "train/p_pos": accum_p_pos,
                        "train/p_neg": accum_p_neg,
                        "lr": lr,
                        "step": global_step,
                    }
                )
                accum_loss = 0.0
                accum_acc = 0.0
                accum_p_pos = 0.0
                accum_p_neg = 0.0

            if max_steps > 0 and global_step >= max_steps:
                break

        if max_steps > 0 and global_step >= max_steps:
            break

    if master:
        tgt = model.module if hasattr(model, "module") else model
        if hasattr(tgt, "save_pretrained"):
            tgt.save_pretrained(weights_path)
        else:
            torch.save(tgt.state_dict(), weights_path)
        if wandb_project:
            wandb.finish()

    if distributed:
        dist.barrier()


def main_worker(local_rank, world_size, cfg):
    os.environ.update(
        WORLD_SIZE=str(world_size),
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29500",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device).gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = TokenCorrectnessDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        thinking_only=cfg.thinking_only,
        thinking_token_id=cfg.thinking_token_id,
        label_column=cfg.label_column,
        ablation_type=cfg.ablation_type,
        examples_per_prompt=cfg.examples_per_prompt,
        supervise_from_trajectory=cfg.supervise_from_trajectory,
    )

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        TokenCorrectnessDataset.collate_fn,
        True,
        42,
        "bfloat16",
        cfg.compile_mode,
        cfg.num_epochs,
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
        cfg.learning_rate,
        0.0,
        cfg.warmup_ratio,
        cfg.weight_decay,
        0.9,
        0.95,
        1.0,
        cfg.wandb_project,
        cfg.dist_backend,
        cfg.correctness_label_smoothing,
        cfg.max_steps,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/token_critic")
    p.add_argument("--data_path", default="data/zip_training_adaptivemath_data_qwen17b_thinking.parquet")

    p.add_argument("--thinking_only", action="store_true", help="Only train on samples that contain thinking token")
    p.add_argument("--thinking_token_id", type=int, default=151667, help="Token ID for detecting thinking samples")
    p.add_argument(
        "--label-column",
        choices=["auto", "correct", "value"],
        default="correct",
        help="Which column to use for correctness supervision. 'auto' prefers 'correct' if present.",
    )

    p.add_argument(
        "--correctness_label_smoothing",
        type=float,
        default=0.1,
        help="Binary label smoothing (targets: 0->s, 1->1-s). Set 0 to disable.",
    )

    p.add_argument(
        "--ablation_type",
        choices=["full", "no_ans", "no_ans_no_rewards", "no_ans_first_reward_only"],
        default="no_ans",
        help="Controls in-context feedback configuration. 'full' is an alias of 'no_ans'.",
    )
    p.add_argument("--examples_per_prompt", type=int, default=1, help="Packed examples per prompt (reshuffle each time).")
    p.add_argument(
        "--supervise_from_trajectory",
        type=int,
        default=4,
        help="1-indexed trajectory number to start supervision (e.g., 4 => supervise trajectories 4..end).",
    )

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=131_072)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, stop training after this many optimizer steps")

    p.add_argument("--compile_mode", default="none", choices=["none", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--wandb_project", default="new_joint_critic")
    p.add_argument("--dist-backend", choices=["ddp"], default="ddp", help="Distributed training backend")

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()

