#!/usr/bin/env python3
"""
Train a student policy with Group Relative Policy Optimization (GRPO).

GRPO computes group-relative advantages by normalizing rewards within each prompt group,
then uses a PPO-style clipped objective to update the policy.

Key features:
- Group-relative advantages: A_i = (R_i - mean(R_group)) / std(R_group)
- PPO clipped objective: L = -min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
- Uses pre-computed reference logprobs (no frozen reference model needed)
- Optional KL regularization term

Input data must have columns:
- prompt_token_ids: List[int] - tokenized prompt
- output_token_ids: List[int] - the actual trajectory tokens
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_ref_logprobs: List[List[float]] - reference model log probs for each candidate
- correct (optional): float - trajectory reward (0 or 1)
- candidate_q_values (fallback): List[List[float]] - Q-values (uses final Q if correct not available)
- prompt_idx: int - prompt identifier for grouping

Example usage:
    python train_policy_grpo.py --model_id Qwen/Qwen3-1.7B \
                                --weights_path models/policy_grpo \
                                --data_path data/labeled.parquet
"""
from __future__ import annotations
import argparse, math, os, time, random
import wandb
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class PolicyGRPODataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
        advantage_eps: float = 1e-8,
    ):
        """
        Dataset for GRPO training.

        Computes group-relative advantages by normalizing rewards within each prompt group.

        Each sample contains:
        - prompt_token_ids: tokenized prompt
        - output_token_ids: tokenized response
        - candidate_ids: List of candidate token IDs per position
        - candidate_ref_logprobs: Reference logprobs for candidates
        - advantage: Group-relative advantage for this trajectory
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.advantage_eps = advantage_eps

        # Load data - support single path, list of paths, or table
        if isinstance(table, str):
            df = pq.read_table(table).to_pandas()
        elif isinstance(table, list):
            # Multiple paths - concatenate all tables
            dfs = []
            for path in table:
                dfs.append(pq.read_table(path).to_pandas())
                print(f"Loaded {len(dfs[-1])} samples from {path}")
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = table

        # Validate required columns
        required_cols = ["prompt_token_ids", "output_token_ids", "candidate_ids", "candidate_ref_logprobs"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        # Check for reward column
        has_correct = "correct" in df.columns
        has_q_values = "candidate_q_values" in df.columns

        if not has_correct and not has_q_values:
            raise ValueError("Dataset must have either 'correct' column or 'candidate_q_values' column")

        # Filter out rows without labels (None values)
        df = df[df["candidate_ids"].notna()].copy()

        print(f"Loaded {len(df)} labeled samples")

        # Compute rewards for each trajectory
        def get_reward(row):
            # Prefer 'correct' column if available
            if has_correct and pd.notna(row.get("correct")):
                return float(row["correct"])
            # Fall back to final Q-value
            if has_q_values:
                q_values = row.get("candidate_q_values")
                if q_values is not None and len(q_values) > 0:
                    last_position = q_values[-1]
                    if last_position is not None and len(last_position) > 0:
                        if hasattr(last_position, 'tolist'):
                            last_position = last_position.tolist()
                        return max(last_position)
            return None

        df["reward"] = df.apply(get_reward, axis=1)
        df = df[df["reward"].notna()].copy()

        print(f"Found {len(df)} samples with valid rewards")

        # Compute group-relative advantages
        if "prompt_idx" not in df.columns:
            print("Warning: No prompt_idx column found. Using global normalization.")
            mean_r = df["reward"].mean()
            std_r = df["reward"].std() + advantage_eps
            df["advantage"] = (df["reward"] - mean_r) / std_r
        else:
            # Group by prompt_idx and compute normalized advantages
            def compute_group_advantages(group):
                rewards = group["reward"].values
                mean_r = rewards.mean()
                std_r = rewards.std() + advantage_eps
                group = group.copy()
                group["advantage"] = (rewards - mean_r) / std_r
                return group

            df = df.groupby("prompt_idx", group_keys=False).apply(compute_group_advantages)

        print(f"Advantage stats: mean={df['advantage'].mean():.4f}, std={df['advantage'].std():.4f}")

        # Build samples
        if "prompt_idx" in df.columns:
            grouped = df.groupby("prompt_idx")
            self.samples = []

            for prompt_idx, group in grouped:
                rows_with_idx = [(idx, row) for idx, row in group.iterrows()]
                if not rows_with_idx:
                    continue

                for _ in range(int(examples_per_prompt)):
                    row_list = list(rows_with_idx)
                    random.shuffle(row_list)
                    for idx, r in row_list:
                        self.samples.append(self._process_row(r.to_dict()))
        else:
            self.samples = [self._process_row(r.to_dict()) for idx, r in df.iterrows()]
            if examples_per_prompt > 1:
                original = self.samples.copy()
                for _ in range(examples_per_prompt - 1):
                    self.samples.extend(original)

        print(f"Constructed {len(self.samples)} training samples")

    def _process_row(self, row: Dict) -> Dict:
        """Process a single row into a training sample."""
        prompt_ids = row["prompt_token_ids"]
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()

        output_ids = row["output_token_ids"]
        if hasattr(output_ids, 'tolist'):
            output_ids = output_ids.tolist()

        candidate_ids = row["candidate_ids"]
        candidate_ref_logprobs = row["candidate_ref_logprobs"]
        candidate_q_values = row.get("candidate_q_values")

        # Convert numpy arrays if needed
        if hasattr(candidate_ids, 'tolist'):
            candidate_ids = candidate_ids.tolist()
        if hasattr(candidate_ref_logprobs, 'tolist'):
            candidate_ref_logprobs = candidate_ref_logprobs.tolist()
        if candidate_q_values is not None and hasattr(candidate_q_values, 'tolist'):
            candidate_q_values = candidate_q_values.tolist()

        return {
            "prompt_ids": list(prompt_ids),
            "output_ids": list(output_ids),
            "candidate_ids": candidate_ids,
            "candidate_ref_logprobs": candidate_ref_logprobs,
            "candidate_q_values": candidate_q_values,
            "advantage": float(row["advantage"]),
            "reward": float(row["reward"]),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_ids = sample["prompt_ids"]
        output_ids = sample["output_ids"]
        candidate_ids = sample["candidate_ids"]
        candidate_ref_logprobs = sample["candidate_ref_logprobs"]
        candidate_q_values = sample["candidate_q_values"]
        advantage = sample["advantage"]
        reward = sample["reward"]

        # Full sequence: prompt + output
        full_ids = prompt_ids + output_ids

        # Truncate if needed
        if len(full_ids) > self.max_length:
            # Keep prompt, truncate output
            max_output = self.max_length - len(prompt_ids)
            if max_output <= 0:
                # Truncate prompt too
                full_ids = full_ids[:self.max_length]
                candidate_ids = []
                candidate_ref_logprobs = []
                candidate_q_values = [] if candidate_q_values else None
                output_ids = []
            else:
                output_ids = output_ids[:max_output]
                full_ids = prompt_ids + output_ids
                candidate_ids = candidate_ids[:max_output]
                candidate_ref_logprobs = candidate_ref_logprobs[:max_output]
                if candidate_q_values:
                    candidate_q_values = candidate_q_values[:max_output]

        # Label positions: positions in full_ids that correspond to output tokens
        # Position i predicts token at position i+1
        # So for output token at position len(prompt_ids) + j, we supervise position len(prompt_ids) + j - 1
        prompt_len = len(prompt_ids)
        label_positions = []
        taken_token_ids = []  # The actual tokens taken by the original policy

        for j in range(len(output_ids)):
            pos = prompt_len + j - 1  # Position that predicts output_ids[j]
            if pos >= 0:
                label_positions.append(pos)
                taken_token_ids.append(output_ids[j])

        # Adjust candidate arrays to match label_positions
        if prompt_len <= 0:
            # Skip first output token (no position to predict it from)
            candidate_ids = candidate_ids[1:] if candidate_ids else []
            candidate_ref_logprobs = candidate_ref_logprobs[1:] if candidate_ref_logprobs else []
            if candidate_q_values:
                candidate_q_values = candidate_q_values[1:]

        input_tensor = torch.tensor(full_ids, dtype=torch.long)

        return {
            "input_ids": input_tensor,
            "label_positions": label_positions,
            "candidate_ids": candidate_ids,
            "candidate_ref_logprobs": candidate_ref_logprobs,
            "candidate_q_values": candidate_q_values,
            "taken_token_ids": taken_token_ids,
            "advantage": advantage,
            "reward": reward,
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(s["input_ids"].size(0) for s in batch)
        pad_val = 0

        return {
            "input_ids": torch.stack([
                F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0)), value=pad_val)
                for s in batch
            ]),
            "label_positions": [s["label_positions"] for s in batch],
            "candidate_ids": [s["candidate_ids"] for s in batch],
            "candidate_ref_logprobs": [s["candidate_ref_logprobs"] for s in batch],
            "candidate_q_values": [s["candidate_q_values"] for s in batch],
            "taken_token_ids": [s["taken_token_ids"] for s in batch],
            "advantages": torch.tensor([s["advantage"] for s in batch], dtype=torch.float32),
            "rewards": torch.tensor([s["reward"] for s in batch], dtype=torch.float32),
        }


def compute_grpo_loss(
    model,
    batch,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute GRPO loss with PPO-style clipping.

    L = -min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) + kl_coef * KL(π_θ || π_ref)

    Where:
    - r(θ) = π_θ(a|s) / π_ref(a|s) is the importance ratio
    - A is the group-relative advantage (same for all tokens in a trajectory)
    - KL is optional regularization
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    advantages = batch["advantages"].to(device)  # [B]

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    total_policy_loss = 0.0
    total_kl = 0.0
    total_positions = 0
    total_clipped = 0
    total_ratio_sum = 0.0
    # Common metrics
    total_entropy_student = 0.0
    total_kl_student_ref = 0.0
    total_advantage_student = 0.0

    for batch_idx, (positions, cand_ids_list, cand_ref_logprobs_list, cand_q_values_list, taken_ids) in enumerate(
        zip(batch["label_positions"], batch["candidate_ids"], batch["candidate_ref_logprobs"],
            batch["candidate_q_values"], batch["taken_token_ids"])
    ):
        if positions is None or len(positions) == 0:
            continue
        if cand_ids_list is None or len(cand_ids_list) == 0:
            continue

        advantage = advantages[batch_idx]

        for pos_idx, (pos, cand_ids, cand_ref_lps, taken_id) in enumerate(
            zip(positions, cand_ids_list, cand_ref_logprobs_list, taken_ids)
        ):
            if cand_ids is None or len(cand_ids) < 1:
                continue

            # Convert to lists if numpy arrays
            cand_ids_list_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
            cand_ref_lps_list = list(cand_ref_lps) if hasattr(cand_ref_lps, 'tolist') else list(cand_ref_lps)

            # Get Q-values for this position if available
            cand_qs = None
            if cand_q_values_list is not None and pos_idx < len(cand_q_values_list):
                cand_qs = cand_q_values_list[pos_idx]
                if cand_qs is not None:
                    cand_qs = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)

            # Find the taken action in candidates
            try:
                taken_idx = cand_ids_list_local.index(taken_id)
            except ValueError:
                # Taken action not in candidates, skip this position
                continue

            # Get logits for this position
            pos_logits = logits[batch_idx, pos, :]  # [V]

            # Candidate tensors
            cand_ids_tensor = torch.tensor(cand_ids_list_local, dtype=torch.long, device=device)
            cand_ref_logprobs_tensor = torch.tensor(cand_ref_lps_list, dtype=torch.float32, device=device)

            # π_θ(a | candidates) - student policy over candidates
            cand_logits = pos_logits[cand_ids_tensor].float()  # [num_candidates]
            log_p_student_cands = F.log_softmax(cand_logits, dim=-1)  # [num_candidates]
            p_student_cands = log_p_student_cands.exp()

            # π_ref(a | candidates) - reference policy over candidates
            log_p_ref_cands = F.log_softmax(cand_ref_logprobs_tensor, dim=-1)  # [num_candidates]
            p_ref_cands = log_p_ref_cands.exp()

            # Log probabilities for the taken action
            log_p_student_taken = log_p_student_cands[taken_idx]
            log_p_ref_taken = log_p_ref_cands[taken_idx]

            # Importance ratio: r(θ) = π_θ(a) / π_ref(a)
            log_ratio = log_p_student_taken - log_p_ref_taken
            ratio = torch.exp(log_ratio)

            # PPO clipped objective
            # L = -min(r * A, clip(r, 1-ε, 1+ε) * A)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            if advantage >= 0:
                # Positive advantage: want to increase probability
                surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)
            else:
                # Negative advantage: want to decrease probability
                surrogate = torch.max(ratio * advantage, clipped_ratio * advantage)

            policy_loss = -surrogate
            total_policy_loss += policy_loss

            # Track if clipped
            if ratio < (1.0 - clip_eps) or ratio > (1.0 + clip_eps):
                total_clipped += 1
            total_ratio_sum += ratio.detach()

            # Optional KL regularization: KL(π_θ || π_ref) over candidates
            if kl_coef > 0:
                kl = (p_student_cands * (log_p_student_cands - log_p_ref_cands)).sum()
                total_kl += kl

            # --- Common metrics (over candidates) ---
            # Entropy of student over candidates
            entropy_student = -(p_student_cands * log_p_student_cands).sum()
            total_entropy_student += entropy_student.detach()

            # KL(student || ref) over candidates
            kl_student_ref = (p_student_cands * (log_p_student_cands - log_p_ref_cands)).sum()
            total_kl_student_ref += kl_student_ref.detach()

            # Advantage: V(s) = E_ref[Q], A = Q - V (if Q-values available)
            if cand_qs is not None and len(cand_qs) == len(cand_ids_list_local):
                cand_qs_tensor = torch.tensor(cand_qs, dtype=torch.float32, device=device)
                v_state = (p_ref_cands * cand_qs_tensor).sum()
                advantages_cands = cand_qs_tensor - v_state
                advantage_student = (p_student_cands * advantages_cands).sum()
                total_advantage_student += advantage_student.detach()

            total_positions += 1

    if total_positions == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "num_positions": 0,
            "clip_fraction": 0.0,
            "mean_ratio": 0.0,
            "kl": zero.detach(),
            "mean_reward": batch["rewards"].mean().item(),
            "mean_advantage": batch["advantages"].mean().item(),
            "entropy_student": 0.0,
            "kl_student_ref": 0.0,
            "advantage_student": 0.0,
        }

    avg_policy_loss = total_policy_loss / total_positions
    avg_kl = total_kl / total_positions if kl_coef > 0 else torch.tensor(0.0, device=device)

    total_loss = avg_policy_loss + kl_coef * avg_kl

    metrics = {
        "num_positions": total_positions,
        "clip_fraction": total_clipped / total_positions,
        "mean_ratio": (total_ratio_sum / total_positions).item(),
        "kl": avg_kl.detach() if isinstance(avg_kl, torch.Tensor) else avg_kl,
        "mean_reward": batch["rewards"].mean().item(),
        "mean_advantage": batch["advantages"].mean().item(),
        # Common metrics
        "entropy_student": (total_entropy_student / total_positions).item(),
        "kl_student_ref": (total_kl_student_ref / total_positions).item(),
        "advantage_student": (total_advantage_student / total_positions).item(),
    }

    return total_loss, metrics


def print_trainable(model):
    print("\n=== Trainable parameters ===")
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
    print(f"TOTAL trainable elements: {total}\n")


def train(
    model,
    dataset,
    weights_path,
    collate_fn,
    shuffle_data,
    seed,
    dtype,
    compile_mode,
    num_epochs,
    batch_size,
    gradient_accumulation_steps,
    learning_rate,
    min_learning_rate,
    warmup_ratio,
    weight_decay,
    beta_1,
    beta_2,
    grad_clip,
    wandb_project,
    clip_eps,
    kl_coef,
    max_steps=-1,
    log_freq=10,
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

    if master and wandb_project and wandb:
        wandb.init(
            project=wandb_project,
            name=f"policy_grpo_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "clip_eps": clip_eps,
                "kl_coef": kl_coef,
                "max_steps": max_steps,
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
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(beta_1, beta_2),
        weight_decay=weight_decay,
    )
    scaler = torch.GradScaler(enabled=(dtype == "float16"))

    total_iters = (num_epochs * len(loader)) // gradient_accumulation_steps
    warmup_iters = int(warmup_ratio * total_iters)

    def lr_schedule(i):
        # Start at 0, linearly increase to learning_rate over warmup_iters
        if i <= warmup_iters:
            return learning_rate * (i - 1) / max(1, warmup_iters)
        progress = (i - warmup_iters) / max(1, total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    global_step = 0
    accum_loss = 0.0
    accum_clip_frac = 0.0
    accum_ratio = 0.0
    accum_kl = 0.0
    accum_reward = 0.0
    accum_advantage = 0.0
    accum_count = 0
    # Common metrics accumulators
    accum_entropy_student = 0.0
    accum_kl_student_ref = 0.0
    accum_advantage_student = 0.0

    for epoch in range(num_epochs):
        if distributed:
            sampler.set_epoch(epoch)

        for it, batch in enumerate(loader):
            update = (it + 1) % gradient_accumulation_steps == 0
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = update

            with torch.autocast(
                "cuda" if torch.cuda.is_available() else "cpu",
                dtype=getattr(torch, dtype),
            ):
                loss, metrics = compute_grpo_loss(model, batch, clip_eps=clip_eps, kl_coef=kl_coef)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_clip_frac += metrics["clip_fraction"]
            accum_ratio += metrics["mean_ratio"]
            accum_kl += metrics["kl"].item() if isinstance(metrics["kl"], torch.Tensor) else metrics["kl"]
            accum_reward += metrics["mean_reward"]
            accum_advantage += metrics["mean_advantage"]
            accum_count += 1
            # Common metrics
            accum_entropy_student += metrics["entropy_student"]
            accum_kl_student_ref += metrics["kl_student_ref"]
            accum_advantage_student += metrics["advantage_student"]

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

            if master and global_step % log_freq == 0:
                avg_loss = accum_loss / max(1, accum_count)
                avg_clip = accum_clip_frac / max(1, accum_count)
                avg_ratio = accum_ratio / max(1, accum_count)
                avg_kl = accum_kl / max(1, accum_count)
                avg_reward = accum_reward / max(1, accum_count)
                avg_adv = accum_advantage / max(1, accum_count)
                # Common metrics
                avg_entropy_student = accum_entropy_student / max(1, accum_count)
                avg_kl_student_ref = accum_kl_student_ref / max(1, accum_count)
                avg_advantage_student = accum_advantage_student / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} "
                    f"clip={avg_clip:.3f} ratio={avg_ratio:.3f} "
                    f"kl={avg_kl:.4f} reward={avg_reward:.3f} adv={avg_adv:.3f} lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/clip_fraction": avg_clip,
                        "train/mean_ratio": avg_ratio,
                        "train/kl": avg_kl,
                        "train/mean_reward": avg_reward,
                        "train/mean_advantage": avg_adv,
                        # Common metrics
                        "train/entropy_student": avg_entropy_student,
                        "train/kl_student_ref": avg_kl_student_ref,
                        "train/advantage_student": avg_advantage_student,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_clip_frac = 0.0
                accum_ratio = 0.0
                accum_kl = 0.0
                accum_reward = 0.0
                accum_advantage = 0.0
                accum_count = 0
                # Reset common metrics accumulators
                accum_entropy_student = 0.0
                accum_kl_student_ref = 0.0
                accum_advantage_student = 0.0

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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Disable Qwen3 thinking mode to ensure consistent behavior with reference
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
        model.generation_config.enable_thinking = False
        print("[INFO] Disabled Qwen3 thinking mode")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = PolicyGRPODataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        examples_per_prompt=cfg.examples_per_prompt,
    )

    if local_rank == 0:
        print(f"[INFO] GRPO with clip_eps={cfg.clip_eps}, kl_coef={cfg.kl_coef}")

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        PolicyGRPODataset.collate_fn,
        shuffle_data=True,
        seed=cfg.seed,
        dtype="bfloat16",
        compile_mode=cfg.compile_mode,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        beta_1=0.9,
        beta_2=0.95,
        grad_clip=cfg.grad_clip,
        wandb_project=cfg.wandb_project,
        clip_eps=cfg.clip_eps,
        kl_coef=cfg.kl_coef,
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with Group Relative Policy Optimization (GRPO)")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_grpo")
    p.add_argument("--data_path", required=True, nargs='+',
                   help="Path(s) to labeled parquet file(s). Multiple files will be concatenated and shuffled.")

    # GRPO parameters
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="PPO clipping epsilon (default: 0.2)")
    p.add_argument("--kl_coef", type=float, default=0.0,
                   help="KL regularization coefficient (default: 0.0, disabled)")

    # Dataset
    p.add_argument("--max_length", type=int, default=131072)
    p.add_argument("--examples_per_prompt", type=int, default=1)

    # Training parameters
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--min_learning_rate", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # Model config
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--compile_mode", default="none",
                   choices=["none", "default", "reduce-overhead", "max-autotune"])

    # Logging
    p.add_argument("--wandb_project", default="policy_grpo")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
