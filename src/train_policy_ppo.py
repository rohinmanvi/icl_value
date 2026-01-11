#!/usr/bin/env python3
"""
Train a student policy with PPO using GAE (Generalized Advantage Estimation).

This implements offline PPO with:
- GAE for advantage computation with tunable λ parameter
- PPO clipped surrogate objective
- Pre-computed reference logprobs (no need for frozen reference model)

GAE interpolates between TD and MC advantages:
- λ=0: A_t = Q_t - Q_{t-1} (TD, low variance, high bias)
- λ=1: A_t = R - V(s_t) (MC, high variance, low bias)

The TD errors are: δ_t = Q_t - Q_{t-1} (since V(s_t) = Q_{t-1} in token MDPs)
GAE: A^GAE_t = Σ_{l=0}^{T-t-1} λ^l δ_{t+l}

Input data must have columns:
- prompt_token_ids: List[int] - tokenized prompt
- output_token_ids: List[int] - the actual trajectory tokens
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate
- candidate_ref_logprobs: List[List[float]] - reference model log probs for each candidate

Example usage:
    python train_policy_ppo.py --model_id Qwen/Qwen3-1.7B \
                               --weights_path models/policy_ppo \
                               --data_path data/labeled.parquet \
                               --gae_lambda 0.95
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


class PolicyPPODataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
    ):
        """
        Dataset for PPO training with GAE.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data - support single path, list of paths, or table
        if isinstance(table, str):
            df = pq.read_table(table).to_pandas()
        elif isinstance(table, list):
            dfs = []
            for path in table:
                dfs.append(pq.read_table(path).to_pandas())
                print(f"Loaded {len(dfs[-1])} samples from {path}")
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = table

        # Validate required columns
        required_cols = ["prompt_token_ids", "output_token_ids", "candidate_ids",
                        "candidate_q_values", "candidate_ref_logprobs"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        # Filter out rows without labels
        df = df[df["candidate_ids"].notna()].copy()
        df = df[df["candidate_q_values"].notna()].copy()

        print(f"Loaded {len(df)} labeled samples")

        # Group by prompt for shuffling
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
        candidate_q_values = row["candidate_q_values"]
        candidate_ref_logprobs = row["candidate_ref_logprobs"]

        if hasattr(candidate_ids, 'tolist'):
            candidate_ids = candidate_ids.tolist()
        if hasattr(candidate_q_values, 'tolist'):
            candidate_q_values = candidate_q_values.tolist()
        if hasattr(candidate_ref_logprobs, 'tolist'):
            candidate_ref_logprobs = candidate_ref_logprobs.tolist()

        return {
            "prompt_ids": list(prompt_ids),
            "output_ids": list(output_ids),
            "candidate_ids": candidate_ids,
            "candidate_q_values": candidate_q_values,
            "candidate_ref_logprobs": candidate_ref_logprobs,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_ids = sample["prompt_ids"]
        output_ids = sample["output_ids"]
        candidate_ids = sample["candidate_ids"]
        candidate_q_values = sample["candidate_q_values"]
        candidate_ref_logprobs = sample["candidate_ref_logprobs"]

        # Full sequence: prompt + output
        full_ids = prompt_ids + output_ids

        # Truncate if needed
        if len(full_ids) > self.max_length:
            max_output = self.max_length - len(prompt_ids)
            if max_output <= 0:
                full_ids = full_ids[:self.max_length]
                candidate_ids = []
                candidate_q_values = []
                candidate_ref_logprobs = []
                output_ids = []
            else:
                output_ids = output_ids[:max_output]
                full_ids = prompt_ids + output_ids
                candidate_ids = candidate_ids[:max_output]
                candidate_q_values = candidate_q_values[:max_output]
                candidate_ref_logprobs = candidate_ref_logprobs[:max_output]

        prompt_len = len(prompt_ids)
        label_positions = []
        taken_token_ids = []

        for j in range(len(output_ids)):
            pos = prompt_len + j - 1
            if pos >= 0:
                label_positions.append(pos)
                taken_token_ids.append(output_ids[j])

        # Adjust candidate arrays
        if prompt_len <= 0:
            candidate_ids = candidate_ids[1:] if candidate_ids else []
            candidate_q_values = candidate_q_values[1:] if candidate_q_values else []
            candidate_ref_logprobs = candidate_ref_logprobs[1:] if candidate_ref_logprobs else []

        input_tensor = torch.tensor(full_ids, dtype=torch.long)

        return {
            "input_ids": input_tensor,
            "label_positions": label_positions,
            "candidate_ids": candidate_ids,
            "candidate_q_values": candidate_q_values,
            "candidate_ref_logprobs": candidate_ref_logprobs,
            "taken_token_ids": taken_token_ids,
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
            "candidate_q_values": [s["candidate_q_values"] for s in batch],
            "candidate_ref_logprobs": [s["candidate_ref_logprobs"] for s in batch],
            "taken_token_ids": [s["taken_token_ids"] for s in batch],
        }


def compute_gae_advantages(
    taken_q_values: List[float],
    gae_lambda: float,
    gamma: float = 1.0,
) -> List[float]:
    """
    Compute GAE advantages from Q-values of taken actions.

    In token-level MDPs with sparse rewards:
    - V(s_t) = Q(s_{t-1}, a_{t-1}) = Q_{t-1}
    - δ_t = Q_t - Q_{t-1} (TD error)
    - A^GAE_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}

    Computed efficiently backwards:
    - A_{T-1} = δ_{T-1}
    - A_t = δ_t + γλ * A_{t+1}
    """
    T = len(taken_q_values)
    if T == 0:
        return []

    # Compute TD errors: δ_t = Q_t - Q_{t-1}
    # For t=0, assume V(s_0) = 0, so δ_0 = Q_0
    deltas = []
    for t in range(T):
        if t == 0:
            delta = taken_q_values[t]  # Q_0 - V(s_0), assuming V(s_0) = 0
        else:
            delta = taken_q_values[t] - taken_q_values[t - 1]
        deltas.append(delta)

    # Compute GAE backwards
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    return advantages


def compute_ppo_loss(
    model,
    batch,
    clip_eps: float = 0.2,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute PPO clipped surrogate loss with GAE advantages.

    L = -min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)

    where r(θ) = π_θ(a|s) / π_ref(a|s) is the importance ratio.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    total_loss = 0.0
    total_positions = 0
    total_advantage = 0.0
    total_ratio = 0.0
    total_clipped = 0
    # Common metrics
    total_entropy_student = 0.0
    total_kl_student_ref = 0.0
    total_advantage_student = 0.0

    for batch_idx, (positions, cand_ids_list, cand_q_list, cand_ref_logprobs_list, taken_ids) in enumerate(
        zip(batch["label_positions"], batch["candidate_ids"],
            batch["candidate_q_values"], batch["candidate_ref_logprobs"],
            batch["taken_token_ids"])
    ):
        if positions is None or len(positions) == 0:
            continue
        if cand_ids_list is None or len(cand_ids_list) == 0:
            continue

        # Extract Q-values for taken actions along the trajectory
        taken_q_values = []
        valid_indices = []  # Track which positions are valid

        for pos_idx, (pos, cand_ids, cand_qs, taken_id) in enumerate(
            zip(positions, cand_ids_list, cand_q_list, taken_ids)
        ):
            if cand_ids is None or len(cand_ids) < 1:
                continue

            cand_ids_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
            cand_qs_local = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)

            try:
                taken_idx = cand_ids_local.index(taken_id)
                taken_q = cand_qs_local[taken_idx]
            except ValueError:
                continue

            taken_q_values.append(taken_q)
            valid_indices.append(pos_idx)

        if len(taken_q_values) == 0:
            continue

        # Compute GAE advantages
        advantages = compute_gae_advantages(taken_q_values, gae_lambda)

        # Optionally normalize advantages
        if normalize_advantages and len(advantages) > 1:
            adv_tensor = torch.tensor(advantages)
            adv_mean = adv_tensor.mean()
            adv_std = adv_tensor.std() + 1e-8
            advantages = ((adv_tensor - adv_mean) / adv_std).tolist()

        # Compute PPO loss for each valid position
        for i, pos_idx in enumerate(valid_indices):
            pos = positions[pos_idx]
            taken_id = taken_ids[pos_idx]
            cand_ids = cand_ids_list[pos_idx]
            cand_qs = cand_q_list[pos_idx]
            cand_ref_lps = cand_ref_logprobs_list[pos_idx]
            advantage = advantages[i]

            cand_ids_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
            cand_qs_local = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)
            cand_ref_lps_local = list(cand_ref_lps) if hasattr(cand_ref_lps, 'tolist') else list(cand_ref_lps)

            try:
                taken_idx = cand_ids_local.index(taken_id)
            except ValueError:
                continue

            # Get logits for this position
            pos_logits = logits[batch_idx, pos, :]  # [V]

            # Compute log probs over candidates
            cand_ids_tensor = torch.tensor(cand_ids_local, dtype=torch.long, device=device)
            cand_qs_tensor = torch.tensor(cand_qs_local, dtype=torch.float32, device=device)
            cand_ref_logprobs_tensor = torch.tensor(cand_ref_lps_local, dtype=torch.float32, device=device)

            # Student policy: π_θ(a | candidates)
            cand_logits = pos_logits[cand_ids_tensor].float()
            log_p_student_cands = F.log_softmax(cand_logits, dim=-1)
            p_student_cands = log_p_student_cands.exp()

            # Reference policy: π_ref(a | candidates)
            log_p_ref_cands = F.log_softmax(cand_ref_logprobs_tensor, dim=-1)
            p_ref_cands = log_p_ref_cands.exp()

            # Log probs for taken action
            log_p_student_taken = log_p_student_cands[taken_idx]
            log_p_ref_taken = log_p_ref_cands[taken_idx]

            # Importance ratio: r(θ) = π_θ(a) / π_ref(a)
            log_ratio = log_p_student_taken - log_p_ref_taken
            ratio = torch.exp(log_ratio)

            # PPO clipped objective
            adv_tensor = torch.tensor(advantage, device=device, dtype=torch.float32)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            if advantage >= 0:
                surrogate = torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)
            else:
                surrogate = torch.max(ratio * adv_tensor, clipped_ratio * adv_tensor)

            loss = -surrogate
            total_loss += loss

            # Track metrics
            if ratio < (1.0 - clip_eps) or ratio > (1.0 + clip_eps):
                total_clipped += 1

            total_positions += 1
            total_advantage += advantage
            total_ratio += ratio.detach().item()

            # --- Common metrics (over candidates) ---
            # Entropy of student over candidates
            entropy_student = -(p_student_cands * log_p_student_cands).sum()
            total_entropy_student += entropy_student.detach()

            # KL(student || ref) over candidates
            kl_student_ref = (p_student_cands * (log_p_student_cands - log_p_ref_cands)).sum()
            total_kl_student_ref += kl_student_ref.detach()

            # Advantage: V(s) = E_ref[Q], A = Q - V
            v_state = (p_ref_cands * cand_qs_tensor).sum()
            advantages_cands = cand_qs_tensor - v_state
            advantage_student = (p_student_cands * advantages_cands).sum()
            total_advantage_student += advantage_student.detach()

    if total_positions == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "num_positions": 0,
            "mean_advantage": 0.0,
            "mean_ratio": 0.0,
            "clip_fraction": 0.0,
            "entropy_student": 0.0,
            "kl_student_ref": 0.0,
            "advantage_student": 0.0,
        }

    avg_loss = total_loss / total_positions

    metrics = {
        "num_positions": total_positions,
        "mean_advantage": total_advantage / total_positions,
        "mean_ratio": total_ratio / total_positions,
        "clip_fraction": total_clipped / total_positions,
        "entropy_student": (total_entropy_student / total_positions).item(),
        "kl_student_ref": (total_kl_student_ref / total_positions).item(),
        "advantage_student": (total_advantage_student / total_positions).item(),
    }

    return avg_loss, metrics


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
    gae_lambda,
    normalize_advantages,
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
            name=f"policy_ppo_gae{gae_lambda}_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "clip_eps": clip_eps,
                "gae_lambda": gae_lambda,
                "normalize_advantages": normalize_advantages,
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
        if i <= warmup_iters:
            return learning_rate * (i - 1) / max(1, warmup_iters)
        progress = (i - warmup_iters) / max(1, total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    global_step = 0
    accum_loss = 0.0
    accum_advantage = 0.0
    accum_ratio = 0.0
    accum_clip_frac = 0.0
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
                loss, metrics = compute_ppo_loss(
                    model, batch,
                    clip_eps=clip_eps,
                    gae_lambda=gae_lambda,
                    normalize_advantages=normalize_advantages,
                )
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_advantage += metrics["mean_advantage"]
            accum_ratio += metrics["mean_ratio"]
            accum_clip_frac += metrics["clip_fraction"]
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
                avg_adv = accum_advantage / max(1, accum_count)
                avg_ratio = accum_ratio / max(1, accum_count)
                avg_clip = accum_clip_frac / max(1, accum_count)
                # Common metrics
                avg_entropy_student = accum_entropy_student / max(1, accum_count)
                avg_kl_student_ref = accum_kl_student_ref / max(1, accum_count)
                avg_advantage_student = accum_advantage_student / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} "
                    f"adv={avg_adv:.4f} ratio={avg_ratio:.3f} "
                    f"clip={avg_clip:.3f} lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/mean_advantage": avg_adv,
                        "train/mean_ratio": avg_ratio,
                        "train/clip_fraction": avg_clip,
                        # Common metrics
                        "train/entropy_student": avg_entropy_student,
                        "train/kl_student_ref": avg_kl_student_ref,
                        "train/advantage_student": avg_advantage_student,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_advantage = 0.0
                accum_ratio = 0.0
                accum_clip_frac = 0.0
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

    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
        model.generation_config.enable_thinking = False
        print("[INFO] Disabled Qwen3 thinking mode")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = PolicyPPODataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        examples_per_prompt=cfg.examples_per_prompt,
    )

    if local_rank == 0:
        print(f"[INFO] PPO with GAE λ={cfg.gae_lambda}, clip_eps={cfg.clip_eps}")
        if cfg.gae_lambda == 0:
            print("[INFO] λ=0: Using TD(0) advantages (A_t = Q_t - Q_{t-1})")
        elif cfg.gae_lambda == 1:
            print("[INFO] λ=1: Using MC advantages (full return)")
        else:
            print(f"[INFO] λ={cfg.gae_lambda}: Interpolating between TD and MC")

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        PolicyPPODataset.collate_fn,
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
        gae_lambda=cfg.gae_lambda,
        normalize_advantages=cfg.normalize_advantages,
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with PPO using GAE advantages")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_ppo")
    p.add_argument("--data_path", required=True, nargs='+',
                   help="Path(s) to labeled parquet file(s)")

    # PPO parameters
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="PPO clipping epsilon (default: 0.2)")
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="GAE lambda: 0=TD(0), 1=MC, 0.95=typical (default: 0.95)")
    p.add_argument("--normalize_advantages", action="store_true", default=True,
                   help="Normalize advantages within each trajectory")
    p.add_argument("--no_normalize_advantages", action="store_false", dest="normalize_advantages",
                   help="Disable advantage normalization")

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
    p.add_argument("--wandb_project", default="policy_ppo")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
