#!/usr/bin/env python3
"""
Train a student policy with Advantage-Weighted Regression (AWR).

AWR performs weighted maximum likelihood where each action is weighted by
exp(A/β), with A being the advantage and β the temperature.

This implementation supports two advantage computation methods:
1. TD-style (default): A_t = Q_t - Q_{t-1}
   - Uses the insight that in token-level MDPs with sparse terminal rewards,
     Q(s_{t-1}, a_{t-1}) = V(s_t), so advantage = Q_t - Q_{t-1}

2. MC-style (paper): A_t = R - V(s_t)
   - R is the trajectory's final reward (from 'correct' column)
   - V(s_t) = E_{π_ref}[Q(s_t, a)] estimated over candidates

Reference: "Advantage-Weighted Regression: Simple and Scalable Off-Policy RL"
           Peng et al., 2019

Input data must have columns:
- prompt_token_ids: List[int] - tokenized prompt
- output_token_ids: List[int] - the actual trajectory tokens
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate
- candidate_ref_logprobs: List[List[float]] - reference model log probs (for MC mode)
- correct (optional): float - trajectory reward (required for MC mode)

Example usage:
    python train_policy_awr.py --model_id Qwen/Qwen3-1.7B \
                               --weights_path models/policy_awr \
                               --data_path data/labeled.parquet \
                               --advantage_type td
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


class PolicyAWRDataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
        advantage_type: str = "td",
    ):
        """
        Dataset for AWR training.

        Args:
            advantage_type: "td" for A_t = Q_t - Q_{t-1}
                           "mc" for A_t = R - V(s_t) (paper style)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.advantage_type = advantage_type

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
        required_cols = ["prompt_token_ids", "output_token_ids", "candidate_ids", "candidate_q_values"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        if advantage_type == "mc":
            if "candidate_ref_logprobs" not in df.columns:
                raise ValueError("MC advantage requires 'candidate_ref_logprobs' column")
            if "correct" not in df.columns:
                raise ValueError("MC advantage requires 'correct' column for trajectory reward")

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

        if hasattr(candidate_ids, 'tolist'):
            candidate_ids = candidate_ids.tolist()
        if hasattr(candidate_q_values, 'tolist'):
            candidate_q_values = candidate_q_values.tolist()

        result = {
            "prompt_ids": list(prompt_ids),
            "output_ids": list(output_ids),
            "candidate_ids": candidate_ids,
            "candidate_q_values": candidate_q_values,
        }

        # For MC mode, also need ref logprobs and reward
        if self.advantage_type == "mc":
            candidate_ref_logprobs = row.get("candidate_ref_logprobs")
            if hasattr(candidate_ref_logprobs, 'tolist'):
                candidate_ref_logprobs = candidate_ref_logprobs.tolist()
            result["candidate_ref_logprobs"] = candidate_ref_logprobs
            result["reward"] = float(row.get("correct", 0.0))

        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_ids = sample["prompt_ids"]
        output_ids = sample["output_ids"]
        candidate_ids = sample["candidate_ids"]
        candidate_q_values = sample["candidate_q_values"]

        # Full sequence: prompt + output
        full_ids = prompt_ids + output_ids

        # Truncate if needed
        if len(full_ids) > self.max_length:
            max_output = self.max_length - len(prompt_ids)
            if max_output <= 0:
                full_ids = full_ids[:self.max_length]
                candidate_ids = []
                candidate_q_values = []
                output_ids = []
            else:
                output_ids = output_ids[:max_output]
                full_ids = prompt_ids + output_ids
                candidate_ids = candidate_ids[:max_output]
                candidate_q_values = candidate_q_values[:max_output]

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

        input_tensor = torch.tensor(full_ids, dtype=torch.long)

        result = {
            "input_ids": input_tensor,
            "label_positions": label_positions,
            "candidate_ids": candidate_ids,
            "candidate_q_values": candidate_q_values,
            "taken_token_ids": taken_token_ids,
        }

        if self.advantage_type == "mc":
            candidate_ref_logprobs = sample.get("candidate_ref_logprobs", [])
            if prompt_len <= 0 and candidate_ref_logprobs:
                candidate_ref_logprobs = candidate_ref_logprobs[1:]
            result["candidate_ref_logprobs"] = candidate_ref_logprobs
            result["reward"] = sample.get("reward", 0.0)

        return result

    @staticmethod
    def collate_fn(batch):
        max_len = max(s["input_ids"].size(0) for s in batch)
        pad_val = 0

        result = {
            "input_ids": torch.stack([
                F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0)), value=pad_val)
                for s in batch
            ]),
            "label_positions": [s["label_positions"] for s in batch],
            "candidate_ids": [s["candidate_ids"] for s in batch],
            "candidate_q_values": [s["candidate_q_values"] for s in batch],
            "taken_token_ids": [s["taken_token_ids"] for s in batch],
        }

        # Optional fields for MC mode
        if "candidate_ref_logprobs" in batch[0]:
            result["candidate_ref_logprobs"] = [s["candidate_ref_logprobs"] for s in batch]
            result["rewards"] = torch.tensor([s["reward"] for s in batch], dtype=torch.float32)

        return result


def compute_awr_loss(
    model,
    batch,
    temperature: float = 1.0,
    weight_clip: float = 20.0,
    advantage_type: str = "td",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute AWR loss: weighted negative log-likelihood.

    L = -E[log π(a|s) * min(exp(A/β), weight_clip)]

    Args:
        temperature: β in exp(A/β), controls weight sharpness
        weight_clip: Maximum weight value (for stability)
        advantage_type: "td" or "mc"
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    total_loss = 0.0
    total_positions = 0
    total_advantage = 0.0
    total_weight = 0.0
    total_clipped = 0

    for batch_idx, (positions, cand_ids_list, cand_q_list, taken_ids) in enumerate(
        zip(batch["label_positions"], batch["candidate_ids"],
            batch["candidate_q_values"], batch["taken_token_ids"])
    ):
        if positions is None or len(positions) == 0:
            continue
        if cand_ids_list is None or len(cand_ids_list) == 0:
            continue

        # For MC mode, get reward and ref logprobs
        if advantage_type == "mc":
            reward = batch["rewards"][batch_idx].item()
            cand_ref_logprobs_list = batch["candidate_ref_logprobs"][batch_idx]

        # Extract Q-values for taken actions along the trajectory
        taken_q_values = []
        valid_positions = []
        valid_taken_ids = []
        value_estimates = []  # V(s_t) for MC mode

        for pos_idx, (pos, cand_ids, cand_qs, taken_id) in enumerate(
            zip(positions, cand_ids_list, cand_q_list, taken_ids)
        ):
            if cand_ids is None or len(cand_ids) < 1:
                continue

            cand_ids_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
            cand_qs_local = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)

            # Find taken action in candidates
            try:
                taken_idx = cand_ids_local.index(taken_id)
                taken_q = cand_qs_local[taken_idx]
            except ValueError:
                continue

            taken_q_values.append(taken_q)
            valid_positions.append(pos)
            valid_taken_ids.append(taken_id)

            # For MC mode, compute V(s_t) = E_{π_ref}[Q(s_t, a)]
            if advantage_type == "mc":
                cand_ref_lps = cand_ref_logprobs_list[pos_idx]
                cand_ref_lps_local = list(cand_ref_lps) if hasattr(cand_ref_lps, 'tolist') else list(cand_ref_lps)

                # Normalize ref logprobs over candidates to get π_ref(a|candidates)
                cand_ref_tensor = torch.tensor(cand_ref_lps_local, dtype=torch.float32, device=device)
                p_ref = F.softmax(cand_ref_tensor, dim=-1)

                cand_qs_tensor = torch.tensor(cand_qs_local, dtype=torch.float32, device=device)
                v_state = (p_ref * cand_qs_tensor).sum().item()
                value_estimates.append(v_state)

        if len(taken_q_values) == 0:
            continue

        # Compute advantages
        advantages = []

        if advantage_type == "td":
            # TD-style: A_t = Q_t - Q_{t-1}
            for i in range(len(taken_q_values)):
                if i == 0:
                    # First position: use Q_0 as advantage (assume V(s_0) = 0)
                    adv = taken_q_values[i]
                else:
                    adv = taken_q_values[i] - taken_q_values[i - 1]
                advantages.append(adv)
        else:
            # MC-style (paper): A_t = R - V(s_t)
            for i in range(len(taken_q_values)):
                adv = reward - value_estimates[i]
                advantages.append(adv)

        # Compute weighted loss for each position
        for i, (pos, taken_id, adv) in enumerate(zip(valid_positions, valid_taken_ids, advantages)):
            # Get logits for this position
            pos_logits = logits[batch_idx, pos, :]  # [V]

            # Log probability of taken action (full vocabulary)
            log_probs = F.log_softmax(pos_logits.float(), dim=-1)
            log_p_taken = log_probs[taken_id]

            # Compute weight: exp(A/β) with clipping
            weight = torch.exp(torch.tensor(adv / temperature, device=device))
            weight_clipped = torch.clamp(weight, max=weight_clip)

            # Track clipping
            if weight > weight_clip:
                total_clipped += 1

            # AWR loss: -log π(a|s) * weight
            loss = -log_p_taken * weight_clipped
            total_loss += loss

            total_positions += 1
            total_advantage += adv
            total_weight += weight_clipped.item()

    if total_positions == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "num_positions": 0,
            "mean_advantage": 0.0,
            "mean_weight": 0.0,
            "clip_fraction": 0.0,
        }

    avg_loss = total_loss / total_positions

    metrics = {
        "num_positions": total_positions,
        "mean_advantage": total_advantage / total_positions,
        "mean_weight": total_weight / total_positions,
        "clip_fraction": total_clipped / total_positions,
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
    temperature,
    weight_clip,
    advantage_type,
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
            name=f"policy_awr_{advantage_type}_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "temperature": temperature,
                "weight_clip": weight_clip,
                "advantage_type": advantage_type,
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
    accum_weight = 0.0
    accum_clip_frac = 0.0
    accum_count = 0

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
                loss, metrics = compute_awr_loss(
                    model, batch,
                    temperature=temperature,
                    weight_clip=weight_clip,
                    advantage_type=advantage_type,
                )
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_advantage += metrics["mean_advantage"]
            accum_weight += metrics["mean_weight"]
            accum_clip_frac += metrics["clip_fraction"]
            accum_count += 1

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
                avg_weight = accum_weight / max(1, accum_count)
                avg_clip = accum_clip_frac / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} "
                    f"adv={avg_adv:.4f} weight={avg_weight:.3f} "
                    f"clip={avg_clip:.3f} lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/mean_advantage": avg_adv,
                        "train/mean_weight": avg_weight,
                        "train/clip_fraction": avg_clip,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_advantage = 0.0
                accum_weight = 0.0
                accum_clip_frac = 0.0
                accum_count = 0

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

    dataset = PolicyAWRDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        examples_per_prompt=cfg.examples_per_prompt,
        advantage_type=cfg.advantage_type,
    )

    if local_rank == 0:
        adv_desc = "A_t = Q_t - Q_{t-1}" if cfg.advantage_type == "td" else "A_t = R - V(s_t)"
        print(f"[INFO] AWR with {cfg.advantage_type} advantage: {adv_desc}")
        print(f"[INFO] Temperature β={cfg.temperature}, weight_clip={cfg.weight_clip}")

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        PolicyAWRDataset.collate_fn,
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
        temperature=cfg.temperature,
        weight_clip=cfg.weight_clip,
        advantage_type=cfg.advantage_type,
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with Advantage-Weighted Regression (AWR)")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_awr")
    p.add_argument("--data_path", required=True, nargs='+',
                   help="Path(s) to labeled parquet file(s)")

    # AWR parameters
    p.add_argument("--advantage_type", type=str, default="td", choices=["td", "mc"],
                   help="Advantage computation: 'td' for A=Q_t-Q_{t-1}, 'mc' for A=R-V(s) (paper)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature β for exp(A/β) weights")
    p.add_argument("--weight_clip", type=float, default=20.0,
                   help="Maximum weight value for stability (default: 20.0)")

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
    p.add_argument("--wandb_project", default="policy_awr")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
