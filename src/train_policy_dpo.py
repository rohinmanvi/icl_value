#!/usr/bin/env python3
"""
Train a student policy with Direct Preference Optimization (DPO).

DPO loss: L = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))

Where:
- y_w = preferred (winning) response
- y_l = rejected (losing) response
- π_θ = policy being trained
- π_ref = reference policy (frozen, from pre-computed logprobs)
- β = temperature parameter

Preference pairs are constructed by grouping trajectories by prompt and selecting
best vs worst based on `correct` column (if available) or final Q-values.

Reference logprobs are extracted from pre-computed candidate data, avoiding the need
to load a frozen reference model.

Input data must have columns:
- prompt_idx: int - prompt identifier for grouping
- prompt_token_ids: List[int] - tokenized prompt
- output_token_ids: List[int] - the actual trajectory tokens
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_ref_logprobs: List[List[float]] - reference model log probs for each candidate
- correct (optional): float - correctness label (preferred if higher)
- candidate_q_values (fallback): List[List[float]] - Q-values for determining preference

Example usage:
    python train_policy_dpo.py --model_id Qwen/Qwen3-1.7B \
                               --weights_path models/policy_dpo \
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


class PolicyDPODataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
    ):
        """
        Dataset for DPO training.

        Groups trajectories by prompt_idx and creates (preferred, rejected) pairs
        using best vs worst based on `correct` column or final Q-values.

        Each sample contains:
        - prompt_token_ids: tokenized prompt
        - preferred_output_ids: tokenized preferred response
        - rejected_output_ids: tokenized rejected response
        - preferred_ref_logprobs: reference log probs for preferred trajectory
        - rejected_ref_logprobs: reference log probs for rejected trajectory
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
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
        required_cols = ["prompt_idx", "prompt_token_ids", "output_token_ids",
                         "candidate_ids", "candidate_ref_logprobs"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        # Filter out rows without labels
        df = df[df["candidate_ids"].notna()].copy()

        # Determine preference criterion
        use_correct = "correct" in df.columns
        if use_correct:
            print("Using 'correct' column for preference ordering")
            df["preference_score"] = df["correct"].apply(
                lambda x: float(x) if x is not None else None
            )
        else:
            print("Using final Q-values for preference ordering")
            df["preference_score"] = df["candidate_q_values"].apply(self._get_final_q)

        df = df[df["preference_score"].notna()].copy()
        print(f"Loaded {len(df)} labeled samples")

        # Group by prompt and create pairs
        self.samples = []
        grouped = df.groupby("prompt_idx")

        num_prompts = 0
        num_pairs = 0

        for prompt_idx, group in grouped:
            if len(group) < 2:
                continue

            # Sort by preference score
            sorted_group = group.sort_values("preference_score", ascending=False)
            rows = list(sorted_group.iterrows())

            # Best vs worst pair
            best_idx, best_row = rows[0]
            worst_idx, worst_row = rows[-1]

            # Skip if same score (no clear preference)
            if best_row["preference_score"] == worst_row["preference_score"]:
                continue

            pair = self._create_pair(best_row, worst_row)
            if pair is not None:
                self.samples.append(pair)
                num_pairs += 1

            num_prompts += 1

        print(f"Created {num_pairs} preference pairs from {num_prompts} prompts")

    def _get_final_q(self, q_values) -> Optional[float]:
        """Extract final Q-value from trajectory."""
        if q_values is None or len(q_values) == 0:
            return None
        last_position = q_values[-1]
        if last_position is None or len(last_position) == 0:
            return None
        if hasattr(last_position, 'tolist'):
            last_position = last_position.tolist()
        return max(last_position)

    def _extract_ref_logprobs(self, output_ids, candidate_ids, candidate_ref_logprobs) -> Optional[List[float]]:
        """
        Extract reference logprobs for the taken tokens.

        For each position, find the taken token in candidate_ids and get its ref logprob.
        """
        if len(output_ids) == 0:
            return []

        ref_logprobs = []

        for pos_idx, token_id in enumerate(output_ids):
            if pos_idx >= len(candidate_ids):
                # No candidate data for this position
                break

            cands = candidate_ids[pos_idx]
            ref_lps = candidate_ref_logprobs[pos_idx]

            if cands is None or ref_lps is None:
                return None

            # Convert to lists if needed
            if hasattr(cands, 'tolist'):
                cands = cands.tolist()
            if hasattr(ref_lps, 'tolist'):
                ref_lps = ref_lps.tolist()

            # Find the taken token in candidates
            try:
                idx = cands.index(token_id)
                ref_logprobs.append(ref_lps[idx])
            except ValueError:
                # Token not found in candidates - this shouldn't happen
                # but we'll handle it gracefully
                return None

        return ref_logprobs

    def _create_pair(self, preferred_row, rejected_row) -> Optional[Dict]:
        """Create a preference pair from two rows."""
        # Process preferred
        pref_prompt_ids = preferred_row["prompt_token_ids"]
        if hasattr(pref_prompt_ids, 'tolist'):
            pref_prompt_ids = pref_prompt_ids.tolist()

        pref_output_ids = preferred_row["output_token_ids"]
        if hasattr(pref_output_ids, 'tolist'):
            pref_output_ids = pref_output_ids.tolist()

        pref_cand_ids = preferred_row["candidate_ids"]
        pref_cand_ref_lps = preferred_row["candidate_ref_logprobs"]
        if hasattr(pref_cand_ids, 'tolist'):
            pref_cand_ids = pref_cand_ids.tolist()
        if hasattr(pref_cand_ref_lps, 'tolist'):
            pref_cand_ref_lps = pref_cand_ref_lps.tolist()

        pref_ref_logprobs = self._extract_ref_logprobs(
            pref_output_ids, pref_cand_ids, pref_cand_ref_lps
        )
        if pref_ref_logprobs is None:
            return None

        # Process rejected
        rej_prompt_ids = rejected_row["prompt_token_ids"]
        if hasattr(rej_prompt_ids, 'tolist'):
            rej_prompt_ids = rej_prompt_ids.tolist()

        rej_output_ids = rejected_row["output_token_ids"]
        if hasattr(rej_output_ids, 'tolist'):
            rej_output_ids = rej_output_ids.tolist()

        rej_cand_ids = rejected_row["candidate_ids"]
        rej_cand_ref_lps = rejected_row["candidate_ref_logprobs"]
        if hasattr(rej_cand_ids, 'tolist'):
            rej_cand_ids = rej_cand_ids.tolist()
        if hasattr(rej_cand_ref_lps, 'tolist'):
            rej_cand_ref_lps = rej_cand_ref_lps.tolist()

        rej_ref_logprobs = self._extract_ref_logprobs(
            rej_output_ids, rej_cand_ids, rej_cand_ref_lps
        )
        if rej_ref_logprobs is None:
            return None

        # Verify prompts match
        if list(pref_prompt_ids) != list(rej_prompt_ids):
            print(f"Warning: prompt mismatch in pair, skipping")
            return None

        return {
            "prompt_ids": list(pref_prompt_ids),
            "preferred_output_ids": list(pref_output_ids),
            "rejected_output_ids": list(rej_output_ids),
            "preferred_ref_logprobs": pref_ref_logprobs,
            "rejected_ref_logprobs": rej_ref_logprobs,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_ids = sample["prompt_ids"]
        pref_output_ids = sample["preferred_output_ids"]
        rej_output_ids = sample["rejected_output_ids"]
        pref_ref_logprobs = sample["preferred_ref_logprobs"]
        rej_ref_logprobs = sample["rejected_ref_logprobs"]

        # Full sequences
        pref_full_ids = prompt_ids + pref_output_ids
        rej_full_ids = prompt_ids + rej_output_ids

        prompt_len = len(prompt_ids)

        # Truncate if needed
        if len(pref_full_ids) > self.max_length:
            max_output = self.max_length - prompt_len
            if max_output <= 0:
                pref_full_ids = pref_full_ids[:self.max_length]
                pref_output_ids = []
                pref_ref_logprobs = []
            else:
                pref_output_ids = pref_output_ids[:max_output]
                pref_full_ids = prompt_ids + pref_output_ids
                pref_ref_logprobs = pref_ref_logprobs[:max_output]

        if len(rej_full_ids) > self.max_length:
            max_output = self.max_length - prompt_len
            if max_output <= 0:
                rej_full_ids = rej_full_ids[:self.max_length]
                rej_output_ids = []
                rej_ref_logprobs = []
            else:
                rej_output_ids = rej_output_ids[:max_output]
                rej_full_ids = prompt_ids + rej_output_ids
                rej_ref_logprobs = rej_ref_logprobs[:max_output]

        return {
            "preferred_input_ids": torch.tensor(pref_full_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rej_full_ids, dtype=torch.long),
            "prompt_len": prompt_len,
            "preferred_ref_logprobs": torch.tensor(pref_ref_logprobs, dtype=torch.float32),
            "rejected_ref_logprobs": torch.tensor(rej_ref_logprobs, dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch):
        # Pad preferred and rejected sequences separately
        max_pref_len = max(s["preferred_input_ids"].size(0) for s in batch)
        max_rej_len = max(s["rejected_input_ids"].size(0) for s in batch)
        pad_val = 0

        preferred_input_ids = torch.stack([
            F.pad(s["preferred_input_ids"], (0, max_pref_len - s["preferred_input_ids"].size(0)), value=pad_val)
            for s in batch
        ])

        rejected_input_ids = torch.stack([
            F.pad(s["rejected_input_ids"], (0, max_rej_len - s["rejected_input_ids"].size(0)), value=pad_val)
            for s in batch
        ])

        return {
            "preferred_input_ids": preferred_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "prompt_lens": [s["prompt_len"] for s in batch],
            "preferred_ref_logprobs": [s["preferred_ref_logprobs"] for s in batch],
            "rejected_ref_logprobs": [s["rejected_ref_logprobs"] for s in batch],
        }


def compute_sequence_logprobs(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute log probabilities for output tokens given a sequence.

    Returns tensor of log probs for each output token.
    """
    outputs = model(input_ids=input_ids.unsqueeze(0), use_cache=False)
    logits = outputs.logits[0]  # [S, V]

    # Get log probs for next token prediction
    log_probs = F.log_softmax(logits, dim=-1)  # [S, V]

    # Extract log probs for the actual tokens
    # Position i predicts token at position i+1
    # Output tokens start at prompt_len
    output_logprobs = []
    for i in range(prompt_len, input_ids.size(0)):
        if i > 0:
            token_id = input_ids[i].item()
            output_logprobs.append(log_probs[i - 1, token_id])

    if len(output_logprobs) == 0:
        return torch.tensor(0.0, device=input_ids.device)

    return torch.stack(output_logprobs)


def compute_dpo_loss(
    model,
    batch,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute DPO loss.

    L = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))
    """
    device = next(model.parameters()).device
    preferred_input_ids = batch["preferred_input_ids"].to(device)
    rejected_input_ids = batch["rejected_input_ids"].to(device)
    prompt_lens = batch["prompt_lens"]
    preferred_ref_logprobs = batch["preferred_ref_logprobs"]
    rejected_ref_logprobs = batch["rejected_ref_logprobs"]

    batch_size = preferred_input_ids.size(0)

    total_loss = 0.0
    total_reward_margin = 0.0
    total_preferred_reward = 0.0
    total_rejected_reward = 0.0
    total_accuracy = 0.0
    num_valid = 0

    for i in range(batch_size):
        prompt_len = prompt_lens[i]

        # Get student log probs for preferred
        pref_ids = preferred_input_ids[i]
        # Find actual length (non-padded)
        pref_len = (pref_ids != 0).sum().item()
        if pref_len <= prompt_len:
            continue
        pref_ids = pref_ids[:pref_len]
        pref_student_logprobs = compute_sequence_logprobs(model, pref_ids, prompt_len)

        # Get student log probs for rejected
        rej_ids = rejected_input_ids[i]
        rej_len = (rej_ids != 0).sum().item()
        if rej_len <= prompt_len:
            continue
        rej_ids = rej_ids[:rej_len]
        rej_student_logprobs = compute_sequence_logprobs(model, rej_ids, prompt_len)

        # Reference log probs (pre-computed)
        pref_ref_lps = preferred_ref_logprobs[i].to(device)
        rej_ref_lps = rejected_ref_logprobs[i].to(device)

        # Ensure lengths match
        min_pref_len = min(len(pref_student_logprobs), len(pref_ref_lps))
        min_rej_len = min(len(rej_student_logprobs), len(rej_ref_lps))

        if min_pref_len == 0 or min_rej_len == 0:
            continue

        pref_student_logprobs = pref_student_logprobs[:min_pref_len]
        pref_ref_lps = pref_ref_lps[:min_pref_len]
        rej_student_logprobs = rej_student_logprobs[:min_rej_len]
        rej_ref_lps = rej_ref_lps[:min_rej_len]

        # Sum log probs for full trajectories
        pref_student_sum = pref_student_logprobs.sum()
        pref_ref_sum = pref_ref_lps.sum()
        rej_student_sum = rej_student_logprobs.sum()
        rej_ref_sum = rej_ref_lps.sum()

        # DPO implicit rewards
        pref_reward = pref_student_sum - pref_ref_sum
        rej_reward = rej_student_sum - rej_ref_sum
        reward_margin = pref_reward - rej_reward

        # DPO loss
        loss = -F.logsigmoid(beta * reward_margin)
        total_loss += loss

        # Metrics
        total_reward_margin += reward_margin.detach()
        total_preferred_reward += pref_reward.detach()
        total_rejected_reward += rej_reward.detach()
        total_accuracy += (reward_margin > 0).float().detach()
        num_valid += 1

    if num_valid == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "num_pairs": 0,
            "reward_margin": 0.0,
            "preferred_reward": 0.0,
            "rejected_reward": 0.0,
            "accuracy": 0.0,
        }

    avg_loss = total_loss / num_valid

    metrics = {
        "num_pairs": num_valid,
        "reward_margin": (total_reward_margin / num_valid).item(),
        "preferred_reward": (total_preferred_reward / num_valid).item(),
        "rejected_reward": (total_rejected_reward / num_valid).item(),
        "accuracy": (total_accuracy / num_valid).item(),
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
    beta=0.1,
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
            name=f"policy_dpo_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "beta": beta,
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
    accum_reward_margin = 0.0
    accum_preferred_reward = 0.0
    accum_rejected_reward = 0.0
    accum_accuracy = 0.0
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
                loss, metrics = compute_dpo_loss(model, batch, beta=beta)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_reward_margin += metrics["reward_margin"]
            accum_preferred_reward += metrics["preferred_reward"]
            accum_rejected_reward += metrics["rejected_reward"]
            accum_accuracy += metrics["accuracy"]
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
                avg_margin = accum_reward_margin / max(1, accum_count)
                avg_pref = accum_preferred_reward / max(1, accum_count)
                avg_rej = accum_rejected_reward / max(1, accum_count)
                avg_acc = accum_accuracy / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} "
                    f"margin={avg_margin:.4f} acc={avg_acc:.3f} "
                    f"pref_r={avg_pref:.4f} rej_r={avg_rej:.4f} lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/dpo_loss": avg_loss,
                        "train/reward_margin": avg_margin,
                        "train/accuracy": avg_acc,
                        "train/preferred_reward": avg_pref,
                        "train/rejected_reward": avg_rej,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_reward_margin = 0.0
                accum_preferred_reward = 0.0
                accum_rejected_reward = 0.0
                accum_accuracy = 0.0
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

    # Disable Qwen3 thinking mode to ensure consistent behavior with reference
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
        model.generation_config.enable_thinking = False
        print("[INFO] Disabled Qwen3 thinking mode")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = PolicyDPODataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        PolicyDPODataset.collate_fn,
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
        beta=cfg.beta,
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with DPO (Direct Preference Optimization)")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_dpo")
    p.add_argument("--data_path", required=True, nargs='+',
                   help="Path(s) to labeled parquet file(s). Multiple files will be concatenated.")

    # DPO parameters
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature parameter β (default: 0.1)")

    # Dataset
    p.add_argument("--max_length", type=int, default=131072)

    # Training parameters
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
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
    p.add_argument("--wandb_project", default="policy_dpo")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
