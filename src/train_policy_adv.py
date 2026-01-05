#!/usr/bin/env python3
"""
Train a policy with advantage-weighted log likelihood loss.

Loss: L = -Σ A(a) * log π(a | s)

where:
- A(a) = Q(a) - mean(Q_candidates) for candidate tokens
- A(a) = 0 for non-candidate tokens

Positive advantages push probabilities up, negative push down.

Input data must have columns:
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate
- output_token_ids: List[int] - the actual trajectory tokens

Example usage:
    python train_policy_adv.py --model_id Qwen/Qwen3-1.7B \
                               --weights_path models/policy_adv \
                               --data_path data/labeled.parquet
"""
from __future__ import annotations
import argparse, math, os, time, wandb, random
from typing import List, Dict, Any, Tuple
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


class PolicyAdvDataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
    ):
        """
        Dataset for training policy with advantage-weighted loss.

        Each sample contains:
        - prompt_token_ids: tokenized prompt
        - output_token_ids: tokenized response
        - candidate_ids: List of candidate token IDs per position
        - candidate_q_values: List of Q-values per candidate per position
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        df = pq.read_table(table).to_pandas() if isinstance(table, str) else table

        # Validate required columns
        required_cols = ["prompt_token_ids", "output_token_ids", "candidate_ids", "candidate_q_values"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        # Filter out rows without labels (None values)
        df = df[df["candidate_ids"].notna()].copy()

        print(f"Loaded {len(df)} labeled samples")

        # Group by prompt for shuffling
        if "prompt_idx" in df.columns:
            grouped = df.groupby("prompt_idx")
            self.samples = []

            for prompt_idx, group in grouped:
                rows = group.to_dict('records')
                if not rows:
                    continue

                for _ in range(int(examples_per_prompt)):
                    row_list = list(rows)
                    random.shuffle(row_list)
                    for r in row_list:
                        self.samples.append(self._process_row(r))
        else:
            self.samples = [self._process_row(r) for _, r in df.iterrows()]
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

        # Convert numpy arrays if needed
        if hasattr(candidate_ids, 'tolist'):
            candidate_ids = candidate_ids.tolist()
        if hasattr(candidate_q_values, 'tolist'):
            candidate_q_values = candidate_q_values.tolist()

        return {
            "prompt_ids": list(prompt_ids),
            "output_ids": list(output_ids),
            "candidate_ids": candidate_ids,
            "candidate_q_values": candidate_q_values,
        }

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
            # Keep prompt, truncate output
            max_output = self.max_length - len(prompt_ids)
            if max_output <= 0:
                # Truncate prompt too
                full_ids = full_ids[:self.max_length]
                candidate_ids = []
                candidate_q_values = []
                output_ids = []
            else:
                output_ids = output_ids[:max_output]
                full_ids = prompt_ids + output_ids
                candidate_ids = candidate_ids[:max_output]
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
        if prompt_len == 0:
            # Skip first output token (no position to predict it from)
            candidate_ids = candidate_ids[1:] if candidate_ids else []
            candidate_q_values = candidate_q_values[1:] if candidate_q_values else []

        input_tensor = torch.tensor(full_ids, dtype=torch.long)

        return {
            "input_ids": input_tensor,
            "label_positions": label_positions,
            "candidate_ids": candidate_ids,
            "candidate_q_values": candidate_q_values,
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
            "taken_token_ids": [s["taken_token_ids"] for s in batch],
        }


def compute_adv_loss(
    model,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute advantage-weighted log likelihood loss.

    L = -Σ A(a) * log π(a | s)

    where A(a) = Q(a) - mean(Q_candidates) for candidates, 0 otherwise.
    Positive advantages push probabilities up, negative push down.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    total_loss = 0.0
    total_positions = 0
    total_advantage = 0.0
    total_entropy = 0.0
    total_min_q = 0.0
    total_max_q = 0.0
    total_mean_q = 0.0
    total_rl_advantage_student = 0.0  # Proper RL advantage: E_π[Q(a) - V(s)]
    total_v_state = 0.0
    vocab_size = logits.shape[-1]

    for batch_idx, (positions, cand_ids_list, cand_q_list, taken_ids) in enumerate(
        zip(batch["label_positions"], batch["candidate_ids"], batch["candidate_q_values"],
            batch["taken_token_ids"])
    ):
        if positions is None or len(positions) == 0:
            continue
        if cand_ids_list is None or len(cand_ids_list) == 0:
            continue

        # Track Q-value of taken action at each position for computing V(s_{t+1})
        prev_q_taken = None

        for pos_idx, (pos, cand_ids, cand_qs, taken_id) in enumerate(
            zip(positions, cand_ids_list, cand_q_list, taken_ids)
        ):
            if cand_ids is None or len(cand_ids) < 2:
                # Need at least 2 candidates for meaningful advantage
                prev_q_taken = None  # Reset since we skipped
                continue

            # Convert to lists if numpy arrays
            cand_ids_list_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
            cand_qs_list_local = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)

            # Get logits for this position
            pos_logits = logits[batch_idx, pos, :]  # [V]

            # Candidate tensors
            cand_ids_tensor = torch.tensor(cand_ids_list_local, dtype=torch.long, device=device)
            cand_qs_tensor = torch.tensor(cand_qs_list_local, dtype=torch.float32, device=device)

            # Compute advantages: A(a) = Q(a) - mean(Q)
            mean_q = cand_qs_tensor.mean()
            advantages = cand_qs_tensor - mean_q  # Can be +/-

            # Get log probs for candidates
            log_probs = F.log_softmax(pos_logits.float(), dim=-1)  # [V]
            cand_log_probs = log_probs[cand_ids_tensor]  # [num_candidates]

            # Loss: -Σ A(a) * log π(a)
            # Only candidates contribute (non-candidates have A=0)
            pos_loss = -(advantages * cand_log_probs).sum()

            total_loss += pos_loss
            total_positions += 1

            # Metrics
            total_advantage += advantages.mean().detach()
            total_min_q += cand_qs_tensor.min().detach()
            total_max_q += cand_qs_tensor.max().detach()
            total_mean_q += cand_qs_tensor.mean().detach()

            # Entropy of student policy over candidates (numerically stable)
            # Normalize in log space to avoid numerical issues
            cand_log_probs_norm = cand_log_probs - torch.logsumexp(cand_log_probs, dim=-1)
            cand_probs_norm = cand_log_probs_norm.exp()
            # Clamp to avoid log(0) = -inf, which causes 0 * -inf = nan
            entropy = -(cand_probs_norm * cand_log_probs_norm.clamp(min=-100)).sum()
            total_entropy += entropy.detach()

            # --- Proper RL Advantage Computation ---
            # V(s_t) = Q(s_{t-1}, a_{t-1}) = prev_q_taken
            # For first position (pos_idx == 0), we don't have prev_q_taken
            if prev_q_taken is not None:
                v_state = prev_q_taken  # V(s_t) = Q-value of previous taken action

                # A(a) = Q(a) - V(s_t) for candidates, 0 for non-candidates
                rl_advantages = torch.zeros(vocab_size, dtype=torch.float32, device=device)
                rl_advantages[cand_ids_tensor] = cand_qs_tensor - v_state

                # Expected advantage under student policy
                p_student = log_probs.exp()
                expected_rl_advantage = (p_student * rl_advantages).sum()

                total_rl_advantage_student += expected_rl_advantage.detach()
                total_v_state += v_state.detach() if isinstance(v_state, torch.Tensor) else v_state

            # Find Q-value of taken action for next position's V(s_{t+1})
            try:
                taken_idx = cand_ids_list_local.index(taken_id)
                prev_q_taken = cand_qs_tensor[taken_idx].detach()
            except ValueError:
                # Taken action not in candidates (shouldn't happen)
                prev_q_taken = None

    if total_positions == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "num_positions": 0,
            "mean_advantage": 0.0,
            "entropy": 0.0,
            "min_q": 0.0,
            "max_q": 0.0,
            "mean_q": 0.0,
            "rl_advantage_student": 0.0,
            "v_state": 0.0,
        }

    avg_loss = total_loss / total_positions

    metrics = {
        "num_positions": total_positions,
        "mean_advantage": (total_advantage / total_positions).item(),
        "entropy": (total_entropy / total_positions).item(),
        "min_q": (total_min_q / total_positions).item(),
        "max_q": (total_max_q / total_positions).item(),
        "mean_q": (total_mean_q / total_positions).item(),
        "rl_advantage_student": total_rl_advantage_student / max(1, total_positions) if isinstance(total_rl_advantage_student, torch.Tensor) else total_rl_advantage_student / max(1, total_positions),
        "v_state": total_v_state / max(1, total_positions) if isinstance(total_v_state, torch.Tensor) else total_v_state / max(1, total_positions),
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

    if master and wandb_project:
        wandb.init(
            project=wandb_project,
            name=f"policy_adv_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
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
    accum_entropy = 0.0
    accum_min_q = 0.0
    accum_max_q = 0.0
    accum_mean_q = 0.0
    accum_rl_advantage_student = 0.0
    accum_v_state = 0.0
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
                loss, metrics = compute_adv_loss(model, batch)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_advantage += metrics["mean_advantage"]
            accum_entropy += metrics["entropy"]
            accum_min_q += metrics["min_q"]
            accum_max_q += metrics["max_q"]
            accum_mean_q += metrics["mean_q"]
            accum_rl_advantage_student += metrics["rl_advantage_student"].item() if isinstance(metrics["rl_advantage_student"], torch.Tensor) else metrics["rl_advantage_student"]
            accum_v_state += metrics["v_state"].item() if isinstance(metrics["v_state"], torch.Tensor) else metrics["v_state"]
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
                avg_ent = accum_entropy / max(1, accum_count)
                avg_min_q = accum_min_q / max(1, accum_count)
                avg_max_q = accum_max_q / max(1, accum_count)
                avg_mean_q = accum_mean_q / max(1, accum_count)
                avg_rl_adv = accum_rl_advantage_student / max(1, accum_count)
                avg_v = accum_v_state / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} "
                    f"adv={avg_adv:.4f} rl_adv={avg_rl_adv:.4f} V={avg_v:.3f} "
                    f"H={avg_ent:.3f} Q=[{avg_min_q:.3f}, {avg_mean_q:.3f}, {avg_max_q:.3f}] lr={lr:.2e}"
                )

                if wandb_project:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/mean_advantage": avg_adv,
                        "train/rl_advantage_student": avg_rl_adv,
                        "train/v_state": avg_v,
                        "train/entropy": avg_ent,
                        "train/min_q": avg_min_q,
                        "train/max_q": avg_max_q,
                        "train/mean_q": avg_mean_q,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_advantage = 0.0
                accum_entropy = 0.0
                accum_min_q = 0.0
                accum_max_q = 0.0
                accum_mean_q = 0.0
                accum_rl_advantage_student = 0.0
                accum_v_state = 0.0
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

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = PolicyAdvDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        examples_per_prompt=cfg.examples_per_prompt,
    )

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        PolicyAdvDataset.collate_fn,
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
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with advantage-weighted log likelihood")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_adv")
    p.add_argument("--data_path", required=True, help="Path to labeled parquet file")

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
    p.add_argument("--wandb_project", default="policy_adv")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
