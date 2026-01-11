#!/usr/bin/env python3
"""
Train a student policy with rejection finetuning (RFT) baseline.

This is a simple baseline that:
1. Filters trajectories based on reward > threshold (success)
2. Performs standard SFT (cross-entropy loss) on successful trajectories

Reward is determined by (in order of preference):
- 'correct' column if available (0 or 1)
- Final Q-value from 'candidate_q_values' as fallback

Input data must have columns:
- prompt_token_ids: List[int] - tokenized prompt
- output_token_ids: List[int] - the actual trajectory tokens
- correct (optional): float - trajectory reward (0 or 1)
- candidate_q_values (fallback): List[List[float]] - Q-values for each candidate

Example usage:
    python train_policy_rft.py --model_id Qwen/Qwen3-1.7B \
                               --weights_path models/policy_rft \
                               --data_path data/labeled.parquet
"""
from __future__ import annotations
import argparse, math, os, time, random
import wandb
from typing import List, Dict, Any, Tuple
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


class PolicyRFTDataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
        success_threshold: float = 0.5,
    ):
        """
        Dataset for rejection finetuning.

        Filters trajectories based on final Q-value > success_threshold,
        then trains with standard SFT loss.

        Each sample contains:
        - prompt_token_ids: tokenized prompt
        - output_token_ids: tokenized response (SFT target)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.success_threshold = success_threshold

        # Load data
        df = pq.read_table(table).to_pandas() if isinstance(table, str) else table

        # Validate required columns
        required_cols = ["prompt_token_ids", "output_token_ids"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        # Check for reward columns
        has_correct = "correct" in df.columns
        has_q_values = "candidate_q_values" in df.columns

        if not has_correct and not has_q_values:
            raise ValueError("Dataset must have either 'correct' column or 'candidate_q_values' column")

        total_samples = len(df)
        print(f"Loaded {total_samples} samples")

        # Compute reward for each trajectory (prefer 'correct', fall back to final Q-value)
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

        # Filter based on reward > threshold
        df = df[df["reward"] > success_threshold].copy()

        success_samples = len(df)
        reward_source = "correct column" if has_correct else "final Q-value"
        print(f"Filtered to {success_samples} successful samples (reward > {success_threshold}, using {reward_source})")
        print(f"Success rate: {success_samples / total_samples * 100:.1f}%")

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

        # Get candidate data for common metrics (if available)
        candidate_ids = row.get("candidate_ids")
        candidate_q_values = row.get("candidate_q_values")
        candidate_ref_logprobs = row.get("candidate_ref_logprobs")

        if candidate_ids is not None and hasattr(candidate_ids, 'tolist'):
            candidate_ids = candidate_ids.tolist()
        if candidate_q_values is not None and hasattr(candidate_q_values, 'tolist'):
            candidate_q_values = candidate_q_values.tolist()
        if candidate_ref_logprobs is not None and hasattr(candidate_ref_logprobs, 'tolist'):
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
            # Keep prompt, truncate output
            max_output = self.max_length - len(prompt_ids)
            if max_output <= 0:
                # Truncate prompt too
                full_ids = full_ids[:self.max_length]
                output_ids = []
                candidate_ids = [] if candidate_ids else None
                candidate_q_values = [] if candidate_q_values else None
                candidate_ref_logprobs = [] if candidate_ref_logprobs else None
            else:
                output_ids = output_ids[:max_output]
                full_ids = prompt_ids + output_ids
                if candidate_ids:
                    candidate_ids = candidate_ids[:max_output]
                if candidate_q_values:
                    candidate_q_values = candidate_q_values[:max_output]
                if candidate_ref_logprobs:
                    candidate_ref_logprobs = candidate_ref_logprobs[:max_output]

        prompt_len = len(prompt_ids)

        # Build label positions and taken token ids for common metrics
        label_positions = []
        taken_token_ids = []
        for j in range(len(output_ids)):
            pos = prompt_len + j - 1
            if pos >= 0:
                label_positions.append(pos)
                taken_token_ids.append(output_ids[j])

        # Adjust candidate arrays for label positions
        if prompt_len <= 0:
            if candidate_ids:
                candidate_ids = candidate_ids[1:]
            if candidate_q_values:
                candidate_q_values = candidate_q_values[1:]
            if candidate_ref_logprobs:
                candidate_ref_logprobs = candidate_ref_logprobs[1:]

        input_tensor = torch.tensor(full_ids, dtype=torch.long)

        # Labels: -100 for prompt positions, actual token ids for output positions
        # For causal LM, we predict next token, so labels are shifted
        labels = torch.full_like(input_tensor, -100)
        # Label positions start from prompt_len (first output token is predicted by last prompt token)
        if len(output_ids) > 0:
            labels[prompt_len:] = input_tensor[prompt_len:]

        return {
            "input_ids": input_tensor,
            "labels": labels,
            "prompt_len": prompt_len,
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

        input_ids = torch.stack([
            F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0)), value=pad_val)
            for s in batch
        ])

        labels = torch.stack([
            F.pad(s["labels"], (0, max_len - s["labels"].size(0)), value=-100)
            for s in batch
        ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_lens": [s["prompt_len"] for s in batch],
            "label_positions": [s["label_positions"] for s in batch],
            "candidate_ids": [s["candidate_ids"] for s in batch],
            "candidate_q_values": [s["candidate_q_values"] for s in batch],
            "candidate_ref_logprobs": [s["candidate_ref_logprobs"] for s in batch],
            "taken_token_ids": [s["taken_token_ids"] for s in batch],
        }


def compute_sft_loss(
    model,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute standard SFT (cross-entropy) loss.

    Loss is computed only on output tokens (labels != -100).
    Also computes common metrics (entropy_student, kl_student_ref, advantage_student)
    if candidate data is available.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Cross-entropy loss (ignores -100)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    # Compute metrics
    with torch.no_grad():
        # Number of tokens trained on
        num_tokens = (shift_labels != -100).sum().item()

        # Perplexity
        perplexity = torch.exp(loss).item()

        # Accuracy (for monitoring)
        valid_mask = shift_labels != -100
        if valid_mask.sum() > 0:
            predictions = shift_logits.argmax(dim=-1)
            correct = (predictions == shift_labels) & valid_mask
            accuracy = correct.sum().float() / valid_mask.sum().float()
        else:
            accuracy = torch.tensor(0.0)

    # Compute common metrics over candidates (if available)
    total_entropy_student = 0.0
    total_kl_student_ref = 0.0
    total_advantage_student = 0.0
    total_common_positions = 0

    with torch.no_grad():
        for batch_idx, (positions, cand_ids_list, cand_q_values_list, cand_ref_logprobs_list, taken_ids) in enumerate(
            zip(batch["label_positions"], batch["candidate_ids"],
                batch["candidate_q_values"], batch["candidate_ref_logprobs"],
                batch["taken_token_ids"])
        ):
            if positions is None or cand_ids_list is None or cand_ref_logprobs_list is None:
                continue
            if len(positions) == 0 or len(cand_ids_list) == 0:
                continue

            for pos_idx, pos in enumerate(positions):
                if pos_idx >= len(cand_ids_list):
                    continue

                cand_ids = cand_ids_list[pos_idx]
                cand_ref_lps = cand_ref_logprobs_list[pos_idx] if cand_ref_logprobs_list else None

                if cand_ids is None or len(cand_ids) < 1:
                    continue
                if cand_ref_lps is None or len(cand_ref_lps) < 1:
                    continue

                cand_ids_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
                cand_ref_lps_local = list(cand_ref_lps) if hasattr(cand_ref_lps, 'tolist') else list(cand_ref_lps)

                # Get Q-values if available
                cand_qs = None
                if cand_q_values_list is not None and pos_idx < len(cand_q_values_list):
                    cand_qs = cand_q_values_list[pos_idx]
                    if cand_qs is not None:
                        cand_qs = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)

                # Get logits for this position
                pos_logits = logits[batch_idx, pos, :]  # [V]

                # Candidate tensors
                cand_ids_tensor = torch.tensor(cand_ids_local, dtype=torch.long, device=device)
                cand_ref_logprobs_tensor = torch.tensor(cand_ref_lps_local, dtype=torch.float32, device=device)

                # Student policy over candidates
                cand_logits = pos_logits[cand_ids_tensor].float()
                log_p_student_cands = F.log_softmax(cand_logits, dim=-1)
                p_student_cands = log_p_student_cands.exp()

                # Reference policy over candidates
                log_p_ref_cands = F.log_softmax(cand_ref_logprobs_tensor, dim=-1)
                p_ref_cands = log_p_ref_cands.exp()

                # Entropy of student over candidates
                entropy_student = -(p_student_cands * log_p_student_cands).sum()
                total_entropy_student += entropy_student.item()

                # KL(student || ref) over candidates
                kl_student_ref = (p_student_cands * (log_p_student_cands - log_p_ref_cands)).sum()
                total_kl_student_ref += kl_student_ref.item()

                # Advantage: V(s) = E_ref[Q], A = Q - V (if Q-values available)
                if cand_qs is not None and len(cand_qs) == len(cand_ids_local):
                    cand_qs_tensor = torch.tensor(cand_qs, dtype=torch.float32, device=device)
                    v_state = (p_ref_cands * cand_qs_tensor).sum()
                    advantages_cands = cand_qs_tensor - v_state
                    advantage_student = (p_student_cands * advantages_cands).sum()
                    total_advantage_student += advantage_student.item()

                total_common_positions += 1

    metrics = {
        "num_tokens": num_tokens,
        "perplexity": perplexity,
        "accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
        # Common metrics
        "entropy_student": total_entropy_student / max(1, total_common_positions),
        "kl_student_ref": total_kl_student_ref / max(1, total_common_positions),
        "advantage_student": total_advantage_student / max(1, total_common_positions),
    }

    return loss, metrics


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

    if master and wandb_project and wandb:
        wandb.init(
            project=wandb_project,
            name=f"policy_rft_{int(time.time())}",
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
        # Start at 0, linearly increase to learning_rate over warmup_iters
        if i <= warmup_iters:
            return learning_rate * (i - 1) / max(1, warmup_iters)
        progress = (i - warmup_iters) / max(1, total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    global_step = 0
    accum_loss = 0.0
    accum_perplexity = 0.0
    accum_accuracy = 0.0
    accum_tokens = 0
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
                loss, metrics = compute_sft_loss(model, batch)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_perplexity += metrics["perplexity"]
            accum_accuracy += metrics["accuracy"]
            accum_tokens += metrics["num_tokens"]
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
                avg_ppl = accum_perplexity / max(1, accum_count)
                avg_acc = accum_accuracy / max(1, accum_count)
                avg_tokens = accum_tokens / max(1, accum_count)
                # Common metrics
                avg_entropy_student = accum_entropy_student / max(1, accum_count)
                avg_kl_student_ref = accum_kl_student_ref / max(1, accum_count)
                avg_advantage_student = accum_advantage_student / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} ppl={avg_ppl:.2f} "
                    f"acc={avg_acc:.3f} tokens={avg_tokens:.0f} lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/perplexity": avg_ppl,
                        "train/accuracy": avg_acc,
                        "train/tokens_per_batch": avg_tokens,
                        # Common metrics
                        "train/entropy_student": avg_entropy_student,
                        "train/kl_student_ref": avg_kl_student_ref,
                        "train/advantage_student": avg_advantage_student,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_perplexity = 0.0
                accum_accuracy = 0.0
                accum_tokens = 0
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

    dataset = PolicyRFTDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        examples_per_prompt=cfg.examples_per_prompt,
        success_threshold=cfg.success_threshold,
    )

    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        os.makedirs(cfg.weights_path, exist_ok=True)
        tokenizer.save_pretrained(cfg.weights_path)

    train(
        model,
        dataset,
        cfg.weights_path,
        PolicyRFTDataset.collate_fn,
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
    p = argparse.ArgumentParser(description="Train policy with rejection finetuning (SFT on successful trajectories)")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_rft")
    p.add_argument("--data_path", required=True, help="Path to labeled parquet file")

    # RFT parameters
    p.add_argument("--success_threshold", type=float, default=0.5,
                   help="Final Q-value threshold for success (default: 0.5)")

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
    p.add_argument("--wandb_project", default="policy_rft")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
