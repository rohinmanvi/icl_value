#!/usr/bin/env python3
"""
Train a student policy with advantage-weighted rejection finetuning (AW-RFT).

This baseline:
1. Computes sequence-level advantage A = Q(taken_last) - Q(taken_first)
2. Filters trajectories where A < 0 (rejection sampling)
3. Weights the SFT loss by A for remaining trajectories

Input data must have columns:
- prompt_token_ids: List[int] - tokenized prompt
- output_token_ids: List[int] - the actual trajectory tokens
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate

Example usage:
    python train_policy_awrft.py --model_id Qwen/Qwen3-1.7B \
                                 --weights_path models/policy_awrft \
                                 --data_path data/labeled.parquet
"""
from __future__ import annotations
import argparse, math, os, time, random
import wandb
from typing import List, Dict, Any, Tuple, Optional
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


class PolicyAWRFTDataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
    ):
        """
        Dataset for advantage-weighted rejection finetuning.

        Computes A = Q(taken_last) - Q(taken_first) for each trajectory.
        Filters trajectories where A < 0, then trains with A-weighted SFT loss.

        Each sample contains:
        - prompt_token_ids: tokenized prompt
        - output_token_ids: tokenized response (SFT target)
        - advantage: sequence-level advantage for weighting
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
        df = df[df["candidate_q_values"].notna()].copy()
        df = df[df["candidate_ids"].notna()].copy()

        total_samples = len(df)
        print(f"Loaded {total_samples} labeled samples")

        # Compute sequence-level advantage: A = Q(taken_last) - Q(taken_first)
        def compute_advantage(row) -> Optional[float]:
            candidate_ids = row["candidate_ids"]
            candidate_q_values = row["candidate_q_values"]
            output_ids = row["output_token_ids"]

            if candidate_ids is None or len(candidate_ids) == 0:
                return None
            if candidate_q_values is None or len(candidate_q_values) == 0:
                return None
            if output_ids is None or len(output_ids) == 0:
                return None

            # Convert if needed
            if hasattr(output_ids, 'tolist'):
                output_ids = output_ids.tolist()

            def get_taken_q(pos_idx: int) -> Optional[float]:
                """Get Q-value of the taken action at position pos_idx."""
                if pos_idx < 0 or pos_idx >= len(candidate_ids):
                    return None
                if pos_idx >= len(output_ids):
                    return None

                cand_ids = candidate_ids[pos_idx]
                cand_qs = candidate_q_values[pos_idx]

                if cand_ids is None or cand_qs is None:
                    return None
                if len(cand_ids) == 0 or len(cand_qs) == 0:
                    return None

                # Convert if needed
                if hasattr(cand_ids, 'tolist'):
                    cand_ids = cand_ids.tolist()
                if hasattr(cand_qs, 'tolist'):
                    cand_qs = cand_qs.tolist()

                taken_token = output_ids[pos_idx]

                # Find Q-value of taken action
                try:
                    taken_idx = list(cand_ids).index(taken_token)
                    return cand_qs[taken_idx]
                except ValueError:
                    # Taken token not in candidates
                    return None

            # Get Q-values at first and last positions
            q_first = get_taken_q(0)
            q_last = get_taken_q(len(candidate_ids) - 1)

            if q_first is None or q_last is None:
                return None

            return q_last - q_first

        df["advantage"] = df.apply(compute_advantage, axis=1)
        df = df[df["advantage"].notna()].copy()

        # Filter: keep only A >= 0
        df_positive = df[df["advantage"] >= 0].copy()

        positive_samples = len(df_positive)
        print(f"Filtered to {positive_samples} samples with A >= 0")
        print(f"Positive advantage rate: {positive_samples / total_samples * 100:.1f}%")

        if positive_samples > 0:
            adv_values = df_positive["advantage"].values
            print(f"Advantage stats: mean={np.mean(adv_values):.4f}, std={np.std(adv_values):.4f}, "
                  f"min={np.min(adv_values):.4f}, max={np.max(adv_values):.4f}")

        # Group by prompt for shuffling
        if "prompt_idx" in df_positive.columns:
            grouped = df_positive.groupby("prompt_idx")
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
            self.samples = [self._process_row(r.to_dict()) for idx, r in df_positive.iterrows()]
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

        return {
            "prompt_ids": list(prompt_ids),
            "output_ids": list(output_ids),
            "advantage": float(row["advantage"]),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_ids = sample["prompt_ids"]
        output_ids = sample["output_ids"]
        advantage = sample["advantage"]

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
            else:
                output_ids = output_ids[:max_output]
                full_ids = prompt_ids + output_ids

        prompt_len = len(prompt_ids)

        input_tensor = torch.tensor(full_ids, dtype=torch.long)

        # Labels: -100 for prompt positions, actual token ids for output positions
        labels = torch.full_like(input_tensor, -100)
        if len(output_ids) > 0:
            labels[prompt_len:] = input_tensor[prompt_len:]

        return {
            "input_ids": input_tensor,
            "labels": labels,
            "prompt_len": prompt_len,
            "advantage": advantage,
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

        advantages = torch.tensor([s["advantage"] for s in batch], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_lens": [s["prompt_len"] for s in batch],
            "advantages": advantages,
        }


def compute_awsft_loss(
    model,
    batch,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute advantage-weighted SFT loss.

    Loss = A * cross_entropy_loss for each sample in batch.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    advantages = batch["advantages"].to(device)  # [B]

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size = shift_logits.size(0)

    # Compute per-sample loss
    total_weighted_loss = 0.0
    total_unweighted_loss = 0.0
    total_tokens = 0

    for i in range(batch_size):
        sample_logits = shift_logits[i]  # [S-1, V]
        sample_labels = shift_labels[i]  # [S-1]
        sample_adv = advantages[i]

        # Mask for valid tokens
        valid_mask = sample_labels != -100
        if valid_mask.sum() == 0:
            continue

        # Cross-entropy for this sample
        sample_loss = F.cross_entropy(
            sample_logits[valid_mask],
            sample_labels[valid_mask],
            reduction='mean'
        )

        # Weight by advantage
        weighted_loss = sample_adv * sample_loss

        total_weighted_loss += weighted_loss
        total_unweighted_loss += sample_loss.detach()
        total_tokens += valid_mask.sum().item()

    # Average over batch
    if batch_size > 0:
        avg_weighted_loss = total_weighted_loss / batch_size
        avg_unweighted_loss = total_unweighted_loss / batch_size
    else:
        avg_weighted_loss = torch.tensor(0.0, device=device, requires_grad=True)
        avg_unweighted_loss = torch.tensor(0.0, device=device)

    # Compute metrics
    with torch.no_grad():
        perplexity = torch.exp(avg_unweighted_loss).item()
        avg_advantage = advantages.mean().item()

        # Accuracy
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        valid_mask = flat_labels != -100
        if valid_mask.sum() > 0:
            predictions = flat_logits.argmax(dim=-1)
            correct = (predictions == flat_labels) & valid_mask
            accuracy = correct.sum().float() / valid_mask.sum().float()
        else:
            accuracy = torch.tensor(0.0)

    metrics = {
        "num_tokens": total_tokens,
        "perplexity": perplexity,
        "accuracy": accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
        "unweighted_loss": avg_unweighted_loss.item(),
        "avg_advantage": avg_advantage,
    }

    return avg_weighted_loss, metrics


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
            name=f"policy_awrft_{int(time.time())}",
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
    accum_unweighted_loss = 0.0
    accum_perplexity = 0.0
    accum_accuracy = 0.0
    accum_advantage = 0.0
    accum_tokens = 0
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
                loss, metrics = compute_awsft_loss(model, batch)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_unweighted_loss += metrics["unweighted_loss"]
            accum_perplexity += metrics["perplexity"]
            accum_accuracy += metrics["accuracy"]
            accum_advantage += metrics["avg_advantage"]
            accum_tokens += metrics["num_tokens"]
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
                avg_uw_loss = accum_unweighted_loss / max(1, accum_count)
                avg_ppl = accum_perplexity / max(1, accum_count)
                avg_acc = accum_accuracy / max(1, accum_count)
                avg_adv = accum_advantage / max(1, accum_count)
                avg_tokens = accum_tokens / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} uw_loss={avg_uw_loss:.4f} ppl={avg_ppl:.2f} "
                    f"acc={avg_acc:.3f} adv={avg_adv:.4f} tokens={avg_tokens:.0f} lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/weighted_loss": avg_loss,
                        "train/unweighted_loss": avg_uw_loss,
                        "train/perplexity": avg_ppl,
                        "train/accuracy": avg_acc,
                        "train/avg_advantage": avg_adv,
                        "train/tokens_per_batch": avg_tokens,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_unweighted_loss = 0.0
                accum_perplexity = 0.0
                accum_accuracy = 0.0
                accum_advantage = 0.0
                accum_tokens = 0
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

    dataset = PolicyAWRFTDataset(
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
        PolicyAWRFTDataset.collate_fn,
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
    p = argparse.ArgumentParser(description="Train policy with advantage-weighted rejection finetuning")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_awrft")
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
    p.add_argument("--wandb_project", default="policy_awrft")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
