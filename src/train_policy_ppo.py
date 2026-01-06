#!/usr/bin/env python3
"""
Train a student policy with PPO-style loss using pre-computed Q-values.

Unlike standard PPO which samples actions and estimates advantages for those samples,
we have Q(s,a) for all candidate actions. This allows computing exact expectations:

- V(s) = E_{a ~ π_ref}[Q(s,a)] = Σ_a π_ref(a) * Q(s,a)  (normalized over candidates)
- A(s,a) = Q(s,a) - V(s)
- Loss = -Σ_a π_θ(a) * Ã(a)  where Ã is clipped advantage

PPO clipping is adapted: we zero out gradients for actions where the ratio
π_θ(a)/π_ref(a) has already moved outside [1-ε, 1+ε] in the beneficial direction.

Input data must have columns:
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate
- candidate_ref_logprobs: List[List[float]] - reference model log probs for each candidate
- output_token_ids: List[int] - the actual trajectory tokens (for reference)

Example usage:
    python train_policy_ppo.py --model_id Qwen/Qwen3-1.7B \
                               --weights_path models/policy_ppo \
                               --data_path data/labeled.parquet
"""
from __future__ import annotations
import argparse, math, os, time, random
import wandb
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


class PolicyPPODataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
    ):
        """
        Dataset for training policy with PPO loss.

        Each sample contains:
        - prompt_token_ids: tokenized prompt
        - output_token_ids: tokenized response
        - candidate_ids: List of candidate token IDs per position
        - candidate_q_values: List of Q-values per candidate per position
        - candidate_ref_logprobs: List of reference log probs per candidate per position
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        df = pq.read_table(table).to_pandas() if isinstance(table, str) else table

        # Validate required columns
        required_cols = ["prompt_token_ids", "output_token_ids", "candidate_ids", "candidate_q_values", "candidate_ref_logprobs"]
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

        # Convert numpy arrays if needed
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
            # Keep prompt, truncate output
            max_output = self.max_length - len(prompt_ids)
            if max_output <= 0:
                # Truncate prompt too
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


def compute_ppo_loss(
    model,
    batch,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute PPO-style loss with pre-computed Q-values for all candidate actions.

    Since we have Q(s,a) for all candidates (not just sampled actions), we compute:
    - V(s) = E_{π_ref}[Q] = Σ_a π_ref(a) * Q(a)  (normalized over candidates)
    - A(s,a) = Q(s,a) - V(s)
    - Loss = -Σ_a π_θ(a) * Ã(a)

    PPO clipping: zero out gradient for actions where ratio is outside [1-ε, 1+ε]
    in the direction that would further increase the objective.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]

    total_loss = 0.0
    total_positions = 0
    total_clipped = 0
    total_candidates = 0
    total_entropy_student = 0.0
    total_advantage = 0.0
    total_advantage_abs = 0.0
    total_v_state = 0.0
    total_ratio_mean = 0.0
    total_ratio_max = 0.0
    total_ratio_min = 0.0

    # Main loss computation
    for batch_idx, (positions, cand_ids_list, cand_q_list, cand_ref_logprobs_list, taken_ids) in enumerate(
        zip(batch["label_positions"], batch["candidate_ids"], batch["candidate_q_values"],
            batch["candidate_ref_logprobs"], batch["taken_token_ids"])
    ):
        if positions is None or len(positions) == 0:
            continue
        if cand_ids_list is None or len(cand_ids_list) == 0:
            continue

        for pos_idx, (pos, cand_ids, cand_qs, cand_ref_lps, taken_id) in enumerate(
            zip(positions, cand_ids_list, cand_q_list, cand_ref_logprobs_list, taken_ids)
        ):
            if cand_ids is None or len(cand_ids) < 2:
                continue

            # Convert to lists if numpy arrays
            cand_ids_list_local = list(cand_ids) if hasattr(cand_ids, 'tolist') else list(cand_ids)
            cand_qs_list_local = list(cand_qs) if hasattr(cand_qs, 'tolist') else list(cand_qs)
            cand_ref_lps_local = list(cand_ref_lps) if hasattr(cand_ref_lps, 'tolist') else list(cand_ref_lps)

            # Get logits for this position
            pos_logits = logits[batch_idx, pos, :]  # [V]

            # Candidate tensors
            cand_ids_tensor = torch.tensor(cand_ids_list_local, dtype=torch.long, device=device)
            cand_qs_tensor = torch.tensor(cand_qs_list_local, dtype=torch.float32, device=device)
            cand_ref_lps_tensor = torch.tensor(cand_ref_lps_local, dtype=torch.float32, device=device)

            # Student policy: π_θ (full vocab softmax, then extract candidates)
            log_p_student = F.log_softmax(pos_logits.float(), dim=-1)
            p_student = log_p_student.exp()

            # Get student probs for candidates
            log_p_student_cands = log_p_student[cand_ids_tensor]
            p_student_cands = p_student[cand_ids_tensor]

            # Reference policy: normalize over candidates
            # p_ref_cands = softmax(ref_logprobs) to get proper distribution over candidates
            p_ref_cands = F.softmax(cand_ref_lps_tensor, dim=-1)

            # V(s) = E_π_ref[Q] = Σ_a π_ref(a) * Q(a)
            V = (p_ref_cands * cand_qs_tensor).sum()

            # A(s,a) = Q(s,a) - V(s)
            advantages = cand_qs_tensor - V

            # Ratio r(a) = π_θ(a) / π_ref(a)
            ratio = p_student_cands / (p_ref_cands.detach() + 1e-10)

            # PPO clipping mask (detached - no gradient through mask decision)
            # Zero out advantage where:
            # - A > 0 and r > 1+ε (already increased enough)
            # - A < 0 and r < 1-ε (already decreased enough)
            clip_high = (advantages > 0) & (ratio > 1 + clip_eps)
            clip_low = (advantages < 0) & (ratio < 1 - clip_eps)
            clip_mask = ~(clip_high | clip_low)
            clip_mask = clip_mask.detach().float()

            # Clipped advantages
            advantages_clipped = advantages * clip_mask

            # Loss: -Σ_a π_θ(a) * Ã(a)
            # This is the policy gradient in expectation form
            loss_pos = -(p_student_cands * advantages_clipped).sum()

            total_loss += loss_pos
            total_positions += 1

            # Track clipping statistics
            num_cands = len(cand_ids_list_local)
            num_clipped = (1 - clip_mask).sum().item()
            total_clipped += num_clipped
            total_candidates += num_cands

            # Entropy of student policy (full vocab)
            total_entropy_student += -(p_student * log_p_student).sum().detach()

            # Advantage statistics
            total_advantage += (p_student_cands.detach() * advantages.detach()).sum()
            total_advantage_abs += advantages.abs().mean().detach()
            total_v_state += V.detach()

            # Ratio statistics
            total_ratio_mean += ratio.mean().detach()
            total_ratio_max += ratio.max().detach()
            total_ratio_min += ratio.min().detach()

    if total_positions == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {
            "num_positions": 0,
            "clip_fraction": 0.0,
            "entropy_student": 0.0,
            "advantage": 0.0,
            "advantage_abs": 0.0,
            "v_state": 0.0,
            "ratio_mean": 0.0,
            "ratio_max": 0.0,
            "ratio_min": 0.0,
        }

    avg_loss = total_loss / total_positions

    metrics = {
        "num_positions": total_positions,
        "clip_fraction": total_clipped / max(1, total_candidates),
        "entropy_student": (total_entropy_student / total_positions).item(),
        "advantage": (total_advantage / total_positions).item(),
        "advantage_abs": (total_advantage_abs / total_positions).item(),
        "v_state": (total_v_state / total_positions).item(),
        "ratio_mean": (total_ratio_mean / total_positions).item(),
        "ratio_max": (total_ratio_max / total_positions).item(),
        "ratio_min": (total_ratio_min / total_positions).item(),
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
            name=f"policy_ppo_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "clip_eps": clip_eps,
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
    accum_clip_frac = 0.0
    accum_entropy = 0.0
    accum_advantage = 0.0
    accum_advantage_abs = 0.0
    accum_v_state = 0.0
    accum_ratio_mean = 0.0
    accum_ratio_max = 0.0
    accum_ratio_min = 0.0
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
                loss, metrics = compute_ppo_loss(
                    model, batch,
                    clip_eps=clip_eps,
                )
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_clip_frac += metrics["clip_fraction"]
            accum_entropy += metrics["entropy_student"]
            accum_advantage += metrics["advantage"]
            accum_advantage_abs += metrics["advantage_abs"]
            accum_v_state += metrics["v_state"]
            accum_ratio_mean += metrics["ratio_mean"]
            accum_ratio_max += metrics["ratio_max"]
            accum_ratio_min += metrics["ratio_min"]
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
                avg_clip = accum_clip_frac / max(1, accum_count)
                avg_ent = accum_entropy / max(1, accum_count)
                avg_adv = accum_advantage / max(1, accum_count)
                avg_adv_abs = accum_advantage_abs / max(1, accum_count)
                avg_v = accum_v_state / max(1, accum_count)
                avg_ratio_mean = accum_ratio_mean / max(1, accum_count)
                avg_ratio_max = accum_ratio_max / max(1, accum_count)
                avg_ratio_min = accum_ratio_min / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} clip={avg_clip:.3f} "
                    f"adv={avg_adv:.4f} |adv|={avg_adv_abs:.4f} V={avg_v:.3f} "
                    f"H={avg_ent:.3f} r=[{avg_ratio_min:.2f},{avg_ratio_mean:.2f},{avg_ratio_max:.2f}] lr={lr:.2e}"
                )

                if wandb_project and wandb:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/clip_fraction": avg_clip,
                        "train/advantage": avg_adv,
                        "train/advantage_abs": avg_adv_abs,
                        "train/v_state": avg_v,
                        "train/entropy_student": avg_ent,
                        "train/ratio_mean": avg_ratio_mean,
                        "train/ratio_max": avg_ratio_max,
                        "train/ratio_min": avg_ratio_min,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_clip_frac = 0.0
                accum_entropy = 0.0
                accum_advantage = 0.0
                accum_advantage_abs = 0.0
                accum_v_state = 0.0
                accum_ratio_mean = 0.0
                accum_ratio_max = 0.0
                accum_ratio_min = 0.0
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

    dataset = PolicyPPODataset(
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
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with PPO loss using pre-computed Q-values")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_ppo")
    p.add_argument("--data_path", required=True, help="Path to labeled parquet file")

    # PPO parameters
    p.add_argument("--clip_eps", type=float, default=0.2,
                   help="PPO clipping epsilon")

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
