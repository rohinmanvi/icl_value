#!/usr/bin/env python3
"""
Train a student policy with KL divergence loss against the extracted policy.

The extracted policy is: π_ext(a) ∝ π_ref(a) * exp(Q(a) / τ)

We minimize KL(π_extracted || π_student) (forward KL for mode-covering).
Non-candidate tokens use min-p filtered reference probabilities with min(Q).

Input data must have columns:
- candidate_ids: List[List[int]] - token IDs of candidates at each position
- candidate_q_values: List[List[float]] - Q-values for each candidate
- candidate_ref_logprobs: List[List[float]] - reference model log probs for each candidate
- output_token_ids: List[int] - the actual trajectory tokens (for reference)

Example usage:
    python train_policy_kl.py --model_id Qwen/Qwen3-1.7B \
                              --weights_path models/policy_kl \
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


class PolicyKLDataset(Dataset):
    def __init__(
        self,
        table,
        tokenizer,
        max_length: int = 131072,
        examples_per_prompt: int = 1,
    ):
        """
        Dataset for training policy with KL loss against extracted policy.

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
        # candidate_ids[j] corresponds to output_ids[j], which is predicted at label_positions[j]
        # But label_positions starts from j=0 when pos >= 0, so we may skip the first output token
        if prompt_len > 0:
            # First output token is predicted at position prompt_len - 1
            # candidate_ids[0] corresponds to this
            pass
        else:
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


def compute_kl_loss(
    model,
    batch,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute KL divergence loss: KL(π_extracted || π_student) - forward KL for mode-covering

    π_student = softmax(student_logits) - full vocabulary from student model
    π_extracted ∝ π_ref * exp(Q / τ) - using stored reference log probs

    For non-candidate tokens, we use min(Q) and assume negligible reference probability.

    Also computes proper RL advantage:
    - V(s_t) = Q(s_{t-1}, a_{t-1}) (Q-value of previous taken action)
    - A(s_t, a) = Q(s_t, a) - V(s_t) for candidates, 0 for non-candidates
    - Expected advantage = Σ π(a) * A(a)
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [B, S, V]
    vocab_size = logits.shape[-1]

    total_kl = 0.0
    total_positions = 0
    total_entropy_student = 0.0
    total_entropy_extracted = 0.0
    total_advantage_student = 0.0
    total_advantage_extracted = 0.0
    total_v_state = 0.0
    total_ref_mass = 0.0  # Probability mass on candidates from reference policy
    total_sft_loss = 0.0  # SFT loss: -log π_student(a_taken | s)
    total_ref_sft_loss = 0.0  # Reference SFT loss: -log π_ref(a_taken | s)

    for batch_idx, (positions, cand_ids_list, cand_q_list, cand_ref_logprobs_list, taken_ids) in enumerate(
        zip(batch["label_positions"], batch["candidate_ids"], batch["candidate_q_values"],
            batch["candidate_ref_logprobs"], batch["taken_token_ids"])
    ):
        if positions is None or len(positions) == 0:
            continue
        if cand_ids_list is None or len(cand_ids_list) == 0:
            continue

        # Track Q-value of taken action at each position for computing V(s_{t+1})
        prev_q_taken = None

        for pos_idx, (pos, cand_ids, cand_qs, cand_ref_lps, taken_id) in enumerate(
            zip(positions, cand_ids_list, cand_q_list, cand_ref_logprobs_list, taken_ids)
        ):
            # Handle numpy arrays and lists
            if cand_ids is None or len(cand_ids) < 2:
                # Need at least 2 candidates for meaningful KL
                prev_q_taken = None  # Reset since we skipped
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

            # π_student = softmax(student_logits) - full vocabulary
            log_p_student = F.log_softmax(pos_logits.float(), dim=-1)  # [V]
            p_student = log_p_student.exp()

            # π_extracted ∝ π_ref * exp(Q / τ)
            # In log space: log π_extracted = log π_ref + Q / τ - log Z
            # For candidates, we have stored ref log probs
            # For non-candidates, use min(Q) and very low reference prob

            min_q = cand_qs_tensor.min()
            min_ref_lp = cand_ref_lps_tensor.min() - 10.0  # Very low prob for non-candidates

            # Build full vocabulary log probs for extracted policy (unnormalized)
            log_ref_extended = torch.full((vocab_size,), min_ref_lp.item(), dtype=torch.float32, device=device)
            log_ref_extended[cand_ids_tensor] = cand_ref_lps_tensor

            q_extended = torch.full((vocab_size,), min_q.item(), dtype=torch.float32, device=device)
            q_extended[cand_ids_tensor] = cand_qs_tensor

            # Log extracted policy: log π_ref + Q / τ, then normalize
            log_p_extracted = F.log_softmax(log_ref_extended + q_extended / temperature, dim=-1)

            # KL(π_extracted || π_student) - forward KL for mode-covering
            # With log_target=True: computes exp(log_target) * (log_target - input)
            # This encourages student to cover all modes of extracted policy
            kl = F.kl_div(log_p_student, log_p_extracted, reduction='sum', log_target=True)

            total_kl += kl
            total_positions += 1

            # Entropy for monitoring
            total_entropy_student += -(p_student * log_p_student).sum()
            total_entropy_extracted += -(log_p_extracted.exp() * log_p_extracted).sum()

            # Reference probability mass on candidates (should be high if min-p filtering worked)
            total_ref_mass += cand_ref_lps_tensor.exp().sum()

            # SFT loss: -log π_student(a_taken | s) - measures alignment with original behavior
            sft_loss = -log_p_student[taken_id]
            total_sft_loss += sft_loss.detach()

            # Reference SFT loss: -log π_ref(a_taken | s) - baseline for comparison
            try:
                taken_idx = cand_ids_list_local.index(taken_id)
                ref_sft_loss = -cand_ref_lps_tensor[taken_idx]
                total_ref_sft_loss += ref_sft_loss.detach()
            except ValueError:
                pass  # Taken action not in candidates, skip

            # --- Proper RL Advantage Computation ---
            # V(s_t) = Q(s_{t-1}, a_{t-1}) = prev_q_taken
            # For first position (pos_idx == 0), we don't have prev_q_taken, so advantage = 0

            if prev_q_taken is not None:
                v_state = prev_q_taken  # V(s_t) = Q-value of previous taken action

                # A(a) = Q(a) - V(s_t) for candidates, 0 for non-candidates
                advantages_extended = torch.zeros(vocab_size, dtype=torch.float32, device=device)
                advantages_extended[cand_ids_tensor] = cand_qs_tensor - v_state

                # Expected advantage for student and extracted policies
                p_extracted = log_p_extracted.exp()
                expected_advantage_student = (p_student * advantages_extended).sum()
                expected_advantage_extracted = (p_extracted * advantages_extended).sum()

                total_advantage_student += expected_advantage_student.detach()
                total_advantage_extracted += expected_advantage_extracted.detach()
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
            "entropy_student": zero.detach(),
            "entropy_extracted": zero.detach(),
            "advantage_student": zero.detach(),
            "advantage_extracted": zero.detach(),
            "v_state": zero.detach(),
            "ref_mass": zero.detach(),
            "sft_loss": zero.detach(),
            "ref_sft_loss": zero.detach(),
        }

    avg_kl = total_kl / total_positions

    metrics = {
        "num_positions": total_positions,
        "entropy_student": (total_entropy_student / total_positions).detach(),
        "entropy_extracted": (total_entropy_extracted / total_positions).detach(),
        "advantage_student": total_advantage_student / max(1, total_positions) if isinstance(total_advantage_student, torch.Tensor) else total_advantage_student / max(1, total_positions),
        "advantage_extracted": total_advantage_extracted / max(1, total_positions) if isinstance(total_advantage_extracted, torch.Tensor) else total_advantage_extracted / max(1, total_positions),
        "v_state": total_v_state / max(1, total_positions) if isinstance(total_v_state, torch.Tensor) else total_v_state / max(1, total_positions),
        "ref_mass": (total_ref_mass / total_positions).detach(),
        "sft_loss": (total_sft_loss / total_positions).item(),
        "ref_sft_loss": (total_ref_sft_loss / total_positions).item(),
    }

    return avg_kl, metrics


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
            name=f"policy_kl_{int(time.time())}",
            config={
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "compile_mode": compile_mode,
                "temperature": temperature,
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
    accum_entropy_student = 0.0
    accum_entropy_extracted = 0.0
    accum_advantage_student = 0.0
    accum_advantage_extracted = 0.0
    accum_v_state = 0.0
    accum_ref_mass = 0.0
    accum_sft_loss = 0.0
    accum_ref_sft_loss = 0.0
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
                loss, metrics = compute_kl_loss(model, batch, temperature=temperature)
                loss_scaled = loss / gradient_accumulation_steps

            scaler.scale(loss_scaled).backward()

            # Aggregate stats
            loss_val = loss.detach()
            if distributed:
                dist.all_reduce(loss_val)
                loss_val /= world

            accum_loss += loss_val.item()
            accum_entropy_student += metrics["entropy_student"].item() if isinstance(metrics["entropy_student"], torch.Tensor) else metrics["entropy_student"]
            accum_entropy_extracted += metrics["entropy_extracted"].item() if isinstance(metrics["entropy_extracted"], torch.Tensor) else metrics["entropy_extracted"]
            accum_advantage_student += metrics["advantage_student"].item() if isinstance(metrics["advantage_student"], torch.Tensor) else metrics["advantage_student"]
            accum_advantage_extracted += metrics["advantage_extracted"].item() if isinstance(metrics["advantage_extracted"], torch.Tensor) else metrics["advantage_extracted"]
            accum_v_state += metrics["v_state"].item() if isinstance(metrics["v_state"], torch.Tensor) else metrics["v_state"]
            accum_ref_mass += metrics["ref_mass"].item() if isinstance(metrics["ref_mass"], torch.Tensor) else metrics["ref_mass"]
            accum_sft_loss += metrics["sft_loss"].item() if isinstance(metrics["sft_loss"], torch.Tensor) else metrics["sft_loss"]
            accum_ref_sft_loss += metrics["ref_sft_loss"].item() if isinstance(metrics["ref_sft_loss"], torch.Tensor) else metrics["ref_sft_loss"]
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
                avg_ent_s = accum_entropy_student / max(1, accum_count)
                avg_ent_e = accum_entropy_extracted / max(1, accum_count)
                avg_adv_s = accum_advantage_student / max(1, accum_count)
                avg_adv_e = accum_advantage_extracted / max(1, accum_count)
                avg_v = accum_v_state / max(1, accum_count)
                avg_ref_mass = accum_ref_mass / max(1, accum_count)
                avg_sft = accum_sft_loss / max(1, accum_count)
                avg_ref_sft = accum_ref_sft_loss / max(1, accum_count)

                print(
                    f"[Step {global_step}] loss={avg_loss:.4f} sft={avg_sft:.3f} ref_sft={avg_ref_sft:.3f} "
                    f"adv_s={avg_adv_s:.4f} adv_e={avg_adv_e:.4f} V={avg_v:.3f} "
                    f"H_s={avg_ent_s:.3f} H_e={avg_ent_e:.3f} ref_mass={avg_ref_mass:.3f} lr={lr:.2e}"
                )

                if wandb_project:
                    wandb.log({
                        "train/kl_loss": avg_loss,
                        "train/sft_loss": avg_sft,
                        "train/ref_sft_loss": avg_ref_sft,
                        "train/advantage_student": avg_adv_s,
                        "train/advantage_extracted": avg_adv_e,
                        "train/v_state": avg_v,
                        "train/entropy_student": avg_ent_s,
                        "train/entropy_extracted": avg_ent_e,
                        "train/ref_mass": avg_ref_mass,
                        "lr": lr,
                        "step": global_step,
                    })

                accum_loss = 0.0
                accum_entropy_student = 0.0
                accum_entropy_extracted = 0.0
                accum_advantage_student = 0.0
                accum_advantage_extracted = 0.0
                accum_v_state = 0.0
                accum_ref_mass = 0.0
                accum_sft_loss = 0.0
                accum_ref_sft_loss = 0.0
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

    dataset = PolicyKLDataset(
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
        PolicyKLDataset.collate_fn,
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
        max_steps=cfg.max_steps,
        log_freq=cfg.log_freq,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train policy with KL loss against extracted policy")

    # Model and data
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/policy_kl")
    p.add_argument("--data_path", required=True, help="Path to labeled parquet file")

    # KL loss parameters
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature τ for extracted policy: π_ext ∝ π_ref * exp(Q/τ)")

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
    p.add_argument("--wandb_project", default="policy_kl")
    p.add_argument("--log_freq", type=int, default=10)

    return p.parse_args()


def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))


if __name__ == "__main__":
    main()
