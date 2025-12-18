#!/usr/bin/env python3
"""
Posterior Policy Distillation (PPD) policy training from pre-generated rollouts.

This script trains a causal LM policy using a Parquet dataset produced by:
  1) `src/data.py` (rollout generation; provides `input_ids` + `label_positions`)
  2) `src/label_ref_logprobs.py` (adds `ref_logprobs`)
  3) `src/label_in_context_critic.py` (adds `critic_p_correct`)

Key idea:
  - Treat stored `ref_logprobs` as log π_old(a_t | s_t) for the sampled tokens.
  - Compute log π_theta(a_t | s_t) via a forward pass of the trainable model.
  - Use a critic that predicts per-token success probability p_t ≈ P(G=1 | prefix up to token t).
  - Define a smoothed log-success "value":
        logP_t = log((p_t + tau) / (1 + tau))
  - Compute a success-shaping advantage via GAE on log-success deltas (gamma=1):
        delta_t = logP_t - logP_{t-1}          (for t >= 1)
        A_success[t] = delta_t + lambda * A_success[t+1]   (backwards; for t >= 1)
        A_success[0] = 0
  - Form the distillation-style reverse-KL advantage decomposition (per token):
        A = A_success - (log π_theta - log π_old)
    and DETACH it (stop-gradient).
  - Optimize the importance-sampling policy-gradient objective on fixed actions:
        L = -E_{x ~ π_old}[ (π_theta/π_old) * A_detached ]

Optional KL regularization:
  - Since only reference action logprobs are available (not full ref logits),
    we add a simple, always-nonnegative estimator:
        kl_term = exp(d) - 1 - d,  where d = logπ_theta - logπ_old
    whose expectation under π_old equals KL(π_old || π_theta).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower().strip()
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


@dataclass(frozen=True)
class Config:
    model_id: str
    data_path: str
    output_dir: str
    seed: int
    dp_size: int
    arrow_batch_size: int
    micro_batch_size: int
    grad_accum_steps: int
    num_epochs: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_ratio: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    dtype: torch.dtype
    attn_implementation: str
    compile_mode: str
    gradient_checkpointing: bool
    ppo_clip_range: float
    adv_smoothing_tau: float
    gae_lambda: float
    adv_clip: float
    kl_coef: float
    log_every: int
    save_every: int
    master_port: str
    wandb_project: str
    wandb_name: str
    columns_input_ids: str
    columns_label_positions: str
    columns_ref_logprobs: str
    columns_p_correct: str


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data-path", required=True, help="Parquet file with rollouts + ref logprobs + critic p_correct.")
    p.add_argument("--output-dir", default="models/policy_ppd")
    p.add_argument("--seed", type=int, default=42)

    # Data loading / DDP
    p.add_argument("--dp-size", type=int, default=0, help="Number of GPU processes (default: all visible GPUs).")
    p.add_argument("--arrow-batch-size", type=int, default=1024, help="Rows per parquet read batch per worker.")
    p.add_argument("--micro-batch-size", type=int, default=1, help="Per-process microbatch size.")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")

    # Training schedule
    p.add_argument("--num-epochs", type=int, default=1, help="Number of dataset passes (best-effort under DDP).")
    p.add_argument("--max-steps", type=int, default=-1, help="If >0, stop after this many optimizer steps.")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--min-learning-rate", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Model runtime
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--compile-mode", default="none", choices=["none", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")

    # Objective knobs
    p.add_argument(
        "--ppo-clip-range",
        type=float,
        default=0.0,
        help="If >0, clamp probability ratio to [1-eps, 1+eps] for stability (0 disables).",
    )
    p.add_argument("--adv-smoothing-tau", type=float, default=0.1)
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda (gamma is fixed to 1).")
    p.add_argument("--adv-clip", type=float, default=0.0, help="If >0, clamp SUCCESS advantages to [-adv-clip, adv-clip].")
    p.add_argument("--kl-coef", type=float, default=0.0, help="Optional KL regularization coefficient (0 disables).")

    # Logging / saving
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0, help="If >0, save every N optimizer steps on rank 0.")
    p.add_argument("--master-port", default="29500")
    p.add_argument("--wandb-project", default="", help="If set, log to Weights & Biases (rank 0 only).")
    p.add_argument("--wandb-name", default="", help="Optional W&B run name.")

    # Column names (keep flexible)
    p.add_argument("--col-input-ids", default="input_ids")
    p.add_argument("--col-label-positions", default="label_positions")
    p.add_argument("--col-ref-logprobs", default="ref_logprobs")
    p.add_argument("--col-p-correct", default="critic_p_correct")

    args = p.parse_args()

    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be >= 1")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.arrow_batch_size <= 0:
        raise ValueError("--arrow-batch-size must be >= 1")
    if args.ppo_clip_range < 0:
        raise ValueError("--ppo-clip-range must be >= 0")
    if args.adv_smoothing_tau <= 0:
        raise ValueError("--adv-smoothing-tau must be > 0")
    if not (0.0 <= float(args.gae_lambda) <= 1.0):
        raise ValueError("--gae-lambda must be in [0, 1]")

    return Config(
        model_id=str(args.model_id),
        data_path=str(args.data_path),
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        dp_size=int(args.dp_size),
        arrow_batch_size=int(args.arrow_batch_size),
        micro_batch_size=int(args.micro_batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        num_epochs=int(args.num_epochs),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        min_learning_rate=float(args.min_learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        grad_clip=float(args.grad_clip),
        dtype=_dtype_from_str(args.dtype),
        attn_implementation=str(args.attn_implementation),
        compile_mode=str(args.compile_mode),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        ppo_clip_range=float(args.ppo_clip_range),
        adv_smoothing_tau=float(args.adv_smoothing_tau),
        gae_lambda=float(args.gae_lambda),
        adv_clip=float(args.adv_clip),
        kl_coef=float(args.kl_coef),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        master_port=str(args.master_port),
        wandb_project=str(args.wandb_project),
        wandb_name=str(args.wandb_name),
        columns_input_ids=str(args.col_input_ids),
        columns_label_positions=str(args.col_label_positions),
        columns_ref_logprobs=str(args.col_ref_logprobs),
        columns_p_correct=str(args.col_p_correct),
    )


def _iter_assigned_batches(
    pf: pq.ParquetFile, *, rank: int, world: int, batch_size: int
) -> Iterable[pa.RecordBatch]:
    """
    Yield record batches assigned to this worker.
    Partition by parquet row groups when possible; otherwise fall back to row-wise striding.
    """
    if pf.num_row_groups >= world:
        row_groups = [rg for rg in range(pf.num_row_groups) if (rg % world) == rank]
        yield from pf.iter_batches(batch_size=batch_size, row_groups=row_groups)
        return

    # Fallback: row-wise striding across full file.
    row_offset = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n = batch.num_rows
        take_idx = [i for i in range(n) if ((row_offset + i) % world) == rank]
        row_offset += n
        if not take_idx:
            continue
        yield batch.take(pa.array(take_idx, type=pa.int32()))


def _as_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


class ParquetRolloutDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        rank: int,
        world_size: int,
        seed: int,
        arrow_batch_size: int,
        shuffle: bool,
        col_input_ids: str,
        col_label_positions: str,
        col_ref_logprobs: str,
        col_p_correct: str,
    ) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.arrow_batch_size = int(arrow_batch_size)
        self.shuffle = bool(shuffle)
        self.col_input_ids = str(col_input_ids)
        self.col_label_positions = str(col_label_positions)
        self.col_ref_logprobs = str(col_ref_logprobs)
        self.col_p_correct = str(col_p_correct)
        self.epoch = 0

        pf = pq.ParquetFile(self.parquet_path)
        names = set(pf.schema_arrow.names)
        missing = [
            c
            for c in [self.col_input_ids, self.col_label_positions, self.col_ref_logprobs, self.col_p_correct]
            if c not in names
        ]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}. Present columns: {sorted(names)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        schema = pf.schema_arrow
        idx_input = schema.get_field_index(self.col_input_ids)
        idx_pos = schema.get_field_index(self.col_label_positions)
        idx_ref = schema.get_field_index(self.col_ref_logprobs)
        idx_p = schema.get_field_index(self.col_p_correct)

        rng = random.Random(self.seed + 1_000_003 * self.epoch + 97 * self.rank)

        for rb in _iter_assigned_batches(pf, rank=self.rank, world=self.world_size, batch_size=self.arrow_batch_size):
            input_ids_col = rb.column(idx_input).to_pylist()
            label_pos_col = rb.column(idx_pos).to_pylist()
            ref_lp_col = rb.column(idx_ref).to_pylist()
            p_corr_col = rb.column(idx_p).to_pylist()

            n = rb.num_rows
            order = list(range(n))
            if self.shuffle:
                rng.shuffle(order)

            for i in order:
                yield {
                    "input_ids": _as_list(input_ids_col[i]),
                    "label_positions": _as_list(label_pos_col[i]),
                    "ref_logprobs": _as_list(ref_lp_col[i]),
                    "critic_p_correct": _as_list(p_corr_col[i]),
                }


def _compute_advantages(
    p_correct: Sequence[float], *, tau: float, gae_lambda: float, adv_clip: float
) -> List[float]:
    """
    Compute per-token SUCCESS advantages from critic p_correct values.

    Define logP_t = log((p_t + tau)/(1+tau)).
    With gamma=1, deltas are:
        delta_t = logP_t - logP_{t-1}  (t>=1)
    Then GAE:
        A_success[t] = delta_t + lambda * A_success[t+1]  (backwards; t>=1)
        A_success[0] = 0

    NOTE: This function returns ONLY A_success.
    The full distillation-style advantage is formed later as:
        A = A_success - (logπ_theta - logπ_old)
    and is detached before optimization.
    """
    values: List[float] = []
    for p in p_correct:
        try:
            p_f = float(p)
        except Exception:
            p_f = float("nan")

        if not (p_f == p_f):  # NaN
            values.append(float("nan"))
            continue

        if p_f < 0.0:
            p_f = 0.0
        elif p_f > 1.0:
            p_f = 1.0

        values.append(math.log((p_f + tau) / (1.0 + tau)))

    n = len(values)
    if n == 0:
        return []

    out: List[float] = [float("nan")] * n
    out[0] = 0.0

    gae = 0.0
    lam = float(gae_lambda)

    for t in range(n - 1, 0, -1):
        v_t = values[t]
        v_prev = values[t - 1]
        if not (math.isfinite(v_t) and math.isfinite(v_prev)):
            out[t] = float("nan")
            gae = 0.0
            continue

        delta = v_t - v_prev
        gae = delta + lam * gae
        out[t] = float(gae)

    if adv_clip and adv_clip > 0.0:
        clip = float(adv_clip)
        for i, a in enumerate(out):
            if not math.isfinite(a):
                continue
            if a > clip:
                out[i] = clip
            elif a < -clip:
                out[i] = -clip

    return out


def _make_collate_fn(*, pad_token_id: int, tau: float, gae_lambda: float, adv_clip: float):
    def collate(rows: List[dict]) -> dict:
        if not rows:
            return {}

        max_len = max(len(r["input_ids"]) for r in rows)
        input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)

        label_positions: List[List[int]] = []
        ref_logprobs: List[List[float]] = []
        advantages: List[List[float]] = []
        num_tokens = 0

        for i, r in enumerate(rows):
            ids = _as_list(r.get("input_ids"))
            if ids:
                t = torch.tensor(ids, dtype=torch.long)
                input_ids[i, : t.numel()] = t
                attention_mask[i, : t.numel()] = 1

            pos_raw = _as_list(r.get("label_positions"))
            pos: List[int] = []
            for x in pos_raw:
                try:
                    pos.append(int(x))
                except Exception:
                    pos.append(-1)

            old_raw = _as_list(r.get("ref_logprobs"))
            old_lp: List[float] = []
            for x in old_raw:
                try:
                    old_lp.append(float(x))
                except Exception:
                    old_lp.append(float("nan"))

            p_corr = _as_list(r.get("critic_p_correct"))

            # Keep alignment by truncating all per-output lists to a common length.
            L = min(len(pos), len(old_lp), len(p_corr))
            pos = pos[:L]
            old_lp = old_lp[:L]
            p_corr = p_corr[:L]

            label_positions.append(pos)
            ref_logprobs.append(old_lp)
            adv = _compute_advantages(p_corr, tau=tau, gae_lambda=gae_lambda, adv_clip=adv_clip)[:L]
            advantages.append(adv)

            # Count trainable tokens (matches the filtering logic in _compute_policy_loss, excluding new logprobs).
            seq_len = len(ids)
            for t in range(L):
                a = float(adv[t])
                if not math.isfinite(a):
                    continue
                o = float(old_lp[t])
                if not math.isfinite(o):
                    continue
                p = int(pos[t])
                if p <= 0 or p >= seq_len:
                    continue
                num_tokens += 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_positions": label_positions,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,  # A_success only
            "num_tokens": int(num_tokens),
        }

    return collate


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    return tgt


def _lr_schedule(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if step >= total_steps:
        return min_lr
    if total_steps <= warmup_steps:
        return min_lr
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + cosine * (base_lr - min_lr))


def _compute_policy_loss(
    *,
    token_logprobs: torch.Tensor,  # [B, S-1]
    attention_mask: torch.Tensor,  # [B, S]
    label_positions: Sequence[Sequence[int]],
    ref_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[Sequence[float]],  # A_success only
    ratio_clip_range: float,  # if >0, clamp ratio to [1-eps,1+eps]
    kl_coef: float,
) -> Tuple[torch.Tensor, dict]:
    device = token_logprobs.device
    bsz = token_logprobs.shape[0]

    flat_b: List[int] = []
    flat_s: List[int] = []
    flat_old: List[float] = []
    flat_adv_success: List[float] = []

    lengths = attention_mask.sum(dim=1).tolist()

    for i in range(bsz):
        seq_len = int(lengths[i])
        pos = list(label_positions[i] or [])
        old = list(ref_logprobs[i] or [])
        adv = list(advantages[i] or [])
        L = min(len(pos), len(old), len(adv))

        for t in range(L):
            a_s = float(adv[t])
            if not (a_s == a_s):  # NaN
                continue
            o = float(old[t])
            if not math.isfinite(o):
                continue

            p = int(pos[t])
            if p <= 0 or p >= seq_len:
                continue
            if p - 1 >= token_logprobs.shape[1]:
                continue

            flat_b.append(i)
            flat_s.append(p - 1)
            flat_old.append(o)
            flat_adv_success.append(a_s)

    if not flat_b:
        zero = token_logprobs.sum() * 0.0
        z = zero.detach()
        return zero, {
            "policy_loss": z,
            "kl_loss": z,
            "approx_kl": z,
            "clip_frac": z,
            "tokens": 0,
            "adv_mean": z,
            "adv_abs_mean": z,
            "ratio_mean": z,
            "ratio_used_mean": z,
        }

    b_idx = torch.tensor(flat_b, device=device, dtype=torch.long)
    s_idx = torch.tensor(flat_s, device=device, dtype=torch.long)

    # log π_theta(a|s)
    new_lp = token_logprobs[b_idx, s_idx].to(torch.float32)
    # log π_old(a|s)
    old_lp = torch.tensor(flat_old, device=device, dtype=torch.float32)
    # A_success (from critic + GAE)
    adv_success = torch.tensor(flat_adv_success, device=device, dtype=torch.float32)

    # log ratio and ratio
    log_ratio = new_lp - old_lp
    ratio = torch.exp(log_ratio)

    eps = float(ratio_clip_range)
    if eps and eps > 0.0:
        ratio_used = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        clip_frac = ((ratio > (1.0 + eps)) | (ratio < (1.0 - eps))).to(torch.float32).mean()
    else:
        ratio_used = ratio
        clip_frac = ratio_used.new_tensor(0.0)

    # Distillation-style reverse-KL advantage decomposition (DETACHED):
    # A = A_success - (logπ_theta - logπ_old)
    adv_final = (adv_success - log_ratio).detach()

    # Importance-sampling policy gradient loss:
    # loss = -E_q[ (p/q) * A_detached ]
    policy_loss = -(ratio_used * adv_final).mean()

    # Optional KL(π_old || π_theta) estimator from sampled actions:
    if kl_coef and kl_coef > 0.0:
        kl_term = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = float(kl_coef) * kl_term.mean()
    else:
        kl_loss = policy_loss * 0.0

    total = policy_loss + kl_loss

    approx_kl = (old_lp - new_lp).mean()

    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_frac": clip_frac.detach(),
        "tokens": int(new_lp.numel()),
        "adv_mean": adv_final.mean().detach(),
        "adv_abs_mean": adv_final.abs().mean().detach(),
        "ratio_mean": ratio.mean().detach(),
        "ratio_used_mean": ratio_used.mean().detach(),
    }
    return total, metrics


def _train_worker(local_rank: int, world_size: int, cfg: Config) -> None:
    os.environ.update(
        WORLD_SIZE=str(world_size),
        RANK=str(local_rank),
        LOCAL_RANK=str(local_rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(cfg.master_port),
    )

    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")

    rank = int(os.environ.get("RANK", "0"))
    master = rank == 0

    # Seed
    torch.manual_seed(int(cfg.seed) + rank)
    random.seed(int(cfg.seed) + rank)

    if master:
        os.makedirs(cfg.output_dir, exist_ok=True)

    # Tokenizer / Model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot pad safely.")

    attn_impl = cfg.attn_implementation
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    if cfg.compile_mode != "none":
        model = torch.compile(model, mode=cfg.compile_mode)

    # Dataset / DataLoader
    dataset = ParquetRolloutDataset(
        cfg.data_path,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        arrow_batch_size=cfg.arrow_batch_size,
        shuffle=True,
        col_input_ids=cfg.columns_input_ids,
        col_label_positions=cfg.columns_label_positions,
        col_ref_logprobs=cfg.columns_ref_logprobs,
        col_p_correct=cfg.columns_p_correct,
    )

    collate_fn = _make_collate_fn(
        pad_token_id=int(tokenizer.pad_token_id),
        tau=float(cfg.adv_smoothing_tau),
        gae_lambda=float(cfg.gae_lambda),
        adv_clip=float(cfg.adv_clip),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # Optimizer
    params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    total_steps = int(cfg.max_steps) if cfg.max_steps and cfg.max_steps > 0 else 0
    if total_steps == 0 and cfg.num_epochs > 0:
        # Best-effort estimate for scheduling only (training loop still uses sync StopIteration).
        try:
            pf = pq.ParquetFile(cfg.data_path)
            total_rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        except Exception:
            total_rows = 0

        if total_rows > 0:
            est_microbatches = (total_rows // max(1, world_size * cfg.micro_batch_size)) * cfg.num_epochs
            est_updates = max(1, est_microbatches // max(1, cfg.grad_accum_steps))
            total_steps = int(est_updates)

    warmup_steps = int(float(cfg.warmup_ratio) * float(total_steps)) if total_steps > 0 else 0

    # Optional W&B
    wb = None
    if master and cfg.wandb_project:
        try:
            import wandb  # type: ignore

            run_name = cfg.wandb_name or f"pg_{int(time.time())}"
            wb = wandb
            wb.init(
                project=cfg.wandb_project,
                name=run_name,
                config={
                    "model_id": cfg.model_id,
                    "data_path": cfg.data_path,
                    "ratio_clip_range": cfg.ppo_clip_range,
                    "adv_smoothing_tau": cfg.adv_smoothing_tau,
                    "gae_lambda": cfg.gae_lambda,
                    "adv_clip": cfg.adv_clip,
                    "kl_coef": cfg.kl_coef,
                    "micro_batch_size": cfg.micro_batch_size,
                    "grad_accum_steps": cfg.grad_accum_steps,
                    "learning_rate": cfg.learning_rate,
                    "min_learning_rate": cfg.min_learning_rate,
                    "warmup_ratio": cfg.warmup_ratio,
                    "weight_decay": cfg.weight_decay,
                    "dtype": str(cfg.dtype),
                    "attn_implementation": cfg.attn_implementation,
                    "compile_mode": cfg.compile_mode,
                    "gradient_checkpointing": cfg.gradient_checkpointing,
                    "world_size": world_size,
                    "max_steps": cfg.max_steps,
                    "num_epochs": cfg.num_epochs,
                },
            )
        except Exception:
            wb = None

    if device.type == "cuda" and cfg.dtype in {torch.float16, torch.bfloat16}:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=cfg.dtype)
    else:
        from contextlib import nullcontext

        autocast_ctx = nullcontext()

    global_step = 0  # optimizer steps
    accum = 0
    t0 = time.perf_counter()
    step_tokens_local = 0
    step_policy_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_kl_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_approx_kl_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_clip_frac_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_adv_mean_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_adv_abs_mean_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_ratio_mean_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_ratio_used_mean_sum = torch.zeros((), device=device, dtype=torch.float32)

    def save_checkpoint(step: int):
        if not master:
            return
        out_dir = cfg.output_dir
        tgt = _unwrap_model(model)
        tgt.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        if master:
            with open(os.path.join(out_dir, "training_state.txt"), "w", encoding="utf-8") as f:
                f.write(f"global_step={step}\n")

    try:
        for epoch in range(max(1, cfg.num_epochs)):
            dataset.set_epoch(epoch)
            it = iter(loader)

            while True:
                batch = None
                has = 1
                try:
                    batch = next(it)
                except StopIteration:
                    has = 0

                if distributed:
                    has_t = torch.tensor(has, device=device, dtype=torch.int32)
                    dist.all_reduce(has_t, op=dist.ReduceOp.MIN)
                    if int(has_t.item()) == 0:
                        break
                    if not has:
                        continue
                else:
                    if not has:
                        break

                if not distributed and int(batch.get("num_tokens", 0)) == 0:
                    continue

                accum += 1

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                with autocast_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    logits = outputs.logits  # [B, S, V]

                    shift_logits = logits[:, :-1, :]
                    shift_labels = input_ids[:, 1:]
                    logit_target = shift_logits.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    logsumexp = torch.logsumexp(shift_logits, dim=-1)
                    token_logprobs = logit_target - logsumexp  # [B, S-1]

                    loss, metrics = _compute_policy_loss(
                        token_logprobs=token_logprobs,
                        attention_mask=attention_mask,
                        label_positions=batch["label_positions"],
                        ref_logprobs=batch["ref_logprobs"],
                        advantages=batch["advantages"],
                        ratio_clip_range=cfg.ppo_clip_range,
                        kl_coef=cfg.kl_coef,
                    )
                    loss = loss / float(cfg.grad_accum_steps)

                loss.backward()

                tok = int(metrics["tokens"]) if isinstance(metrics["tokens"], int) else int(metrics["tokens"].item())
                step_tokens_local += tok
                tok_f = float(tok)
                step_policy_loss_sum = step_policy_loss_sum + metrics["policy_loss"].to(torch.float32) * tok_f
                step_kl_loss_sum = step_kl_loss_sum + metrics["kl_loss"].to(torch.float32) * tok_f
                step_approx_kl_sum = step_approx_kl_sum + metrics["approx_kl"].to(torch.float32) * tok_f
                step_clip_frac_sum = step_clip_frac_sum + metrics["clip_frac"].to(torch.float32) * tok_f
                step_adv_mean_sum = step_adv_mean_sum + metrics["adv_mean"].to(torch.float32) * tok_f
                step_adv_abs_mean_sum = step_adv_abs_mean_sum + metrics["adv_abs_mean"].to(torch.float32) * tok_f
                step_ratio_mean_sum = step_ratio_mean_sum + metrics["ratio_mean"].to(torch.float32) * tok_f
                step_ratio_used_mean_sum = step_ratio_used_mean_sum + metrics["ratio_used_mean"].to(torch.float32) * tok_f

                update = (accum % cfg.grad_accum_steps) == 0
                if update:
                    step_tokens_t = torch.tensor(step_tokens_local, device=device, dtype=torch.long)
                    if distributed:
                        dist.all_reduce(step_tokens_t, op=dist.ReduceOp.SUM)
                    step_tokens_global = int(step_tokens_t.item())

                    if step_tokens_global == 0:
                        optimizer.zero_grad(set_to_none=True)
                        step_tokens_local = 0
                        step_policy_loss_sum.zero_()
                        step_kl_loss_sum.zero_()
                        step_approx_kl_sum.zero_()
                        step_clip_frac_sum.zero_()
                        step_adv_mean_sum.zero_()
                        step_adv_abs_mean_sum.zero_()
                        step_ratio_mean_sum.zero_()
                        step_ratio_used_mean_sum.zero_()
                        continue

                    if cfg.grad_clip and cfg.grad_clip > 0:
                        clip_grad_norm_(params, cfg.grad_clip)

                    global_step += 1
                    lr = _lr_schedule(
                        global_step,
                        total_steps=total_steps,
                        base_lr=cfg.learning_rate,
                        min_lr=cfg.min_learning_rate,
                        warmup_steps=warmup_steps,
                    )
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    do_log = (global_step % max(1, cfg.log_every) == 0)
                    if do_log:
                        sums = torch.stack(
                            [
                                step_policy_loss_sum,
                                step_kl_loss_sum,
                                step_approx_kl_sum,
                                step_clip_frac_sum,
                                step_adv_mean_sum,
                                step_adv_abs_mean_sum,
                                step_ratio_mean_sum,
                                step_ratio_used_mean_sum,
                            ]
                        )
                        if distributed:
                            dist.all_reduce(sums, op=dist.ReduceOp.SUM)

                        denom = float(max(1, step_tokens_global))
                        mean_policy_loss = sums[0] / denom
                        mean_kl_loss = sums[1] / denom
                        mean_approx_kl = sums[2] / denom
                        mean_clip_frac = sums[3] / denom
                        mean_adv = sums[4] / denom
                        mean_adv_abs = sums[5] / denom
                        mean_ratio = sums[6] / denom
                        mean_ratio_used = sums[7] / denom

                    if master and do_log:
                        elapsed = time.perf_counter() - t0
                        log_dict = {
                            "train/step": global_step,
                            "train/lr": lr,
                            "train/tokens": int(step_tokens_global),
                            "train/policy_loss": float(mean_policy_loss.item()),
                            "train/kl_loss": float(mean_kl_loss.item()),
                            "train/approx_kl": float(mean_approx_kl.item()),
                            "train/clip_frac": float(mean_clip_frac.item()),
                            "train/adv_mean": float(mean_adv.item()),
                            "train/adv_abs_mean": float(mean_adv_abs.item()),
                            "train/ratio_mean": float(mean_ratio.item()),
                            "train/ratio_used_mean": float(mean_ratio_used.item()),
                            "train/elapsed_s": elapsed,
                        }
                        print(
                            f"[step {global_step}] loss={log_dict['train/policy_loss']:.6f} "
                            f"kl={log_dict['train/approx_kl']:.6f} clip={log_dict['train/clip_frac']:.3f} "
                            f"tokens={int(step_tokens_global)} lr={lr:.3e}",
                            flush=True,
                        )
                        if wb is not None:
                            wb.log(log_dict, step=global_step)

                    if master and cfg.save_every and cfg.save_every > 0 and (global_step % cfg.save_every == 0):
                        save_checkpoint(global_step)

                    step_tokens_local = 0
                    step_policy_loss_sum.zero_()
                    step_kl_loss_sum.zero_()
                    step_approx_kl_sum.zero_()
                    step_clip_frac_sum.zero_()
                    step_adv_mean_sum.zero_()
                    step_adv_abs_mean_sum.zero_()
                    step_ratio_mean_sum.zero_()
                    step_ratio_used_mean_sum.zero_()

                    if cfg.max_steps and cfg.max_steps > 0 and global_step >= cfg.max_steps:
                        break

            if cfg.max_steps and cfg.max_steps > 0 and global_step >= cfg.max_steps:
                break

        if master:
            save_checkpoint(global_step)
        if wb is not None and master:
            wb.finish()

    finally:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


def main() -> None:
    cfg = parse_args()

    ngpus = torch.cuda.device_count()
    if cfg.dp_size and cfg.dp_size > 0:
        world = int(cfg.dp_size)
    else:
        world = max(1, ngpus)

    if ngpus > 0 and world > ngpus:
        raise RuntimeError(f"Requested dp_size={world}, but only {ngpus} CUDA devices are visible.")
    if ngpus == 0 and world != 1:
        raise RuntimeError("No CUDA devices visible; run with --dp-size 1 for CPU training.")

    mp.spawn(_train_worker, nprocs=world, args=(world, cfg))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        raise
