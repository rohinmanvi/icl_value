#!/usr/bin/env python3
"""
Offline GRPO policy training from pre-generated rollouts.

This script mirrors `src/train_policy_ppo.py` but implements DeepSeek-style GRPO:
  - Group rollouts by `prompt_idx`.
  - Optionally subsample K rollouts per prompt group (`--group-size`).
  - Use a sequence-level reward (default column: `correct`) and compute the
    standardized group advantage:
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
  - Optimize a PPO-style clipped objective using stored `ref_logprobs` as
    log π_old(a_t | s_t) for the sampled output tokens.

This trainer does not require critic estimates (e.g., `critic_p_correct`).
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
    group_size: int
    kl_coef: float
    log_every: int
    save_every: int
    master_port: str
    wandb_project: str
    wandb_name: str
    columns_prompt_idx: str
    columns_input_ids: str
    columns_label_positions: str
    columns_ref_logprobs: str
    columns_reward: str


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data-path", required=True, help="Parquet file with rollouts + ref logprobs + reward column.")
    p.add_argument("--output-dir", default="models/policy_grpo")
    p.add_argument("--seed", type=int, default=42)

    # Data loading / DDP
    p.add_argument("--dp-size", type=int, default=0, help="Number of GPU processes (default: all visible GPUs).")
    p.add_argument("--arrow-batch-size", type=int, default=1024, help="Rows per parquet read batch per worker.")
    p.add_argument("--micro-batch-size", type=int, default=1, help="Per-process microbatch size.")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")

    # Training schedule
    p.add_argument("--num-epochs", type=int, default=1, help="Number of dataset passes (best-effort under DDP).")
    p.add_argument("--max-steps", type=int, default=-1, help="If >0, stop after this many optimizer steps.")
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--min-learning-rate", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Model runtime
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--compile-mode", default="none", choices=["none", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")

    # GRPO / PPO objective
    p.add_argument("--ppo-clip-range", type=float, default=0.2)
    p.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Subsample K rollouts per prompt group (-1 uses all available).",
    )
    p.add_argument("--kl-coef", type=float, default=0.0, help="Optional KL regularization coefficient (0 disables).")

    # Logging / saving
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0, help="If >0, save every N optimizer steps on rank 0.")
    p.add_argument("--master-port", default="29500")
    p.add_argument("--wandb-project", default="", help="If set, log to Weights & Biases (rank 0 only).")
    p.add_argument("--wandb-name", default="", help="Optional W&B run name.")

    # Column names (keep flexible)
    p.add_argument("--col-prompt-idx", default="prompt_idx")
    p.add_argument("--col-input-ids", default="input_ids")
    p.add_argument("--col-label-positions", default="label_positions")
    p.add_argument("--col-ref-logprobs", default="ref_logprobs")
    p.add_argument("--col-reward", default="correct", help="Scalar reward column used for GRPO (default: correct).")

    args = p.parse_args()

    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be >= 1")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.arrow_batch_size <= 0:
        raise ValueError("--arrow-batch-size must be >= 1")
    if args.ppo_clip_range <= 0:
        raise ValueError("--ppo-clip-range must be > 0")
    if int(args.group_size) == 0 or int(args.group_size) < -1:
        raise ValueError("--group-size must be -1 (all) or >= 1")

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
        group_size=int(args.group_size),
        kl_coef=float(args.kl_coef),
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        master_port=str(args.master_port),
        wandb_project=str(args.wandb_project),
        wandb_name=str(args.wandb_name),
        columns_prompt_idx=str(args.col_prompt_idx),
        columns_input_ids=str(args.col_input_ids),
        columns_label_positions=str(args.col_label_positions),
        columns_ref_logprobs=str(args.col_ref_logprobs),
        columns_reward=str(args.col_reward),
    )


def _as_list(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _row_group_range(num_row_groups: int, *, rank: int, world: int) -> Tuple[int, int]:
    start = (num_row_groups * rank) // world
    end = (num_row_groups * (rank + 1)) // world
    return start, end


def _iter_row_groups(pf: pq.ParquetFile, *, start_rg: int, batch_size: int) -> Iterable[Tuple[int, pa.RecordBatch]]:
    for rg in range(start_rg, pf.num_row_groups):
        for rb in pf.iter_batches(row_groups=[rg], batch_size=batch_size):
            yield rg, rb


def _standardized_advantages(rewards: Sequence[float], *, eps: float) -> List[float]:
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(float(r) for r in rewards) / float(n)
    var = sum((float(r) - mean) ** 2 for r in rewards) / float(n)
    std = math.sqrt(max(0.0, var))
    denom = std + float(eps)
    return [(float(r) - mean) / denom for r in rewards]


class ParquetGRPODataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        rank: int,
        world_size: int,
        seed: int,
        arrow_batch_size: int,
        group_size: int,
        col_prompt_idx: str,
        col_input_ids: str,
        col_label_positions: str,
        col_ref_logprobs: str,
        col_reward: str,
    ) -> None:
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.arrow_batch_size = int(arrow_batch_size)
        self.group_size = int(group_size)
        self.col_prompt_idx = str(col_prompt_idx)
        self.col_input_ids = str(col_input_ids)
        self.col_label_positions = str(col_label_positions)
        self.col_ref_logprobs = str(col_ref_logprobs)
        self.col_reward = str(col_reward)
        self.epoch = 0

        pf = pq.ParquetFile(self.parquet_path)
        names = set(pf.schema_arrow.names)
        required = [
            self.col_prompt_idx,
            self.col_input_ids,
            self.col_label_positions,
            self.col_ref_logprobs,
            self.col_reward,
        ]
        missing = [c for c in required if c not in names]
        if missing:
            raise ValueError(f"Parquet missing required columns: {missing}. Present columns: {sorted(names)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        schema = pf.schema_arrow
        idx_prompt = schema.get_field_index(self.col_prompt_idx)
        idx_input = schema.get_field_index(self.col_input_ids)
        idx_pos = schema.get_field_index(self.col_label_positions)
        idx_ref = schema.get_field_index(self.col_ref_logprobs)
        idx_reward = schema.get_field_index(self.col_reward)

        # If there are too few row groups to split evenly, fall back to scanning the full
        # file on every rank and assigning prompt groups by prompt_idx % world_size.
        use_full_scan = pf.num_row_groups < self.world_size
        if use_full_scan:
            start_rg, end_rg = 0, pf.num_row_groups
            read_start_rg = 0
        else:
            start_rg, end_rg = _row_group_range(pf.num_row_groups, rank=self.rank, world=self.world_size)
            read_start_rg = max(0, start_rg - 1)

        adv_eps = 1e-8

        current_prompt_idx = None
        current_group_start_rg = None
        current_batches: List[pa.RecordBatch] = []

        def flush_group():
            nonlocal current_prompt_idx, current_group_start_rg, current_batches
            if current_prompt_idx is None or not current_batches:
                current_prompt_idx = None
                current_group_start_rg = None
                current_batches = []
                return

            pid = current_prompt_idx

            if use_full_scan:
                try:
                    pid_int = int(pid) if pid is not None else -1
                except Exception:
                    pid_int = int(abs(hash(str(pid))) % (2**31))
                should_process = (pid_int % self.world_size) == self.rank
            else:
                should_process = (current_group_start_rg is not None) and (start_rg <= current_group_start_rg < end_rg)

            if should_process:
                table = pa.Table.from_batches(current_batches, schema=schema)

                input_ids_col = table.column(idx_input).to_pylist()
                label_pos_col = table.column(idx_pos).to_pylist()
                ref_lp_col = table.column(idx_ref).to_pylist()
                reward_col = table.column(idx_reward).to_pylist()

                n = table.num_rows
                order = list(range(n))

                # Deterministic subsampling per prompt group (varies per epoch).
                try:
                    pid_int = int(pid) if pid is not None else -1
                except Exception:
                    pid_int = int(abs(hash(str(pid))) % (2**31))
                rng = random.Random(int(self.seed) + 1_000_003 * self.epoch + 1009 * pid_int)
                if self.group_size > 0 and n > self.group_size:
                    order = rng.sample(order, k=self.group_size)
                rng.shuffle(order)

                rewards_sel: List[float] = []
                for row_idx in order:
                    r = reward_col[row_idx]
                    try:
                        r_f = float(r) if r is not None else 0.0
                    except Exception:
                        r_f = 0.0
                    if not math.isfinite(r_f):
                        r_f = 0.0
                    rewards_sel.append(r_f)

                advs = _standardized_advantages(rewards_sel, eps=adv_eps)

                for row_idx, reward, adv in zip(order, rewards_sel, advs):
                    yield {
                        "prompt_idx": pid_int,
                        "input_ids": _as_list(input_ids_col[row_idx]),
                        "label_positions": _as_list(label_pos_col[row_idx]),
                        "ref_logprobs": _as_list(ref_lp_col[row_idx]),
                        "reward": float(reward),
                        "advantage": float(adv),
                    }

            current_prompt_idx = None
            current_group_start_rg = None
            current_batches = []

        past_primary = False

        for rg, rb in _iter_row_groups(pf, start_rg=read_start_rg, batch_size=self.arrow_batch_size):
            if not use_full_scan and rg >= end_rg:
                past_primary = True

            prompt_idx_col = rb.column(idx_prompt).to_pylist()
            n_rows = len(prompt_idx_col)
            i = 0
            while i < n_rows:
                pid = prompt_idx_col[i]
                j = i + 1
                while j < n_rows and prompt_idx_col[j] == pid:
                    j += 1
                chunk = rb.slice(i, j - i)

                if current_prompt_idx is None:
                    current_prompt_idx = pid
                    current_group_start_rg = rg
                    current_batches = [chunk]
                elif pid == current_prompt_idx:
                    current_batches.append(chunk)
                else:
                    yield from flush_group()
                    current_prompt_idx = pid
                    current_group_start_rg = rg
                    current_batches = [chunk]

                i = j

            if not use_full_scan and past_primary and current_prompt_idx is None:
                break

        yield from flush_group()


def _make_collate_fn(*, pad_token_id: int):
    def collate(rows: List[dict]) -> dict:
        if not rows:
            return {}

        max_len = max(len(r.get("input_ids") or []) for r in rows)
        input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)

        label_positions: List[List[int]] = []
        ref_logprobs: List[List[float]] = []
        advantages: List[List[float]] = []

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

            adv_scalar = float(r.get("advantage") or 0.0)

            L = min(len(pos), len(old_lp))
            pos = pos[:L]
            old_lp = old_lp[:L]

            label_positions.append(pos)
            ref_logprobs.append(old_lp)
            advantages.append([adv_scalar] * L)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_positions": label_positions,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
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
    advantages: Sequence[Sequence[float]],
    ppo_clip_range: float,
    kl_coef: float,
) -> Tuple[torch.Tensor, dict]:
    device = token_logprobs.device
    bsz = token_logprobs.shape[0]

    flat_b: List[int] = []
    flat_s: List[int] = []
    flat_old: List[float] = []
    flat_adv: List[float] = []

    lengths = attention_mask.sum(dim=1).tolist()

    for i in range(bsz):
        seq_len = int(lengths[i])
        pos = list(label_positions[i] or [])
        old = list(ref_logprobs[i] or [])
        adv = list(advantages[i] or [])
        L = min(len(pos), len(old), len(adv))

        for t in range(L):
            a = float(adv[t])
            if not (a == a):  # NaN
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
            flat_adv.append(a)

    if not flat_b:
        zero = token_logprobs.sum() * 0.0
        return zero, {
            "policy_loss": float(0.0),
            "kl_loss": float(0.0),
            "approx_kl": float("nan"),
            "clip_frac": float("nan"),
            "tokens": 0,
        }

    b_idx = torch.tensor(flat_b, device=device, dtype=torch.long)
    s_idx = torch.tensor(flat_s, device=device, dtype=torch.long)
    new_lp = token_logprobs[b_idx, s_idx].to(torch.float32)
    old_lp = torch.tensor(flat_old, device=device, dtype=torch.float32)
    adv_t = torch.tensor(flat_adv, device=device, dtype=torch.float32)

    log_ratio = new_lp - old_lp
    ratio = torch.exp(log_ratio)
    ratio_clipped = torch.clamp(ratio, 1.0 - ppo_clip_range, 1.0 + ppo_clip_range)

    surr1 = ratio * adv_t
    surr2 = ratio_clipped * adv_t
    obj = torch.min(surr1, surr2)
    policy_loss = -obj.mean()

    # Always-nonnegative reverse-KL estimate whose expectation under ref equals KL(ref||new):
    # E_ref[exp(d) - 1 - d] = -E_ref[d] = KL(ref||new), since E_ref[exp(d)] = 1.
    if kl_coef and kl_coef > 0.0:
        kl_rev = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = float(kl_coef) * kl_rev.mean()
    else:
        kl_loss = policy_loss * 0.0

    total = policy_loss + kl_loss

    approx_kl = (old_lp - new_lp).mean()
    clip_frac = ((ratio > (1.0 + ppo_clip_range)) | (ratio < (1.0 - ppo_clip_range))).to(torch.float32).mean()

    metrics = {
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "clip_frac": clip_frac.detach(),
        "tokens": int(new_lp.numel()),
        "adv_mean": adv_t.mean().detach(),
        "adv_abs_mean": adv_t.abs().mean().detach(),
        "ratio_mean": ratio.mean().detach(),
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
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    if cfg.compile_mode != "none":
        model = torch.compile(model, mode=cfg.compile_mode)

    # Dataset / DataLoader
    dataset = ParquetGRPODataset(
        cfg.data_path,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        arrow_batch_size=cfg.arrow_batch_size,
        group_size=cfg.group_size,
        col_prompt_idx=cfg.columns_prompt_idx,
        col_input_ids=cfg.columns_input_ids,
        col_label_positions=cfg.columns_label_positions,
        col_ref_logprobs=cfg.columns_ref_logprobs,
        col_reward=cfg.columns_reward,
    )

    collate_fn = _make_collate_fn(pad_token_id=int(tokenizer.pad_token_id))

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

            run_name = cfg.wandb_name or f"grpo_{int(time.time())}"
            wb = wandb
            wb.init(project=cfg.wandb_project, name=run_name, config=cfg.__dict__)
        except Exception:
            wb = None

    # Autocast
    if device.type == "cuda" and cfg.dtype in {torch.float16, torch.bfloat16}:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=cfg.dtype)
    else:
        from contextlib import nullcontext

        autocast_ctx = nullcontext()

    global_step = 0  # optimizer steps
    accum = 0
    t0 = time.perf_counter()

    def save_checkpoint(step: int):
        if not master:
            return
        out_dir = cfg.output_dir
        tgt = _unwrap_model(model)
        tgt.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
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
                        ppo_clip_range=cfg.ppo_clip_range,
                        kl_coef=cfg.kl_coef,
                    )
                    loss = loss / float(cfg.grad_accum_steps)

                loss.backward()

                update = (accum % cfg.grad_accum_steps) == 0
                if update:
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

                    if master and (global_step % max(1, cfg.log_every) == 0):
                        elapsed = time.perf_counter() - t0
                        tok = int(metrics["tokens"]) if isinstance(metrics["tokens"], int) else int(metrics["tokens"].item())
                        log_dict = {
                            "train/step": global_step,
                            "train/lr": lr,
                            "train/tokens": tok,
                            "train/policy_loss": float(metrics["policy_loss"].item()),
                            "train/kl_loss": float(metrics["kl_loss"].item()),
                            "train/approx_kl": float(metrics["approx_kl"].item()),
                            "train/clip_frac": float(metrics["clip_frac"].item()),
                            "train/adv_mean": float(metrics["adv_mean"].item()),
                            "train/adv_abs_mean": float(metrics["adv_abs_mean"].item()),
                            "train/ratio_mean": float(metrics["ratio_mean"].item()),
                            "train/elapsed_s": elapsed,
                        }
                        print(
                            f"[step {global_step}] loss={log_dict['train/policy_loss']:.6f} "
                            f"kl={log_dict['train/approx_kl']:.6f} clip={log_dict['train/clip_frac']:.3f} "
                            f"tokens={tok} lr={lr:.3e}",
                            flush=True,
                        )
                        if wb is not None:
                            wb.log(log_dict, step=global_step)

                    if master and cfg.save_every and cfg.save_every > 0 and (global_step % cfg.save_every == 0):
                        save_checkpoint(global_step)

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
