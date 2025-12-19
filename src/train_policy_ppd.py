#!/usr/bin/env python3
"""
Posterior Policy Distillation (PPD) policy training from pre-generated rollouts.

This script trains a causal LM policy from a Parquet dataset produced by:
  1) `src/data.py` (rollout generation; provides `input_ids` + `label_positions`)
  2) `src/label_ref_logprobs.py` (adds `ref_logprobs`)
  3) `src/label_in_context_critic.py` (adds `critic_p_correct`)

To make training fast and simple, you can also run an in-script preprocessing step
that writes a minimal "train-ready" parquet with ONLY:
  - `input_ids`
  - `labels` (same length, -100 where no loss)
  - `ref_logprobs` (one per output token; suffix-aligned)
  - `advantages`  (one per output token; suffix-aligned)
via `--preprocess-out`.

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

Notes:
  - Preprocessing validates that output tokens form a contiguous suffix so training can use fast slicing.
  - Training uses a fused cross-entropy path to get per-token logprobs.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import time
import traceback
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
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
    log_every: int
    save_every: int
    master_port: str
    wandb_project: str
    wandb_name: str
    columns_input_ids: str
    columns_label_positions: str
    columns_ref_logprobs: str
    columns_p_correct: str

    # NEW
    max_seq_len: int
    ddp_timeout_seconds: int

    # Preprocess / simplified training
    preprocess_out: str
    preprocess_only: bool
    overwrite_preprocess_out: bool
    max_rows: int
    preprocess_row_group_size: int
    columns_labels: str
    columns_advantages: str
    logprob_chunk_size: int


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
    p.add_argument("--learning-rate", type=float, default=1e-6)
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

    # Logging / saving
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0, help="If >0, save every N optimizer steps on rank 0.")
    p.add_argument("--master-port", default="29500")
    p.add_argument("--wandb-project", default="ppd_policy", help="If set, log to Weights & Biases (rank 0 only).")
    p.add_argument("--wandb-name", default="", help="Optional W&B run name.")

    # Column names (keep flexible)
    p.add_argument("--col-input-ids", default="input_ids")
    p.add_argument("--col-label-positions", default="label_positions")
    p.add_argument("--col-ref-logprobs", default="ref_logprobs")
    p.add_argument("--col-p-correct", default="critic_p_correct")

    # NEW
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help="If >0, truncate input_ids on the RIGHT to at most this many tokens (keeps prompt/prefix).",
    )
    p.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        default=7200,
        help="DDP/NCCL process group timeout in seconds.",
    )

    # Preprocess / simplified training
    p.add_argument(
        "--preprocess-out",
        default="",
        help=(
            "If set, preprocess --data-path into this parquet containing ONLY: "
            "input_ids, labels, ref_logprobs, advantages; then train on it."
        ),
    )
    p.add_argument("--preprocess-only", action="store_true", help="Run preprocessing then exit (no training).")
    p.add_argument(
        "--overwrite-preprocess-out",
        action="store_true",
        help="Allow overwriting an existing --preprocess-out file.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="If >0, limit rows processed (preprocess: total rows; train: rows per rank).",
    )
    p.add_argument(
        "--preprocess-row-group-size",
        type=int,
        default=1024,
        help="Target parquet row group size (rows) when writing --preprocess-out.",
    )
    p.add_argument("--col-labels", default="labels", help="Column name for labels in preprocessed parquet.")
    p.add_argument("--col-advantages", default="advantages", help="Column name for advantages in preprocessed parquet.")
    p.add_argument(
        "--logprob-chunk-size",
        type=int,
        default=512,
        help="Compute policy logprobs in chunks of this many output tokens to reduce peak memory at long contexts.",
    )

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
    if int(args.max_seq_len) < 0:
        raise ValueError("--max-seq-len must be >= 0")
    if int(args.ddp_timeout_seconds) < 60:
        raise ValueError("--ddp-timeout-seconds must be >= 60")
    if int(args.max_rows) == 0 or int(args.max_rows) < -1:
        raise ValueError("--max-rows must be -1 (all) or > 0")
    if int(args.preprocess_row_group_size) <= 0:
        raise ValueError("--preprocess-row-group-size must be > 0")
    if int(args.logprob_chunk_size) <= 0:
        raise ValueError("--logprob-chunk-size must be > 0")

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
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        master_port=str(args.master_port),
        wandb_project=str(args.wandb_project),
        wandb_name=str(args.wandb_name),
        columns_input_ids=str(args.col_input_ids),
        columns_label_positions=str(args.col_label_positions),
        columns_ref_logprobs=str(args.col_ref_logprobs),
        columns_p_correct=str(args.col_p_correct),
        max_seq_len=int(args.max_seq_len),
        ddp_timeout_seconds=int(args.ddp_timeout_seconds),

        preprocess_out=str(args.preprocess_out),
        preprocess_only=bool(args.preprocess_only),
        overwrite_preprocess_out=bool(args.overwrite_preprocess_out),
        max_rows=int(args.max_rows),
        preprocess_row_group_size=int(args.preprocess_row_group_size),
        columns_labels=str(args.col_labels),
        columns_advantages=str(args.col_advantages),
        logprob_chunk_size=int(args.logprob_chunk_size),
    )


def _iter_assigned_batches(
    pf: pq.ParquetFile, *, rank: int, world: int, batch_size: int, columns: Sequence[str] | None = None
) -> Iterable[pa.RecordBatch]:
    """
    Yield record batches assigned to this worker.
    Partition by parquet row groups when possible; otherwise fall back to row-wise striding.
    """
    if pf.num_row_groups >= world:
        row_groups = [rg for rg in range(pf.num_row_groups) if (rg % world) == rank]
        yield from pf.iter_batches(batch_size=batch_size, row_groups=row_groups, columns=list(columns) if columns else None)
        return

    # Fallback: row-wise striding across full file.
    row_offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
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


class PreprocessedParquetDataset(IterableDataset):
    """
    Iterable dataset over a preprocessed parquet file containing:
      - input_ids
      - labels (unused at train time; kept for auditability)
      - ref_logprobs (suffix-aligned, one per output token)
      - advantages  (suffix-aligned, one per output token)
    """

    def __init__(
        self,
        parquet_path: str,
        *,
        rank: int,
        world_size: int,
        seed: int,
        arrow_batch_size: int,
        shuffle: bool,
        max_rows: int,
        col_input_ids: str,
        col_labels: str,
        col_ref_logprobs: str,
        col_advantages: str,
    ) -> None:
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.arrow_batch_size = int(arrow_batch_size)
        self.shuffle = bool(shuffle)
        self.max_rows = int(max_rows)
        self.col_input_ids = str(col_input_ids)
        self.col_labels = str(col_labels)
        self.col_ref_logprobs = str(col_ref_logprobs)
        self.col_advantages = str(col_advantages)
        self.epoch = 0

        pf = pq.ParquetFile(self.parquet_path)
        names = set(pf.schema_arrow.names)
        missing = [c for c in [self.col_input_ids, self.col_labels, self.col_ref_logprobs, self.col_advantages] if c not in names]
        if missing:
            raise ValueError(
                f"Preprocessed parquet missing required columns: {missing}. "
                f"Present columns: {sorted(names)}. "
                "If this is a raw rollout parquet, run with --preprocess-out first."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)

        rng = random.Random(self.seed + 1_000_003 * self.epoch + 97 * self.rank)

        cols = [self.col_input_ids, self.col_ref_logprobs, self.col_advantages]
        processed = 0

        for rb in _iter_assigned_batches(
            pf,
            rank=self.rank,
            world=self.world_size,
            batch_size=self.arrow_batch_size,
            columns=cols,
        ):
            rb_names = list(rb.schema.names)
            idx_input = rb_names.index(self.col_input_ids)
            idx_ref = rb_names.index(self.col_ref_logprobs)
            idx_adv = rb_names.index(self.col_advantages)

            input_ids_col = rb.column(idx_input).to_pylist()
            ref_lp_col = rb.column(idx_ref).to_pylist()
            adv_col = rb.column(idx_adv).to_pylist()

            n = rb.num_rows
            order = list(range(n))
            if self.shuffle:
                rng.shuffle(order)

            for i in order:
                if self.max_rows > 0 and processed >= self.max_rows:
                    return
                processed += 1
                yield {
                    "input_ids": _as_list(input_ids_col[i]),
                    "ref_logprobs": _as_list(ref_lp_col[i]),
                    "advantages": _as_list(adv_col[i]),
                }


def _compute_advantages(
    p_correct: Sequence[float], *, tau: float, gae_lambda: float
) -> List[float]:
    """
    Compute per-token GAE advantages from critic p_correct values.

    V_t = log((p_t + tau)/(1+tau))
    delta_t = V_t - V_{t-1} (t>=1)
    A_t = delta_t + lambda * A_{t+1} (backwards; t>=1)
    A_0 = 0
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

    return out


def _preprocess_parquet(cfg: Config) -> str:
    """
    Create a simplified parquet file for fast training.

    Output columns:
      - input_ids: List[int32]
      - labels:    List[int32] (same length; -100 for no loss)
      - ref_logprobs: List[float32] (one per output token; suffix-aligned)
      - advantages:   List[float32] (one per output token; suffix-aligned)
    """
    out_path = str(cfg.preprocess_out or "").strip()
    if not out_path:
        raise ValueError("_preprocess_parquet called with empty cfg.preprocess_out")

    in_abs = os.path.abspath(cfg.data_path)
    out_abs = os.path.abspath(out_path)
    if in_abs == out_abs:
        raise ValueError("Refusing to overwrite --data-path; choose a different --preprocess-out.")

    if os.path.exists(out_path) and not cfg.overwrite_preprocess_out:
        raise FileExistsError(f"--preprocess-out already exists: {out_path} (pass --overwrite-preprocess-out to overwrite)")

    pf = pq.ParquetFile(cfg.data_path)
    names = set(pf.schema_arrow.names)
    required_in = [cfg.columns_input_ids, cfg.columns_label_positions, cfg.columns_ref_logprobs, cfg.columns_p_correct]
    missing = [c for c in required_in if c not in names]
    if missing:
        raise ValueError(f"Cannot preprocess: input parquet missing columns: {missing}. Present: {sorted(names)}")

    out_schema = pa.schema(
        [
            pa.field(cfg.columns_input_ids, pa.list_(pa.int32())),
            pa.field(cfg.columns_labels, pa.list_(pa.int32())),
            pa.field(cfg.columns_ref_logprobs, pa.list_(pa.float32())),
            pa.field(cfg.columns_advantages, pa.list_(pa.float32())),
        ]
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = pq.ParquetWriter(out_path, out_schema)

    processed = 0
    t0 = time.perf_counter()

    try:
        for rb in pf.iter_batches(batch_size=cfg.arrow_batch_size, columns=required_in):
            if cfg.max_rows > 0 and processed >= cfg.max_rows:
                break

            rb_names = list(rb.schema.names)
            idx_input = rb_names.index(cfg.columns_input_ids)
            idx_pos = rb_names.index(cfg.columns_label_positions)
            idx_ref = rb_names.index(cfg.columns_ref_logprobs)
            idx_p = rb_names.index(cfg.columns_p_correct)

            input_ids_col = rb.column(idx_input).to_pylist()
            label_pos_col = rb.column(idx_pos).to_pylist()
            ref_lp_col = rb.column(idx_ref).to_pylist()
            p_corr_col = rb.column(idx_p).to_pylist()

            n = rb.num_rows
            if cfg.max_rows > 0:
                n = min(n, cfg.max_rows - processed)

            out_input_ids: List[List[int]] = []
            out_labels: List[List[int]] = []
            out_ref: List[List[float]] = []
            out_adv: List[List[float]] = []

            for i in range(n):
                ids = _as_list(input_ids_col[i])
                if cfg.max_seq_len and cfg.max_seq_len > 0 and len(ids) > cfg.max_seq_len:
                    ids = ids[: cfg.max_seq_len]
                seq_len = len(ids)

                pos_raw = _as_list(label_pos_col[i])
                pos: List[int] = []
                for x in pos_raw:
                    try:
                        pos.append(int(x))
                    except Exception:
                        pos.append(-1)

                old_raw = _as_list(ref_lp_col[i])
                old_lp: List[float] = []
                for x in old_raw:
                    try:
                        old_lp.append(float(x))
                    except Exception:
                        old_lp.append(float("nan"))

                p_raw = _as_list(p_corr_col[i])
                p_corr_f: List[float] = []
                for x in p_raw:
                    try:
                        p_corr_f.append(float(x))
                    except Exception:
                        p_corr_f.append(float("nan"))

                # Common length among per-output lists
                L0 = min(len(pos), len(old_lp), len(p_corr_f))
                pos = pos[:L0]
                old_lp = old_lp[:L0]
                p_corr_f = p_corr_f[:L0]

                # Keep only tokens whose label_position is inside (possibly truncated) ids
                keep = [t for t, p in enumerate(pos) if 0 <= int(p) < seq_len]
                pos = [pos[t] for t in keep]
                old_lp = [old_lp[t] for t in keep]
                p_corr_f = [p_corr_f[t] for t in keep]

                # Validate output suffix contiguity so training can use fast slicing.
                if pos:
                    prompt_len = int(pos[0])
                    if prompt_len <= 0:
                        raise ValueError(
                            f"Invalid prompt_len={prompt_len} inferred from label_positions; "
                            "need at least 1 token of context for causal LM logprobs."
                        )
                    if int(pos[-1]) != seq_len - 1:
                        raise ValueError(
                            f"label_positions do not reach end of input_ids after truncation: "
                            f"last={int(pos[-1])}, seq_len={seq_len}"
                        )
                    for j, p in enumerate(pos):
                        if int(p) != prompt_len + j:
                            raise ValueError(
                                "label_positions are not a contiguous suffix; cannot use fast suffix alignment. "
                                f"Expected {prompt_len + j} at offset {j}, got {int(p)}."
                            )
                else:
                    prompt_len = seq_len

                adv = _compute_advantages(p_corr_f, tau=float(cfg.adv_smoothing_tau), gae_lambda=float(cfg.gae_lambda))
                if len(adv) != len(old_lp):
                    raise ValueError(f"advantages/ref_logprobs length mismatch: {len(adv)} vs {len(old_lp)}")

                if any(not math.isfinite(float(x)) for x in old_lp):
                    raise ValueError("Found non-finite ref_logprobs after truncation/filtering.")
                if any(not math.isfinite(float(x)) for x in adv):
                    raise ValueError("Found non-finite advantages after computation.")

                labels = [-100] * seq_len
                for p in pos:
                    if p < 0 or p >= seq_len:
                        continue
                    labels[int(p)] = int(ids[int(p)])
                if seq_len > 0:
                    labels[0] = -100  # causal LM has no label for position 0

                # Final sanity: output tokens must be the trailing segment.
                output_len = len(old_lp)
                if pos and output_len != max(0, seq_len - prompt_len):
                    raise ValueError(
                        f"Output length mismatch: len(ref_logprobs)={output_len} but "
                        f"seq_len={seq_len}, prompt_len={prompt_len} (expected {max(0, seq_len - prompt_len)})."
                    )

                out_input_ids.append([int(x) for x in ids])
                out_labels.append([int(x) for x in labels])
                out_ref.append([float(x) for x in old_lp])
                out_adv.append([float(x) for x in adv])

            if out_input_ids:
                table = pa.Table.from_pydict(
                    {
                        cfg.columns_input_ids: out_input_ids,
                        cfg.columns_labels: out_labels,
                        cfg.columns_ref_logprobs: out_ref,
                        cfg.columns_advantages: out_adv,
                    },
                    schema=out_schema,
                )
                writer.write_table(table, row_group_size=int(cfg.preprocess_row_group_size))
                processed += len(out_input_ids)

        elapsed = time.perf_counter() - t0
        print(f"[preprocess] wrote {processed} rows to {out_path} in {elapsed:.2f}s", flush=True)
        return out_path
    finally:
        writer.close()


def _make_preprocessed_collate_fn(*, pad_token_id: int):
    def collate(rows: List[dict]) -> dict:
        if not rows:
            return {}

        max_len = 0
        input_ids_list: List[List[int]] = []
        ref_logprobs: List[List[float]] = []
        advantages: List[List[float]] = []
        seq_lens: List[int] = []
        prompt_lens: List[int] = []
        num_tokens = 0

        for r in rows:
            ids = [int(x) for x in _as_list(r.get("input_ids"))]
            old_lp = [float(x) for x in _as_list(r.get("ref_logprobs"))]
            adv = [float(x) for x in _as_list(r.get("advantages"))]

            if len(old_lp) != len(adv):
                raise ValueError(f"ref_logprobs/advantages length mismatch: {len(old_lp)} vs {len(adv)}")

            seq_len = len(ids)
            out_len = len(old_lp)
            prompt_len = seq_len - out_len

            if prompt_len < 0:
                raise ValueError(f"Invalid row: len(input_ids)={seq_len} < len(ref_logprobs)={out_len}")
            if out_len > 0 and prompt_len <= 0:
                raise ValueError("Need at least 1 prompt token when training on output tokens (prompt_len must be > 0).")
            if seq_len > 0 and ids[0] < 0:
                raise ValueError("input_ids contains a negative token id.")

            input_ids_list.append(ids)
            ref_logprobs.append(old_lp)
            advantages.append(adv)
            seq_lens.append(seq_len)
            prompt_lens.append(prompt_len)
            max_len = max(max_len, seq_len)
            num_tokens += out_len

        input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            if ids:
                t = torch.tensor(ids, dtype=torch.long)
                input_ids[i, : t.numel()] = t
                attention_mask[i, : t.numel()] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seq_lens": seq_lens,
            "prompt_lens": prompt_lens,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "num_tokens": int(num_tokens),
        }

    return collate


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    return tgt


class _BackboneOnlyCausalLM(torch.nn.Module):
    """
    Wrap a HF AutoModelForCausalLM but only run the backbone (no full-vocab logits).

    This avoids materializing [B, S, V] logits at long sequence lengths, which is both
    slow and can exceed kernel indexing limits for large (S*V).
    """

    def __init__(self, hf_model: torch.nn.Module):
        super().__init__()
        self.hf_model = hf_model

        base = None
        prefix = getattr(hf_model, "base_model_prefix", None)
        if isinstance(prefix, str) and prefix:
            base = getattr(hf_model, prefix, None)
        if base is None:
            base = getattr(hf_model, "model", None)
        if base is None:
            base = getattr(hf_model, "transformer", None)
        if base is None:
            raise ValueError("Could not locate base transformer on the provided HF causal LM.")
        self.base_model = base

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        # BaseModelOutputWithPastAndCrossAttentions-compatible
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]


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


def _compute_policy_loss_from_hidden_suffix(
    *,
    hidden_states: torch.Tensor,  # [B, S, H]
    input_ids: torch.Tensor,  # [B, S]
    seq_lens: Sequence[int],
    prompt_lens: Sequence[int],
    ref_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[Sequence[float]],
    lm_head: torch.nn.Module,
    ppo_clip_range: float,
    logprob_chunk_size: int,
) -> Tuple[torch.Tensor, int]:
    """
    Fast path for the common case where output tokens are a contiguous suffix.

    For each row:
      - prompt_len = len(input_ids) - len(ref_logprobs)
      - output token positions are [prompt_len .. seq_len-1]
      - logits from hidden_states[:, t-1] predict token x_t
    """
    device = hidden_states.device
    bsz = int(hidden_states.shape[0])

    # Unwrap possible torch.compile and/or DDP wrappers for lm_head access
    tgt = lm_head._orig_mod if hasattr(lm_head, "_orig_mod") else lm_head
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    weight = tgt.weight
    bias = tgt.bias if hasattr(tgt, "bias") else None
    if bias is not None and not isinstance(bias, torch.Tensor):
        bias = None

    # Grad-connected accumulator for the loss.
    policy_sum = hidden_states.sum() * 0.0

    tokens = 0

    for i in range(bsz):
        out_len = len(ref_logprobs[i] or [])
        if out_len == 0:
            continue

        seq_len = int(seq_lens[i])
        prompt_len = int(prompt_lens[i])

        # Need hidden at positions that predict the output tokens.
        h_start = prompt_len - 1
        h_end = h_start + out_len

        if h_start < 0:
            raise ValueError(f"Invalid prompt_len={prompt_len} for seq_len={seq_len} (need prompt_len>=1 when out_len>0).")
        if (prompt_len + out_len) != seq_len:
            raise ValueError(
                f"Suffix alignment mismatch: prompt_len({prompt_len}) + out_len({out_len}) != seq_len({seq_len})."
            )
        if h_end > (seq_len - 1) or h_end > int(hidden_states.shape[1] - 1):
            raise ValueError(
                f"Suffix hidden slice out of range: start={h_start}, end={h_end}, seq_len={seq_len}, hidden_S={int(hidden_states.shape[1])}"
            )

        old_lp_full = torch.tensor(ref_logprobs[i], device=device, dtype=torch.float32)
        adv_full = torch.tensor(advantages[i], device=device, dtype=torch.float32)

        h = hidden_states[i, h_start:h_end, :]  # [out_len, H]
        y = input_ids[i, prompt_len:seq_len]  # [out_len]

        if int(h.shape[0]) != out_len or int(y.numel()) != out_len:
            raise ValueError(f"Internal slice mismatch for row {i}: h={tuple(h.shape)} y={int(y.numel())} out_len={out_len}")

        for j in range(0, out_len, int(logprob_chunk_size)):
            j2 = min(out_len, j + int(logprob_chunk_size))
            h_chunk = h[j:j2]
            y_chunk = y[j:j2]
            old_lp = old_lp_full[j:j2]
            adv_t = adv_full[j:j2]

            logits = F.linear(h_chunk, weight, bias)  # [chunk, V]
            nll = F.cross_entropy(logits, y_chunk, reduction="none")
            new_lp = (-nll).to(torch.float32)

            log_ratio = new_lp - old_lp
            ratio = torch.exp(log_ratio)

            if ppo_clip_range and ppo_clip_range > 0.0:
                lo = 1.0 - float(ppo_clip_range)
                hi = 1.0 + float(ppo_clip_range)
                ratio_used = torch.clamp(ratio, lo, hi)
            else:
                ratio_used = ratio

            adv_final = (adv_t - log_ratio).detach()
            policy_tokens = -(ratio_used * adv_final)

            policy_sum = policy_sum + policy_tokens.sum()
            tokens += int(new_lp.numel())

    if tokens == 0:
        zero = hidden_states.sum() * 0.0
        zero = zero + weight.sum() * 0.0
        if bias is not None:
            zero = zero + bias.sum() * 0.0
        return zero, 0

    denom = float(tokens)
    loss = policy_sum / denom

    return loss, int(tokens)


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
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            timeout=datetime.timedelta(seconds=int(cfg.ddp_timeout_seconds)),
        )

    rank = int(os.environ.get("RANK", "0"))
    master = rank == 0

    torch.manual_seed(int(cfg.seed) + rank)
    random.seed(int(cfg.seed) + rank)

    if master:
        os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot pad safely.")

    attn_impl = cfg.attn_implementation
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"

    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    try:
        hf_model.config.use_cache = False
    except Exception:
        pass

    if cfg.gradient_checkpointing:
        hf_model.gradient_checkpointing_enable()

    hf_model.to(device)

    model: torch.nn.Module = _BackboneOnlyCausalLM(hf_model).to(device)
    if distributed:
        model = DDP(model, device_ids=[device] if device.type == "cuda" else None)

    if cfg.compile_mode != "none":
        model = torch.compile(model, mode=cfg.compile_mode)

    lm_head = _unwrap_model(model).hf_model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("Model has no output embeddings; cannot compute token logprobs.")

    dataset = PreprocessedParquetDataset(
        cfg.data_path,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        arrow_batch_size=cfg.arrow_batch_size,
        shuffle=True,
        max_rows=cfg.max_rows,
        col_input_ids=cfg.columns_input_ids,
        col_labels=cfg.columns_labels,
        col_ref_logprobs=cfg.columns_ref_logprobs,
        col_advantages=cfg.columns_advantages,
    )

    collate_fn = _make_preprocessed_collate_fn(pad_token_id=int(tokenizer.pad_token_id))

    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    total_steps = int(cfg.max_steps) if cfg.max_steps and cfg.max_steps > 0 else 0
    if total_steps == 0 and cfg.num_epochs > 0:
        try:
            pf = pq.ParquetFile(cfg.data_path)
            total_rows = int(pf.metadata.num_rows) if pf.metadata is not None else 0
        except Exception:
            total_rows = 0

        if total_rows > 0:
            if cfg.max_rows and cfg.max_rows > 0:
                total_rows = min(total_rows, int(cfg.max_rows) * int(world_size))
            est_microbatches = (total_rows // max(1, world_size * cfg.micro_batch_size)) * cfg.num_epochs
            est_updates = max(1, est_microbatches // max(1, cfg.grad_accum_steps))
            total_steps = int(est_updates)

    warmup_steps = int(float(cfg.warmup_ratio) * float(total_steps)) if total_steps > 0 else 0

    wb = None
    if master and cfg.wandb_project:
        try:
            import wandb  # type: ignore

            run_name = cfg.wandb_name or f"ppd_{int(time.time())}"
            wb = wandb
            wb.init(
                project=cfg.wandb_project,
                name=run_name,
                config={
                    "model_id": cfg.model_id,
                    "data_path": cfg.data_path,
                    "ppo_clip_range": cfg.ppo_clip_range,
                    "adv_smoothing_tau": cfg.adv_smoothing_tau,
                    "gae_lambda": cfg.gae_lambda,
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
                    "max_seq_len": cfg.max_seq_len,
                    "ddp_timeout_seconds": cfg.ddp_timeout_seconds,
                    "logprob_chunk_size": cfg.logprob_chunk_size,
                },
            )
        except Exception:
            wb = None

    if device.type == "cuda" and cfg.dtype in {torch.float16, torch.bfloat16}:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=cfg.dtype)
    else:
        from contextlib import nullcontext

        autocast_ctx = nullcontext()

    global_step = 0
    accum = 0
    t0 = time.perf_counter()
    step_tokens_local = 0
    step_policy_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    step_max_seq_local = 0
    step_max_out_local = 0

    def save_checkpoint(step: int):
        if not master:
            return
        out_dir = cfg.output_dir
        tgt = _unwrap_model(model)
        tgt.hf_model.save_pretrained(out_dir)
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

                try:
                    seq_max = max(int(x) for x in (batch.get("seq_lens") or []))
                except ValueError:
                    seq_max = 0
                try:
                    out_max = max(int(s) - int(p) for s, p in zip(batch.get("seq_lens") or [], batch.get("prompt_lens") or []))
                except ValueError:
                    out_max = 0
                step_max_seq_local = max(step_max_seq_local, int(seq_max))
                step_max_out_local = max(step_max_out_local, int(out_max))

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                with autocast_ctx:
                    hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)

                    loss_unscaled, tok = _compute_policy_loss_from_hidden_suffix(
                        hidden_states=hidden_states,
                        input_ids=input_ids,
                        seq_lens=batch["seq_lens"],
                        prompt_lens=batch["prompt_lens"],
                        ref_logprobs=batch["ref_logprobs"],
                        advantages=batch["advantages"],
                        lm_head=lm_head,
                        ppo_clip_range=cfg.ppo_clip_range,
                        logprob_chunk_size=cfg.logprob_chunk_size,
                    )
                    loss = loss_unscaled / float(cfg.grad_accum_steps)

                loss.backward()

                step_tokens_local += tok
                tok_f = float(tok)
                step_policy_loss_sum = step_policy_loss_sum + loss_unscaled.detach().to(torch.float32) * tok_f

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
                        step_max_seq_local = 0
                        step_max_out_local = 0
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
                        step_max_seq_t = torch.tensor(step_max_seq_local, device=device, dtype=torch.long)
                        step_max_out_t = torch.tensor(step_max_out_local, device=device, dtype=torch.long)
                        if distributed:
                            dist.all_reduce(step_policy_loss_sum, op=dist.ReduceOp.SUM)
                            dist.all_reduce(step_max_seq_t, op=dist.ReduceOp.MAX)
                            dist.all_reduce(step_max_out_t, op=dist.ReduceOp.MAX)

                        denom = float(max(1, step_tokens_global))
                        mean_policy_loss = step_policy_loss_sum / denom

                    if master and do_log:
                        elapsed = time.perf_counter() - t0
                        log_dict = {
                            "train/step": global_step,
                            "train/lr": lr,
                            "train/tokens": int(step_tokens_global),
                            "train/policy_loss": float(mean_policy_loss.item()),
                            "train/max_seq_len": int(step_max_seq_t.item()),
                            "train/max_out_len": int(step_max_out_t.item()),
                            "train/elapsed_s": elapsed,
                        }
                        print(
                            f"[step {global_step}] loss={log_dict['train/policy_loss']:.6f} "
                            f"tokens={int(step_tokens_global)} maxS={log_dict['train/max_seq_len']} "
                            f"maxOut={log_dict['train/max_out_len']} lr={lr:.3e}",
                            flush=True,
                        )
                        if wb is not None:
                            wb.log(log_dict, step=global_step)

                    if master and cfg.save_every and cfg.save_every > 0 and (global_step % cfg.save_every == 0):
                        save_checkpoint(global_step)

                    step_tokens_local = 0
                    step_policy_loss_sum.zero_()
                    step_max_seq_local = 0
                    step_max_out_local = 0

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

    if cfg.preprocess_out:
        out_path = _preprocess_parquet(cfg)
        if cfg.preprocess_only:
            return
        cfg = replace(cfg, data_path=out_path)

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
