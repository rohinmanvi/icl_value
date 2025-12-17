#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) for the policy on the same rollout Parquet format used by PPO.

This trains a causal LM with standard next-token cross-entropy, but only on the tokens
corresponding to the sampled response (as indicated by `label_positions`) and only for
rows whose correctness label indicates a correct solution.

Supports optionally subsampling up to K (correct) samples per `prompt_idx` to avoid always
training on all rollouts (e.g., 32 per prompt).
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
    log_every: int
    save_every: int
    master_port: str
    wandb_project: str
    wandb_name: str
    samples_per_prompt: int
    correct_threshold: float
    shard_by_prompt_idx: bool
    columns_input_ids: str
    columns_label_positions: str
    columns_prompt_idx: str
    columns_correct: str


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data-path", required=True, help="Parquet file with rollouts (must include correctness labels).")
    p.add_argument("--output-dir", default="models/policy_sft")
    p.add_argument("--seed", type=int, default=42)

    # Data loading / DDP
    p.add_argument("--dp-size", type=int, default=0, help="Number of GPU processes (default: all visible GPUs).")
    p.add_argument("--arrow-batch-size", type=int, default=1024, help="Rows per parquet read batch per worker.")
    p.add_argument("--micro-batch-size", type=int, default=1, help="Per-process microbatch size.")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument(
        "--shard-by-prompt-idx",
        action="store_true",
        help=(
            "If set, shard data by (prompt_idx % world_size == rank), which guarantees per-prompt subsampling is "
            "globally enforced across ranks but requires each rank to scan the full parquet."
        ),
    )

    # Selection
    p.add_argument(
        "--samples-per-prompt",
        type=int,
        default=-1,
        help="If >0, keep at most this many *correct* samples per prompt_idx per epoch. If 0, keep none. If <0, keep all.",
    )
    p.add_argument(
        "--correct-threshold",
        type=float,
        default=0.5,
        help="Treat rows with correct >= threshold as correct (default: 0.5).",
    )

    # Training schedule
    p.add_argument("--num-epochs", type=int, default=1, help="Number of dataset passes (best-effort under DDP).")
    p.add_argument("--max-steps", type=int, default=-1, help="If >0, stop after this many optimizer steps.")
    p.add_argument("--learning-rate", type=float, default=2e-5)
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

    # Logging / saving
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0, help="If >0, save every N optimizer steps on rank 0.")
    p.add_argument("--master-port", default="29501")
    p.add_argument("--wandb-project", default="", help="If set, log to Weights & Biases (rank 0 only).")
    p.add_argument("--wandb-name", default="", help="Optional W&B run name.")

    # Column names (keep flexible)
    p.add_argument("--col-input-ids", default="input_ids")
    p.add_argument("--col-label-positions", default="label_positions")
    p.add_argument("--col-prompt-idx", default="prompt_idx")
    p.add_argument("--col-correct", default="correct")

    args = p.parse_args()

    if args.micro_batch_size <= 0:
        raise ValueError("--micro-batch-size must be >= 1")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.arrow_batch_size <= 0:
        raise ValueError("--arrow-batch-size must be >= 1")
    if args.samples_per_prompt < -1:
        raise ValueError("--samples-per-prompt must be >= -1")
    if not math.isfinite(float(args.correct_threshold)):
        raise ValueError("--correct-threshold must be finite")

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
        log_every=int(args.log_every),
        save_every=int(args.save_every),
        master_port=str(args.master_port),
        wandb_project=str(args.wandb_project),
        wandb_name=str(args.wandb_name),
        samples_per_prompt=int(args.samples_per_prompt),
        correct_threshold=float(args.correct_threshold),
        shard_by_prompt_idx=bool(args.shard_by_prompt_idx),
        columns_input_ids=str(args.col_input_ids),
        columns_label_positions=str(args.col_label_positions),
        columns_prompt_idx=str(args.col_prompt_idx),
        columns_correct=str(args.col_correct),
    )


def _iter_assigned_batches(
    pf: pq.ParquetFile, *, rank: int, world: int, batch_size: int
) -> Iterable[pa.RecordBatch]:
    if pf.num_row_groups >= world:
        row_groups = [rg for rg in range(pf.num_row_groups) if (rg % world) == rank]
        yield from pf.iter_batches(batch_size=batch_size, row_groups=row_groups)
        return

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


class ParquetSFTDataset(IterableDataset):
    def __init__(
        self,
        parquet_path: str,
        *,
        rank: int,
        world_size: int,
        seed: int,
        arrow_batch_size: int,
        shuffle: bool,
        samples_per_prompt: int,
        correct_threshold: float,
        shard_by_prompt_idx: bool,
        col_input_ids: str,
        col_label_positions: str,
        col_prompt_idx: str,
        col_correct: str,
    ) -> None:
        super().__init__()
        self.parquet_path = str(parquet_path)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.arrow_batch_size = int(arrow_batch_size)
        self.shuffle = bool(shuffle)
        self.samples_per_prompt = int(samples_per_prompt)
        self.correct_threshold = float(correct_threshold)
        self.shard_by_prompt_idx = bool(shard_by_prompt_idx)
        self.col_input_ids = str(col_input_ids)
        self.col_label_positions = str(col_label_positions)
        self.col_prompt_idx = str(col_prompt_idx)
        self.col_correct = str(col_correct)
        self.epoch = 0

        pf = pq.ParquetFile(self.parquet_path)
        names = set(pf.schema_arrow.names)
        missing = [
            c
            for c in [self.col_input_ids, self.col_label_positions, self.col_prompt_idx, self.col_correct]
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
        idx_prompt = schema.get_field_index(self.col_prompt_idx)
        idx_correct = schema.get_field_index(self.col_correct)

        rng = random.Random(self.seed + 1_000_003 * self.epoch + 97 * self.rank)
        kept_per_prompt: dict[int, int] = {} if self.samples_per_prompt >= 0 else {}

        if self.shard_by_prompt_idx:
            batches = pf.iter_batches(batch_size=self.arrow_batch_size)
        else:
            batches = _iter_assigned_batches(pf, rank=self.rank, world=self.world_size, batch_size=self.arrow_batch_size)

        for rb in batches:
            input_ids_col = rb.column(idx_input).to_pylist()
            label_pos_col = rb.column(idx_pos).to_pylist()
            prompt_idx_col = rb.column(idx_prompt).to_pylist()
            correct_col = rb.column(idx_correct).to_pylist()

            n = rb.num_rows
            order = list(range(n))
            if self.shuffle:
                rng.shuffle(order)

            for i in order:
                try:
                    prompt_idx = int(prompt_idx_col[i])
                except Exception:
                    continue

                if self.shard_by_prompt_idx and (prompt_idx % max(1, self.world_size)) != self.rank:
                    continue

                try:
                    corr = float(correct_col[i])
                except Exception:
                    corr = float("nan")
                if not (math.isfinite(corr) and corr >= self.correct_threshold):
                    continue

                if self.samples_per_prompt == 0:
                    continue
                if self.samples_per_prompt > 0:
                    prev = kept_per_prompt.get(prompt_idx, 0)
                    if prev >= self.samples_per_prompt:
                        continue
                    kept_per_prompt[prompt_idx] = prev + 1

                yield {
                    "input_ids": _as_list(input_ids_col[i]),
                    "label_positions": _as_list(label_pos_col[i]),
                }


def _make_collate_fn(*, pad_token_id: int):
    def collate(rows: List[dict]) -> dict:
        if not rows:
            return {}

        max_len = max(len(r["input_ids"]) for r in rows)
        input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        label_positions: List[List[int]] = []

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
            label_positions.append(pos)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_positions": label_positions,
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


def _compute_sft_loss(
    *,
    token_logprobs: torch.Tensor,  # [B, S-1]
    attention_mask: torch.Tensor,  # [B, S]
    label_positions: Sequence[Sequence[int]],
) -> Tuple[torch.Tensor, dict]:
    device = token_logprobs.device
    bsz = token_logprobs.shape[0]

    flat_b: List[int] = []
    flat_s: List[int] = []
    lengths = attention_mask.sum(dim=1).tolist()

    for i in range(bsz):
        seq_len = int(lengths[i])
        pos = list(label_positions[i] or [])
        for p_raw in pos:
            try:
                p = int(p_raw)
            except Exception:
                continue
            if p <= 0 or p >= seq_len:
                continue
            if (p - 1) >= token_logprobs.shape[1]:
                continue
            flat_b.append(i)
            flat_s.append(p - 1)

    if not flat_b:
        zero = token_logprobs.sum() * 0.0
        nan = zero.detach() + float("nan")
        return zero, {"sft_loss": zero.detach(), "tokens": 0, "logprob_mean": nan}

    b_idx = torch.tensor(flat_b, device=device, dtype=torch.long)
    s_idx = torch.tensor(flat_s, device=device, dtype=torch.long)
    lp = token_logprobs[b_idx, s_idx].to(torch.float32)
    loss = -lp.mean()

    metrics = {
        "sft_loss": loss.detach(),
        "tokens": int(lp.numel()),
        "logprob_mean": lp.mean().detach(),
    }
    return loss, metrics


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

    dataset = ParquetSFTDataset(
        cfg.data_path,
        rank=rank,
        world_size=world_size,
        seed=cfg.seed,
        arrow_batch_size=cfg.arrow_batch_size,
        shuffle=True,
        samples_per_prompt=cfg.samples_per_prompt,
        correct_threshold=cfg.correct_threshold,
        shard_by_prompt_idx=cfg.shard_by_prompt_idx,
        col_input_ids=cfg.columns_input_ids,
        col_label_positions=cfg.columns_label_positions,
        col_prompt_idx=cfg.columns_prompt_idx,
        col_correct=cfg.columns_correct,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=_make_collate_fn(pad_token_id=int(tokenizer.pad_token_id)),
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
            est_microbatches = (total_rows // max(1, world_size * cfg.micro_batch_size)) * cfg.num_epochs
            est_updates = max(1, est_microbatches // max(1, cfg.grad_accum_steps))
            total_steps = int(est_updates)

    warmup_steps = int(float(cfg.warmup_ratio) * float(total_steps)) if total_steps > 0 else 0

    wb = None
    if master and cfg.wandb_project:
        try:
            import wandb  # type: ignore

            run_name = cfg.wandb_name or f"sft_{int(time.time())}"
            wb = wandb
            wb.init(
                project=cfg.wandb_project,
                name=run_name,
                config={
                    "model_id": cfg.model_id,
                    "data_path": cfg.data_path,
                    "samples_per_prompt": cfg.samples_per_prompt,
                    "correct_threshold": cfg.correct_threshold,
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
                    "shard_by_prompt_idx": cfg.shard_by_prompt_idx,
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

                    loss, metrics = _compute_sft_loss(
                        token_logprobs=token_logprobs,
                        attention_mask=attention_mask,
                        label_positions=batch["label_positions"],
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
                            "train/sft_loss": float(metrics["sft_loss"].item()),
                            "train/logprob_mean": float(metrics["logprob_mean"].item()),
                            "train/elapsed_s": elapsed,
                        }
                        print(
                            f"[step {global_step}] loss={log_dict['train/sft_loss']:.6f} "
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
