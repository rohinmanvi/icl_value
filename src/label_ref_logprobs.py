#!/usr/bin/env python3
"""
Label a generated Parquet dataset with reference-policy token logprobs.

This script is inference-only: it reads pre-tokenized `input_ids` (and
`label_positions`) produced by `src/data.py` and computes log p_ref(token|prefix)
for the output tokens, then writes an updated Parquet file with the new columns.

It supports simple data-parallel execution by spawning one worker per GPU and
merging shard outputs at the end (similar to `src/data.py`).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


NEW_COLUMNS = (
    ("ref_model_id", pa.string()),
    ("ref_logprobs", pa.list_(pa.float32())),
    ("ref_logprob_sum", pa.float32()),
    ("ref_logprob_mean", pa.float32()),
)

# Candidate columns for policy distillation (only added when --min-p is set)
CANDIDATE_COLUMNS = (
    ("candidate_ids", pa.list_(pa.list_(pa.int32()))),       # [num_positions][num_candidates]
    ("candidate_logprobs", pa.list_(pa.list_(pa.float32()))), # [num_positions][num_candidates]
)


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
    in_parquet: str
    out_parquet: str
    dp_size: int
    batch_size: int
    arrow_batch_size: int
    dtype: torch.dtype
    attn_implementation: str
    allow_partial_merge: bool
    max_rows: int
    min_p: float  # If > 0, extract candidate tokens passing min_p filter


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="Reference policy model to score with (HF id/path).")
    p.add_argument("--in-parquet", required=True, help="Input parquet produced by `src/data.py`.")
    p.add_argument("--out-parquet", required=True, help="Output parquet with ref logprobs added.")
    p.add_argument("--dp-size", type=int, default=0, help="Number of GPU workers (default: all visible GPUs).")
    p.add_argument("--batch-size", type=int, default=1, help="Sequences per forward pass (default: 1).")
    p.add_argument("--arrow-batch-size", type=int, default=64, help="Rows per parquet read batch (per worker).")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for inference.",
    )
    p.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation to use when loading the model.",
    )
    p.add_argument(
        "--allow-partial-merge",
        action="store_true",
        help="If set, merge whatever shard outputs exist; otherwise require all dp shards.",
    )
    p.add_argument("--max-rows", type=int, default=-1, help="If >0, stop after scoring this many rows per worker.")
    p.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help=(
            "If >0, extract candidate tokens at each position where p(token) >= min_p * max_p. "
            "Adds candidate_ids and candidate_logprobs columns for policy distillation. "
            "Typical value: 0.05"
        ),
    )
    args = p.parse_args()

    in_abs = os.path.abspath(args.in_parquet)
    out_abs = os.path.abspath(args.out_parquet)
    if in_abs == out_abs:
        raise ValueError("Refusing to overwrite input parquet; choose a different --out-parquet.")

    dtype = _dtype_from_str(args.dtype)
    dp_size = int(args.dp_size)
    if dp_size < 0:
        raise ValueError("--dp-size must be >= 0")

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    arrow_batch_size = int(args.arrow_batch_size)
    if arrow_batch_size <= 0:
        raise ValueError("--arrow-batch-size must be >= 1")

    min_p = float(args.min_p)
    if min_p < 0.0 or min_p >= 1.0:
        raise ValueError("--min-p must be in [0, 1)")

    return Config(
        model_id=args.model_id,
        in_parquet=args.in_parquet,
        out_parquet=args.out_parquet,
        dp_size=dp_size,
        batch_size=batch_size,
        arrow_batch_size=arrow_batch_size,
        dtype=dtype,
        attn_implementation=str(args.attn_implementation),
        allow_partial_merge=bool(args.allow_partial_merge),
        max_rows=int(args.max_rows),
        min_p=min_p,
    )


def _iter_assigned_batches(pf: pq.ParquetFile, *, rank: int, world: int, batch_size: int) -> Iterable[pa.RecordBatch]:
    """
    Yield record batches assigned to this worker.

    Preferred partitioning is by parquet row groups (no duplicated reads).
    If there are fewer row groups than workers, falls back to row-wise striding,
    which requires each worker to scan the file but keeps memory bounded.
    """
    if pf.num_row_groups >= world:
        row_groups = [rg for rg in range(pf.num_row_groups) if (rg % world) == rank]
        yield from pf.iter_batches(batch_size=batch_size, row_groups=row_groups)
        return

    # Fallback: row-wise striding across the full file.
    row_offset = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n = batch.num_rows
        take_idx = [i for i in range(n) if ((row_offset + i) % world) == rank]
        row_offset += n
        if not take_idx:
            continue
        yield batch.take(pa.array(take_idx, type=pa.int32()))


def _ensure_required_columns(schema: pa.Schema) -> None:
    names = set(schema.names)
    if "input_ids" not in names:
        raise ValueError("Input parquet missing required column 'input_ids'.")
    if "label_positions" not in names:
        # We can reconstruct label positions if prompt/output token ids exist.
        if not ({"prompt_token_ids", "output_token_ids"} <= names):
            raise ValueError(
                "Input parquet missing required column 'label_positions' (and cannot reconstruct; "
                "need 'prompt_token_ids' and 'output_token_ids')."
            )


def _add_new_columns_to_schema(schema: pa.Schema, *, include_candidates: bool = False) -> pa.Schema:
    names = set(schema.names)
    all_columns = list(NEW_COLUMNS)
    if include_candidates:
        all_columns.extend(CANDIDATE_COLUMNS)
    for col, _typ in all_columns:
        if col in names:
            raise ValueError(
                f"Input parquet already has column '{col}'. "
                "Write to a new --out-parquet (or remove the existing columns first)."
            )
    out = schema
    for col, typ in all_columns:
        out = out.append(pa.field(col, typ))
    return out


def _as_pylist_maybe(x):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _resolve_input_ids_and_labels(
    batch: pa.RecordBatch,
) -> tuple[List[List[int]], List[List[int]]]:
    names = batch.schema.names

    input_ids_col = batch.column(names.index("input_ids")).to_pylist()

    if "label_positions" in names:
        label_positions_col = batch.column(names.index("label_positions")).to_pylist()
        # Normalize to python lists.
        input_ids = [_as_pylist_maybe(x) or [] for x in input_ids_col]
        label_positions = [_as_pylist_maybe(x) or [] for x in label_positions_col]
        return input_ids, label_positions

    # Reconstruct label positions from prompt/output token ids.
    prompt_ids_col = batch.column(names.index("prompt_token_ids")).to_pylist()
    output_ids_col = batch.column(names.index("output_token_ids")).to_pylist()

    input_ids = [_as_pylist_maybe(x) or [] for x in input_ids_col]
    prompt_ids = [_as_pylist_maybe(x) or [] for x in prompt_ids_col]
    output_ids = [_as_pylist_maybe(x) or [] for x in output_ids_col]

    out_input_ids: List[List[int]] = []
    label_positions: List[List[int]] = []
    for p_ids, o_ids, in_ids in zip(prompt_ids, output_ids, input_ids):
        prompt_len = len(p_ids)
        if not in_ids and (p_ids or o_ids):
            in_ids = p_ids + o_ids
        out_input_ids.append(list(in_ids))
        label_positions.append(list(range(prompt_len, prompt_len + len(o_ids))))
    return out_input_ids, label_positions


def _pad_batch(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    lengths = [len(s) for s in sequences]
    max_len = max(lengths) if lengths else 0
    if max_len == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0, 0), dtype=torch.long, device=device),
            lengths,
        )

    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(sequences):
        if not ids:
            continue
        t = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, : t.numel()] = t
        attention_mask[i, : t.numel()] = 1
    return input_ids, attention_mask, lengths


def _extract_candidates_at_positions(
    log_probs: torch.Tensor,  # [B, S, V] or [S, V]
    *,
    min_p: float,
    valid_mask: torch.Tensor,  # [B, S] or [S] - True for valid positions
) -> List[List[Tuple[List[int], List[float]]]]:
    """
    Extract candidate tokens at each position using min_p filtering.

    For each position, a token is a candidate if p(token) >= min_p * max_p(position).

    Returns: List[List[Tuple[List[int], List[float]]]]
        Outer list: batch dimension
        Middle list: sequence positions
        Inner tuple: (candidate_token_ids, candidate_log_probs)
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)

    B, S, V = log_probs.shape
    probs = log_probs.exp()  # [B, S, V]
    max_probs = probs.max(dim=-1, keepdim=True).values  # [B, S, 1]
    threshold = min_p * max_probs  # [B, S, 1]

    results: List[List[Tuple[List[int], List[float]]]] = []

    for b in range(B):
        seq_results: List[Tuple[List[int], List[float]]] = []
        for s in range(S):
            if not valid_mask[b, s]:
                seq_results.append(([], []))
                continue

            mask = probs[b, s] >= threshold[b, s, 0]
            indices = mask.nonzero(as_tuple=True)[0]
            cand_ids = indices.tolist()
            cand_lps = log_probs[b, s, indices].tolist()
            seq_results.append((cand_ids, cand_lps))
        results.append(seq_results)

    return results


@torch.inference_mode()
def _score_minibatch(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    min_p: float = 0.0,
) -> Tuple[torch.Tensor, Optional[List[List[Tuple[List[int], List[float]]]]]]:
    """
    Return per-token logprobs for the *next token* labels.

    Output shape: [B, S-1] where logprobs[b, t-1] = log p(x_t | x_<t).

    If min_p > 0, also returns candidate tokens at each position:
        List[List[Tuple[List[int], List[float]]]] - [B][S-1][(cand_ids, cand_lps)]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, S, V]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    bsz, seqlen_minus1, vocab = shift_logits.shape

    # Compute log probs for all tokens (needed for candidates)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)  # [B, S-1, V]

    # Extract actual token logprobs
    logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, S-1]

    # Mask out padded labels (positions where the target token is padding).
    label_mask = attention_mask[:, 1:].to(torch.bool)
    logprobs = torch.where(label_mask, logprobs, torch.full_like(logprobs, float("nan")))

    # Extract candidates if requested
    candidates = None
    if min_p > 0.0:
        candidates = _extract_candidates_at_positions(
            log_probs, min_p=min_p, valid_mask=label_mask
        )

    return logprobs, candidates


def _gather_logprobs(
    token_logprobs: torch.Tensor,
    *,
    input_lengths: Sequence[int],
    label_positions: Sequence[Sequence[int]],
) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Convert [B, S-1] token logprobs into per-row lists aligned to label_positions.
    """
    out_lists: List[List[float]] = []
    out_sums: List[float] = []
    out_means: List[float] = []

    for i in range(token_logprobs.shape[0]):
        seq_len = int(input_lengths[i])
        positions = list(label_positions[i])
        if not positions:
            out_lists.append([])
            out_sums.append(0.0)
            out_means.append(float("nan"))
            continue

        if seq_len < 2:
            raise ValueError("Encountered sequence with length < 2 but non-empty label_positions.")

        row_lp: List[float] = []
        for pos in positions:
            if not isinstance(pos, int):
                pos = int(pos)
            if pos <= 0 or pos >= seq_len:
                raise ValueError(f"label_positions contains out-of-range index {pos} for seq_len={seq_len}.")
            lp = float(token_logprobs[i, pos - 1].item())
            if lp != lp:  # NaN check
                raise ValueError("Got NaN logprob (likely due to padding/positions mismatch).")
            row_lp.append(lp)

        s = float(sum(row_lp))
        out_lists.append(row_lp)
        out_sums.append(s)
        out_means.append(float(s / len(row_lp)))
    return out_lists, out_sums, out_means


def _gather_candidates(
    candidates: List[List[Tuple[List[int], List[float]]]],
    *,
    input_lengths: Sequence[int],
    label_positions: Sequence[Sequence[int]],
) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
    """
    Extract candidates at label_positions from the full [B][S-1] candidate list.

    Returns:
        candidate_ids: List[List[List[int]]] - [B][num_label_positions][num_candidates]
        candidate_logprobs: List[List[List[float]]] - [B][num_label_positions][num_candidates]
    """
    out_ids: List[List[List[int]]] = []
    out_lps: List[List[List[float]]] = []

    for i, cand_seq in enumerate(candidates):
        seq_len = int(input_lengths[i])
        positions = list(label_positions[i])

        if not positions:
            out_ids.append([])
            out_lps.append([])
            continue

        row_ids: List[List[int]] = []
        row_lps: List[List[float]] = []

        for pos in positions:
            if not isinstance(pos, int):
                pos = int(pos)
            # pos is the label position (1-indexed for token at that position)
            # candidates are stored at index pos-1 (0-indexed, shifted like logprobs)
            cand_idx = pos - 1
            if cand_idx < 0 or cand_idx >= len(cand_seq):
                # Out of range - append empty
                row_ids.append([])
                row_lps.append([])
            else:
                ids, lps = cand_seq[cand_idx]
                row_ids.append(ids)
                row_lps.append(lps)

        out_ids.append(row_ids)
        out_lps.append(row_lps)

    return out_ids, out_lps


def _load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id; cannot pad batches safely.")
    return tokenizer


def _load_model(model_id: str, *, dtype: torch.dtype, attn_implementation: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return model


def _worker(local_rank: int, world: int, cfg: Config) -> None:
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = _load_tokenizer(cfg.model_id)
    attn_impl = cfg.attn_implementation
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = "sdpa"
    model = _load_model(cfg.model_id, dtype=cfg.dtype, attn_implementation=attn_impl).to(device)

    pf = pq.ParquetFile(cfg.in_parquet)
    _ensure_required_columns(pf.schema_arrow)
    include_candidates = cfg.min_p > 0.0
    out_schema = _add_new_columns_to_schema(pf.schema_arrow, include_candidates=include_candidates)

    os.makedirs(os.path.dirname(cfg.out_parquet) or ".", exist_ok=True)
    shard_path = f"{cfg.out_parquet}.part{local_rank}"
    writer = pq.ParquetWriter(shard_path, out_schema)

    start_t = time.perf_counter()
    processed = 0

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=cfg.dtype)
        if device.type == "cuda" and cfg.dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )

    try:
        for rb in _iter_assigned_batches(pf, rank=local_rank, world=world, batch_size=cfg.arrow_batch_size):
            if cfg.max_rows > 0 and processed >= cfg.max_rows:
                break

            input_ids_list, label_positions_list = _resolve_input_ids_and_labels(rb)
            n_rows = len(input_ids_list)
            if n_rows == 0:
                continue

            ref_logprobs_all: List[List[float]] = []
            ref_sum_all: List[float] = []
            ref_mean_all: List[float] = []
            candidate_ids_all: List[List[List[int]]] = []
            candidate_lps_all: List[List[List[float]]] = []

            for start in range(0, n_rows, cfg.batch_size):
                end = min(n_rows, start + cfg.batch_size)
                mb_input_ids = input_ids_list[start:end]
                mb_label_positions = label_positions_list[start:end]

                if not mb_input_ids:
                    continue

                input_ids_t, attention_mask_t, lengths = _pad_batch(
                    mb_input_ids, pad_token_id=int(tokenizer.pad_token_id), device=device
                )
                if input_ids_t.numel() == 0:
                    # Degenerate empty sequences; keep alignment.
                    for _ in range(len(mb_input_ids)):
                        ref_logprobs_all.append([])
                        ref_sum_all.append(0.0)
                        ref_mean_all.append(float("nan"))
                        if include_candidates:
                            candidate_ids_all.append([])
                            candidate_lps_all.append([])
                    continue

                with autocast_ctx:
                    token_logprobs, candidates = _score_minibatch(
                        model, input_ids=input_ids_t, attention_mask=attention_mask_t, min_p=cfg.min_p
                    )

                lp_lists, lp_sums, lp_means = _gather_logprobs(
                    token_logprobs, input_lengths=lengths, label_positions=mb_label_positions
                )
                ref_logprobs_all.extend(lp_lists)
                ref_sum_all.extend(lp_sums)
                ref_mean_all.extend(lp_means)

                if include_candidates and candidates is not None:
                    cand_ids, cand_lps = _gather_candidates(
                        candidates, input_lengths=lengths, label_positions=mb_label_positions
                    )
                    candidate_ids_all.extend(cand_ids)
                    candidate_lps_all.extend(cand_lps)

            # Truncate if max_rows cuts mid-batch.
            if cfg.max_rows > 0:
                remaining = max(0, cfg.max_rows - processed)
                if remaining < n_rows:
                    rb = rb.slice(0, remaining)
                    ref_logprobs_all = ref_logprobs_all[:remaining]
                    ref_sum_all = ref_sum_all[:remaining]
                    ref_mean_all = ref_mean_all[:remaining]
                    if include_candidates:
                        candidate_ids_all = candidate_ids_all[:remaining]
                        candidate_lps_all = candidate_lps_all[:remaining]
                    n_rows = remaining

            rb = rb.append_column("ref_model_id", pa.array([cfg.model_id] * n_rows, type=pa.string()))
            rb = rb.append_column("ref_logprobs", pa.array(ref_logprobs_all, type=pa.list_(pa.float32())))
            rb = rb.append_column("ref_logprob_sum", pa.array(ref_sum_all, type=pa.float32()))
            rb = rb.append_column("ref_logprob_mean", pa.array(ref_mean_all, type=pa.float32()))

            if include_candidates:
                rb = rb.append_column(
                    "candidate_ids",
                    pa.array(candidate_ids_all, type=pa.list_(pa.list_(pa.int32())))
                )
                rb = rb.append_column(
                    "candidate_logprobs",
                    pa.array(candidate_lps_all, type=pa.list_(pa.list_(pa.float32())))
                )

            writer.write_table(pa.Table.from_batches([rb], schema=out_schema))
            processed += n_rows

        writer.close()
    except Exception:
        writer.close()
        raise

    elapsed = time.perf_counter() - start_t
    print(
        f"[rank {local_rank}] wrote {shard_path} ({processed} rows) in {elapsed:.2f}s",
        flush=True,
    )


def _merge_shards(out_parquet: str, *, dp_size: int, allow_partial: bool) -> tuple[bool, int, int]:
    part_paths = [f"{out_parquet}.part{r}" for r in range(dp_size) if os.path.exists(f"{out_parquet}.part{r}")]
    found = len(part_paths)
    expected = dp_size

    if found == 0:
        return (False, 0, expected)

    if found != expected and not allow_partial:
        return (False, found, expected)

    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)

    schema = pq.ParquetFile(part_paths[0]).schema_arrow
    writer = pq.ParquetWriter(out_parquet, schema)
    try:
        for part in part_paths:
            pf = pq.ParquetFile(part)
            for rb in pf.iter_batches():
                writer.write_table(pa.Table.from_batches([rb], schema=schema))
        writer.close()
    except Exception:
        writer.close()
        raise

    for part in part_paths:
        try:
            os.remove(part)
        except OSError:
            pass

    return (True, found, expected)


def main() -> None:
    cfg = parse_args()

    # Decide how many workers to spawn.
    ngpus = torch.cuda.device_count()
    if cfg.dp_size == 0:
        dp = max(1, ngpus)
    else:
        dp = cfg.dp_size

    if ngpus > 0 and dp > ngpus:
        raise RuntimeError(f"Requested dp_size={dp}, but only {ngpus} CUDA devices are visible.")
    if ngpus == 0 and dp != 1:
        raise RuntimeError("No CUDA devices visible; run with --dp-size 1 for CPU scoring.")

    min_p_info = f"  min_p:    {cfg.min_p} (extracting candidates)\n" if cfg.min_p > 0 else ""
    print(
        f"Scoring parquet with reference model:\n"
        f"  model_id: {cfg.model_id}\n"
        f"  in:       {cfg.in_parquet}\n"
        f"  out:      {cfg.out_parquet}\n"
        f"  dp_size:  {dp}\n"
        f"  batch:    {cfg.batch_size} (arrow batch {cfg.arrow_batch_size})\n"
        f"{min_p_info}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs = []
    for r in range(dp):
        p = ctx.Process(target=_worker, args=(r, dp, cfg), daemon=False)
        p.start()
        procs.append(p)

    any_fail = False
    for r, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            any_fail = True
            print(f"[main] worker rank {r} (pid={p.pid}) exited with code {p.exitcode}", flush=True)

    merged, found, expected = _merge_shards(cfg.out_parquet, dp_size=dp, allow_partial=cfg.allow_partial_merge)
    if not merged:
        if found == 0:
            print("✗ No shards were produced; check worker errors above.", flush=True)
        else:
            print(f"✗ Shards missing ({found}/{expected}).", flush=True)
        if any_fail or found != expected:
            sys.exit(1)
        return

    suffix = "" if found == expected else f" (PARTIAL: {found}/{expected} shards)"
    print(f"✓ Wrote {cfg.out_parquet}{suffix}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
