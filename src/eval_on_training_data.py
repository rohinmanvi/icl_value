#!/usr/bin/env python3
"""
Evaluate a trained model on its training data prompts.

Extracts unique prompts from the training parquet, generates responses,
and evaluates them.

Example:
    python src/eval_on_training_data.py \
        --model models/policy_kl \
        --training-data data/training.parquet \
        --out data/eval_results.parquet \
        --dp-size 8
"""

import argparse
import os
import sys
from multiprocessing import get_context
from typing import List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"


def extract_unique_prompts(
    data_path: str,
    tokenizer,
    max_prompts: int = None,
) -> Tuple[List[str], List[int]]:
    """Extract unique prompts from training data by decoding prompt_token_ids."""
    df = pq.read_table(data_path).to_pandas()

    # Group by prompt_idx if available, otherwise use prompt_token_ids
    if "prompt_idx" in df.columns:
        # Get one row per unique prompt
        unique_df = df.groupby("prompt_idx").first().reset_index()
    else:
        # Deduplicate by prompt_token_ids (convert to tuple for hashing)
        df["prompt_tuple"] = df["prompt_token_ids"].apply(
            lambda x: tuple(x.tolist() if hasattr(x, 'tolist') else x)
        )
        unique_df = df.drop_duplicates(subset=["prompt_tuple"])

    prompts = []
    prompt_indices = []

    for idx, row in unique_df.iterrows():
        prompt_ids = row["prompt_token_ids"]
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()

        # Decode the prompt
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompts.append(prompt_text)
        prompt_indices.append(row.get("prompt_idx", idx))

        if max_prompts and len(prompts) >= max_prompts:
            break

    print(f"Extracted {len(prompts)} unique prompts from {len(df)} training samples")
    return prompts, prompt_indices


def worker(
    rank: int,
    dp_size: int,
    assigned_gpus: List[int],
    model_id: str,
    out_path: str,
    prompts: List[str],
    prompt_indices: List[int],
    tp_size: int,
    max_model_len: int,
    max_num_seqs: int,
    num_samples: int,
    min_p: float,
):
    """Worker process for generation."""
    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))

    # Shard prompts
    my_prompts = prompts[rank::dp_size]
    my_indices = prompt_indices[rank::dp_size]

    if not my_prompts:
        print(f"[Rank {rank}] No prompts assigned, exiting.")
        return

    print(f"[Rank {rank}] Processing {len(my_prompts)} prompts on GPUs {assigned_gpus}")

    # Load model
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        max_num_seqs=max_num_seqs or 256,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # Sampling params
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=0.6,
        top_p=0.95,
        min_p=min_p,
        max_tokens=max(128, max_model_len - 1024),
    )

    # Generate
    outputs = llm.generate(my_prompts, sampling_params)

    # Collect results
    records = []
    for prompt, prompt_idx, output in zip(my_prompts, my_indices, outputs):
        for sample_idx, completion in enumerate(output.outputs):
            records.append({
                "prompt": prompt,
                "prompt_idx": prompt_idx,
                "response": completion.text,
                "sample_idx": sample_idx,
            })

    # Save shard
    shard_path = out_path.replace(".parquet", f".shard_{rank}.parquet")
    table = pa.Table.from_pandas(pd.DataFrame(records))
    pq.write_table(table, shard_path)
    print(f"[Rank {rank}] Wrote {len(records)} records to {shard_path}")


def merge_shards(out_path: str, dp_size: int):
    """Merge all shards into final output."""
    tables = []
    for rank in range(dp_size):
        shard_path = out_path.replace(".parquet", f".shard_{rank}.parquet")
        if os.path.exists(shard_path):
            tables.append(pq.read_table(shard_path))
            os.remove(shard_path)

    if tables:
        merged = pa.concat_tables(tables)
        pq.write_table(merged, out_path)
        print(f"Merged {len(tables)} shards -> {out_path} ({merged.num_rows} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--training-data", required=True, help="Training parquet file")
    parser.add_argument("--out", required=True, help="Output parquet file")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--min-p", type=float, default=0.1)
    args = parser.parse_args()

    # Load tokenizer to decode prompts
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Extract unique prompts
    prompts, prompt_indices = extract_unique_prompts(
        args.training_data, tokenizer, args.max_prompts
    )

    # Determine GPU allocation
    import torch
    num_gpus = torch.cuda.device_count()
    if args.dp_size * args.tp_size > num_gpus:
        raise RuntimeError(f"Need {args.dp_size * args.tp_size} GPUs, only {num_gpus} available")

    # Spawn workers
    ctx = get_context("spawn")
    processes = []

    for rank in range(args.dp_size):
        gpu_start = rank * args.tp_size
        assigned_gpus = list(range(gpu_start, gpu_start + args.tp_size))

        p = ctx.Process(
            target=worker,
            args=(
                rank, args.dp_size, assigned_gpus, args.model, args.out,
                prompts, prompt_indices, args.tp_size, args.max_model_len,
                args.max_num_seqs, args.num_samples, args.min_p,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge shards
    merge_shards(args.out, args.dp_size)


if __name__ == "__main__":
    main()
