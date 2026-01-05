#!/usr/bin/env python3
"""
Debug script to compare stored reference log probs with model's actual log probs.
Loads one sample and runs the model to verify alignment.
"""
import argparse
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1, help="Check multiple samples")
    parser.add_argument("--min_prompt_len", type=int, default=0, help="Only check samples with prompt >= this")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Disable thinking mode if present
    if hasattr(model, 'generation_config') and hasattr(model.generation_config, 'enable_thinking'):
        model.generation_config.enable_thinking = False
        print("Disabled thinking mode")

    print(f"\nLoading data: {args.data_path}")
    table = pq.read_table(args.data_path)
    df = table.to_pandas()

    # Find samples to check
    samples_to_check = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        prompt_ids = row["prompt_token_ids"]
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        if len(prompt_ids) >= args.min_prompt_len:
            samples_to_check.append(idx)
        if len(samples_to_check) >= args.num_samples:
            break

    print(f"Checking {len(samples_to_check)} samples with prompt_len >= {args.min_prompt_len}")

    all_diffs = []

    for sample_idx in samples_to_check:
        row = df.iloc[sample_idx]
        print(f"\n{'='*80}")
        print(f"Sample {sample_idx}:")

        prompt_ids = row["prompt_token_ids"]
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        prompt_ids = list(prompt_ids)

        output_ids = row["output_token_ids"]
        if hasattr(output_ids, 'tolist'):
            output_ids = output_ids.tolist()
        output_ids = list(output_ids)

        candidate_ids = row["candidate_ids"]
        candidate_ref_logprobs = row["candidate_ref_logprobs"]

        if hasattr(candidate_ids, 'tolist'):
            candidate_ids = candidate_ids.tolist()
        if hasattr(candidate_ref_logprobs, 'tolist'):
            candidate_ref_logprobs = candidate_ref_logprobs.tolist()

        print(f"  Prompt length: {len(prompt_ids)}")
        print(f"  Output length: {len(output_ids)}")
        print(f"  Num positions with candidates: {len(candidate_ids)}")

        # Build full sequence
        full_ids = prompt_ids + output_ids
        print(f"  Full sequence length: {len(full_ids)}")

        # Run model
        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_tensor, use_cache=False)
            logits = outputs.logits[0]  # [S, V]

        # Compare log probs at each position
        prompt_len = len(prompt_ids)
        sample_diffs = []

        for j in range(min(5, len(output_ids))):
            pos = prompt_len + j - 1
            if pos < 0:
                continue

            next_token = output_ids[j]
            cand_ids = candidate_ids[j] if j < len(candidate_ids) else None
            cand_ref_lps = candidate_ref_logprobs[j] if j < len(candidate_ref_logprobs) else None

            if cand_ids is None or cand_ref_lps is None:
                continue

            # Convert to lists
            if hasattr(cand_ids, 'tolist'):
                cand_ids = cand_ids.tolist()
            if hasattr(cand_ref_lps, 'tolist'):
                cand_ref_lps = cand_ref_lps.tolist()

            if len(cand_ids) < 2:  # Skip positions with only 1 candidate
                continue

            # Get model's log probs at this position
            pos_logits = logits[pos]
            model_log_probs = F.log_softmax(pos_logits.float(), dim=-1)

            # Compare for each candidate
            diffs = []
            for i, (cid, stored_lp) in enumerate(zip(cand_ids, cand_ref_lps)):
                model_lp = model_log_probs[cid].item()
                diff = abs(model_lp - stored_lp)
                diffs.append(diff)

            mean_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)
            sample_diffs.append(mean_diff)
            all_diffs.append(mean_diff)

            print(f"  Pos {j} (pos={pos}): {len(cand_ids)} cands, mean_diff={mean_diff:.4f}, max_diff={max_diff:.4f}")

        if sample_diffs:
            print(f"  Sample mean diff: {sum(sample_diffs)/len(sample_diffs):.4f}")

    if all_diffs:
        print(f"\n{'='*80}")
        print(f"OVERALL: {len(all_diffs)} positions checked")
        print(f"  Mean diff: {sum(all_diffs)/len(all_diffs):.4f}")
        print(f"  Max diff: {max(all_diffs):.4f}")
        print(f"  Positions with diff > 0.1: {sum(1 for d in all_diffs if d > 0.1)}")


if __name__ == "__main__":
    main()
