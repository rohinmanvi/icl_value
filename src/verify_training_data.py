#!/usr/bin/env python3
"""
Verify that responses in training data are evaluated as correct.

Takes a training parquet file and converts it to evaluation format,
then you can run label_and_evaluate.py on the output.

Example:
    python src/verify_training_data.py \
        --training-data data/training.parquet \
        --out data/training_for_eval.parquet \
        --model Qwen/Qwen3-1.7B

    python src/label_and_evaluate.py --data data/training_for_eval.parquet
"""

import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True, help="Training parquet file")
    parser.add_argument("--out", required=True, help="Output parquet for evaluation")
    parser.add_argument("--model", default=None, help="Model to use for decoding tokens (if needed)")
    parser.add_argument("--answer-source", default=None, help="HuggingFace dataset to load answers from")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows to process")
    args = parser.parse_args()

    df = pq.read_table(args.training_data).to_pandas()
    print(f"Loaded {len(df)} rows from {args.training_data}")
    print(f"Columns: {list(df.columns)}")

    if args.max_rows:
        df = df.head(args.max_rows)
        print(f"Limited to {len(df)} rows")

    # Check what columns we have
    has_prompt_text = "prompt" in df.columns
    has_response_text = "response" in df.columns
    has_prompt_ids = "prompt_token_ids" in df.columns
    has_response_ids = "output_token_ids" in df.columns or "response_token_ids" in df.columns

    # Load tokenizer if we need to decode
    tokenizer = None
    if (not has_prompt_text or not has_response_text) and args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load answers from external source if provided
    answer_map = {}
    if args.answer_source:
        from datasets import load_dataset
        print(f"Loading answers from {args.answer_source}")
        ans_ds = load_dataset(args.answer_source, split="train")
        prob_col = next((c for c in ["problem", "question", "prompt"] if c in ans_ds.column_names), None)
        ans_col = next((c for c in ["answer", "solution", "response"] if c in ans_ds.column_names), None)
        if prob_col and ans_col:
            for row in ans_ds:
                answer_map[row[prob_col].strip()] = str(row[ans_col])
            print(f"Loaded {len(answer_map)} answers")

    records = []
    for idx, row in df.iterrows():
        # Get prompt text
        if has_prompt_text:
            prompt = row["prompt"]
        elif has_prompt_ids and tokenizer:
            prompt_ids = row["prompt_token_ids"]
            if hasattr(prompt_ids, 'tolist'):
                prompt_ids = prompt_ids.tolist()
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        else:
            raise ValueError("No prompt text or token IDs found, or no model provided for decoding")

        # Get response text
        if has_response_text:
            response = row["response"]
        elif has_response_ids and tokenizer:
            resp_col = "output_token_ids" if "output_token_ids" in df.columns else "response_token_ids"
            response_ids = row[resp_col]
            if hasattr(response_ids, 'tolist'):
                response_ids = response_ids.tolist()
            response = tokenizer.decode(response_ids, skip_special_tokens=False)
        else:
            raise ValueError("No response text or token IDs found, or no model provided for decoding")

        # Get answer
        answer = row.get("answer", "")
        if pd.isna(answer) or answer == "":
            # Try to extract raw problem from chat format
            raw_problem = prompt
            if "<|im_start|>user" in raw_problem:
                start = raw_problem.find("<|im_start|>user") + len("<|im_start|>user")
                end = raw_problem.find("<|im_end|>", start)
                if end > start:
                    raw_problem = raw_problem[start:end].strip()
            answer = answer_map.get(raw_problem, "")

        # Get other fields
        finished = row.get("finished", True)
        length = row.get("length", len(response))
        output_token_ids = row.get("output_token_ids", row.get("response_token_ids", []))
        if hasattr(output_token_ids, 'tolist'):
            output_token_ids = output_token_ids.tolist()

        records.append({
            "prompt": prompt,
            "response": response,
            "answer": str(answer) if answer else "",
            "finished": bool(finished),
            "length": int(length),
            "output_token_ids": list(output_token_ids) if output_token_ids is not None else [],
            "prompt_idx": row.get("prompt_idx", idx),
        })

    out_df = pd.DataFrame(records)
    pq.write_table(pa.Table.from_pandas(out_df), args.out)
    print(f"Wrote {len(out_df)} rows to {args.out}")
    print(f"Columns: {list(out_df.columns)}")

    # Summary stats
    print(f"\nSummary:")
    print(f"  Finished: {out_df['finished'].sum()}/{len(out_df)}")
    print(f"  Has answer: {(out_df['answer'] != '').sum()}/{len(out_df)}")


if __name__ == "__main__":
    main()
