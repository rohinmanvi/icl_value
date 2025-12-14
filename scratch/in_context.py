#!/usr/bin/env python3
import argparse
import os
import torch
import re
from transformers import AutoTokenizer
from termcolor import colored  # pip install termcolor

# Import the dataset class from your train.py (assumes it is in the same folder)
from train_in_context_critic import JointDistributionDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/zip_training_adaptivemath_data_qwen17b_thinking_4_min_p_001.parquet", help="Path to your .parquet file")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B", help="Model ID for tokenizer")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of distinct prompts to visualize")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="visualization_output.txt", help="Path to save text output")
    args = parser.parse_args()

    # Regex to strip ANSI color codes for the text file
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    # Open the output file
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        
        # Helper function to print to both console (colored) and file (clean)
        def log(*msg_args, end="\n"):
            # Create the full string for the file
            text = " ".join(map(str, msg_args))
            
            # Print to console normally (preserves color)
            print(*msg_args, end=end)
            
            # Write to file (strips color)
            clean_text = ansi_escape.sub('', text)
            f_out.write(clean_text + end)
            f_out.flush()

        log(f"Logging output to: {args.output_file}")
        log(f"Loading Tokenizer: {args.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        log(f"Initializing Dataset from {args.data_path}...")
        # Initialize dataset with defaults from your script
        dataset = JointDistributionDataset(
            table=args.data_path,
            tokenizer=tokenizer,
            max_length=131072,
            thinking_only=False, # Set to True if your data relies on it
            label_column="correct" # or "value" depending on your data
        )

        log(f"\nLoaded {len(dataset)} samples (context-target pairs).")
        log("="*80)
        log("VISUALIZATION KEY:")
        log(colored("GRAY TEXT   = Context (Prompt + Sibling Response + Ground Truth Injection)", "light_grey"))
        log(colored("GREEN TEXT  = Target (The response we are calculating loss on)", "green", attrs=["bold"]))
        log("="*80 + "\n")

        indices = list(range(len(dataset)))
        # Simple slice or shuffle
        import random
        random.seed(args.seed)
        selected_indices = random.sample(indices, min(args.num_samples, len(dataset)))

        for i, idx in enumerate(selected_indices):
            item = dataset[idx]
            input_ids = item["input_ids"]
            label_positions = item["label_positions"]
            
            # Determine where the target starts based on label positions
            # The dataset logic: label_positions are the indices of the target response
            if not label_positions:
                log(f"Sample {idx}: No labels found (skipped)")
                continue
                
            target_start_idx = min(label_positions)
            
            # Split tokens
            context_tokens = input_ids[:target_start_idx]
            target_tokens = input_ids[target_start_idx:]
            
            # Decode
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=False)
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
            
            log(f"SAMPLE #{i+1} (Dataset Index {idx})")
            log("-" * 20)
            
            # Print Context
            log(colored(context_text, "light_grey"), end="")
            
            # Print Target (Continuous flow)
            log(colored(target_text, "green", attrs=["bold"]))
            
            # log("\n" + "-" * 20)
            # log(f"Target Reward: {dataset.samples[idx]['target_reward']}")
            # log(f"Target Length: {len(target_tokens)} tokens")
            log("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()