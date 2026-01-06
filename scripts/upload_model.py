#!/usr/bin/env python3
"""
Upload a trained model to Hugging Face Hub.

Example:
    python scripts/upload_model.py \
        --model_path models/joint_distribution_critic_no_ans_supervise_from_8_with_32 \
        --repo_id rohinm/joint-distribution-critic \
        --private
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model_path", required=True, help="Local path to the model")
    parser.add_argument("--repo_id", required=True, help="HF repo ID (e.g., username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    parser.add_argument("--commit_message", default="Upload model", help="Commit message")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        print(f"Created/verified repo: {args.repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the entire folder
    print(f"Uploading {args.model_path} to {args.repo_id}...")
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    print(f"Done! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
