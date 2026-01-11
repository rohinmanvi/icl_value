# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an RL/ML research project implementing **In-Context Value Learning (ICL-V)** - a method for training critic models and policies using in-context learning for joint value and cost prediction. The system trains models to predict reward and remaining length jointly, then extracts improved policies using Q-weighted sampling.

## Environment Setup

```bash
# Activate conda environment
conda activate zip

# Uses Python 3.10, PyTorch 2.6, vLLM 0.8.5, Flash Attention
```

## Key Commands

### Generate Training Data
```bash
python src/data.py \
    --model Qwen/Qwen3-1.7B \
    --dataset <hf_dataset> \
    --out data/output.parquet \
    --thinking-samples 2 --non-thinking-samples 2 \
    --dp-size 8 --tp-size 1
```

### Train In-Context Critic (Joint Value-Cost Prediction)
```bash
python src/train_in_context_critic.py \
    --model_id Qwen/Qwen3-1.7B \
    --data_path data/training_data.parquet \
    --weights_path models/critic \
    --ablation_type no_ans \
    --supervise_from_trajectory 4
```

### Label Trajectories with Q-Values
```bash
python src/label_in_context_critic.py \
    --critic_path models/critic \
    --ref_path Qwen/Qwen3-1.7B \
    --in_parquet data/input.parquet \
    --out_parquet data/labeled.parquet
```

### Train Policy with KL Divergence Loss
```bash
python src/train_policy_kl.py \
    --model_id Qwen/Qwen3-1.7B \
    --data_path data/labeled.parquet \
    --weights_path models/policy \
    --temperature 1.0
```

### Evaluate Results
```bash
python src/label_and_evaluate.py --data results.parquet --use-consistency
```

### Generate with Extracted Policy (Q-weighted sampling)
```bash
python src/extract_policy_generate.py \
    --critic_path models/critic \
    --ref_path Qwen/Qwen3-1.7B \
    --data_path data/input.parquet \
    --out_dir output/
```

## Architecture

### Pipeline Flow
1. **Data Generation** (`data.py`): Sample responses from base model with/without thinking tokens using vLLM
2. **Critic Training** (`train_in_context_critic.py`): Train joint distribution predictor P(reward, tokens_remaining) using packed sequences with in-context examples
3. **Q-Value Labeling** (`label_in_context_critic.py`): Label trajectories with per-token Q-values for all min-p filtered candidates
4. **Policy Training** (`train_policy_kl.py`): Distill Q-weighted policy using reverse KL divergence loss
5. **Evaluation** (`label_and_evaluate.py`): Evaluate correctness using LLM judge with best-of-n selection

### Core Components

**Critic Model**: Predicts joint distribution over (reward, remaining_tokens) using special token embeddings (starting at token ID 151669). Outputs binned probabilities marginalized to get reward probability (Q-value).

**In-Context Format**: Trajectories are packed with format:
```
[user prompt] [ctx trajectory] [feedback: "Reward: X\nLength: Y"] ... [target trajectory]
```
Supervision starts from trajectory N (configurable via `--supervise_from_trajectory`).

**Extracted Policy**: π_new(a|s) ∝ π_ref(a|s) · exp(Q(s,a) / τ) where Q is computed via min-p filtered candidate evaluation.

### Key Files
- `src/train_in_context_critic.py`: Joint value-cost critic with packed multi-trajectory training
- `src/train_policy_kl.py`: KL divergence policy training (reverse KL for mode-seeking)
- `src/train_policy_rft.py`: Rejection finetuning baseline (SFT on successful trajectories)
- `src/label_in_context_critic.py`: Per-token Q-value labeling with candidate expansion
- `src/extract_policy_generate.py`: Q-weighted generation with HTML visualization

### Qwen Token IDs (used throughout codebase)
- `IM_START_TOKEN_ID = 151644`
- `IM_END_TOKEN_ID = 151645`
- Distribution tokens start at `151669`
- Thinking token: `151667`

## Distributed Training

All training scripts support multi-GPU via PyTorch DDP:
- Workers spawn automatically based on `torch.cuda.device_count()`
- Data generation uses `--dp-size` for data parallel workers
- Environment variables for NCCL are set in shell scripts

## Data Format

Parquet files with columns:
- `prompt_idx`: Integer prompt identifier
- `prompt`: Text prompt
- `prompt_token_ids`, `output_token_ids`: Pre-tokenized sequences
- `correct` or `value`: Reward signal (float 0-1)
- `candidate_ids`, `candidate_q_values`, `candidate_ref_logprobs`: Per-token Q-value labels (after labeling)
