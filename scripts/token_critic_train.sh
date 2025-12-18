#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=data_token_critic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/icl_value/logs/token_critic_train.out
#SBATCH --error=/home/rohin/icl_value/logs/token_critic_train.err
#SBATCH --account=liquidai
#SBATCH --exclude=liquid-gpu-[054]

# Network configuration
export PMI_DEBUG=1
export MPI_ROOT=/usr/mpi/gcc/openmpi-4.1.7a1/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_ROOT/lib
export OMPI_MCA_btl_tcp_if_include=bond0
export UCX_TLS=self,shm,tcp
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_13,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8"
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=1
export NCCL_SOCKET_IFNAME=bond0
export LC_CTYPE=en_US.UTF-8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

cd $HOME/icl_value
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip

# === Configuration ===
model_id="Qwen/Qwen3-1.7B"
data_path="/home/rohin/icl_value/data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001.parquet"
weights_path="models/token_correctness_critic_from_8_with_32"
learning_rate=1e-4
label_column="correct"
correctness_label_smoothing=0.1
batch_size=1
num_epochs=1
wandb_project="new_joint_critic"

# Ablation Choice: no_ans, no_ans_no_rewards, no_ans_first_reward_only (note: "full" is an alias of "no_ans"; no answer injection)
ablation_type="no_ans"

# Dataset packing / supervision
# 1-indexed trajectory number to start supervision (e.g., 4 => supervise trajectories 4..end; first 3 are context-only).
supervise_from_trajectory=8
# Number of packed examples per prompt (reshuffle trajectories each time). With multi-trajectory supervision, 1 is usually sufficient.
examples_per_prompt=1

echo "=================================================="
echo "Starting Token Correctness Critic Training"
echo "  Model: $model_id"
echo "  Data: $data_path"
echo "  Output: $weights_path"
echo "  Ablation: $ablation_type"
echo "  Supervise From Trajectory: $supervise_from_trajectory"
echo "  Examples/Prompt: $examples_per_prompt"
echo "  Label Smoothing: $correctness_label_smoothing"
echo "=================================================="

python3 -u src/train_in_context_token_critic.py \
    --model_id "$model_id" \
    --data_path "$data_path" \
    --weights_path "$weights_path" \
    --learning_rate $learning_rate \
    --label-column "$label_column" \
    --correctness_label_smoothing $correctness_label_smoothing \
    --batch_size $batch_size \
    --num_epochs $num_epochs \
    --wandb_project "$wandb_project" \
    --ablation_type "$ablation_type" \
    --supervise_from_trajectory $supervise_from_trajectory \
    --examples_per_prompt $examples_per_prompt \
    --dist-backend "ddp" 2>&1 | tee -a /home/rohin/icl_value/logs/train_token_critic.log

exit_code=${PIPESTATUS[0]}
echo "Done. Exit code: $exit_code"
exit $exit_code

