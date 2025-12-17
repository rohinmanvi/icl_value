#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=icl_policy_ppo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/icl_value/logs/policy_ppo.out
#SBATCH --error=/home/rohin/icl_value/logs/policy_ppo.err
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
data_path="data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001_with_ref_logprobs_with_critic.parquet"
output_dir="models/policy_ppo_qwen17b_tau01"

dp_size=8
micro_batch_size=1
grad_accum_steps=1
learning_rate=1e-6
ppo_clip_range=0.2
adv_smoothing_tau=0.1
kl_coef=0.0
max_steps=-1

echo "Starting offline PPO policy training:"
echo "  Model:       $model_id"
echo "  Data:        $data_path"
echo "  Output dir:  $output_dir"
echo "  DP size:     $dp_size"
echo "  Micro batch: $micro_batch_size"
echo "  Grad accum:  $grad_accum_steps"
echo "  LR:          $learning_rate"
echo "  PPO clip:    $ppo_clip_range"
echo "  Tau:         $adv_smoothing_tau"
echo "  KL coef:     $kl_coef"
echo "  Max steps:   $max_steps"
echo "  Start time:  $(date)"

python3 -u src/train_policy_ppo.py \
  --model-id "$model_id" \
  --data-path "$data_path" \
  --output-dir "$output_dir" \
  --dp-size $dp_size \
  --micro-batch-size $micro_batch_size \
  --grad-accum-steps $grad_accum_steps \
  --learning-rate $learning_rate \
  --ppo-clip-range $ppo_clip_range \
  --adv-smoothing-tau $adv_smoothing_tau \
  --kl-coef $kl_coef \
  --max-steps $max_steps \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --gradient-checkpointing 2>&1 | tee -a /home/rohin/icl_value/logs/policy_ppo_$(basename ${output_dir}).log

exit_code=${PIPESTATUS[0]}
echo "Training completed with exit code: $exit_code at $(date)"
exit $exit_code

