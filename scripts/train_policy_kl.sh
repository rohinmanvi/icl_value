#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=train_policy_kl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/icl_value/logs/train_policy_kl.out
#SBATCH --error=/home/rohin/icl_value/logs/train_policy_kl.err
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

cd $HOME/icl_value
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate && conda deactivate
conda activate zip

# === Configuration ===
model_id="Qwen/Qwen3-1.7B"
data_path="data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001_with_q_labels.parquet"
weights_path="models/policy_kl_tau1"

temperature=1.0
max_length=8192
batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-5
min_learning_rate=0
warmup_ratio=0.05
num_epochs=1
max_steps=-1  # -1 for full training
log_freq=10

echo "Starting policy KL training:"
echo "  Model:       $model_id"
echo "  Data:        $data_path"
echo "  Output:      $weights_path"
echo "  Temperature: $temperature"
echo "  Max length:  $max_length"
echo "  Batch size:  $batch_size"
echo "  Grad accum:  $gradient_accumulation_steps"
echo "  LR:          $learning_rate"
echo "  Epochs:      $num_epochs"
echo "  Max steps:   $max_steps"
echo "  Start time:  $(date)"

python3 -u src/train_policy_kl.py \
    --model_id "$model_id" \
    --data_path "$data_path" \
    --weights_path "$weights_path" \
    --temperature $temperature \
    --max_length $max_length \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --min_learning_rate $min_learning_rate \
    --warmup_ratio $warmup_ratio \
    --num_epochs $num_epochs \
    --max_steps $max_steps \
    --log_freq $log_freq \
    --gradient_checkpointing 2>&1 | tee -a /home/rohin/icl_value/logs/train_policy_kl_$(basename ${weights_path}).log

exit_code=${PIPESTATUS[0]}
echo "Training completed with exit code: $exit_code at $(date)"
exit $exit_code
