#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=ref_logprobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/icl_value/logs/ref_logprobs.out
#SBATCH --error=/home/rohin/icl_value/logs/ref_logprobs.err
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

# Configuration
model="Qwen/Qwen3-1.7B"
in_file="data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001.parquet"
out_file="data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001_with_ref_logprobs.parquet"

dp_size=8
batch_size=1
arrow_batch_size=64

echo "Starting reference logprob labeling:"
echo "  Model: $model"
echo "  Input: $in_file"
echo "  Output: $out_file"
echo "  DP size: $dp_size"
echo "  Batch size: $batch_size (arrow batch $arrow_batch_size)"
echo "  Start time: $(date)"

python3 -u src/label_ref_logprobs.py \
    --model-id "$model" \
    --in-parquet "$in_file" \
    --out-parquet "$out_file" \
    --dp-size $dp_size \
    --batch-size $batch_size \
    --arrow-batch-size $arrow_batch_size 2>&1 | tee -a /home/rohin/icl_value/logs/ref_logprobs_$(basename ${out_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Labeling completed with exit code: $exit_code at $(date)"

[ -f "$out_file" ] && echo "Output file size: $(du -h $out_file | cut -f1)"
exit $exit_code

