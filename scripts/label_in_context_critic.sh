#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=data_icl_critic_label
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --output=/home/rohin/icl_value/logs/critic_label.out
#SBATCH --error=/home/rohin/icl_value/logs/critic_label.err
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
critic_model_id="models/joint_distribution_critic_no_ans_supervise_from_8_with_32"
in_file="data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001_with_ref_logprobs.parquet"
out_file="data/icl_value_training_adaptivemath_data_qwen17b_non_thinking_32_min_p_001_with_ref_logprobs_with_critic.parquet"

dp_size=8
arrow_batch_size=1024
max_length=131072
shuffle_seed=0
label_rollouts_per_prompt=1  # -1=all, 0=none, K>0=subsample K per prompt
label_seed=0

echo "Starting in-context critic labeling:"
echo "  Critic model: $critic_model_id"
echo "  Input:       $in_file"
echo "  Output:      $out_file"
echo "  DP size:     $dp_size"
echo "  Max length:  $max_length"
echo "  Seed:        $shuffle_seed"
echo "  Label/prompt:$label_rollouts_per_prompt"
echo "  Label seed:  $label_seed"
echo "  Start time:  $(date)"

python3 -u src/label_in_context_critic.py \
    --critic-model-id "$critic_model_id" \
    --in-parquet "$in_file" \
    --out-parquet "$out_file" \
    --dp-size $dp_size \
    --arrow-batch-size $arrow_batch_size \
    --max-length $max_length \
    --shuffle-seed $shuffle_seed \
    --label-rollouts-per-prompt $label_rollouts_per_prompt \
    --label-seed $label_seed 2>&1 | tee -a /home/rohin/icl_value/logs/critic_label_$(basename ${out_file} .parquet).log

exit_code=${PIPESTATUS[0]}
echo "Labeling completed with exit code: $exit_code at $(date)"

[ -f "$out_file" ] && echo "Output file size: $(du -h $out_file | cut -f1)"
exit $exit_code
