#!/bin/bash
#SBATCH --job-name=topological_diffusion_training
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive

# Load modules
module load python/3.9
module load cuda/11.8
module load nccl/2.12
module load openmpi/4.1.0

# Set up environment
source activate topological_diffusion

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# CUDA and performance settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# Memory and performance optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Create necessary directories
mkdir -p logs checkpoints results

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: 4 H100"
echo "Memory per node: 256G"
echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"

# Launch distributed training
srun python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/training_example.py \
    --config training_config.json \
    --distributed \
    --checkpoint-dir checkpoints \
    --log-dir logs \
    --mixed-precision \
    --gradient-compression \
    --zero-optimizer

echo "Training job completed at $(date)"