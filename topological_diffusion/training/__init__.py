"""
Training Module for Topological Diffusion Generator
==================================================

This module contains training infrastructure and optimization:
- Distributed training with SLURM integration
- Physics-informed training loops
- Checkpoint management and fault tolerance
- Resource monitoring and optimization
- Multi-task learning frameworks
"""

from .distributed_training import (
    DistributedTrainer,
    DistributedTrainingConfig,
    SLURMTrainingManager,
    CheckpointManager,
    ResourceMonitor,
    CommunicationOptimizer,
    setup_distributed_training,
    create_distributed_data_loader
)

__all__ = [
    'DistributedTrainer',
    'DistributedTrainingConfig',
    'SLURMTrainingManager', 
    'CheckpointManager',
    'ResourceMonitor',
    'CommunicationOptimizer',
    'setup_distributed_training',
    'create_distributed_data_loader'
]