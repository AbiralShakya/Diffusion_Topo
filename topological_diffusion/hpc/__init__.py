"""
HPC Module for Topological Diffusion Generator
==============================================

This module contains high-performance computing utilities:
- SLURM job management and submission
- Distributed computing orchestration
- Resource monitoring and optimization
- Fault-tolerant job execution
"""

from .slurm_scripts import (
    SlurmJobManager,
    SlurmJobConfig,
    PhysicsCalculationJob,
    create_distributed_training_script
)

__all__ = [
    'SlurmJobManager',
    'SlurmJobConfig', 
    'PhysicsCalculationJob',
    'create_distributed_training_script'
]