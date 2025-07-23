"""
Topological Diffusion Generator
==============================

A comprehensive framework for generating topological materials using
physics-informed diffusion models with electric field perturbations.

This package combines state-of-the-art physics theory with advanced
generative machine learning to model and predict topological insulators
that can transition between insulating and conducting states under
electric field perturbations.

Key Features:
- Physics-informed diffusion models with topological constraints
- Multi-scale physics integration (DFT → tight-binding → transport)
- Electric field perturbation modeling
- HPC-optimized distributed training
- Comprehensive topological invariant calculations
- Integration with established physics codes

Modules:
- physics: Quantum mechanics and topological physics calculations
- models: Machine learning models and architectures
- data: Data processing and DFT calculation pipelines
- training: Distributed training infrastructure
- hpc: High-performance computing utilities
- analysis: Analysis and visualization tools
"""

__version__ = "0.1.0"
__author__ = "Topological Materials Research Team"

# Core physics components
from .physics import (
    TopologicalHamiltonian,
    KaneMeleModel,
    BHZModel,
    FuKaneMeleModel,
    WeylSemimetalModel,
    HamiltonianFactory,
    LatticeParameters,
    SpinOrbitParameters,
    TopologicalInvariantCalculator,
    ElectricFieldSolver,
    StarkEffectCalculator
)

# Machine learning models
from .models import (
    PhysicsInformedDiffusion,
    TopologicalTransformer,
    PhysicsAwareAttention,
    MultiTaskLearningFramework,
    PhysicsConstraints,
    PhysicsValidator
)

# Training infrastructure
from .training import (
    DistributedTrainer,
    DistributedTrainingConfig,
    CheckpointManager,
    setup_distributed_training
)

# Data processing
from .data import (
    DFTCalculator,
    VASPCalculator,
    QuantumESPRESSOCalculator,
    WorkflowManager,
    DFTParameters
)

# HPC utilities
from .hpc import (
    SlurmJobManager,
    SlurmJobConfig
)

__all__ = [
    # Physics
    'TopologicalHamiltonian',
    'KaneMeleModel',
    'BHZModel', 
    'FuKaneMeleModel',
    'WeylSemimetalModel',
    'HamiltonianFactory',
    'LatticeParameters',
    'SpinOrbitParameters',
    'TopologicalInvariantCalculator',
    'ElectricFieldSolver',
    'StarkEffectCalculator',
    
    # Models
    'PhysicsInformedDiffusion',
    'TopologicalTransformer',
    'PhysicsAwareAttention', 
    'MultiTaskLearningFramework',
    'PhysicsConstraints',
    'PhysicsValidator',
    
    # Training
    'DistributedTrainer',
    'DistributedTrainingConfig',
    'CheckpointManager',
    'setup_distributed_training',
    
    # Data
    'DFTCalculator',
    'VASPCalculator',
    'QuantumESPRESSOCalculator',
    'WorkflowManager',
    'DFTParameters',
    
    # HPC
    'SlurmJobManager',
    'SlurmJobConfig'
]

# Package metadata
__title__ = "topological_diffusion"
__description__ = "Physics-informed diffusion models for topological materials generation"
__url__ = "https://github.com/topological-materials/topological-diffusion"
__license__ = "MIT"