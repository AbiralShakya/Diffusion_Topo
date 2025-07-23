"""
Physics Module for Topological Diffusion Generator
================================================

This module contains all physics calculations for topological materials:
- Quantum Hamiltonians with spin-orbit coupling
- Electric field perturbations and transport
- Topological invariant calculations
- Berry curvature and quantum geometry
"""

from .quantum_hamiltonian import (
    TopologicalHamiltonian,
    KaneMeleModel,
    BHZModel, 
    FuKaneMeleModel,
    WeylSemimetalModel,
    HamiltonianFactory,
    LatticeParameters,
    SpinOrbitParameters,
    BlochHamiltonianSolver,
    generate_k_path,
    generate_k_grid
)

__all__ = [
    'TopologicalHamiltonian',
    'KaneMeleModel',
    'BHZModel',
    'FuKaneMeleModel', 
    'WeylSemimetalModel',
    'HamiltonianFactory',
    'LatticeParameters',
    'SpinOrbitParameters',
    'BlochHamiltonianSolver',
    'generate_k_path',
    'generate_k_grid'
]