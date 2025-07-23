"""
Basic Usage Examples for Topological Diffusion Generator
========================================================

This script demonstrates basic usage of the topological diffusion generator
for creating and analyzing topological materials.
"""

import numpy as np
import torch
import logging
from pathlib import Path

# Import topological diffusion components
import sys
sys.path.append('..')

from topological_diffusion import (
    # Physics components
    HamiltonianFactory,
    LatticeParameters,
    SpinOrbitParameters,
    TopologicalInvariantCalculator,
    ElectricFieldSolver,
    StarkEffectCalculator,
    
    # ML components
    TopologicalTransformer,
    PhysicsInformedDiffusion,
    PhysicsConstraints,
    PhysicsValidator,
    
    # Training components
    DistributedTrainingConfig,
    
    # Data components
    DFTParameters,
    VASPCalculator
)

from topological_diffusion.physics.electric_field import (
    ElectricFieldConfig, create_material_properties, FieldSolverFactory
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_basic_hamiltonian():
    """Example 1: Create and analyze a basic topological Hamiltonian"""
    logger.info("=== Example 1: Basic Hamiltonian Analysis ===")
    
    # Create graphene with Kane-Mele spin-orbit coupling
    graphene = HamiltonianFactory.create_from_material('graphene')
    
    # Define k-points for band structure calculation
    from topological_diffusion.physics.quantum_hamiltonian import generate_k_path
    
    high_sym_points = {
        'Γ': np.array([0, 0, 0]),
        'K': np.array([4*np.pi/(3*np.sqrt(3)), 0, 0]),
        'M': np.array([np.pi/np.sqrt(3), np.pi/3, 0])
    }
    
    k_path, distances = generate_k_path(high_sym_points, ['Γ', 'K', 'M', 'Γ'], 200)
    
    # Compute band structure
    eigenvalues, eigenvectors = graphene.get_band_structure(k_path)
    
    logger.info(f"Computed band structure with {len(eigenvalues)} k-points")
    logger.info(f"Number of bands: {eigenvalues.shape[1]}")
    logger.info(f"Energy range: {eigenvalues.min():.3f} to {eigenvalues.max():.3f} eV")
    
    return graphene, eigenvalues, eigenvectors

def example_2_topological_invariants():
    """Example 2: Calculate topological invariants"""
    logger.info("=== Example 2: Topological Invariant Calculation ===")
    
    # Create Bi2Se3 3D topological insulator
    bi2se3 = HamiltonianFactory.create_from_material('bi2se3')
    
    # Create topological invariant calculator
    topo_calc = TopologicalInvariantCalculator(bi2se3)
    
    # Generate k-grid for invariant calculation
    from topological_diffusion.physics.quantum_hamiltonian import generate_k_grid
    k_grid = generate_k_grid(bi2se3.reciprocal_vectors, (10, 10, 10))
    
    # Assume half-filled system
    H_test = bi2se3.build_hamiltonian(np.array([0, 0, 0]))
    n_bands = H_test.shape[0]
    occupied_bands = list(range(n_bands // 2))
    
    # Compute all topological invariants
    invariants = topo_calc.compute_all_invariants(occupied_bands, k_grid, system_dimension=3)
    
    # Classify topological phase
    phase = topo_calc.classify_topological_phase(invariants)
    logger.info(f"Topological phase: {phase}")
    
    # Print results
    for name, invariant in invariants.items():
        logger.info(f"{invariant.name}: {invariant.value}")
        
    return invariants

def example_3_electric_field_effects():
    """Example 3: Electric field effects on topological materials"""
    logger.info("=== Example 3: Electric Field Effects ===")
    
    # Create field configuration
    field_config = ElectricFieldConfig(
        field_strength=1e6,  # 1 MV/m
        field_direction=np.array([1, 0, 0]),
        temperature=300.0
    )
    
    # Create material properties for Bi2Se3
    material_props = create_material_properties('bi2se3')
    
    # Create uniform field solver
    field_solver = FieldSolverFactory.create_solver('uniform', field_config, material_props)
    
    # Create Hamiltonian
    bi2se3 = HamiltonianFactory.create_from_material('bi2se3')
    
    # Create Stark effect calculator
    stark_calc = StarkEffectCalculator(bi2se3, field_solver)
    
    # Test coordinates (simplified)
    coordinates = np.array([[0, 0, 0], [1e-10, 0, 0], [0, 1e-10, 0]])
    
    # Apply Stark effect at Γ point
    k_gamma = np.array([0, 0, 0])
    H_perturbed = stark_calc.apply_stark_effect(k_gamma, coordinates)
    
    # Compute perturbed eigenvalues
    eigenvals_perturbed = np.linalg.eigvals(H_perturbed)
    
    # Compare with unperturbed system
    H_unperturbed = bi2se3.build_hamiltonian(k_gamma)
    eigenvals_unperturbed = np.linalg.eigvals(H_unperturbed)
    
    logger.info(f"Unperturbed eigenvalues: {np.sort(np.real(eigenvals_unperturbed))}")
    logger.info(f"Perturbed eigenvalues: {np.sort(np.real(eigenvals_perturbed))}")
    
    # Calculate field-dependent band structure
    k_path = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])
    field_strengths = np.array([0, 5e5, 1e6, 2e6])  # V/m
    
    band_energies = stark_calc.compute_field_dependent_bands(k_path, field_strengths, coordinates)
    logger.info(f"Field-dependent band energies shape: {band_energies.shape}")
    
    return stark_calc, band_energies

def example_4_physics_constraints():
    """Example 4: Physics constraints for generation"""
    logger.info("=== Example 4: Physics Constraints Setup ===")
    
    # Create physics constraints for topological insulator generation
    constraints = PhysicsConstraints(
        space_group=166,  # R-3m (Bi2Se3 space group)
        time_reversal_symmetry=True,
        inversion_symmetry=True,
        band_gap_range=(0.1, 0.5),  # eV
        topological_class='TI',
        target_z2_invariant=(1, [0, 0, 0]),  # Strong TI
        max_field_strength=1e7,  # V/m
        allowed_elements=['Bi', 'Se', 'Te', 'Sb'],
        max_atoms_per_cell=15,
        stability_threshold=0.1  # eV/atom above hull
    )
    
    # Create physics validator
    validator = PhysicsValidator(constraints)
    
    # Test validation with mock structure data
    test_structure_data = {
        'space_group': 166,
        'has_inversion': True,
        'formation_energy': 0.05,  # eV/atom
        'band_gap': 0.3,  # eV
        'topological_class': 'TI',
        'z2_invariant': (1, [0, 0, 0])
    }
    
    validation_result = validator.validate_structure(test_structure_data)
    
    logger.info(f"Validation result: {validation_result}")
    logger.info(f"Structure is valid: {validation_result['is_valid']}")
    logger.info(f"Confidence score: {validation_result['confidence']:.3f}")
    
    return constraints, validator

def example_5_dft_setup():
    """Example 5: Set up DFT calculations"""
    logger.info("=== Example 5: DFT Calculation Setup ===")
    
    # Create DFT parameters for topological materials
    dft_params = DFTParameters(
        functional="PBE",
        encut=520.0,  # eV
        kpoint_density=1000.0,  # k-points per Å⁻³
        lsorbit=True,  # Enable spin-orbit coupling
        ispin=2,  # Spin-polarized
        nbands=None,  # Auto-determine number of bands
        nedos=2000
    )
    
    logger.info("DFT Parameters:")
    logger.info(f"  Functional: {dft_params.functional}")
    logger.info(f"  Cutoff energy: {dft_params.encut} eV")
    logger.info(f"  Spin-orbit coupling: {dft_params.lsorbit}")
    logger.info(f"  k-point density: {dft_params.kpoint_density} per Å⁻³")
    
    # Convert to VASP format
    vasp_dict = dft_params.to_vasp_dict()
    logger.info(f"VASP INCAR parameters: {len(vasp_dict)} entries")
    
    return dft_params

def example_6_training_config():
    """Example 6: Set up training configuration"""
    logger.info("=== Example 6: Training Configuration ===")
    
    # Create distributed training configuration
    training_config = DistributedTrainingConfig(
        # Model parameters
        model_config={
            'num_species': 50,
            'hidden_dim': 256,
            'num_topo_classes': 4
        },
        physics_config={
            'topological_class': 'TI',
            'band_gap_range': [0.1, 0.5],
            'target_chern_number': 1,
            'stability_threshold': 0.1,
            'constraint_weight': 1.0
        },
        
        # Training parameters
        batch_size=16,
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_epochs=1000,
        
        # Distributed parameters
        world_size=4,
        mixed_precision=True,
        gradient_compression=True,
        zero_optimizer=True,
        
        # SLURM parameters
        slurm_nodes=1,
        slurm_gpus_per_node=4,
        slurm_time="72:00:00",
        slurm_memory="128G"
    )
    
    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {training_config.batch_size}")
    logger.info(f"  Learning rate: {training_config.learning_rate}")
    logger.info(f"  Max epochs: {training_config.max_epochs}")
    logger.info(f"  World size: {training_config.world_size}")
    logger.info(f"  Mixed precision: {training_config.mixed_precision}")
    logger.info(f"  SLURM nodes: {training_config.slurm_nodes}")
    logger.info(f"  GPUs per node: {training_config.slurm_gpus_per_node}")
    
    return training_config

def main():
    """Run all examples"""
    logger.info("Starting Topological Diffusion Generator Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Basic Hamiltonian
        graphene, eigenvals, eigenvecs = example_1_basic_hamiltonian()
        
        # Example 2: Topological invariants
        invariants = example_2_topological_invariants()
        
        # Example 3: Electric field effects
        stark_calc, band_energies = example_3_electric_field_effects()
        
        # Example 4: Physics constraints
        constraints, validator = example_4_physics_constraints()
        
        # Example 5: DFT setup
        dft_params = example_5_dft_setup()
        
        # Example 6: Training configuration
        training_config = example_6_training_config()
        
        logger.info("=" * 60)
        logger.info("All examples completed successfully!")
        
        return {
            'hamiltonian': graphene,
            'invariants': invariants,
            'stark_calculator': stark_calc,
            'constraints': constraints,
            'dft_params': dft_params,
            'training_config': training_config
        }
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    results = main()