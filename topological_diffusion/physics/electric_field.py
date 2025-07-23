"""
Electric Field Perturbation Engine for Topological Materials
==========================================================

This module implements electric field effects on topological materials:
- Stark effect calculations for uniform electric fields
- Gradient field solvers for spatially varying perturbations  
- Self-consistent Poisson-Schrödinger solver
- Temperature-dependent screening effects
- Field-induced topological phase transitions

Key Physics:
- Linear Stark effect: ΔE = -μ·E (for polar molecules)
- Quadratic Stark effect: ΔE = -½α·E² (for atoms/nonpolar)
- Poisson equation: ∇²φ = -ρ/ε₀ε_r
- Thomas-Fermi screening: k_TF = √(e²n/ε₀ε_r k_B T)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from .quantum_hamiltonian import TopologicalHamiltonian
import logging

logger = logging.getLogger(__name__)

@dataclass
class ElectricFieldConfig:
    """Configuration for electric field calculations"""
    field_strength: float  # V/m
    field_direction: np.ndarray  # Unit vector
    field_type: str = "uniform"  # "uniform", "gradient", "oscillating"
    frequency: float = 0.0  # For oscillating fields (Hz)
    temperature: float = 300.0  # K
    dielectric_constant: float = 1.0
    screening_length: float = 1e-9  # m (Thomas-Fermi screening)
    
    def __post_init__(self):
        # Normalize field direction
        self.field_direction = self.field_direction / np.linalg.norm(self.field_direction)

@dataclass
class MaterialProperties:
    """Material properties for field calculations"""
    dielectric_tensor: np.ndarray  # 3x3 dielectric tensor
    polarizability: np.ndarray  # 3x3 polarizability tensor
    born_charges: Dict[str, np.ndarray]  # Born effective charges by atom type
    carrier_density: float = 1e16  # m^-3
    mobility: float = 1000.0  # cm²/V·s
    band_gap: float = 0.1  # eV
    effective_mass: float = 0.1  # m_e units

class ElectricFieldSolver(ABC):
    """Abstract base class for electric field solvers"""
    
    def __init__(self, field_config: ElectricFieldConfig, material_props: MaterialProperties):
        self.field_config = field_config
        self.material_props = material_props
        self.kb = 8.617e-5  # Boltzmann constant (eV/K)
        self.e = 1.602e-19  # Elementary charge (C)
        self.eps0 = 8.854e-12  # Vacuum permittivity (F/m)
        
    @abstractmethod
    def solve_field(self, coordinates: np.ndarray) -> np.ndarray:
        """Solve for electric field at given coordinates"""
        pass
    
    @abstractmethod
    def compute_potential(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute electric potential at given coordinates"""
        pass

class UniformFieldSolver(ElectricFieldSolver):
    """Solver for uniform electric fields"""
    
    def solve_field(self, coordinates: np.ndarray) -> np.ndarray:
        """Return uniform field at all coordinates"""
        field_vector = self.field_config.field_strength * self.field_config.field_direction
        return np.tile(field_vector, (len(coordinates), 1))
    
    def compute_potential(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute potential for uniform field: φ = -E·r"""
        field_vector = self.field_config.field_strength * self.field_config.field_direction
        return -np.dot(coordinates, field_vector)

class GradientFieldSolver(ElectricFieldSolver):
    """Solver for spatially varying electric fields"""
    
    def __init__(self, field_config: ElectricFieldConfig, material_props: MaterialProperties,
                 gradient_tensor: np.ndarray):
        super().__init__(field_config, material_props)
        self.gradient_tensor = gradient_tensor  # 3x3 tensor ∂E_i/∂x_j
        
    def solve_field(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute field with linear gradient: E(r) = E₀ + ∇E·r"""
        base_field = self.field_config.field_strength * self.field_config.field_direction
        gradient_contribution = coordinates @ self.gradient_tensor.T
        return base_field + gradient_contribution
    
    def compute_potential(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute potential including gradient terms"""
        # φ = -E₀·r - ½r·(∇E)·r
        base_potential = -np.dot(coordinates, 
                                self.field_config.field_strength * self.field_config.field_direction)
        gradient_potential = -0.5 * np.sum(coordinates * (coordinates @ self.gradient_tensor.T), axis=1)
        return base_potential + gradient_potential

class PoissonSolver(ElectricFieldSolver):
    """Self-consistent Poisson-Schrödinger solver"""
    
    def __init__(self, field_config: ElectricFieldConfig, material_props: MaterialProperties,
                 grid_spacing: float = 1e-10, max_iterations: int = 100, tolerance: float = 1e-6):
        super().__init__(field_config, material_props)
        self.grid_spacing = grid_spacing
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def solve_poisson_equation(self, charge_density: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation: ∇²φ = -ρ/ε₀ε_r
        
        Args:
            charge_density: Charge density at grid points (C/m³)
            grid_points: Grid coordinates
            
        Returns:
            potential: Electric potential at grid points (V)
        """
        # Create finite difference Laplacian matrix
        n_points = len(grid_points)
        laplacian = self._build_laplacian_matrix(grid_points)
        
        # Right-hand side: -ρ/ε₀ε_r
        rhs = -charge_density / (self.eps0 * self.material_props.dielectric_tensor[0,0])
        
        # Solve linear system
        potential = spsolve(laplacian, rhs)
        
        return potential
    
    def _build_laplacian_matrix(self, grid_points: np.ndarray) -> sp.csr_matrix:
        """Build finite difference Laplacian matrix"""
        n = len(grid_points)
        h = self.grid_spacing
        
        # Simple 1D Laplacian for demonstration
        # In practice, would need full 3D finite element implementation
        diagonals = [
            np.ones(n-1),      # Upper diagonal
            -2*np.ones(n),     # Main diagonal  
            np.ones(n-1)       # Lower diagonal
        ]
        offsets = [1, 0, -1]
        
        laplacian = sp.diags(diagonals, offsets, shape=(n, n), format='csr') / h**2
        
        return laplacian
    
    def solve_field(self, coordinates: np.ndarray) -> np.ndarray:
        """Solve self-consistent field"""
        # Initial guess for charge density
        charge_density = np.zeros(len(coordinates))
        
        for iteration in range(self.max_iterations):
            # Solve Poisson equation
            potential = self.solve_poisson_equation(charge_density, coordinates)
            
            # Compute electric field: E = -∇φ
            field = -np.gradient(potential, self.grid_spacing)
            
            # Update charge density based on field (simplified)
            new_charge_density = self._compute_charge_density(field, coordinates)
            
            # Check convergence
            if np.max(np.abs(new_charge_density - charge_density)) < self.tolerance:
                logger.info(f"Poisson solver converged in {iteration+1} iterations")
                break
                
            charge_density = new_charge_density
            
        return field.reshape(-1, 1) * np.array([1, 0, 0])  # Simplified 1D case
    
    def _compute_charge_density(self, field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Compute charge density from field (simplified Thomas-Fermi model)"""
        # Thomas-Fermi screening
        screening_length = self.field_config.screening_length
        potential_energy = self.e * np.linalg.norm(field) * coordinates[:, 0]  # Simplified
        
        # Fermi-Dirac distribution (simplified)
        beta = 1 / (self.kb * self.field_config.temperature)
        charge_density = -self.e * self.material_props.carrier_density * np.tanh(beta * potential_energy / 2)
        
        return charge_density
    
    def compute_potential(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute self-consistent potential"""
        field = self.solve_field(coordinates)
        # Integrate field to get potential (simplified)
        return -np.cumsum(field[:, 0]) * self.grid_spacing

class StarkEffectCalculator:
    """Calculator for Stark effect in topological materials"""
    
    def __init__(self, hamiltonian: TopologicalHamiltonian, field_solver: ElectricFieldSolver):
        self.hamiltonian = hamiltonian
        self.field_solver = field_solver
        
    def apply_stark_effect(self, k_point: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Apply Stark effect to Hamiltonian
        
        Args:
            k_point: k-point in reciprocal space
            coordinates: Real space coordinates of atoms
            
        Returns:
            perturbed_hamiltonian: Hamiltonian with Stark effect
        """
        # Get base Hamiltonian
        H0 = self.hamiltonian.build_hamiltonian(k_point)
        
        # Compute electric field at atomic positions
        electric_field = self.field_solver.solve_field(coordinates)
        
        # Build Stark perturbation
        H_stark = self._build_stark_perturbation(electric_field, coordinates)
        
        return H0 + H_stark
    
    def _build_stark_perturbation(self, electric_field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """
        Build Stark effect perturbation Hamiltonian
        
        H_Stark = -μ·E - ½α·E²
        where μ is dipole moment and α is polarizability
        """
        n_orbitals = len(coordinates)
        H_stark = np.zeros((n_orbitals * 2, n_orbitals * 2), dtype=complex)  # Include spin
        
        for i, (coord, field) in enumerate(zip(coordinates, electric_field)):
            # Linear Stark effect (dipole interaction)
            dipole_energy = -np.dot(self._get_dipole_moment(i), field)
            
            # Quadratic Stark effect (polarizability)
            polarizability = self.field_solver.material_props.polarizability
            quadratic_energy = -0.5 * field.T @ polarizability @ field
            
            # Add to diagonal (on-site energies)
            H_stark[2*i, 2*i] += dipole_energy + quadratic_energy
            H_stark[2*i+1, 2*i+1] += dipole_energy + quadratic_energy
            
        return H_stark
    
    def _get_dipole_moment(self, atom_index: int) -> np.ndarray:
        """Get dipole moment for atom (simplified)"""
        # In practice, would compute from wavefunctions
        return np.array([1e-30, 0, 0])  # Simplified dipole moment (C·m)
    
    def compute_field_dependent_bands(self, k_path: np.ndarray, field_strengths: np.ndarray,
                                    coordinates: np.ndarray) -> np.ndarray:
        """
        Compute band structure as function of electric field strength
        
        Returns:
            band_energies: Shape (n_k_points, n_bands, n_field_strengths)
        """
        n_k = len(k_path)
        n_fields = len(field_strengths)
        
        # Get number of bands from base Hamiltonian
        H0 = self.hamiltonian.build_hamiltonian(k_path[0])
        n_bands = H0.shape[0]
        
        band_energies = np.zeros((n_k, n_bands, n_fields))
        
        for i, field_strength in enumerate(field_strengths):
            # Update field strength
            original_strength = self.field_solver.field_config.field_strength
            self.field_solver.field_config.field_strength = field_strength
            
            for j, k_point in enumerate(k_path):
                H_perturbed = self.apply_stark_effect(k_point, coordinates)
                eigenvalues = np.linalg.eigvals(H_perturbed)
                band_energies[j, :, i] = np.sort(np.real(eigenvalues))
                
            # Restore original field strength
            self.field_solver.field_config.field_strength = original_strength
            
        return band_energies

class TopologicalPhaseTransitionDetector:
    """Detector for field-induced topological phase transitions"""
    
    def __init__(self, stark_calculator: StarkEffectCalculator):
        self.stark_calculator = stark_calculator
        
    def find_critical_field(self, k_point: np.ndarray, coordinates: np.ndarray,
                          field_range: Tuple[float, float], tolerance: float = 1e-6) -> float:
        """
        Find critical field strength for gap closing
        
        Args:
            k_point: k-point to monitor (usually high-symmetry point)
            coordinates: Atomic coordinates
            field_range: (min_field, max_field) to search
            tolerance: Convergence tolerance
            
        Returns:
            critical_field: Field strength at gap closing
        """
        def gap_function(field_strength):
            """Function to minimize - returns band gap"""
            # Update field strength
            original_strength = self.stark_calculator.field_solver.field_config.field_strength
            self.stark_calculator.field_solver.field_config.field_strength = field_strength
            
            # Compute perturbed Hamiltonian
            H = self.stark_calculator.apply_stark_effect(k_point, coordinates)
            eigenvalues = np.linalg.eigvals(H)
            eigenvalues = np.sort(np.real(eigenvalues))
            
            # Find gap (assuming half-filled system)
            n_bands = len(eigenvalues)
            gap = eigenvalues[n_bands//2] - eigenvalues[n_bands//2 - 1]
            
            # Restore original field strength
            self.stark_calculator.field_solver.field_config.field_strength = original_strength
            
            return abs(gap)  # Minimize absolute gap
        
        # Find minimum gap
        result = minimize_scalar(gap_function, bounds=field_range, method='bounded')
        
        if result.success:
            critical_field = result.x
            logger.info(f"Found critical field: {critical_field:.2e} V/m")
            return critical_field
        else:
            logger.warning("Critical field search did not converge")
            return None
    
    def map_phase_diagram(self, k_points: np.ndarray, coordinates: np.ndarray,
                         field_strengths: np.ndarray, field_directions: np.ndarray) -> np.ndarray:
        """
        Map topological phase diagram in field parameter space
        
        Returns:
            phase_map: Array indicating topological phase at each parameter point
        """
        n_strengths = len(field_strengths)
        n_directions = len(field_directions)
        phase_map = np.zeros((n_strengths, n_directions))
        
        for i, strength in enumerate(field_strengths):
            for j, direction in enumerate(field_directions):
                # Update field parameters
                self.stark_calculator.field_solver.field_config.field_strength = strength
                self.stark_calculator.field_solver.field_config.field_direction = direction
                
                # Check if system is topological
                is_topological = self._check_topological_phase(k_points, coordinates)
                phase_map[i, j] = int(is_topological)
                
        return phase_map
    
    def _check_topological_phase(self, k_points: np.ndarray, coordinates: np.ndarray) -> bool:
        """
        Check if system is in topological phase (simplified)
        
        In practice, would compute topological invariants
        """
        # Simplified: check if gap is inverted at Γ point
        gamma_point = np.array([0, 0, 0])
        H = self.stark_calculator.apply_stark_effect(gamma_point, coordinates)
        eigenvalues = np.sort(np.real(np.linalg.eigvals(H)))
        
        # Check band inversion (simplified criterion)
        n_bands = len(eigenvalues)
        valence_band = eigenvalues[n_bands//2 - 1]
        conduction_band = eigenvalues[n_bands//2]
        
        # Topological if bands are inverted compared to reference
        # This is a simplified criterion - real implementation would compute Z2 invariant
        return valence_band > conduction_band

class FieldSolverFactory:
    """Factory for creating electric field solvers"""
    
    @staticmethod
    def create_solver(solver_type: str, field_config: ElectricFieldConfig,
                     material_props: MaterialProperties, **kwargs) -> ElectricFieldSolver:
        """
        Create electric field solver
        
        Args:
            solver_type: Type of solver ('uniform', 'gradient', 'poisson')
            field_config: Field configuration
            material_props: Material properties
            **kwargs: Solver-specific parameters
            
        Returns:
            ElectricFieldSolver instance
        """
        solvers = {
            'uniform': UniformFieldSolver,
            'gradient': GradientFieldSolver,
            'poisson': PoissonSolver
        }
        
        if solver_type not in solvers:
            raise ValueError(f"Unknown solver type: {solver_type}")
            
        if solver_type == 'gradient':
            if 'gradient_tensor' not in kwargs:
                raise ValueError("GradientFieldSolver requires 'gradient_tensor' parameter")
            return solvers[solver_type](field_config, material_props, kwargs['gradient_tensor'])
        else:
            return solvers[solver_type](field_config, material_props, **kwargs)

# Utility functions
def compute_thomas_fermi_screening(carrier_density: float, temperature: float, 
                                 dielectric_constant: float) -> float:
    """
    Compute Thomas-Fermi screening length
    
    k_TF = √(e²n/ε₀ε_r k_B T)
    """
    e = 1.602e-19  # C
    eps0 = 8.854e-12  # F/m
    kb = 1.381e-23  # J/K
    
    k_tf_squared = (e**2 * carrier_density) / (eps0 * dielectric_constant * kb * temperature)
    screening_length = 1 / np.sqrt(k_tf_squared)
    
    return screening_length

def create_material_properties(material_name: str) -> MaterialProperties:
    """Create material properties for known materials"""
    
    materials = {
        'bi2se3': MaterialProperties(
            dielectric_tensor=np.diag([100, 100, 30]),  # Anisotropic
            polarizability=np.diag([1e-39, 1e-39, 5e-40]),  # m³
            born_charges={'Bi': np.array([2.5, 0, 0]), 'Se': np.array([-1.25, 0, 0])},
            carrier_density=1e17,  # m⁻³
            mobility=1500,  # cm²/V·s
            band_gap=0.3,  # eV
            effective_mass=0.15
        ),
        'hgte': MaterialProperties(
            dielectric_tensor=np.diag([20, 20, 20]),
            polarizability=np.diag([2e-39, 2e-39, 2e-39]),
            born_charges={'Hg': np.array([1.5, 0, 0]), 'Te': np.array([-1.5, 0, 0])},
            carrier_density=5e16,
            mobility=2000,
            band_gap=0.0,  # Semimetal
            effective_mass=0.03
        ),
        'graphene': MaterialProperties(
            dielectric_tensor=np.diag([2.4, 2.4, 1.0]),
            polarizability=np.diag([1e-40, 1e-40, 1e-41]),
            born_charges={'C': np.array([0, 0, 0])},  # No Born charges
            carrier_density=1e12,  # m⁻²
            mobility=10000,
            band_gap=0.0,
            effective_mass=0.0  # Massless Dirac fermions
        )
    }
    
    if material_name.lower() not in materials:
        raise ValueError(f"Unknown material: {material_name}")
        
    return materials[material_name.lower()]

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create field configuration
    field_config = ElectricFieldConfig(
        field_strength=1e6,  # 1 MV/m
        field_direction=np.array([1, 0, 0]),
        temperature=300.0
    )
    
    # Create material properties
    material_props = create_material_properties('bi2se3')
    
    # Create uniform field solver
    field_solver = FieldSolverFactory.create_solver('uniform', field_config, material_props)
    
    # Test field calculation
    coordinates = np.array([[0, 0, 0], [1e-10, 0, 0], [0, 1e-10, 0]])
    field = field_solver.solve_field(coordinates)
    potential = field_solver.compute_potential(coordinates)
    
    logger.info(f"Electric field shape: {field.shape}")
    logger.info(f"Field strength: {np.linalg.norm(field[0]):.2e} V/m")
    logger.info(f"Potential range: {potential.min():.2e} to {potential.max():.2e} V")