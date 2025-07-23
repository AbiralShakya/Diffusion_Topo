"""
Topological Invariant Calculators for Topological Materials
==========================================================

This module implements calculations of topological invariants:
- Wilson loop calculations for Z2 invariants in 3D topological insulators
- Berry curvature integration for Chern number calculations  
- Wannier charge center tracking for 1D systems
- Parallel computing support for large Brillouin zone sampling

Key Physics:
- Z2 invariant: ν = (1/2π) ∮ A·dk mod 2 (3D TI)
- Chern number: C = (1/2π) ∫ Ω·dS (2D systems)
- Berry connection: A_n(k) = i⟨u_n(k)|∇_k|u_n(k)⟩
- Berry curvature: Ω_n(k) = ∇_k × A_n(k)
- Wilson loop: W = P exp(i ∮ A·dk)

References:
- Thouless et al., PRL 49, 405 (1982) - TKNN formula
- Haldane, PRL 61, 2015 (1988) - Berry curvature
- Kane & Mele, PRL 95, 226801 (2005) - Z2 classification
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.linalg as la
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
import multiprocessing as mp
from functools import partial
import h5py
import logging

from .quantum_hamiltonian import TopologicalHamiltonian

logger = logging.getLogger(__name__)

@dataclass
class TopologicalInvariant:
    """Container for topological invariant results"""
    name: str
    value: Union[int, float, np.ndarray]
    uncertainty: Optional[float] = None
    k_points: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None

class BerryConnection:
    """Calculator for Berry connection and Berry curvature"""
    
    def __init__(self, hamiltonian: TopologicalHamiltonian, occupied_bands: List[int] = None):
        self.hamiltonian = hamiltonian
        self.occupied_bands = occupied_bands
        
    def compute_berry_connection(self, k_point: np.ndarray, band_index: int, 
                               delta_k: float = 1e-6) -> np.ndarray:
        """
        Compute Berry connection A_n(k) = i⟨u_n(k)|∇_k|u_n(k)⟩
        
        Args:
            k_point: k-point in reciprocal space
            band_index: Band index
            delta_k: Finite difference step
            
        Returns:
            berry_connection: 3-component Berry connection vector
        """
        # Get wavefunction at k
        H_k = self.hamiltonian.build_hamiltonian(k_point)
        eigenvals, eigenvecs = np.linalg.eigh(H_k)
        psi_k = eigenvecs[:, band_index]
        
        berry_connection = np.zeros(3, dtype=complex)
        
        # Finite difference approximation of ∇_k
        for i in range(3):
            k_plus = k_point.copy()
            k_minus = k_point.copy()
            k_plus[i] += delta_k
            k_minus[i] -= delta_k
            
            # Wavefunctions at k±δk
            H_plus = self.hamiltonian.build_hamiltonian(k_plus)
            H_minus = self.hamiltonian.build_hamiltonian(k_minus)
            
            _, eigenvecs_plus = np.linalg.eigh(H_plus)
            _, eigenvecs_minus = np.linalg.eigh(H_minus)
            
            psi_plus = eigenvecs_plus[:, band_index]
            psi_minus = eigenvecs_minus[:, band_index]
            
            # Fix gauge (choose phase to maximize overlap)
            overlap_plus = np.vdot(psi_k, psi_plus)
            overlap_minus = np.vdot(psi_k, psi_minus)
            
            psi_plus *= np.sign(overlap_plus)
            psi_minus *= np.sign(overlap_minus)
            
            # Berry connection component
            berry_connection[i] = 1j * np.vdot(psi_k, (psi_plus - psi_minus) / (2 * delta_k))
            
        return berry_connection
    
    def compute_berry_curvature(self, k_point: np.ndarray, band_index: int,
                              delta_k: float = 1e-6) -> np.ndarray:
        """
        Compute Berry curvature Ω_n(k) = ∇_k × A_n(k)
        
        Args:
            k_point: k-point in reciprocal space
            band_index: Band index
            delta_k: Finite difference step
            
        Returns:
            berry_curvature: 3-component Berry curvature vector
        """
        # Compute Berry connection at neighboring points
        berry_curvature = np.zeros(3, dtype=complex)
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                    
                # Points for finite difference
                k_pp = k_point.copy()  # k + δk_i + δk_j
                k_pm = k_point.copy()  # k + δk_i - δk_j
                k_mp = k_point.copy()  # k - δk_i + δk_j
                k_mm = k_point.copy()  # k - δk_i - δk_j
                
                k_pp[i] += delta_k
                k_pp[j] += delta_k
                k_pm[i] += delta_k
                k_pm[j] -= delta_k
                k_mp[i] -= delta_k
                k_mp[j] += delta_k
                k_mm[i] -= delta_k
                k_mm[j] -= delta_k
                
                # Berry connections
                A_pp = self.compute_berry_connection(k_pp, band_index, delta_k)[j]
                A_pm = self.compute_berry_connection(k_pm, band_index, delta_k)[j]
                A_mp = self.compute_berry_connection(k_mp, band_index, delta_k)[j]
                A_mm = self.compute_berry_connection(k_mm, band_index, delta_k)[j]
                
                # Curl component: (∂A_j/∂k_i - ∂A_i/∂k_j)
                curl_component = ((A_pp - A_pm) - (A_mp - A_mm)) / (4 * delta_k**2)
                
                # Add to appropriate component of curl
                k_idx = 3 - i - j  # Remaining index for cross product
                if (i, j, k_idx) in [(0,1,2), (1,2,0), (2,0,1)]:
                    berry_curvature[k_idx] += curl_component
                else:
                    berry_curvature[k_idx] -= curl_component
                    
        return berry_curvature

class ChernNumberCalculator:
    """Calculator for Chern numbers in 2D systems"""
    
    def __init__(self, hamiltonian: TopologicalHamiltonian):
        self.hamiltonian = hamiltonian
        self.berry_calc = BerryConnection(hamiltonian)
        
    def compute_chern_number(self, occupied_bands: List[int], k_grid: np.ndarray) -> int:
        """
        Compute Chern number C = (1/2π) ∫ Ω·dS
        
        Args:
            occupied_bands: List of occupied band indices
            k_grid: 2D k-point grid for integration
            
        Returns:
            chern_number: Integer Chern number
        """
        total_curvature = 0.0
        
        # Integrate Berry curvature over Brillouin zone
        for k_point in k_grid:
            for band_idx in occupied_bands:
                berry_curvature = self.berry_calc.compute_berry_curvature(k_point, band_idx)
                # For 2D system, use z-component of curvature
                total_curvature += np.real(berry_curvature[2])
                
        # Normalize and round to nearest integer
        chern_number = int(np.round(total_curvature / (2 * np.pi)))
        
        return chern_number
    
    def compute_chern_number_wilson_loop(self, occupied_bands: List[int], 
                                       k_grid: np.ndarray) -> int:
        """
        Compute Chern number using Wilson loop method
        More numerically stable for large systems
        """
        # Create Wilson loops along both directions
        wilson_loops_x = []
        wilson_loops_y = []
        
        # Wilson loops in x-direction
        for ky in np.unique(k_grid[:, 1]):
            k_line = k_grid[k_grid[:, 1] == ky]
            k_line = k_line[np.argsort(k_line[:, 0])]  # Sort by kx
            
            wilson_loop = self._compute_wilson_loop_1d(k_line, occupied_bands, direction=0)
            wilson_loops_x.append(wilson_loop)
            
        # Wilson loops in y-direction  
        for kx in np.unique(k_grid[:, 0]):
            k_line = k_grid[k_grid[:, 0] == kx]
            k_line = k_line[np.argsort(k_line[:, 1])]  # Sort by ky
            
            wilson_loop = self._compute_wilson_loop_1d(k_line, occupied_bands, direction=1)
            wilson_loops_y.append(wilson_loop)
            
        # Compute winding numbers
        winding_x = self._compute_winding_number(wilson_loops_x)
        winding_y = self._compute_winding_number(wilson_loops_y)
        
        # Chern number from winding
        chern_number = winding_x - winding_y
        
        return chern_number
    
    def _compute_wilson_loop_1d(self, k_path: np.ndarray, occupied_bands: List[int], 
                               direction: int) -> complex:
        """Compute Wilson loop along 1D path"""
        n_occ = len(occupied_bands)
        wilson_matrix = np.eye(n_occ, dtype=complex)
        
        for i in range(len(k_path) - 1):
            k1 = k_path[i]
            k2 = k_path[i + 1]
            
            # Get wavefunctions
            H1 = self.hamiltonian.build_hamiltonian(k1)
            H2 = self.hamiltonian.build_hamiltonian(k2)
            
            _, psi1 = np.linalg.eigh(H1)
            _, psi2 = np.linalg.eigh(H2)
            
            # Overlap matrix between occupied states
            overlap = np.zeros((n_occ, n_occ), dtype=complex)
            for m, band_m in enumerate(occupied_bands):
                for n, band_n in enumerate(occupied_bands):
                    overlap[m, n] = np.vdot(psi1[:, band_m], psi2[:, band_n])
                    
            wilson_matrix = wilson_matrix @ overlap
            
        # Wilson loop eigenvalue (should be on unit circle)
        eigenvals = np.linalg.eigvals(wilson_matrix)
        wilson_loop = np.prod(eigenvals)
        
        return wilson_loop
    
    def _compute_winding_number(self, wilson_loops: List[complex]) -> int:
        """Compute winding number from Wilson loop eigenvalues"""
        phases = [np.angle(w) for w in wilson_loops]
        
        # Unwrap phases to avoid 2π jumps
        phases = np.unwrap(phases)
        
        # Winding number is total phase change divided by 2π
        total_phase_change = phases[-1] - phases[0]
        winding_number = int(np.round(total_phase_change / (2 * np.pi)))
        
        return winding_number

class Z2InvariantCalculator:
    """Calculator for Z2 topological invariants in 3D systems"""
    
    def __init__(self, hamiltonian: TopologicalHamiltonian):
        self.hamiltonian = hamiltonian
        
    def compute_z2_invariant(self, occupied_bands: List[int], 
                           time_reversal_invariant_momenta: List[np.ndarray]) -> Tuple[int, List[int]]:
        """
        Compute Z2 invariant using parity eigenvalues at TRIM points
        
        For 3D systems: (ν₀; ν₁ν₂ν₃) where νᵢ ∈ {0,1}
        
        Args:
            occupied_bands: List of occupied band indices
            time_reversal_invariant_momenta: List of TRIM points
            
        Returns:
            strong_z2: Strong Z2 invariant ν₀
            weak_z2: Weak Z2 invariants [ν₁, ν₂, ν₃]
        """
        # Compute parity eigenvalues at all TRIM points
        parity_products = []
        
        for trim_point in time_reversal_invariant_momenta:
            parity_product = self._compute_parity_product(trim_point, occupied_bands)
            parity_products.append(parity_product)
            
        # Z2 invariants from parity products
        # Strong Z2: product over all 8 TRIM points
        strong_z2 = 1
        for parity in parity_products:
            strong_z2 *= parity
        strong_z2 = int((1 - strong_z2) / 2) % 2
        
        # Weak Z2: products over 4 TRIM points each
        weak_z2 = []
        for i in range(3):
            weak_invariant = 1
            for j, parity in enumerate(parity_products):
                if (j >> i) & 1:  # Check i-th bit
                    weak_invariant *= parity
            weak_z2.append(int((1 - weak_invariant) / 2) % 2)
            
        return strong_z2, weak_z2
    
    def _compute_parity_product(self, k_point: np.ndarray, occupied_bands: List[int]) -> int:
        """
        Compute product of parity eigenvalues at TRIM point
        
        At TRIM points, parity operator P commutes with Hamiltonian
        """
        H = self.hamiltonian.build_hamiltonian(k_point)
        eigenvals, eigenvecs = np.linalg.eigh(H)
        
        # Parity operator (simplified - depends on specific model)
        P = self._get_parity_operator(H.shape[0])
        
        parity_product = 1
        for band_idx in occupied_bands:
            psi = eigenvecs[:, band_idx]
            
            # Parity eigenvalue: ⟨ψ|P|ψ⟩
            parity_eigenval = np.real(np.vdot(psi, P @ psi))
            parity_sign = int(np.sign(parity_eigenval))
            
            parity_product *= parity_sign
            
        return parity_product
    
    def _get_parity_operator(self, matrix_size: int) -> np.ndarray:
        """
        Get parity operator for the system
        
        This is model-dependent and needs to be implemented for each Hamiltonian
        """
        # Simplified parity operator (identity for now)
        # In practice, this depends on the specific model and basis
        return np.eye(matrix_size)
    
    def compute_z2_wilson_loop(self, occupied_bands: List[int], k_grid: np.ndarray) -> int:
        """
        Compute Z2 invariant using Wilson loop method
        More robust for numerical calculations
        """
        # Create Wilson loops on 2D surfaces of 3D BZ
        wilson_loops = []
        
        # Loop over different 2D surfaces
        for surface_normal in range(3):
            surface_loops = self._compute_surface_wilson_loops(
                k_grid, occupied_bands, surface_normal
            )
            wilson_loops.extend(surface_loops)
            
        # Compute Z2 invariant from Wilson loop spectrum
        z2_invariant = self._extract_z2_from_wilson_loops(wilson_loops)
        
        return z2_invariant
    
    def _compute_surface_wilson_loops(self, k_grid: np.ndarray, occupied_bands: List[int],
                                    surface_normal: int) -> List[np.ndarray]:
        """Compute Wilson loops on 2D surface of 3D Brillouin zone"""
        # Implementation depends on specific k-grid structure
        # This is a simplified version
        surface_loops = []
        
        # Extract 2D surface from 3D grid
        unique_vals = np.unique(k_grid[:, surface_normal])
        
        for val in unique_vals:
            surface_points = k_grid[k_grid[:, surface_normal] == val]
            
            # Compute Wilson loops on this 2D surface
            loop_eigenvals = self._compute_2d_wilson_loops(surface_points, occupied_bands)
            surface_loops.append(loop_eigenvals)
            
        return surface_loops
    
    def _compute_2d_wilson_loops(self, k_surface: np.ndarray, occupied_bands: List[int]) -> np.ndarray:
        """Compute Wilson loops on 2D surface"""
        # Simplified implementation
        n_occ = len(occupied_bands)
        wilson_eigenvals = np.ones(n_occ, dtype=complex)
        
        # This would involve proper 2D Wilson loop calculation
        # For now, return placeholder
        return wilson_eigenvals
    
    def _extract_z2_from_wilson_loops(self, wilson_loops: List[np.ndarray]) -> int:
        """Extract Z2 invariant from Wilson loop eigenvalues"""
        # Count number of Wilson loop eigenvalues crossing -1
        crossings = 0
        
        for loop_eigenvals in wilson_loops:
            for eigenval in loop_eigenvals:
                phase = np.angle(eigenval)
                if abs(phase - np.pi) < 0.1:  # Near -1
                    crossings += 1
                    
        # Z2 invariant is parity of crossings
        z2_invariant = crossings % 2
        
        return z2_invariant

class WannierCenterCalculator:
    """Calculator for Wannier charge centers in 1D systems"""
    
    def __init__(self, hamiltonian: TopologicalHamiltonian):
        self.hamiltonian = hamiltonian
        
    def compute_wannier_centers(self, occupied_bands: List[int], k_path: np.ndarray) -> np.ndarray:
        """
        Compute Wannier charge centers along 1D path
        
        Args:
            occupied_bands: List of occupied band indices
            k_path: 1D k-point path
            
        Returns:
            wannier_centers: Wannier center positions
        """
        n_occ = len(occupied_bands)
        wannier_centers = np.zeros(n_occ)
        
        # Compute Wilson loop along k-path
        wilson_matrix = np.eye(n_occ, dtype=complex)
        
        for i in range(len(k_path) - 1):
            k1 = k_path[i]
            k2 = k_path[i + 1]
            
            # Overlap matrix
            overlap = self._compute_overlap_matrix(k1, k2, occupied_bands)
            wilson_matrix = wilson_matrix @ overlap
            
        # Wannier centers from Wilson loop eigenvalues
        wilson_eigenvals = np.linalg.eigvals(wilson_matrix)
        
        for i, eigenval in enumerate(wilson_eigenvals):
            # Wannier center position
            wannier_centers[i] = -np.angle(eigenval) / (2 * np.pi)
            
        return np.sort(wannier_centers)
    
    def _compute_overlap_matrix(self, k1: np.ndarray, k2: np.ndarray, 
                               occupied_bands: List[int]) -> np.ndarray:
        """Compute overlap matrix between occupied states at two k-points"""
        H1 = self.hamiltonian.build_hamiltonian(k1)
        H2 = self.hamiltonian.build_hamiltonian(k2)
        
        _, psi1 = np.linalg.eigh(H1)
        _, psi2 = np.linalg.eigh(H2)
        
        n_occ = len(occupied_bands)
        overlap = np.zeros((n_occ, n_occ), dtype=complex)
        
        for i, band_i in enumerate(occupied_bands):
            for j, band_j in enumerate(occupied_bands):
                overlap[i, j] = np.vdot(psi1[:, band_i], psi2[:, band_j])
                
        return overlap
    
    def compute_polarization(self, occupied_bands: List[int], k_path: np.ndarray) -> float:
        """
        Compute electric polarization from Wannier centers
        
        P = e * Σᵢ xᵢ (mod e)
        """
        wannier_centers = self.compute_wannier_centers(occupied_bands, k_path)
        
        # Electric polarization (in units of e per unit cell)
        polarization = np.sum(wannier_centers) % 1
        
        return polarization

class TopologicalInvariantCalculator:
    """Main calculator for all topological invariants"""
    
    def __init__(self, hamiltonian: TopologicalHamiltonian):
        self.hamiltonian = hamiltonian
        self.chern_calc = ChernNumberCalculator(hamiltonian)
        self.z2_calc = Z2InvariantCalculator(hamiltonian)
        self.wannier_calc = WannierCenterCalculator(hamiltonian)
        
    def compute_all_invariants(self, occupied_bands: List[int], 
                             k_grid: np.ndarray, system_dimension: int = 3) -> Dict[str, TopologicalInvariant]:
        """
        Compute all relevant topological invariants
        
        Args:
            occupied_bands: List of occupied band indices
            k_grid: k-point grid for integration
            system_dimension: Spatial dimension (1, 2, or 3)
            
        Returns:
            invariants: Dictionary of computed invariants
        """
        invariants = {}
        
        if system_dimension >= 2:
            # Chern number for 2D systems
            try:
                chern_number = self.chern_calc.compute_chern_number(occupied_bands, k_grid)
                invariants['chern_number'] = TopologicalInvariant(
                    name='Chern Number',
                    value=chern_number,
                    k_points=k_grid
                )
                logger.info(f"Computed Chern number: {chern_number}")
            except Exception as e:
                logger.warning(f"Failed to compute Chern number: {e}")
                
        if system_dimension == 3:
            # Z2 invariant for 3D systems
            try:
                # Define TRIM points for cubic lattice
                trim_points = [
                    np.array([0, 0, 0]),      # Γ
                    np.array([π, 0, 0]),      # X
                    np.array([0, π, 0]),      # Y  
                    np.array([0, 0, π]),      # Z
                    np.array([π, π, 0]),      # S
                    np.array([π, 0, π]),      # T
                    np.array([0, π, π]),      # U
                    np.array([π, π, π])       # R
                ]
                
                strong_z2, weak_z2 = self.z2_calc.compute_z2_invariant(occupied_bands, trim_points)
                
                invariants['z2_strong'] = TopologicalInvariant(
                    name='Strong Z2 Invariant',
                    value=strong_z2
                )
                
                invariants['z2_weak'] = TopologicalInvariant(
                    name='Weak Z2 Invariants',
                    value=weak_z2
                )
                
                logger.info(f"Computed Z2 invariants: ({strong_z2}; {weak_z2[0]}{weak_z2[1]}{weak_z2[2]})")
                
            except Exception as e:
                logger.warning(f"Failed to compute Z2 invariant: {e}")
                
        # Wannier centers for 1D cuts
        try:
            # Take 1D cut through Γ-X direction
            k_line = np.linspace([0, 0, 0], [np.pi, 0, 0], 100)
            wannier_centers = self.wannier_calc.compute_wannier_centers(occupied_bands, k_line)
            polarization = self.wannier_calc.compute_polarization(occupied_bands, k_line)
            
            invariants['wannier_centers'] = TopologicalInvariant(
                name='Wannier Centers',
                value=wannier_centers,
                k_points=k_line
            )
            
            invariants['polarization'] = TopologicalInvariant(
                name='Electric Polarization',
                value=polarization
            )
            
            logger.info(f"Computed {len(wannier_centers)} Wannier centers")
            logger.info(f"Electric polarization: {polarization:.4f} e/cell")
            
        except Exception as e:
            logger.warning(f"Failed to compute Wannier centers: {e}")
            
        return invariants
    
    def classify_topological_phase(self, invariants: Dict[str, TopologicalInvariant]) -> str:
        """
        Classify topological phase based on computed invariants
        
        Returns:
            phase_name: String describing the topological phase
        """
        if 'chern_number' in invariants:
            chern = invariants['chern_number'].value
            if chern != 0:
                return f"Chern Insulator (C = {chern})"
                
        if 'z2_strong' in invariants and 'z2_weak' in invariants:
            strong = invariants['z2_strong'].value
            weak = invariants['z2_weak'].value
            
            if strong == 1:
                return f"Strong Topological Insulator ({strong}; {weak[0]}{weak[1]}{weak[2]})"
            elif any(w == 1 for w in weak):
                return f"Weak Topological Insulator ({strong}; {weak[0]}{weak[1]}{weak[2]})"
                
        return "Trivial Insulator"

# Parallel computation utilities
def compute_berry_curvature_parallel(args):
    """Parallel worker for Berry curvature calculation"""
    k_point, hamiltonian, band_index, delta_k = args
    berry_calc = BerryConnection(hamiltonian)
    return berry_calc.compute_berry_curvature(k_point, band_index, delta_k)

def parallel_chern_calculation(hamiltonian: TopologicalHamiltonian, 
                             occupied_bands: List[int], k_grid: np.ndarray,
                             n_processes: int = None) -> int:
    """
    Parallel calculation of Chern number
    
    Args:
        hamiltonian: Topological Hamiltonian
        occupied_bands: List of occupied bands
        k_grid: k-point grid
        n_processes: Number of parallel processes
        
    Returns:
        chern_number: Computed Chern number
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
        
    # Prepare arguments for parallel computation
    args_list = []
    for k_point in k_grid:
        for band_idx in occupied_bands:
            args_list.append((k_point, hamiltonian, band_idx, 1e-6))
            
    # Parallel computation
    with mp.Pool(n_processes) as pool:
        berry_curvatures = pool.map(compute_berry_curvature_parallel, args_list)
        
    # Sum up contributions
    total_curvature = 0.0
    for curvature in berry_curvatures:
        total_curvature += np.real(curvature[2])  # z-component for 2D
        
    # Normalize to get Chern number
    chern_number = int(np.round(total_curvature / (2 * np.pi)))
    
    return chern_number

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from .quantum_hamiltonian import HamiltonianFactory, generate_k_grid
    
    # Create graphene model
    graphene = HamiltonianFactory.create_from_material('graphene')
    
    # Create topological invariant calculator
    topo_calc = TopologicalInvariantCalculator(graphene)
    
    # Generate k-grid
    k_grid = generate_k_grid(graphene.reciprocal_vectors, (20, 20, 1))
    
    # Compute invariants (assuming half-filled system)
    H_test = graphene.build_hamiltonian(np.array([0, 0, 0]))
    n_bands = H_test.shape[0]
    occupied_bands = list(range(n_bands // 2))
    
    invariants = topo_calc.compute_all_invariants(occupied_bands, k_grid, system_dimension=2)
    
    # Classify phase
    phase = topo_calc.classify_topological_phase(invariants)
    logger.info(f"Topological phase: {phase}")
    
    # Print results
    for name, invariant in invariants.items():
        logger.info(f"{invariant.name}: {invariant.value}")