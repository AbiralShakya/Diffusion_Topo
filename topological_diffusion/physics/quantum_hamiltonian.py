"""
Quantum Hamiltonian Framework for Topological Materials
=====================================================

This module implements tight-binding Hamiltonians with spin-orbit coupling
for topological insulators and related materials. Supports multi-orbital
systems and various topological models.

Based on:
- Kane-Mele model for graphene-like systems
- Bernevig-Hughes-Zhang (BHZ) model for HgTe quantum wells  
- Fu-Kane-Mele model for 3D topological insulators
- Weyl semimetal models with broken symmetries
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import h5py
import logging

logger = logging.getLogger(__name__)

@dataclass
class LatticeParameters:
    """Crystal lattice parameters"""
    a: float  # Lattice constant a
    b: float  # Lattice constant b  
    c: float  # Lattice constant c
    alpha: float = 90.0  # Angle alpha (degrees)
    beta: float = 90.0   # Angle beta (degrees)
    gamma: float = 90.0  # Angle gamma (degrees)
    
    def to_vectors(self) -> np.ndarray:
        """Convert to lattice vectors"""
        alpha_rad = np.radians(self.alpha)
        beta_rad = np.radians(self.beta)
        gamma_rad = np.radians(self.gamma)
        
        # Standard crystallographic convention
        a1 = np.array([self.a, 0, 0])
        a2 = np.array([self.b * np.cos(gamma_rad), self.b * np.sin(gamma_rad), 0])
        
        cx = self.c * np.cos(beta_rad)
        cy = self.c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        cz = self.c * np.sqrt(1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2 + 
                             2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        a3 = np.array([cx, cy, cz])
        
        return np.array([a1, a2, a3])

@dataclass 
class SpinOrbitParameters:
    """Spin-orbit coupling parameters"""
    rashba_alpha: float = 0.0      # Rashba SOC strength
    dresselhaus_beta: float = 0.0  # Dresselhaus SOC strength
    intrinsic_lambda: float = 0.0  # Intrinsic SOC (Kane-Mele type)
    atomic_soc: Dict[str, float] = None  # Atomic SOC by element
    
    def __post_init__(self):
        if self.atomic_soc is None:
            # Default atomic SOC values (eV)
            self.atomic_soc = {
                'Bi': 1.5,   # Strong SOC in Bi
                'Pb': 1.2,   # Strong SOC in Pb
                'Hg': 0.8,   # Moderate SOC in Hg
                'Te': 0.4,   # Moderate SOC in Te
                'Se': 0.3,   # Moderate SOC in Se
                'C': 0.006,  # Weak SOC in C (graphene)
                'Si': 0.02,  # Weak SOC in Si
                'Ge': 0.3,   # Moderate SOC in Ge
            }

class PauliMatrices:
    """Pauli matrices and common combinations"""
    
    @staticmethod
    def sigma_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def sigma_y():
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def sigma_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def sigma_0():
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @classmethod
    def all_pauli(cls):
        return [cls.sigma_0(), cls.sigma_x(), cls.sigma_y(), cls.sigma_z()]

class TopologicalHamiltonian(ABC):
    """
    Abstract base class for topological Hamiltonians
    """
    
    def __init__(self, lattice_params: LatticeParameters, soc_params: SpinOrbitParameters):
        self.lattice_params = lattice_params
        self.soc_params = soc_params
        self.lattice_vectors = lattice_params.to_vectors()
        self.reciprocal_vectors = self._compute_reciprocal_vectors()
        self.pauli = PauliMatrices()
        
    def _compute_reciprocal_vectors(self) -> np.ndarray:
        """Compute reciprocal lattice vectors"""
        a1, a2, a3 = self.lattice_vectors
        volume = np.dot(a1, np.cross(a2, a3))
        
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume  
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        
        return np.array([b1, b2, b3])
    
    @abstractmethod
    def build_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Build Hamiltonian matrix at given k-point"""
        pass
    
    @abstractmethod
    def get_band_structure(self, k_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute band structure along k-path"""
        pass

class KaneMeleModel(TopologicalHamiltonian):
    """
    Kane-Mele model for graphene with spin-orbit coupling
    
    H = -t ∑⟨i,j⟩,σ (c†ᵢσ cⱼσ + h.c.) + iλSO ∑⟨⟨i,j⟩⟩,σσ' sᵢⱼ σᶻσσ' c†ᵢσ cⱼσ'
    
    Reference: Kane & Mele, PRL 95, 226801 (2005)
    """
    
    def __init__(self, lattice_params: LatticeParameters, soc_params: SpinOrbitParameters,
                 t: float = 2.8, t2: float = 0.0):
        super().__init__(lattice_params, soc_params)
        self.t = t  # Nearest neighbor hopping
        self.t2 = t2  # Next-nearest neighbor hopping
        self.lambda_so = soc_params.intrinsic_lambda
        
        # Graphene lattice vectors
        self.delta = np.array([
            [1, 0],
            [-1/2, np.sqrt(3)/2], 
            [-1/2, -np.sqrt(3)/2]
        ]) * lattice_params.a / np.sqrt(3)
        
    def build_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Build Kane-Mele Hamiltonian at k-point"""
        kx, ky = k_point[:2]
        
        # Nearest neighbor terms
        f_k = np.sum([np.exp(1j * np.dot(k_point[:2], delta)) for delta in self.delta])
        
        # Pauli matrices
        sigma_x, sigma_y, sigma_z, sigma_0 = self.pauli.all_pauli()
        
        # Kinetic energy term
        H_kinetic = -self.t * np.kron(
            np.array([[0, f_k], [np.conj(f_k), 0]]), sigma_0
        )
        
        # Spin-orbit coupling term
        if self.lambda_so != 0:
            # Next-nearest neighbor vectors
            b1 = self.delta[0] - self.delta[1]
            b2 = self.delta[1] - self.delta[2] 
            b3 = self.delta[2] - self.delta[0]
            
            # SOC phase factors
            phi_so = 2 * np.sum([
                np.sin(np.dot(k_point[:2], b)) for b in [b1, b2, b3]
            ])
            
            H_soc = self.lambda_so * phi_so * np.kron(
                np.array([[1, 0], [0, -1]]), sigma_z
            )
        else:
            H_soc = np.zeros_like(H_kinetic)
            
        return H_kinetic + H_soc
    
    def get_band_structure(self, k_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute band structure along k-path"""
        eigenvalues = []
        eigenvectors = []
        
        for k in k_path:
            H = self.build_hamiltonian(k)
            evals, evecs = np.linalg.eigh(H)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
            
        return np.array(eigenvalues), np.array(eigenvectors)

class BHZModel(TopologicalHamiltonian):
    """
    Bernevig-Hughes-Zhang model for HgTe quantum wells
    
    H(k) = ε(k) + d₁(k)σₓ + d₂(k)σᵧ + d₃(k)σᵤ
    
    Reference: Bernevig, Hughes & Zhang, Science 314, 1757 (2006)
    """
    
    def __init__(self, lattice_params: LatticeParameters, soc_params: SpinOrbitParameters,
                 A: float = 3.65, B: float = -68.6, C: float = 0.0, D: float = -51.2, M: float = -0.01):
        super().__init__(lattice_params, soc_params)
        self.A = A  # Linear dispersion coefficient
        self.B = B  # Quadratic dispersion coefficient  
        self.C = C  # Quadratic dispersion coefficient
        self.D = D  # Quadratic dispersion coefficient
        self.M = M  # Mass term
        
    def build_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Build BHZ Hamiltonian at k-point"""
        kx, ky = k_point[:2]
        k_squared = kx**2 + ky**2
        
        # Pauli matrices
        sigma_x, sigma_y, sigma_z, sigma_0 = self.pauli.all_pauli()
        
        # BHZ Hamiltonian components
        epsilon = self.C + self.D * k_squared
        d1 = self.A * kx
        d2 = self.A * ky  
        d3 = self.M + self.B * k_squared
        
        # Build Hamiltonian
        H = (epsilon * sigma_0 + d1 * sigma_x + d2 * sigma_y + d3 * sigma_z)
        
        return H
    
    def get_band_structure(self, k_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute band structure along k-path"""
        eigenvalues = []
        eigenvectors = []
        
        for k in k_path:
            H = self.build_hamiltonian(k)
            evals, evecs = np.linalg.eigh(H)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
            
        return np.array(eigenvalues), np.array(eigenvectors)

class FuKaneMeleModel(TopologicalHamiltonian):
    """
    Fu-Kane-Mele model for 3D topological insulators (Bi₂Se₃, Bi₂Te₃)
    
    Effective low-energy model around Γ point with strong spin-orbit coupling
    
    Reference: Zhang et al., Nature Physics 5, 438 (2009)
    """
    
    def __init__(self, lattice_params: LatticeParameters, soc_params: SpinOrbitParameters,
                 v_f: float = 5.0, lambda_so: float = 0.3, m: float = 0.28):
        super().__init__(lattice_params, soc_params)
        self.v_f = v_f      # Fermi velocity
        self.lambda_so = lambda_so  # SOC strength
        self.m = m          # Mass gap
        
    def build_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Build 3D TI Hamiltonian at k-point"""
        kx, ky, kz = k_point if len(k_point) >= 3 else [*k_point, 0]
        
        # Pauli matrices for orbital and spin degrees of freedom
        sigma_x, sigma_y, sigma_z, sigma_0 = self.pauli.all_pauli()
        tau_x, tau_y, tau_z, tau_0 = self.pauli.all_pauli()  # Orbital pseudospin
        
        # 3D TI Hamiltonian (4x4 matrix)
        H = (self.v_f * kx * np.kron(tau_z, sigma_x) + 
             self.v_f * ky * np.kron(tau_z, sigma_y) +
             (self.m + self.v_f * kz) * np.kron(tau_x, sigma_0) +
             self.lambda_so * kz * np.kron(tau_0, sigma_z))
        
        return H
    
    def get_band_structure(self, k_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute band structure along k-path"""
        eigenvalues = []
        eigenvectors = []
        
        for k in k_path:
            H = self.build_hamiltonian(k)
            evals, evecs = np.linalg.eigh(H)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
            
        return np.array(eigenvalues), np.array(eigenvectors)

class WeylSemimetalModel(TopologicalHamiltonian):
    """
    Weyl semimetal model with broken time-reversal or inversion symmetry
    
    H = v_f (k·σ) + b·σ (magnetic field term)
    """
    
    def __init__(self, lattice_params: LatticeParameters, soc_params: SpinOrbitParameters,
                 v_f: float = 1.0, b_field: np.ndarray = None, weyl_separation: float = 0.1):
        super().__init__(lattice_params, soc_params)
        self.v_f = v_f
        self.b_field = b_field if b_field is not None else np.array([0, 0, 0.1])
        self.weyl_separation = weyl_separation
        
    def build_hamiltonian(self, k_point: np.ndarray) -> np.ndarray:
        """Build Weyl semimetal Hamiltonian"""
        kx, ky, kz = k_point if len(k_point) >= 3 else [*k_point, 0]
        
        # Shift k-points to create Weyl nodes
        kz_plus = kz + self.weyl_separation/2
        kz_minus = kz - self.weyl_separation/2
        
        sigma_x, sigma_y, sigma_z, sigma_0 = self.pauli.all_pauli()
        
        # Weyl Hamiltonian (2x2 for each chirality)
        H_plus = self.v_f * (kx * sigma_x + ky * sigma_y + kz_plus * sigma_z) + np.dot(self.b_field, [sigma_x, sigma_y, sigma_z])
        H_minus = self.v_f * (kx * sigma_x + ky * sigma_y + kz_minus * sigma_z) - np.dot(self.b_field, [sigma_x, sigma_y, sigma_z])
        
        # Block diagonal Hamiltonian
        H = np.block([[H_plus, np.zeros((2,2))], [np.zeros((2,2)), H_minus]])
        
        return H
    
    def get_band_structure(self, k_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute band structure along k-path"""
        eigenvalues = []
        eigenvectors = []
        
        for k in k_path:
            H = self.build_hamiltonian(k)
            evals, evecs = np.linalg.eigh(H)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
            
        return np.array(eigenvalues), np.array(eigenvectors)

class BlochHamiltonianSolver:
    """
    Solver for Bloch Hamiltonians with periodic boundary conditions
    Supports parallel eigenvalue calculations for HPC environments
    """
    
    def __init__(self, hamiltonian: TopologicalHamiltonian, use_sparse: bool = True):
        self.hamiltonian = hamiltonian
        self.use_sparse = use_sparse
        
    def solve_eigenvalue_problem(self, k_points: np.ndarray, 
                                num_bands: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for multiple k-points
        
        Args:
            k_points: Array of k-points, shape (N, 3)
            num_bands: Number of bands to compute (None for all)
            
        Returns:
            eigenvalues: Shape (N, num_bands)
            eigenvectors: Shape (N, num_bands, matrix_size)
        """
        eigenvalues = []
        eigenvectors = []
        
        for k in k_points:
            H = self.hamiltonian.build_hamiltonian(k)
            
            if self.use_sparse and sp.issparse(H):
                if num_bands is None:
                    num_bands = H.shape[0] // 2
                evals, evecs = eigsh(H, k=num_bands, which='SM')
            else:
                evals, evecs = np.linalg.eigh(H)
                if num_bands is not None:
                    evals = evals[:num_bands]
                    evecs = evecs[:, :num_bands]
                    
            eigenvalues.append(evals)
            eigenvectors.append(evecs)
            
        return np.array(eigenvalues), np.array(eigenvectors)
    
    def compute_dos(self, k_grid: np.ndarray, energy_range: Tuple[float, float], 
                   num_points: int = 1000, broadening: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute density of states using Gaussian broadening
        
        Args:
            k_grid: Grid of k-points
            energy_range: (E_min, E_max) 
            num_points: Number of energy points
            broadening: Gaussian broadening parameter
            
        Returns:
            energies: Energy grid
            dos: Density of states
        """
        eigenvalues, _ = self.solve_eigenvalue_problem(k_grid)
        
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        dos = np.zeros(num_points)
        
        for evals in eigenvalues:
            for eval in evals:
                dos += np.exp(-(energies - eval)**2 / (2 * broadening**2))
                
        dos /= (len(k_grid) * np.sqrt(2 * np.pi) * broadening)
        
        return energies, dos

class HamiltonianFactory:
    """Factory for creating different topological Hamiltonian models"""
    
    @staticmethod
    def create_model(model_type: str, lattice_params: LatticeParameters, 
                    soc_params: SpinOrbitParameters, **kwargs) -> TopologicalHamiltonian:
        """
        Create a topological Hamiltonian model
        
        Args:
            model_type: Type of model ('kane_mele', 'bhz', 'fu_kane_mele', 'weyl')
            lattice_params: Lattice parameters
            soc_params: Spin-orbit coupling parameters
            **kwargs: Model-specific parameters
            
        Returns:
            TopologicalHamiltonian instance
        """
        models = {
            'kane_mele': KaneMeleModel,
            'bhz': BHZModel, 
            'fu_kane_mele': FuKaneMeleModel,
            'weyl': WeylSemimetalModel
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return models[model_type](lattice_params, soc_params, **kwargs)
    
    @staticmethod
    def create_from_material(material_name: str) -> TopologicalHamiltonian:
        """Create model from known material parameters"""
        
        material_params = {
            'graphene': {
                'model_type': 'kane_mele',
                'lattice_params': LatticeParameters(a=2.46, b=2.46, c=10.0),
                'soc_params': SpinOrbitParameters(intrinsic_lambda=0.006),
                'kwargs': {'t': 2.8}
            },
            'hgte': {
                'model_type': 'bhz', 
                'lattice_params': LatticeParameters(a=6.46, b=6.46, c=6.46),
                'soc_params': SpinOrbitParameters(),
                'kwargs': {'A': 3.65, 'B': -68.6, 'M': -0.01}
            },
            'bi2se3': {
                'model_type': 'fu_kane_mele',
                'lattice_params': LatticeParameters(a=4.14, b=4.14, c=28.6),
                'soc_params': SpinOrbitParameters(atomic_soc={'Bi': 1.5, 'Se': 0.3}),
                'kwargs': {'v_f': 5.0, 'lambda_so': 0.3, 'm': 0.28}
            }
        }
        
        if material_name.lower() not in material_params:
            raise ValueError(f"Unknown material: {material_name}")
            
        params = material_params[material_name.lower()]
        return HamiltonianFactory.create_model(
            params['model_type'], 
            params['lattice_params'],
            params['soc_params'],
            **params['kwargs']
        )

# Utility functions for k-point generation
def generate_k_path(high_symmetry_points: Dict[str, np.ndarray], 
                   path: List[str], num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate k-point path through high-symmetry points
    
    Args:
        high_symmetry_points: Dictionary of high-symmetry points
        path: List of point names defining the path
        num_points: Total number of k-points
        
    Returns:
        k_points: Array of k-points
        distances: Cumulative distances along path
    """
    k_points = []
    distances = [0]
    
    points_per_segment = num_points // (len(path) - 1)
    
    for i in range(len(path) - 1):
        start = high_symmetry_points[path[i]]
        end = high_symmetry_points[path[i + 1]]
        
        segment = np.linspace(start, end, points_per_segment, endpoint=False)
        k_points.extend(segment)
        
        # Calculate distances
        segment_length = np.linalg.norm(end - start)
        segment_distances = np.linspace(0, segment_length, points_per_segment, endpoint=False)
        distances.extend(distances[-1] + segment_distances[1:])
    
    # Add final point
    k_points.append(high_symmetry_points[path[-1]])
    distances.append(distances[-1] + np.linalg.norm(high_symmetry_points[path[-1]] - high_symmetry_points[path[-2]]))
    
    return np.array(k_points), np.array(distances)

def generate_k_grid(reciprocal_vectors: np.ndarray, grid_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Generate uniform k-point grid in Brillouin zone
    
    Args:
        reciprocal_vectors: Reciprocal lattice vectors
        grid_size: Grid dimensions (nx, ny, nz)
        
    Returns:
        k_grid: Array of k-points
    """
    nx, ny, nz = grid_size
    
    # Generate fractional coordinates
    kx = np.linspace(0, 1, nx, endpoint=False)
    ky = np.linspace(0, 1, ny, endpoint=False) 
    kz = np.linspace(0, 1, nz, endpoint=False)
    
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Convert to Cartesian coordinates
    k_frac = np.stack([kx_grid.ravel(), ky_grid.ravel(), kz_grid.ravel()], axis=1)
    k_cart = k_frac @ reciprocal_vectors
    
    return k_cart

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create graphene with Kane-Mele SOC
    graphene = HamiltonianFactory.create_from_material('graphene')
    
    # Define high-symmetry points for graphene
    high_sym_points = {
        'Γ': np.array([0, 0, 0]),
        'K': np.array([4*np.pi/(3*np.sqrt(3)), 0, 0]),
        'M': np.array([np.pi/np.sqrt(3), np.pi/3, 0])
    }
    
    # Generate k-path
    k_path, distances = generate_k_path(high_sym_points, ['Γ', 'K', 'M', 'Γ'], 200)
    
    # Compute band structure
    eigenvalues, eigenvectors = graphene.get_band_structure(k_path)
    
    logger.info(f"Computed band structure with {len(eigenvalues)} k-points")
    logger.info(f"Number of bands: {eigenvalues.shape[1]}")
    logger.info(f"Energy range: {eigenvalues.min():.3f} to {eigenvalues.max():.3f} eV")