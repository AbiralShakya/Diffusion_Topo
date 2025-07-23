"""
Physics-Based Field Response Data Generator
==========================================

This module addresses the data scarcity problem for electric field effects
by generating synthetic field-response datasets using physics-based models.

Key Features:
- Parameter estimation from material properties
- Theoretical Stark effect calculations
- Synthetic field-response dataset generation
- Uncertainty quantification for synthetic data
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from scipy.optimize import minimize_scalar
from scipy.constants import elementary_charge, epsilon_0, hbar

from .quantum_hamiltonian import TopologicalHamiltonian, HamiltonianFactory
from .electric_field import StarkEffectCalculator, ElectricFieldConfig, MaterialProperties
from .topological_invariants import TopologicalInvariantCalculator

logger = logging.getLogger(__name__)

@dataclass
class FieldResponseData:
    """Container for field response data"""
    structure: Dict
    field_vector: np.ndarray
    band_gap_shift: float
    energy_shifts: np.ndarray
    topological_transition: bool
    critical_field: Optional[float]
    confidence: float  # Confidence in synthetic data
    source: str = "synthetic"

class StarkParameterEstimator:
    """Estimate Stark effect parameters from basic material properties"""
    
    def __init__(self):
        # Empirical scaling relationships from literature
        self.ionic_radii = {
            'H': 0.31, 'Li': 0.76, 'Be': 0.27, 'B': 0.11, 'C': 0.16, 'N': 0.146,
            'O': 0.140, 'F': 0.133, 'Na': 1.02, 'Mg': 0.72, 'Al': 0.535, 'Si': 0.40,
            'P': 0.38, 'S': 0.37, 'Cl': 0.181, 'K': 1.38, 'Ca': 1.00, 'Sc': 0.745,
            'Ti': 0.605, 'V': 0.58, 'Cr': 0.615, 'Mn': 0.67, 'Fe': 0.645, 'Co': 0.65,
            'Ni': 0.69, 'Cu': 0.73, 'Zn': 0.74, 'Ga': 0.62, 'Ge': 0.53, 'As': 0.58,
            'Se': 0.50, 'Br': 0.196, 'Rb': 1.52, 'Sr': 1.18, 'Y': 0.90, 'Zr': 0.72,
            'Nb': 0.64, 'Mo': 0.69, 'Tc': 0.645, 'Ru': 0.68, 'Rh': 0.665, 'Pd': 0.86,
            'Ag': 1.15, 'Cd': 0.95, 'In': 0.80, 'Sn': 0.69, 'Sb': 0.76, 'Te': 0.70,
            'I': 0.220, 'Cs': 1.67, 'Ba': 1.35, 'La': 1.032, 'Ce': 1.01, 'Pr': 0.99,
            'Nd': 0.983, 'Pm': 0.97, 'Sm': 0.958, 'Eu': 0.947, 'Gd': 0.938, 'Tb': 0.923,
            'Dy': 0.912, 'Ho': 0.901, 'Er': 0.890, 'Tm': 0.880, 'Yb': 0.868, 'Lu': 0.861,
            'Hf': 0.71, 'Ta': 0.64, 'W': 0.66, 'Re': 0.63, 'Os': 0.63, 'Ir': 0.625,
            'Pt': 0.625, 'Au': 1.37, 'Hg': 1.02, 'Tl': 1.50, 'Pb': 1.19, 'Bi': 1.03,
            'Po': 0.97, 'At': 0.62, 'Rn': 1.20
        }
        
        # Born effective charge estimates (typical values)
        self.typical_born_charges = {
            'ionic': {'cation': 2.0, 'anion': -2.0},
            'covalent': {'all': 0.5},
            'metallic': {'all': 0.1}
        }
    
    def estimate_polarizability(self, material_props: Dict) -> np.ndarray:
        """
        Estimate polarizability tensor from material properties
        
        Uses multiple approaches:
        1. Ionic polarizability from ionic radii
        2. Electronic polarizability from band gap
        3. Dielectric constant correlation
        """
        elements = material_props.get('elements', [])
        band_gap = material_props.get('band_gap', 1.0)
        dielectric = material_props.get('dielectric_constant', 10.0)
        
        # Ionic contribution (Clausius-Mossotti relation)
        alpha_ionic = self._estimate_ionic_polarizability(elements)
        
        # Electronic contribution (Penn model)
        alpha_electronic = self._estimate_electronic_polarizability(band_gap, dielectric)
        
        # Total polarizability
        alpha_total = alpha_ionic + alpha_electronic
        
        # Assume isotropic for simplicity (can be enhanced)
        return np.diag([alpha_total, alpha_total, alpha_total])
    
    def _estimate_ionic_polarizability(self, elements: List[str]) -> float:
        """Estimate ionic polarizability from ionic radii"""
        if not elements:
            return 1e-40  # Default value
            
        alpha_ionic = 0.0
        for element in elements:
            if element in self.ionic_radii:
                radius = self.ionic_radii[element] * 1e-10  # Convert to meters
                # Classical ionic polarizability: α = 4πε₀r³
                alpha_ionic += 4 * np.pi * epsilon_0 * radius**3
            else:
                alpha_ionic += 1e-40  # Default for unknown elements
                
        return alpha_ionic / len(elements)  # Average
    
    def _estimate_electronic_polarizability(self, band_gap: float, dielectric: float) -> float:
        """Estimate electronic polarizability using Penn model"""
        # Penn model: α_e ∝ (ε_∞ - 1) / (4π * n * E_g²)
        # Simplified version
        eps_inf = dielectric  # High-frequency dielectric constant
        alpha_electronic = (eps_inf - 1) * epsilon_0 * 1e-30 / (band_gap**2 + 0.1)
        
        return alpha_electronic
    
    def estimate_dipole_moments(self, structure: Dict) -> List[np.ndarray]:
        """
        Estimate atomic dipole moments from structure and bonding
        
        Uses Born effective charges and atomic displacements
        """
        coordinates = np.array(structure.get('coordinates', []))
        elements = structure.get('elements', [])
        
        if len(coordinates) == 0:
            return [np.zeros(3) for _ in elements]
        
        # Estimate Born effective charges
        born_charges = self._estimate_born_charges(elements, structure)
        
        # Compute dipole moments
        dipole_moments = []
        centroid = np.mean(coordinates, axis=0)
        
        for i, (coord, element) in enumerate(zip(coordinates, elements)):
            # Displacement from centroid
            displacement = coord - centroid
            
            # Dipole moment = charge × displacement
            dipole = born_charges[i] * displacement * elementary_charge
            dipole_moments.append(dipole)
            
        return dipole_moments
    
    def _estimate_born_charges(self, elements: List[str], structure: Dict) -> List[float]:
        """Estimate Born effective charges"""
        bonding_type = self._classify_bonding(elements, structure)
        
        born_charges = []
        for element in elements:
            if bonding_type == 'ionic':
                # Simple ionic model
                if self._is_cation(element):
                    charge = self.typical_born_charges['ionic']['cation']
                else:
                    charge = self.typical_born_charges['ionic']['anion']
            elif bonding_type == 'covalent':
                charge = self.typical_born_charges['covalent']['all']
            else:  # metallic
                charge = self.typical_born_charges['metallic']['all']
                
            born_charges.append(charge)
            
        return born_charges
    
    def _classify_bonding(self, elements: List[str], structure: Dict) -> str:
        """Classify bonding type (ionic/covalent/metallic)"""
        # Simplified classification based on elements
        metals = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 
                 'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi'}
        nonmetals = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I'}
        
        element_set = set(elements)
        
        if element_set.intersection(metals) and element_set.intersection(nonmetals):
            return 'ionic'
        elif element_set.issubset(nonmetals):
            return 'covalent'
        else:
            return 'metallic'
    
    def _is_cation(self, element: str) -> bool:
        """Check if element is likely to be a cation"""
        cations = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
                  'Al', 'Ga', 'In', 'Tl', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                  'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh',
                  'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                  'Au', 'Hg', 'Pb', 'Bi'}
        return element in cations

class FieldResponseDataGenerator:
    """Generate synthetic field-response datasets using physics models"""
    
    def __init__(self, materials_database: List[Dict]):
        self.materials_db = materials_database
        self.parameter_estimator = StarkParameterEstimator()
        
    def generate_field_response_dataset(self, n_samples: int = 10000,
                                      field_strength_range: Tuple[float, float] = (1e4, 1e7),
                                      n_field_points: int = 15) -> List[FieldResponseData]:
        """
        Generate comprehensive field-response dataset
        
        Args:
            n_samples: Target number of samples
            field_strength_range: Range of field strengths (V/m)
            n_field_points: Number of field strength points
            
        Returns:
            List of FieldResponseData objects
        """
        dataset = []
        
        # Field parameters
        field_strengths = np.logspace(
            np.log10(field_strength_range[0]), 
            np.log10(field_strength_range[1]), 
            n_field_points
        )
        
        field_directions = self._generate_field_directions()
        
        samples_per_material = max(1, n_samples // len(self.materials_db))
        
        for material in self.materials_db:
            try:
                # Generate field responses for this material
                material_responses = self._generate_material_field_responses(
                    material, field_strengths, field_directions, samples_per_material
                )
                dataset.extend(material_responses)
                
                if len(dataset) >= n_samples:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to generate field response for material: {e}")
                continue
        
        logger.info(f"Generated {len(dataset)} field-response data points")
        return dataset[:n_samples]
    
    def _generate_field_directions(self) -> List[np.ndarray]:
        """Generate diverse field directions"""
        directions = [
            np.array([1, 0, 0]),  # x
            np.array([0, 1, 0]),  # y
            np.array([0, 0, 1]),  # z
            np.array([1, 1, 0]) / np.sqrt(2),  # xy diagonal
            np.array([1, 0, 1]) / np.sqrt(2),  # xz diagonal
            np.array([0, 1, 1]) / np.sqrt(2),  # yz diagonal
            np.array([1, 1, 1]) / np.sqrt(3),  # xyz diagonal
        ]
        
        # Add some random directions
        for _ in range(8):
            random_dir = np.random.randn(3)
            random_dir /= np.linalg.norm(random_dir)
            directions.append(random_dir)
            
        return directions
    
    def _generate_material_field_responses(self, material: Dict, 
                                         field_strengths: np.ndarray,
                                         field_directions: List[np.ndarray],
                                         max_samples: int) -> List[FieldResponseData]:
        """Generate field responses for a single material"""
        responses = []
        
        # Estimate Stark parameters
        polarizability = self.parameter_estimator.estimate_polarizability(material)
        dipole_moments = self.parameter_estimator.estimate_dipole_moments(material)
        
        # Create Hamiltonian for material
        hamiltonian = self._create_material_hamiltonian(material)
        if hamiltonian is None:
            return responses
        
        sample_count = 0
        for E_strength in field_strengths:
            for E_direction in field_directions:
                if sample_count >= max_samples:
                    break
                    
                try:
                    # Compute field response
                    response = self._compute_stark_response(
                        material, hamiltonian, E_strength, E_direction,
                        polarizability, dipole_moments
                    )
                    
                    responses.append(response)
                    sample_count += 1
                    
                except Exception as e:
                    logger.debug(f"Failed to compute response for E={E_strength:.1e}: {e}")
                    continue
                    
            if sample_count >= max_samples:
                break
        
        return responses
    
    def _create_material_hamiltonian(self, material: Dict) -> Optional[TopologicalHamiltonian]:
        """Create appropriate Hamiltonian for material"""
        try:
            # Try to identify material type and create appropriate model
            elements = set(material.get('elements', []))
            
            # Simple heuristics for model selection
            if elements == {'C'}:  # Graphene-like
                return HamiltonianFactory.create_from_material('graphene')
            elif 'Bi' in elements and 'Se' in elements:  # Bi2Se3-like
                return HamiltonianFactory.create_from_material('bi2se3')
            elif 'Hg' in elements and 'Te' in elements:  # HgTe-like
                return HamiltonianFactory.create_from_material('hgte')
            else:
                # Default to simple tight-binding model
                return self._create_generic_hamiltonian(material)
                
        except Exception as e:
            logger.debug(f"Failed to create Hamiltonian: {e}")
            return None
    
    def _create_generic_hamiltonian(self, material: Dict) -> TopologicalHamiltonian:
        """Create generic tight-binding Hamiltonian"""
        # This is a simplified implementation
        # In practice, would use more sophisticated parameter estimation
        from .quantum_hamiltonian import LatticeParameters, SpinOrbitParameters, BHZModel
        
        # Estimate lattice parameters
        lattice_params = LatticeParameters(a=5.0, b=5.0, c=5.0)  # Default
        
        # Estimate SOC parameters
        soc_params = SpinOrbitParameters(intrinsic_lambda=0.1)  # Default
        
        # Use BHZ model as generic 2-band model
        return BHZModel(lattice_params, soc_params)
    
    def _compute_stark_response(self, material: Dict, hamiltonian: TopologicalHamiltonian,
                              E_strength: float, E_direction: np.ndarray,
                              polarizability: np.ndarray, 
                              dipole_moments: List[np.ndarray]) -> FieldResponseData:
        """Compute theoretical Stark response"""
        
        # Create field configuration
        field_config = ElectricFieldConfig(
            field_strength=E_strength,
            field_direction=E_direction,
            temperature=300.0
        )
        
        # Create material properties
        material_props = MaterialProperties(
            dielectric_tensor=np.diag([10.0, 10.0, 10.0]),  # Default
            polarizability=polarizability,
            born_charges={},  # Simplified
            carrier_density=1e16,
            band_gap=material.get('band_gap', 0.1)
        )
        
        # Create field solver and Stark calculator
        from .electric_field import FieldSolverFactory
        field_solver = FieldSolverFactory.create_solver('uniform', field_config, material_props)
        stark_calc = StarkEffectCalculator(hamiltonian, field_solver)
        
        # Compute response at Γ point
        k_gamma = np.array([0, 0, 0])
        coordinates = np.array(material.get('coordinates', [[0, 0, 0]]))
        
        # Zero-field eigenvalues
        H0 = hamiltonian.build_hamiltonian(k_gamma)
        eigenvals_0 = np.sort(np.real(np.linalg.eigvals(H0)))
        
        # Field-perturbed eigenvalues
        H_field = stark_calc.apply_stark_effect(k_gamma, coordinates)
        eigenvals_E = np.sort(np.real(np.linalg.eigvals(H_field)))
        
        # Compute response properties
        band_gap_shift = self._compute_gap_shift(eigenvals_0, eigenvals_E)
        energy_shifts = eigenvals_E - eigenvals_0
        topological_transition = self._check_topological_transition(eigenvals_0, eigenvals_E)
        critical_field = self._estimate_critical_field(material, E_direction)
        
        # Estimate confidence based on material properties
        confidence = self._estimate_confidence(material, E_strength)
        
        return FieldResponseData(
            structure=material,
            field_vector=E_strength * E_direction,
            band_gap_shift=band_gap_shift,
            energy_shifts=energy_shifts,
            topological_transition=topological_transition,
            critical_field=critical_field,
            confidence=confidence,
            source="physics_based_synthetic"
        )
    
    def _compute_gap_shift(self, eigenvals_0: np.ndarray, eigenvals_E: np.ndarray) -> float:
        """Compute band gap shift"""
        n_bands = len(eigenvals_0)
        
        # Assume half-filled system
        gap_0 = eigenvals_0[n_bands//2] - eigenvals_0[n_bands//2 - 1]
        gap_E = eigenvals_E[n_bands//2] - eigenvals_E[n_bands//2 - 1]
        
        return gap_E - gap_0
    
    def _check_topological_transition(self, eigenvals_0: np.ndarray, 
                                    eigenvals_E: np.ndarray) -> bool:
        """Check for topological phase transition (simplified)"""
        # Simplified criterion: band inversion
        n_bands = len(eigenvals_0)
        
        # Check if valence/conduction band ordering changes
        valence_0 = eigenvals_0[n_bands//2 - 1]
        conduction_0 = eigenvals_0[n_bands//2]
        
        valence_E = eigenvals_E[n_bands//2 - 1]
        conduction_E = eigenvals_E[n_bands//2]
        
        # Transition if band ordering changes
        return (valence_0 < conduction_0) != (valence_E < conduction_E)
    
    def _estimate_critical_field(self, material: Dict, field_direction: np.ndarray) -> Optional[float]:
        """Estimate critical field for topological transition"""
        band_gap = material.get('band_gap', 0.1)
        
        # Simple estimate: E_c ~ E_g / (e * a)
        # where a is lattice constant
        lattice_constant = 5e-10  # Default 5 Å
        critical_field = band_gap * elementary_charge / (elementary_charge * lattice_constant)
        
        return critical_field
    
    def _estimate_confidence(self, material: Dict, field_strength: float) -> float:
        """Estimate confidence in synthetic data"""
        confidence = 1.0
        
        # Lower confidence for very high fields
        if field_strength > 1e7:
            confidence *= 0.5
        
        # Lower confidence for materials with unknown properties
        if 'band_gap' not in material:
            confidence *= 0.7
        
        if 'dielectric_constant' not in material:
            confidence *= 0.8
        
        # Higher confidence for well-known material types
        elements = set(material.get('elements', []))
        if elements in [{'C'}, {'Bi', 'Se'}, {'Hg', 'Te'}]:
            confidence *= 1.2
        
        return min(confidence, 1.0)

# Utility functions for integration with existing pipeline
def augment_dataset_with_field_responses(materials_dataset: List[Dict],
                                       n_field_samples: int = 5000) -> List[Dict]:
    """
    Augment existing materials dataset with field-response data
    
    Args:
        materials_dataset: Original materials database
        n_field_samples: Number of field-response samples to generate
        
    Returns:
        Augmented dataset with field-response information
    """
    generator = FieldResponseDataGenerator(materials_dataset)
    field_responses = generator.generate_field_response_dataset(n_field_samples)
    
    # Convert to format compatible with existing pipeline
    augmented_dataset = list(materials_dataset)  # Copy original data
    
    for response in field_responses:
        augmented_sample = {
            **response.structure,  # Original structure data
            'field_vector': response.field_vector,
            'band_gap_shift': response.band_gap_shift,
            'energy_shifts': response.energy_shifts,
            'topological_transition': response.topological_transition,
            'critical_field': response.critical_field,
            'data_confidence': response.confidence,
            'data_source': response.source,
            'is_field_augmented': True
        }
        augmented_dataset.append(augmented_sample)
    
    logger.info(f"Augmented dataset: {len(materials_dataset)} → {len(augmented_dataset)} samples")
    return augmented_dataset

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock materials database
    mock_materials = [
        {
            'elements': ['C'],
            'coordinates': [[0, 0, 0], [1.42e-10, 0, 0]],
            'band_gap': 0.0,
            'dielectric_constant': 2.4
        },
        {
            'elements': ['Bi', 'Se'],
            'coordinates': [[0, 0, 0], [2e-10, 0, 0], [0, 2e-10, 0]],
            'band_gap': 0.3,
            'dielectric_constant': 100
        }
    ]
    
    # Generate field-response dataset
    generator = FieldResponseDataGenerator(mock_materials)
    field_dataset = generator.generate_field_response_dataset(100)
    
    logger.info(f"Generated {len(field_dataset)} field-response samples")
    
    # Show example
    if field_dataset:
        example = field_dataset[0]
        logger.info(f"Example: E = {np.linalg.norm(example.field_vector):.1e} V/m")
        logger.info(f"Band gap shift: {example.band_gap_shift:.3f} eV")
        logger.info(f"Topological transition: {example.topological_transition}")
        logger.info(f"Confidence: {example.confidence:.2f}")