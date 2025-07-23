"""Advanced data augmentation strategies for topological materials."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import random
from scipy.spatial.transform import Rotation
from pymatgen.core import Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..physics.structures import Structure, TopologicalMaterial


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation strategies."""
    # Symmetry operations
    use_symmetry_operations: bool = True
    max_symmetry_ops: int = 48  # Maximum number of symmetry operations to apply
    
    # Electric field perturbations
    field_augmentation: bool = True
    field_strength_range: Tuple[float, float] = (1e4, 1e7)  # V/m
    field_directions: int = 6  # Number of field directions to sample
    
    # Structural perturbations
    atomic_displacement: bool = True
    max_displacement: float = 0.1  # Angstroms
    strain_augmentation: bool = True
    max_strain: float = 0.05  # 5% strain
    
    # Composition variations
    substitution_augmentation: bool = True
    substitution_probability: float = 0.1
    allowed_substitutions: Dict[str, List[str]] = None
    
    # Temperature effects
    thermal_augmentation: bool = True
    temperature_range: Tuple[float, float] = (0, 500)  # Kelvin
    
    # Noise injection
    property_noise: bool = True
    noise_level: float = 0.01  # 1% noise


class CrystalSymmetryAugmenter:
    """Applies crystallographic symmetry operations for data augmentation."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def get_symmetry_operations(self, structure: Structure) -> List[np.ndarray]:
        """Get all symmetry operations for a crystal structure."""
        # Convert to pymatgen structure for symmetry analysis
        pmg_structure = PMGStructure(
            lattice=structure.lattice,
            species=structure.species,
            coords=structure.positions
        )
        
        # Get space group analyzer
        sga = SpacegroupAnalyzer(pmg_structure)
        
        # Get symmetry operations
        sym_ops = sga.get_symmetry_operations()
        
        # Convert to rotation matrices
        operations = []
        for op in sym_ops[:self.config.max_symmetry_ops]:
            operations.append(op.rotation_matrix)
        
        return operations
    
    def apply_symmetry_operation(self, material: TopologicalMaterial, 
                                operation: np.ndarray) -> TopologicalMaterial:
        """Apply a symmetry operation to a material."""
        # Apply rotation to atomic positions
        new_positions = np.dot(material.structure.positions, operation.T)
        
        # Apply rotation to lattice vectors
        new_lattice = np.dot(material.structure.lattice, operation.T)
        
        # Create new structure
        new_structure = Structure(
            lattice=new_lattice,
            positions=new_positions,
            species=material.structure.species.copy(),
            pbc=material.structure.pbc,
            magnetic_moments=material.structure.magnetic_moments
        )
        
        # Create new material with transformed structure
        new_material = TopologicalMaterial(
            structure=new_structure,
            topological_invariants=material.topological_invariants.copy(),
            material_id=f"{material.material_id}_sym",
            confidence_score=material.confidence_score
        )
        
        return new_material
    
    def augment_with_symmetry(self, material: TopologicalMaterial, 
                             num_augmentations: int = 5) -> List[TopologicalMaterial]:
        """Generate augmented materials using symmetry operations."""
        if not self.config.use_symmetry_operations:
            return [material]
        
        # Get symmetry operations
        sym_ops = self.get_symmetry_operations(material.structure)
        
        # Generate augmented materials
        augmented = [material]  # Include original
        
        # Randomly sample symmetry operations
        selected_ops = random.sample(sym_ops, min(num_augmentations, len(sym_ops)))
        
        for op in selected_ops:
            try:
                aug_material = self.apply_symmetry_operation(material, op)
                augmented.append(aug_material)
            except Exception as e:
                print(f"Warning: Failed to apply symmetry operation: {e}")
                continue
        
        return augmented


class ElectricFieldAugmenter:
    """Generates data with systematic electric field perturbations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def generate_field_directions(self, num_directions: int = None) -> List[np.ndarray]:
        """Generate systematic field directions."""
        if num_directions is None:
            num_directions = self.config.field_directions
        
        directions = []
        
        # Add principal directions
        directions.extend([
            np.array([1, 0, 0]),  # x
            np.array([0, 1, 0]),  # y
            np.array([0, 0, 1]),  # z
        ])
        
        # Add diagonal directions
        directions.extend([
            np.array([1, 1, 0]) / np.sqrt(2),    # xy
            np.array([1, 0, 1]) / np.sqrt(2),    # xz
            np.array([0, 1, 1]) / np.sqrt(2),    # yz
            np.array([1, 1, 1]) / np.sqrt(3),    # xyz
        ])
        
        # Add random directions if needed
        while len(directions) < num_directions:
            # Generate random unit vector
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            directions.append(vec)
        
        return directions[:num_directions]
    
    def generate_field_strengths(self, num_strengths: int = 5) -> List[float]:
        """Generate systematic field strengths."""
        min_field, max_field = self.config.field_strength_range
        
        # Use logarithmic spacing for field strengths
        strengths = np.logspace(
            np.log10(min_field), 
            np.log10(max_field), 
            num_strengths
        )
        
        return strengths.tolist()
    
    def augment_with_fields(self, material: TopologicalMaterial) -> List[TopologicalMaterial]:
        """Generate field-augmented materials."""
        if not self.config.field_augmentation:
            return [material]
        
        augmented = [material]  # Include original (zero field)
        
        # Generate field directions and strengths
        directions = self.generate_field_directions()
        strengths = self.generate_field_strengths()
        
        # Create materials with different field configurations
        for direction in directions:
            for strength in strengths:
                field_vector = direction * strength
                
                # Create field response data
                from ..physics.structures import FieldResponse
                field_response = FieldResponse(
                    critical_field=strength,
                    field_induced_gap=(np.array([0, strength]), np.array([0.1, 0.05])),
                    polarization=direction * 1e-12  # Typical polarization scale
                )
                
                # Create new material with field response
                new_material = TopologicalMaterial(
                    structure=material.structure,
                    topological_invariants=material.topological_invariants.copy(),
                    field_response=field_response,
                    material_id=f"{material.material_id}_field_{strength:.0e}",
                    confidence_score=material.confidence_score
                )
                
                augmented.append(new_material)
        
        return augmented


class StructuralPerturbationAugmenter:
    """Applies structural perturbations while preserving topology."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def apply_atomic_displacement(self, structure: Structure) -> Structure:
        """Apply small random displacements to atomic positions."""
        if not self.config.atomic_displacement:
            return structure
        
        # Generate random displacements
        displacements = np.random.normal(
            0, self.config.max_displacement, 
            structure.positions.shape
        )
        
        # Apply displacements
        new_positions = structure.positions + displacements
        
        # Ensure positions remain in unit cell
        new_positions = new_positions % 1.0
        
        return Structure(
            lattice=structure.lattice,
            positions=new_positions,
            species=structure.species.copy(),
            pbc=structure.pbc,
            magnetic_moments=structure.magnetic_moments
        )
    
    def apply_strain(self, structure: Structure) -> Structure:
        """Apply uniform strain to the lattice."""
        if not self.config.strain_augmentation:
            return structure
        
        # Generate random strain tensor (symmetric)
        strain = np.random.normal(0, self.config.max_strain, (3, 3))
        strain = (strain + strain.T) / 2  # Make symmetric
        
        # Apply strain to lattice vectors
        strain_matrix = np.eye(3) + strain
        new_lattice = np.dot(structure.lattice, strain_matrix)
        
        return Structure(
            lattice=new_lattice,
            positions=structure.positions.copy(),
            species=structure.species.copy(),
            pbc=structure.pbc,
            magnetic_moments=structure.magnetic_moments
        )
    
    def augment_with_perturbations(self, material: TopologicalMaterial, 
                                  num_augmentations: int = 3) -> List[TopologicalMaterial]:
        """Generate structurally perturbed materials."""
        augmented = [material]  # Include original
        
        for i in range(num_augmentations):
            # Apply random combination of perturbations
            new_structure = material.structure
            
            if random.random() < 0.5:  # 50% chance of displacement
                new_structure = self.apply_atomic_displacement(new_structure)
            
            if random.random() < 0.3:  # 30% chance of strain
                new_structure = self.apply_strain(new_structure)
            
            # Create new material
            new_material = TopologicalMaterial(
                structure=new_structure,
                topological_invariants=material.topological_invariants.copy(),
                material_id=f"{material.material_id}_pert_{i}",
                confidence_score=material.confidence_score * 0.95  # Slightly lower confidence
            )
            
            augmented.append(new_material)
        
        return augmented


class CompositionAugmenter:
    """Handles chemical substitutions and composition variations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        
        # Default substitution rules for common topological materials
        if config.allowed_substitutions is None:
            self.substitutions = {
                'Bi': ['Sb', 'As'],
                'Se': ['Te', 'S'],
                'Te': ['Se', 'S'],
                'Hg': ['Cd', 'Zn'],
                'Sn': ['Pb', 'Ge'],
                'Ti': ['Zr', 'Hf'],
                'W': ['Mo', 'Cr']
            }
        else:
            self.substitutions = config.allowed_substitutions
    
    def apply_substitution(self, material: TopologicalMaterial) -> TopologicalMaterial:
        """Apply chemical substitution to a material."""
        if not self.config.substitution_augmentation:
            return material
        
        new_species = material.structure.species.copy()
        substituted = False
        
        # Try to substitute each atom with some probability
        for i, species in enumerate(new_species):
            if random.random() < self.config.substitution_probability:
                if species in self.substitutions:
                    # Choose random substitution
                    new_element = random.choice(self.substitutions[species])
                    new_species[i] = new_element
                    substituted = True
        
        if not substituted:
            return material
        
        # Create new structure with substituted species
        new_structure = Structure(
            lattice=material.structure.lattice,
            positions=material.structure.positions.copy(),
            species=new_species,
            pbc=material.structure.pbc,
            magnetic_moments=material.structure.magnetic_moments
        )
        
        # Create new material
        new_material = TopologicalMaterial(
            structure=new_structure,
            topological_invariants={},  # Will need to be recalculated
            material_id=f"{material.material_id}_sub",
            confidence_score=material.confidence_score * 0.8  # Lower confidence for substituted
        )
        
        return new_material


class PropertyNoiseAugmenter:
    """Adds realistic noise to material properties."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def add_property_noise(self, material: TopologicalMaterial) -> TopologicalMaterial:
        """Add noise to material properties."""
        if not self.config.property_noise:
            return material
        
        # Add noise to topological invariants
        new_invariants = {}
        for key, value in material.topological_invariants.items():
            if isinstance(value, (int, float)):
                # Add Gaussian noise
                noise = np.random.normal(0, abs(value) * self.config.noise_level)
                new_invariants[key] = value + noise
            else:
                new_invariants[key] = value
        
        # Reduce confidence score to reflect noise
        new_confidence = material.confidence_score * (1 - self.config.noise_level)
        
        # Create new material with noisy properties
        new_material = TopologicalMaterial(
            structure=material.structure,
            topological_invariants=new_invariants,
            field_response=material.field_response,
            transport_properties=material.transport_properties,
            material_id=f"{material.material_id}_noise",
            confidence_score=new_confidence
        )
        
        return new_material


class ComprehensiveAugmenter:
    """Combines all augmentation strategies."""
    
    def __init__(self, config: AugmentationConfig = None):
        if config is None:
            config = AugmentationConfig()
        
        self.config = config
        self.symmetry_augmenter = CrystalSymmetryAugmenter(config)
        self.field_augmenter = ElectricFieldAugmenter(config)
        self.structural_augmenter = StructuralPerturbationAugmenter(config)
        self.composition_augmenter = CompositionAugmenter(config)
        self.noise_augmenter = PropertyNoiseAugmenter(config)
    
    def augment_material(self, material: TopologicalMaterial, 
                        augmentation_factor: int = 10) -> List[TopologicalMaterial]:
        """Apply comprehensive augmentation to a material."""
        augmented_materials = [material]
        
        # Apply different augmentation strategies
        strategies = []
        
        if self.config.use_symmetry_operations:
            strategies.append(('symmetry', self.symmetry_augmenter.augment_with_symmetry))
        
        if self.config.field_augmentation:
            strategies.append(('field', self.field_augmenter.augment_with_fields))
        
        if self.config.atomic_displacement or self.config.strain_augmentation:
            strategies.append(('structural', self.structural_augmenter.augment_with_perturbations))
        
        # Apply strategies in combination
        current_materials = [material]
        
        for strategy_name, strategy_func in strategies:
            new_materials = []
            
            for mat in current_materials:
                try:
                    if strategy_name == 'field':
                        # Field augmentation generates many variants
                        augmented = strategy_func(mat)
                        # Limit to avoid explosion
                        new_materials.extend(augmented[:5])
                    else:
                        augmented = strategy_func(mat, num_augmentations=2)
                        new_materials.extend(augmented)
                except Exception as e:
                    print(f"Warning: {strategy_name} augmentation failed: {e}")
                    new_materials.append(mat)
            
            current_materials = new_materials
            
            # Limit total number to prevent explosion
            if len(current_materials) > augmentation_factor:
                current_materials = random.sample(current_materials, augmentation_factor)
        
        # Apply composition and noise augmentation to a subset
        final_materials = current_materials.copy()
        
        for mat in current_materials[:augmentation_factor//2]:
            # Try composition substitution
            if random.random() < 0.3:  # 30% chance
                try:
                    sub_mat = self.composition_augmenter.apply_substitution(mat)
                    if sub_mat != mat:  # Only add if actually substituted
                        final_materials.append(sub_mat)
                except Exception as e:
                    print(f"Warning: Composition augmentation failed: {e}")
            
            # Add property noise
            if random.random() < 0.5:  # 50% chance
                try:
                    noise_mat = self.noise_augmenter.add_property_noise(mat)
                    final_materials.append(noise_mat)
                except Exception as e:
                    print(f"Warning: Noise augmentation failed: {e}")
        
        # Final limiting
        if len(final_materials) > augmentation_factor:
            final_materials = random.sample(final_materials, augmentation_factor)
        
        return final_materials
    
    def augment_dataset(self, materials: List[TopologicalMaterial], 
                       augmentation_factor: int = 10) -> List[TopologicalMaterial]:
        """Augment an entire dataset of materials."""
        augmented_dataset = []
        
        for i, material in enumerate(materials):
            print(f"Augmenting material {i+1}/{len(materials)}: {material.material_id}")
            
            try:
                augmented = self.augment_material(material, augmentation_factor)
                augmented_dataset.extend(augmented)
            except Exception as e:
                print(f"Warning: Failed to augment material {material.material_id}: {e}")
                augmented_dataset.append(material)  # Keep original
        
        print(f"Dataset augmented from {len(materials)} to {len(augmented_dataset)} materials")
        return augmented_dataset