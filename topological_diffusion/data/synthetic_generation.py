"""Synthetic data generation using physics-based models."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import random
from scipy.spatial.transform import Rotation

from ..physics.structures import Structure, TopologicalMaterial, PhysicsConstraints
from ..physics.hamiltonian import TopologicalHamiltonian
from ..physics.invariants import TopologicalInvariantCalculator


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    # Generation parameters
    n_materials: int = 1000
    material_types: List[str] = None  # ['TI', 'WSM', 'NI', 'Chern']
    
    # Structure generation
    lattice_types: List[str] = None  # ['cubic', 'hexagonal', 'tetragonal']
    size_range: Tuple[int, int] = (2, 20)  # Number of atoms per unit cell
    
    # Physics parameters
    soc_strength_range: Tuple[float, float] = (0.01, 0.5)  # eV
    hopping_strength_range: Tuple[float, float] = (0.5, 3.0)  # eV
    onsite_energy_range: Tuple[float, float] = (-2.0, 2.0)  # eV
    
    # Electric field parameters
    field_strength_range: Tuple[float, float] = (1e4, 1e7)  # V/m
    include_field_response: bool = True
    
    # Noise and realism
    add_structural_noise: bool = True
    noise_level: float = 0.05
    add_property_noise: bool = True
    property_noise_level: float = 0.02
    
    # Constraints
    physics_constraints: Optional[PhysicsConstraints] = None
    enforce_stability: bool = True
    stability_threshold: float = 0.1  # eV/atom


class StructureGenerator:
    """Generates synthetic crystal structures."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        
        # Common elements for topological materials
        self.topological_elements = [
            'Bi', 'Sb', 'Te', 'Se', 'S',  # Topological insulators
            'Hg', 'Cd', 'Zn',             # HgTe family
            'Sn', 'Pb', 'Ge',             # Tin-based
            'Ti', 'Zr', 'Hf',             # Transition metals
            'W', 'Mo', 'Ta', 'Nb',        # Weyl semimetals
            'C', 'Si'                      # Graphene-like
        ]
        
        # Lattice parameters for different crystal systems
        self.lattice_systems = {
            'cubic': self._generate_cubic_lattice,
            'hexagonal': self._generate_hexagonal_lattice,
            'tetragonal': self._generate_tetragonal_lattice,
            'orthorhombic': self._generate_orthorhombic_lattice,
            'rhombohedral': self._generate_rhombohedral_lattice
        }
    
    def generate_structure(self, target_type: str = None) -> Structure:
        """Generate a synthetic crystal structure."""
        # Choose lattice system
        if self.config.lattice_types:
            lattice_type = random.choice(self.config.lattice_types)
        else:
            lattice_type = random.choice(list(self.lattice_systems.keys()))
        
        # Generate lattice
        lattice = self.lattice_systems[lattice_type]()
        
        # Choose number of atoms
        min_atoms, max_atoms = self.config.size_range
        n_atoms = random.randint(min_atoms, max_atoms)
        
        # Generate atomic positions and species
        positions, species = self._generate_atoms(n_atoms, target_type)
        
        # Add structural noise if requested
        if self.config.add_structural_noise:
            positions = self._add_structural_noise(positions)
        
        # Create structure
        structure = Structure(
            lattice=lattice,
            positions=positions,
            species=species,
            pbc=(True, True, True)
        )
        
        return structure
    
    def _generate_cubic_lattice(self) -> np.ndarray:
        """Generate cubic lattice vectors."""
        a = np.random.uniform(3.0, 8.0)  # Lattice parameter in Angstroms
        return np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a]
        ])
    
    def _generate_hexagonal_lattice(self) -> np.ndarray:
        """Generate hexagonal lattice vectors."""
        a = np.random.uniform(3.0, 6.0)
        c = np.random.uniform(4.0, 12.0)
        return np.array([
            [a, 0, 0],
            [-a/2, a*np.sqrt(3)/2, 0],
            [0, 0, c]
        ])
    
    def _generate_tetragonal_lattice(self) -> np.ndarray:
        """Generate tetragonal lattice vectors."""
        a = np.random.uniform(3.0, 6.0)
        c = np.random.uniform(4.0, 10.0)
        return np.array([
            [a, 0, 0],
            [0, a, 0],
            [0, 0, c]
        ])
    
    def _generate_orthorhombic_lattice(self) -> np.ndarray:
        """Generate orthorhombic lattice vectors."""
        a = np.random.uniform(3.0, 8.0)
        b = np.random.uniform(3.0, 8.0)
        c = np.random.uniform(4.0, 12.0)
        return np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c]
        ])
    
    def _generate_rhombohedral_lattice(self) -> np.ndarray:
        """Generate rhombohedral lattice vectors."""
        a = np.random.uniform(4.0, 8.0)
        alpha = np.random.uniform(60, 120) * np.pi / 180  # Convert to radians
        
        # Rhombohedral lattice vectors
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        return np.array([
            [a, 0, 0],
            [a * cos_alpha, a * sin_alpha, 0],
            [a * cos_alpha, a * cos_alpha / sin_alpha, 
             a * np.sqrt(1 - cos_alpha**2 - (cos_alpha / sin_alpha)**2)]
        ])
    
    def _generate_atoms(self, n_atoms: int, target_type: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate atomic positions and species."""
        positions = []
        species = []
        
        # Choose elements based on target topological type
        if target_type == 'TI':  # Topological Insulator
            element_pool = ['Bi', 'Sb', 'Te', 'Se', 'S']
        elif target_type == 'WSM':  # Weyl Semimetal
            element_pool = ['W', 'Mo', 'Ta', 'Nb', 'Ti']
        elif target_type == 'Chern':  # Chern Insulator
            element_pool = ['C', 'Si', 'Ge', 'Sn']
        else:
            element_pool = self.topological_elements
        
        # Apply element constraints if specified
        if self.config.physics_constraints and self.config.physics_constraints.allowed_elements:
            element_pool = [e for e in element_pool if e in self.config.physics_constraints.allowed_elements]
        
        if self.config.physics_constraints and self.config.physics_constraints.forbidden_elements:
            element_pool = [e for e in element_pool if e not in self.config.physics_constraints.forbidden_elements]
        
        if not element_pool:
            element_pool = ['C']  # Fallback
        
        # Generate positions and species
        for i in range(n_atoms):
            # Generate random fractional coordinates
            pos = np.random.random(3)
            positions.append(pos)
            
            # Choose element
            element = random.choice(element_pool)
            species.append(element)
        
        return np.array(positions), np.array(species)
    
    def _add_structural_noise(self, positions: np.ndarray) -> np.ndarray:
        """Add realistic structural noise to atomic positions."""
        noise = np.random.normal(0, self.config.noise_level, positions.shape)
        noisy_positions = positions + noise
        
        # Ensure positions remain in unit cell
        noisy_positions = noisy_positions % 1.0
        
        return noisy_positions


class PhysicsPropertyGenerator:
    """Generates synthetic physics properties using simplified models."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
    
    def generate_properties(self, structure: Structure, target_type: str = None) -> Dict:
        """Generate physics properties for a structure."""
        properties = {}
        
        # Generate Hamiltonian parameters
        soc_strength = np.random.uniform(*self.config.soc_strength_range)
        hopping_strength = np.random.uniform(*self.config.hopping_strength_range)
        
        # Generate on-site energies for each element
        unique_elements = np.unique(structure.species)
        onsite_energies = {}
        for element in unique_elements:
            energy = np.random.uniform(*self.config.onsite_energy_range)
            onsite_energies[element] = energy
        
        # Create simplified Hamiltonian
        try:
            hamiltonian = TopologicalHamiltonian(
                structure=structure,
                soc_strength=soc_strength,
                max_neighbors=2,
                on_site_energies=onsite_energies
            )
            
            # Calculate topological invariants
            invariant_calc = TopologicalInvariantCalculator(hamiltonian)
            
            # Simplified invariant calculations (faster than full calculation)
            properties['z2'] = self._estimate_z2_invariant(structure, target_type)
            properties['chern'] = self._estimate_chern_number(structure, target_type)
            properties['band_gap'] = self._estimate_band_gap(structure, soc_strength)
            
            # Classify topological phase
            properties['phase'] = self._classify_phase(properties)
            
        except Exception as e:
            # Fallback to simple estimates if Hamiltonian construction fails
            print(f\"Warning: Hamiltonian construction failed: {e}\")\n            properties = self._generate_fallback_properties(structure, target_type)\n        \n        # Add electric field response if requested\n        if self.config.include_field_response:\n            properties.update(self._generate_field_response(structure))\n        \n        # Add property noise if requested\n        if self.config.add_property_noise:\n            properties = self._add_property_noise(properties)\n        \n        return properties\n    \n    def _estimate_z2_invariant(self, structure: Structure, target_type: str = None) -> int:\n        \"\"\"Estimate Z2 invariant using heuristics.\"\"\"\n        if target_type == 'TI':\n            # Topological insulators typically have Z2 = 1\n            return 1 if np.random.random() > 0.2 else 0\n        elif target_type == 'NI':\n            # Normal insulators have Z2 = 0\n            return 0 if np.random.random() > 0.1 else 1\n        else:\n            # Random for other types\n            return np.random.choice([0, 1])\n    \n    def _estimate_chern_number(self, structure: Structure, target_type: str = None) -> int:\n        \"\"\"Estimate Chern number using heuristics.\"\"\"\n        if target_type == 'Chern':\n            # Chern insulators have non-zero Chern number\n            return np.random.choice([-2, -1, 1, 2])\n        else:\n            # Most other materials have Chern = 0\n            return 0 if np.random.random() > 0.1 else np.random.choice([-1, 1])\n    \n    def _estimate_band_gap(self, structure: Structure, soc_strength: float) -> float:\n        \"\"\"Estimate band gap using simple heuristics.\"\"\"\n        # Base gap depends on structure and composition\n        base_gap = 0.1 + 0.5 * np.random.random()\n        \n        # SOC can open or close gaps\n        soc_contribution = soc_strength * (0.5 - np.random.random())\n        \n        # Size effects\n        size_factor = 1.0 / np.sqrt(structure.num_atoms)\n        \n        gap = base_gap + soc_contribution + size_factor * 0.1\n        \n        # Ensure gap is positive for insulators\n        return max(0.01, gap)\n    \n    def _classify_phase(self, properties: Dict) -> str:\n        \"\"\"Classify topological phase based on invariants.\"\"\"\n        z2 = properties.get('z2', 0)\n        chern = properties.get('chern', 0)\n        gap = properties.get('band_gap', 0.1)\n        \n        if gap < 0.01:\n            return 'Metal'\n        elif chern != 0:\n            return f'Chern insulator (C={chern})'\n        elif z2 == 1:\n            return 'Z2 topological insulator'\n        else:\n            return 'Trivial insulator'\n    \n    def _generate_field_response(self, structure: Structure) -> Dict:\n        \"\"\"Generate electric field response properties.\"\"\"\n        # Critical field for topological transition\n        critical_field = np.random.uniform(*self.config.field_strength_range)\n        \n        # Field-induced gap modulation\n        field_values = np.linspace(0, critical_field, 10)\n        gap_values = 0.1 * np.exp(-field_values / critical_field) + 0.01\n        \n        # Polarization (simplified)\n        polarization = np.random.normal(0, 1e-12, 3)  # C/m²\n        \n        return {\n            'critical_field': critical_field,\n            'field_induced_gap': (field_values, gap_values),\n            'polarization': polarization\n        }\n    \n    def _generate_fallback_properties(self, structure: Structure, target_type: str = None) -> Dict:\n        \"\"\"Generate fallback properties when physics calculations fail.\"\"\"\n        properties = {\n            'z2': self._estimate_z2_invariant(structure, target_type),\n            'chern': self._estimate_chern_number(structure, target_type),\n            'band_gap': np.random.uniform(0.01, 2.0)\n        }\n        \n        properties['phase'] = self._classify_phase(properties)\n        \n        return properties\n    \n    def _add_property_noise(self, properties: Dict) -> Dict:\n        \"\"\"Add realistic noise to properties.\"\"\"\n        noisy_properties = properties.copy()\n        \n        for key, value in properties.items():\n            if isinstance(value, (int, float)) and key != 'z2' and key != 'chern':\n                # Add Gaussian noise (preserve integer invariants)\n                noise = np.random.normal(0, abs(value) * self.config.property_noise_level)\n                noisy_properties[key] = value + noise\n        \n        return noisy_properties\n\n\nclass SyntheticDatasetGenerator:\n    \"\"\"Main class for generating synthetic topological materials datasets.\"\"\"\n    \n    def __init__(self, config: SyntheticDataConfig = None):\n        if config is None:\n            config = SyntheticDataConfig()\n        \n        self.config = config\n        self.structure_generator = StructureGenerator(config)\n        self.physics_generator = PhysicsPropertyGenerator(config)\n    \n    def generate_dataset(self, n_materials: int = None) -> List[TopologicalMaterial]:\n        \"\"\"Generate a complete synthetic dataset.\"\"\"\n        if n_materials is None:\n            n_materials = self.config.n_materials\n        \n        materials = []\n        \n        # Determine target types distribution\n        if self.config.material_types:\n            type_distribution = self.config.material_types\n        else:\n            type_distribution = ['TI', 'WSM', 'NI', 'Chern', 'Metal']\n        \n        print(f\"Generating {n_materials} synthetic materials...\")\n        \n        for i in range(n_materials):\n            if i % 100 == 0:\n                print(f\"Generated {i}/{n_materials} materials\")\n            \n            try:\n                # Choose target type\n                target_type = random.choice(type_distribution)\n                \n                # Generate structure\n                structure = self.structure_generator.generate_structure(target_type)\n                \n                # Generate properties\n                properties = self.physics_generator.generate_properties(structure, target_type)\n                \n                # Create material\n                material = TopologicalMaterial(\n                    structure=structure,\n                    topological_invariants=properties,\n                    material_id=f\"synthetic_{i:06d}\",\n                    confidence_score=0.8 + 0.2 * np.random.random()  # Synthetic confidence\n                )\n                \n                # Apply stability filter if requested\n                if self.config.enforce_stability:\n                    if self._is_stable(material):\n                        materials.append(material)\n                else:\n                    materials.append(material)\n                    \n            except Exception as e:\n                print(f\"Warning: Failed to generate material {i}: {e}\")\n                continue\n        \n        print(f\"Successfully generated {len(materials)} materials\")\n        return materials\n    \n    def generate_balanced_dataset(self, materials_per_class: int = 200) -> List[TopologicalMaterial]:\n        \"\"\"Generate a balanced dataset with equal representation of each class.\"\"\"\n        if not self.config.material_types:\n            material_types = ['TI', 'WSM', 'NI', 'Chern']\n        else:\n            material_types = self.config.material_types\n        \n        balanced_materials = []\n        \n        for material_type in material_types:\n            print(f\"Generating {materials_per_class} {material_type} materials...\")\n            \n            type_materials = []\n            attempts = 0\n            max_attempts = materials_per_class * 3  # Allow some failures\n            \n            while len(type_materials) < materials_per_class and attempts < max_attempts:\n                try:\n                    # Generate structure\n                    structure = self.structure_generator.generate_structure(material_type)\n                    \n                    # Generate properties\n                    properties = self.physics_generator.generate_properties(structure, material_type)\n                    \n                    # Check if material matches target type\n                    if self._matches_target_type(properties, material_type):\n                        material = TopologicalMaterial(\n                            structure=structure,\n                            topological_invariants=properties,\n                            material_id=f\"synthetic_{material_type}_{len(type_materials):06d}\",\n                            confidence_score=0.8 + 0.2 * np.random.random()\n                        )\n                        \n                        if not self.config.enforce_stability or self._is_stable(material):\n                            type_materials.append(material)\n                    \n                    attempts += 1\n                    \n                except Exception as e:\n                    attempts += 1\n                    continue\n            \n            balanced_materials.extend(type_materials)\n            print(f\"Generated {len(type_materials)} {material_type} materials\")\n        \n        # Shuffle the dataset\n        random.shuffle(balanced_materials)\n        \n        return balanced_materials\n    \n    def _matches_target_type(self, properties: Dict, target_type: str) -> bool:\n        \"\"\"Check if generated properties match target type.\"\"\"\n        phase = properties.get('phase', '')\n        z2 = properties.get('z2', 0)\n        chern = properties.get('chern', 0)\n        gap = properties.get('band_gap', 0.1)\n        \n        if target_type == 'TI':\n            return z2 == 1 and gap > 0.01\n        elif target_type == 'WSM':\n            return gap < 0.05  # Semimetal\n        elif target_type == 'NI':\n            return z2 == 0 and chern == 0 and gap > 0.01\n        elif target_type == 'Chern':\n            return chern != 0 and gap > 0.01\n        else:\n            return True  # Accept any for unspecified types\n    \n    def _is_stable(self, material: TopologicalMaterial) -> bool:\n        \"\"\"Simple stability check based on heuristics.\"\"\"\n        structure = material.structure\n        \n        # Check for reasonable atomic distances\n        distances = structure.get_all_distances()\n        min_distance = np.min(distances[distances > 0])\n        \n        if min_distance < 1.0:  # Too close\n            return False\n        \n        # Check for reasonable composition\n        unique_elements = len(np.unique(structure.species))\n        if unique_elements > 5:  # Too many different elements\n            return False\n        \n        # Check band gap reasonableness\n        gap = material.topological_invariants.get('band_gap', 0.1)\n        if gap < 0 or gap > 5.0:  # Unreasonable gap\n            return False\n        \n        return True\n    \n    def save_dataset(self, materials: List[TopologicalMaterial], filename: str):\n        \"\"\"Save dataset to file.\"\"\"\n        import pickle\n        \n        with open(filename, 'wb') as f:\n            pickle.dump(materials, f)\n        \n        print(f\"Saved {len(materials)} materials to {filename}\")\n    \n    def load_dataset(self, filename: str) -> List[TopologicalMaterial]:\n        \"\"\"Load dataset from file.\"\"\"\n        import pickle\n        \n        with open(filename, 'rb') as f:\n            materials = pickle.load(f)\n        \n        print(f\"Loaded {len(materials)} materials from {filename}\")\n        return materials\n    \n    def get_dataset_statistics(self, materials: List[TopologicalMaterial]) -> Dict:\n        \"\"\"Get statistics about the generated dataset.\"\"\"\n        stats = {\n            'total_materials': len(materials),\n            'phase_distribution': {},\n            'element_distribution': {},\n            'size_distribution': [],\n            'gap_distribution': []\n        }\n        \n        # Analyze phase distribution\n        for material in materials:\n            phase = material.topological_invariants.get('phase', 'Unknown')\n            stats['phase_distribution'][phase] = stats['phase_distribution'].get(phase, 0) + 1\n        \n        # Analyze element distribution\n        for material in materials:\n            for element in material.structure.species:\n                stats['element_distribution'][element] = stats['element_distribution'].get(element, 0) + 1\n        \n        # Analyze size and gap distributions\n        for material in materials:\n            stats['size_distribution'].append(material.structure.num_atoms)\n            stats['gap_distribution'].append(material.topological_invariants.get('band_gap', 0))\n        \n        return stats"