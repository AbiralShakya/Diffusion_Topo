"""Comprehensive data pipeline integrating all augmentation and active learning strategies."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import random
import os
import pickle
from pathlib import Path

from ..physics.structures import TopologicalMaterial, PhysicsConstraints
from .augmentation import ComprehensiveAugmenter, AugmentationConfig
from .active_learning import ActiveLearningStrategy, ActiveLearningConfig
from .intelligent_sampling import IntelligentSampler, SamplingConfig
from .synthetic_generation import SyntheticDatasetGenerator, SyntheticDataConfig


@dataclass
class DataPipelineConfig:
    """Configuration for the complete data pipeline."""
    # Data sources
    use_existing_databases: bool = True
    database_paths: List[str] = None
    
    # Synthetic data generation
    generate_synthetic_data: bool = True
    synthetic_data_config: Optional[SyntheticDataConfig] = None
    
    # Data augmentation
    apply_augmentation: bool = True
    augmentation_config: Optional[AugmentationConfig] = None
    augmentation_factor: int = 5
    
    # Active learning
    use_active_learning: bool = True
    active_learning_config: Optional[ActiveLearningConfig] = None
    
    # Intelligent sampling
    use_intelligent_sampling: bool = True
    sampling_config: Optional[SamplingConfig] = None
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Quality control
    apply_quality_filters: bool = True
    min_confidence_score: float = 0.5
    max_duplicate_similarity: float = 0.95
    
    # Caching and storage
    cache_processed_data: bool = True
    cache_directory: str = "./data_cache"
    
    # Physics constraints
    physics_constraints: Optional[PhysicsConstraints] = None


class DataQualityController:
    """Handles data quality control and filtering."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
    
    def filter_by_confidence(self, materials: List[TopologicalMaterial]) -> List[TopologicalMaterial]:
        """Filter materials by confidence score."""
        if not self.config.apply_quality_filters:
            return materials
        
        filtered = [
            mat for mat in materials 
            if mat.confidence_score >= self.config.min_confidence_score
        ]
        
        print(f"Confidence filtering: {len(materials)} -> {len(filtered)} materials")
        return filtered
    
    def remove_duplicates(self, materials: List[TopologicalMaterial]) -> List[TopologicalMaterial]:
        """Remove duplicate or very similar materials."""
        if not self.config.apply_quality_filters:
            return materials
        
        unique_materials = []
        
        for material in materials:
            is_duplicate = False
            
            for existing in unique_materials:
                similarity = self._calculate_similarity(material, existing)
                if similarity > self.config.max_duplicate_similarity:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_materials.append(material)
        
        print(f"Duplicate removal: {len(materials)} -> {len(unique_materials)} materials")
        return unique_materials
    
    def _calculate_similarity(self, mat1: TopologicalMaterial, mat2: TopologicalMaterial) -> float:
        """Calculate similarity between two materials."""
        # Simple similarity based on structure and properties
        similarity_score = 0.0
        
        # Structure similarity
        if mat1.structure.num_atoms == mat2.structure.num_atoms:
            similarity_score += 0.3
        
        # Species similarity
        species1 = set(mat1.structure.species)
        species2 = set(mat2.structure.species)
        species_overlap = len(species1.intersection(species2)) / len(species1.union(species2))
        similarity_score += 0.3 * species_overlap
        
        # Property similarity
        props1 = mat1.topological_invariants
        props2 = mat2.topological_invariants
        
        common_keys = set(props1.keys()).intersection(set(props2.keys()))
        if common_keys:
            prop_similarity = 0.0
            for key in common_keys:
                val1, val2 = props1[key], props2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 == val2:
                        prop_similarity += 1.0
                    elif abs(val1 - val2) < 0.1:
                        prop_similarity += 0.5
                elif val1 == val2:
                    prop_similarity += 1.0
            
            similarity_score += 0.4 * (prop_similarity / len(common_keys))
        
        return similarity_score
    
    def validate_physics(self, materials: List[TopologicalMaterial]) -> List[TopologicalMaterial]:
        """Validate materials against physics constraints."""
        if not self.config.physics_constraints:
            return materials
        
        valid_materials = []
        
        for material in materials:
            if self._satisfies_physics_constraints(material):
                valid_materials.append(material)
        
        print(f"Physics validation: {len(materials)} -> {len(valid_materials)} materials")
        return valid_materials
    
    def _satisfies_physics_constraints(self, material: TopologicalMaterial) -> bool:
        """Check if material satisfies physics constraints."""
        constraints = self.config.physics_constraints
        
        # Check element constraints
        elements = set(material.structure.species)
        
        if constraints.allowed_elements:
            if not elements.issubset(set(constraints.allowed_elements)):
                return False
        
        if constraints.forbidden_elements:
            if elements.intersection(set(constraints.forbidden_elements)):
                return False
        
        # Check size constraints
        if material.structure.num_atoms > constraints.max_atoms_per_cell:
            return False
        
        # Check band gap constraints
        gap = material.topological_invariants.get('band_gap', 0.1)
        min_gap, max_gap = constraints.band_gap_range
        if not (min_gap <= gap <= max_gap):
            return False
        
        # Check topological class
        if constraints.topological_class:
            phase = material.topological_invariants.get('phase', '')
            if constraints.topological_class not in phase:
                return False
        
        return True


class DataPipeline:
    """Main data pipeline orchestrating all data processing steps."""
    
    def __init__(self, config: DataPipelineConfig = None):
        if config is None:
            config = DataPipelineConfig()
        
        self.config = config
        
        # Initialize components
        self.quality_controller = DataQualityController(config)
        
        # Initialize augmentation
        if config.apply_augmentation:
            aug_config = config.augmentation_config or AugmentationConfig()
            self.augmenter = ComprehensiveAugmenter(aug_config)
        else:
            self.augmenter = None
        
        # Initialize active learning
        if config.use_active_learning:
            al_config = config.active_learning_config or ActiveLearningConfig()
            self.active_learner = ActiveLearningStrategy(al_config)
        else:
            self.active_learner = None
        
        # Initialize intelligent sampling
        if config.use_intelligent_sampling:
            sampling_config = config.sampling_config or SamplingConfig()
            self.intelligent_sampler = IntelligentSampler(sampling_config)
        else:
            self.intelligent_sampler = None
        
        # Initialize synthetic data generator
        if config.generate_synthetic_data:
            synthetic_config = config.synthetic_data_config or SyntheticDataConfig()
            self.synthetic_generator = SyntheticDatasetGenerator(synthetic_config)
        else:
            self.synthetic_generator = None
        
        # Create cache directory
        if config.cache_processed_data:
            Path(config.cache_directory).mkdir(parents=True, exist_ok=True)
    
    def load_existing_data(self) -> List[TopologicalMaterial]:
        """Load data from existing databases."""
        if not self.config.use_existing_databases or not self.config.database_paths:
            return []
        
        all_materials = []
        
        for db_path in self.config.database_paths:
            try:
                if db_path.endswith('.pkl') or db_path.endswith('.pickle'):
                    with open(db_path, 'rb') as f:
                        materials = pickle.load(f)
                    all_materials.extend(materials)
                    print(f"Loaded {len(materials)} materials from {db_path}")
                else:
                    print(f"Warning: Unsupported database format: {db_path}")
            except Exception as e:
                print(f"Warning: Failed to load database {db_path}: {e}")
        
        return all_materials
    
    def generate_synthetic_data(self, n_materials: int = 1000) -> List[TopologicalMaterial]:
        """Generate synthetic training data."""
        if not self.config.generate_synthetic_data or not self.synthetic_generator:
            return []
        
        print(f"Generating {n_materials} synthetic materials...")
        synthetic_materials = self.synthetic_generator.generate_dataset(n_materials)
        
        return synthetic_materials
    
    def augment_data(self, materials: List[TopologicalMaterial]) -> List[TopologicalMaterial]:
        """Apply data augmentation strategies."""
        if not self.config.apply_augmentation or not self.augmenter:
            return materials
        
        print(f"Applying data augmentation (factor: {self.config.augmentation_factor})...")
        augmented_materials = self.augmenter.augment_dataset(
            materials, 
            augmentation_factor=self.config.augmentation_factor
        )
        
        return augmented_materials
    
    def apply_quality_control(self, materials: List[TopologicalMaterial]) -> List[TopologicalMaterial]:
        """Apply quality control filters."""
        print("Applying quality control...")
        
        # Filter by confidence
        materials = self.quality_controller.filter_by_confidence(materials)
        
        # Remove duplicates
        materials = self.quality_controller.remove_duplicates(materials)
        
        # Validate physics
        materials = self.quality_controller.validate_physics(materials)
        
        return materials
    
    def split_data(self, materials: List[TopologicalMaterial]) -> Tuple[List, List, List]:
        """Split data into train/validation/test sets."""
        # Shuffle materials
        random.shuffle(materials)
        
        n_total = len(materials)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.val_split)
        
        train_data = materials[:n_train]
        val_data = materials[n_train:n_train + n_val]
        test_data = materials[n_train + n_val:]
        
        print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def active_learning_iteration(self, candidate_pool: List[TopologicalMaterial],
                                 model: torch.nn.Module,
                                 existing_data: List[TopologicalMaterial]) -> List[TopologicalMaterial]:
        """Perform one iteration of active learning."""
        if not self.config.use_active_learning or not self.active_learner:
            return random.sample(candidate_pool, min(10, len(candidate_pool)))
        
        print("Selecting candidates using active learning...")
        selected_candidates = self.active_learner.select_candidates(
            candidate_pool, model, existing_data
        )
        
        return selected_candidates
    
    def intelligent_parameter_sampling(self, parameter_space: Dict, 
                                     n_samples: int = 100) -> List[Dict]:
        """Generate intelligent parameter samples."""
        if not self.config.use_intelligent_sampling or not self.intelligent_sampler:
            # Fallback to random sampling
            return self._random_parameter_sampling(parameter_space, n_samples)
        
        print(f"Generating {n_samples} intelligent parameter samples...")
        samples = self.intelligent_sampler.sample_parameters(n_samples, parameter_space)
        
        return samples
    
    def _random_parameter_sampling(self, parameter_space: Dict, n_samples: int) -> List[Dict]:
        """Fallback random parameter sampling."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, param_range in parameter_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range
                    sample[param_name] = np.random.uniform(min_val, max_val)
                elif isinstance(param_range, list):
                    sample[param_name] = random.choice(param_range)
                else:
                    sample[param_name] = param_range
            
            samples.append(sample)
        
        return samples
    
    def cache_data(self, data: List[TopologicalMaterial], cache_key: str):
        """Cache processed data."""
        if not self.config.cache_processed_data:
            return
        
        cache_path = os.path.join(self.config.cache_directory, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Cached {len(data)} materials to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to cache data: {e}")
    
    def load_cached_data(self, cache_key: str) -> Optional[List[TopologicalMaterial]]:
        """Load cached data."""
        if not self.config.cache_processed_data:
            return None
        
        cache_path = os.path.join(self.config.cache_directory, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} materials from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"Warning: Failed to load cached data: {e}")
            return None
    
    def process_complete_pipeline(self, n_synthetic: int = 1000) -> Tuple[List, List, List]:
        """Run the complete data processing pipeline."""
        print("Starting complete data processing pipeline...")
        
        # Check for cached results
        cached_data = self.load_cached_data("complete_pipeline")
        if cached_data is not None:
            print("Using cached pipeline results")
            return self.split_data(cached_data)
        
        # Step 1: Load existing data
        existing_materials = self.load_existing_data()
        print(f"Loaded {len(existing_materials)} existing materials")
        
        # Step 2: Generate synthetic data
        synthetic_materials = self.generate_synthetic_data(n_synthetic)
        print(f"Generated {len(synthetic_materials)} synthetic materials")
        
        # Step 3: Combine all materials
        all_materials = existing_materials + synthetic_materials
        print(f"Total materials before processing: {len(all_materials)}")
        
        # Step 4: Apply quality control
        all_materials = self.apply_quality_control(all_materials)
        print(f"Materials after quality control: {len(all_materials)}")
        
        # Step 5: Apply augmentation
        all_materials = self.augment_data(all_materials)
        print(f"Materials after augmentation: {len(all_materials)}")
        
        # Step 6: Final quality control
        all_materials = self.apply_quality_control(all_materials)
        print(f"Final material count: {len(all_materials)}")
        
        # Cache the processed data
        self.cache_data(all_materials, "complete_pipeline")
        
        # Step 7: Split data
        train_data, val_data, test_data = self.split_data(all_materials)
        
        return train_data, val_data, test_data
    
    def get_pipeline_statistics(self, materials: List[TopologicalMaterial]) -> Dict:
        """Get comprehensive statistics about the processed data."""
        stats = {
            'total_materials': len(materials),
            'topological_phases': {},
            'elements': {},
            'size_distribution': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0
            },
            'confidence_distribution': {
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0
            }
        }
        
        if not materials:
            return stats
        
        # Analyze topological phases
        for material in materials:
            phase = material.topological_invariants.get('phase', 'Unknown')
            stats['topological_phases'][phase] = stats['topological_phases'].get(phase, 0) + 1
        
        # Analyze elements
        for material in materials:
            for element in material.structure.species:
                stats['elements'][element] = stats['elements'].get(element, 0) + 1
        
        # Analyze size distribution
        sizes = [mat.structure.num_atoms for mat in materials]
        stats['size_distribution'] = {
            'mean': np.mean(sizes),
            'std': np.std(sizes),
            'min': np.min(sizes),
            'max': np.max(sizes)
        }
        
        # Analyze confidence distribution
        confidences = [mat.confidence_score for mat in materials]
        stats['confidence_distribution'] = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        
        return stats