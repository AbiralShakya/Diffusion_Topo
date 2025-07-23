"""Advanced active learning strategies for efficient topological materials discovery."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

from ..physics.structures import TopologicalMaterial, PhysicsConstraints


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning strategies."""
    # Acquisition function settings
    acquisition_function: str = 'uncertainty'  # 'uncertainty', 'diversity', 'expected_improvement'
    batch_size: int = 10  # Number of samples to select per iteration
    
    # Uncertainty quantification
    uncertainty_method: str = 'ensemble'  # 'ensemble', 'dropout', 'bayesian'
    num_ensemble_models: int = 5
    dropout_samples: int = 100
    
    # Diversity settings
    diversity_metric: str = 'euclidean'  # 'euclidean', 'cosine', 'hamming'
    diversity_weight: float = 0.3  # Weight for diversity vs uncertainty
    
    # Physics-informed settings
    physics_constraints: Optional[PhysicsConstraints] = None
    constraint_weight: float = 0.5
    
    # Budget and stopping criteria
    max_iterations: int = 50
    max_dft_calculations: int = 1000
    convergence_threshold: float = 0.01
    
    # Transfer learning
    use_transfer_learning: bool = True
    source_domain_weight: float = 0.2


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def score(self, candidates: List[TopologicalMaterial], 
              model: nn.Module, **kwargs) -> np.ndarray:
        """Score candidates for acquisition."""
        pass


class UncertaintyAcquisition(AcquisitionFunction):
    """Uncertainty-based acquisition function."""
    
    def __init__(self, method: str = 'ensemble'):
        self.method = method
    
    def score(self, candidates: List[TopologicalMaterial], 
              model: nn.Module, **kwargs) -> np.ndarray:
        """Score based on prediction uncertainty."""
        if self.method == 'ensemble':
            return self._ensemble_uncertainty(candidates, model, **kwargs)
        elif self.method == 'dropout':
            return self._dropout_uncertainty(candidates, model, **kwargs)
        elif self.method == 'bayesian':
            return self._bayesian_uncertainty(candidates, model, **kwargs)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
    
    def _ensemble_uncertainty(self, candidates: List[TopologicalMaterial], 
                            model: nn.Module, **kwargs) -> np.ndarray:
        """Calculate uncertainty using ensemble of models."""
        ensemble_models = kwargs.get('ensemble_models', [model])
        
        # Get predictions from all ensemble models
        all_predictions = []
        
        for ensemble_model in ensemble_models:
            ensemble_model.eval()
            predictions = []
            
            with torch.no_grad():
                for material in candidates:
                    # Convert material to model input
                    # This would depend on the specific model architecture
                    model_input = self._material_to_tensor(material)
                    pred = ensemble_model(model_input)
                    predictions.append(pred.cpu().numpy())
            
            all_predictions.append(np.array(predictions))
        
        # Calculate variance across ensemble predictions
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_candidates, n_outputs)
        uncertainties = np.var(all_predictions, axis=0)  # Variance across models
        
        # Return mean uncertainty across all outputs
        return np.mean(uncertainties, axis=1)
    
    def _dropout_uncertainty(self, candidates: List[TopologicalMaterial], 
                           model: nn.Module, **kwargs) -> np.ndarray:
        """Calculate uncertainty using Monte Carlo dropout."""
        n_samples = kwargs.get('dropout_samples', 100)
        
        # Enable dropout during inference
        model.train()  # This enables dropout
        
        all_predictions = []
        
        for _ in range(n_samples):
            predictions = []
            
            with torch.no_grad():
                for material in candidates:
                    model_input = self._material_to_tensor(material)
                    pred = model(model_input)
                    predictions.append(pred.cpu().numpy())
            
            all_predictions.append(np.array(predictions))
        
        # Calculate variance across dropout samples
        all_predictions = np.array(all_predictions)
        uncertainties = np.var(all_predictions, axis=0)
        
        return np.mean(uncertainties, axis=1)
    
    def _bayesian_uncertainty(self, candidates: List[TopologicalMaterial], 
                            model: nn.Module, **kwargs) -> np.ndarray:
        """Calculate uncertainty using Bayesian neural networks."""
        # This would require a Bayesian model implementation
        # For now, fall back to ensemble method
        return self._ensemble_uncertainty(candidates, model, **kwargs)
    
    def _material_to_tensor(self, material: TopologicalMaterial) -> torch.Tensor:
        """Convert material to tensor for model input."""
        # This is a placeholder - actual implementation would depend on model architecture
        # Could use crystal graph representations, composition vectors, etc.
        
        # Simple example: use composition as features
        composition_features = np.zeros(100)  # Placeholder
        
        # Add structural features
        structure = material.structure
        composition_features[0] = structure.num_atoms
        composition_features[1] = structure.volume
        
        # Add any available topological invariants
        for i, (key, value) in enumerate(material.topological_invariants.items()):
            if i < 10 and isinstance(value, (int, float)):
                composition_features[10 + i] = value
        
        return torch.tensor(composition_features, dtype=torch.float32)


class DiversityAcquisition(AcquisitionFunction):
    """Diversity-based acquisition function."""
    
    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
    
    def score(self, candidates: List[TopologicalMaterial], 
              model: nn.Module, **kwargs) -> np.ndarray:
        """Score based on diversity from existing training data."""
        existing_materials = kwargs.get('existing_materials', [])
        
        if not existing_materials:
            # If no existing data, return uniform scores
            return np.ones(len(candidates))
        
        # Convert materials to feature vectors
        candidate_features = np.array([
            self._material_to_features(mat) for mat in candidates
        ])
        
        existing_features = np.array([
            self._material_to_features(mat) for mat in existing_materials
        ])
        
        # Calculate diversity scores
        diversity_scores = []
        
        for candidate_feat in candidate_features:
            # Calculate minimum distance to existing data
            distances = pairwise_distances(
                candidate_feat.reshape(1, -1), 
                existing_features, 
                metric=self.metric
            )
            
            # Use minimum distance as diversity score
            diversity_score = np.min(distances)
            diversity_scores.append(diversity_score)
        
        return np.array(diversity_scores)
    
    def _material_to_features(self, material: TopologicalMaterial) -> np.ndarray:
        """Convert material to feature vector for diversity calculation."""
        features = []
        
        # Structural features
        structure = material.structure
        features.extend([
            structure.num_atoms,
            structure.volume,
            len(np.unique(structure.species))  # Number of unique elements
        ])
        
        # Composition features (element counts)
        element_counts = {}
        for species in structure.species:
            element_counts[species] = element_counts.get(species, 0) + 1
        
        # Convert to fixed-size vector (top 20 most common elements)
        common_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                          'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca']
        
        for element in common_elements:
            features.append(element_counts.get(element, 0))
        
        # Topological features
        for key in ['z2', 'chern', 'band_gap']:
            value = material.topological_invariants.get(key, 0)
            if isinstance(value, (int, float)):
                features.append(value)
            else:
                features.append(0)
        
        return np.array(features)


class PhysicsInformedAcquisition(AcquisitionFunction):
    """Physics-informed acquisition function."""
    
    def __init__(self, constraints: PhysicsConstraints):
        self.constraints = constraints
    
    def score(self, candidates: List[TopologicalMaterial], 
              model: nn.Module, **kwargs) -> np.ndarray:
        """Score based on physics constraints and expected information gain."""
        scores = []
        
        for material in candidates:
            score = 0.0
            
            # Check topological class preference
            if self.constraints.topological_class:
                predicted_class = self._predict_topological_class(material, model)
                if predicted_class == self.constraints.topological_class:
                    score += 1.0
            
            # Check band gap range
            if hasattr(material, 'band_gap'):
                gap = material.band_gap
                min_gap, max_gap = self.constraints.band_gap_range
                if min_gap <= gap <= max_gap:
                    score += 1.0
                else:
                    # Penalize materials outside desired range
                    score -= abs(gap - (min_gap + max_gap) / 2) / max_gap
            
            # Check element constraints
            if self.constraints.allowed_elements:
                allowed_set = set(self.constraints.allowed_elements)
                material_elements = set(material.structure.species)
                if material_elements.issubset(allowed_set):
                    score += 0.5
            
            if self.constraints.forbidden_elements:
                forbidden_set = set(self.constraints.forbidden_elements)
                material_elements = set(material.structure.species)
                if not material_elements.intersection(forbidden_set):
                    score += 0.5
            
            # Check size constraints
            if material.structure.num_atoms <= self.constraints.max_atoms_per_cell:
                score += 0.5
            
            scores.append(max(0, score))  # Ensure non-negative
        
        return np.array(scores)
    
    def _predict_topological_class(self, material: TopologicalMaterial, 
                                  model: nn.Module) -> str:
        """Predict topological class using the model."""
        # This would depend on the specific model architecture
        # For now, return a placeholder
        return 'TI'  # Topological Insulator


class ExpectedImprovementAcquisition(AcquisitionFunction):
    """Expected improvement acquisition function."""
    
    def score(self, candidates: List[TopologicalMaterial], 
              model: nn.Module, **kwargs) -> np.ndarray:
        """Score based on expected improvement over current best."""
        current_best = kwargs.get('current_best_score', 0.0)
        
        # Get model predictions and uncertainties
        predictions = []
        uncertainties = []
        
        model.eval()
        with torch.no_grad():
            for material in candidates:
                model_input = self._material_to_tensor(material)
                pred = model(model_input)
                
                # Assume model outputs mean and variance
                if pred.shape[-1] >= 2:
                    mean = pred[..., 0].cpu().numpy()
                    var = pred[..., 1].cpu().numpy()
                else:
                    mean = pred.cpu().numpy()
                    var = 0.1  # Default uncertainty
                
                predictions.append(mean)
                uncertainties.append(var)
        
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        # Calculate expected improvement
        improvement = predictions - current_best
        std = np.sqrt(uncertainties)
        
        # Expected improvement formula
        from scipy.stats import norm
        z = improvement / (std + 1e-8)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
    
    def _material_to_tensor(self, material: TopologicalMaterial) -> torch.Tensor:
        """Convert material to tensor for model input."""
        # Reuse implementation from UncertaintyAcquisition
        composition_features = np.zeros(100)
        
        structure = material.structure
        composition_features[0] = structure.num_atoms
        composition_features[1] = structure.volume
        
        for i, (key, value) in enumerate(material.topological_invariants.items()):
            if i < 10 and isinstance(value, (int, float)):
                composition_features[10 + i] = value
        
        return torch.tensor(composition_features, dtype=torch.float32)


class ActiveLearningStrategy:
    """Main active learning strategy coordinator."""
    
    def __init__(self, config: ActiveLearningConfig = None):
        if config is None:
            config = ActiveLearningConfig()
        
        self.config = config
        
        # Initialize acquisition function
        if config.acquisition_function == 'uncertainty':
            self.acquisition_fn = UncertaintyAcquisition(config.uncertainty_method)
        elif config.acquisition_function == 'diversity':
            self.acquisition_fn = DiversityAcquisition(config.diversity_metric)
        elif config.acquisition_function == 'expected_improvement':
            self.acquisition_fn = ExpectedImprovementAcquisition()
        elif config.acquisition_function == 'physics_informed':
            self.acquisition_fn = PhysicsInformedAcquisition(config.physics_constraints)
        else:
            raise ValueError(f"Unknown acquisition function: {config.acquisition_function}")
    
    def select_candidates(self, candidate_pool: List[TopologicalMaterial],
                         model: nn.Module,
                         existing_materials: List[TopologicalMaterial] = None,
                         **kwargs) -> List[TopologicalMaterial]:
        """Select the most informative candidates for labeling."""
        if len(candidate_pool) <= self.config.batch_size:
            return candidate_pool
        
        # Calculate acquisition scores
        scores = self.acquisition_fn.score(
            candidate_pool, 
            model, 
            existing_materials=existing_materials,
            **kwargs
        )
        
        # Apply physics constraints if specified
        if self.config.physics_constraints:
            physics_scores = PhysicsInformedAcquisition(
                self.config.physics_constraints
            ).score(candidate_pool, model)
            
            # Combine scores
            combined_scores = (
                (1 - self.config.constraint_weight) * scores + 
                self.config.constraint_weight * physics_scores
            )
        else:
            combined_scores = scores
        
        # Select top candidates
        top_indices = np.argsort(combined_scores)[-self.config.batch_size:]
        selected_candidates = [candidate_pool[i] for i in top_indices]
        
        return selected_candidates
    
    def diversified_selection(self, candidate_pool: List[TopologicalMaterial],
                            model: nn.Module,
                            existing_materials: List[TopologicalMaterial] = None) -> List[TopologicalMaterial]:
        """Select diverse candidates using uncertainty and diversity."""
        # Get uncertainty scores
        uncertainty_scores = UncertaintyAcquisition().score(
            candidate_pool, model
        )
        
        # Get diversity scores
        diversity_scores = DiversityAcquisition().score(
            candidate_pool, model, existing_materials=existing_materials
        )
        
        # Normalize scores
        uncertainty_scores = (uncertainty_scores - np.min(uncertainty_scores)) / (
            np.max(uncertainty_scores) - np.min(uncertainty_scores) + 1e-8
        )
        
        diversity_scores = (diversity_scores - np.min(diversity_scores)) / (
            np.max(diversity_scores) - np.min(diversity_scores) + 1e-8
        )
        
        # Combine scores
        combined_scores = (
            (1 - self.config.diversity_weight) * uncertainty_scores + 
            self.config.diversity_weight * diversity_scores
        )
        
        # Select top candidates
        top_indices = np.argsort(combined_scores)[-self.config.batch_size:]
        selected_candidates = [candidate_pool[i] for i in top_indices]
        
        return selected_candidates
    
    def adaptive_sampling(self, candidate_pool: List[TopologicalMaterial],
                         model: nn.Module,
                         iteration: int,
                         performance_history: List[float]) -> List[TopologicalMaterial]:
        """Adaptive sampling that changes strategy based on performance."""
        # Analyze performance trend
        if len(performance_history) >= 3:
            recent_improvement = performance_history[-1] - performance_history[-3]
            
            if recent_improvement < self.config.convergence_threshold:
                # Performance plateauing, increase diversity
                return self.diversified_selection(candidate_pool, model)
            else:
                # Good improvement, continue with uncertainty-based selection
                return self.select_candidates(candidate_pool, model)
        else:
            # Early iterations, use standard selection
            return self.select_candidates(candidate_pool, model)
    
    def budget_aware_selection(self, candidate_pool: List[TopologicalMaterial],
                             model: nn.Module,
                             remaining_budget: int,
                             total_iterations: int) -> List[TopologicalMaterial]:
        """Select candidates considering remaining computational budget."""
        # Adjust batch size based on remaining budget
        if remaining_budget < self.config.batch_size:
            adjusted_batch_size = remaining_budget
        else:
            # Use larger batches early, smaller batches later for refinement
            progress = 1 - (remaining_budget / self.config.max_dft_calculations)
            if progress < 0.5:
                adjusted_batch_size = self.config.batch_size
            else:
                adjusted_batch_size = max(1, self.config.batch_size // 2)
        
        # Temporarily adjust config
        original_batch_size = self.config.batch_size
        self.config.batch_size = adjusted_batch_size
        
        # Select candidates
        selected = self.select_candidates(candidate_pool, model)
        
        # Restore original config
        self.config.batch_size = original_batch_size
        
        return selected


class TransferLearningStrategy:
    """Handles transfer learning from related domains."""
    
    def __init__(self, config: ActiveLearningConfig):
        self.config = config
    
    def adapt_from_source_domain(self, source_materials: List[TopologicalMaterial],
                                target_materials: List[TopologicalMaterial],
                                model: nn.Module) -> nn.Module:
        """Adapt model using source domain knowledge."""
        if not self.config.use_transfer_learning:
            return model
        
        # Simple domain adaptation: weight source domain samples
        # In practice, this would involve more sophisticated techniques
        
        # Create weighted training data
        weighted_materials = []
        
        # Add source domain materials with reduced weight
        for material in source_materials:
            material.confidence_score *= self.config.source_domain_weight
            weighted_materials.append(material)
        
        # Add target domain materials with full weight
        weighted_materials.extend(target_materials)
        
        # Retrain model (this would be implemented in the training module)
        # For now, just return the original model
        return model
    
    def identify_transferable_knowledge(self, source_materials: List[TopologicalMaterial],
                                      target_materials: List[TopologicalMaterial]) -> Dict:
        """Identify which knowledge can be transferred between domains."""
        transferable_knowledge = {
            'common_elements': set(),
            'similar_structures': [],
            'shared_topological_classes': set()
        }
        
        # Find common elements
        source_elements = set()
        target_elements = set()
        
        for material in source_materials:
            source_elements.update(material.structure.species)
        
        for material in target_materials:
            target_elements.update(material.structure.species)
        
        transferable_knowledge['common_elements'] = source_elements.intersection(target_elements)
        
        # Find shared topological classes
        source_classes = set()
        target_classes = set()
        
        for material in source_materials:
            if 'phase' in material.topological_invariants:
                source_classes.add(material.topological_invariants['phase'])
        
        for material in target_materials:
            if 'phase' in material.topological_invariants:
                target_classes.add(material.topological_invariants['phase'])
        
        transferable_knowledge['shared_topological_classes'] = source_classes.intersection(target_classes)
        
        return transferable_knowledge