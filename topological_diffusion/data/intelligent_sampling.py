"""Intelligent sampling strategies for efficient phase space exploration."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import itertools

from ..physics.structures import TopologicalMaterial, PhysicsConstraints


@dataclass
class SamplingConfig:
    """Configuration for intelligent sampling strategies."""
    # Sampling strategy
    strategy: str = 'adaptive'  # 'random', 'grid', 'adaptive', 'bayesian', 'evolutionary'
    
    # Phase space exploration
    exploration_weight: float = 0.7  # Balance between exploration and exploitation
    exploitation_weight: float = 0.3
    
    # Clustering and diversity
    use_clustering: bool = True
    n_clusters: int = 10
    cluster_method: str = 'kmeans'  # 'kmeans', 'dbscan', 'hierarchical'
    
    # Bayesian optimization
    acquisition_function: str = 'ei'  # 'ei', 'ucb', 'poi'
    gp_kernel: str = 'rbf'  # 'rbf', 'matern', 'linear'
    
    # Evolutionary sampling
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Constraints
    physics_constraints: Optional[PhysicsConstraints] = None
    constraint_penalty: float = 1000.0
    
    # Efficiency settings
    max_evaluations: int = 1000
    convergence_tolerance: float = 1e-6
    parallel_evaluations: int = 4


class SamplingStrategy(ABC):
    """Abstract base class for sampling strategies."""
    
    @abstractmethod
    def sample(self, n_samples: int, **kwargs) -> List[Dict]:
        """Generate n_samples parameter combinations."""
        pass
    
    @abstractmethod
    def update(self, new_data: List[Tuple[Dict, float]], **kwargs):
        """Update strategy with new evaluation results."""
        pass


class AdaptiveSamplingStrategy(SamplingStrategy):
    """Adaptive sampling that learns from previous evaluations."""
    
    def __init__(self, config: SamplingConfig, parameter_space: Dict):
        self.config = config
        self.parameter_space = parameter_space
        self.evaluated_points = []
        self.evaluation_results = []
        self.best_score = -np.inf
        self.best_parameters = None
        
        # Initialize surrogate model for adaptive sampling
        self.surrogate_model = None
        self._initialize_surrogate_model()
    
    def _initialize_surrogate_model(self):
        """Initialize surrogate model for parameter space."""
        # Simple Gaussian Process surrogate
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern
        
        if self.config.gp_kernel == 'rbf':
            kernel = RBF()
        elif self.config.gp_kernel == 'matern':
            kernel = Matern()
        else:
            kernel = RBF()  # Default
        
        self.surrogate_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
    
    def sample(self, n_samples: int, **kwargs) -> List[Dict]:
        """Generate adaptive samples based on current knowledge."""
        if len(self.evaluated_points) < 5:
            # Not enough data for adaptive sampling, use random sampling
            return self._random_sample(n_samples)
        
        # Use acquisition function to select promising regions
        candidates = self._generate_candidates(n_samples * 10)  # Generate more candidates
        
        # Score candidates using acquisition function
        scores = self._score_candidates(candidates)
        
        # Select top candidates
        top_indices = np.argsort(scores)[-n_samples:]
        selected_samples = [candidates[i] for i in top_indices]
        
        return selected_samples
    
    def _random_sample(self, n_samples: int) -> List[Dict]:
        """Generate random samples from parameter space."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, param_range in self.parameter_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Continuous parameter
                    min_val, max_val = param_range
                    sample[param_name] = np.random.uniform(min_val, max_val)
                elif isinstance(param_range, list):
                    # Discrete parameter
                    sample[param_name] = random.choice(param_range)
                else:
                    # Single value
                    sample[param_name] = param_range
            
            samples.append(sample)
        
        return samples
    
    def _generate_candidates(self, n_candidates: int) -> List[Dict]:
        """Generate candidate parameter combinations."""
        candidates = []
        
        # Mix of random and guided sampling
        n_random = n_candidates // 2
        n_guided = n_candidates - n_random
        
        # Random candidates
        candidates.extend(self._random_sample(n_random))
        
        # Guided candidates around best points
        if self.best_parameters:
            for _ in range(n_guided):
                candidate = self._perturb_parameters(self.best_parameters)
                candidates.append(candidate)
        else:
            candidates.extend(self._random_sample(n_guided))
        
        return candidates
    
    def _perturb_parameters(self, base_params: Dict, perturbation_scale: float = 0.1) -> Dict:
        """Create perturbed version of parameter set."""
        perturbed = {}
        
        for param_name, value in base_params.items():
            param_range = self.parameter_space[param_name]
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous parameter
                min_val, max_val = param_range
                range_size = max_val - min_val
                perturbation = np.random.normal(0, perturbation_scale * range_size)
                new_value = np.clip(value + perturbation, min_val, max_val)
                perturbed[param_name] = new_value
            elif isinstance(param_range, list):
                # Discrete parameter - occasionally change to random choice
                if np.random.random() < 0.3:  # 30% chance to change
                    perturbed[param_name] = random.choice(param_range)
                else:
                    perturbed[param_name] = value
            else:
                perturbed[param_name] = value
        
        return perturbed
    
    def _score_candidates(self, candidates: List[Dict]) -> np.ndarray:
        """Score candidates using acquisition function."""
        if len(self.evaluated_points) < 2:
            return np.random.random(len(candidates))
        
        # Convert candidates to feature vectors
        candidate_features = np.array([
            self._params_to_features(candidate) for candidate in candidates
        ])
        
        # Fit surrogate model if needed
        if len(self.evaluated_points) != len(self.surrogate_model.X_train_):
            evaluated_features = np.array([
                self._params_to_features(params) for params in self.evaluated_points
            ])
            self.surrogate_model.fit(evaluated_features, self.evaluation_results)
        
        # Get predictions and uncertainties
        predictions, std = self.surrogate_model.predict(candidate_features, return_std=True)
        
        # Calculate acquisition scores
        if self.config.acquisition_function == 'ei':
            scores = self._expected_improvement(predictions, std)
        elif self.config.acquisition_function == 'ucb':
            scores = self._upper_confidence_bound(predictions, std)
        elif self.config.acquisition_function == 'poi':
            scores = self._probability_of_improvement(predictions, std)
        else:
            scores = predictions  # Default to predicted value
        
        return scores
    
    def _params_to_features(self, params: Dict) -> np.ndarray:
        """Convert parameter dictionary to feature vector."""
        features = []
        
        for param_name in sorted(self.parameter_space.keys()):
            value = params[param_name]
            param_range = self.parameter_space[param_name]
            
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Normalize continuous parameters to [0, 1]
                min_val, max_val = param_range
                normalized = (value - min_val) / (max_val - min_val)
                features.append(normalized)
            elif isinstance(param_range, list):
                # One-hot encode discrete parameters
                one_hot = [0] * len(param_range)
                if value in param_range:
                    one_hot[param_range.index(value)] = 1
                features.extend(one_hot)
            else:
                features.append(1.0)  # Single value parameter
        
        return np.array(features)
    
    def _expected_improvement(self, predictions: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate expected improvement acquisition function."""
        improvement = predictions - self.best_score
        z = improvement / (std + 1e-8)
        
        from scipy.stats import norm
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return ei
    
    def _upper_confidence_bound(self, predictions: np.ndarray, std: np.ndarray, 
                               kappa: float = 2.0) -> np.ndarray:
        """Calculate upper confidence bound acquisition function."""
        return predictions + kappa * std
    
    def _probability_of_improvement(self, predictions: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate probability of improvement acquisition function."""
        improvement = predictions - self.best_score
        z = improvement / (std + 1e-8)
        
        from scipy.stats import norm
        return norm.cdf(z)
    
    def update(self, new_data: List[Tuple[Dict, float]], **kwargs):
        """Update strategy with new evaluation results."""
        for params, score in new_data:
            self.evaluated_points.append(params)
            self.evaluation_results.append(score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_parameters = params.copy()


class EvolutionaryStrategy(SamplingStrategy):
    """Evolutionary algorithm for parameter optimization."""
    
    def __init__(self, config: SamplingConfig, parameter_space: Dict):
        self.config = config
        self.parameter_space = parameter_space
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.config.population_size):
            individual = {}
            for param_name, param_range in self.parameter_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range
                    individual[param_name] = np.random.uniform(min_val, max_val)
                elif isinstance(param_range, list):
                    individual[param_name] = random.choice(param_range)
                else:
                    individual[param_name] = param_range
            
            self.population.append(individual)
        
        self.fitness_scores = [0.0] * len(self.population)
    
    def sample(self, n_samples: int, **kwargs) -> List[Dict]:
        """Generate samples using evolutionary algorithm."""
        if self.generation == 0:
            # Return initial population
            return self.population[:n_samples]
        
        # Evolve population
        new_population = self._evolve_population()
        
        # Return best individuals from new population
        return new_population[:n_samples]
    
    def _evolve_population(self) -> List[Dict]:
        """Evolve the population for one generation."""
        new_population = []
        
        # Elitism: keep best individuals
        elite_size = max(1, self.config.population_size // 10)
        elite_indices = np.argsort(self.fitness_scores)[-elite_size:]
        
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if np.random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        new_population = new_population[:self.config.population_size]
        
        self.generation += 1
        return new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict:
        """Select individual using tournament selection."""
        tournament_indices = np.random.choice(
            len(self.population), 
            size=min(tournament_size, len(self.population)), 
            replace=False
        )
        
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Perform crossover between two parents."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param_name in self.parameter_space.keys():
            if np.random.random() < 0.5:  # 50% chance to swap
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict) -> Dict:
        """Mutate an individual."""
        mutated = individual.copy()
        
        for param_name, param_range in self.parameter_space.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Gaussian mutation for continuous parameters
                    min_val, max_val = param_range
                    current_val = mutated[param_name]
                    mutation_strength = (max_val - min_val) * 0.1
                    new_val = current_val + np.random.normal(0, mutation_strength)
                    mutated[param_name] = np.clip(new_val, min_val, max_val)
                elif isinstance(param_range, list):
                    # Random choice for discrete parameters
                    mutated[param_name] = random.choice(param_range)
        
        return mutated
    
    def update(self, new_data: List[Tuple[Dict, float]], **kwargs):
        """Update population fitness scores."""
        # Update fitness scores for evaluated individuals
        for params, score in new_data:
            # Find matching individual in population
            for i, individual in enumerate(self.population):
                if self._params_match(individual, params):
                    self.fitness_scores[i] = score
                    break
    
    def _params_match(self, params1: Dict, params2: Dict, tolerance: float = 1e-6) -> bool:
        """Check if two parameter sets match."""
        if set(params1.keys()) != set(params2.keys()):
            return False
        
        for key in params1.keys():
            val1, val2 = params1[key], params2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if abs(val1 - val2) > tolerance:
                    return False
            elif val1 != val2:
                return False
        
        return True


class ClusterBasedSampling:
    """Cluster-based sampling for diverse exploration."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.clusters = None
        self.cluster_centers = None
    
    def cluster_parameter_space(self, evaluated_points: List[Dict]) -> Dict:
        """Cluster the evaluated parameter space."""
        if len(evaluated_points) < self.config.n_clusters:
            return {'clusters': None, 'centers': None}
        
        # Convert parameters to feature vectors
        features = np.array([
            self._params_to_features(params) for params in evaluated_points
        ])
        
        # Perform clustering
        if self.config.cluster_method == 'kmeans':
            clusterer = KMeans(n_clusters=self.config.n_clusters, random_state=42)
        elif self.config.cluster_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            clusterer = KMeans(n_clusters=self.config.n_clusters, random_state=42)
        
        cluster_labels = clusterer.fit_predict(features)
        
        # Store clustering results
        self.clusters = cluster_labels
        if hasattr(clusterer, 'cluster_centers_'):
            self.cluster_centers = clusterer.cluster_centers_
        
        return {
            'clusters': cluster_labels,
            'centers': self.cluster_centers,
            'n_clusters': len(np.unique(cluster_labels))
        }
    
    def sample_from_clusters(self, n_samples: int, parameter_space: Dict) -> List[Dict]:
        """Sample points from underexplored clusters."""
        if self.cluster_centers is None:
            # No clustering available, use random sampling
            return self._random_sample(n_samples, parameter_space)
        
        samples = []
        samples_per_cluster = max(1, n_samples // len(self.cluster_centers))
        
        for center in self.cluster_centers:
            # Generate samples around cluster center
            for _ in range(samples_per_cluster):
                sample = self._sample_around_center(center, parameter_space)
                samples.append(sample)
        
        # Fill remaining samples randomly
        while len(samples) < n_samples:
            sample = self._random_sample(1, parameter_space)[0]
            samples.append(sample)
        
        return samples[:n_samples]
    
    def _params_to_features(self, params: Dict) -> np.ndarray:
        """Convert parameters to feature vector."""
        # This should match the implementation in AdaptiveSamplingStrategy
        features = []
        
        for param_name in sorted(params.keys()):
            value = params[param_name]
            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                # Handle categorical parameters
                features.append(hash(str(value)) % 1000 / 1000.0)
        
        return np.array(features)
    
    def _sample_around_center(self, center: np.ndarray, parameter_space: Dict) -> Dict:
        """Generate sample around cluster center."""
        sample = {}
        feature_idx = 0
        
        for param_name, param_range in parameter_space.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous parameter
                min_val, max_val = param_range
                center_val = center[feature_idx] * (max_val - min_val) + min_val
                
                # Add noise around center
                noise_scale = (max_val - min_val) * 0.1
                new_val = center_val + np.random.normal(0, noise_scale)
                sample[param_name] = np.clip(new_val, min_val, max_val)
                feature_idx += 1
            elif isinstance(param_range, list):
                # Discrete parameter - choose randomly
                sample[param_name] = random.choice(param_range)
            else:
                sample[param_name] = param_range
        
        return sample
    
    def _random_sample(self, n_samples: int, parameter_space: Dict) -> List[Dict]:
        """Generate random samples."""
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


class IntelligentSampler:
    """Main intelligent sampling coordinator."""
    
    def __init__(self, config: SamplingConfig = None):
        if config is None:
            config = SamplingConfig()
        
        self.config = config
        self.strategy = None
        self.cluster_sampler = ClusterBasedSampling(config)
        self.evaluation_history = []
        
    def initialize_strategy(self, parameter_space: Dict):
        """Initialize sampling strategy."""
        if self.config.strategy == 'adaptive':
            self.strategy = AdaptiveSamplingStrategy(self.config, parameter_space)
        elif self.config.strategy == 'evolutionary':
            self.strategy = EvolutionaryStrategy(self.config, parameter_space)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def sample_parameters(self, n_samples: int, parameter_space: Dict) -> List[Dict]:
        """Generate intelligent parameter samples."""
        if self.strategy is None:
            self.initialize_strategy(parameter_space)
        
        # Get samples from main strategy
        samples = self.strategy.sample(n_samples)
        
        # Apply physics constraints if specified
        if self.config.physics_constraints:
            samples = self._filter_by_constraints(samples)
        
        return samples
    
    def update_with_results(self, parameter_results: List[Tuple[Dict, float]]):
        """Update sampler with evaluation results."""
        self.evaluation_history.extend(parameter_results)
        
        if self.strategy:
            self.strategy.update(parameter_results)
    
    def _filter_by_constraints(self, samples: List[Dict]) -> List[Dict]:
        """Filter samples based on physics constraints."""
        filtered_samples = []
        
        for sample in samples:
            if self._satisfies_constraints(sample):
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _satisfies_constraints(self, params: Dict) -> bool:
        """Check if parameters satisfy physics constraints."""
        constraints = self.config.physics_constraints
        
        # Check element constraints
        if 'elements' in params:
            elements = params['elements']
            if isinstance(elements, str):
                elements = [elements]
            
            if constraints.allowed_elements:
                if not all(elem in constraints.allowed_elements for elem in elements):
                    return False
            
            if constraints.forbidden_elements:
                if any(elem in constraints.forbidden_elements for elem in elements):
                    return False
        
        # Check size constraints
        if 'num_atoms' in params:
            if params['num_atoms'] > constraints.max_atoms_per_cell:
                return False
        
        # Check band gap constraints
        if 'band_gap' in params:
            gap = params['band_gap']
            min_gap, max_gap = constraints.band_gap_range
            if not (min_gap <= gap <= max_gap):
                return False
        
        return True
    
    def get_sampling_statistics(self) -> Dict:
        """Get statistics about sampling performance."""
        if not self.evaluation_history:
            return {}
        
        scores = [score for _, score in self.evaluation_history]
        
        stats = {
            'n_evaluations': len(self.evaluation_history),
            'best_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'improvement_rate': 0.0
        }
        
        # Calculate improvement rate
        if len(scores) >= 10:
            recent_scores = scores[-10:]
            early_scores = scores[:10]
            stats['improvement_rate'] = (np.mean(recent_scores) - np.mean(early_scores)) / np.mean(early_scores)
        
        return stats