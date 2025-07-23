"""Data processing and augmentation modules for topological materials."""

from .augmentation import (
    AugmentationConfig,
    CrystalSymmetryAugmenter,
    ElectricFieldAugmenter,
    StructuralPerturbationAugmenter,
    CompositionAugmenter,
    PropertyNoiseAugmenter,
    ComprehensiveAugmenter
)

from .active_learning import (
    ActiveLearningConfig,
    AcquisitionFunction,
    UncertaintyAcquisition,
    DiversityAcquisition,
    PhysicsInformedAcquisition,
    ExpectedImprovementAcquisition,
    ActiveLearningStrategy,
    TransferLearningStrategy
)

from .intelligent_sampling import (
    SamplingConfig,
    SamplingStrategy,
    AdaptiveSamplingStrategy,
    EvolutionaryStrategy,
    ClusterBasedSampling,
    IntelligentSampler
)

from .synthetic_generation import (
    SyntheticDataConfig,
    StructureGenerator,
    PhysicsPropertyGenerator,
    SyntheticDatasetGenerator
)

__all__ = [
    # Augmentation
    'AugmentationConfig',
    'CrystalSymmetryAugmenter',
    'ElectricFieldAugmenter', 
    'StructuralPerturbationAugmenter',
    'CompositionAugmenter',
    'PropertyNoiseAugmenter',
    'ComprehensiveAugmenter',
    
    # Active Learning
    'ActiveLearningConfig',
    'AcquisitionFunction',
    'UncertaintyAcquisition',
    'DiversityAcquisition',
    'PhysicsInformedAcquisition',
    'ExpectedImprovementAcquisition',
    'ActiveLearningStrategy',
    'TransferLearningStrategy',
    
    # Intelligent Sampling
    'SamplingConfig',
    'SamplingStrategy',
    'AdaptiveSamplingStrategy',
    'EvolutionaryStrategy',
    'ClusterBasedSampling',
    'IntelligentSampler',
    
    # Synthetic Generation
    'SyntheticDataConfig',
    'StructureGenerator',
    'PhysicsPropertyGenerator',
    'SyntheticDatasetGenerator'
]