# Requirements Document

## Introduction

This project aims to develop physically accurate generative machine learning models for studying and predicting topological insulators under perturbed electric fields. The system will combine state-of-the-art physics theory (including quantum Hall effect principles) with advanced diffusion-based generative models to simulate the transition between regular insulating states and topological conducting states. The computational framework will enable researchers to generate novel topological insulator configurations and predict their electronic transport properties under various electric field perturbations.

## Requirements

### Requirement 1: Physics-Based Data Representation

**User Story:** As a topological matter physicist, I want to represent topological insulator systems with their complete electronic structure and field perturbations, so that I can accurately model the physics of topological phase transitions.

#### Acceptance Criteria

1. WHEN a topological insulator structure is input THEN the system SHALL encode the crystal lattice, band structure, and topological invariants (Chern numbers, Z2 invariants)
2. WHEN electric field perturbations are applied THEN the system SHALL represent field strength, direction, and spatial distribution with physical units
3. WHEN modeling electronic states THEN the system SHALL include spin-orbit coupling effects and time-reversal symmetry considerations
4. IF the system has magnetic impurities THEN the system SHALL account for magnetic scattering effects on topological surface states
5. WHEN representing quantum Hall states THEN the system SHALL encode filling factors, Landau levels, and edge state properties

### Requirement 2: Physically Accurate Diffusion Model Architecture

**User Story:** As a generative ML researcher, I want to design diffusion models that respect physical constraints and symmetries, so that generated samples are physically realizable topological states.

#### Acceptance Criteria

1. WHEN training the diffusion model THEN the system SHALL enforce gauge invariance and time-reversal symmetry constraints
2. WHEN generating new configurations THEN the system SHALL preserve crystal symmetries and topological protection mechanisms
3. WHEN modeling the denoising process THEN the system SHALL use physics-informed loss functions that include band gap constraints and conductivity requirements
4. IF the model generates unphysical states THEN the system SHALL apply rejection sampling based on physical validity checks
5. WHEN conditioning on electric fields THEN the system SHALL ensure the generated states satisfy Maxwell's equations and current continuity

### Requirement 3: Topological Phase Classification and Prediction

**User Story:** As a researcher, I want to classify and predict topological phases of generated materials, so that I can identify novel topological insulators with desired properties.

#### Acceptance Criteria

1. WHEN analyzing a generated structure THEN the system SHALL compute topological invariants (Z2, Chern numbers) accurately
2. WHEN electric fields are applied THEN the system SHALL predict the critical field strength for topological phase transitions
3. WHEN classifying phases THEN the system SHALL distinguish between trivial insulators, topological insulators, and Weyl semimetals
4. IF surface states are present THEN the system SHALL identify and characterize Dirac cone dispersions
5. WHEN quantum Hall regime is accessed THEN the system SHALL predict quantized conductance plateaus and edge state properties

### Requirement 4: Electric Field Perturbation Modeling

**User Story:** As a physicist studying transport properties, I want to model how electric fields affect topological insulators, so that I can understand the transition from insulating to conducting behavior.

#### Acceptance Criteria

1. WHEN applying uniform electric fields THEN the system SHALL model field-induced band tilting and gap closure
2. WHEN using spatially varying fields THEN the system SHALL account for local field gradients and their effects on electronic states
3. WHEN fields exceed critical values THEN the system SHALL predict breakdown of topological protection and onset of bulk conduction
4. IF time-dependent fields are applied THEN the system SHALL model dynamic responses and Floquet topological states
5. WHEN modeling transport THEN the system SHALL compute conductivity tensors including Hall conductivity contributions

### Requirement 5: Multi-Scale Physics Integration

**User Story:** As a computational physicist, I want to integrate physics across multiple scales from atomic to mesoscopic, so that I can capture all relevant physical phenomena.

#### Acceptance Criteria

1. WHEN modeling at atomic scale THEN the system SHALL include tight-binding Hamiltonians with spin-orbit coupling
2. WHEN scaling to mesoscopic systems THEN the system SHALL use effective low-energy theories and k·p models
3. WHEN including disorder effects THEN the system SHALL model impurity scattering while preserving topological protection
4. IF temperature effects are relevant THEN the system SHALL include thermal broadening and phonon interactions
5. WHEN connecting scales THEN the system SHALL ensure consistent physical parameters across different model resolutions

### Requirement 6: Training Data Generation and Validation

**User Story:** As a machine learning researcher, I want high-quality training data with known physical properties, so that I can train robust and accurate generative models.

#### Acceptance Criteria

1. WHEN generating training data THEN the system SHALL use first-principles DFT calculations for ground truth electronic structures
2. WHEN creating field-perturbed states THEN the system SHALL solve self-consistent field equations with proper boundary conditions
3. WHEN validating generated samples THEN the system SHALL compare against experimental data and theoretical predictions
4. IF computational resources are limited THEN the system SHALL use active learning to select most informative training examples
5. WHEN augmenting datasets THEN the system SHALL apply physically meaningful transformations that preserve topological properties

### Requirement 7: Generative Model Training and Optimization

**User Story:** As a researcher, I want to train diffusion models efficiently while maintaining physical accuracy, so that I can generate novel topological materials quickly.

#### Acceptance Criteria

1. WHEN training the model THEN the system SHALL use physics-informed loss functions that penalize unphysical configurations
2. WHEN optimizing hyperparameters THEN the system SHALL balance generation quality with physical constraint satisfaction
3. WHEN monitoring training THEN the system SHALL track both ML metrics and physical property distributions
4. IF training becomes unstable THEN the system SHALL apply gradient clipping and regularization techniques
5. WHEN evaluating model performance THEN the system SHALL measure both sample quality and physical property accuracy

### Requirement 8: Interactive Research Interface

**User Story:** As a researcher, I want an intuitive interface to explore generated topological materials and their properties, so that I can efficiently discover interesting physics.

#### Acceptance Criteria

1. WHEN visualizing generated structures THEN the system SHALL display band structures, surface states, and field configurations
2. WHEN exploring parameter space THEN the system SHALL provide interactive controls for field strength, direction, and material parameters
3. WHEN analyzing results THEN the system SHALL compute and display relevant physical quantities (conductivity, topological invariants)
4. IF interesting configurations are found THEN the system SHALL allow saving and further analysis of specific samples
5. WHEN comparing samples THEN the system SHALL provide tools for statistical analysis of generated material properties

### Requirement 9: Computational Performance and Scalability

**User Story:** As a computational researcher, I want the system to handle large-scale calculations efficiently, so that I can study realistic material systems.

#### Acceptance Criteria

1. WHEN running on GPU clusters THEN the system SHALL efficiently parallelize diffusion model training and inference
2. WHEN handling large unit cells THEN the system SHALL use memory-efficient representations and sparse matrix operations
3. WHEN scaling to many samples THEN the system SHALL implement batch processing for physics calculations
4. IF memory becomes limiting THEN the system SHALL use gradient checkpointing and model sharding techniques
5. WHEN deploying models THEN the system SHALL support both local computation and cloud-based inference

### Requirement 10: Integration with Existing Physics Codes

**User Story:** As a physicist, I want to integrate with established computational physics tools, so that I can leverage existing validated implementations.

#### Acceptance Criteria

1. WHEN interfacing with DFT codes THEN the system SHALL read and write standard file formats (VASP, Quantum ESPRESSO)
2. WHEN using tight-binding models THEN the system SHALL interface with Wannier90 and PythTB libraries
3. WHEN computing transport properties THEN the system SHALL integrate with Kwant for quantum transport calculations
4. IF topological analysis is needed THEN the system SHALL use Z2Pack and other specialized topological tools
5. WHEN validating results THEN the system SHALL compare against established benchmarks and experimental databases