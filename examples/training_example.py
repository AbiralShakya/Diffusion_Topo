"""
Training Example for Topological Diffusion Generator
====================================================

This script demonstrates how to set up and run distributed training
for the topological diffusion model on HPC clusters with SLURM.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add parent directory to path
sys.path.append('..')

from topological_diffusion import (
    # Training components
    DistributedTrainer,
    DistributedTrainingConfig,
    setup_distributed_training,
    create_distributed_data_loader,
    
    # Model components
    TopologicalTransformer,
    PhysicsInformedDiffusion,
    PhysicsConstraints,
    PhysicsValidator,
    setup_physics_informed_training,
    
    # HPC components
    SlurmJobManager
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockTopologicalDataset(Dataset):
    """Mock dataset for demonstration purposes"""
    
    def __init__(self, num_samples: int = 1000, max_atoms: int = 20):
        self.num_samples = num_samples
        self.max_atoms = max_atoms
        
        # Generate mock data
        self.data = []
        for i in range(num_samples):
            # Random number of atoms
            num_atoms = np.random.randint(5, max_atoms + 1)
            
            # Mock structure data
            sample = {
                'lattice': torch.randn(3, 3),  # Random lattice
                'frac_coords': torch.rand(num_atoms, 3),  # Fractional coordinates
                'atom_types': torch.randint(0, 50, (num_atoms,)),  # Random atom types
                'num_atoms': num_atoms,
                
                # Mock targets
                'lattice_target': torch.randn(3, 3),
                'coord_target': torch.randn(num_atoms, 3),
                'species_target': torch.randint(0, 50, (num_atoms,)),
                
                # Mock physics properties
                'band_gap_target': torch.rand(1) * 2.0,  # 0-2 eV
                'formation_energy_target': torch.randn(1) * 0.1,  # Around 0 eV/atom
                'topo_class_target': torch.randint(0, 4, (1,)),  # 4 topological classes
                'conductivity_target': torch.randn(3),  # 3-component conductivity
                
                # Mock physics constraints
                'physics_constraints': {
                    'band_gap_range': (0.0, 2.0),
                    'topological_class': 'TI',
                    'stability_threshold': 0.1
                },
                
                # Timestep for diffusion
                'timestep': torch.randint(0, 1000, (1,))
            }
            
            self.data.append(sample)
            
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Custom collate function for variable-size structures"""
    # This is a simplified collate function
    # In practice, would need proper batching for variable-size graphs
    
    batch_size = len(batch)
    
    # Stack lattices
    lattices = torch.stack([item['lattice'] for item in batch])
    
    # Handle variable-size coordinates and atom types
    max_atoms = max(item['num_atoms'] for item in batch)
    
    # Pad coordinates and atom types
    frac_coords = torch.zeros(batch_size, max_atoms, 3)
    atom_types = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    
    for i, item in enumerate(batch):
        n_atoms = item['num_atoms']
        frac_coords[i, :n_atoms] = item['frac_coords']
        atom_types[i, :n_atoms] = item['atom_types']
    
    # Create mock edge indices (simplified)
    edge_indices = []
    edge_attrs = []
    batch_indices = []
    
    node_offset = 0
    for i, item in enumerate(batch):
        n_atoms = item['num_atoms']
        
        # Create simple edge connectivity (all-to-all within cutoff)
        for j in range(n_atoms):
            for k in range(j + 1, n_atoms):
                edge_indices.extend([[node_offset + j, node_offset + k],
                                   [node_offset + k, node_offset + j]])
                edge_attrs.extend([torch.randn(4), torch.randn(4)])  # Mock edge features
                
        # Batch indices for nodes
        batch_indices.extend([i] * n_atoms)
        node_offset += n_atoms
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.stack(edge_attrs)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4))
    
    batch_idx = torch.tensor(batch_indices, dtype=torch.long)
    
    # Collect other data
    timesteps = torch.cat([item['timestep'] for item in batch])
    
    # Targets
    band_gap_targets = torch.cat([item['band_gap_target'] for item in batch])
    formation_energy_targets = torch.cat([item['formation_energy_target'] for item in batch])
    topo_class_targets = torch.cat([item['topo_class_target'] for item in batch])
    
    return {
        'lattice': lattices,
        'frac_coords': frac_coords.view(-1, 3),  # Flatten for graph processing
        'atom_types': atom_types.view(-1),
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'batch': batch_idx,
        'timestep': timesteps,
        
        # Targets
        'lattice_target': torch.stack([item['lattice_target'] for item in batch]),
        'band_gap_target': band_gap_targets,
        'formation_energy_target': formation_energy_targets,
        'topo_class_target': topo_class_targets,
        
        # Physics constraints
        'physics_constraints': batch[0]['physics_constraints']  # Simplified
    }

def create_mock_data_loaders(config: DistributedTrainingConfig):
    """Create mock data loaders for training"""
    
    # Create datasets
    train_dataset = MockTopologicalDataset(num_samples=5000)
    val_dataset = MockTopologicalDataset(num_samples=1000)
    
    # Create distributed data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main_training(config_path: str, distributed: bool = False):
    """Main training function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = DistributedTrainingConfig(**config_dict)
    
    logger.info(f"Starting training with config: {config_path}")
    logger.info(f"Distributed training: {distributed}")
    
    # Create data loaders
    train_loader, val_loader = create_mock_data_loaders(config)
    
    logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Set up physics-informed training
    model, diffusion = setup_physics_informed_training(
        config.model_config,
        config.physics_config
    )
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    if distributed:
        # Set up distributed training
        trainer = setup_distributed_training(
            config, model, train_loader, val_loader, diffusion
        )
        
        logger.info("Set up distributed training")
        
        # Start training
        trainer.train(train_loader, val_loader, diffusion)
        
        # Cleanup
        trainer.cleanup()
        
    else:
        # Single-process training (for testing)
        logger.info("Running single-process training")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Simple training loop
        model.train()
        for epoch in range(min(5, config.max_epochs)):  # Limited epochs for demo
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                # Move to device
                batch_data = {k: v.to(device) if torch.is_tensor(v) else v 
                             for k, v in batch_data.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                Lt = batch_data['lattice']
                Ft = batch_data['frac_coords']
                At = batch_data['atom_types']
                edge_index = batch_data['edge_index']
                edge_attr = batch_data['edge_attr']
                batch = batch_data['batch']
                t = batch_data['timestep']
                
                try:
                    model_output = model(Lt, Ft, At, edge_index, edge_attr, batch, t)
                    
                    # Simple loss (mock)
                    if len(model_output) >= 3:
                        loss = sum(torch.mean(output**2) for output in model_output[:3])
                    else:
                        loss = torch.tensor(0.0, device=device)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                        
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
                    
                # Limit batches for demo
                if batch_idx >= 20:
                    break
                    
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch} completed, Average Loss: {avg_loss:.6f}")
    
    logger.info("Training completed successfully!")

def create_example_config():
    """Create example training configuration"""
    
    config = {
        # Model parameters
        "model_config": {
            "num_species": 50,
            "hidden_dim": 128,  # Smaller for demo
            "num_topo_classes": 4
        },
        
        # Physics parameters
        "physics_config": {
            "topological_class": "TI",
            "band_gap_range": [0.1, 0.5],
            "target_chern_number": 1,
            "stability_threshold": 0.1,
            "constraint_weight": 1.0
        },
        
        # Training parameters
        "batch_size": 4,  # Small for demo
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "max_epochs": 10,  # Limited for demo
        "warmup_epochs": 2,
        
        # Distributed parameters
        "world_size": 1,  # Single process for demo
        "backend": "nccl",
        "mixed_precision": True,
        "gradient_compression": False,  # Disabled for single process
        "zero_optimizer": False,
        
        # Checkpointing
        "checkpoint_dir": "./checkpoints",
        "checkpoint_interval": 5,
        "keep_n_checkpoints": 3,
        
        # Monitoring
        "log_interval": 5,
        "eval_interval": 10,
        
        # SLURM parameters (for distributed runs)
        "slurm_job_name": "topological_diffusion_demo",
        "slurm_partition": "gpu",
        "slurm_time": "02:00:00",
        "slurm_nodes": 1,
        "slurm_gpus_per_node": 1,
        "slurm_cpus_per_task": 4,
        "slurm_memory": "32G"
    }
    
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topological Diffusion Training Example")
    parser.add_argument("--config", type=str, default="training_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    parser.add_argument("--create-config", action="store_true",
                       help="Create example configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        # Create example configuration
        config = create_example_config()
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created example configuration: {args.config}")
        
    else:
        # Check if config exists
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            logger.info("Run with --create-config to create an example configuration")
            sys.exit(1)
            
        # Run training
        main_training(args.config, args.distributed)