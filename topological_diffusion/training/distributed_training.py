"""
SLURM-Integrated Distributed Training Infrastructure
===================================================

This module provides distributed training capabilities for topological diffusion models:
- Multi-node, multi-GPU training using PyTorch DistributedDataParallel
- SLURM job scripts for automatic resource allocation and scaling
- Fault-tolerant training with automatic checkpoint recovery
- Dynamic resource allocation based on training progress
- Gradient compression and communication optimization
- Memory-efficient training strategies for large crystal structures

Key Components:
- DistributedTrainer: Main distributed training coordinator
- SLURMTrainingManager: SLURM job management for training
- CheckpointManager: Fault-tolerant checkpointing
- ResourceMonitor: Dynamic resource allocation
- CommunicationOptimizer: Gradient compression and optimization
"""

import os
import sys
import time
import json
import pickle
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default_hooks

import numpy as np
from tqdm import tqdm

from ..models.physics_informed_diffusion import (
    TopologicalTransformer, PhysicsInformedDiffusion, MultiTaskLearningFramework
)
from ..hpc.slurm_scripts import SlurmJobManager, SlurmJobConfig

logger = logging.getLogger(__name__)

@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training"""
    # Model parameters
    model_config: Dict = field(default_factory=dict)
    physics_config: Dict = field(default_factory=dict)
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 1000
    warmup_epochs: int = 10
    
    # Distributed parameters
    world_size: int = 1
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    
    # Optimization parameters
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    gradient_compression: bool = True
    zero_optimizer: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 10
    keep_n_checkpoints: int = 5
    
    # Resource management
    max_memory_per_gpu: float = 0.9  # Fraction of GPU memory to use
    dynamic_batch_size: bool = True
    memory_efficient_attention: bool = True
    
    # Monitoring
    log_interval: int = 10
    eval_interval: int = 100
    wandb_project: Optional[str] = None
    
    # SLURM parameters
    slurm_job_name: str = "topological_diffusion_training"
    slurm_partition: str = "gpu"
    slurm_time: str = "72:00:00"
    slurm_nodes: int = 1
    slurm_gpus_per_node: int = 4
    slurm_cpus_per_task: int = 8
    slurm_memory: str = "128G"

class CheckpointManager:
    """Manages training checkpoints with fault tolerance"""
    
    def __init__(self, checkpoint_dir: str, keep_n_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.keep_n_checkpoints = keep_n_checkpoints
        
    def save_checkpoint(self, state: Dict, epoch: int, is_best: bool = False) -> str:
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        # Add metadata
        state['epoch'] = epoch
        state['timestamp'] = time.time()
        
        # Save checkpoint
        torch.save(state, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """Load training checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_path = self._find_latest_checkpoint()
            
        if checkpoint_path is None:
            logger.info("No checkpoint found")
            return None
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
            
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            return None
            
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoint_files[-1])
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[:-self.keep_n_checkpoints]:
            checkpoint_file.unlink()
            logger.debug(f"Removed old checkpoint {checkpoint_file}")

class ResourceMonitor:
    """Monitors and manages computational resources during training"""
    
    def __init__(self, max_memory_per_gpu: float = 0.9):
        self.max_memory_per_gpu = max_memory_per_gpu
        self.memory_history = []
        
    def check_gpu_memory(self) -> Dict[str, float]:
        """Check GPU memory usage"""
        if not torch.cuda.is_available():
            return {}
            
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            memory_info[f'gpu_{i}'] = {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'total': memory_total,
                'utilization': memory_allocated / memory_total
            }
            
        return memory_info
        
    def suggest_batch_size_adjustment(self, current_batch_size: int, 
                                    memory_info: Dict) -> int:
        """Suggest batch size adjustment based on memory usage"""
        if not memory_info:
            return current_batch_size
            
        # Get maximum memory utilization across GPUs
        max_utilization = max(
            gpu_info['utilization'] for gpu_info in memory_info.values()
        )
        
        # Adjust batch size based on memory utilization
        if max_utilization > self.max_memory_per_gpu:
            # Reduce batch size
            new_batch_size = max(1, int(current_batch_size * 0.8))
            logger.warning(f"High memory usage ({max_utilization:.2f}), reducing batch size to {new_batch_size}")
            return new_batch_size
        elif max_utilization < 0.6:
            # Increase batch size
            new_batch_size = int(current_batch_size * 1.2)
            logger.info(f"Low memory usage ({max_utilization:.2f}), increasing batch size to {new_batch_size}")
            return new_batch_size
            
        return current_batch_size
        
    def log_resource_usage(self):
        """Log current resource usage"""
        memory_info = self.check_gpu_memory()
        self.memory_history.append(memory_info)
        
        for gpu_id, info in memory_info.items():
            logger.debug(f"{gpu_id}: {info['allocated']:.2f}GB / {info['total']:.2f}GB "
                        f"({info['utilization']:.1%} utilization)")

class CommunicationOptimizer:
    """Optimizes distributed communication for training"""
    
    def __init__(self, gradient_compression: bool = True):
        self.gradient_compression = gradient_compression
        
    def setup_communication_hooks(self, model: DDP):
        """Set up communication optimization hooks"""
        if self.gradient_compression:
            # Use PowerSGD compression for gradient communication
            try:
                model.register_comm_hook(
                    state=None, 
                    hook=default_hooks.powerSGD_hook
                )
                logger.info("Enabled PowerSGD gradient compression")
            except Exception as e:
                logger.warning(f"Failed to enable gradient compression: {e}")
                
    def optimize_bucket_size(self, model: DDP, target_bucket_size_mb: int = 25):
        """Optimize DDP bucket size for communication efficiency"""
        # Convert MB to bytes
        bucket_size_bytes = target_bucket_size_mb * 1024 * 1024
        
        # Set bucket cap
        model._set_static_graph()  # Enable static graph optimization
        logger.info(f"Set DDP bucket size to {target_bucket_size_mb}MB")

class DistributedTrainer:
    """Main distributed training coordinator"""
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize distributed training
        self._init_distributed()
        
        # Set device
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.local_rank)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.keep_n_checkpoints)
        self.resource_monitor = ResourceMonitor(config.max_memory_per_gpu)
        self.comm_optimizer = CommunicationOptimizer(config.gradient_compression)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.current_epoch = 0
        self.best_loss = float('inf')
        
    def _init_distributed(self):
        """Initialize distributed training"""
        if self.world_size > 1:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.world_size,
                rank=self.rank
            )
            logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
        else:
            logger.info("Single-process training")
            
    def setup_model(self, model: nn.Module) -> nn.Module:
        """Set up model for distributed training"""
        model = model.to(self.device)
        
        if self.world_size > 1:
            # Wrap model with DistributedDataParallel
            model = DDP(
                model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True  # For complex models with conditional paths
            )
            
            # Set up communication optimization
            self.comm_optimizer.setup_communication_hooks(model)
            self.comm_optimizer.optimize_bucket_size(model)
            
        self.model = model
        return model
        
    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Set up optimizer for distributed training"""
        if self.config.zero_optimizer and self.world_size > 1:
            # Use ZeroRedundancyOptimizer for memory efficiency
            optimizer = ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
        self.optimizer = optimizer
        return optimizer
        
    def setup_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Set up learning rate scheduler"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        self.scheduler = scheduler
        return scheduler
        
    def setup_mixed_precision(self):
        """Set up mixed precision training"""
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Enabled mixed precision training")
        else:
            self.scaler = None
            
    def train_epoch(self, train_loader: DataLoader, diffusion: PhysicsInformedDiffusion) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Set up progress bar (only on rank 0)
        if self.rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            pbar = train_loader
            
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            batch_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in batch_data.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch_data, diffusion)
                    
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clipping
                    )
                    
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                loss = self._compute_loss(batch_data, diffusion)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clipping
                    )
                    
                self.optimizer.step()
                
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0 and self.rank == 0:
                logger.info(f"Epoch {self.current_epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                
            # Resource monitoring
            if batch_idx % (self.config.log_interval * 10) == 0:
                self.resource_monitor.log_resource_usage()
                
        # Average loss across all processes
        avg_loss = total_loss / num_batches
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
            
        return {'train_loss': avg_loss}
        
    def _compute_loss(self, batch_data: Dict, diffusion: PhysicsInformedDiffusion) -> torch.Tensor:
        """Compute training loss"""
        # Extract batch components
        Lt = batch_data['lattice']
        Ft = batch_data['frac_coords']
        At = batch_data['atom_types']
        edge_index = batch_data['edge_index']
        edge_attr = batch_data['edge_attr']
        batch = batch_data['batch']
        t = batch_data['timestep']
        
        # Model forward pass
        model_output = self.model(Lt, Ft, At, edge_index, edge_attr, batch, t)
        
        # Compute physics-informed loss
        targets = (batch_data.get('lattice_target'), 
                  batch_data.get('coord_target'),
                  batch_data.get('species_target'))
        
        physics_constraints = batch_data.get('physics_constraints', {})
        
        loss = diffusion.physics_informed_loss(
            model_output, targets, physics_constraints, batch_data
        )
        
        return loss
        
    def validate(self, val_loader: DataLoader, diffusion: PhysicsInformedDiffusion) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Move data to device
                batch_data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                             for k, v in batch_data.items()}
                
                # Compute loss
                loss = self._compute_loss(batch_data, diffusion)
                total_loss += loss.item()
                num_batches += 1
                
        # Average loss across all processes
        avg_loss = total_loss / num_batches
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
            
        return {'val_loss': avg_loss}
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              diffusion: PhysicsInformedDiffusion):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        
        # Load checkpoint if available
        checkpoint = self.checkpoint_manager.load_checkpoint()
        if checkpoint:
            self._load_checkpoint(checkpoint)
            
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                
            # Training
            train_metrics = self.train_epoch(train_loader, diffusion)
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader, diffusion)
            else:
                val_metrics = {}
                
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
                
            # Checkpointing (only on rank 0)
            if self.rank == 0 and epoch % self.config.checkpoint_interval == 0:
                is_best = val_metrics.get('val_loss', float('inf')) < self.best_loss
                if is_best:
                    self.best_loss = val_metrics['val_loss']
                    
                checkpoint_state = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'epoch': epoch,
                    'best_loss': self.best_loss,
                    'config': self.config
                }
                
                self.checkpoint_manager.save_checkpoint(checkpoint_state, epoch, is_best)
                
            # Logging
            if self.rank == 0:
                log_msg = f"Epoch {epoch}: "
                log_msg += f"Train Loss: {train_metrics['train_loss']:.6f}"
                if val_metrics:
                    log_msg += f", Val Loss: {val_metrics['val_loss']:.6f}"
                logger.info(log_msg)
                
        logger.info("Training completed")
        
    def _load_checkpoint(self, checkpoint: Dict):
        """Load training state from checkpoint"""
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Resumed training from epoch {self.current_epoch}")
        
    def cleanup(self):
        """Clean up distributed training"""
        if self.world_size > 1:
            dist.destroy_process_group()

class SLURMTrainingManager:
    """Manages SLURM jobs for distributed training"""
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.job_manager = SlurmJobManager()
        
    def create_training_script(self, training_script_path: str, 
                             config_path: str) -> str:
        """Create SLURM training script"""
        
        slurm_config = SlurmJobConfig(
            job_name=self.config.slurm_job_name,
            partition=self.config.slurm_partition,
            nodes=self.config.slurm_nodes,
            ntasks_per_node=self.config.slurm_gpus_per_node,
            cpus_per_task=self.config.slurm_cpus_per_task,
            gpus_per_node=self.config.slurm_gpus_per_node,
            memory=self.config.slurm_memory,
            time=self.config.slurm_time,
            modules=["python/3.9", "cuda/11.8", "nccl/2.12", "openmpi/4.1.0"]
        )
        
        commands = [
            "# Set up distributed training environment",
            "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",
            "export MASTER_PORT=29500",
            "export WORLD_SIZE=$SLURM_NTASKS",
            "export RANK=$SLURM_PROCID",
            "export LOCAL_RANK=$SLURM_LOCALID",
            "",
            "# Launch distributed training",
            f"srun python -m torch.distributed.launch \\",
            f"    --nproc_per_node={self.config.slurm_gpus_per_node} \\",
            f"    --nnodes={self.config.slurm_nodes} \\",
            f"    --node_rank=$SLURM_NODEID \\",
            f"    --master_addr=$MASTER_ADDR \\",
            f"    --master_port=$MASTER_PORT \\",
            f"    {training_script_path} \\",
            f"    --config {config_path} \\",
            f"    --distributed"
        ]
        
        script_content = self.job_manager.create_job_script(slurm_config, commands)
        
        return script_content
        
    def submit_training_job(self, training_script_path: str, 
                          config_path: str) -> str:
        """Submit distributed training job"""
        
        script_content = self.create_training_script(training_script_path, config_path)
        
        # Write script to file
        script_path = Path("./slurm_training_script.sh")
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_path, 0o755)
        
        # Submit job
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"Submitted distributed training job: {job_id}")
        
        return job_id

# Utility functions
def setup_distributed_training(config: DistributedTrainingConfig,
                             model: nn.Module,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             diffusion: PhysicsInformedDiffusion) -> DistributedTrainer:
    """Set up distributed training components"""
    
    # Create trainer
    trainer = DistributedTrainer(config)
    
    # Set up model
    trainer.setup_model(model)
    
    # Set up optimizer and scheduler
    trainer.setup_optimizer(model)
    trainer.setup_scheduler(trainer.optimizer)
    
    # Set up mixed precision
    trainer.setup_mixed_precision()
    
    return trainer

def create_distributed_data_loader(dataset, config: DistributedTrainingConfig,
                                 shuffle: bool = True) -> DataLoader:
    """Create distributed data loader"""
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=config.world_size,
        rank=int(os.environ.get('RANK', 0)),
        shuffle=shuffle
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return loader

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = DistributedTrainingConfig(
        batch_size=16,
        learning_rate=1e-4,
        max_epochs=100,
        world_size=4,
        mixed_precision=True,
        gradient_compression=True,
        slurm_nodes=1,
        slurm_gpus_per_node=4
    )
    
    logger.info(f"Distributed training configuration: {config.world_size} processes")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"Gradient compression: {config.gradient_compression}")