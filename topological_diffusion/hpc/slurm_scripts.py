"""
SLURM Job Management for HPC Calculations
========================================

This module provides utilities for submitting and managing SLURM jobs
for computationally intensive physics calculations.
"""

import os
import subprocess
import time
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

@dataclass
class SlurmJobConfig:
    """Configuration for SLURM job submission"""
    job_name: str
    partition: str = "gpu"
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 8
    gpus_per_node: int = 1
    memory: str = "32G"
    time: str = "24:00:00"
    output_file: str = "slurm-%j.out"
    error_file: str = "slurm-%j.err"
    email: Optional[str] = None
    email_type: str = "END,FAIL"
    modules: List[str] = None
    conda_env: Optional[str] = None
    working_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.modules is None:
            self.modules = ["python/3.9", "cuda/11.8", "openmpi/4.1.0"]

class SlurmJobManager:
    """Manager for SLURM job submission and monitoring"""
    
    def __init__(self, base_dir: str = "./slurm_jobs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.job_history = {}
        
    def create_job_script(self, config: SlurmJobConfig, commands: List[str]) -> str:
        """Create SLURM job script"""
        
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={config.job_name}",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --gres=gpu:{config.gpus_per_node}",
            f"#SBATCH --mem={config.memory}",
            f"#SBATCH --time={config.time}",
            f"#SBATCH --output={config.output_file}",
            f"#SBATCH --error={config.error_file}",
        ]
        
        if config.email:
            script_lines.extend([
                f"#SBATCH --mail-user={config.email}",
                f"#SBATCH --mail-type={config.email_type}"
            ])
            
        if config.working_dir:
            script_lines.append(f"#SBATCH --chdir={config.working_dir}")
            
        script_lines.extend([
            "",
            "# Load modules",
        ])
        
        for module in config.modules:
            script_lines.append(f"module load {module}")
            
        if config.conda_env:
            script_lines.extend([
                "",
                f"# Activate conda environment", 
                f"source activate {config.conda_env}",
            ])
            
        script_lines.extend([
            "",
            "# Set environment variables",
            "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
            "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID",
            "",
            "# Job commands",
        ])
        
        script_lines.extend(commands)
        
        return "\n".join(script_lines)
    
    def submit_job(self, config: SlurmJobConfig, commands: List[str]) -> str:
        """Submit job to SLURM queue"""
        
        # Create job script
        script_content = self.create_job_script(config, commands)
        script_path = self.base_dir / f"{config.job_name}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Submit job
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract job ID
            job_id = result.stdout.strip().split()[-1]
            
            # Store job info
            self.job_history[job_id] = {
                'config': config,
                'script_path': script_path,
                'submit_time': time.time(),
                'status': 'SUBMITTED'
            }
            
            logger.info(f"Submitted job {job_id}: {config.job_name}")
            return job_id
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    def check_job_status(self, job_id: str) -> str:
        """Check status of submitted job"""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                check=True
            )
            
            status = result.stdout.strip()
            if job_id in self.job_history:
                self.job_history[job_id]['status'] = status
                
            return status
            
        except subprocess.CalledProcessError:
            # Job might be completed and no longer in queue
            return "COMPLETED"
    
    def wait_for_job(self, job_id: str, check_interval: int = 60) -> str:
        """Wait for job to complete"""
        logger.info(f"Waiting for job {job_id} to complete...")
        
        while True:
            status = self.check_job_status(job_id)
            
            if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                logger.info(f"Job {job_id} finished with status: {status}")
                return status
                
            logger.debug(f"Job {job_id} status: {status}")
            time.sleep(check_interval)
    
    def cancel_job(self, job_id: str):
        """Cancel submitted job"""
        try:
            subprocess.run(["scancel", job_id], check=True)
            logger.info(f"Cancelled job {job_id}")
            
            if job_id in self.job_history:
                self.job_history[job_id]['status'] = 'CANCELLED'
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
    
    def get_job_info(self, job_id: str) -> Dict:
        """Get detailed job information"""
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job info (simplified)
            info = {}
            for line in result.stdout.split('\n'):
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        info[key] = value
                        
            return info
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get job info for {job_id}: {e}")
            return {}

class PhysicsCalculationJob:
    """Specialized job for physics calculations"""
    
    def __init__(self, job_manager: SlurmJobManager):
        self.job_manager = job_manager
        
    def submit_hamiltonian_calculation(self, 
                                     material_name: str,
                                     k_grid_size: tuple,
                                     output_dir: str,
                                     config_overrides: Dict = None) -> str:
        """Submit Hamiltonian eigenvalue calculation job"""
        
        config = SlurmJobConfig(
            job_name=f"hamiltonian_{material_name}",
            partition="gpu",
            cpus_per_task=16,
            memory="64G",
            time="12:00:00"
        )
        
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        
        commands = [
            f"cd {output_dir}",
            "python -c \"",
            "import sys",
            "sys.path.append('/path/to/topological_diffusion')",
            "from topological_diffusion.physics import HamiltonianFactory",
            "import numpy as np",
            "",
            f"# Create {material_name} model",
            f"model = HamiltonianFactory.create_from_material('{material_name}')",
            "",
            "# Generate k-grid",
            f"k_grid = generate_k_grid(model.reciprocal_vectors, {k_grid_size})",
            "",
            "# Solve eigenvalue problem",
            "solver = BlochHamiltonianSolver(model)",
            "eigenvalues, eigenvectors = solver.solve_eigenvalue_problem(k_grid)",
            "",
            "# Save results",
            "np.savez('hamiltonian_results.npz', ",
            "         eigenvalues=eigenvalues,",
            "         eigenvectors=eigenvectors,",
            "         k_grid=k_grid)",
            "print('Calculation completed successfully')",
            "\""
        ]
        
        return self.job_manager.submit_job(config, commands)
    
    def submit_band_structure_calculation(self,
                                        material_name: str, 
                                        k_path_points: List[str],
                                        num_k_points: int,
                                        output_dir: str) -> str:
        """Submit band structure calculation job"""
        
        config = SlurmJobConfig(
            job_name=f"bands_{material_name}",
            partition="cpu",
            cpus_per_task=8,
            memory="32G", 
            time="4:00:00"
        )
        
        commands = [
            f"cd {output_dir}",
            "python -c \"",
            "import sys",
            "sys.path.append('/path/to/topological_diffusion')",
            "from topological_diffusion.physics import HamiltonianFactory, generate_k_path",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "",
            f"# Create {material_name} model",
            f"model = HamiltonianFactory.create_from_material('{material_name}')",
            "",
            "# Define high-symmetry points (material-specific)",
            "if 'graphene' in material_name.lower():",
            "    high_sym_points = {",
            "        'Γ': np.array([0, 0, 0]),",
            "        'K': np.array([4*np.pi/(3*np.sqrt(3)), 0, 0]),",
            "        'M': np.array([np.pi/np.sqrt(3), np.pi/3, 0])",
            "    }",
            "else:",
            "    # Default cubic points",
            "    high_sym_points = {",
            "        'Γ': np.array([0, 0, 0]),",
            "        'X': np.array([np.pi, 0, 0]),",
            "        'M': np.array([np.pi, np.pi, 0]),",
            "        'R': np.array([np.pi, np.pi, np.pi])",
            "    }",
            "",
            f"# Generate k-path",
            f"k_path, distances = generate_k_path(high_sym_points, {k_path_points}, {num_k_points})",
            "",
            "# Compute band structure",
            "eigenvalues, eigenvectors = model.get_band_structure(k_path)",
            "",
            "# Save results",
            "np.savez('band_structure.npz',",
            "         eigenvalues=eigenvalues,",
            "         eigenvectors=eigenvectors,", 
            "         k_path=k_path,",
            "         distances=distances)",
            "",
            "# Create plot",
            "plt.figure(figsize=(10, 6))",
            "for band in range(eigenvalues.shape[1]):",
            "    plt.plot(distances, eigenvalues[:, band], 'b-', alpha=0.7)",
            "plt.xlabel('k-path')",
            "plt.ylabel('Energy (eV)')",
            f"plt.title('{material_name} Band Structure')",
            "plt.grid(True, alpha=0.3)",
            "plt.savefig('band_structure.png', dpi=300, bbox_inches='tight')",
            "print('Band structure calculation completed')",
            "\""
        ]
        
        return self.job_manager.submit_job(config, commands)

def create_distributed_training_script(num_nodes: int, gpus_per_node: int, 
                                     training_script: str, config_file: str) -> str:
    """Create SLURM script for distributed training"""
    
    config = SlurmJobConfig(
        job_name="topological_diffusion_training",
        partition="gpu",
        nodes=num_nodes,
        ntasks_per_node=gpus_per_node,
        cpus_per_task=8,
        gpus_per_node=gpus_per_node,
        memory="128G",
        time="72:00:00",
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
        f"    --nproc_per_node={gpus_per_node} \\",
        f"    --nnodes={num_nodes} \\",
        f"    --node_rank=$SLURM_NODEID \\",
        f"    --master_addr=$MASTER_ADDR \\",
        f"    --master_port=$MASTER_PORT \\",
        f"    {training_script} \\",
        f"    --config {config_file} \\",
        f"    --distributed"
    ]
    
    job_manager = SlurmJobManager()
    script_content = job_manager.create_job_script(config, commands)
    
    return script_content

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create job manager
    job_manager = SlurmJobManager()
    physics_jobs = PhysicsCalculationJob(job_manager)
    
    # Submit Hamiltonian calculation
    job_id = physics_jobs.submit_hamiltonian_calculation(
        material_name="graphene",
        k_grid_size=(50, 50, 1),
        output_dir="./calculations/graphene"
    )
    
    print(f"Submitted job: {job_id}")
    
    # Wait for completion
    status = job_manager.wait_for_job(job_id)
    print(f"Job completed with status: {status}")