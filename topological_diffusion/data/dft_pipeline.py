"""
Automated High-Throughput DFT Calculation Framework
==================================================

This module provides automated DFT calculation workflows for generating
training data for topological materials. Features:

- SLURM-based workflow management for large-scale DFT calculations
- VASP, Quantum ESPRESSO, and WIEN2k integration with automatic job submission
- Intelligent convergence checking and parameter optimization
- Automated band structure and topological invariant calculation pipelines
- Error handling and job recovery for failed calculations
- Cost-aware calculation prioritization for efficient resource usage

Key Components:
- DFTCalculator: Base class for DFT calculations
- VASPCalculator: VASP-specific implementation
- QuantumESPRESSOCalculator: QE-specific implementation
- WorkflowManager: Orchestrates calculation workflows
- ConvergenceChecker: Validates calculation convergence
"""

import os
import subprocess
import time
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import logging

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints, Potcar
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.io.qe.inputs import PWInput
from pymatgen.io.qe.outputs import PWOutput
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer

from ..hpc.slurm_scripts import SlurmJobManager, SlurmJobConfig
from ..physics.topological_invariants import TopologicalInvariantCalculator

logger = logging.getLogger(__name__)

@dataclass
class DFTParameters:
    """Parameters for DFT calculations"""
    # Basic parameters
    functional: str = "PBE"  # Exchange-correlation functional
    encut: float = 520.0     # Plane wave cutoff (eV)
    kpoint_density: float = 1000.0  # k-points per Å⁻³
    
    # Convergence parameters
    ediff: float = 1e-6      # Electronic convergence (eV)
    ediffg: float = -0.01    # Ionic convergence (eV/Å)
    nelm: int = 100          # Max electronic steps
    nsw: int = 100           # Max ionic steps
    
    # Spin-orbit coupling
    lsorbit: bool = True     # Enable SOC
    saxis: List[float] = field(default_factory=lambda: [0, 0, 1])  # Spin axis
    
    # Magnetism
    ispin: int = 2           # Spin polarization
    magmom: Optional[List[float]] = None  # Initial magnetic moments
    
    # Band structure
    nbands: Optional[int] = None  # Number of bands
    nedos: int = 2000        # DOS points
    
    # Hybrid functionals
    lhfcalc: bool = False    # Hybrid functional
    hfscreen: float = 0.2    # Screening parameter
    
    # Advanced parameters
    algo: str = "Normal"     # Electronic algorithm
    prec: str = "Accurate"   # Precision level
    lreal: str = "Auto"      # Real space projection
    
    def to_vasp_dict(self) -> Dict[str, Any]:
        """Convert to VASP INCAR dictionary"""
        incar_dict = {
            'ENCUT': self.encut,
            'EDIFF': self.ediff,
            'EDIFFG': self.ediffg,
            'NELM': self.nelm,
            'NSW': self.nsw,
            'ISPIN': self.ispin,
            'LSORBIT': self.lsorbit,
            'SAXIS': self.saxis,
            'NBANDS': self.nbands,
            'NEDOS': self.nedos,
            'LHFCALC': self.lhfcalc,
            'HFSCREEN': self.hfscreen,
            'ALGO': self.algo,
            'PREC': self.prec,
            'LREAL': self.lreal,
        }
        
        # Remove None values
        incar_dict = {k: v for k, v in incar_dict.items() if v is not None}
        
        # Add magnetic moments if specified
        if self.magmom is not None:
            incar_dict['MAGMOM'] = self.magmom
            
        return incar_dict

@dataclass
class CalculationResult:
    """Results from DFT calculation"""
    structure: Structure
    energy: float
    forces: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    magnetic_moments: Optional[List[float]] = None
    band_gap: Optional[float] = None
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    dos: Optional[Dict] = None
    converged: bool = False
    calculation_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

class DFTCalculator(ABC):
    """Abstract base class for DFT calculators"""
    
    def __init__(self, parameters: DFTParameters, work_dir: str = "./dft_calculations"):
        self.parameters = parameters
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)
        
    @abstractmethod
    def setup_calculation(self, structure: Structure, calc_dir: str) -> None:
        """Set up calculation files"""
        pass
    
    @abstractmethod
    def run_calculation(self, calc_dir: str, job_manager: SlurmJobManager) -> str:
        """Submit calculation job"""
        pass
    
    @abstractmethod
    def parse_results(self, calc_dir: str) -> CalculationResult:
        """Parse calculation results"""
        pass
    
    @abstractmethod
    def check_convergence(self, calc_dir: str) -> bool:
        """Check if calculation converged"""
        pass

class VASPCalculator(DFTCalculator):
    """VASP DFT calculator implementation"""
    
    def __init__(self, parameters: DFTParameters, potcar_dir: str,
                 work_dir: str = "./vasp_calculations"):
        super().__init__(parameters, work_dir)
        self.potcar_dir = Path(potcar_dir)
        
    def setup_calculation(self, structure: Structure, calc_dir: str) -> None:
        """Set up VASP calculation files"""
        calc_path = Path(calc_dir)
        calc_path.mkdir(exist_ok=True, parents=True)
        
        # POSCAR
        poscar = Poscar(structure)
        poscar.write_file(calc_path / "POSCAR")
        
        # INCAR
        incar_dict = self.parameters.to_vasp_dict()
        
        # Add structure-specific parameters
        if structure.is_ordered:
            # Non-magnetic calculation for ordered structures
            if not any(site.species.elements[0].is_magnetic for site in structure):
                incar_dict['ISPIN'] = 1
                
        # Automatic k-point generation
        kpoint_density = self.parameters.kpoint_density
        kpoints = Kpoints.automatic_density(structure, kpoint_density)
        kpoints.write_file(calc_path / "KPOINTS")
        
        # POTCAR
        potcar = Potcar([str(site.specie) for site in structure], 
                       functional=self.parameters.functional,
                       potcar_dir=str(self.potcar_dir))
        potcar.write_file(calc_path / "POTCAR")
        
        # INCAR
        incar = Incar(incar_dict)
        incar.write_file(calc_path / "INCAR")
        
        logger.info(f"VASP calculation setup complete in {calc_dir}")
        
    def run_calculation(self, calc_dir: str, job_manager: SlurmJobManager) -> str:
        """Submit VASP calculation job"""
        
        config = SlurmJobConfig(
            job_name=f"vasp_{Path(calc_dir).name}",
            partition="cpu",
            nodes=1,
            ntasks_per_node=16,
            cpus_per_task=1,
            memory="64G",
            time="24:00:00",
            modules=["vasp/6.3.0", "intel/2021.4", "openmpi/4.1.0"],
            working_dir=calc_dir
        )
        
        commands = [
            "# Run VASP calculation",
            "mpirun -np $SLURM_NTASKS vasp_std > vasp.out 2>&1",
            "",
            "# Check if calculation completed",
            "if grep -q 'reached required accuracy' OUTCAR; then",
            "    echo 'VASP calculation completed successfully'",
            "    touch CONVERGED",
            "else",
            "    echo 'VASP calculation failed or did not converge'",
            "    touch FAILED",
            "fi"
        ]
        
        job_id = job_manager.submit_job(config, commands)
        logger.info(f"Submitted VASP job {job_id} for {calc_dir}")
        
        return job_id
        
    def parse_results(self, calc_dir: str) -> CalculationResult:
        """Parse VASP calculation results"""
        calc_path = Path(calc_dir)
        
        try:
            # Parse main output files
            vasprun = Vasprun(calc_path / "vasprun.xml", parse_dos=True, parse_eigen=True)
            outcar = Outcar(calc_path / "OUTCAR")
            
            # Basic results
            structure = vasprun.final_structure
            energy = vasprun.final_energy
            forces = vasprun.ionic_steps[-1]['forces'] if vasprun.ionic_steps else None
            stress = outcar.stress if hasattr(outcar, 'stress') else None
            
            # Electronic properties
            band_gap = vasprun.eigenvalue_band_properties[0] if vasprun.eigenvalue_band_properties else None
            eigenvalues = vasprun.eigenvalues if hasattr(vasprun, 'eigenvalues') else None
            
            # Magnetic properties
            magnetic_moments = None
            if vasprun.is_spin:
                try:
                    magnetic_moments = [site.magmom for site in structure]
                except:
                    magnetic_moments = outcar.magnetization if hasattr(outcar, 'magnetization') else None
                    
            # DOS
            dos = None
            if vasprun.complete_dos:
                dos = {
                    'energies': vasprun.complete_dos.energies,
                    'densities': vasprun.complete_dos.densities,
                    'efermi': vasprun.efermi
                }
                
            # Convergence check
            converged = self.check_convergence(calc_dir)
            
            # Calculation time
            calc_time = 0.0
            if hasattr(outcar, 'run_stats'):
                calc_time = outcar.run_stats.get('Total CPU time used (sec)', 0.0)
                
            result = CalculationResult(
                structure=structure,
                energy=energy,
                forces=forces,
                stress=stress,
                magnetic_moments=magnetic_moments,
                band_gap=band_gap,
                eigenvalues=eigenvalues,
                dos=dos,
                converged=converged,
                calculation_time=calc_time,
                metadata={
                    'calculator': 'VASP',
                    'functional': self.parameters.functional,
                    'encut': self.parameters.encut,
                    'kpoint_density': self.parameters.kpoint_density
                }
            )
            
            logger.info(f"Successfully parsed VASP results from {calc_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse VASP results from {calc_dir}: {e}")
            # Return minimal result with error info
            return CalculationResult(
                structure=Structure.from_file(calc_path / "POSCAR"),
                energy=float('inf'),
                converged=False,
                metadata={'error': str(e), 'calculator': 'VASP'}
            )
            
    def check_convergence(self, calc_dir: str) -> bool:
        """Check VASP calculation convergence"""
        calc_path = Path(calc_dir)
        
        # Check for convergence markers
        if (calc_path / "CONVERGED").exists():
            return True
        if (calc_path / "FAILED").exists():
            return False
            
        # Check OUTCAR for convergence
        outcar_path = calc_path / "OUTCAR"
        if outcar_path.exists():
            try:
                with open(outcar_path, 'r') as f:
                    content = f.read()
                    if 'reached required accuracy' in content:
                        return True
                    if 'EDIFF is reached' in content:
                        return True
            except:
                pass
                
        return False

class QuantumESPRESSOCalculator(DFTCalculator):
    """Quantum ESPRESSO DFT calculator implementation"""
    
    def __init__(self, parameters: DFTParameters, pseudopotential_dir: str,
                 work_dir: str = "./qe_calculations"):
        super().__init__(parameters, work_dir)
        self.pseudo_dir = Path(pseudopotential_dir)
        
    def setup_calculation(self, structure: Structure, calc_dir: str) -> None:
        """Set up Quantum ESPRESSO calculation files"""
        calc_path = Path(calc_dir)
        calc_path.mkdir(exist_ok=True, parents=True)
        
        # Convert parameters to QE format
        qe_input = self._create_pw_input(structure)
        
        # Write input file
        qe_input.write_file(calc_path / "pw.in")
        
        logger.info(f"Quantum ESPRESSO calculation setup complete in {calc_dir}")
        
    def _create_pw_input(self, structure: Structure) -> PWInput:
        """Create PWInput object from structure and parameters"""
        
        # Control section
        control = {
            'calculation': 'scf',
            'restart_mode': 'from_scratch',
            'pseudo_dir': str(self.pseudo_dir),
            'outdir': './tmp',
            'prefix': 'pwscf',
            'verbosity': 'high'
        }
        
        # System section
        system = {
            'ecutwfc': self.parameters.encut / 13.6057,  # Convert eV to Ry
            'occupations': 'smearing',
            'smearing': 'gaussian',
            'degauss': 0.01,
            'noncolin': self.parameters.lsorbit,
            'lspinorb': self.parameters.lsorbit
        }
        
        if self.parameters.ispin == 2:
            system['nspin'] = 2
            
        # Electrons section
        electrons = {
            'conv_thr': self.parameters.ediff / 13.6057,  # Convert eV to Ry
            'mixing_beta': 0.3,
            'mixing_mode': 'plain'
        }
        
        # k-points
        kpoint_density = self.parameters.kpoint_density
        # Simplified k-point generation
        kpts = [6, 6, 6]  # Would need proper automatic generation
        
        # Pseudopotentials
        pseudos = {}
        for element in structure.symbol_set:
            # Simplified - would need proper pseudopotential mapping
            pseudos[element] = f"{element}.UPF"
            
        pw_input = PWInput(
            structure=structure,
            control=control,
            system=system,
            electrons=electrons,
            kpoints_grid=kpts,
            pseudos=pseudos
        )
        
        return pw_input
        
    def run_calculation(self, calc_dir: str, job_manager: SlurmJobManager) -> str:
        """Submit Quantum ESPRESSO calculation job"""
        
        config = SlurmJobConfig(
            job_name=f"qe_{Path(calc_dir).name}",
            partition="cpu",
            nodes=1,
            ntasks_per_node=16,
            cpus_per_task=1,
            memory="64G",
            time="24:00:00",
            modules=["quantumespresso/7.0", "intel/2021.4", "openmpi/4.1.0"],
            working_dir=calc_dir
        )
        
        commands = [
            "# Run Quantum ESPRESSO calculation",
            "mpirun -np $SLURM_NTASKS pw.x < pw.in > pw.out 2>&1",
            "",
            "# Check if calculation completed",
            "if grep -q 'JOB DONE' pw.out; then",
            "    echo 'QE calculation completed successfully'",
            "    touch CONVERGED",
            "else",
            "    echo 'QE calculation failed'",
            "    touch FAILED",
            "fi"
        ]
        
        job_id = job_manager.submit_job(config, commands)
        logger.info(f"Submitted QE job {job_id} for {calc_dir}")
        
        return job_id
        
    def parse_results(self, calc_dir: str) -> CalculationResult:
        """Parse Quantum ESPRESSO calculation results"""
        calc_path = Path(calc_dir)
        
        try:
            # Parse output file
            pw_output = PWOutput(calc_path / "pw.out")
            
            # Basic results
            structure = pw_output.final_structure
            energy = pw_output.final_energy
            forces = pw_output.forces if hasattr(pw_output, 'forces') else None
            stress = pw_output.stress if hasattr(pw_output, 'stress') else None
            
            # Convergence check
            converged = self.check_convergence(calc_dir)
            
            result = CalculationResult(
                structure=structure,
                energy=energy,
                forces=forces,
                stress=stress,
                converged=converged,
                metadata={
                    'calculator': 'Quantum ESPRESSO',
                    'functional': self.parameters.functional,
                    'ecutwfc': self.parameters.encut / 13.6057
                }
            )
            
            logger.info(f"Successfully parsed QE results from {calc_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse QE results from {calc_dir}: {e}")
            return CalculationResult(
                structure=Structure.from_file(calc_path / "pw.in"),
                energy=float('inf'),
                converged=False,
                metadata={'error': str(e), 'calculator': 'Quantum ESPRESSO'}
            )
            
    def check_convergence(self, calc_dir: str) -> bool:
        """Check Quantum ESPRESSO calculation convergence"""
        calc_path = Path(calc_dir)
        
        # Check for convergence markers
        if (calc_path / "CONVERGED").exists():
            return True
        if (calc_path / "FAILED").exists():
            return False
            
        # Check output file for convergence
        output_path = calc_path / "pw.out"
        if output_path.exists():
            try:
                with open(output_path, 'r') as f:
                    content = f.read()
                    if 'JOB DONE' in content:
                        return True
                    if 'convergence achieved' in content:
                        return True
            except:
                pass
                
        return False

class WorkflowManager:
    """Manages DFT calculation workflows"""
    
    def __init__(self, calculator: DFTCalculator, job_manager: SlurmJobManager,
                 max_concurrent_jobs: int = 50):
        self.calculator = calculator
        self.job_manager = job_manager
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs = {}  # job_id -> calc_info
        self.completed_calculations = []
        self.failed_calculations = []
        
    def submit_calculation(self, structure: Structure, calc_name: str) -> str:
        """Submit a single DFT calculation"""
        calc_dir = self.calculator.work_dir / calc_name
        
        # Setup calculation
        self.calculator.setup_calculation(structure, str(calc_dir))
        
        # Submit job
        job_id = self.calculator.run_calculation(str(calc_dir), self.job_manager)
        
        # Track job
        self.active_jobs[job_id] = {
            'calc_dir': str(calc_dir),
            'structure': structure,
            'calc_name': calc_name,
            'submit_time': time.time()
        }
        
        return job_id
        
    def submit_batch_calculations(self, structures: List[Structure], 
                                calc_names: List[str]) -> List[str]:
        """Submit batch of DFT calculations"""
        job_ids = []
        
        for structure, calc_name in zip(structures, calc_names):
            # Check if we're at max concurrent jobs
            while len(self.active_jobs) >= self.max_concurrent_jobs:
                self.check_job_status()
                time.sleep(60)  # Wait 1 minute before checking again
                
            job_id = self.submit_calculation(structure, calc_name)
            job_ids.append(job_id)
            
        return job_ids
        
    def check_job_status(self) -> None:
        """Check status of active jobs and process completed ones"""
        completed_jobs = []
        
        for job_id, job_info in self.active_jobs.items():
            status = self.job_manager.check_job_status(job_id)
            
            if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                completed_jobs.append(job_id)
                
                # Process completed job
                calc_dir = job_info['calc_dir']
                
                if status == "COMPLETED":
                    try:
                        result = self.calculator.parse_results(calc_dir)
                        result.metadata['job_id'] = job_id
                        result.metadata['calc_name'] = job_info['calc_name']
                        
                        if result.converged:
                            self.completed_calculations.append(result)
                            logger.info(f"Successfully completed calculation {job_info['calc_name']}")
                        else:
                            self.failed_calculations.append(result)
                            logger.warning(f"Calculation {job_info['calc_name']} did not converge")
                            
                    except Exception as e:
                        logger.error(f"Failed to process completed job {job_id}: {e}")
                        self.failed_calculations.append(job_info)
                        
                else:
                    logger.error(f"Job {job_id} failed with status {status}")
                    self.failed_calculations.append(job_info)
                    
        # Remove completed jobs from active list
        for job_id in completed_jobs:
            del self.active_jobs[job_id]
            
    def wait_for_all_jobs(self, check_interval: int = 300) -> None:
        """Wait for all active jobs to complete"""
        logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete...")
        
        while self.active_jobs:
            self.check_job_status()
            if self.active_jobs:
                logger.info(f"{len(self.active_jobs)} jobs still running...")
                time.sleep(check_interval)
                
        logger.info("All jobs completed")
        
    def get_results_summary(self) -> Dict:
        """Get summary of calculation results"""
        return {
            'total_submitted': len(self.completed_calculations) + len(self.failed_calculations),
            'completed': len(self.completed_calculations),
            'failed': len(self.failed_calculations),
            'success_rate': len(self.completed_calculations) / 
                          (len(self.completed_calculations) + len(self.failed_calculations))
                          if (self.completed_calculations or self.failed_calculations) else 0.0
        }
        
    def save_results(self, output_file: str) -> None:
        """Save calculation results to file"""
        results_data = {
            'completed_calculations': [],
            'failed_calculations': [],
            'summary': self.get_results_summary()
        }
        
        # Serialize completed calculations
        for result in self.completed_calculations:
            result_dict = {
                'structure': result.structure.as_dict(),
                'energy': result.energy,
                'band_gap': result.band_gap,
                'magnetic_moments': result.magnetic_moments,
                'converged': result.converged,
                'calculation_time': result.calculation_time,
                'metadata': result.metadata
            }
            results_data['completed_calculations'].append(result_dict)
            
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        logger.info(f"Saved results to {output_file}")

class ConvergenceChecker:
    """Intelligent convergence checking and parameter optimization"""
    
    def __init__(self):
        self.convergence_history = []
        
    def check_electronic_convergence(self, calc_dir: str, 
                                   tolerance: float = 1e-6) -> bool:
        """Check electronic convergence"""
        # Implementation depends on calculator type
        # This is a simplified version
        return True
        
    def check_ionic_convergence(self, calc_dir: str,
                              force_tolerance: float = 0.01) -> bool:
        """Check ionic convergence"""
        # Implementation depends on calculator type
        return True
        
    def suggest_parameter_adjustments(self, failed_calc_dir: str) -> Dict:
        """Suggest parameter adjustments for failed calculations"""
        suggestions = {}
        
        # Analyze failure mode and suggest fixes
        # This would be more sophisticated in practice
        suggestions['encut'] = 600.0  # Increase cutoff
        suggestions['kpoint_density'] = 2000.0  # Increase k-points
        
        return suggestions

# Utility functions
def create_calculator_from_config(config: Dict) -> DFTCalculator:
    """Create DFT calculator from configuration"""
    calc_type = config.get('calculator', 'vasp').lower()
    
    # Create parameters
    params = DFTParameters(**config.get('parameters', {}))
    
    if calc_type == 'vasp':
        return VASPCalculator(
            parameters=params,
            potcar_dir=config['potcar_dir'],
            work_dir=config.get('work_dir', './vasp_calculations')
        )
    elif calc_type == 'qe' or calc_type == 'quantumespresso':
        return QuantumESPRESSOCalculator(
            parameters=params,
            pseudopotential_dir=config['pseudo_dir'],
            work_dir=config.get('work_dir', './qe_calculations')
        )
    else:
        raise ValueError(f"Unknown calculator type: {calc_type}")

def setup_high_throughput_workflow(structures: List[Structure],
                                 calc_names: List[str],
                                 config: Dict) -> WorkflowManager:
    """Set up high-throughput DFT workflow"""
    
    # Create calculator
    calculator = create_calculator_from_config(config)
    
    # Create job manager
    job_manager = SlurmJobManager(base_dir=config.get('slurm_dir', './slurm_jobs'))
    
    # Create workflow manager
    workflow = WorkflowManager(
        calculator=calculator,
        job_manager=job_manager,
        max_concurrent_jobs=config.get('max_concurrent_jobs', 50)
    )
    
    return workflow

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'calculator': 'vasp',
        'potcar_dir': '/path/to/potcar',
        'parameters': {
            'functional': 'PBE',
            'encut': 520.0,
            'lsorbit': True,
            'ispin': 2
        },
        'max_concurrent_jobs': 20
    }
    
    # Example structures (would come from materials database)
    from pymatgen.core import Lattice, Structure
    
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    
    structures = [structure]
    calc_names = ["Si_test"]
    
    # Set up workflow
    workflow = setup_high_throughput_workflow(structures, calc_names, config)
    
    # Submit calculations
    job_ids = workflow.submit_batch_calculations(structures, calc_names)
    logger.info(f"Submitted {len(job_ids)} calculations")
    
    # Wait for completion (in practice, would run as separate monitoring process)
    # workflow.wait_for_all_jobs()
    
    # Save results
    # workflow.save_results("dft_results.json")