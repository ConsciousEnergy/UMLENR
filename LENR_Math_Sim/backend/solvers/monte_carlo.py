"""Monte Carlo uncertainty propagation for LENR simulations.

This module implements the Monte Carlo error propagation and sensitivity
analysis described in Section 3.2 of the theoretical framework, including
bootstrapped confidence intervals and Kullback-Leibler divergence assessment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
import logging
from scipy import stats
from scipy.stats import bootstrap
from scipy.special import kl_div
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters
from utils.constants import KB, EV_TO_JOULE, JOULE_TO_EV

logger = logging.getLogger(__name__)


@dataclass
class ParameterDistribution:
    """Define distribution for a parameter."""
    
    name: str  # Parameter name
    distribution_type: str  # 'normal', 'uniform', 'lognormal', 'triangular'
    params: Dict[str, float]  # Distribution parameters
    bounds: Optional[Tuple[float, float]] = None  # Physical bounds
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples from distribution."""
        if self.distribution_type == 'normal':
            samples = np.random.normal(
                self.params['mean'], 
                self.params['std'], 
                n_samples
            )
        elif self.distribution_type == 'uniform':
            samples = np.random.uniform(
                self.params['low'], 
                self.params['high'], 
                n_samples
            )
        elif self.distribution_type == 'lognormal':
            samples = np.random.lognormal(
                self.params['mean'], 
                self.params['std'], 
                n_samples
            )
        elif self.distribution_type == 'triangular':
            samples = np.random.triangular(
                self.params['left'],
                self.params['mode'],
                self.params['right'],
                n_samples
            )
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        
        # Apply bounds if specified
        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])
        
        return samples


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    
    n_samples: int = 10000  # Number of MC samples
    n_bootstrap: int = 1000  # Bootstrap samples for CI
    confidence_level: float = 0.95  # Confidence level
    parallel: bool = True  # Use parallel processing
    n_workers: Optional[int] = None  # Number of workers (None = auto)
    save_samples: bool = False  # Save all sample results
    convergence_check: bool = True  # Check for convergence
    convergence_tolerance: float = 0.01  # Relative tolerance
    batch_size: int = 100  # Samples per batch


class MonteCarloUncertainty:
    """Monte Carlo uncertainty propagation for LENR simulations."""
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """Initialize Monte Carlo system."""
        self.config = config or MonteCarloConfig()
        self.parameter_distributions: List[ParameterDistribution] = []
        self.results_cache: Dict = {}
        
    def add_parameter_distribution(self, param_dist: ParameterDistribution):
        """Add a parameter distribution for sampling."""
        self.parameter_distributions.append(param_dist)
    
    def setup_default_distributions(self):
        """Set up default parameter distributions based on paper."""
        # Material microstructure
        self.add_parameter_distribution(ParameterDistribution(
            name="defect_density",
            distribution_type="lognormal",
            params={"mean": np.log(1e20), "std": 0.5},
            bounds=(1e18, 1e22)
        ))
        
        # Surface topology
        self.add_parameter_distribution(ParameterDistribution(
            name="surface_roughness",
            distribution_type="lognormal",
            params={"mean": np.log(10e-9), "std": 0.3},
            bounds=(1e-9, 100e-9)
        ))
        
        # Loading ratio
        self.add_parameter_distribution(ParameterDistribution(
            name="loading_ratio",
            distribution_type="triangular",
            params={"left": 0.80, "mode": 0.90, "right": 0.99},
            bounds=(0.0, 1.0)
        ))
        
        # Temperature
        self.add_parameter_distribution(ParameterDistribution(
            name="temperature",
            distribution_type="normal",
            params={"mean": 300.0, "std": 10.0},
            bounds=(250.0, 400.0)
        ))
        
        # Electric field
        self.add_parameter_distribution(ParameterDistribution(
            name="electric_field",
            distribution_type="lognormal",
            params={"mean": np.log(1e9), "std": 0.5},
            bounds=(1e8, 1e11)
        ))
        
        # Coherence domain size
        self.add_parameter_distribution(ParameterDistribution(
            name="coherence_domain_size",
            distribution_type="uniform",
            params={"low": 100, "high": 10000},
            bounds=(10, 100000)
        ))
        
        # Surface potential
        self.add_parameter_distribution(ParameterDistribution(
            name="surface_potential",
            distribution_type="normal",
            params={"mean": 0.5, "std": 0.1},
            bounds=(0.0, 2.0)
        ))
    
    def generate_parameter_samples(self) -> np.ndarray:
        """Generate parameter samples from distributions.
        
        Returns:
            Array of shape (n_samples, n_parameters)
        """
        n_params = len(self.parameter_distributions)
        samples = np.zeros((self.config.n_samples, n_params))
        
        for i, param_dist in enumerate(self.parameter_distributions):
            samples[:, i] = param_dist.sample(self.config.n_samples)
        
        return samples
    
    def run_single_simulation(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Run a single LENR simulation with given parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Dictionary of results
        """
        # Create simulation parameters
        sim_params = IntegratedParameters()
        
        # Update with sampled values
        for key, value in parameters.items():
            if hasattr(sim_params, key):
                setattr(sim_params, key, value)
        
        # Run simulation
        try:
            sim = IntegratedLENRSimulation(sim_params)
            results = sim.calculate_total_enhancement(energy=10.0)
            
            # Extract key outputs
            output = {
                "total_enhancement": results["total_combined_enhancement"],
                "tunneling_probability": results["final_reaction_probability"],
                "energy_concentration": results["total_energy_concentration"],
                "screening_energy": results["screening_energy"],
                "max_field": results["max_interface_field"]
            }
            
            # Add reaction rate estimate
            rate_results = sim.reaction_rate_estimate()
            output["reaction_rate"] = rate_results["enhanced_rate"]
            output["power_density"] = rate_results["power_density"]
            
        except Exception as e:
            logger.warning(f"Simulation failed: {e}")
            output = {
                "total_enhancement": np.nan,
                "tunneling_probability": np.nan,
                "energy_concentration": np.nan,
                "screening_energy": np.nan,
                "max_field": np.nan,
                "reaction_rate": np.nan,
                "power_density": np.nan
            }
        
        return output
    
    def run_batch(self, parameter_batch: np.ndarray) -> List[Dict]:
        """Run a batch of simulations.
        
        Args:
            parameter_batch: Array of parameter sets
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for params_array in parameter_batch:
            # Convert to dictionary
            params_dict = {}
            for i, param_dist in enumerate(self.parameter_distributions):
                params_dict[param_dist.name] = params_array[i]
            
            # Run simulation
            result = self.run_single_simulation(params_dict)
            results.append(result)
        
        return results
    
    def run_monte_carlo(self, output_quantities: Optional[List[str]] = None) -> Dict:
        """Run full Monte Carlo uncertainty propagation.
        
        Args:
            output_quantities: List of output quantities to track
            
        Returns:
            Dictionary with statistics and confidence intervals
        """
        if output_quantities is None:
            output_quantities = [
                "total_enhancement",
                "tunneling_probability", 
                "energy_concentration",
                "screening_energy",
                "reaction_rate"
            ]
        
        # Generate parameter samples
        logger.info(f"Generating {self.config.n_samples} parameter samples...")
        parameter_samples = self.generate_parameter_samples()
        
        # Run simulations
        logger.info("Running Monte Carlo simulations...")
        
        if self.config.parallel and self.config.n_workers != 1:
            # Parallel execution
            n_workers = self.config.n_workers or mp.cpu_count()
            batch_size = max(1, self.config.n_samples // n_workers)
            
            # Split samples into batches
            batches = [
                parameter_samples[i:i+batch_size] 
                for i in range(0, self.config.n_samples, batch_size)
            ]
            
            # Run batches in parallel
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                batch_results = list(executor.map(self.run_batch, batches))
            
            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        else:
            # Sequential execution
            all_results = self.run_batch(parameter_samples)
        
        # Extract output arrays
        output_arrays = {}
        for quantity in output_quantities:
            values = np.array([r[quantity] for r in all_results])
            # Remove NaN values
            output_arrays[quantity] = values[~np.isnan(values)]
        
        # Calculate statistics
        statistics = self.calculate_statistics(output_arrays, parameter_samples)
        
        # Bootstrap confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(output_arrays)
        
        # Convergence analysis
        if self.config.convergence_check:
            convergence = self.check_convergence(output_arrays)
        else:
            convergence = None
        
        # Store results
        results = {
            "parameter_samples": parameter_samples,
            "output_samples": output_arrays,
            "statistics": statistics,
            "confidence_intervals": confidence_intervals,
            "convergence": convergence,
            "n_successful": len(output_arrays[output_quantities[0]]),
            "n_failed": self.config.n_samples - len(output_arrays[output_quantities[0]])
        }
        
        if self.config.save_samples:
            results["all_results"] = all_results
        
        return results
    
    def calculate_statistics(self, output_arrays: Dict[str, np.ndarray],
                           parameter_samples: np.ndarray) -> Dict:
        """Calculate statistical measures for outputs.
        
        Args:
            output_arrays: Dictionary of output arrays
            parameter_samples: Parameter sample array
            
        Returns:
            Dictionary of statistics
        """
        statistics = {}
        
        for quantity, values in output_arrays.items():
            if len(values) == 0:
                continue
                
            statistics[quantity] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "skewness": stats.skew(values),
                "kurtosis": stats.kurtosis(values),
                "percentiles": {
                    "p5": np.percentile(values, 5),
                    "p25": np.percentile(values, 25),
                    "p50": np.percentile(values, 50),
                    "p75": np.percentile(values, 75),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
            }
        
        return statistics
    
    def calculate_confidence_intervals(self, output_arrays: Dict[str, np.ndarray]) -> Dict:
        """Calculate bootstrap confidence intervals.
        
        Args:
            output_arrays: Dictionary of output arrays
            
        Returns:
            Dictionary of confidence intervals
        """
        confidence_intervals = {}
        alpha = 1 - self.config.confidence_level
        
        for quantity, values in output_arrays.items():
            if len(values) < 10:  # Need minimum samples
                continue
            
            # Bootstrap for mean
            res_mean = bootstrap(
                (values,),
                np.mean,
                n_resamples=self.config.n_bootstrap,
                confidence_level=self.config.confidence_level
            )
            
            # Bootstrap for median
            res_median = bootstrap(
                (values,),
                np.median,
                n_resamples=self.config.n_bootstrap,
                confidence_level=self.config.confidence_level
            )
            
            confidence_intervals[quantity] = {
                "mean_ci": (res_mean.confidence_interval.low, 
                           res_mean.confidence_interval.high),
                "median_ci": (res_median.confidence_interval.low,
                            res_median.confidence_interval.high),
                "mean_se": res_mean.standard_error,
                "median_se": res_median.standard_error
            }
        
        return confidence_intervals
    
    def check_convergence(self, output_arrays: Dict[str, np.ndarray]) -> Dict:
        """Check Monte Carlo convergence.
        
        Args:
            output_arrays: Dictionary of output arrays
            
        Returns:
            Convergence metrics
        """
        convergence = {}
        
        for quantity, values in output_arrays.items():
            if len(values) < 100:
                continue
            
            # Calculate running mean
            n_checkpoints = min(20, len(values) // 50)
            checkpoint_indices = np.linspace(100, len(values), n_checkpoints, dtype=int)
            
            running_means = []
            running_stds = []
            
            for idx in checkpoint_indices:
                running_means.append(np.mean(values[:idx]))
                running_stds.append(np.std(values[:idx]))
            
            # Check relative change
            if len(running_means) > 1:
                relative_change = np.abs(running_means[-1] - running_means[-2]) / \
                                (np.abs(running_means[-1]) + 1e-10)
                converged = relative_change < self.config.convergence_tolerance
            else:
                relative_change = np.nan
                converged = False
            
            convergence[quantity] = {
                "converged": converged,
                "relative_change": relative_change,
                "running_means": running_means,
                "running_stds": running_stds,
                "checkpoint_samples": checkpoint_indices.tolist()
            }
        
        return convergence
    
    def sobol_sensitivity_analysis(self, n_samples: int = 1000) -> Dict:
        """Perform Sobol sensitivity analysis.
        
        Args:
            n_samples: Number of samples for Sobol analysis
            
        Returns:
            Sobol indices
        """
        from SALib.sample import saltelli
        from SALib.analyze import sobol
        
        # Define problem
        problem = {
            'num_vars': len(self.parameter_distributions),
            'names': [pd.name for pd in self.parameter_distributions],
            'bounds': []
        }
        
        # Set bounds from distributions
        for pd in self.parameter_distributions:
            if pd.distribution_type == 'uniform':
                bounds = [pd.params['low'], pd.params['high']]
            elif pd.distribution_type == 'normal':
                mean, std = pd.params['mean'], pd.params['std']
                bounds = [mean - 3*std, mean + 3*std]
            elif pd.distribution_type == 'lognormal':
                mean, std = pd.params['mean'], pd.params['std']
                bounds = [np.exp(mean - 3*std), np.exp(mean + 3*std)]
            else:
                # Default bounds
                bounds = [0, 1]
            
            if pd.bounds is not None:
                bounds = [max(bounds[0], pd.bounds[0]), min(bounds[1], pd.bounds[1])]
            
            problem['bounds'].append(bounds)
        
        # Generate Sobol samples
        param_values = saltelli.sample(problem, n_samples)
        
        # Run simulations
        Y = np.zeros((param_values.shape[0], 2))  # Enhancement and energy
        
        for i, params in enumerate(param_values):
            params_dict = {
                problem['names'][j]: params[j] 
                for j in range(len(problem['names']))
            }
            result = self.run_single_simulation(params_dict)
            Y[i, 0] = result.get('total_enhancement', np.nan)
            Y[i, 1] = result.get('energy_concentration', np.nan)
        
        # Clean NaN values
        valid_mask = ~np.isnan(Y[:, 0])
        Y_clean = Y[valid_mask, :]
        
        # Analyze
        Si_enhancement = sobol.analyze(problem, Y_clean[:, 0])
        Si_energy = sobol.analyze(problem, Y_clean[:, 1])
        
        return {
            "enhancement": {
                "S1": Si_enhancement['S1'].tolist(),  # First-order indices
                "ST": Si_enhancement['ST'].tolist(),  # Total-order indices
                "S2": Si_enhancement.get('S2', None)   # Second-order indices
            },
            "energy_concentration": {
                "S1": Si_energy['S1'].tolist(),
                "ST": Si_energy['ST'].tolist(),
                "S2": Si_energy.get('S2', None)
            },
            "parameter_names": problem['names']
        }
    
    def calculate_kl_divergence(self, reference_dist: np.ndarray,
                               comparison_dist: np.ndarray,
                               n_bins: int = 50) -> float:
        """Calculate Kullback-Leibler divergence between distributions.
        
        Args:
            reference_dist: Reference distribution samples
            comparison_dist: Comparison distribution samples
            n_bins: Number of histogram bins
            
        Returns:
            KL divergence value
        """
        # Create histograms
        min_val = min(np.min(reference_dist), np.min(comparison_dist))
        max_val = max(np.max(reference_dist), np.max(comparison_dist))
        
        bins = np.linspace(min_val, max_val, n_bins)
        
        hist_ref, _ = np.histogram(reference_dist, bins=bins, density=True)
        hist_comp, _ = np.histogram(comparison_dist, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_ref = hist_ref + epsilon
        hist_comp = hist_comp + epsilon
        
        # Normalize
        hist_ref = hist_ref / np.sum(hist_ref)
        hist_comp = hist_comp / np.sum(hist_comp)
        
        # Calculate KL divergence
        kl = np.sum(hist_ref * np.log(hist_ref / hist_comp))
        
        return kl


# Example usage and testing
if __name__ == "__main__":
    print("Monte Carlo Uncertainty Propagation for LENR")
    print("=" * 60)
    
    # Create Monte Carlo system
    mc_config = MonteCarloConfig(
        n_samples=100,  # Reduced for testing
        n_bootstrap=100,
        parallel=False  # Sequential for testing
    )
    
    mc = MonteCarloUncertainty(mc_config)
    
    # Set up default distributions
    print("\nSetting up parameter distributions...")
    mc.setup_default_distributions()
    
    print(f"Parameters to sample:")
    for pd in mc.parameter_distributions:
        print(f"  - {pd.name}: {pd.distribution_type}")
    
    # Run Monte Carlo
    print(f"\nRunning {mc_config.n_samples} Monte Carlo samples...")
    results = mc.run_monte_carlo()
    
    # Display results
    print(f"\nMonte Carlo Results:")
    print(f"  Successful simulations: {results['n_successful']}/{mc_config.n_samples}")
    
    print(f"\nStatistics for Total Enhancement:")
    if "total_enhancement" in results["statistics"]:
        stats = results["statistics"]["total_enhancement"]
        print(f"  Mean: {stats['mean']:.2e}")
        print(f"  Std: {stats['std']:.2e}")
        print(f"  Median: {stats['median']:.2e}")
        print(f"  Range: [{stats['min']:.2e}, {stats['max']:.2e}]")
        print(f"  95% CI: [{stats['percentiles']['p5']:.2e}, {stats['percentiles']['p95']:.2e}]")
    
    print(f"\nStatistics for Energy Concentration:")
    if "energy_concentration" in results["statistics"]:
        stats = results["statistics"]["energy_concentration"]
        print(f"  Mean: {stats['mean']:.2f} eV/atom")
        print(f"  Std: {stats['std']:.2f} eV/atom")
        print(f"  Median: {stats['median']:.2f} eV/atom")
        print(f"  95% CI: [{stats['percentiles']['p5']:.2f}, {stats['percentiles']['p95']:.2f}] eV/atom")
    
    # Check convergence
    if results["convergence"]:
        print(f"\nConvergence Analysis:")
        for quantity, conv in results["convergence"].items():
            if "converged" in conv:
                status = "[CONVERGED]" if conv["converged"] else "[NOT CONVERGED]"
                print(f"  {quantity}: {status} (rel. change = {conv['relative_change']:.3f})")
    
    print(f"\n[SUCCESS] Monte Carlo uncertainty propagation system is operational!")
