"""Machine learning components for LENR parameter optimization.

Implements ML models from Section 3.4 of the paper for parameter discovery
and optimization using Gaussian Process Regression and neural networks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    optimal_params: Dict[str, float]
    expected_enhancement: float
    uncertainty: float
    acquisition_value: float


class LENRParameterOptimizer:
    """ML-based parameter optimizer for LENR simulations."""
    
    def __init__(self):
        """Initialize optimizer with Gaussian Process."""
        # Define parameter bounds
        self.param_bounds = {
            'loading_ratio': (0.80, 0.99),
            'electric_field': (1e8, 1e11),
            'temperature': (250.0, 400.0),
            'defect_density': (1e19, 1e22),
            'coherence_domain_size': (100, 10000),
            'surface_potential': (0.1, 2.0)
        }
        
        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
        # Scaler for normalization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Store training data
        self.X_train = []
        self.y_train = []
        
    def add_observation(self, params: Dict[str, float], enhancement: float):
        """Add a new observation to the training set.
        
        Args:
            params: Parameter values
            enhancement: Observed enhancement factor (log scale)
        """
        # Convert params to array
        X = np.array([params.get(k, 0) for k in sorted(self.param_bounds.keys())])
        y = np.log10(enhancement) if enhancement > 0 else 0
        
        self.X_train.append(X)
        self.y_train.append(y)
        
    def train(self):
        """Train the Gaussian Process on accumulated data."""
        if len(self.X_train) < 2:
            raise ValueError("Need at least 2 observations to train")
        
        X = np.array(self.X_train)
        y = np.array(self.y_train).reshape(-1, 1)
        
        # Normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Train GP
        self.gp.fit(X_scaled, y_scaled.ravel())
        
        logger.info(f"Trained GP on {len(self.X_train)} observations")
        
    def predict(self, params: Dict[str, float]) -> Tuple[float, float]:
        """Predict enhancement for given parameters.
        
        Args:
            params: Parameter values
            
        Returns:
            Tuple of (mean_enhancement, std_enhancement)
        """
        X = np.array([params.get(k, 0) for k in sorted(self.param_bounds.keys())])
        X = X.reshape(1, -1)
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled, std_scaled = self.gp.predict(X_scaled, return_std=True)
        
        # Inverse transform
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
        
        # Convert from log scale
        enhancement = 10**y_pred
        
        # Approximate uncertainty in original scale
        uncertainty = enhancement * std_scaled[0] * np.log(10)
        
        return enhancement, uncertainty
    
    def acquisition_function(self, params: Dict[str, float], 
                           exploration_weight: float = 1.0) -> float:
        """Calculate acquisition function value (Upper Confidence Bound).
        
        Args:
            params: Parameter values
            exploration_weight: Balance exploration vs exploitation
            
        Returns:
            Acquisition value
        """
        mean, std = self.predict(params)
        return mean + exploration_weight * std
    
    def suggest_next_parameters(self, n_candidates: int = 1000,
                              exploration_weight: float = 1.0) -> OptimizationResult:
        """Suggest next parameters to test using Bayesian optimization.
        
        Args:
            n_candidates: Number of random candidates to evaluate
            exploration_weight: UCB exploration parameter
            
        Returns:
            Optimal parameters to test next
        """
        if len(self.X_train) < 2:
            # Random exploration initially
            params = {}
            for param, (low, high) in self.param_bounds.items():
                if 'electric_field' in param or 'defect_density' in param:
                    params[param] = 10**np.random.uniform(np.log10(low), np.log10(high))
                else:
                    params[param] = np.random.uniform(low, high)
            
            return OptimizationResult(
                optimal_params=params,
                expected_enhancement=1e6,  # Prior expectation
                uncertainty=1e6,
                acquisition_value=2e6
            )
        
        # Generate random candidates
        candidates = []
        for _ in range(n_candidates):
            params = {}
            for param, (low, high) in self.param_bounds.items():
                if 'electric_field' in param or 'defect_density' in param:
                    params[param] = 10**np.random.uniform(np.log10(low), np.log10(high))
                else:
                    params[param] = np.random.uniform(low, high)
            candidates.append(params)
        
        # Evaluate acquisition function
        best_params = None
        best_acquisition = -np.inf
        best_mean = 0
        best_std = 0
        
        for params in candidates:
            acq = self.acquisition_function(params, exploration_weight)
            if acq > best_acquisition:
                best_acquisition = acq
                best_params = params
                best_mean, best_std = self.predict(params)
        
        return OptimizationResult(
            optimal_params=best_params,
            expected_enhancement=best_mean,
            uncertainty=best_std,
            acquisition_value=best_acquisition
        )
    
    def optimize(self, n_iterations: int = 20, 
                simulation_func: Optional[Any] = None) -> List[OptimizationResult]:
        """Run Bayesian optimization loop.
        
        Args:
            n_iterations: Number of optimization iterations
            simulation_func: Function to run actual simulations
            
        Returns:
            List of optimization results
        """
        results = []
        
        for i in range(n_iterations):
            logger.info(f"Optimization iteration {i+1}/{n_iterations}")
            
            # Suggest next parameters
            result = self.suggest_next_parameters(
                exploration_weight=2.0 * (1 - i/n_iterations)  # Decrease exploration
            )
            
            # Run simulation if function provided
            if simulation_func:
                actual_enhancement = simulation_func(result.optimal_params)
                self.add_observation(result.optimal_params, actual_enhancement)
                
                # Retrain GP
                if len(self.X_train) >= 2:
                    self.train()
            
            results.append(result)
        
        return results
    
    def get_best_parameters(self) -> Dict[str, float]:
        """Get the best parameters found so far.
        
        Returns:
            Best parameter set
        """
        if not self.y_train:
            return {k: (v[0] + v[1])/2 for k, v in self.param_bounds.items()}
        
        best_idx = np.argmax(self.y_train)
        best_X = self.X_train[best_idx]
        
        params = {}
        for i, key in enumerate(sorted(self.param_bounds.keys())):
            params[key] = best_X[i]
        
        return params
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        model_data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'param_bounds': self.param_bounds,
            'gp_params': self.gp.get_params(),
            'scaler_X_params': {
                'mean': self.scaler_X.mean_.tolist() if hasattr(self.scaler_X, 'mean_') else None,
                'scale': self.scaler_X.scale_.tolist() if hasattr(self.scaler_X, 'scale_') else None
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']
        self.param_bounds = model_data['param_bounds']
        
        if len(self.X_train) >= 2:
            self.train()


class PatternRecognizer:
    """Simple pattern recognition for high-yield conditions."""
    
    def __init__(self):
        """Initialize pattern recognizer."""
        self.patterns = []
        self.threshold_enhancement = 1e6
        
    def analyze_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data for patterns in high-yield conditions.
        
        Args:
            data: List of simulation results with parameters and enhancement
            
        Returns:
            Pattern analysis results
        """
        if not data:
            return {}
        
        # Filter high-yield results
        high_yield = [d for d in data if d.get('enhancement', 0) > self.threshold_enhancement]
        
        if not high_yield:
            return {"message": "No high-yield conditions found"}
        
        # Analyze parameter ranges
        param_stats = {}
        param_keys = ['loading_ratio', 'electric_field', 'temperature', 
                     'defect_density', 'coherence_domain_size']
        
        for key in param_keys:
            values = [d.get(key, 0) for d in high_yield if key in d]
            if values:
                param_stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'optimal_range': (np.percentile(values, 25), np.percentile(values, 75))
                }
        
        # Identify correlations
        correlations = self._calculate_correlations(high_yield)
        
        # Find critical thresholds
        thresholds = self._find_thresholds(data)
        
        return {
            'n_high_yield': len(high_yield),
            'total_samples': len(data),
            'success_rate': len(high_yield) / len(data),
            'parameter_statistics': param_stats,
            'correlations': correlations,
            'critical_thresholds': thresholds
        }
    
    def _calculate_correlations(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate parameter correlations with enhancement."""
        if len(data) < 3:
            return {}
        
        from scipy.stats import pearsonr
        
        correlations = {}
        param_keys = ['loading_ratio', 'electric_field', 'temperature']
        
        enhancements = [np.log10(d.get('enhancement', 1)) for d in data]
        
        for key in param_keys:
            values = [d.get(key, 0) for d in data]
            if len(set(values)) > 1:  # Need variation
                corr, p_value = pearsonr(values, enhancements)
                correlations[key] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return correlations
    
    def _find_thresholds(self, data: List[Dict]) -> Dict[str, float]:
        """Find critical parameter thresholds."""
        thresholds = {}
        
        # Loading ratio threshold
        loading_ratios = [d.get('loading_ratio', 0) for d in data]
        enhancements = [d.get('enhancement', 0) for d in data]
        
        if loading_ratios and enhancements:
            # Find where enhancement jumps
            sorted_data = sorted(zip(loading_ratios, enhancements))
            
            max_jump = 0
            threshold_lr = 0.85
            
            for i in range(1, len(sorted_data)):
                if sorted_data[i][0] > 0:
                    jump = sorted_data[i][1] / sorted_data[i-1][1] if sorted_data[i-1][1] > 0 else 0
                    if jump > max_jump:
                        max_jump = jump
                        threshold_lr = sorted_data[i][0]
            
            thresholds['loading_ratio_critical'] = threshold_lr
        
        return thresholds


# Test the ML components
if __name__ == "__main__":
    print("ML Parameter Optimizer for LENR")
    print("=" * 50)
    
    # Create optimizer
    optimizer = LENRParameterOptimizer()
    
    # Add some synthetic training data
    print("\nAdding synthetic training data...")
    np.random.seed(42)
    
    for _ in range(10):
        params = {
            'loading_ratio': np.random.uniform(0.85, 0.99),
            'electric_field': 10**np.random.uniform(9, 10),
            'temperature': np.random.uniform(280, 320),
            'defect_density': 10**np.random.uniform(20, 21),
            'coherence_domain_size': np.random.uniform(500, 2000),
            'surface_potential': np.random.uniform(0.3, 0.7)
        }
        
        # Synthetic enhancement (higher loading ratio = better)
        enhancement = 10**(6 + 2*params['loading_ratio'] + 
                          0.5*np.log10(params['electric_field']/1e9))
        enhancement *= np.random.uniform(0.5, 2.0)  # Add noise
        
        optimizer.add_observation(params, enhancement)
    
    # Train model
    print("Training Gaussian Process...")
    optimizer.train()
    
    # Suggest next parameters
    print("\nSuggesting optimal parameters...")
    result = optimizer.suggest_next_parameters()
    
    print(f"\nOptimal Parameters:")
    for key, value in result.optimal_params.items():
        if 'field' in key or 'density' in key:
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value:.3f}")
    
    print(f"\nExpected enhancement: {result.expected_enhancement:.2e}")
    print(f"Uncertainty: {result.uncertainty:.2e}")
    print(f"Acquisition value: {result.acquisition_value:.2e}")
    
    # Test pattern recognition
    print("\n" + "=" * 50)
    print("Pattern Recognition")
    print("=" * 50)
    
    recognizer = PatternRecognizer()
    
    # Create test data
    test_data = []
    for i in range(20):
        test_data.append({
            'loading_ratio': np.random.uniform(0.8, 0.99),
            'electric_field': 10**np.random.uniform(8, 11),
            'temperature': np.random.uniform(250, 350),
            'enhancement': 10**np.random.uniform(4, 8)
        })
    
    patterns = recognizer.analyze_patterns(test_data)
    
    print(f"High-yield conditions: {patterns.get('n_high_yield', 0)}/{patterns.get('total_samples', 0)}")
    print(f"Success rate: {patterns.get('success_rate', 0):.1%}")
    
    print("\n[SUCCESS] ML components are operational!")
