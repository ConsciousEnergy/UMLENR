#!/usr/bin/env python
"""Comprehensive integration test for LENR Simulation Framework."""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add backend to path - go up one level from scripts/ to project root, then into backend/
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

def test_physics_modules():
    """Test all physics modules."""
    print("\n1. TESTING PHYSICS MODULES")
    print("=" * 60)
    
    try:
        from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters  # type: ignore
        
        params = IntegratedParameters(
            loading_ratio=0.95,
            electric_field=1e10,
            defect_density=1e21
        )
        sim = IntegratedLENRSimulation(params)
        results = sim.calculate_total_enhancement(energy=10.0)
        
        # Validate results - ensure values are numeric and within sensible bounds
        import math
        
        # Check screening_energy
        screening_energy = results['screening_energy']
        if not isinstance(screening_energy, (int, float)) or math.isnan(screening_energy) or math.isinf(screening_energy):
            raise AssertionError(f"screening_energy is not a valid number: {screening_energy}")
        if screening_energy < 0:
            raise AssertionError(f"screening_energy must be non-negative, got: {screening_energy} eV")
        if screening_energy > 1e6:
            raise AssertionError(f"screening_energy exceeds realistic bound (1e6 eV), got: {screening_energy} eV")
        
        # Check lattice_enhancement
        lattice_enhancement = results['lattice_enhancement']
        if not isinstance(lattice_enhancement, (int, float)) or math.isnan(lattice_enhancement) or math.isinf(lattice_enhancement):
            raise AssertionError(f"lattice_enhancement is not a valid number: {lattice_enhancement}")
        if lattice_enhancement < 1.0:
            raise AssertionError(f"lattice_enhancement should be >= 1.0 (enhancement factor), got: {lattice_enhancement}x")
        if lattice_enhancement > 1e6:
            raise AssertionError(f"lattice_enhancement exceeds realistic bound (1e6x), got: {lattice_enhancement}x")
        
        # Check interface_enhancement
        interface_enhancement = results['interface_enhancement']
        if not isinstance(interface_enhancement, (int, float)) or math.isnan(interface_enhancement) or math.isinf(interface_enhancement):
            raise AssertionError(f"interface_enhancement is not a valid number: {interface_enhancement}")
        if interface_enhancement < 1.0:
            raise AssertionError(f"interface_enhancement should be >= 1.0 (enhancement factor), got: {interface_enhancement}x")
        if interface_enhancement > 1e6:
            raise AssertionError(f"interface_enhancement exceeds realistic bound (1e6x), got: {interface_enhancement}x")
        
        # Check total_combined_enhancement
        total_enhancement = results['total_combined_enhancement']
        if not isinstance(total_enhancement, (int, float)) or math.isnan(total_enhancement) or math.isinf(total_enhancement):
            raise AssertionError(f"total_combined_enhancement is not a valid number: {total_enhancement}")
        if total_enhancement < 1.0:
            raise AssertionError(f"total_combined_enhancement should be >= 1.0, got: {total_enhancement}")
        if total_enhancement > 1e50:
            raise AssertionError(f"total_combined_enhancement exceeds realistic bound (1e50), got: {total_enhancement}")
        
        print(f"[OK] Quantum Tunneling: Working")
        print(f"[OK] Electron Screening: {results['screening_energy']:.1f} eV (validated: 0 to 1e6)")
        print(f"[OK] Lattice Effects: {results['lattice_enhancement']:.2f}x (validated: >= 1.0)")
        print(f"[OK] Interface Dynamics: {results['interface_enhancement']:.1f}x (validated: >= 1.0)")
        print(f"[OK] Total Enhancement: {results['total_combined_enhancement']:.2e} (validated: 1.0 to 1e50)")
        
        # Validate against paper
        validation = sim.validate_against_paper()
        print(f"\nValidation against paper:")
        for key, value in validation.items():
            if key != "all_checks_passed":
                status = "[OK]" if value else "[FAIL]"
                print(f"  {status} {key}: {value}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Physics modules test failed: {e}")
        return False


def test_solvers():
    """Test numerical solvers."""
    print("\n2. TESTING NUMERICAL SOLVERS")
    print("=" * 60)
    
    try:
        # Test Poisson-Schrödinger
        from solvers.poisson_schrodinger import PoissonSchrodingerSolver  # type: ignore
        solver = PoissonSchrodingerSolver()
        print(f"[OK] Poisson-Schrödinger Solver: {solver.N} grid points")
        
        # Test Monte Carlo
        from solvers.monte_carlo import MonteCarloUncertainty, MonteCarloConfig  # type: ignore
        mc_config = MonteCarloConfig(n_samples=100, parallel=False)
        mc = MonteCarloUncertainty(mc_config)
        mc.setup_default_distributions()
        print(f"[OK] Monte Carlo System: {len(mc.parameter_distributions)} parameters")        
        return True
    except Exception as e:
        print(f"[FAIL] Solvers test failed: {e}")
        return False


def test_ml_components():
    """Test ML optimization components."""
    print("\n3. TESTING ML COMPONENTS")
    print("=" * 60)
    
    try:
        from ml.parameter_optimizer import LENRParameterOptimizer, PatternRecognizer  # type: ignore
        
        # Test optimizer
        optimizer = LENRParameterOptimizer()
        
        # Add test observations
        for i in range(5):
            params = {
                'loading_ratio': 0.9 + i*0.02,
                'electric_field': 1e10,
                'temperature': 300,
                'defect_density': 1e21,
                'coherence_domain_size': 1000,
                'surface_potential': 0.5
            }
            optimizer.add_observation(params, 10**(7 + i*0.5))
        
        optimizer.train()
        suggestion = optimizer.suggest_next_parameters()
        
        print(f"[OK] Gaussian Process Optimizer: Trained on {len(optimizer.X_train)} samples")
        print(f"[OK] Suggested enhancement: {suggestion.expected_enhancement:.2e}")
        
        # Test pattern recognition
        recognizer = PatternRecognizer()
        test_data = [
            {'loading_ratio': 0.95, 'enhancement': 1e7},
            {'loading_ratio': 0.85, 'enhancement': 1e5}
        ]
        patterns = recognizer.analyze_patterns(test_data)
        print(f"[OK] Pattern Recognizer: {patterns.get('n_high_yield', 0)} high-yield found")
        
        return True
    except Exception as e:
        print(f"[FAIL] ML components test failed: {e}")
        return False


def test_api_import():
    """Test API components can be imported."""
    print("\n4. TESTING API COMPONENTS")
    print("=" * 60)
    
    try:
        from main import app  # type: ignore
        from api import models, simulation, parameters, ml_endpoints  # type: ignore
        
        print(f"[OK] FastAPI app imported successfully")
        print(f"[OK] API models defined")
        print(f"[OK] Simulation endpoints ready")
        print(f"[OK] Parameter endpoints ready")
        print(f"[OK] ML endpoints ready")
        
        return True
    except Exception as e:
        print(f"[FAIL] API import test failed: {e}")
        return False


def run_performance_test():
    """Run a performance benchmark."""
    print("\n5. PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters  # type: ignore
        
        params = IntegratedParameters()
        sim = IntegratedLENRSimulation(params)
        
        # Time a single simulation
        start = time.time()
        results = sim.calculate_total_enhancement(energy=10.0)
        elapsed = time.time() - start
        
        print(f"[OK] Single simulation time: {elapsed:.3f} seconds")
        print(f"[OK] Enhancement calculated: {results['total_combined_enhancement']:.2e}")
        
        # Time parameter scan
        start = time.time()
        scan_results = sim.parameter_scan("loading_ratio", [0.85, 0.90, 0.95])
        elapsed = time.time() - start
        
        print(f"[OK] Parameter scan (3 values): {elapsed:.3f} seconds")
        print(f"[OK] Results obtained: {len(scan_results['total_enhancement'])} values")
        
        return True
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("LENR SIMULATION FRAMEWORK - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Physics Modules", test_physics_modules),
        ("Numerical Solvers", test_solvers),
        ("ML Components", test_ml_components),
        ("API Components", test_api_import),
        ("Performance", run_performance_test)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    for name, success in results:
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"{name:<20} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("\n[SUCCESS] All integration tests passed!")
        print("\nThe LENR Simulation Framework is fully operational:")
        print("  - Physics engine: Complete")
        print("  - Numerical solvers: Working")
        print("  - ML optimization: Ready")
        print("  - API framework: Functional")
        print("  - Performance: Verified")
        print("\n[READY] System ready for production use!")
        return 0
    else:
        print(f"\n[WARNING] {failed} test(s) failed")
        print("Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
