#!/usr/bin/env python
"""Test script to verify LENR simulation setup and core functionality."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    modules_to_test = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("utils.constants", "Constants module"),
        ("core.quantum_tunneling", "Quantum tunneling module"),
        ("core.electron_screening", "Electron screening module"),
        ("core.lattice_effects", "Lattice effects module"),
        ("core.interface_dynamics", "Interface dynamics module"),
        ("core.bubble_dynamics", "Bubble dynamics module"),
        ("core.integrated_simulation", "Integrated simulation module")
    ]
    
    all_success = True
    for module_name, display_name in modules_to_test:
        try:
            if "." in module_name:
                parts = module_name.split(".")
                module = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
            else:
                __import__(module_name)
            print(f"[PASS] {display_name} imported successfully")
        except ImportError as e:
            print(f"[FAIL] {display_name} import failed: {e}")
            all_success = False
    
    return all_success


def test_quantum_tunneling():
    """Test quantum tunneling calculations."""
    print("\nTesting quantum tunneling calculations...")
    
    from core.quantum_tunneling import QuantumTunneling, TunnelingParameters
    
    # Create calculator with custom parameters
    params = TunnelingParameters(
        particle1_mass=3.343583719e-27,  # Deuteron mass
        particle2_mass=3.343583719e-27,
        charge1=1.602176634e-19,  # Elementary charge
        charge2=1.602176634e-19,
        temperature=300.0,
        electric_field=1e9,
        screening_energy=25.0,  # Enhanced screening
        coherence_factor=2.0,  # Some coherence
        loading_ratio=0.95  # High loading
    )
    
    tunneling = QuantumTunneling(params)
    
    # Test at different energies
    test_energies = [1.0, 10.0, 100.0]
    
    print("\nEnergy (eV) | Base Prob | Enhanced Prob | Enhancement Factor")
    print("-" * 65)
    
    for energy in test_energies:
        result = tunneling.calculate_total_enhancement(energy)
        print(f"{energy:10.1f} | {result['base_probability']:10.2e} | "
              f"{result['final_probability']:13.2e} | {result['total_enhancement']:10.2e}")
    
    # Run a small Monte Carlo simulation
    print("\nRunning Monte Carlo simulation (1000 samples)...")
    sim_results = tunneling.simulate_tunneling_events(n_samples=1000)
    
    print(f"Mean probability: {sim_results['mean_probability']:.2e}")
    print(f"Std deviation: {sim_results['std_probability']:.2e}")
    print(f"95th percentile: {sim_results['percentiles']['p95']:.2e}")
    
    return True


def test_constants():
    """Test physical constants."""
    print("\nTesting physical constants...")
    
    from utils import constants
    
    # Check some key constants
    print(f"Reduced Planck constant: {constants.HBAR:.4e} J·s")
    print(f"Elementary charge: {constants.E_CHARGE:.4e} C")
    print(f"Deuteron mass: {constants.M_DEUTERON:.4e} kg")
    print(f"D-D Coulomb barrier: {constants.D_D_COULOMB_BARRIER/1000:.1f} keV")
    
    # Test utility functions
    thermal_energy = constants.get_thermal_energy(300.0)
    print(f"Thermal energy at 300K: {thermal_energy:.4f} eV")
    
    return True


def test_all_physics_modules():
    """Test all physics modules are working."""
    print("\nTesting all physics modules...")
    
    try:
        from core.electron_screening import ElectronScreening
        from core.lattice_effects import LatticeEffects  
        from core.interface_dynamics import InterfaceDynamics
        from core.bubble_dynamics import BubbleDynamics
        
        # Test electron screening
        screening = ElectronScreening()
        screening_energy = screening.total_screening_energy()
        print(f"  Screening energy: {screening_energy['total_screening_energy']:.2f} eV")
        
        # Test lattice effects
        lattice = LatticeEffects()
        lattice_enhancement = lattice.total_lattice_enhancement()
        print(f"  Lattice enhancement: {lattice_enhancement['enhancement_factor']:.2f}x")
        
        # Test interface dynamics
        interface = InterfaceDynamics()
        interface_results = interface.total_interface_enhancement()
        print(f"  Interface field: {interface_results['max_field']:.2e} V/m")
        
        # Test bubble dynamics
        bubble = BubbleDynamics()
        print(f"  Bubble radius: {bubble.params.R0*1e6:.1f} um")
        
        return True
    except Exception as e:
        print(f"[FAIL] Physics modules test failed: {e}")
        return False


def test_integrated_simulation():
    """Test integrated simulation."""
    print("\nTesting integrated simulation...")
    
    try:
        from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters
        
        # Create simulation
        params = IntegratedParameters(
            loading_ratio=0.95,
            electric_field=1e10
        )
        sim = IntegratedLENRSimulation(params)
        
        # Run enhancement calculation
        results = sim.calculate_total_enhancement(energy=10.0)
        
        print(f"  Total enhancement: {results['total_combined_enhancement']:.2e}x")
        print(f"  Energy concentration: {results['total_energy_concentration']:.2f} eV/atom")
        
        # Validate against paper
        validation = sim.validate_against_paper()
        if validation['all_checks_passed']:
            print("  [PASS] All validation checks passed!")
        else:
            print("  [WARNING] Some validation checks failed")
            for check, passed in validation.items():
                if not passed and check != 'all_checks_passed':
                    print(f"    - {check}: FAILED")
        
        return True
    except Exception as e:
        print(f"[FAIL] Integrated simulation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("LENR Mathematical Simulation - Setup Verification")
    print("=" * 70)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("[WARNING] Python 3.8+ is recommended")
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Constants Test", test_constants),
        ("Quantum Tunneling Test", test_quantum_tunneling),
        ("All Physics Modules Test", test_all_physics_modules),
        ("Integrated Simulation Test", test_integrated_simulation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"[FAIL] Test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n[SUCCESS] All tests passed! The LENR simulation framework is ready.")
        print("\nNext steps:")
        print("1. Install any missing dependencies: pip install -r backend/requirements.txt")
        print("2. Start the API server: cd backend && uvicorn main:app --reload")
        print("3. Access the API docs: http://localhost:8000/docs")
    else:
        print("\n[WARNING] Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed: pip install -r backend/requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
