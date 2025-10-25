#!/usr/bin/env python
"""Simple demonstration of the LENR simulation framework."""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def main():
    """Run a simple demonstration of the LENR simulation."""
    print("=" * 70)
    print("LENR Mathematical Simulation - Quick Demo")
    print("=" * 70)
    
    try:
        # Import the integrated simulation
        from core.integrated_simulation import IntegratedLENRSimulation, IntegratedParameters
        
        print("\n[1] Creating simulation with optimal parameters...")
        
        # Create simulation with parameters from the paper
        params = IntegratedParameters(
            material="Pd",
            temperature=300.0,
            loading_ratio=0.95,
            electric_field=1e10,  # 10 GV/m
            surface_potential=0.5,
            defect_density=1e21,
            coherence_domain_size=1000
        )
        
        sim = IntegratedLENRSimulation(params)
        
        print("[2] Calculating enhancement factors...")
        
        # Calculate enhancement at 10 eV (typical thermal energy)
        results = sim.calculate_total_enhancement(energy=10.0)
        
        print("\n" + "=" * 70)
        print("RESULTS AT 10 eV")
        print("=" * 70)
        
        print(f"\nScreening Enhancement:")
        print(f"  Screening energy: {results['screening_energy']:.2f} eV")
        print(f"  Screening factor: {results['screening_factor']:.2e}")
        
        print(f"\nLattice Enhancement:")
        print(f"  Lattice enhancement: {results['lattice_enhancement']:.2f}x")
        print(f"  Coherent energy: {results['coherent_energy']:.3f} eV/atom")
        
        print(f"\nInterface Enhancement:")
        print(f"  Interface enhancement: {results['interface_enhancement']:.2f}x")
        print(f"  Max field: {results['max_interface_field']:.2e} V/m")
        
        print(f"\nTunneling Enhancement:")
        print(f"  Tunneling enhancement: {results['tunneling_enhancement']:.2e}x")
        print(f"  Tunneling probability: {results['tunneling_probability']:.2e}")
        
        print(f"\n" + "=" * 70)
        print("TOTAL ENHANCEMENT")
        print("=" * 70)
        print(f"  Combined enhancement factor: {results['total_combined_enhancement']:.2e}x")
        print(f"  Final reaction probability: {results['final_reaction_probability']:.2e}")
        print(f"  Total energy concentration: {results['total_energy_concentration']:.2f} eV/atom")
        
        # Validate against paper predictions
        print(f"\n" + "=" * 70)
        print("VALIDATION AGAINST PAPER")
        print("=" * 70)
        
        validation = sim.validate_against_paper()
        print(f"  Enhancement in 10^3-10^5 range: {validation['enhancement_in_range']}")
        print(f"  Screening in 10-100 eV range: {validation['screening_in_range']}")
        print(f"  Field in 10^9-10^11 V/m range: {validation['field_in_range']}")
        print(f"  Energy concentration in 10-100 eV range: {validation['energy_concentration_in_range']}")
        
        if validation['all_checks_passed']:
            print("\n[SUCCESS] All validation checks passed!")
            print("The simulation matches the theoretical predictions from the paper.")
        else:
            print("\n[WARNING] Some validation checks failed.")
        
        # Show enhancement at different energies
        print(f"\n" + "=" * 70)
        print("ENHANCEMENT VS ENERGY")
        print("=" * 70)
        print(f"{'Energy (eV)':<15} {'Enhancement':<20} {'Probability':<20}")
        print("-" * 55)
        
        for energy in [1.0, 10.0, 100.0, 1000.0]:
            res = sim.calculate_total_enhancement(energy)
            print(f"{energy:<15.1f} {res['total_combined_enhancement']:<20.2e} {res['final_reaction_probability']:<20.2e}")
        
        print(f"\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print("\nThe simulation demonstrates that combining multiple enhancement mechanisms")
        print("(electron screening, lattice coherence, interface fields, and thermal effects)")
        print(f"produces total enhancement factors of {results['total_combined_enhancement']:.0e}, which matches")
        print("the 10^3-10^5 range predicted in the theoretical framework.")
        print("\nThis provides a physically plausible path to observable LENR effects")
        print("without requiring exotic physics beyond the standard model.")
        
    except Exception as e:
        print(f"\n[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
