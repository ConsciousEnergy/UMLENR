#!/usr/bin/env python
"""Test script for LENR Simulation API."""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint."""
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Version: {data['version']}")
        print(f"   Physics Modules: {sum(data['physics_modules'].values())}/{len(data['physics_modules'])} operational")
        print(f"   Solvers: {sum(data['solvers'].values())}/{len(data['solvers'])} operational")
        return True
    except Exception as e:
        print(f"   [FAIL] Health check failed: {e}")
        return False


def test_root():
    """Test root endpoint."""
    print("\n2. Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        print(f"   API Name: {data['name']}")
        print(f"   Status: {data['status']}")
        print(f"   Available endpoints: {len(data['endpoints'])}")
        return True
    except Exception as e:
        print(f"   [FAIL] Root endpoint failed: {e}")
        return False


def test_default_parameters():
    """Test getting default parameters."""
    print("\n3. Testing Default Parameters...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/parameters/defaults")
        response.raise_for_status()
        data = response.json()
        print(f"   Material: {data['material']}")
        print(f"   Temperature: {data['temperature']} K")
        print(f"   Loading Ratio: {data['loading_ratio']}")
        print(f"   Electric Field: {data['electric_field']:.1e} V/m")
        return True
    except Exception as e:
        print(f"   [FAIL] Default parameters failed: {e}")
        return False


def test_parameter_ranges():
    """Test getting parameter ranges."""
    print("\n4. Testing Parameter Ranges...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/parameters/ranges")
        response.raise_for_status()
        data = response.json()
        print(f"   Parameters defined: {len(data)}")
        for param, info in list(data.items())[:3]:
            print(f"   {param}: {info['min']:.1e} - {info['max']:.1e} {info['unit']}")
        return True
    except Exception as e:
        print(f"   [FAIL] Parameter ranges failed: {e}")
        return False


def test_create_simulation():
    """Test creating and running a simulation."""
    print("\n5. Testing Simulation Creation...")
    try:
        # Create simulation request
        request_data = {
            "parameters": {
                "material": "Pd",
                "temperature": 300.0,
                "loading_ratio": 0.95,
                "electric_field": 1e10,
                "surface_potential": 0.5,
                "defect_density": 1e21,
                "coherence_domain_size": 1000
            },
            "energy": 10.0,
            "calculate_rate": True,
            "include_validation": True
        }
        
        # Send request
        response = requests.post(
            f"{BASE_URL}/api/v1/simulations/",
            json=request_data
        )
        response.raise_for_status()
        data = response.json()
        
        simulation_id = data['simulation_id']
        print(f"   Simulation ID: {simulation_id}")
        print(f"   Status: {data['status']}")
        
        # Wait for completion
        print("   Waiting for completion", end="")
        max_wait = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            time.sleep(0.5)
            print(".", end="", flush=True)
            
            # Check status
            response = requests.get(f"{BASE_URL}/api/v1/simulations/{simulation_id}")
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'completed':
                print(" COMPLETED")
                results = data['results']
                print(f"   Total Enhancement: {results['total_enhancement']:.2e}")
                print(f"   Energy Concentration: {results['energy_concentration']:.2f} eV/atom")
                print(f"   Screening Energy: {results['screening_energy']:.2f} eV")
                
                # Check validation
                if 'validation' in results:
                    validation = results['validation']
                    print(f"   Validation: {'PASSED' if validation['all_checks_passed'] else 'FAILED'}")
                
                return True
            elif data['status'] == 'failed':
                print(" FAILED")
                print(f"   Error: {data.get('error', 'Unknown error')}")
                return False
        
        print(" TIMEOUT")
        return False
        
    except Exception as e:
        print(f"\n   [FAIL] Simulation creation failed: {e}")
        return False


def test_parameter_scan():
    """Test parameter scanning."""
    print("\n6. Testing Parameter Scan...")
    try:
        request_data = {
            "parameter_name": "loading_ratio",
            "values": [0.80, 0.85, 0.90, 0.95],
            "energy": 10.0
        }
        
        response = requests.post(
            f"{BASE_URL}/api/v1/parameters/scan",
            json=request_data
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"   Scan ID: {data['scan_id']}")
        print(f"   Parameter: {data['parameter_name']}")
        print(f"   Values tested: {len(data['parameter_values'])}")
        
        # Display results
        print("\n   Results:")
        print("   Loading Ratio | Enhancement")
        print("   " + "-" * 30)
        for i, value in enumerate(data['parameter_values']):
            enhancement = data['total_enhancement'][i]
            print(f"   {value:12.2f} | {enhancement:.2e}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Parameter scan failed: {e}")
        return False


def test_batch_simulations():
    """Test batch simulation creation."""
    print("\n7. Testing Batch Simulations...")
    try:
        # Create multiple simulation requests
        requests_data = []
        temperatures = [280, 300, 320]
        
        for temp in temperatures:
            requests_data.append({
                "parameters": {
                    "material": "Pd",
                    "temperature": temp,
                    "loading_ratio": 0.90,
                    "electric_field": 5e9,
                    "surface_potential": 0.4,
                    "defect_density": 5e20,
                    "coherence_domain_size": 500
                },
                "energy": 10.0,
                "calculate_rate": False,
                "include_validation": False
            })
        
        response = requests.post(
            f"{BASE_URL}/api/v1/simulations/batch",
            json=requests_data
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"   Created {len(data)} simulations")
        for sim in data:
            print(f"   - {sim['simulation_id']}: T={sim['parameters']['temperature']}K, Status={sim['status']}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Batch simulations failed: {e}")
        return False


def test_optimal_parameters():
    """Test getting optimal parameters."""
    print("\n8. Testing Optimal Parameters...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/parameters/optimal")
        response.raise_for_status()
        data = response.json()
        
        print(f"   Configurations available: {len(data)}")
        for config_name, config in data.items():
            print(f"\n   {config_name}:")
            print(f"   - Description: {config['description']}")
            print(f"   - Expected enhancement: {config['expected_enhancement']}")
            print(f"   - Material: {config['parameters']['material']}")
            print(f"   - Loading ratio: {config['parameters']['loading_ratio']}")
        
        return True
        
    except Exception as e:
        print(f"   [FAIL] Optimal parameters failed: {e}")
        return False


def main():
    """Run all API tests."""
    print("=" * 60)
    print("LENR Simulation API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    print("\nChecking if API server is running...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("[OK] Server is running")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Server is not running!")
        print("\nPlease start the server with:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Default Parameters", test_default_parameters),
        ("Parameter Ranges", test_parameter_ranges),
        ("Create Simulation", test_create_simulation),
        ("Parameter Scan", test_parameter_scan),
        ("Batch Simulations", test_batch_simulations),
        ("Optimal Parameters", test_optimal_parameters)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{name:20} {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 60)
    print(f"Total: {passed} passed, {failed} failed out of {len(tests)} tests")
    
    if failed == 0:
        print("\n[SUCCESS] All API tests passed!")
        return 0
    else:
        print(f"\n[WARNING] {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
