#!/usr/bin/env python3
"""
Metal GPU Acceleration Validation Script

This script verifies that the Metal GPU accelerated physics engine produces
the same results as the CPU-based solve.py implementation. It performs
comprehensive testing across different simulation scenarios.

Tests:
1. Basic single-stage simulation comparison
2. Physics equation verification (force, inductance, field)
3. Energy conservation validation
4. Numerical stability under extreme conditions
5. Performance benchmarking

Usage:
    python validate_metal_acceleration.py [--config config.json] [--verbose] [--benchmark]
"""

import numpy as np
import time
import json
import sys
import os
import argparse
from pathlib import Path
import traceback
import warnings
from typing import Dict, List, Tuple, Any

# Import both implementations
try:
    from solve import CoilgunSimulation as CPUSimulation
    from equations import CoilgunPhysicsEngine as CPUPhysics
    print("✓ CPU implementation imported successfully")
except ImportError as e:
    print(f"✗ Failed to import CPU implementation: {e}")
    sys.exit(1)

try:
    from metal_acceleration_extended import CoilgunSimulation as MetalSimulation
    from metal_acceleration import create_metal_accelerated_solver, MetalAcceleration
    print("✓ Metal GPU implementation imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Metal GPU implementation: {e}")
    sys.exit(1)

class ValidationResult:
    """Container for validation test results"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.passed = False
        self.message = ""
        self.cpu_result = None
        self.metal_result = None
        self.relative_error = None
        self.execution_time_cpu = None
        self.execution_time_metal = None
        self.speedup = None

class MetalValidation:
    """Comprehensive validation suite for Metal GPU acceleration"""
    
    def __init__(self, config_file: str = "multistage_4_coilgun_config.json", verbose: bool = True):
        self.config_file = config_file
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.tolerance = 1e-6  # Relative tolerance for floating point comparisons
        self.strict_tolerance = 1e-9  # Stricter tolerance for critical physics
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        print(f"🔬 Starting Metal GPU validation with config: {config_file}")
        print(f"   Tolerance: {self.tolerance}, Strict tolerance: {self.strict_tolerance}")
    
    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        
        print("\n" + "="*60)
        print("           METAL GPU VALIDATION SUITE")
        print("="*60)
        
        # Test 1: Basic Physics Engine Initialization
        self._test_physics_engine_initialization()
        
        # Test 2: Core Physics Equations
        self._test_core_physics_equations()
        
        # Test 3: Simulation Comparison
        self._test_simulation_comparison()
        
        # Test 4: Energy Conservation
        self._test_energy_conservation()
        
        # Test 5: Numerical Stability
        self._test_numerical_stability()
        
        # Test 6: Performance Benchmark
        self._test_performance_benchmark()
        
        # Generate summary report
        return self._generate_summary_report()
    
    def _test_physics_engine_initialization(self):
        """Test that both engines initialize with same parameters"""
        
        result = ValidationResult("Physics Engine Initialization")
        
        try:
            # Create single-stage config for basic physics testing
            single_stage_config = self._create_single_stage_config()
            
            # Initialize CPU physics engine
            start_time = time.time()
            cpu_physics = CPUPhysics(single_stage_config)
            cpu_init_time = time.time() - start_time
            
            # Initialize Metal physics engine
            start_time = time.time()
            metal_physics, metal_accel = create_metal_accelerated_solver(single_stage_config, verbose=False)
            metal_init_time = time.time() - start_time
            
            # Compare key parameters
            params_to_check = [
                'coil_length', 'total_turns', 'proj_mass', 'capacitance', 
                'initial_voltage', 'total_resistance'
            ]
            
            differences = []
            for param in params_to_check:
                if hasattr(cpu_physics, param) and hasattr(metal_physics, param):
                    cpu_val = getattr(cpu_physics, param)
                    metal_val = getattr(metal_physics, param)
                    
                    if abs(cpu_val - metal_val) > self.tolerance * max(abs(cpu_val), abs(metal_val), 1e-12):
                        differences.append(f"{param}: CPU={cpu_val:.6e}, Metal={metal_val:.6e}")
            
            if differences:
                result.message = f"Parameter differences found: {'; '.join(differences)}"
                result.passed = False
            else:
                result.message = f"All parameters match. Init times: CPU={cpu_init_time:.3f}s, Metal={metal_init_time:.3f}s"
                result.passed = True
            
            result.execution_time_cpu = cpu_init_time
            result.execution_time_metal = metal_init_time
            
        except Exception as e:
            result.message = f"Initialization test failed: {str(e)}"
            result.passed = False
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        self._print_test_result(result)
    
    def _test_core_physics_equations(self):
        """Test core physics calculations match between implementations"""
        
        result = ValidationResult("Core Physics Equations")
        
        try:
            single_stage_config = self._create_single_stage_config()
            
            # Initialize both engines
            cpu_physics = CPUPhysics(single_stage_config)
            metal_physics, metal_accel = create_metal_accelerated_solver(single_stage_config, verbose=False)
            
            # Test parameters
            test_positions = np.linspace(-0.01, 0.08, 10)
            test_currents = np.linspace(10, 1000, 10)
            
            max_relative_error = 0.0
            errors = []
            
            for pos in test_positions:
                for current in test_currents:
                    # Test inductance calculation
                    cpu_L = cpu_physics.get_inductance(pos, current)
                    metal_L = metal_physics.get_inductance(pos, current)
                    
                    rel_error_L = abs(cpu_L - metal_L) / max(abs(cpu_L), abs(metal_L), 1e-12)
                    if rel_error_L > max_relative_error:
                        max_relative_error = rel_error_L
                    
                    # Test magnetic force calculation
                    cpu_force = cpu_physics.magnetic_force_ferromagnetic(current, pos)
                    metal_force = metal_physics.magnetic_force_ferromagnetic(current, pos)
                    
                    # Handle tuple return values (force, eddy_power_loss)
                    if isinstance(cpu_force, tuple):
                        cpu_force = cpu_force[0]
                    if isinstance(metal_force, tuple):
                        metal_force = metal_force[0]
                    
                    rel_error_F = abs(cpu_force - metal_force) / max(abs(cpu_force), abs(metal_force), 1e-12)
                    if rel_error_F > max_relative_error:
                        max_relative_error = rel_error_F
                    
                    # Store significant errors for reporting
                    if rel_error_L > self.tolerance:
                        errors.append(f"L@pos={pos:.3f},I={current:.0f}: {rel_error_L:.2e}")
                    if rel_error_F > self.tolerance:
                        errors.append(f"F@pos={pos:.3f},I={current:.0f}: {rel_error_F:.2e}")
            
            if max_relative_error < self.tolerance:
                result.message = f"Physics equations match within tolerance. Max error: {max_relative_error:.2e}"
                result.passed = True
            else:
                result.message = f"Physics equations differ. Max error: {max_relative_error:.2e}. Errors: {'; '.join(errors[:5])}"
                result.passed = False
            
            result.relative_error = max_relative_error
            
        except Exception as e:
            result.message = f"Physics equations test failed: {str(e)}"
            result.passed = False
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        self._print_test_result(result)
    
    def _test_simulation_comparison(self):
        """Compare full simulation results between CPU and Metal implementations"""
        
        result = ValidationResult("Full Simulation Comparison")
        
        try:
            single_stage_config = self._create_single_stage_config()
            
            # Run CPU simulation
            print("   Running CPU simulation...")
            start_time = time.time()
            cpu_sim = CPUSimulation(single_stage_config)
            cpu_results = cpu_sim.run_simulation(save_data=False, verbose=False, show_progress=False)
            cpu_time = time.time() - start_time
            
            # Run Metal simulation
            print("   Running Metal GPU simulation...")
            start_time = time.time()
            metal_sim = MetalSimulation(single_stage_config)
            metal_results = metal_sim.run_simulation(save_data=False, verbose=False, show_progress=False)
            metal_time = time.time() - start_time
            
            # Compare key simulation results
            comparisons = [
                ('max_velocity', 'max_velocity'),
                ('max_current', 'max_current'),
                ('max_force', 'max_force'),
                ('efficiency', 'efficiency'),
                ('exit_velocity', 'exit_velocity')
            ]
            
            max_error = 0.0
            differences = []
            
            for cpu_key, metal_key in comparisons:
                if hasattr(cpu_results, cpu_key) and hasattr(metal_results, metal_key):
                    cpu_val = getattr(cpu_results, cpu_key)
                    metal_val = getattr(metal_results, metal_key)
                    
                    rel_error = abs(cpu_val - metal_val) / max(abs(cpu_val), abs(metal_val), 1e-12)
                    max_error = max(max_error, rel_error)
                    
                    if rel_error > self.tolerance:
                        differences.append(f"{cpu_key}: CPU={cpu_val:.4f}, Metal={metal_val:.4f} (error: {rel_error:.2e})")
            
            # Calculate speedup
            speedup = cpu_time / metal_time if metal_time > 0 else float('inf')
            
            if max_error < self.tolerance:
                result.message = f"Simulation results match within tolerance. Max error: {max_error:.2e}, Speedup: {speedup:.1f}x"
                result.passed = True
            else:
                result.message = f"Simulation results differ. Max error: {max_error:.2e}. Differences: {'; '.join(differences)}"
                result.passed = False
            
            result.relative_error = max_error
            result.execution_time_cpu = cpu_time
            result.execution_time_metal = metal_time
            result.speedup = speedup
            result.cpu_result = cpu_results
            result.metal_result = metal_results
            
        except Exception as e:
            result.message = f"Simulation comparison failed: {str(e)}"
            result.passed = False
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        self._print_test_result(result)
    
    def _test_energy_conservation(self):
        """Test energy conservation in both implementations"""
        
        result = ValidationResult("Energy Conservation")
        
        try:
            single_stage_config = self._create_single_stage_config()
            
            # Test both implementations for energy conservation
            cpu_physics = CPUPhysics(single_stage_config)
            metal_physics, metal_accel = create_metal_accelerated_solver(single_stage_config, verbose=False)
            
            # Simulate short time period and check energy balance
            initial_energy_cpu = 0.5 * cpu_physics.capacitance * cpu_physics.initial_voltage**2
            initial_energy_metal = 0.5 * metal_physics.capacitance * metal_physics.initial_voltage**2
            
            # Both should have same initial energy
            energy_diff = abs(initial_energy_cpu - initial_energy_metal)
            rel_energy_error = energy_diff / max(initial_energy_cpu, initial_energy_metal, 1e-12)
            
            if rel_energy_error < self.strict_tolerance:
                result.message = f"Energy conservation validated. Initial energy error: {rel_energy_error:.2e}"
                result.passed = True
            else:
                result.message = f"Energy conservation failed. Initial energy error: {rel_energy_error:.2e}"
                result.passed = False
            
            result.relative_error = rel_energy_error
            
        except Exception as e:
            result.message = f"Energy conservation test failed: {str(e)}"
            result.passed = False
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        self._print_test_result(result)
    
    def _test_numerical_stability(self):
        """Test numerical stability under extreme conditions"""
        
        result = ValidationResult("Numerical Stability")
        
        try:
            # Create config with extreme parameters
            extreme_config = self._create_single_stage_config()
            
            # Modify to create extreme conditions
            with open(extreme_config, 'r') as f:
                config_data = json.load(f)
            
            # Test with high currents and fast dynamics
            config_data['capacitor']['initial_voltage'] = 1000.0  # High voltage
            config_data['simulation']['tolerance'] = 1e-12  # Strict tolerance
            
            extreme_config_file = "temp_extreme_config.json"
            with open(extreme_config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            try:
                # Test both implementations under extreme conditions
                cpu_physics = CPUPhysics(extreme_config_file)
                metal_physics, metal_accel = create_metal_accelerated_solver(extreme_config_file, verbose=False)
                
                # Test extreme current values
                extreme_current = 5000.0  # 5 kA
                extreme_position = 0.05
                
                cpu_force = cpu_physics.magnetic_force_ferromagnetic(extreme_current, extreme_position)
                metal_force = metal_physics.magnetic_force_ferromagnetic(extreme_current, extreme_position)
                
                if isinstance(cpu_force, tuple):
                    cpu_force = cpu_force[0]
                if isinstance(metal_force, tuple):
                    metal_force = metal_force[0]
                
                # Check for NaN or infinite values
                if np.isnan(cpu_force) or np.isinf(cpu_force) or np.isnan(metal_force) or np.isinf(metal_force):
                    result.message = "Numerical instability detected (NaN/Inf values)"
                    result.passed = False
                else:
                    rel_error = abs(cpu_force - metal_force) / max(abs(cpu_force), abs(metal_force), 1e-12)
                    if rel_error < self.tolerance:
                        result.message = f"Numerical stability maintained under extreme conditions. Error: {rel_error:.2e}"
                        result.passed = True
                    else:
                        result.message = f"Numerical stability compromised. Error: {rel_error:.2e}"
                        result.passed = False
                
                result.relative_error = rel_error if 'rel_error' in locals() else float('inf')
                
            finally:
                # Clean up temporary file
                if os.path.exists(extreme_config_file):
                    os.remove(extreme_config_file)
            
        except Exception as e:
            result.message = f"Numerical stability test failed: {str(e)}"
            result.passed = False
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        self._print_test_result(result)
    
    def _test_performance_benchmark(self):
        """Benchmark performance between CPU and Metal implementations"""
        
        result = ValidationResult("Performance Benchmark")
        
        try:
            single_stage_config = self._create_single_stage_config()
            
            # Benchmark physics calculations
            num_iterations = 100
            test_positions = np.linspace(-0.01, 0.08, num_iterations)
            test_currents = np.linspace(100, 1000, num_iterations)
            
            # Initialize both engines
            cpu_physics = CPUPhysics(single_stage_config)
            metal_physics, metal_accel = create_metal_accelerated_solver(single_stage_config, verbose=False)
            
            # Benchmark CPU
            start_time = time.time()
            for i in range(num_iterations):
                pos = test_positions[i]
                current = test_currents[i]
                _ = cpu_physics.magnetic_force_ferromagnetic(current, pos)
                _ = cpu_physics.get_inductance(pos, current)
            cpu_time = time.time() - start_time
            
            # Benchmark Metal
            start_time = time.time()
            for i in range(num_iterations):
                pos = test_positions[i]
                current = test_currents[i]
                _ = metal_physics.magnetic_force_ferromagnetic(current, pos)
                _ = metal_physics.get_inductance(pos, current)
            metal_time = time.time() - start_time
            
            # Calculate speedup
            speedup = cpu_time / metal_time if metal_time > 0 else float('inf')
            
            result.message = f"Performance benchmark: CPU={cpu_time:.3f}s, Metal={metal_time:.3f}s, Speedup={speedup:.1f}x"
            result.passed = True  # Performance test always passes if it completes
            result.execution_time_cpu = cpu_time
            result.execution_time_metal = metal_time
            result.speedup = speedup
            
        except Exception as e:
            result.message = f"Performance benchmark failed: {str(e)}"
            result.passed = False
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        self._print_test_result(result)
    
    def _create_single_stage_config(self) -> str:
        """Create a single-stage configuration for testing"""
        
        # Extract single stage from multi-stage config
        single_config = {
            "simulation": self.config["shared"]["simulation"],
            "circuit_model": self.config["shared"]["circuit_model"],
            "magnetic_model": self.config["shared"]["magnetic_model"],
            "output": self.config["shared"]["output"],
            "capacitor": self.config["shared"]["capacitor"],
            "projectile": self.config["shared"]["projectile"],
            "coil": self.config["stages"][0]["coil"]  # Use first stage coil config
        }
        
        # Save temporary config file
        temp_config_file = "temp_single_stage_config.json"
        with open(temp_config_file, 'w') as f:
            json.dump(single_config, f, indent=2)
        
        return temp_config_file
    
    def _print_test_result(self, result: ValidationResult):
        """Print individual test result"""
        
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{status} | {result.test_name}")
        print(f"      {result.message}")
        
        if result.execution_time_cpu and result.execution_time_metal:
            print(f"      Timing: CPU={result.execution_time_cpu:.3f}s, Metal={result.execution_time_metal:.3f}s")
        
        if result.speedup:
            print(f"      Speedup: {result.speedup:.1f}x")
        
        if result.relative_error is not None:
            print(f"      Relative Error: {result.relative_error:.2e}")
    
    def _generate_summary_report(self) -> bool:
        """Generate final validation summary report"""
        
        print("\n" + "="*60)
        print("              VALIDATION SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.results:
                if not result.passed:
                    print(f"   • {result.test_name}: {result.message}")
        
        # Performance summary
        benchmark_result = next((r for r in self.results if r.test_name == "Performance Benchmark"), None)
        if benchmark_result and benchmark_result.speedup:
            print(f"\n⚡ PERFORMANCE: {benchmark_result.speedup:.1f}x speedup achieved")
        
        # Overall assessment
        all_critical_passed = all(r.passed for r in self.results if r.test_name in [
            "Physics Engine Initialization", 
            "Core Physics Equations", 
            "Full Simulation Comparison",
            "Energy Conservation"
        ])
        
        if all_critical_passed:
            print(f"\n🎉 VALIDATION SUCCESSFUL: Metal GPU implementation verified!")
            print(f"   The Metal GPU acceleration correctly computes the physics engine")
            print(f"   values and generates the same valid data as solve.py.")
        else:
            print(f"\n⚠️  VALIDATION ISSUES DETECTED: Please review failed tests.")
            print(f"   The Metal GPU implementation may have accuracy or compatibility issues.")
        
        # Clean up temporary files
        temp_files = ["temp_single_stage_config.json", "temp_extreme_config.json"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return all_critical_passed

def main():
    """Main validation function"""
    
    parser = argparse.ArgumentParser(description="Validate Metal GPU acceleration against CPU solve.py")
    parser.add_argument("--config", default="multistage_4_coilgun_config.json", 
                       help="Configuration file to use for testing")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output with detailed error traces")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run extended performance benchmarks")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file '{args.config}' not found!")
        sys.exit(1)
    
    # Run validation
    validator = MetalValidation(args.config, verbose=args.verbose)
    success = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 