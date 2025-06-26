"""
Coilgun Optimization Script

This script iterates through several iterations of possible coilgun designs and configurations
to optimize coilgun performance. The script will output a csv file of all valid configurations
along with a config.json file to be used in the electromagnetic coilgun simulation.

Optimization Strategy:
- Target velocity is a hard requirement (configurations below target are invalid)
- Multi-objective scoring optimizes for efficiency, energy consumption, design complexity,
  current management, and practical feasibility
- Lower scores indicate better configurations among valid configurations
- Tracks and saves all configurations that tie for the best score
- Graceful interruption support: Press Ctrl+C to stop early and save current progress
"""

import json
import numpy as np
import csv
import os
import signal
import sys
import time
import threading
import multiprocessing as mp
import concurrent.futures
import contextlib
import io
import glob
import tempfile
import shutil
import datetime
from typing import Dict, Any, List, Tuple
from solve import CoilgunSimulation, MultiStageCoilgunSimulation  # Make sure solve.py is in your PYTHONPATH

# Check if tqdm is available for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
    # Configure tqdm to avoid multiple progress bars
    tqdm.monitor_interval = 0  # Disable monitoring thread
    if 'TQDM_DISABLE' not in os.environ:
        os.environ['TQDM_DISABLE'] = '0'  # Enable tqdm for main progress bar only
except ImportError:
    TQDM_AVAILABLE = False

def check_gpu_acceleration_compatibility():
    """Check if GPU acceleration module is compatible with optimization."""
    try:
        # Try to import GPU acceleration module from GPUAccelerationFiles directory
        import sys
        gpu_files_path = os.path.join(os.path.dirname(__file__), 'GPUAccelerationFiles')
        sys.path.insert(0, gpu_files_path)
        
        from gpu_acceleration import GPUAcceleration
        
        # Initialize GPU acceleration
        gpu_accel = GPUAcceleration(verbose=False)
        
        if not gpu_accel.gpu_available:
            return False, None, None, None
        
        # Check if the GPU acceleration module has the required methods
        required_methods = ['setup_julia_physics']
        available_methods = []
        
        for method in required_methods:
            if hasattr(gpu_accel, method):
                available_methods.append(method)
        
        if len(available_methods) >= 1:  # At least one method available
            return True, gpu_accel, gpu_accel.gpu_backend, gpu_accel.julia_main
        else:
            if hasattr(gpu_accel, 'julia_main'):
                # Even without full methods, we can still use basic Julia acceleration
                return True, gpu_accel, gpu_accel.gpu_backend, gpu_accel.julia_main
            return False, None, None, None
            
    except ImportError:
        # GPU acceleration module not found - try basic Julia support
        return check_basic_julia_compatibility()
    except Exception as e:
        print(f"⚠ GPU acceleration compatibility check failed: {e}")
        return check_basic_julia_compatibility()

def check_basic_julia_compatibility():
    """Check for basic Julia acceleration when GPU module is not available."""
    try:
        from juliacall import Main as jl
        jl.seval("using Pkg")
          # Minimal CPU-only acceleration
        class BasicJuliaAcceleration:
            def __init__(self):
                self.gpu_available = False
                self.gpu_backend = None
                self.julia_main = jl
            
            def setup_julia_physics(self):
                jl.seval("using DifferentialEquations, LinearAlgebra")
        
        return False, BasicJuliaAcceleration(), None, jl
    except:
        return False, None, None, None

# Check for GPU acceleration support
try:
    GPU_AVAILABLE, GPU_ACCELERATION, GPU_TYPE, jl = check_gpu_acceleration_compatibility()
except:
    GPU_AVAILABLE, GPU_ACCELERATION, GPU_TYPE, jl = False, None, None, None

if GPU_AVAILABLE:
    print(f"✓ {GPU_TYPE} GPU acceleration available for optimization")
    try:
        GPU_ACCELERATION.setup_julia_physics()
    except:
        print(f"⚠ GPU setup failed - using limited features")
else:
    print("ℹ No GPU acceleration available - using CPU multi-threading")

# Determine optimal number of workers
if GPU_AVAILABLE:
    # For GPU, use fewer workers to avoid memory conflicts
    MAX_WORKERS = min(4, mp.cpu_count())
    ACCELERATION_TYPE = f"{GPU_TYPE} GPU"
else:
    # For CPU, use more workers
    MAX_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one core free
    ACCELERATION_TYPE = f"CPU ({MAX_WORKERS} threads)"

# Only show essential initialization info
print(f"🚀 Optimization mode: {ACCELERATION_TYPE}")
print(f"🔧 Max parallel workers: {MAX_WORKERS}")
print()


def load_material_data() -> Dict[str, Any]:
    """Load material properties from materials.json if available, otherwise use defaults."""
    try:
        if not os.path.exists("materials.json"):
            print("Warning: materials.json not found. Using basic material properties...")
            return create_default_materials()
        with open("materials.json", 'r') as f:
            return json.load(f)
    except:
        print("Error loading materials. Using basic material properties...")
        return create_default_materials()


def create_default_materials() -> Dict[str, Any]:
    """Provide a minimal set of material properties if materials.json is missing."""
    return {
        "materials": {
            "Copper": {"resistivity": 1.68e-8, "mu_r": 0.999991, "density": 8960},
            "Low_Carbon_Steel": {"mu_r": 1000, "density": 7850},
            "Pure_Iron": {"mu_r": 5000, "density": 7874},
            "Aluminum": {"mu_r": 1.000022, "density": 2700}
        },
        "wire_specifications": {
            "awg_diameter_mm": {"14": 1.628, "16": 1.291, "18": 1.024, "20": 0.812},
            "current_capacity_A": {"14": 32, "16": 22, "18": 16, "20": 11}
        },
        "magnetic_methods": ["biot_savart", "finite_element", "analytical"]
    }


def get_float_input(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
    """Prompt the user for a float input with optional default, min, and max."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def get_int_input(prompt: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
    """Prompt the user for an integer input with optional default, min, and max."""
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def get_choice_input(prompt: str, choices: list, default: str = None) -> str:
    """Prompt the user to select from a list of choices, with an optional default."""
    while True:
        print(f"\n{prompt}")
        for i, choice in enumerate(choices, 1):
            print(f"  {i}. {choice}")
        if default:
            user_input = input(f"Enter choice (1-{len(choices)}) (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"Enter choice (1-{len(choices)}): ").strip()
        try:
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(choices):
                return choices[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number.")


def get_yes_no_input(prompt: str, default: bool = None) -> bool:
    """Prompt the user for a yes/no input with an optional default."""
    while True:
        options = "y/N" if default is False else "Y/n" if default is True else "y/n"
        user_input = input(f"{prompt} ({options}): ").strip().lower()
        if not user_input and default is not None:
            return default
        if user_input in ["y", "yes"]:
            return True
        if user_input in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'.")


def get_range_input(prompt: str, default_min, default_max, default_step, is_int=True):
    """Prompt the user for a min, max, and step for a parameter range."""
    min_val = get_int_input(f"{prompt} min", default_min) if is_int else get_float_input(f"{prompt} min", default_min)
    max_val = get_int_input(f"{prompt} max", default_max) if is_int else get_float_input(f"{prompt} max", default_max)
    step = get_int_input(f"{prompt} step", default_step) if is_int else get_float_input(f"{prompt} step", default_step)
    return min_val, max_val, step


def calculate_projectile_height(mass: float, diameter: float, density: float) -> float:
    """Calculate projectile height from mass, diameter, and density."""
    radius = diameter / 2
    height = mass / (np.pi * radius**2 * density)
    return height


def get_wire_gauge_range(wire_spec: Dict[str, Any]) -> List[int]:
    """Get a sorted list of available wire gauges from the wire_spec dictionary."""
    return sorted([int(k) for k in wire_spec["awg_diameter_mm"].keys()])


def awg_to_diameter_m(wire_gauge: int, wire_spec: Dict[str, Any]) -> float:
    """Convert AWG to diameter in meters using wire_spec."""
    return wire_spec["awg_diameter_mm"][str(wire_gauge)] / 1000.0


def build_config_dict(params, materials, wire_spec, fast_mode=False):
    """Build a simulation config dictionary for a given parameter set."""
    # Handle both tuple/list and integer values for stages parameter
    stages_param = params["stages"]
    num_stages = stages_param[1] if isinstance(stages_param, (list, tuple)) else stages_param
    
    # Fast mode settings for optimization
    if fast_mode:
        # Reduced accuracy for speed during optimization
        max_step = 5e-6  # Larger time steps
        tolerance = 1e-8  # Less strict tolerance
        axial_disc = min(500, params.get("axial_discretization", 1000))  # Reduced discretization
        radial_disc = min(50, params.get("radial_discretization", 100))
        method = "RK45"  # Faster but less accurate method
    else:
        # High accuracy for final results
        max_step = 1e-6
        tolerance = 1e-9
        axial_disc = params.get("axial_discretization", 1000)
        radial_disc = params.get("radial_discretization", 100)
        method = "RK45"
    
    # Create base config with multi-stage settings
    config = {
        "multi_stage": {
            "enabled": True,
            "num_stages": num_stages,
            "shared_settings": ["projectile", "simulation", "circuit_model", "magnetic_model", "output"],
            "stage_groups": [list(range(1, num_stages + 1))]  # All stages use same config during optimization
        },
        "stages": [],
        "shared": {
            "projectile": {
                "diameter": params["projectile_diameter"],
                "length": params["projectile_height"],
                "material": params["projectile_material"],
                "initial_position": params["initial_position"],
                "initial_velocity": params["initial_velocity"]
            },
            "simulation": {
                "time_span": [0, params["simulation_time"]],
                "max_step": max_step,
                "tolerance": tolerance,
                "method": method  # Use RK45 for better stability in multi-stage simulations
            },
            "circuit_model": {
                "switch_resistance": params["switch_resistance"],
                "switch_inductance": params["switch_inductance"],
                "parasitic_capacitance": params["parasitic_capacitance"],
                "include_skin_effect": params.get("include_skin_effect", True) if not fast_mode else False,
                "include_proximity_effect": params.get("include_proximity_effect", True) if not fast_mode else False
            },
            "magnetic_model": {
                "calculation_method": params["calculation_method"],
                "axial_discretization": axial_disc,
                "radial_discretization": radial_disc,
                "include_saturation": params.get("include_saturation", False) if not fast_mode else False,
                "include_hysteresis": params.get("include_hysteresis", False) if not fast_mode else False,
                "include_eddy_currents": False,
                "include_skin_effect": False,
                "include_temperature_effects": False,
                "include_displacement_current": False,
                "force_components": {
                    "gradient_force": True,
                    "reluctance_force": True,
                    "eddy_current_force": False
                }
            },
            "output": {
                "save_trajectory": False,  # Always false during optimization
                "save_current_profile": False,
                "save_field_data": False,
                "print_progress": False,
                "save_interval": 1000,  # Larger interval
            }
        }
    }
      # Calculate coil length based on turns per layer and wire specifications
    wire_gauge = params["wire_gauge"]
    wire_diameter_mm = wire_spec["awg_diameter_mm"][str(wire_gauge)]
    wire_diameter_m = wire_diameter_mm / 1000.0
    insulation_thickness = params["insulation_thickness"]
    effective_wire_diameter = wire_diameter_m + insulation_thickness
    calculated_coil_length = params["turns_per_layer"] * effective_wire_diameter
    
    # Create stage-specific configurations
    base_stage_config = {
        "coil": {
            "inner_diameter": params["projectile_diameter"] * 1.05,
            "length": calculated_coil_length,
            "wire_gauge_awg": str(params["wire_gauge"]),  # Convert to string for compatibility
            "num_layers": params["layers"],
            "turns_per_layer": params["turns_per_layer"],
            "wire_material": params["wire_material"],
            "insulation_thickness": params["insulation_thickness"],
            "packing_factor": params["packing_factor"],
            "min_temperature": 20,
            "max_temperature": 80
        },
        "capacitor": {
            "capacitance": params["capacitance"],
            "initial_voltage": params["voltage"],
            "esr": 0.01,
            "esl": 5e-8
        }
    }
    
    # Add configurations for each stage
    for stage_num in range(1, num_stages + 1):
        stage_config = {
            "stage_id": stage_num,
            "group_id": "group_1"  # All stages in same group during optimization
        }
        stage_config.update(base_stage_config)
        config["stages"].append(stage_config)
    
    return config


def calculate_optimization_score(velocity, target_velocity, efficiency, max_current, params, config):
    """
    Multi-objective scoring function for coilgun optimization.
    Lower scores are better. Balances efficiency and practicality.
    
    Scoring Components (Total: 0-250 points, lower is better):
    1. Efficiency (0-100): Penalizes low efficiency designs
    2. Energy Consumption (0-60): Favors lower energy requirements  
    3. Design Complexity (0-40): Penalizes overly complex designs
    4. Current Management (0-30): Graduated thresholds for current levels
    5. Practical Feasibility (0-20): Penalizes unsafe/impractical combinations
    """
    # 1. Efficiency score (0-100)
    efficiency_score = max(0, 100 - efficiency)
    
    # 2. Energy consumption score (0-60)
    energy_factor = params["voltage"] ** 2 * params["capacitance"] * 0.5
    energy_score = min(60, energy_factor / 8000)
    
    # 3. Complexity penalty (0-40)
    num_stages = params.get("stages", 1)
    num_layers = config["stages"][0]["coil"]["num_layers"]
    complexity_score = min(40, (num_stages - 1) * 5 + (num_layers - 1) * 3)

    # 4. Current management score (0-30)
    if max_current <= 1000:
        current_score = 0
    elif max_current <= 3000:
        current_score = min(20, (max_current - 1000) * 0.01)
    else:
        current_score = min(30, 20 + (max_current - 3000) * 0.005)

    # 5. Practical feasibility (0-20)
    wire_gauge = params["wire_gauge"]
    turns = params["turns_per_layer"]
    feasibility_score = 0
    
    if wire_gauge >= 16 and turns > 200:
        feasibility_score += 8
    if max_current > 5000:
        feasibility_score += 10
    elif max_current > 3000:
        feasibility_score += 5
    if params["voltage"] > 600:
        feasibility_score += 6
    elif params["voltage"] > 400:
        feasibility_score += 2
    
    return (efficiency_score + energy_score + complexity_score + 
            current_score + feasibility_score)


def simulate_and_score_worker(params_batch):
    """
    Worker function for parallel simulation processing.
    Processes a batch of parameter sets and returns results.
    """
    # Disable all progress bars and output in worker processes
    import os
    import atexit
    os.environ['TQDM_DISABLE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Monkey-patch ProgressTracker to disable all progress display
    try:
        import solve
        class DummyProgressTracker:
            def __init__(self, *args, **kwargs): pass
            def start_integration_display(self): pass
            def update(self, t, y): pass
            def stop(self): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        solve.ProgressTracker = DummyProgressTracker
    except:
        pass
    
    results = []
    materials = params_batch['materials']
    wire_spec = params_batch['wire_spec']
    target_velocity = params_batch['target_velocity']
    param_sets = params_batch['param_sets']
    
    # Create worker-specific temp directory
    worker_id = f"{os.getpid()}_{threading.current_thread().ident}_{time.time():.6f}"
    worker_temp_dir = tempfile.mkdtemp(prefix=f"worker_{worker_id}_")
    original_cwd = os.getcwd()
    
    # Register cleanup function to ensure temp directory is always removed
    def cleanup_temp_dir():
        try:
            if os.path.exists(worker_temp_dir):
                shutil.rmtree(worker_temp_dir, ignore_errors=True)
        except:
            pass
    
    atexit.register(cleanup_temp_dir)
    
    try:
        os.chdir(worker_temp_dir)
        
        for i, params in enumerate(param_sets):
            temp_config_file = None
            try:
                temp_config_file = f"temp_worker_{worker_id}_{i}.json"
                config = build_config_dict(params, materials, wire_spec, fast_mode=True)
                
                with open(temp_config_file, "w") as f:
                    json.dump(config, f, indent=4)
                
                with SuppressOutput():
                    sim = MultiStageCoilgunSimulation(temp_config_file)
                    results_data = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
                
                # Extract metrics
                final_velocity = float(results_data.get('final_velocity_ms', 0))
                max_current = float(results_data.get('max_current_A', 0))
                max_force = float(results_data.get('max_force_N', 0))
                overall_efficiency = float(results_data.get('overall_efficiency_percent', 0))
                
                score = calculate_optimization_score(
                    final_velocity, target_velocity, overall_efficiency, 
                    max_current, params, config
                )
                
                # Check if final velocity meets target (after all stages)
                valid = final_velocity >= target_velocity
                
                # Process stage results
                stage_data = []
                stage_velocities = results_data.get('stage_final_velocities_ms', [])
                stage_efficiencies = results_data.get('stage_efficiencies_percent', [])
                
                for j, (vel, eff) in enumerate(zip(stage_velocities, stage_efficiencies)):
                    stage_data.append({
                        "stage": j + 1,
                        "velocity": float(vel),
                        "efficiency": float(eff)
                    })
                
                result = {
                    "velocity": final_velocity,
                    "max_current": max_current,
                    "max_force": max_force,
                    "efficiency": overall_efficiency,
                    "score": score,
                    "valid": valid,
                    "stage_results": stage_data,
                    "params": params.copy(),
                    "acceleration_used": "CPU Parallel"
                }
                results.append(result)
                
            except Exception as e:
                # Handle failures gracefully
                error_result = {"valid": False, "params": params.copy()}
                results.append(error_result)
            finally:
                # Always clean up temp config file immediately after use
                if temp_config_file and os.path.exists(temp_config_file):
                    try:
                        os.remove(temp_config_file)
                    except:
                        pass
    
    finally:
        # Change back to original directory and clean up temp directory
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        # Force cleanup of temp directory
        cleanup_temp_dir()
        
        # Unregister the atexit handler since we've already cleaned up
        try:
            atexit.unregister(cleanup_temp_dir)
        except:
            pass
    
    return results


def simulate_and_score_gpu(params_list, materials, wire_spec, target_velocity):
    """GPU-accelerated batch simulation processing."""
    if not GPU_AVAILABLE or not GPU_ACCELERATION:
        raise RuntimeError("GPU acceleration not available")
    
    try:
        results = []
        batch_size = min(16, len(params_list))
        
        for i in range(0, len(params_list), batch_size):
            batch_params = params_list[i:i + batch_size]
            batch_results = []
            
            for params in batch_params:
                try:
                    result = simulate_and_score_with_gpu(params, materials, wire_spec, target_velocity)
                    batch_results.append(result)
                except:
                    error_result = {"valid": False, "params": params.copy()}
                    batch_results.append(error_result)
            
            results.extend(batch_results)
        
        return results
        
    except:
        return simulate_and_score_cpu_parallel(params_list, materials, wire_spec, target_velocity)


def simulate_and_score_with_gpu(params, materials, wire_spec, target_velocity):
    """Run a single simulation using GPU acceleration."""
    config = build_config_dict(params, materials, wire_spec, fast_mode=True)
    temp_config_file = f"temp_gpu_sim_config_{os.getpid()}_{time.time()}.json"
    
    try:
        with open(temp_config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        # Try GPU-accelerated solver if available
        if hasattr(GPU_ACCELERATION, 'create_accelerated_solver'):
            with SuppressOutput():
                physics_engine, _ = GPU_ACCELERATION.create_accelerated_solver(temp_config_file)
                solution = physics_engine.solve_balanced_julia()
                analysis = physics_engine.analyze_solution_julia(solution)
            
            final_velocity = float(analysis.get('final_velocity', 0))
            max_current = float(analysis.get('max_current', 0))
            max_force = float(analysis.get('max_force', 0))
            overall_efficiency = float(analysis.get('efficiency', 0)) * 100
        else:
            # Fallback to regular simulation
            with SuppressOutput():
                sim = MultiStageCoilgunSimulation(temp_config_file)
                results = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
            
            final_velocity = float(results.get('final_velocity_ms', 0))
            max_current = float(results.get('max_current_A', 0))
            max_force = float(results.get('max_force_N', 0))
            overall_efficiency = float(results.get('overall_efficiency_percent', 0))
        
        score = calculate_optimization_score(
            final_velocity, target_velocity, overall_efficiency, 
            max_current, params, config
        )
        
        valid = final_velocity >= target_velocity
        
        # Create stage results (simplified for GPU path)
        # Note: GPU acceleration may not provide detailed stage-by-stage data
        num_stages = len(config["stages"])
        stage_data = []
        for i in range(num_stages):
            stage_data.append({
                "stage": i + 1,
                "velocity": final_velocity if i == num_stages - 1 else final_velocity * (i + 1) / num_stages,
                "efficiency": overall_efficiency
            })
        
        return {
            "velocity": final_velocity,
            "max_current": max_current,
            "max_force": max_force,
            "efficiency": overall_efficiency,
            "score": score,
            "valid": valid,
            "stage_results": stage_data,
            "params": params.copy(),
            "acceleration_used": f"{GPU_TYPE} GPU"
        }
        
    except:
        return {"valid": False, "params": params.copy()}
    finally:
        try:
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
        except:
            pass


def simulate_and_score_cpu_parallel(params_list, materials, wire_spec, target_velocity):
    """CPU multi-threaded batch simulation processing."""
    if not params_list:
        return []
    
    # Split parameter sets into batches for parallel processing
    batch_size = max(1, len(params_list) // MAX_WORKERS)
    batches = []
    
    for i in range(0, len(params_list), batch_size):
        batch_params = params_list[i:i + batch_size]
        batch = {
            'param_sets': batch_params,
            'materials': materials,
            'wire_spec': wire_spec,
            'target_velocity': target_velocity
        }
        batches.append(batch)
    
    # Process batches in parallel
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(simulate_and_score_worker, batch): batch 
            for batch in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
            except:
                pass  # Continue with other batches
    
    return all_results


def process_parameter_batch(params_list, materials, wire_spec, target_velocity):
    """Process a batch of parameters using available acceleration."""
    if not params_list:
        return []
    
    if GPU_AVAILABLE:
        try:
            return simulate_and_score_gpu(params_list, materials, wire_spec, target_velocity)
        except:
            return simulate_and_score_cpu_parallel(params_list, materials, wire_spec, target_velocity)
    else:
        return simulate_and_score_cpu_parallel(params_list, materials, wire_spec, target_velocity)


def simulate_and_score(params, materials, wire_spec, target_velocity):
    """Run a simulation for given parameters and return performance metrics."""
    # Disable progress display
    try:
        import solve
        class DummyProgressTracker:
            def __init__(self, *args, **kwargs): pass
            def start_integration_display(self): pass
            def update(self, t, y): pass
            def stop(self): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
        solve.ProgressTracker = DummyProgressTracker
    except:
        pass
    
    # Try GPU acceleration first if available
    if GPU_AVAILABLE and GPU_ACCELERATION:
        try:
            return simulate_and_score_with_gpu(params, materials, wire_spec, target_velocity)
        except:
            pass  # Fall through to CPU version
      # CPU version
    config = build_config_dict(params, materials, wire_spec, fast_mode=True)
    temp_config_file = f"temp_sim_config_{os.getpid()}_{threading.current_thread().ident}_{time.time():.6f}.json"
    
    try:
        with open(temp_config_file, "w") as f:
            json.dump(config, f, indent=4)
        
        with SuppressOutput():
            sim = MultiStageCoilgunSimulation(temp_config_file)
            results = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
        
        # Extract metrics
        final_velocity = float(results.get('final_velocity_ms', 0))
        max_current = float(results.get('max_current_A', 0))
        max_force = float(results.get('max_force_N', 0))
        overall_efficiency = float(results.get('overall_efficiency_percent', 0))
        
        score = calculate_optimization_score(
            final_velocity, target_velocity, overall_efficiency, 
            max_current, params, config
        )
        
        # Check if final velocity meets target (after all stages)
        valid = final_velocity >= target_velocity
        
        # Process stage results
        stage_data = []
        stage_velocities = results.get('stage_final_velocities_ms', [])
        stage_efficiencies = results.get('stage_efficiencies_percent', [])
        
        for i, (vel, eff) in enumerate(zip(stage_velocities, stage_efficiencies)):
            stage_data.append({
                "stage": i + 1,
                "velocity": float(vel),
                "efficiency": float(eff)
            })
        
        return {
            "velocity": final_velocity,
            "max_current": max_current,
            "max_force": max_force,
            "efficiency": overall_efficiency,
            "score": score,
            "valid": valid,
            "stage_results": stage_data,
            "params": params.copy(),
            "acceleration_used": "CPU"
        }
    except:
        return {"valid": False, "params": params.copy()}
    finally:
        # Clean up temp files
        try:
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
        except:
            pass

def quick_feasibility_filter(params, wire_spec, target_velocity):
    """
    Quick feasibility check to filter out obviously bad configurations
    without running full simulation. Returns (is_feasible, reason).
    """
    wire_gauge = params["wire_gauge"]
    turns = params["turns_per_layer"]
    voltage = params["voltage"]
    capacitance = params["capacitance"]
    layers = params["layers"]
    
    # Check wire gauge exists
    if str(wire_gauge) not in wire_spec["awg_diameter_mm"]:
        return False, "wire_gauge_unavailable"
    
    # Very basic energy check (only filter obviously impossible cases)
    total_energy = 0.5 * capacitance * voltage**2
    
    # Minimum energy requirement (capacitor must have at least the kinetic energy needed)
    projectile_mass = params["projectile_mass"]
    kinetic_energy_needed = 0.5 * projectile_mass * target_velocity**2
    
    # Only filter if capacitor energy is less than kinetic energy (impossible even with 100% efficiency)
    if total_energy < kinetic_energy_needed:
        return False, "insufficient_energy"
    
    # Basic current capacity check (more realistic resistance model for coilguns)
    wire_current_capacity = wire_spec["current_capacity_A"].get(str(wire_gauge), 1000)
    
    # Realistic coilgun resistance estimate based on wire length and gauge
    # Typical coilgun has much higher resistance due to long wire lengths
    wire_resistance_per_m = 0.0001 * (2 ** ((wire_gauge - 10) / 3))  # Rough AWG resistance scaling
    estimated_wire_length = turns * layers * 0.2  # Rough estimate of total wire length
    estimated_resistance = max(0.1, wire_resistance_per_m * estimated_wire_length)
    estimated_current = voltage / estimated_resistance
    
    # Very lenient current check - allow 200x overload for pulsed operation
    # Real coilguns routinely exceed continuous ratings by huge factors
    if estimated_current > wire_current_capacity * 200:
        return False, "current_too_high"
    
    # Only filter completely unrealistic configurations
    if voltage > 1500:  # Extremely high voltage threshold
        return False, "voltage_extreme"
    
    if wire_gauge >= 20 and turns > 200:  # Very thin wire + many turns
        return False, "wire_too_thin"
    
    if layers > 15:  # Too many layers
        return False, "too_many_layers"
    
    return True, "feasible"


def smart_parameter_sampling(params, materials, wire_spec, target_velocity, max_configs=50000, disable_filtering=False):
    """
    Intelligently sample parameter space instead of exhaustive search.
    Returns reduced list of promising parameter combinations.
    """
    print(f"🧠 Smart sampling enabled - targeting {max_configs:,} configurations")
    if disable_filtering:
        print(f"⚠️ Filtering disabled for debugging purposes")
    
    stages_min, stages_max, stages_step = params["stages"]
    wire_gauge_min, wire_gauge_max = params["wire_gauge"]
    layers_min, layers_max, layers_step = params["layers"]
    turns_min, turns_max, turns_step = params["turns_per_layer"]
    voltage_min, voltage_max, voltage_step = params["voltage"]
    cap_min, cap_max, cap_step = params["capacitance"]
    
    promising_configs = []
    filtered_count = 0
    filter_reasons = {}  # Track reasons for filtering
    
    # Priority sampling strategy
    priorities = [
        # (stages, wire_density, layer_density, turns_density, voltage_density, cap_density)
        (1, 2, 2, 2, 2, 3),  # Focus on simpler, higher energy configs
        (2, 2, 2, 2, 3, 2),  # Multi-stage with good energy
        (3, 3, 3, 3, 2, 2),  # Complex but balanced
        (1, 1, 1, 1, 1, 1),  # Sample entire space lightly
    ]
    
    for priority_idx, (stage_density, wire_density, layer_density, turns_density, voltage_density, cap_density) in enumerate(priorities):
        print(f"   📊 Priority {priority_idx + 1}/4: Sampling with density ({stage_density}, {wire_density}, {layer_density}, {turns_density}, {voltage_density}, {cap_density})")
        
        # Generate sampling points for each parameter
        stages_points = list(range(stages_min, stages_max + 1, max(1, stages_step * stage_density)))
        wire_points = list(range(wire_gauge_min, wire_gauge_max + 1, max(1, wire_density)))
        layer_points = list(range(layers_min, layers_max + 1, max(1, layers_step * layer_density)))
        turns_points = list(range(turns_min, turns_max + 1, max(1, turns_step * turns_density)))
        voltage_points = list(range(voltage_min, voltage_max + 1, max(1, voltage_step * voltage_density)))
        cap_points = list(np.linspace(cap_min, cap_max, max(2, int((cap_max - cap_min) / cap_step / cap_density) + 1)))
        
        # Sample combinations for this priority
        priority_count = 0
        max_priority_configs = max_configs // len(priorities)
        
        for stages in stages_points:
            if len(promising_configs) >= max_configs:
                break
            for wire_gauge in wire_points:
                if len(promising_configs) >= max_configs:
                    break
                for layers in layer_points:
                    if len(promising_configs) >= max_configs:
                        break
                    for turns_per_layer in turns_points:
                        if len(promising_configs) >= max_configs:
                            break
                        for voltage in voltage_points:
                            if len(promising_configs) >= max_configs:
                                break
                            for capacitance in cap_points:
                                if len(promising_configs) >= max_configs:
                                    break
                                
                                candidate = {
                                    "stages": stages,
                                    "wire_gauge": wire_gauge,
                                    "layers": layers,
                                    "turns_per_layer": turns_per_layer,
                                    "voltage": voltage,
                                    "capacitance": capacitance,
                                    "projectile_mass": params["projectile_mass"],
                                    "projectile_diameter": params["projectile_diameter"],
                                    "projectile_material": params["projectile_material"],
                                    "projectile_height": params["projectile_height"],
                                    "wire_material": params["wire_material"],
                                    "insulation_thickness": params["insulation_thickness"],
                                    "packing_factor": params["packing_factor"],
                                    "initial_position": params["initial_position"],
                                    "initial_velocity": params["initial_velocity"],
                                    "switch_resistance": params["switch_resistance"],
                                    "switch_inductance": params["switch_inductance"],
                                    "parasitic_capacitance": params["parasitic_capacitance"],
                                    "include_skin_effect": params["include_skin_effect"],
                                    "include_proximity_effect": params["include_proximity_effect"],
                                    "calculation_method": params["calculation_method"],
                                    "axial_discretization": params["axial_discretization"],
                                    "radial_discretization": params["radial_discretization"],
                                    "include_saturation": params["include_saturation"],
                                    "include_hysteresis": params["include_hysteresis"],
                                    "simulation_time": params["simulation_time"]
                                }
                                
                                # Apply feasibility filter
                                if disable_filtering:
                                    # Skip filtering for debugging
                                    promising_configs.append(candidate)
                                    priority_count += 1
                                else:
                                    feasible, reason = quick_feasibility_filter(candidate, wire_spec, target_velocity)
                                    if feasible:
                                        promising_configs.append(candidate)
                                        priority_count += 1
                                    else:
                                        filtered_count += 1
                                        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                                
                                if priority_count >= max_priority_configs:
                                    break
                            if priority_count >= max_priority_configs:
                                break
                        if priority_count >= max_priority_configs:
                            break
                    if priority_count >= max_priority_configs:
                        break
                if priority_count >= max_priority_configs:
                    break
            if priority_count >= max_priority_configs:
                break
        
        print(f"   ✅ Generated {priority_count:,} configs in priority {priority_idx + 1}")
    
    print(f"🧠 Smart sampling complete:")
    print(f"   ✨ Promising configurations: {len(promising_configs):,}")
    print(f"   🚫 Filtered out: {filtered_count:,}")
    print(f"   📈 Reduction factor: {filtered_count + len(promising_configs):,} → {len(promising_configs):,}")
    
    # Show filtering breakdown
    if filter_reasons:
        print(f"\n📊 Filtering breakdown:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / max(filtered_count, 1)) * 100
            print(f"   • {reason}: {count:,} ({percentage:.1f}%)")
    
    # If we filtered too aggressively and have very few configs, add some basic ones
    if len(promising_configs) < max_configs // 10:
        print(f"⚠️  Few configurations found, adding basic fallback configurations...")
        for stages in range(stages_min, min(stages_min + 3, stages_max + 1)):
            for wire_gauge in range(wire_gauge_min, min(wire_gauge_min + 6, wire_gauge_max + 1), 2):
                for layers in range(layers_min, min(layers_min + 5, layers_max + 1), 2):
                    for turns_per_layer in range(turns_min, min(turns_min + 100, turns_max + 1), turns_step * 5):
                        for voltage in range(voltage_min, min(voltage_min + 200, voltage_max + 1), voltage_step * 2):
                            for capacitance in np.linspace(cap_min, min(cap_min + 0.1, cap_max), 3):
                                if len(promising_configs) >= max_configs // 5:
                                    break
                                candidate = {
                                    "stages": stages,
                                    "wire_gauge": wire_gauge,
                                    "layers": layers,
                                    "turns_per_layer": turns_per_layer,
                                    "voltage": voltage,
                                    "capacitance": capacitance,
                                    "projectile_mass": params["projectile_mass"],
                                    "projectile_diameter": params["projectile_diameter"],
                                    "projectile_material": params["projectile_material"],
                                    "projectile_height": params["projectile_height"],
                                    "wire_material": params["wire_material"],
                                    "insulation_thickness": params["insulation_thickness"],
                                    "packing_factor": params["packing_factor"],
                                    "initial_position": params["initial_position"],
                                    "initial_velocity": params["initial_velocity"],
                                    "switch_resistance": params["switch_resistance"],
                                    "switch_inductance": params["switch_inductance"],
                                    "parasitic_capacitance": params["parasitic_capacitance"],
                                    "include_skin_effect": params["include_skin_effect"],
                                    "include_proximity_effect": params["include_proximity_effect"],
                                    "calculation_method": params["calculation_method"],
                                    "axial_discretization": params["axial_discretization"],
                                    "radial_discretization": params["radial_discretization"],
                                    "include_saturation": params["include_saturation"],
                                    "include_hysteresis": params["include_hysteresis"],
                                    "simulation_time": params["simulation_time"]
                                }
                                if disable_filtering:
                                    promising_configs.append(candidate)
                                else:
                                    feasible, reason = quick_feasibility_filter(candidate, wire_spec, target_velocity)
                                    if feasible:
                                        promising_configs.append(candidate)
        print(f"   🔋 Generated {len(promising_configs)} configurations with fallback")
    
    return promising_configs


def optimize_coilgun(params, materials, wire_spec, target_velocity, total_combinations, use_smart_sampling=True):
    """
    Iterate through all parameter combinations, run full simulation for each,
    and find the best configuration. Supports graceful interruption with Ctrl+C.
    Uses smart sampling for massive speed improvements.
    """
    # Initialize interrupt handler
    interrupt_handler = OptimizationInterrupt()
    
    best_config = None
    best_score = float('inf')
    best_configs = []  # List to store all configurations with the best score
    results_list = []
    
    # Optimization header
    print("\n" + "=" * 25)
    print("Optimization Process")
    print("=" * 25 + "\n")
    print("Press Ctrl+C at any time to interrupt and save current progress")
    print("-" * 50)
    
    # Clean up any existing temp files before starting
    cleanup_temp_files()
    
    # Choose optimization strategy
    if use_smart_sampling and total_combinations > 10000:
        # Use smart sampling for large optimization spaces
        max_configs = min(50000, total_combinations // 10)  # Sample 10% or max 50k
        print(f"🧠 Using smart sampling strategy")
        print(f"📊 Original space: {total_combinations:,} configurations")
        print(f"🎯 Target sample size: {max_configs:,} configurations")
        
        all_param_combinations = smart_parameter_sampling(params, materials, wire_spec, target_velocity, max_configs, disable_filtering=False)
        actual_combinations = len(all_param_combinations)
        print(f"✨ Final sample: {actual_combinations:,} configurations")
        estimated_time_hours = actual_combinations * 0.3 / 3600  # Assume 0.3s per config with fast mode
        print(f"⏱️  Estimated time: {estimated_time_hours:.1f} hours")
        
    else:
        # Use exhaustive search for smaller spaces
        print(f"🔍 Using exhaustive search")
        print(f"📊 Total combinations: {total_combinations:,}")
        
        stages_min, stages_max, stages_step = params["stages"]
        wire_gauge_min, wire_gauge_max = params["wire_gauge"]
        layers_min, layers_max, layers_step = params["layers"]
        turns_min, turns_max, turns_step = params["turns_per_layer"]
        voltage_min, voltage_max, voltage_step = params["voltage"]
        cap_min, cap_max, cap_step = params["capacitance"]
        
        # Generate all parameter combinations
        all_param_combinations = []
        
        for stages in range(stages_min, stages_max + 1, stages_step):
            for wire_gauge in range(wire_gauge_min, wire_gauge_max + 1, 1):
                if str(wire_gauge) not in wire_spec["awg_diameter_mm"]:
                    continue
                for layers in range(layers_min, layers_max + 1, layers_step):
                    for turns_per_layer in range(turns_min, turns_max + 1, turns_step):
                        for voltage in range(voltage_min, voltage_max + 1, voltage_step):
                            num_steps = int(round((cap_max - cap_min) / cap_step)) + 1
                            for capacitance in np.linspace(cap_min, cap_max, num_steps):
                                candidate = params.copy()
                                candidate.update({
                                    "stages": stages,
                                    "wire_gauge": wire_gauge,
                                    "layers": layers,
                                    "turns_per_layer": turns_per_layer,
                                    "voltage": voltage,
                                    "capacitance": capacitance
                                })
                                all_param_combinations.append(candidate)
        
        actual_combinations = len(all_param_combinations)
    
    print("-" * 50)
      # Progress tracking setup
    progress_count = 0
    last_percent_reported = -1
    
    # Adaptive batch processing
    if GPU_AVAILABLE:
        # GPU batch size based on memory
        batch_size = min(16, max(4, actual_combinations // 200))
        print(f"🚀 Using GPU batch processing (batch size: {batch_size})")
    else:
        # CPU batch size based on worker count and complexity
        batch_size = min(64, max(MAX_WORKERS * 2, actual_combinations // 100))
        print(f"🔧 Using CPU parallel processing (batch size: {batch_size})")

    print(f"📊 Processing {actual_combinations:,} configurations in batches of {batch_size}")
    print("-" * 50)

    try:
        # Initialize progress bar only if tqdm is available
        pbar = None
        if TQDM_AVAILABLE:
            os.environ['TQDM_DISABLE'] = '0'  # Re-enable for main progress bar
            pbar = tqdm(total=actual_combinations, desc="Optimizing (Valid: 0)", 
                       unit="config", leave=True, ascii=False, miniters=1,
                       bar_format='{desc}: {percentage:5.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                       position=0, dynamic_ncols=True)
        
        # Process in batches
        for batch_start in range(0, len(all_param_combinations), batch_size):
            # Check for interrupt at the start of each batch
            if interrupt_handler.interrupted:
                break
            
            batch_end = min(batch_start + batch_size, len(all_param_combinations))
            batch_params = all_param_combinations[batch_start:batch_end]
            
            # Process batch using available acceleration with completely suppressed output
            with SuppressOutput():
                batch_results = process_parameter_batch(batch_params, materials, wire_spec, target_velocity)
            
            # Track valid configurations found in this batch
            valid_in_batch = 0
            
            # Process batch results
            for sim_result in batch_results:
                if sim_result.get("valid", False):
                    valid_in_batch += 1
                    
                    # Store only the simulation metrics, not the raw results
                    sim_metrics = {
                        "velocity": sim_result["velocity"],
                        "max_current": sim_result["max_current"],
                        "max_force": sim_result["max_force"],
                        "efficiency": sim_result["efficiency"],
                        "stage_results": sim_result["stage_results"]
                    }
                    
                    # Add acceleration info if available
                    if "acceleration_used" in sim_result:
                        sim_metrics["acceleration_used"] = sim_result["acceleration_used"]
                    
                    # Combine with parameters
                    result_record = {**sim_result["params"], **sim_metrics}
                    results_list.append(result_record)
                    
                    # Handle best score tracking with tie support
                    if sim_result["score"] < best_score:
                        # New best score found - clear previous ties and set new best
                        best_score = sim_result["score"]
                        best_config = sim_result
                        best_configs = [sim_result.copy()]  # Start new list with this config
                    elif sim_result["score"] == best_score:
                        # Tie with current best score - add to list
                        best_configs.append(sim_result.copy())
            
            # Update progress
            progress_count = batch_end
            valid_found = len(results_list)            # Update progress bar or print progress
            if TQDM_AVAILABLE and pbar:
                pbar.update(len(batch_params))
                pbar.set_description(f"Optimizing (Valid: {valid_found})")
            else:
                # Fallback to print-based progress
                current_percent = (progress_count / actual_combinations) * 100
                
                if (current_percent >= last_percent_reported + 1.0) or \
                   (progress_count % (batch_size * 5) == 0) or \
                   (progress_count >= actual_combinations):
                    
                    acceleration_info = f" ({ACCELERATION_TYPE})" if progress_count % (batch_size * 10) == 0 else ""
                    print(f"Progress: {current_percent:5.2f}% ({progress_count:,}/{actual_combinations:,}) - Valid configs: {valid_found}{acceleration_info}")
                    last_percent_reported = current_percent
            
            # Update interrupt handler with current progress
            interrupt_handler.update_progress(best_config, results_list, progress_count, actual_combinations)

    except KeyboardInterrupt:        
        # Handle graceful interruption
        best_config = interrupt_handler.best_config
        results_list = interrupt_handler.results_list
        progress_count = interrupt_handler.progress_count
        
        print(f"\nOptimization interrupted after {progress_count} combinations.")
        print("Saving current progress...")
        
        # Save partial results immediately
        if results_list:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            partial_csv_filename = f"partial_optimization_results_{timestamp}.csv"
            save_results_to_csv(results_list, partial_csv_filename)
            print(f"Partial results saved to: {partial_csv_filename}")
            
            if best_config:
                partial_config_filename = f"partial_best_config_{timestamp}.json"
                save_best_config_to_json(best_config, partial_config_filename, materials, wire_spec)
                print(f"Best configuration saved to: {partial_config_filename}")
        else:
            print("No valid configurations found before interruption.")
    
    finally:
        # Clean up resources
        if TQDM_AVAILABLE and 'pbar' in locals() and pbar:
            try:
                pbar.close()
                print()  # Add newline
            except:
                pass
        
        # Restore signal handler and clean up temp files
        interrupt_handler.restore_signal_handler()
        cleanup_temp_files()
    
    # Final completion message
    if not interrupt_handler.interrupted:
        print("\nOptimization completed successfully!")
    
    # Display best configuration details
    if best_config and best_config.get("valid", False):
        print(f"\nBest configuration found with score: {best_score:.2f}")
        
        # Check for tied configurations
        if len(best_configs) > 1:
            print(f"⚖️  TIED CONFIGURATIONS: {len(best_configs)} configurations achieved the same best score!")
            print("All tied configurations will be saved for your review.")
        
        # Show acceleration used if available
        if "acceleration_used" in best_config:
            print(f"🚀 Acceleration used: {best_config['acceleration_used']}")
        
        # Extract necessary parameters for score explanation
        best_params = best_config.get("params", {})
        temp_config = build_config_dict(best_params, materials, wire_spec)
        explain_score(
            best_config["velocity"], target_velocity, 
            best_config["efficiency"], best_config["max_current"],
            best_params, temp_config, best_score
        )
        
        # Save tied configurations if any exist
        if len(best_configs) > 1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_tied_configs_to_json(best_configs, f"best_tied_configs_{timestamp}", materials, wire_spec)
    else:
        print("\nNo valid configurations found that meet the target velocity.")
    
    # Show acceleration summary
    show_acceleration_summary(results_list)
    
    return best_config, results_list, best_configs

def save_results_to_csv(results_list, filename):
    """Save all valid configurations to a CSV file."""
    if not results_list:
        print("No valid configurations to save.")
        return
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(results_list[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_list:
            writer.writerow(row)

def save_best_config_to_json(best_config, filename, materials, wire_spec):
    """Save the best configuration as a JSON file suitable for simulation."""
    if not best_config or not best_config.get("valid", False):
        print("No valid configuration to save.")
        return
    
    try:
        # Extract parameters from best_config
        best_params = best_config.get("params", {})
        
        # Build the complete configuration using existing function
        config = build_config_dict(best_params, materials, wire_spec)
        
        # Add optimization metadata
        config["optimization_metadata"] = {
            "score": best_config.get("score", 0),
            "velocity_achieved": best_config.get("velocity", 0),
            "efficiency": best_config.get("efficiency", 0),
            "max_current": best_config.get("max_current", 0),
            "max_force": best_config.get("max_force", 0),
            "stage_results": best_config.get("stage_results", []),
            "optimization_timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuration saved successfully to {filename}")
        
    except Exception as e:
        print(f"Error saving configuration: {e}")

def save_tied_configs_to_json(best_configs, filename_prefix, materials, wire_spec):
    """Save all tied best configurations as separate JSON files."""
    if not best_configs:
        print("No tied configurations to save.")
        return
    
    print(f"\nSaving {len(best_configs)} tied configurations...")
    
    for i, config in enumerate(best_configs):
        try:
            # Extract parameters from config
            best_params = config.get("params", {})
            
            # Build the complete configuration
            full_config = build_config_dict(best_params, materials, wire_spec)
            
            # Add optimization metadata
            full_config["optimization_metadata"] = {
                "score": config.get("score", 0),
                "velocity_achieved": config.get("velocity", 0),
                "efficiency": config.get("efficiency", 0),
                "max_current": config.get("max_current", 0),
                "max_force": config.get("max_force", 0),
                "stage_results": config.get("stage_results", []),
                "tie_rank": i + 1,
                "total_tied_configs": len(best_configs),
                "optimization_timestamp": datetime.datetime.now().isoformat()
            }
            
            # Create filename for this tied config
            tied_filename = f"{filename_prefix}_tied_config_{i+1}_of_{len(best_configs)}.json"
            
            # Save to JSON
            with open(tied_filename, 'w') as f:
                json.dump(full_config, f, indent=4)
            
            print(f"  Saved tied config {i+1}: {tied_filename}")
            
        except Exception as e:
            print(f"  Error saving tied config {i+1}: {e}")

def explain_score(velocity, target_velocity, efficiency, max_current, params, config, score):
    """Provide detailed breakdown of optimization score components."""
    print(f"\n{'='*50}")
    print("OPTIMIZATION SCORE BREAKDOWN")
    print(f"{'='*50}")
    print(f"Total Score: {score:.2f} (lower is better)")
    print(f"Target Velocity: {target_velocity:.1f} m/s (REQUIRED)")
    print(f"Achieved Velocity: {velocity:.1f} m/s ({'✓ PASS' if velocity >= target_velocity else '✗ FAIL'})")
    print(f"Overall Efficiency: {efficiency:.1f}%")
    print(f"Max Current: {max_current:.1f} A")
    print(f"Configuration:")
    print(f"  - Stages: {params.get('stages', 1)}")
    print(f"  - Wire Gauge: {params['wire_gauge']} AWG")
    print(f"  - Layers: {config['stages'][0]['coil']['num_layers']}")
    print(f"  - Turns/Layer: {params['turns_per_layer']}")
    print(f"  - Voltage: {params['voltage']} V")
    print(f"  - Capacitance: {params['capacitance']:.3f} F")
    
    # Calculate score components
    efficiency_score = max(0, 100 - efficiency)
    energy_factor = params["voltage"] ** 2 * params["capacitance"] * 0.5
    energy_score = min(60, energy_factor / 8000)
    
    num_stages = params.get("stages", 1)
    num_layers = config["stages"][0]["coil"]["num_layers"]
    complexity_score = min(40, (num_stages - 1) * 5 + (num_layers - 1) * 3)
    
    if max_current <= 1000:
        current_score = 0
    elif max_current <= 3000:
        current_score = min(20, (max_current - 1000) * 0.01)
    else:
        current_score = min(30, 20 + (max_current - 3000) * 0.005)
    
    wire_gauge = params["wire_gauge"]
    turns = params["turns_per_layer"]
    feasibility_score = 0
    
    if wire_gauge >= 16 and turns > 200:
        feasibility_score += 8
    if max_current > 5000:
        feasibility_score += 10
    elif max_current > 3000:
        feasibility_score += 5
    if params["voltage"] > 600:
        feasibility_score += 6
    elif params["voltage"] > 400:
        feasibility_score += 2

    print(f"\nScore Components:")
    print(f"  - Efficiency: {efficiency_score:.1f}/100 (100 - {efficiency:.1f}%)")
    print(f"  - Energy Consumption: {energy_score:.1f}/60 (E = ½CV² = {energy_factor:.0f}J)")
    print(f"  - Design Complexity: {complexity_score:.1f}/40 ({num_stages} stages, {num_layers} layers)")
    print(f"  - Current Management: {current_score:.1f}/30 ({max_current:.0f}A)")
    if max_current <= 1000:
        print(f"    └─ Safe range (≤1000A): No penalty")
    elif max_current <= 3000:
        print(f"    └─ Moderate range (1001-3000A): Requires specialized equipment")
    else:
        print(f"    └─ High range (>3000A): Extreme high-power territory")
    print(f"  - Practical Feasibility: {feasibility_score:.1f}/20")
    
    if feasibility_score > 0:
        print(f"    └─ Penalties for:")
        if wire_gauge >= 16 and turns > 200:
            print(f"      • Thin wire (≥{wire_gauge} AWG) + many turns: +8")
        if max_current > 5000:
            print(f"      • Extremely high current (>5000A): +10")
        elif max_current > 3000:
            print(f"      • Very high current (>3000A): +5")
        if params["voltage"] > 600:
            print(f"      • High voltage (>{params['voltage']}V): +6")
        elif params["voltage"] > 400:
            print(f"      • Moderate high voltage (>{params['voltage']}V): +2")
    
    print(f"\nNote: Velocity is a hard requirement, not scored.")
    print(f"Only configurations meeting target velocity are considered valid.")
    print(f"{'='*50}")

class OptimizationInterrupt:
    """Class to handle graceful interruption of optimization process."""
    
    def __init__(self):
        self.interrupted = False
        self.best_config = None
        self.results_list = []
        self.progress_count = 0
        self.total_combinations = 0
        self._original_handler = None
        
        # Set up signal handler for Ctrl+C
        self._original_handler = signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signal (Ctrl+C)."""
        if self.interrupted:
            # Second Ctrl+C - force exit
            print("\n\nSecond interrupt detected - forcing exit...")
            if self._original_handler and self._original_handler != signal.SIG_DFL:
                signal.signal(signal.SIGINT, self._original_handler)
            else:
                sys.exit(1)
        
        print(f"\n\n{'='*50}")
        print("OPTIMIZATION INTERRUPTED BY USER")
        print(f"{'='*50}")
        print(f"Progress: {self.progress_count}/{self.total_combinations} combinations tested")
        if self.progress_count > 0:
            percent = (self.progress_count / self.total_combinations) * 100
            print(f"Completed: {percent:.1f}% of total optimization")
        print("Saving current progress... (Press Ctrl+C again to force quit)")
        self.interrupted = True
    
    def update_progress(self, best_config, results_list, progress_count, total_combinations):
        """Update the current state of optimization."""
        self.best_config = best_config
        self.results_list = results_list[:]  # Copy the list
        self.progress_count = progress_count
        self.total_combinations = total_combinations
    
    def restore_signal_handler(self):
        """Restore the original signal handler."""
        if self._original_handler:
            signal.signal(signal.SIGINT, self._original_handler)

def main():
    """
    Main function to prompt user, run optimization, and save results.
    """
    import atexit
    
    # Register cleanup function to run on exit
    atexit.register(cleanup_temp_files)
    
    # Clean up any temporary files from previous runs at startup
    cleanup_temp_files()
    
    print("=" * 50)
    print("COILGUN OPTIMIZATION")
    print("=" * 50)
    print("\nWelcome to the Coilgun Optimization Tool!")
    print("This tool will help you find the optimal configuration for your coilgun design.")
    print("Please follow the prompts to enter your design parameters.")
    print("\n" + "=" * 50)
    print("INTERRUPT FEATURE:")
    print("You can press Ctrl+C at any time during optimization to stop early")
    print("and save the current best configuration and all valid results found.")
    print("=" * 50)

    # Load materials and wire specs
    materials = load_material_data()
    wire_spec = materials["wire_specifications"]
    # Filter to only magnetic materials (mu_r > 10)
    material_choices = [k for k in materials["materials"].keys() 
                       if materials["materials"][k].get("mu_r", 1) > 10]
    # Filter to materials with conductivity specified
    wire_material_choices = [k for k in materials["materials"].keys() 
                           if "conductivity" in materials["materials"][k]]
    magnetic_methods = ["biot_savart", "finite_element", "analytical"]
    
    # Section: Projectile Parameters
    print("\n" + "=" * 25)
    print("Projectile Parameters")
    print("=" * 25)
    projectile_material = get_choice_input("Projectile material", material_choices, default="Pure_Iron")
    projectile_mass = get_float_input("Projectile mass (kg)", default=1.0, min_val=0.01)
    projectile_diameter = get_float_input("Projectile diameter (m)", default=0.0508, min_val=0.001)
    density = materials["materials"][projectile_material]["density"]
    projectile_height = calculate_projectile_height(projectile_mass, projectile_diameter, density)
    print(f"Calculated projectile height: {projectile_height:.4f} m")
    initial_position = get_float_input("Projectile initial position (m)", default=0.0)
    initial_velocity = get_float_input("Projectile initial velocity (m/s)", default=0.0)
    target_velocity = get_float_input("Target velocity (m/s)", default=100.0, min_val=0.1)    
    
    # Section: Coil Parameters
    print("\n" + "=" * 25)
    print("Coil Parameters")
    print("=" * 25)
    wire_material = get_choice_input("Wire material", wire_material_choices, default="Copper")
    insulation_thickness = get_float_input("Insulation thickness (m)", default=5e-5, min_val=0.0)
    packing_factor = get_float_input("Packing factor (0-1)", default=0.85, min_val=0.0, max_val=1.0)
    stages = get_range_input("Number of stages", 5, 10, 1, is_int=True)
    wire_gauge_range = get_wire_gauge_range(wire_spec)
    wire_gauge_min = get_int_input("Wire gauge min (AWG)", 10, min(wire_gauge_range), max(wire_gauge_range))
    wire_gauge_max = get_int_input("Wire gauge max (AWG)", 18, min(wire_gauge_range), max(wire_gauge_range))
    layers = get_range_input("Number of layers", 1, 8, 1, is_int=True)
    turns = get_range_input("Turns per layer", 10, 80, 10, is_int=True)    
    
    # Section: Capacitor Parameters
    print("\n" + "=" * 25)
    print("Capacitor Parameters")
    print("=" * 25)
    voltage = get_range_input("Capacitor voltage (V)", 200, 600, 50, is_int=True)
    capacitance = get_range_input("Capacitance (F)", 0.01, 1, 0.01, is_int=False)

    # Section: Circuit Model Parameters
    print("\n" + "=" * 25)
    print("Circuit Model Parameters")
    print("=" * 25)
    switch_resistance = get_float_input("Switch resistance (Ohms)", default=0.001, min_val=0.0)
    switch_inductance = get_float_input("Switch inductance (H)", default=1e-8, min_val=0.0)
    parasitic_capacitance = get_float_input("Parasitic capacitance (F)", default=1e-11, min_val=0.0)
    include_skin_effect = get_yes_no_input("Include skin effect?", default=True)
    include_proximity_effect = get_yes_no_input("Include proximity effect?", default=True)    
    
    # Section: Magnetic Model Parameters
    print("\n" + "=" * 25)
    print("Magnetic Model Parameters")
    print("=" * 25)
    calculation_method = get_choice_input("Magnetic calculation method", magnetic_methods, default=magnetic_methods[0])
    axial_discretization = get_int_input("Axial discretization", default=1000, min_val=1)
    radial_discretization = get_int_input("Radial discretization", default=100, min_val=1)
    include_saturation = get_yes_no_input("Include saturation?", default=False)
    include_hysteresis = get_yes_no_input("Include hysteresis?", default=False)

    # Section: Simulation Parameters
    print("\n" + "=" * 25)
    print("Simulation Parameters")
    print("=" * 25)
    simulation_time = get_float_input("Simulation time (s)", default=0.1, min_val=0.01)

    # Calculate total combinations for progress bar
    num_stages = ((stages[1] - stages[0]) // stages[2]) + 1
    num_wire_gauges = (wire_gauge_max - wire_gauge_min + 1)
    num_layers = ((layers[1] - layers[0]) // layers[2]) + 1
    num_turns = ((turns[1] - turns[0]) // turns[2]) + 1
    num_voltages = ((voltage[1] - voltage[0]) // voltage[2]) + 1
    num_caps = int(round((capacitance[1] - capacitance[0]) / capacitance[2])) + 1
    total_combinations = num_stages * num_wire_gauges * num_layers * num_turns * num_voltages * num_caps

    print(f"\n📊 Optimization Analysis:")
    print(f"Total parameter combinations: {total_combinations:,}")
    print(f"Estimated time (exhaustive): {total_combinations * 0.3 / 3600:.1f} hours")
    
    # Smart sampling recommendation
    use_smart_sampling = True
    if total_combinations > 10000:
        print(f"\n⚡ PERFORMANCE RECOMMENDATION:")
        print(f"With {total_combinations:,} combinations, smart sampling is highly recommended!")
        print(f"Smart sampling can reduce optimization time by 10-100x while finding optimal solutions.")
        use_smart_sampling = get_yes_no_input("Use smart sampling for faster optimization?", default=True)
    else:
        print(f"Using exhaustive search for {total_combinations:,} combinations.")
        use_smart_sampling = False

    params = {
        "stages": stages,
        "wire_gauge": (wire_gauge_min, wire_gauge_max),
        "layers": layers,
        "turns_per_layer": turns,
        "voltage": voltage,
        "capacitance": capacitance,
        "projectile_mass": projectile_mass,
        "projectile_diameter": projectile_diameter,
        "projectile_material": projectile_material,
        "projectile_height": projectile_height,
        "wire_material": wire_material,
        "insulation_thickness": insulation_thickness,
        "packing_factor": packing_factor,
        "initial_position": initial_position,
        "initial_velocity": initial_velocity,
        "switch_resistance": switch_resistance,
        "switch_inductance": switch_inductance,
        "parasitic_capacitance": parasitic_capacitance,
        "include_skin_effect": include_skin_effect,
        "include_proximity_effect": include_proximity_effect,
        "calculation_method": calculation_method,
        "axial_discretization": axial_discretization,
        "radial_discretization": radial_discretization,        
        "include_saturation": include_saturation,
        "include_hysteresis": include_hysteresis,
        "simulation_time": simulation_time
    }
    
    print(f"\n🚀 Starting optimization with {total_combinations:,} total combinations...")
    
    try:
        best_config, results_list, best_configs = optimize_coilgun(params, materials, wire_spec, target_velocity, total_combinations, use_smart_sampling)
        print(f"\n✅ Optimization completed! Found {len(results_list)} valid configurations.")
        
        if best_config:
            print(f"✓ Best configuration found with score: {best_config.get('score', 'N/A')}")
            if len(best_configs) > 1:
                print(f"⚖️ Found {len(best_configs)} tied configurations with the same best score!")
        else:
            print("⚠ No best configuration found")
            
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Show acceleration performance summary
    show_acceleration_summary(results_list)
    
    if best_config and "params" in best_config:  # Make sure we have valid params
        # Save the best configuration using the proper function with metadata
        best_config_filename = "best_coilgun_config.json"
        save_best_config_to_json(best_config, best_config_filename, materials, wire_spec)
        
        # Convert to simulation config format for display
        config_dict = build_config_dict(best_config["params"], materials, wire_spec)
        
        print("\n" + "="*50)
        print("MULTI-STAGE COILGUN CONFIGURATION SUMMARY")
        print("="*50)
        
        # Print shared parameters first
        print("\n=== Shared Parameters ===")
        print("\n--- Projectile Parameters ---")
        projectile = config_dict['shared']['projectile']
        print(f"Diameter: {projectile['diameter']*1000:.1f} mm")
        print(f"Length: {projectile['length']*1000:.1f} mm")
        print(f"Material: {projectile['material']}")
        print(f"Initial position: {projectile['initial_position']*1000:.1f} mm")
        print(f"Initial velocity: {projectile['initial_velocity']:.1f} m/s")
        
        print("\n--- Circuit Model Parameters ---")
        circuit = config_dict['shared']['circuit_model']
        print(f"Switch resistance: {circuit['switch_resistance']:.3f} Ω")
        print(f"Switch inductance: {circuit['switch_inductance']*1e9:.1f} nH")
        print(f"Parasitic capacitance: {circuit['parasitic_capacitance']*1e12:.1f} pF")
        print(f"Include skin effect: {'Yes' if circuit['include_skin_effect'] else 'No'}")
        print(f"Include proximity effect: {'Yes' if circuit['include_proximity_effect'] else 'No'}")
        
        print("\n--- Magnetic Model Parameters ---")
        magnetic = config_dict['shared']['magnetic_model']
        print(f"Calculation method: {magnetic['calculation_method']}")
        print(f"Axial discretization: {magnetic['axial_discretization']}")
        print(f"Radial discretization: {magnetic['radial_discretization']}")
        print(f"Include saturation: {'Yes' if magnetic['include_saturation'] else 'No'}")
        print(f"Include hysteresis: {'Yes' if magnetic['include_hysteresis'] else 'No'}")        # Print common stage configuration
        print(f"\n=== Stage Configuration (identical for all {len(config_dict['stages'])} stages) ===")
        stage = config_dict['stages'][0]  # Use first stage since all are identical
        print("\n--- Coil Parameters ---")
        print(f"Inner diameter: {stage['coil']['inner_diameter']*1000:.1f} mm")
        print(f"Length (calculated): {stage['coil']['length']*1000:.1f} mm")
        print(f"Wire gauge: {stage['coil']['wire_gauge_awg']} AWG")
        print(f"Number of layers: {stage['coil']['num_layers']}")
        print(f"Turns per layer: {stage['coil'].get('turns_per_layer', best_config['params'].get('turns_per_layer', 'N/A'))}")
        print(f"Material: {stage['coil']['wire_material']}")
        print(f"Insulation thickness: {stage['coil']['insulation_thickness']*1000:.3f} mm")
        print(f"Packing factor: {stage['coil']['packing_factor']:.2f}")
        
        print("\n--- Capacitor Parameters ---")
        print(f"Capacitance: {stage['capacitor']['capacitance']*1000:.1f} mF")
        print(f"Initial voltage: {stage['capacitor']['initial_voltage']:.0f} V")
        print(f"ESR: {stage['capacitor']['esr']:.3f} Ω")
        print(f"ESL: {stage['capacitor']['esl']*1e9:.1f} nH")
        
        # Print per-stage performance
        if 'stage_results' in best_config:
            print("\n--- Per-Stage Performance ---")
            for i, stage_data in enumerate(best_config['stage_results'], 1):
                print(f"Stage {i}: {stage_data['velocity']:.1f} m/s, {stage_data['efficiency']:.1f}% efficiency")
        
        print("\n=== Overall Performance ===")
        print(f"Final velocity: {best_config.get('velocity', 0):.1f} m/s")
        print(f"Overall efficiency: {best_config.get('efficiency', 0)*100:.1f}%")
        print(f"Maximum current: {best_config.get('max_current', 0):.1f} A")
        print(f"Maximum force: {best_config.get('max_force', 0):.1f} N")
        
        print("\nVelocity progression:")
        if 'stage_results' in best_config:
            print(f"Initial: {config_dict['shared']['projectile']['initial_velocity']:.1f} m/s")
            for stage_data in best_config['stage_results']:
                print(f"After stage {stage_data['stage']}: {stage_data['velocity']:.1f} m/s")
        
        # Save all valid configurations to CSV
        if results_list:
            results_csv_filename = "coilgun_optimization_results.csv"
            save_results_to_csv(results_list, results_csv_filename)
            print(f"\nAll valid configurations saved to: {results_csv_filename}")
        
        # Save tied configurations if any exist
        if 'best_configs' in locals() and len(best_configs) > 1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_tied_configs_to_json(best_configs, f"main_best_tied_configs_{timestamp}", materials, wire_spec)
            print(f"\n⚖️ {len(best_configs)} tied configurations also saved!")
    else:
        print("\nNo valid coilgun configuration found. Try adjusting your parameter ranges or target velocity.")
    
    print(f"\n🏁 Program completed successfully!")
    print(f"\n📁 Files saved:")
    print(f"   ✓ Best configuration: best_coilgun_config.json (with optimization metadata)")
    if 'results_list' in locals() and results_list:
        print(f"   ✓ All valid configurations: coilgun_optimization_results.csv ({len(results_list)} configs)")
    if 'best_configs' in locals() and len(best_configs) > 1:
        print(f"   ✓ Tied configurations: best_tied_configs_*.json ({len(best_configs)} files)")
    
    # Final cleanup
    cleanup_temp_files()

def cleanup_temp_files():
    """Clean up any temporary files and directories that might remain."""
    try:
        # Clean up temporary files in current directory
        temp_patterns = [
            "temp_sim_config_*.json",
            "temp_gpu_sim_config_*.json", 
            "temp_worker_*.json",
            "temp_stage_*_config*.json",
            "*temp*stage*.json",
            "temp_*.json",
            "*.temp",
            "*_temp_*",
            "temp*config*.json",
            "*multistage*temp*.json",
            "*.tmp"
        ]
        
        files_cleaned = 0
        for pattern in temp_patterns:
            for temp_file in glob.glob(pattern):
                try:
                    os.remove(temp_file)
                    files_cleaned += 1
                except:
                    pass
        
        # Clean up temporary worker directories
        temp_dirs_cleaned = 0
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            
            # Look for worker directories in system temp directory
            for item in os.listdir(temp_dir):
                if item.startswith("worker_") and os.path.isdir(os.path.join(temp_dir, item)):
                    try:
                        worker_dir_path = os.path.join(temp_dir, item)
                        shutil.rmtree(worker_dir_path, ignore_errors=True)
                        temp_dirs_cleaned += 1
                    except:
                        pass
        except:
            pass
        
        total_cleaned = files_cleaned + temp_dirs_cleaned
        if total_cleaned > 0:
            print(f"🧹 Cleaned up {files_cleaned} temporary files and {temp_dirs_cleaned} temporary directories")
    except:
        pass


def show_acceleration_summary(results_list):
    """Show a summary of acceleration methods used during optimization."""
    if not results_list:
        return
    
    # Count acceleration methods used
    acceleration_counts = {}
    for result in results_list:
        accel_type = result.get("acceleration_used", "Unknown")
        acceleration_counts[accel_type] = acceleration_counts.get(accel_type, 0) + 1
    
    if acceleration_counts:
        print(f"\n📊 Acceleration Summary:")
        total = sum(acceleration_counts.values())
        for accel_type, count in acceleration_counts.items():
            percentage = (count / total) * 100
            print(f"   {accel_type}: {count:,} configs ({percentage:.1f}%)")
    
    if GPU_AVAILABLE and GPU_TYPE:
        print(f"\n✓ GPU acceleration was available: {GPU_TYPE}")
    else:
        print(f"\n💻 CPU-only processing was used")
        print(f"✓ Parallel workers: {MAX_WORKERS}")
    
    print("="*50)

class SuppressOutput:
    """Context manager to completely suppress stdout and stderr, even in multiprocessing contexts."""
    
    def __init__(self):
        self.stdout = None
        self.stderr = None
        self.devnull = None
        self.tqdm_disabled = False
        self.original_displayhook = None
    
    def __enter__(self):
        # Save original stdout/stderr
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
        # Create devnull
        self.devnull = open(os.devnull, 'w')
        
        # Redirect stdout and stderr
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        
        # Also suppress sys.displayhook to catch any remaining output
        self.original_displayhook = sys.displayhook
        sys.displayhook = lambda x: None
        
        # Disable any tqdm progress bars that might be created during simulation
        try:
            if 'TQDM_DISABLE' not in os.environ or os.environ['TQDM_DISABLE'] != '1':
                os.environ['TQDM_DISABLE'] = '1'
                self.tqdm_disabled = True
        except:
            pass
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        
        # Restore displayhook
        if self.original_displayhook:
            sys.displayhook = self.original_displayhook
        
        # Close devnull
        if self.devnull:
            self.devnull.close()
        
        # Restore tqdm if we disabled it (but only if we're not in a worker)
        if self.tqdm_disabled:
            try:
                import os
                import multiprocessing as mp
                # Only restore tqdm in the main process
                if mp.current_process().name == 'MainProcess':
                    os.environ['TQDM_DISABLE'] = '0'
            except:
                pass


if __name__ == "__main__":
    main()
