"""
Coilgun Optimization Script

This script iterates through several iterations of possible coilgun designs and configurations
to optimize coilgun performance. The script will output a csv file of all valid configurations
along with a config.json file to be used in the electromagnetic coilgun simulation.
"""

import json
import numpy as np
import csv
import os
from typing import Dict, Any, List, Tuple
from solve import CoilgunSimulation, MultiStageCoilgunSimulation  # Make sure solve.py is in your PYTHONPATH

# Check if tqdm is available for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def load_material_data() -> Dict[str, Any]:
    """Load material properties from materials.json if available, otherwise use defaults."""
    try:
        if not os.path.exists("materials.json"):
            print("Warning: materials.json not found. Using basic material properties...")
            return create_default_materials()
        with open("materials.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading materials: {e}")
        print("Using basic material properties...")
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


def build_config_dict(params, materials, wire_spec):
    """Build a simulation config dictionary for a given parameter set."""
    # Handle both tuple/list and integer values for stages parameter
    stages_param = params["stages"]
    num_stages = stages_param[1] if isinstance(stages_param, (list, tuple)) else stages_param
    
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
                "max_step": 1e-6,
                "tolerance": 1e-9,
                "method": "RK45"
            },
            "circuit_model": {
                "switch_resistance": params["switch_resistance"],
                "switch_inductance": params["switch_inductance"],
                "parasitic_capacitance": params["parasitic_capacitance"],
                "include_skin_effect": params["include_skin_effect"],
                "include_proximity_effect": params["include_proximity_effect"]
            },
            "magnetic_model": {
                "calculation_method": params["calculation_method"],
                "axial_discretization": params["axial_discretization"],
                "radial_discretization": params["radial_discretization"],
                "include_saturation": params["include_saturation"],
                "include_hysteresis": params["include_hysteresis"]
            },
            "output": {
                "save_trajectory": True,
                "save_current_profile": True,
                "save_field_data": False,
                "print_progress": False,
                "save_interval": 100,
                "suppress_init_output": True  # Suppress physics initialization output during optimization
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
            "wire_gauge_awg": params["wire_gauge"],
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


def simulate_and_score(params, materials, wire_spec, target_velocity):
    """
    Run a full multi-stage simulation for the given parameters and return performance metrics.
    """
    config = build_config_dict(params, materials, wire_spec)
    temp_config_file = "temp_sim_config.json"
    try:
        with open(temp_config_file, "w") as f:
            json.dump(config, f, indent=4)
        sim = MultiStageCoilgunSimulation(temp_config_file)
        results = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
          # Extract the key metrics from results
        final_velocity = float(results.get('final_velocity_ms', 0))
        max_current = float(results.get('max_current_A', 0))
        max_force = float(results.get('max_force_N', 0))
        overall_efficiency = float(results.get('overall_efficiency_percent', 0))
        
        score = (params["voltage"] + params["capacitance"] * 1000 +
                config["stages"][0]["coil"]["num_layers"] * 100 + max_current * 0.1)
        
        valid = final_velocity >= target_velocity
        
        # Process stage-specific results
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
            "params": params.copy()
        }
    except Exception as e:
        return {"valid": False, "error": str(e), "params": params.copy()}
    finally:
        # Always try to remove the temp file
        try:
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
        except Exception:
            pass

def optimize_coilgun(params, materials, wire_spec, target_velocity, total_combinations):
    """
    Iterate through all parameter combinations, run full simulation for each,
    and find the best configuration. Shows a progress bar.
    """
    best_config = None
    best_score = float('inf')
    results_list = []
    stages_min, stages_max, stages_step = params["stages"]
    wire_gauge_min, wire_gauge_max = params["wire_gauge"]
    layers_min, layers_max, layers_step = params["layers"]
    turns_min, turns_max, turns_step = params["turns_per_layer"]
    voltage_min, voltage_max, voltage_step = params["voltage"]
    cap_min, cap_max, cap_step = params["capacitance"]

    # Optimization header
    print("\n" + "=" * 25)
    print("Optimization Process")
    print("=" * 25 + "\n")
    
    # Progress bar setup    
    if TQDM_AVAILABLE:
        pbar = tqdm(total=total_combinations, desc="Optimizing")
    else:
        pbar = None
        progress_count = 0

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
                            
                            sim_result = simulate_and_score(candidate, materials, wire_spec, target_velocity)
                            if sim_result["valid"]:
                                # Store only the simulation metrics, not the raw results
                                sim_metrics = {
                                    "velocity": sim_result["velocity"],
                                    "max_current": sim_result["max_current"],
                                    "max_force": sim_result["max_force"],
                                    "efficiency": sim_result["efficiency"],
                                    "stage_results": sim_result["stage_results"]
                                }
                                results_list.append({**candidate, **sim_metrics})
                                if sim_result["score"] < best_score:
                                    best_score = sim_result["score"]
                                    best_config = sim_result
                            if TQDM_AVAILABLE:
                                pbar.update(1)
                            else:
                                progress_count += 1
                                if progress_count % max(1, total_combinations // 100) == 0:
                                    percent = (progress_count / total_combinations) * 100
                                    print(f"Progress: {percent:.1f}%")
    if TQDM_AVAILABLE:
        pbar.close()
    return best_config, results_list

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

def main():
    """
    Main function to prompt user, run optimization, and save results.
    """
    print("=" * 50)
    print("COILGUN OPTIMIZATION")
    print("=" * 50)
    print("\nWelcome to the Coilgun Optimization Tool!")
    print("This tool will help you find the optimal configuration for your coilgun design.")
    print("Please follow the prompts to enter your design parameters.")

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
    initial_position = get_float_input("Projectile initial position (m)", default=-0.05)
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
    voltage = get_range_input("Capacitor voltage (V)", 100, 600, 50, is_int=True)
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
    best_config, results_list = optimize_coilgun(params, materials, wire_spec, target_velocity, total_combinations)
    
    if best_config and "params" in best_config:  # Make sure we have valid params
        # Convert to simulation config format
        config_dict = build_config_dict(best_config["params"], materials, wire_spec)
        best_config_filename = "best_coilgun_config.json"
        with open(best_config_filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
            print(f"\nBest configuration saved to: {best_config_filename}")
        
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
        print(f"Include hysteresis: {'Yes' if magnetic['include_hysteresis'] else 'No'}")

        # Print common stage configuration        print(f"\n=== Stage Configuration (identical for all {len(config_dict['stages'])} stages) ===")
        stage = config_dict['stages'][0]  # Use first stage since all are identical
        print("\n--- Coil Parameters ---")
        print(f"Inner diameter: {stage['coil']['inner_diameter']*1000:.1f} mm")
        print(f"Length (calculated): {stage['coil']['length']*1000:.1f} mm")
        print(f"Projectile length: {config_dict['shared']['projectile']['length']*1000:.1f} mm")
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
        
        # Print per-stage velocity and efficiency results
        if 'stage_results' in best_config:
            print("\n--- Per-Stage Performance ---")
            for i, stage_data in enumerate(best_config['stage_results'], 1):
                print(f"Stage {i}: {stage_data['velocity']:.1f} m/s, {stage_data['efficiency']:.1f}% efficiency")
          # Circuit and Magnetic model parameters are already shown in Shared Parameters section

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
    else:
        print("\nNo valid coilgun configuration found. Try adjusting your parameter ranges or target velocity.")

if __name__ == "__main__":
    main()