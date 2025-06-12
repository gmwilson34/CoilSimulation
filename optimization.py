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
from solve import CoilgunSimulation  # Make sure solve.py is in your PYTHONPATH

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
        "magnetic_methods": ["biot_savart"]
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
        default_str = "y" if default else "n" if default is not None else ""
        user_input = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
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
    config = {
        "coil": {
            "inner_diameter": params["projectile_diameter"] * 1.05,
            "length": params["projectile_height"],
            "wire_gauge_awg": params["wire_gauge"],
            "num_layers": params["layers"],
            "wire_material": params["wire_material"],
            "insulation_thickness": params["insulation_thickness"],
            "packing_factor": params["packing_factor"]
        },
        "projectile": {
            "diameter": params["projectile_diameter"],
            "length": params["projectile_height"],
            "material": params["projectile_material"],
            "initial_position": params["initial_position"],
            "initial_velocity": params["initial_velocity"]
        },
        "capacitor": {
            "capacitance": params["capacitance"],
            "initial_voltage": params["voltage"],
            "esr": 0.01,
            "esl": 5e-8
        },
        "simulation": {
            "time_span": [0, 0.02],
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
            "save_trajectory": False,
            "save_current_profile": False,
            "save_field_data": False,
            "print_progress": False,
            "save_interval": 100
        }
    }
    return config

def simulate_and_score(params, materials, wire_spec, target_velocity):
    """
    Run a full simulation for the given parameters and return performance metrics.
    """
    config = build_config_dict(params, materials, wire_spec)
    temp_config_file = "temp_sim_config.json"
    try:
        with open(temp_config_file, "w") as f:
            json.dump(config, f, indent=4)
        sim = CoilgunSimulation(temp_config_file)
        results = sim.run_simulation(save_data=False, verbose=False)
        velocity = results.get('final_velocity', results.get('final_velocity_ms', 0))
        resistance = getattr(sim.physics, 'coil_resistance', None)
        inductance = getattr(sim.physics, 'coil_inductance', None)
        max_current = results.get('max_current', results.get('max_current_A', None))
        score = (params["voltage"] + params["capacitance"] * 1000 +
                 (resistance or 0) * 100 + (inductance or 0) * 100)
        valid = velocity >= target_velocity
        return {
            "velocity": velocity,
            "resistance": resistance,
            "inductance": inductance,
            "max_current": max_current,
            "score": score,
            "valid": valid,
            "results": results,
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
                                results_list.append({**candidate, **sim_result["results"]})
                                if sim_result["score"] < best_score:
                                    best_score = sim_result["score"]
                                    best_config = {**candidate, **sim_result["results"]}
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
    print("=== COILGUN OPTIMIZATION ===")

    # Load materials and wire specs
    materials = load_material_data()
    wire_spec = materials["wire_specifications"]
    material_choices = [k for k in materials["materials"].keys() if "density" in materials["materials"][k]]
    wire_material_choices = [k for k in materials["materials"].keys() if "resistivity" in materials["materials"][k]]
    magnetic_methods = materials.get("magnetic_methods", ["biot_savart"])

    # --- Section: Projectile Parameters ---
    print("\n--- Projectile Parameters ---")
    projectile_material = get_choice_input("Projectile material", material_choices, default="Low_Carbon_Steel")
    projectile_mass = get_float_input("Projectile mass (kg)", default=1.0, min_val=0.01)
    projectile_diameter = get_float_input("Projectile diameter (m)", default=0.012, min_val=0.001)
    density = materials["materials"][projectile_material]["density"]
    projectile_height = calculate_projectile_height(projectile_mass, projectile_diameter, density)
    print(f"Calculated projectile height: {projectile_height:.4f} m")
    initial_position = get_float_input("Projectile initial position (m)", default=-0.05)
    initial_velocity = get_float_input("Projectile initial velocity (m/s)", default=0.0)
    target_velocity = get_float_input("Target velocity (m/s)", default=105.0, min_val=0.1)

    # --- Section: Coil Parameters ---
    print("\n--- Coil Parameters ---")
    wire_material = get_choice_input("Wire material", wire_material_choices, default="Copper")
    insulation_thickness = get_float_input("Insulation thickness (m)", default=5e-5, min_val=0.0)
    packing_factor = get_float_input("Packing factor (0-1)", default=0.85, min_val=0.0, max_val=1.0)
    stages = get_range_input("Number of stages", 6, 10, 1, is_int=True)
    wire_gauge_range = get_wire_gauge_range(wire_spec)
    wire_gauge_min = get_int_input("Wire gauge min (AWG)", 14, min(wire_gauge_range), max(wire_gauge_range))
    wire_gauge_max = get_int_input("Wire gauge max (AWG)", 18, min(wire_gauge_range), max(wire_gauge_range))
    layers = get_range_input("Number of layers", 1, 6, 1, is_int=True)
    turns = get_range_input("Turns per layer", 10, 100, 10, is_int=True)

    # --- Section: Capacitor Parameters ---
    print("\n--- Capacitor Parameters ---")
    voltage = get_range_input("Capacitor voltage (V)", 200, 600, 50, is_int=True)
    capacitance = get_range_input("Capacitance (F)", 0.01, 0.1, 0.01, is_int=False)

    # --- Section: Circuit Model Parameters ---
    print("\n--- Circuit Model Parameters ---")
    switch_resistance = get_float_input("Switch resistance (Ohms)", default=0.001, min_val=0.0)
    switch_inductance = get_float_input("Switch inductance (H)", default=1e-8, min_val=0.0)
    parasitic_capacitance = get_float_input("Parasitic capacitance (F)", default=1e-11, min_val=0.0)
    include_skin_effect = get_yes_no_input("Include skin effect?", default=False)
    include_proximity_effect = get_yes_no_input("Include proximity effect?", default=False)

    # --- Section: Magnetic Model Parameters ---
    print("\n--- Magnetic Model Parameters ---")
    calculation_method = get_choice_input("Magnetic calculation method", magnetic_methods, default=magnetic_methods[0])
    axial_discretization = get_int_input("Axial discretization", default=1000, min_val=1)
    radial_discretization = get_int_input("Radial discretization", default=100, min_val=1)
    include_saturation = get_yes_no_input("Include saturation?", default=False)
    include_hysteresis = get_yes_no_input("Include hysteresis?", default=False)

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
        "include_hysteresis": include_hysteresis
    }
    best_config, results_list = optimize_coilgun(params, materials, wire_spec, target_velocity, total_combinations)
    # Save best configuration to JSON
    if best_config:
        best_config_filename = "best_coilgun_config.json"
        with open(best_config_filename, 'w') as f:
            json.dump(best_config, f, indent=4)
        print(f"Best configuration saved to: {best_config_filename}")
    # Save all valid configurations to CSV
    if results_list:
        results_csv_filename = "coilgun_optimization_results.csv"
        save_results_to_csv(results_list, results_csv_filename)
        print(f"All valid configurations saved to: {results_csv_filename}")
    # Print best configuration summary
    if best_config:
        print("\n=== Best Coilgun Configuration Found ===")
        for k, v in best_config.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        print("\nNo valid coilgun configuration found. Try adjusting your parameter ranges or target velocity.")

if __name__ == "__main__":
    main()