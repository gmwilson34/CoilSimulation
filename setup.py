"""
Coilgun Simulation Setup Script

This script collects user inputs and generates appropriate JSON configuration files
for the electromagnetic coilgun simulation.
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple


def load_material_data() -> Dict[str, Any]:
    """Load material properties from materials.json"""
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
    """Create basic material properties if materials.json is not available"""
    return {
        "materials": {
            "Copper": {"description": "Copper wire", "resistivity": 1.68e-8, "mu_r": 0.999991},
            "Low_Carbon_Steel": {"description": "Steel projectile", "mu_r": 1000, "density": 7850},
            "Pure_Iron": {"description": "Iron projectile", "mu_r": 5000, "density": 7874},
            "Aluminum": {"description": "Aluminum projectile", "mu_r": 1.000022, "density": 2700}
        },
        "wire_specifications": {
            "awg_diameter_mm": {"14": 1.628, "16": 1.291, "18": 1.024, "20": 0.812},
            "current_capacity_A": {"14": 32, "16": 22, "18": 16, "20": 11}
        }
    }


def get_float_input(prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
    """Get validated float input from user"""
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
    """Get validated integer input from user"""
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


def get_choice_input(prompt: str, choices: List[str], default: str = None) -> str:
    """Get validated choice input from user"""
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
    """Get yes/no input from user"""
    while True:
        if default is not None:
            default_str = "Y/n" if default else "y/N"
            user_input = input(f"{prompt} ({default_str}): ").strip().lower()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt} (y/n): ").strip().lower()
        
        if user_input in ['y', 'yes', 'true', '1']:
            return True
        elif user_input in ['n', 'no', 'false', '0']:
            return False
        else:
            print("Please enter 'y' or 'n'")


def setup_coil_parameters(materials: Dict[str, Any]) -> Dict[str, Any]:
    """Collect coil configuration parameters"""
    print("\n" + "="*50)
    print("COIL CONFIGURATION")
    print("="*50)
    
    # Available wire materials
    wire_materials = [mat for mat in materials["materials"].keys() 
                     if "conductivity" in materials["materials"][mat] or mat == "Copper"]
    
    # Available wire gauges
    wire_gauges = list(materials["wire_specifications"]["awg_diameter_mm"].keys())
    
    return {
        "inner_diameter": get_float_input(
            "Coil inner diameter (m)", 
            default=0.015, min_val=0.001, max_val=0.1
        ),
        "length": get_float_input(
            "Coil length (m)", 
            default=0.075, min_val=0.01, max_val=0.5
        ),
        "wire_gauge_awg": int(get_choice_input(
            "Wire gauge (AWG)", 
            wire_gauges, 
            default="16"
        )),
        "num_layers": get_int_input(
            "Number of wire layers", 
            default=6, min_val=1, max_val=20
        ),
        "wire_material": get_choice_input(
            "Wire material", 
            wire_materials, 
            default="Copper"
        ),
        "insulation_thickness": get_float_input(
            "Wire insulation thickness (m)", 
            default=5e-5, min_val=1e-6, max_val=1e-3
        ),
        "packing_factor": get_float_input(
            "Wire packing factor (0-1)", 
            default=0.85, min_val=0.5, max_val=0.95
        )
    }


def setup_projectile_parameters(materials: Dict[str, Any]) -> Dict[str, Any]:
    """Collect projectile configuration parameters"""
    print("\n" + "="*50)
    print("PROJECTILE CONFIGURATION")
    print("="*50)
    
    # Available projectile materials (magnetic materials)
    projectile_materials = [mat for mat in materials["materials"].keys() 
                           if "mu_r" in materials["materials"][mat] and 
                           materials["materials"][mat]["mu_r"] > 10]
    
    if not projectile_materials:
        projectile_materials = ["Low_Carbon_Steel", "Pure_Iron", "Silicon_Steel"]
    
    return {
        "diameter": get_float_input(
            "Projectile diameter (m)", 
            default=0.012, min_val=0.001, max_val=0.05
        ),
        "length": get_float_input(
            "Projectile length (m)", 
            default=0.025, min_val=0.005, max_val=0.1
        ),
        "material": get_choice_input(
            "Projectile material", 
            projectile_materials, 
            default="Low_Carbon_Steel"
        ),
        "initial_position": get_float_input(
            "Initial position relative to coil center (m)", 
            default=-0.05, min_val=-0.2, max_val=0.0
        ),
        "initial_velocity": get_float_input(
            "Initial velocity (m/s)", 
            default=0.0, min_val=0.0, max_val=100.0
        )
    }


def setup_capacitor_parameters() -> Dict[str, Any]:
    """Collect capacitor configuration parameters"""
    print("\n" + "="*50)
    print("CAPACITOR CONFIGURATION")
    print("="*50)
    
    return {
        "capacitance": get_float_input(
            "Capacitance (F)", 
            default=0.0033, min_val=1e-6, max_val=0.1
        ),
        "initial_voltage": get_float_input(
            "Initial voltage (V)", 
            default=360.0, min_val=10.0, max_val=1000.0
        ),
        "esr": get_float_input(
            "Equivalent series resistance (Ω)", 
            default=0.01, min_val=0.001, max_val=1.0
        ),
        "esl": get_float_input(
            "Equivalent series inductance (H)", 
            default=5e-8, min_val=1e-9, max_val=1e-6
        )
    }


def setup_simulation_parameters() -> Dict[str, Any]:
    """Collect simulation configuration parameters"""
    print("\n" + "="*50)
    print("SIMULATION CONFIGURATION")
    print("="*50)
    
    time_end = get_float_input(
        "Simulation end time (s)", 
        default=0.02, min_val=0.001, max_val=1.0
    )
    
    methods = ["RK45", "RK23", "Radau", "BDF", "LSODA"]
    
    return {
        "time_span": [0, time_end],
        "max_step": get_float_input(
            "Maximum time step (s)", 
            default=1e-6, min_val=1e-9, max_val=1e-3
        ),
        "tolerance": get_float_input(
            "Solver tolerance", 
            default=1e-9, min_val=1e-12, max_val=1e-6
        ),
        "method": get_choice_input(
            "Integration method", 
            methods, 
            default="RK45"
        )
    }


def setup_circuit_model_parameters() -> Dict[str, Any]:
    """Collect circuit model configuration parameters"""
    print("\n" + "="*50)
    print("CIRCUIT MODEL CONFIGURATION")
    print("="*50)
    
    return {
        "switch_resistance": get_float_input(
            "Switch resistance (Ω)", 
            default=0.001, min_val=1e-6, max_val=1.0
        ),
        "switch_inductance": get_float_input(
            "Switch inductance (H)", 
            default=1e-8, min_val=1e-12, max_val=1e-6
        ),
        "parasitic_capacitance": get_float_input(
            "Parasitic capacitance (F)", 
            default=1e-11, min_val=1e-15, max_val=1e-9
        ),
        "include_skin_effect": get_yes_no_input(
            "Include skin effect?", 
            default=False
        ),
        "include_proximity_effect": get_yes_no_input(
            "Include proximity effect?", 
            default=False
        )
    }


def setup_magnetic_model_parameters() -> Dict[str, Any]:
    """Collect magnetic model configuration parameters"""
    print("\n" + "="*50)
    print("MAGNETIC MODEL CONFIGURATION")
    print("="*50)
    
    methods = ["biot_savart", "finite_element", "analytical"]
    
    return {
        "calculation_method": get_choice_input(
            "Magnetic field calculation method", 
            methods, 
            default="biot_savart"
        ),
        "axial_discretization": get_int_input(
            "Axial discretization points", 
            default=1000, min_val=100, max_val=10000
        ),
        "radial_discretization": get_int_input(
            "Radial discretization points", 
            default=100, min_val=10, max_val=1000
        ),
        "include_saturation": get_yes_no_input(
            "Include magnetic saturation?", 
            default=False
        ),
        "include_hysteresis": get_yes_no_input(
            "Include magnetic hysteresis?", 
            default=False
        )
    }


def setup_output_parameters() -> Dict[str, Any]:
    """Collect output configuration parameters"""
    print("\n" + "="*50)
    print("OUTPUT CONFIGURATION")
    print("="*50)
    
    return {
        "save_trajectory": get_yes_no_input(
            "Save projectile trajectory?", 
            default=True
        ),
        "save_current_profile": get_yes_no_input(
            "Save current profile?", 
            default=True
        ),
        "save_field_data": get_yes_no_input(
            "Save magnetic field data?", 
            default=False
        ),
        "print_progress": get_yes_no_input(
            "Print simulation progress?", 
            default=True
        ),
        "save_interval": get_int_input(
            "Data save interval (steps)", 
            default=100, min_val=1, max_val=10000
        )
    }


def create_configuration() -> Dict[str, Any]:
    """Create complete configuration by collecting all parameters"""
    print("Welcome to the Coilgun Simulation Setup!")
    print("This script will help you create a configuration file for your simulation.")
    
    # Load materials
    materials = load_material_data()
    
    # Collect all configuration sections
    config = {
        "coil": setup_coil_parameters(materials),
        "projectile": setup_projectile_parameters(materials),
        "capacitor": setup_capacitor_parameters(),
        "simulation": setup_simulation_parameters(),
        "circuit_model": setup_circuit_model_parameters(),
        "magnetic_model": setup_magnetic_model_parameters(),
        "output": setup_output_parameters()
    }
    
    return config


def save_configuration(config: Dict[str, Any], filename: str) -> None:
    """Save configuration to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\nConfiguration saved to: {filename}")
    except Exception as e:
        print(f"Error saving configuration: {e}")


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("COILGUN SIMULATION SETUP")
    print("="*60)
    
    # Create configuration
    config = create_configuration()
    
    # Get filename for saving
    print("\n" + "="*50)
    print("SAVE CONFIGURATION")
    print("="*50)
    
    default_filename = "my_coilgun_config.json"
    filename = input(f"Enter filename for configuration (default: {default_filename}): ").strip()
    if not filename:
        filename = default_filename
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Save configuration
    save_configuration(config, filename)
    
    # Display summary
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Coil: {config['coil']['inner_diameter']*1000:.1f}mm ID, {config['coil']['length']*1000:.1f}mm length")
    print(f"Wire: AWG {config['coil']['wire_gauge_awg']}, {config['coil']['num_layers']} layers")
    print(f"Projectile: {config['projectile']['diameter']*1000:.1f}mm × {config['projectile']['length']*1000:.1f}mm {config['projectile']['material']}")
    print(f"Capacitor: {config['capacitor']['capacitance']*1000:.1f}mF @ {config['capacitor']['initial_voltage']:.0f}V")
    print(f"Energy: {0.5 * config['capacitor']['capacitance'] * config['capacitor']['initial_voltage']**2:.1f}J")
    
    print(f"\nTo run the simulation:")
    print(f"python solve.py {filename}")
    print(f"\nTo visualize results:")
    print(f"python view.py {filename}")


if __name__ == "__main__":
    main()

