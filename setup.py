"""
Coilgun Simulation Setup Script

This script collects user inputs and generates appropriate JSON configuration files
for the electromagnetic coilgun simulation. Supports both single-stage and multi-stage
coilgun configurations with advanced physics modeling.
"""

import json
import os
import sys
import signal
from typing import Dict, Any, List, Tuple, Optional


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


def get_float_input(prompt: str, default: Optional[float] = None, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
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
        except KeyboardInterrupt:
            print("\nSetup cancelled by user.")
            sys.exit(0)


def get_int_input(prompt: str, default: Optional[int] = None, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
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
        except KeyboardInterrupt:
            print("\nSetup cancelled by user.")
            sys.exit(0)


def get_choice_input(prompt: str, choices: List[str], default: Optional[str] = None) -> str:
    """Get validated choice input from user"""
    while True:
        try:
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
        except KeyboardInterrupt:
            print("\nSetup cancelled by user.")
            sys.exit(0)


def get_yes_no_input(prompt: str, default: Optional[bool] = None) -> bool:
    """Get yes/no input from user"""
    while True:
        try:
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
        except KeyboardInterrupt:
            print("\nSetup cancelled by user.")
            sys.exit(0)


def get_stage_grouping(num_stages: int) -> List[List[int]]:
    """Get how stages should be grouped for configuration sharing"""
    print(f"\nStage Grouping Configuration")
    print("="*50)
    print(f"You have {num_stages} stages. You can group stages to share the same coil configuration.")
    print("For example, with 6 stages you could group them as:")
    print("  - Group 1: Stages 1-3 (same config)")
    print("  - Group 2: Stages 4-6 (same config)")
    print("  - Or all different configurations")
    print("  - Or all the same configuration")
    
    all_same = get_yes_no_input("Do you want all stages to use the same coil configuration?", default=True)
    
    if all_same:
        return [list(range(1, num_stages + 1))]
    
    # Manual grouping
    groups = []
    remaining_stages = set(range(1, num_stages + 1))
    group_num = 1
    
    while remaining_stages:
        print(f"\nRemaining stages: {sorted(remaining_stages)}")
        print(f"Defining group {group_num}:")
        
        # Get start and end of this group
        start_stage = get_int_input(
            f"Starting stage for group {group_num}", 
            min_val=min(remaining_stages), 
            max_val=max(remaining_stages)
        )
        
        if start_stage not in remaining_stages:
            print(f"Stage {start_stage} is already assigned to a group.")
            continue
        
        end_stage = get_int_input(
            f"Ending stage for group {group_num} (inclusive)", 
            default=start_stage,
            min_val=start_stage, 
            max_val=max(remaining_stages)
        )
        
        # Validate the range
        group_stages = []
        for stage in range(start_stage, end_stage + 1):
            if stage in remaining_stages:
                group_stages.append(stage)
                remaining_stages.remove(stage)
            else:
                print(f"Warning: Stage {stage} is already assigned to a group.")
        
        if group_stages:
            groups.append(group_stages)
            print(f"Group {group_num}: Stages {group_stages}")
            group_num += 1
        
        if remaining_stages:
            continue_grouping = get_yes_no_input("Continue grouping remaining stages?", default=True)
            if not continue_grouping:
                # Put remaining stages in individual groups
                for stage in remaining_stages:
                    groups.append([stage])
                break
    
    return groups


def setup_shared_settings() -> List[str]:
    """Determine which settings should be shared across all stages"""
    print("\n" + "="*50)
    print("SHARED SETTINGS CONFIGURATION")
    print("="*50)
    print("Some settings can be shared across all stages to save time.")
    
    shared_settings = []
    
    setting_options = [
        ("simulation", "Simulation parameters (time span, tolerance, method)"),
        ("circuit_model", "Circuit model parameters (switch resistance, parasitic effects)"),
        ("magnetic_model", "Magnetic model parameters (calculation method, discretization)"),
        ("output", "Output parameters (what data to save)"),
        ("capacitor", "Capacitor parameters (capacitance, voltage, ESR)")
    ]
    
    for setting_key, description in setting_options:
        if get_yes_no_input(f"Share {description} across all stages?", default=True):
            shared_settings.append(setting_key)
    
    return shared_settings


def setup_coil_parameters(materials: Dict[str, Any], stage_info: str = "") -> Dict[str, Any]:
    """Collect coil configuration parameters"""
    print("\n" + "="*50)
    print(f"COIL CONFIGURATION{stage_info}")
    print("="*50)
    
    # Available wire materials
    wire_materials = [mat for mat in materials["materials"].keys() 
                     if "conductivity" in materials["materials"][mat] or mat == "Copper"]
    
    # Available wire gauges
    wire_gauges = list(materials["wire_specifications"]["awg_diameter_mm"].keys())
    
    return {
        "inner_diameter": get_float_input(
            "Coil inner diameter (m)", 
            default=0.015, min_val=0.001, max_val=0.5
        ),
        "length": get_float_input(
            "Coil length (m)", 
            default=0.075, min_val=0.01, max_val= 15.0
        ),
        "wire_gauge_awg": get_choice_input(
            "Wire gauge (AWG)", 
            wire_gauges, 
            default="16"
        ),
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
    """Collect projectile configuration parameters (shared across all stages)"""
    print("\n" + "="*50)
    print("PROJECTILE CONFIGURATION")
    print("="*50)
    print("NOTE: Projectile properties are shared across all stages.")
    
    # Available projectile materials (magnetic materials)
    projectile_materials = [mat for mat in materials["materials"].keys() 
                           if "mu_r" in materials["materials"][mat] and 
                           materials["materials"][mat]["mu_r"] > 10]
    
    if not projectile_materials:
        projectile_materials = ["Low_Carbon_Steel", "Pure_Iron", "Silicon_Steel"]
    
    return {
        "diameter": get_float_input(
            "Projectile diameter (m)", 
            default=0.012, min_val=0.001, max_val=0.5
        ),
        "length": get_float_input(
            "Projectile length (m)", 
            default=0.025, min_val=0.005, max_val=0.5
        ),
        "material": get_choice_input(
            "Projectile material", 
            projectile_materials, 
            default="Pure_Iron"
        ),
        "initial_position": get_float_input(
            "Initial position relative to first coil center (m)", 
            default=-0.05, min_val=-0.2, max_val=0.0
        ),
        "initial_velocity": get_float_input(
            "Initial velocity (m/s)", 
            default=0.0, min_val=0.0, max_val=100.0
        )
    }


def setup_capacitor_parameters(stage_info: str = "") -> Dict[str, Any]:
    """Collect capacitor configuration parameters"""
    print("\n" + "="*50)
    print(f"CAPACITOR CONFIGURATION{stage_info}")
    print("="*50)
    
    return {
        "capacitance": get_float_input(
            "Capacitance (F)", 
            default=0.0033, min_val=1e-6, max_val=1.0
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
    print("Configure the electromagnetic physics engine for accurate coilgun simulation.")
    
    # Enhanced calculation methods from equations.py
    methods = ["exact_elliptic", "biot_savart", "finite_element", "analytical"]
    
    config = {
        "calculation_method": get_choice_input(
            "Magnetic field calculation method", 
            methods, 
            default="exact_elliptic"
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
            "Include magnetic saturation effects (Jiles-Atherton model)?", 
            default=True
        ),
        "include_hysteresis": get_yes_no_input(
            "Include magnetic hysteresis (memory effects)?", 
            default=False
        ),
        "include_eddy_currents": get_yes_no_input(
            "Include eddy current effects (3D modeling)?", 
            default=True
        ),
        "include_skin_effect": get_yes_no_input(
            "Include frequency-dependent skin effect?", 
            default=True
        ),
        "include_temperature_effects": get_yes_no_input(
            "Include temperature-dependent material properties?", 
            default=False
        ),
        "include_displacement_current": get_yes_no_input(
            "Include displacement current effects for fast transients?", 
            default=False
        )
    }
    
    # Advanced force calculation options
    print("\nForce Calculation Options:")
    print("Configure electromagnetic force components for comprehensive physics modeling.")
    config["force_components"] = {
        "reluctance_force": get_yes_no_input(
            "Include reluctance force (gradient method: F = 0.5*I²*∂L/∂x)?", 
            default=True
        ),
        "lorentz_force": get_yes_no_input(
            "Include Lorentz force on induced currents (F = ∫J×B dV)?", 
            default=True
        ),
        "maxwell_stress": get_yes_no_input(
            "Include Maxwell stress tensor force (electromagnetic stress)?", 
            default=True
        ),
        "image_force": get_yes_no_input(
            "Include magnetic image force (ferromagnetic attraction)?", 
            default=True
        ),
        "eddy_force": get_yes_no_input(
            "Include eddy current force effects (velocity-dependent damping)?", 
            default=config["include_eddy_currents"]
        )
    }
    
    # Enhanced field solver parameters
    if config["calculation_method"] == "exact_elliptic":
        print("\nElliptic Integral Solver Options:")
        config["elliptic_solver"] = {
            "tolerance": get_float_input(
                "Elliptic integral tolerance", 
                default=1e-12, min_val=1e-15, max_val=1e-9
            ),
            "max_iterations": get_int_input(
                "Maximum elliptic solver iterations", 
                default=100, min_val=10, max_val=1000
            ),
            "use_approximation_fallback": get_yes_no_input(
                "Use approximation fallback for numerical issues?", 
                default=True
            )
        }
    
    return config


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
            default=True
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


# Timing optimization configuration removed


def setup_advanced_physics_parameters() -> Dict[str, Any]:
    """Collect advanced physics model configuration parameters"""
    print("\n" + "="*50)
    print("ADVANCED PHYSICS CONFIGURATION")
    print("="*50)
    print("Configure PhD-level electromagnetic physics models for maximum accuracy:")
    print("• Jiles-Atherton magnetic hysteresis model")
    print("• 3D eddy current patterns with skin depth effects")
    print("• Temperature-dependent material properties")
    print("• Energy conservation validation")
    
    config: Dict[str, Any] = {}
    
    # Jiles-Atherton hysteresis model parameters
    if get_yes_no_input("Configure Jiles-Atherton hysteresis model parameters?", default=False):
        print("\nJiles-Atherton Hysteresis Model:")
        config["jiles_atherton"] = {
            "enabled": True,
            "Ms": get_float_input(
                "Saturation magnetization Ms (A/m)", 
                default=1.7e6, min_val=1e5, max_val=5e6
            ),
            "a": get_float_input(
                "Shape parameter 'a' (A/m)", 
                default=1000.0, min_val=100.0, max_val=10000.0
            ),
            "alpha": get_float_input(
                "Interdomain coupling 'alpha'", 
                default=1e-3, min_val=1e-6, max_val=1e-1
            ),
            "c": get_float_input(
                "Reversible magnetization fraction 'c'", 
                default=0.2, min_val=0.0, max_val=1.0
            ),
            "k": get_float_input(
                "Pinning parameter 'k' (A/m)", 
                default=500.0, min_val=10.0, max_val=5000.0
            )
        }
    else:
        config["jiles_atherton"] = {"enabled": False}
    
    # Eddy current modeling parameters
    if get_yes_no_input("Configure advanced eddy current modeling?", default=True):
        print("\nEddy Current Modeling:")
        config["eddy_currents"] = {
            "enabled": True,
            "3d_modeling": get_yes_no_input(
                "Enable 3D eddy current modeling?", 
                default=False
            ),
            "skin_depth_calculation": get_yes_no_input(
                "Calculate frequency-dependent skin depth?", 
                default=True
            ),
            "proximity_effects": get_yes_no_input(
                "Include proximity effects between conductors?", 
                default=False
            ),
            "frequency_analysis": get_yes_no_input(
                "Perform frequency domain analysis?", 
                default=True
            ),
            "eddy_damping_factor": get_float_input(
                "Eddy current damping factor", 
                default=1.0, min_val=0.1, max_val=10.0
            )
        }
    else:
        config["eddy_currents"] = {"enabled": False}
    
    # Thermal modeling parameters
    if get_yes_no_input("Configure thermal modeling?", default=False):
        print("\nThermal Modeling:")
        config["thermal"] = {
            "enabled": True,
            "ambient_temperature": get_float_input(
                "Ambient temperature (K)", 
                default=293.15, min_val=200.0, max_val=400.0
            ),
            "thermal_time_constant": get_float_input(
                "Thermal time constant (s)", 
                default=1.0, min_val=0.1, max_val=100.0
            ),
            "temperature_dependent_resistance": get_yes_no_input(
                "Enable temperature-dependent resistance?", 
                default=True
            ),
            "temperature_dependent_permeability": get_yes_no_input(
                "Enable temperature-dependent permeability?", 
                default=False
            )
        }
    else:
        config["thermal"] = {"enabled": False}
    
    # Energy conservation validation
    energy_conservation_enabled = get_yes_no_input(
        "Enable energy conservation validation?", 
        default=True
    )
    config["energy_conservation"] = {
        "enabled": energy_conservation_enabled,
        "tolerance": get_float_input(
            "Energy conservation tolerance", 
            default=1e-6, min_val=1e-9, max_val=1e-3
        ) if energy_conservation_enabled else 1e-6
    }
    
    return config


def create_single_stage_configuration(materials: Dict[str, Any]) -> Dict[str, Any]:
    """Create single-stage configuration (original behavior)"""
    print("Creating single-stage coilgun configuration...")
    
    config = {
        "coil": setup_coil_parameters(materials),
        "projectile": setup_projectile_parameters(materials),
        "capacitor": setup_capacitor_parameters(),
        "simulation": setup_simulation_parameters(),
        "circuit_model": setup_circuit_model_parameters(),
        "magnetic_model": setup_magnetic_model_parameters(),
        "output": setup_output_parameters()
    }
    
    # Add advanced physics options
    if get_yes_no_input("Configure advanced physics models?", default=True):
        config["advanced_physics"] = setup_advanced_physics_parameters()
    
    # Timing optimization removed
    
    return config


def create_multi_stage_configuration(materials: Dict[str, Any]) -> Dict[str, Any]:
    """Create multi-stage configuration"""
    print("Creating multi-stage coilgun configuration...")
    
    # Get number of stages
    num_stages = get_int_input(
        "Number of stages", 
        default=2, min_val=2, max_val=10
    )
    
    # Determine shared settings
    shared_settings = setup_shared_settings()
    
    # Get stage grouping for coil configurations
    stage_groups = get_stage_grouping(num_stages)
    
    print(f"\nConfiguration Summary:")
    print(f"Number of stages: {num_stages}")
    print(f"Shared settings: {shared_settings}")
    print(f"Stage groups: {stage_groups}")
    
    # Initialize configuration structure
    config = {
        "multi_stage": {
            "enabled": True,
            "num_stages": num_stages,
            "shared_settings": shared_settings,
            "stage_groups": stage_groups
        },
        "stages": [],
        "shared": {}
    }
    
    # Setup shared settings first
    for setting in shared_settings:
        if setting == "simulation":
            config["shared"]["simulation"] = setup_simulation_parameters()
        elif setting == "circuit_model":
            config["shared"]["circuit_model"] = setup_circuit_model_parameters()
        elif setting == "magnetic_model":
            config["shared"]["magnetic_model"] = setup_magnetic_model_parameters()
        elif setting == "output":
            config["shared"]["output"] = setup_output_parameters()
        elif setting == "capacitor":
            config["shared"]["capacitor"] = setup_capacitor_parameters(" (SHARED)")
    
    # Setup projectile (always shared across stages)
    config["shared"]["projectile"] = setup_projectile_parameters(materials)
    
    # Setup advanced physics (shared across all stages for consistency)
    if get_yes_no_input("Configure advanced physics models for all stages?", default=True):
        config["shared"]["advanced_physics"] = setup_advanced_physics_parameters()
    
    # Timing optimization removed from multi-stage operation
    
    # Setup configurations for each group
    group_configs = {}
    for i, group in enumerate(stage_groups):
        group_id = f"group_{i+1}"
        group_info = f" (GROUP {i+1}: Stages {group})"
        
        print(f"\n" + "="*60)
        print(f"CONFIGURING GROUP {i+1}: STAGES {group}")
        print("="*60)
        
        # Setup coil configuration for this group
        group_configs[group_id] = {
            "coil": setup_coil_parameters(materials, group_info)
        }
        
        # Setup capacitor if not shared
        if "capacitor" not in shared_settings:
            group_configs[group_id]["capacitor"] = setup_capacitor_parameters(group_info)
        
        # Setup other non-shared settings
        for setting in ["simulation", "circuit_model", "magnetic_model", "output"]:
            if setting not in shared_settings:
                if setting == "simulation":
                    group_configs[group_id]["simulation"] = setup_simulation_parameters()
                elif setting == "circuit_model":
                    group_configs[group_id]["circuit_model"] = setup_circuit_model_parameters()
                elif setting == "magnetic_model":
                    group_configs[group_id]["magnetic_model"] = setup_magnetic_model_parameters()
                elif setting == "output":
                    group_configs[group_id]["output"] = setup_output_parameters()
    
    # Create individual stage configurations
    for stage_num in range(1, num_stages + 1):
        # Find which group this stage belongs to
        group_id = None
        for i, group in enumerate(stage_groups):
            if stage_num in group:
                group_id = f"group_{i+1}"
                break
        
        stage_config = {
            "stage_id": stage_num,
            "group_id": group_id
        }
        
        # Add group-specific configurations
        if group_id in group_configs:
            stage_config.update(group_configs[group_id])
        
        config["stages"].append(stage_config)
    
    return config


def create_configuration() -> Dict[str, Any]:
    """Create complete configuration by collecting all parameters"""
    print("Welcome to the Coilgun Simulation Setup!")
    print("This script will help you create a configuration file for your simulation.")
    
    # Load materials
    materials = load_material_data()
    
    # Ask if user wants multi-stage modeling
    multi_stage = get_yes_no_input(
        "\nDo you want multi-stage modeling?", 
        default=False
    )
    
    if multi_stage:
        return create_multi_stage_configuration(materials)
    else:
        return create_single_stage_configuration(materials)


def validate_configuration(config: Dict[str, Any]) -> bool:
    """Validate configuration for physics engine compatibility"""
    print("\n" + "="*50)
    print("CONFIGURATION VALIDATION")
    print("="*50)
    
    valid = True
    warnings = []
    errors = []
    
    # Validate physics engine compatibility
    try:
        # Check if we can import physics engine
        from equations import CoilgunPhysicsEngine
        
        # For multi-stage configurations, create a test single-stage config
        test_config = config
        if config.get("multi_stage", {}).get("enabled", False):
            # Create a test single-stage configuration from the first stage
            test_config = create_test_single_stage_config(config)
        
        # Create temporary file for validation
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(test_config, temp_file, indent=4)
            temp_filename = temp_file.name
        
        try:
            # Try to initialize physics engine
            test_engine = CoilgunPhysicsEngine(temp_filename)
            print("✓ Physics engine initialization: PASSED")
            
            # Validate advanced physics features
            if hasattr(test_engine, 'enable_advanced_physics') and test_engine.enable_advanced_physics:
                print("✓ Advanced physics integration: ENABLED")
                
                # Check specific features
                features = [
                    ("Magnetic saturation", getattr(test_engine, 'saturation_enabled', False)),
                    ("Eddy currents", getattr(test_engine, 'eddy_current_enabled', False)),
                    ("Hysteresis effects", getattr(test_engine, 'hysteresis_enabled', False)),
                    ("Thermal effects", getattr(test_engine, 'thermal_enabled', False)),
                    ("Energy conservation", getattr(test_engine, 'energy_conservation_enabled', True))
                ]
                
                for feature_name, enabled in features:
                    status = "ENABLED" if enabled else "DISABLED"
                    print(f"  • {feature_name}: {status}")
            
        except Exception as e:
            errors.append(f"Physics engine initialization failed: {e}")
            valid = False
        finally:
            # Clean up temporary file
            import os
            os.unlink(temp_filename)
            
    except ImportError as e:
        errors.append(f"Cannot import physics engine: {e}")
        valid = False
    
    # Validate configuration structure
    if config.get("multi_stage", {}).get("enabled", False):
        num_stages = config["multi_stage"]["num_stages"]
        if num_stages < 2:
            errors.append("Multi-stage configuration requires at least 2 stages")
            valid = False
        elif len(config.get("stages", [])) != num_stages:
            errors.append(f"Stage count mismatch: expected {num_stages}, got {len(config.get('stages', []))}")
            valid = False
    
    # Check for essential parameters
    essential_params = ["simulation", "coil", "projectile", "capacitor"]
    for param in essential_params:
        if config.get("multi_stage", {}).get("enabled", False):
            # Multi-stage: check shared or stage-specific
            if param not in config.get("shared", {}) and not any(param in stage for stage in config.get("stages", [])):
                warnings.append(f"Missing {param} configuration in multi-stage setup")
        else:
            # Single-stage: check directly
            if param not in config:
                errors.append(f"Missing essential parameter: {param}")
                valid = False
    
    # Report results
    if errors:
        print("\n❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print("\n⚠  VALIDATION WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if valid and not errors:
        print("\n✅ Configuration validation: PASSED")
        print("Configuration is compatible with the physics engine.")
    
    return valid


def save_configuration(config: Dict[str, Any], filename: str) -> None:
    """Save configuration to JSON file with validation"""
    try:
        # Validate configuration before saving
        if not validate_configuration(config):
            if not get_yes_no_input("\nConfiguration has validation errors. Save anyway?", default=False):
                print("Configuration not saved.")
                return
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\nConfiguration saved to: {filename}")
        
        # Test load the configuration
        try:
            with open(filename, 'r') as f:
                test_config = json.load(f)
            print("✓ Configuration file is valid JSON and can be loaded.")
        except Exception as e:
            print(f"⚠  Warning: Saved file may have issues: {e}")
            
    except Exception as e:
        print(f"Error saving configuration: {e}")


def print_configuration_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the configuration"""
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    
    if config.get("multi_stage", {}).get("enabled", False):
        # Multi-stage summary
        num_stages = config["multi_stage"]["num_stages"]
        shared_settings = config["multi_stage"]["shared_settings"]
        stage_groups = config["multi_stage"]["stage_groups"]
        
        print(f"Multi-stage configuration: {num_stages} stages")
        print(f"Shared settings: {', '.join(shared_settings)}")
        print(f"Stage groups: {stage_groups}")
        
        # Print projectile info (always shared)
        projectile = config["shared"]["projectile"]
        print(f"Projectile: {projectile['diameter']*1000:.1f}mm × {projectile['length']*1000:.1f}mm {projectile['material']}")
        
        # Print group configurations
        for group_num, group_stages in enumerate(stage_groups):
            group_id = f"group_{group_num+1}"
            # Find a stage in this group to get the config
            stage_with_config = None
            for stage in config["stages"]:
                if stage["group_id"] == group_id:
                    stage_with_config = stage
                    break
            
            if stage_with_config and "coil" in stage_with_config:
                coil = stage_with_config["coil"]
                print(f"Group {group_num+1} (Stages {group_stages}):")
                print(f"  Coil: {coil['inner_diameter']*1000:.1f}mm ID, {coil['length']*1000:.1f}mm length")
                print(f"  Wire: AWG {coil['wire_gauge_awg']}, {coil['num_layers']} layers")
                
                # Print capacitor if available
                capacitor = None
                if "capacitor" in stage_with_config:
                    capacitor = stage_with_config["capacitor"]
                elif "capacitor" in config["shared"]:
                    capacitor = config["shared"]["capacitor"]
                
                if capacitor:
                    energy = 0.5 * capacitor['capacitance'] * capacitor['initial_voltage']**2
                    print(f"  Capacitor: {capacitor['capacitance']*1000:.1f}mF @ {capacitor['initial_voltage']:.0f}V")
                    print(f"  Energy: {energy:.1f}J")
    else:
        # Single-stage summary
        print(f"Single-stage configuration")
        print(f"Coil: {config['coil']['inner_diameter']*1000:.1f}mm ID, {config['coil']['length']*1000:.1f}mm length")
        print(f"Wire: AWG {config['coil']['wire_gauge_awg']}, {config['coil']['num_layers']} layers")
        print(f"Projectile: {config['projectile']['diameter']*1000:.1f}mm × {config['projectile']['length']*1000:.1f}mm {config['projectile']['material']}")
        print(f"Capacitor: {config['capacitor']['capacitance']*1000:.1f}mF @ {config['capacitor']['initial_voltage']:.0f}V")
        print(f"Energy: {0.5 * config['capacitor']['capacitance'] * config['capacitor']['initial_voltage']**2:.1f}J")


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("COILGUN SIMULATION SETUP")
    print("="*60)
    print("Press Ctrl+C at any time to cancel setup gracefully.")
    
    try:
        # Create configuration
        config = create_configuration()
        
        # Get filename for saving
        print("\n" + "="*50)
        print("SAVE CONFIGURATION")
        print("="*50)
        
        # Suggest filename based on configuration type
        if config.get("multi_stage", {}).get("enabled", False):
            num_stages = config["multi_stage"]["num_stages"]
            default_filename = f"multistage_{num_stages}_coilgun_config.json"
        else:
            default_filename = "my_coilgun_config.json"
        
        try:
            filename = input(f"Enter filename for configuration (default: {default_filename}): ").strip()
        except KeyboardInterrupt:
            print("\nSetup cancelled by user.")
            sys.exit(0)
            
        if not filename:
            filename = default_filename
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Save configuration
        save_configuration(config, filename)
        
        # Display summary
        print_configuration_summary(config)
        
        print(f"\nConfiguration setup complete!")
        print(f"\nTo run the simulation:")
        print(f"python solve.py {filename}")
        print(f"\nTo visualize results:")
        print(f"python view.py {filename}")
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user (Ctrl+C)")
        print("No configuration file was saved.")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle signals gracefully"""
    print("\n\nReceived interrupt signal.")
    print("Setup cancelled. No configuration file was saved.")
    print("Exiting gracefully...")
    sys.exit(0)


if __name__ == "__main__":
    import signal
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnhandled error: {e}")
        sys.exit(1)


def create_test_single_stage_config(multi_stage_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a test single-stage configuration from a multi-stage configuration
    for validation purposes.
    
    Args:
        multi_stage_config: Multi-stage configuration dictionary
        
    Returns:
        Single-stage configuration dictionary that can be used by physics engine
    """
    if not multi_stage_config.get("multi_stage", {}).get("enabled", False):
        # Already single-stage
        return multi_stage_config
    
    # Create single-stage config from first stage
    test_config = {}
    
    # Copy shared settings to top level
    shared = multi_stage_config.get("shared", {})
    for key, value in shared.items():
        test_config[key] = value
    
    # Get first stage configuration
    stages = multi_stage_config.get("stages", [])
    if stages:
        first_stage = stages[0]
        # Copy stage-specific settings to top level
        for key, value in first_stage.items():
            if key not in ["stage_id", "group_id"]:
                test_config[key] = value
    
    return test_config

