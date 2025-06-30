"""
Solver Utility Functions

This module provides utility functions for configuration management,
file operations, and common solver tasks.
"""

import json
import os
import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .core import SolverUtils as CoreUtils


def find_config_files(directory: str = ".") -> List[Path]:
    """
    Find all coilgun configuration files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of configuration file paths
    """
    return CoreUtils.find_config_files(directory)


def select_config_file() -> Optional[str]:
    """
    Interactive config file selection.
    
    Returns:
        Selected configuration file path or None
    """
    return CoreUtils.select_config_file()


def setup_signal_handlers(progress_tracker=None):
    """
    Setup signal handlers for graceful interruption.
    
    Args:
        progress_tracker: Optional progress tracker to stop on interruption
    """
    def signal_handler(signum, frame):
        print("\n⚠  Simulation interrupted by user")
        if progress_tracker:
            progress_tracker.stop()
        
        # Clean up temporary files
        cleanup_temp_files()
        
        print("Exiting...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def cleanup_temp_files():
    """Clean up temporary files created during simulation."""
    temp_patterns = [
        "temp_stage_*.json",
        "temp_param_*.json", 
        "temp_opt_*.json",
        "temp_multi_opt_*.json",
        "temp_final_opt.json"
    ]
    
    current_dir = Path(".")
    for pattern in temp_patterns:
        for temp_file in current_dir.glob(pattern):
            try:
                temp_file.unlink()
                print(f"Cleaned up: {temp_file}")
            except Exception:
                pass


def validate_config_structure(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required top-level sections
    required_sections = ['coil', 'projectile', 'capacitor']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate coil section
    if 'coil' in config:
        coil_config = config['coil']
        # Core required parameters - total_turns can be calculated automatically
        required_coil_params = ['inner_diameter', 'length']
        
        for param in required_coil_params:
            if param not in coil_config:
                errors.append(f"Missing coil parameter: {param}")
            elif not isinstance(coil_config[param], (int, float)) or coil_config[param] <= 0:
                errors.append(f"Invalid coil parameter {param}: must be positive number")
        
        # Check if we have enough info to calculate total_turns if it's missing
        if 'total_turns' not in coil_config:
            # Need wire info to calculate turns
            if 'wire_gauge_awg' not in coil_config or 'num_layers' not in coil_config:
                errors.append("Missing total_turns and insufficient wire info (need wire_gauge_awg and num_layers) to calculate it")
    
    # Validate projectile section
    if 'projectile' in config:
        proj_config = config['projectile']
        # Core required parameters - mass can be calculated from material and dimensions
        required_proj_params = ['length', 'diameter', 'material']
        
        for param in required_proj_params:
            if param not in proj_config:
                errors.append(f"Missing projectile parameter: {param}")
            elif param != 'material' and (not isinstance(proj_config[param], (int, float)) or proj_config[param] <= 0):
                errors.append(f"Invalid projectile parameter {param}: must be positive number")
        
        # Mass is optional if we have material and dimensions (physics engine can calculate it)
        if 'mass' not in proj_config:
            # This is OK - physics engine will calculate from material density and volume
            pass
    
    # Validate capacitor section
    if 'capacitor' in config:
        cap_config = config['capacitor']
        required_cap_params = ['capacitance', 'initial_voltage']
        
        for param in required_cap_params:
            if param not in cap_config:
                errors.append(f"Missing capacitor parameter: {param}")
            elif not isinstance(cap_config[param], (int, float)) or cap_config[param] <= 0:
                errors.append(f"Invalid capacitor parameter {param}: must be positive number")
    
    return len(errors) == 0, errors


def load_and_validate_config(config_file: str) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Load and validate a configuration file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Tuple of (config_dict, is_valid, error_messages)
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        is_valid, errors = validate_config_structure(config)
        return config, is_valid, errors
        
    except FileNotFoundError:
        return {}, False, [f"Configuration file not found: {config_file}"]
    except json.JSONDecodeError as e:
        return {}, False, [f"Invalid JSON in configuration file: {str(e)}"]
    except Exception as e:
        return {}, False, [f"Error loading configuration: {str(e)}"]


def create_default_config() -> Dict[str, Any]:
    """
    Create a default coilgun configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "coil": {
            "inner_diameter": 0.02,
            "length": 0.05,
            "wire_diameter": 0.001,
            "num_layers": 1,
            "total_turns": 1000,
            "resistance_per_turn": 0.0001
        },
        "projectile": {
            "mass": 0.01,
            "length": 0.01,
            "diameter": 0.008,
            "material": "Low_Carbon_Steel",
            "initial_position": -0.03
        },
        "capacitor": {
            "capacitance": 0.001,
            "initial_voltage": 400,
            "esr": 0.1
        },
        "magnetic_model": {
            "calculation_method": "finite_solenoid",
            "include_fringing": True,
            "include_eddy_currents": True
        },
        "solver": {
            "method": "RK45",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_step": 1e-4
        }
    }


def save_config(config: Dict[str, Any], filename: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with overrides
        
    Returns:
        Merged configuration
    """
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return deep_merge(base_config, override_config)


def extract_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a summary of key configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Summary dictionary
    """
    summary = {}
    
    # Coil summary
    if 'coil' in config:
        coil = config['coil']
        summary['coil'] = {
            'diameter': coil.get('inner_diameter', 0) * 1000,  # mm
            'length': coil.get('length', 0) * 1000,  # mm
            'turns': coil.get('total_turns', 0),
            'layers': coil.get('num_layers', 1)
        }
    
    # Projectile summary
    if 'projectile' in config:
        proj = config['projectile']
        summary['projectile'] = {
            'mass': proj.get('mass', 0) * 1000,  # g
            'diameter': proj.get('diameter', 0) * 1000,  # mm
            'length': proj.get('length', 0) * 1000,  # mm
            'material': proj.get('material', 'Unknown')
        }
    
    # Capacitor summary
    if 'capacitor' in config:
        cap = config['capacitor']
        summary['capacitor'] = {
            'capacitance': cap.get('capacitance', 0) * 1000,  # mF
            'voltage': cap.get('initial_voltage', 0),  # V
            'energy': 0.5 * cap.get('capacitance', 0) * cap.get('initial_voltage', 0)**2  # J
        }
    
    return summary


def print_config_summary(config: Dict[str, Any]):
    """
    Print a formatted summary of configuration parameters.
    
    Args:
        config: Configuration dictionary
    """
    summary = extract_config_summary(config)
    
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    
    if 'coil' in summary:
        coil = summary['coil']
        print(f"Coil:")
        print(f"  Inner diameter: {coil['diameter']:.1f} mm")
        print(f"  Length: {coil['length']:.1f} mm")
        print(f"  Total turns: {coil['turns']:,}")
        print(f"  Layers: {coil['layers']}")
    
    if 'projectile' in summary:
        proj = summary['projectile']
        print(f"\nProjectile:")
        print(f"  Mass: {proj['mass']:.1f} g")
        print(f"  Diameter: {proj['diameter']:.1f} mm")
        print(f"  Length: {proj['length']:.1f} mm")
        print(f"  Material: {proj['material']}")
    
    if 'capacitor' in summary:
        cap = summary['capacitor']
        print(f"\nCapacitor:")
        print(f"  Capacitance: {cap['capacitance']:.1f} mF")
        print(f"  Initial voltage: {cap['voltage']:.0f} V")
        print(f"  Stored energy: {cap['energy']:.1f} J")
    
    print("="*50)


def estimate_simulation_time(config: Dict[str, Any]) -> float:
    """
    Estimate simulation time based on configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Estimated simulation time in seconds
    """
    # Get capacitor parameters
    cap_config = config.get('capacitor', {})
    capacitance = cap_config.get('capacitance', 0.001)
    voltage = cap_config.get('initial_voltage', 400)
    
    # Estimate coil resistance
    coil_config = config.get('coil', {})
    turns = coil_config.get('total_turns', 1000)
    resistance_per_turn = coil_config.get('resistance_per_turn', 0.0001)
    total_resistance = turns * resistance_per_turn
    
    # RC time constant gives a rough estimate
    RC = total_resistance * capacitance
    
    # Add time for projectile transit
    coil_length = coil_config.get('length', 0.05)
    estimated_velocity = 50.0  # m/s rough estimate
    transit_time = coil_length / estimated_velocity
    
    # Total estimated time (with safety margin)
    estimated_time = max(5 * RC, 2 * transit_time, 0.01)
    
    return min(estimated_time, 1.0)  # Cap at 1 second


def format_simulation_results(results: Dict[str, Any]) -> str:
    """
    Format simulation results for display.
    
    Args:
        results: Simulation results dictionary
        
    Returns:
        Formatted results string
    """
    output = []
    
    # Basic results
    final_velocity = results.get('final_velocity', 0)
    max_velocity = results.get('max_velocity', 0)
    efficiency = results.get('energy_analysis', {}).get('efficiency', 0) * 100
    
    output.append(f"Final velocity: {final_velocity:.2f} m/s")
    output.append(f"Maximum velocity: {max_velocity:.2f} m/s")
    output.append(f"Efficiency: {efficiency:.1f}%")
    
    # Performance metrics
    max_current = results.get('max_current', 0)
    max_force = results.get('max_force', 0)
    
    output.append(f"Maximum current: {max_current:.1f} A")
    output.append(f"Maximum force: {max_force:.1f} N")
    
    # Energy analysis
    energy_analysis = results.get('energy_analysis', {})
    initial_energy = energy_analysis.get('initial_energy', 0)
    final_kinetic = energy_analysis.get('final_kinetic_energy', 0)
    
    output.append(f"Initial energy: {initial_energy:.1f} J")
    output.append(f"Final kinetic energy: {final_kinetic:.3f} J")
    
    return "\n".join(output)


def create_results_report(results: Dict[str, Any], config: Dict[str, Any], 
                         output_file: Optional[str] = None) -> str:
    """
    Create a comprehensive results report.
    
    Args:
        results: Simulation results
        config: Configuration used
        output_file: Optional output file path
        
    Returns:
        Report text
    """
    report_lines = []
    
    # Header
    report_lines.append("="*60)
    report_lines.append("COILGUN SIMULATION REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Configuration summary
    report_lines.append("CONFIGURATION:")
    report_lines.append("-" * 30)
    config_summary = extract_config_summary(config)
    
    if 'coil' in config_summary:
        coil = config_summary['coil']
        report_lines.append(f"Coil: {coil['diameter']:.1f}mm ID × {coil['length']:.1f}mm, {coil['turns']:,} turns")
    
    if 'projectile' in config_summary:
        proj = config_summary['projectile']
        report_lines.append(f"Projectile: {proj['mass']:.1f}g {proj['material']}, {proj['diameter']:.1f}mm × {proj['length']:.1f}mm")
    
    if 'capacitor' in config_summary:
        cap = config_summary['capacitor']
        report_lines.append(f"Capacitor: {cap['capacitance']:.1f}mF @ {cap['voltage']:.0f}V ({cap['energy']:.1f}J)")
    
    report_lines.append("")
    
    # Results
    report_lines.append("RESULTS:")
    report_lines.append("-" * 30)
    report_lines.append(format_simulation_results(results))
    report_lines.append("")
    
    # Simulation info
    sim_time = results.get('simulation_time', 0)
    step_count = results.get('step_count', 0)
    method = results.get('integration_method', 'Unknown')
    
    report_lines.append("SIMULATION INFO:")
    report_lines.append("-" * 30)
    report_lines.append(f"Integration method: {method}")
    report_lines.append(f"Computation time: {sim_time:.2f} seconds")
    report_lines.append(f"Integration steps: {step_count:,}")
    report_lines.append("")
    
    report_lines.append("="*60)
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    return report_text 