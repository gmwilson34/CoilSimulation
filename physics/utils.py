"""
Physics Utilities and Helper Functions

This module contains utility functions, validation helpers, and common calculations
used throughout the physics engine.
"""

import numpy as np
import json
import warnings
from typing import Dict, Any, Optional, Tuple, List, Union
from .core import PhysicsConstants, SafetyLimits, validate_physical_parameter


def validate_coilgun_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate coilgun configuration for completeness and physical consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required sections
    required_sections = ['coil', 'projectile', 'capacitor', 'simulation']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate coil parameters
    if 'coil' in config:
        coil_cfg = config['coil']
        
        try:
            validate_physical_parameter(
                coil_cfg.get('inner_diameter', 0), 'coil inner_diameter', 
                min_val=0.001, max_val=1.0, allow_zero=False
            )
            validate_physical_parameter(
                coil_cfg.get('length', 0), 'coil length',
                min_val=0.001, max_val=2.0, allow_zero=False
            )
            validate_physical_parameter(
                coil_cfg.get('total_turns', 0), 'coil total_turns',
                min_val=10, max_val=100000, allow_zero=False
            )
        except Exception as e:
            errors.append(f"Coil validation error: {e}")
    
    # Validate projectile parameters
    if 'projectile' in config:
        proj_cfg = config['projectile']
        
        try:
            validate_physical_parameter(
                proj_cfg.get('mass', 0), 'projectile mass',
                min_val=1e-6, max_val=5.0, allow_zero=False
            )
            validate_physical_parameter(
                proj_cfg.get('diameter', 0), 'projectile diameter',
                min_val=1e-6, max_val=0.1, allow_zero=False
            )
            validate_physical_parameter(
                proj_cfg.get('length', 0), 'projectile length',
                min_val=1e-6, max_val=1.0, allow_zero=False
            )
        except Exception as e:
            errors.append(f"Projectile validation error: {e}")
    
    # Validate capacitor parameters
    if 'capacitor' in config:
        cap_cfg = config['capacitor']
        
        try:
            validate_physical_parameter(
                cap_cfg.get('capacitance', 0), 'capacitor capacitance',
                min_val=1e-6, max_val=1.0, allow_zero=False
            )
            validate_physical_parameter(
                cap_cfg.get('initial_voltage', 0), 'capacitor initial_voltage',
                min_val=1.0, max_val=10000, allow_zero=False
            )
        except Exception as e:
            errors.append(f"Capacitor validation error: {e}")
    
    return len(errors) == 0, errors


def calculate_coil_metrics(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate derived coil metrics from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with calculated metrics
    """
    coil_cfg = config.get('coil', {})
    
    # Basic parameters
    inner_radius = coil_cfg.get('inner_diameter', 0.02) / 2.0
    outer_radius = coil_cfg.get('outer_diameter', 0.04) / 2.0
    length = coil_cfg.get('length', 0.05)
    total_turns = coil_cfg.get('total_turns', 1000)
    wire_awg = coil_cfg.get('wire_gauge_awg', 16)
    num_layers = coil_cfg.get('num_layers', 1)
    
    # Calculated metrics
    turn_density = total_turns / length  # turns per meter
    coil_volume = np.pi * inner_radius**2 * length
    
    # CORRECTED: Wire length estimation accounting for layers
    if num_layers > 1:
        # Multi-layer coil
        layer_thickness = (outer_radius - inner_radius) / num_layers
        turns_per_layer = total_turns / num_layers
        total_wire_length = 0.0
        
        for layer in range(num_layers):
            layer_radius = inner_radius + (layer + 0.5) * layer_thickness
            # Account for helical pitch
            axial_pitch = length / turns_per_layer
            helix_turn_length = np.sqrt((2 * np.pi * layer_radius)**2 + axial_pitch**2)
            total_wire_length += helix_turn_length * turns_per_layer
    else:
        # Single layer
        avg_turn_radius = inner_radius
        axial_pitch = length / total_turns
        helix_turn_length = np.sqrt((2 * np.pi * avg_turn_radius)**2 + axial_pitch**2)
        total_wire_length = helix_turn_length * total_turns
    
    # AWG wire diameter (simplified lookup)
    awg_diameters = {
        10: 2.588e-3, 12: 2.053e-3, 14: 1.628e-3, 16: 1.291e-3,
        18: 1.024e-3, 20: 0.812e-3, 22: 0.644e-3, 24: 0.511e-3
    }
    wire_diameter = awg_diameters.get(wire_awg, 1.291e-3)  # Default AWG 16
    wire_area = np.pi * (wire_diameter / 2.0)**2
    
    return {
        'inner_radius': inner_radius,
        'outer_radius': outer_radius,
        'turn_density': turn_density,
        'coil_volume': coil_volume,
        'total_wire_length': total_wire_length,
        'wire_diameter': wire_diameter,
        'wire_area': wire_area,
        'aspect_ratio': length / (2 * inner_radius),
        'num_layers': num_layers
    }


def calculate_projectile_metrics(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate derived projectile metrics from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with calculated metrics
    """
    proj_cfg = config.get('projectile', {})
    
    # Basic parameters
    mass = proj_cfg.get('mass', 0.01)
    diameter = proj_cfg.get('diameter', 0.008)
    length = proj_cfg.get('length', 0.01)
    material = proj_cfg.get('material', 'Low_Carbon_Steel')
    
    # Calculated metrics
    radius = diameter / 2.0
    volume = np.pi * radius**2 * length
    cross_sectional_area = np.pi * radius**2
    surface_area = 2 * np.pi * radius * (radius + length)
    
    # Material density (if mass and volume are inconsistent)
    calculated_density = mass / volume if volume > 0 else 0
    
    return {
        'mass': mass,
        'diameter': diameter,
        'length': length,
        'radius': radius,
        'volume': volume,
        'cross_sectional_area': cross_sectional_area,
        'surface_area': surface_area,
        'calculated_density': calculated_density,
        'aspect_ratio': length / diameter if diameter > 0 else 0,
        'material': material
    }


def estimate_system_performance(config: Dict[str, Any]) -> Dict[str, Union[float, str]]:
    """
    Estimate basic system performance metrics from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with performance estimates (floats and strings)
    """
    # Get component metrics
    coil_metrics = calculate_coil_metrics(config)
    proj_metrics = calculate_projectile_metrics(config)
    
    # Capacitor parameters
    cap_cfg = config.get('capacitor', {})
    capacitance = cap_cfg.get('capacitance', 0.001)
    initial_voltage = cap_cfg.get('initial_voltage', 400)
    
    # Energy estimates
    initial_energy = 0.5 * capacitance * initial_voltage**2
    
    # CORRECTED: More realistic inductance estimate using Wheeler's formula
    N = config['coil']['total_turns']
    L_coil = config['coil']['length']
    r = coil_metrics['inner_radius']
    
    # CORRECTED: Wheeler's formula for air-core inductance (SI units)
    # L = μ₀N²a / (9 + 10(l/a)) where a is radius, l is length
    aspect_ratio = L_coil / r
    L_estimate = (PhysicsConstants.MU_0 * N**2 * r) / (9 + 10 * aspect_ratio)
    
    # CORRECTED: Resistance estimate based on wire properties
    # Use copper resistivity and actual wire geometry
    copper_resistivity = 1.68e-8  # Ω⋅m at 20°C
    R_estimate = copper_resistivity * coil_metrics['total_wire_length'] / coil_metrics['wire_area']
    
    # CORRECTED: Time constant analysis for RLC circuits
    # Natural frequency and damping analysis
    omega_0 = 1.0 / np.sqrt(L_estimate * capacitance)  # Natural frequency (rad/s)
    damping_ratio = R_estimate / (2 * np.sqrt(L_estimate / capacitance))
    
    if damping_ratio < 1.0:
        # Underdamped oscillation
        # CORRECTED: For underdamped circuits, the exponential decay time constant is τ = 1/(ζω₀)
        # The oscillation period is T = 2π/ω_d where ω_d = ω₀√(1-ζ²)
        decay_time_constant = 1.0 / (damping_ratio * omega_0)  # Exponential decay envelope
        time_constant = decay_time_constant  # Use decay time as characteristic time
        circuit_type = "underdamped"
        # Damped frequency for reference
        omega_d = omega_0 * np.sqrt(1 - damping_ratio**2)
    elif damping_ratio == 1.0:
        # Critically damped
        time_constant = 2.0 / omega_0  # Characteristic time for critical damping
        circuit_type = "critically_damped"
        decay_time_constant = time_constant
    else:
        # Overdamped - two time constants, use the dominant (slower) one
        # τ₁,₂ = 2/(ζω₀ ± ω₀√(ζ²-1))
        sqrt_term = np.sqrt(damping_ratio**2 - 1)
        tau_1 = 2.0 / (damping_ratio * omega_0 + omega_0 * sqrt_term)  # Fast time constant
        tau_2 = 2.0 / (damping_ratio * omega_0 - omega_0 * sqrt_term)  # Slow time constant
        time_constant = max(tau_1, tau_2)  # Use the dominant (slower) time constant
        circuit_type = "overdamped"
        decay_time_constant = time_constant
    
    # Energy-based velocity estimate (no arbitrary efficiency factor)
    # Assume reasonable energy transfer based on circuit dynamics
    # Peak current estimation: I_peak ≈ V₀√(C/L)
    peak_current = initial_voltage * np.sqrt(capacitance / L_estimate)
    
    # Magnetic energy at peak current
    magnetic_energy = 0.5 * L_estimate * peak_current**2
    
    # Theoretical maximum kinetic energy (energy conservation)
    max_kinetic_energy = min(initial_energy, magnetic_energy)
    theoretical_max_velocity = np.sqrt(2 * max_kinetic_energy / proj_metrics['mass'])
    
    return {
        'initial_energy': initial_energy,
        'inductance_estimate': L_estimate,
        'resistance_estimate': R_estimate,
        'time_constant': time_constant,
        'circuit_type': circuit_type,
        'damping_ratio': damping_ratio,
        'natural_frequency': omega_0,
        'peak_current_estimate': peak_current,
        'magnetic_energy_peak': magnetic_energy,
        'theoretical_max_velocity': theoretical_max_velocity
    }


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between common physics units.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    # Length conversions
    length_to_meters = {
        'm': 1.0, 'cm': 0.01, 'mm': 0.001, 'in': 0.0254, 'ft': 0.3048
    }
    
    # Mass conversions
    mass_to_kg = {
        'kg': 1.0, 'g': 0.001, 'lb': 0.453592, 'oz': 0.0283495
    }
    
    # Energy conversions
    energy_to_joules = {
        'J': 1.0, 'kJ': 1000, 'cal': 4.184, 'Btu': 1055.06, 'eV': 1.602e-19
    }
    
    # Voltage conversions
    voltage_to_volts = {
        'V': 1.0, 'kV': 1000, 'mV': 0.001
    }
    
    # Current conversions
    current_to_amps = {
        'A': 1.0, 'mA': 0.001, 'kA': 1000
    }
    
    # Select appropriate conversion table
    conversion_tables = {
        ('m', 'cm', 'mm', 'in', 'ft'): length_to_meters,
        ('kg', 'g', 'lb', 'oz'): mass_to_kg,
        ('J', 'kJ', 'cal', 'Btu', 'eV'): energy_to_joules,
        ('V', 'kV', 'mV'): voltage_to_volts,
        ('A', 'mA', 'kA'): current_to_amps
    }
    
    # Find appropriate conversion table
    conversion_dict = None
    for units, table in conversion_tables.items():
        if from_unit in units and to_unit in units:
            conversion_dict = table
            break
    
    if conversion_dict is None:
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    
    # Convert: value -> base unit -> target unit
    base_value = value * conversion_dict[from_unit]
    result = base_value / conversion_dict[to_unit]
    
    return result


def create_parameter_sweep(base_config: Dict[str, Any], parameter_path: str, 
                         values: List[float]) -> List[Dict[str, Any]]:
    """
    Create parameter sweep configurations.
    
    Args:
        base_config: Base configuration
        parameter_path: Dot-separated path to parameter (e.g., 'coil.total_turns')
        values: List of values for the parameter
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    for value in values:
        # Deep copy base config
        config = json.loads(json.dumps(base_config))
        
        # Navigate to parameter location
        keys = parameter_path.split('.')
        current_dict = config
        
        # Navigate to parent dictionary
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        
        # Set the parameter value
        current_dict[keys[-1]] = value
        
        configs.append(config)
    
    return configs


def analyze_dimensional_consistency(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze dimensional consistency of configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Analysis results
    """
    issues = []
    warnings_list = []
    
    # Check projectile vs coil sizing
    if 'projectile' in config and 'coil' in config:
        proj_diameter = config['projectile'].get('diameter', 0)
        coil_inner_diameter = config['coil'].get('inner_diameter', 0)
        
        if proj_diameter >= coil_inner_diameter:
            issues.append("Projectile diameter >= coil inner diameter")
        
        if proj_diameter < 0.5 * coil_inner_diameter:
            warnings_list.append("Projectile diameter is less than 50% of coil diameter - may reduce efficiency")
        
        proj_length = config['projectile'].get('length', 0)
        coil_length = config['coil'].get('length', 0)
        
        if proj_length > coil_length:
            warnings_list.append("Projectile longer than coil")
    
    # Check electrical parameters
    if 'capacitor' in config:
        voltage = config['capacitor'].get('initial_voltage', 0)
        capacitance = config['capacitor'].get('capacitance', 0)
        
        if voltage > 1000:
            warnings_list.append("High voltage system - ensure proper safety measures")
        
        energy = 0.5 * capacitance * voltage**2
        if energy > 1000:  # More than 1 kJ
            warnings_list.append("High energy system - requires careful design")
    
    return {
        'dimensional_issues': issues,
        'warnings': warnings_list,
        'is_consistent': len(issues) == 0
    }


def format_scientific_notation(value: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation with specified precision.
    
    Args:
        value: Value to format
        precision: Number of decimal places
        
    Returns:
        Formatted string
    """
    if value == 0:
        return "0"
    
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    
    if abs(exponent) < 3:  # Use regular notation for small exponents
        return f"{value:.{precision}f}"
    else:
        return f"{mantissa:.{precision}f}e{exponent:+d}"


def interpolate_1d(x_values: np.ndarray, y_values: np.ndarray, 
                  x_new: float, method: str = 'linear') -> float:
    """
    Simple 1D interpolation.
    
    Args:
        x_values: X coordinates
        y_values: Y coordinates  
        x_new: New X coordinate
        method: Interpolation method ('linear', 'nearest')
        
    Returns:
        Interpolated value
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have same length")
    
    if len(x_values) < 2:
        return y_values[0] if len(y_values) > 0 else 0.0
    
    # Handle boundary cases
    if x_new <= x_values[0]:
        return y_values[0]
    if x_new >= x_values[-1]:
        return y_values[-1]
    
    # Find surrounding points
    idx = np.searchsorted(x_values, x_new) - 1
    
    if method == 'nearest':
        # Return nearest neighbor
        if abs(x_new - x_values[idx]) < abs(x_new - x_values[idx + 1]):
            return y_values[idx]
        else:
            return y_values[idx + 1]
    
    elif method == 'linear':
        # Linear interpolation
        x0, x1 = x_values[idx], x_values[idx + 1]
        y0, y1 = y_values[idx], y_values[idx + 1]
        
        t = (x_new - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def calculate_rms(values: np.ndarray) -> float:
    """
    Calculate root mean square of array.
    
    Args:
        values: Input array
        
    Returns:
        RMS value
    """
    if len(values) == 0:
        return 0.0
    
    return np.sqrt(np.mean(values**2)) 