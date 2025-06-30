"""
Utility functions for visualization and data processing.

This module provides helper functions for data loading, processing,
file management, and other utility operations for the view package.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import pandas as pd


def extract_actual_current_data(time_series_data=None, simulation_results=None, physics_engine=None):
    """
    Extract actual current data from various sources.
    
    Args:
        time_series_data: Time series data dictionary
        simulation_results: Simulation results dictionary
        physics_engine: Physics engine instance
        
    Returns:
        Dictionary with processed current data
    """
    current_data = {}
    
    # Try to extract from time series data first
    if time_series_data and 'current' in time_series_data:
        current_data['time'] = time_series_data.get('time', [])
        current_data['current'] = time_series_data['current']
        current_data['source'] = 'time_series'
    
    # Fall back to simulation results
    elif simulation_results and 'current' in simulation_results:
        current_data['time'] = simulation_results.get('time', [])
        current_data['current'] = simulation_results['current']
        current_data['source'] = 'simulation'
    
    # Generate synthetic data if physics engine available
    elif physics_engine:
        print("Generating synthetic current data from physics parameters...")
        time_points = np.linspace(0, 0.01, 1000)  # 10ms simulation
        
        # Simple RLC discharge model
        R = getattr(physics_engine, 'resistance', 0.1)
        L = getattr(physics_engine, 'inductance', 1e-3)
        C = getattr(physics_engine, 'capacitance', 1e-3)
        V0 = getattr(physics_engine, 'initial_voltage', 400)
        
        # Calculate circuit parameters
        omega_0 = 1 / np.sqrt(L * C)
        alpha = R / (2 * L)
        
        if alpha < omega_0:  # Underdamped
            omega_d = np.sqrt(omega_0**2 - alpha**2)
            current = (V0 / (L * omega_d)) * np.exp(-alpha * time_points) * np.sin(omega_d * time_points)
        else:  # Overdamped or critically damped
            current = (V0 / L) * np.exp(-alpha * time_points) * time_points
        
        current_data['time'] = time_points.tolist()
        current_data['current'] = current.tolist()
        current_data['source'] = 'synthetic'
    
    else:
        print("No current data available from any source")
        current_data = {'time': [], 'current': [], 'source': 'none'}
    
    return current_data


def find_results_directories():
    """
    Find all results directories in the current workspace.
    
    Returns:
        List of results directory paths
    """
    results_dirs = []
    current_dir = Path('.')
    
    # Look for common result directory patterns
    patterns = ['*results*', '*output*', '*simulation*', '*data*']
    
    for pattern in patterns:
        for path in current_dir.glob(pattern):
            if path.is_dir():
                results_dirs.append(path)
    
    # Also check for timestamped directories
    for item in current_dir.iterdir():
        if item.is_dir():
            # Check if directory contains result files
            has_results = any(
                item.glob('*.json') or 
                item.glob('*.npz') or 
                item.glob('*.pkl') or
                item.glob('*result*') or
                item.glob('*simulation*')
            )
            if has_results:
                results_dirs.append(item)
    
    return sorted(list(set(results_dirs)))


def select_results_directory():
    """
    Interactively select a results directory.
    
    Returns:
        Selected directory path or None
    """
    results_dirs = find_results_directories()
    
    if not results_dirs:
        print("No results directories found.")
        return None
    
    print("\nAvailable results directories:")
    for i, dir_path in enumerate(results_dirs):
        print(f"{i+1}. {dir_path}")
    
    try:
        selection = input("\nSelect directory (1-{}) or 'q' to quit: ".format(len(results_dirs)))
        
        if selection.lower() == 'q':
            return None
        
        index = int(selection) - 1
        if 0 <= index < len(results_dirs):
            return results_dirs[index]
        else:
            print("Invalid selection.")
            return None
    except (ValueError, KeyboardInterrupt):
        print("Invalid input or interrupted.")
        return None


def load_simulation_from_results(results_dir):
    """
    Load simulation results and configuration from a results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Tuple of (config, simulation_results, time_series_data)
    """
    results_path = Path(results_dir)
    
    config = None
    simulation_results = None
    time_series_data = None
    
    # Look for configuration files
    config_files = list(results_path.glob('*config*.json'))
    if config_files:
        try:
            with open(config_files[0], 'r') as f:
                config = json.load(f)
            print(f"Loaded config from: {config_files[0]}")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Look for simulation results
    result_files = list(results_path.glob('*result*.json')) + list(results_path.glob('simulation*.json'))
    if result_files:
        try:
            with open(result_files[0], 'r') as f:
                simulation_results = json.load(f)
            print(f"Loaded simulation results from: {result_files[0]}")
        except Exception as e:
            print(f"Error loading simulation results: {e}")
    
    # Look for time series data
    timeseries_files = list(results_path.glob('*timeseries*.json')) + list(results_path.glob('*time_series*.json'))
    if timeseries_files:
        try:
            with open(timeseries_files[0], 'r') as f:
                time_series_data = json.load(f)
            print(f"Loaded time series data from: {timeseries_files[0]}")
        except Exception as e:
            print(f"Error loading time series data: {e}")
    
    # Look for numpy data files
    npz_files = list(results_path.glob('*.npz'))
    if npz_files and simulation_results is None:
        try:
            data = np.load(npz_files[0])
            simulation_results = {key: data[key].tolist() if hasattr(data[key], 'tolist') else data[key] 
                                for key in data.files}
            print(f"Loaded numpy data from: {npz_files[0]}")
        except Exception as e:
            print(f"Error loading numpy data: {e}")
    
    return config, simulation_results, time_series_data


def create_output_directory(base_name="visualizations"):
    """
    Create a unique output directory for saving visualizations.
    
    Args:
        base_name: Base name for the directory
        
    Returns:
        Path to created directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{base_name}_{timestamp}"
    output_dir = Path(dir_name)
    
    try:
        output_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {output_dir}")
        return output_dir
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return Path('.')


def validate_simulation_data(simulation_results):
    """
    Validate simulation data integrity and completeness.
    
    Args:
        simulation_results: Simulation results dictionary
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'data_quality': {}
    }
    
    if not simulation_results:
        validation['valid'] = False
        validation['errors'].append("No simulation results provided")
        return validation
    
    # Check for required fields
    required_fields = ['time']
    for field in required_fields:
        if field not in simulation_results:
            validation['valid'] = False
            validation['errors'].append(f"Missing required field: {field}")
    
    # Check data consistency
    if 'time' in simulation_results:
        time_data = simulation_results['time']
        time_length = len(time_data)
        
        # Check other arrays have consistent length
        for key, value in simulation_results.items():
            if isinstance(value, (list, np.ndarray)) and len(value) != time_length:
                validation['warnings'].append(f"Data length mismatch for {key}: {len(value)} vs {time_length}")
        
        # Check for NaN or infinite values
        for key, value in simulation_results.items():
            if isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                if np.any(np.isnan(arr)):
                    validation['warnings'].append(f"NaN values found in {key}")
                if np.any(np.isinf(arr)):
                    validation['warnings'].append(f"Infinite values found in {key}")
        
        # Data quality metrics
        validation['data_quality']['time_span'] = max(time_data) - min(time_data) if time_data else 0
        validation['data_quality']['sample_count'] = time_length
        validation['data_quality']['avg_timestep'] = np.mean(np.diff(time_data)) if len(time_data) > 1 else 0
    
    return validation


def check_and_fix_empty_simulation_data(sim, config_file, time_series_data):
    """
    Check for empty simulation data and attempt to fix by running simulation.
    
    Args:
        sim: Simulation object
        config_file: Configuration file path
        time_series_data: Time series data
        
    Returns:
        Updated simulation results
    """
    if not sim or not hasattr(sim, 'results') or not sim.results:
        print("No simulation results found. Attempting to run simulation...")
        
        if hasattr(sim, 'run_simulation'):
            try:
                # Try to run the simulation
                sim.run_simulation()
                if hasattr(sim, 'results') and sim.results:
                    print("Successfully generated simulation results")
                    return sim.results
            except Exception as e:
                print(f"Error running simulation: {e}")
        
        # Fall back to time series data
        if time_series_data:
            print("Using time series data as simulation results")
            return time_series_data
        
        # Generate minimal synthetic data
        print("Generating minimal synthetic data for demonstration")
        t = np.linspace(0, 0.01, 1000)
        synthetic_results = {
            'time': t.tolist(),
            'current': (100 * np.exp(-t/0.005) * np.sin(2*np.pi*1000*t)).tolist(),
            'voltage': (400 * np.exp(-t/0.005)).tolist(),
            'position': (0.01 * t**2).tolist(),
            'velocity': (0.02 * t).tolist()
        }
        return synthetic_results
    
    return sim.results if hasattr(sim, 'results') else {}


def interpolate_missing_data(data_dict, target_length=None):
    """
    Interpolate missing or incomplete data arrays.
    
    Args:
        data_dict: Dictionary of data arrays
        target_length: Target length for all arrays
        
    Returns:
        Dictionary with interpolated data
    """
    if not data_dict:
        return data_dict
    
    # Determine target length
    if target_length is None:
        lengths = [len(v) for v in data_dict.values() if isinstance(v, (list, np.ndarray))]
        target_length = max(lengths) if lengths else 100
    
    interpolated_data = {}
    
    for key, value in data_dict.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value)
            if len(arr) != target_length:
                # Interpolate to target length
                x_old = np.linspace(0, 1, len(arr))
                x_new = np.linspace(0, 1, target_length)
                interpolated_data[key] = np.interp(x_new, x_old, arr).tolist()
            else:
                interpolated_data[key] = value
        else:
            interpolated_data[key] = value
    
    return interpolated_data


def calculate_performance_metrics(results):
    """
    Calculate performance metrics from simulation results.
    
    Args:
        results: Simulation results dictionary
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {}
    
    if 'velocity' in results and results['velocity']:
        velocities = np.array(results['velocity'])
        metrics['max_velocity'] = np.max(velocities)
        metrics['final_velocity'] = velocities[-1] if len(velocities) > 0 else 0
        metrics['avg_acceleration'] = np.mean(np.diff(velocities)) if len(velocities) > 1 else 0
    
    if 'force' in results and results['force']:
        forces = np.array(results['force'])
        metrics['max_force'] = np.max(forces)
        metrics['avg_force'] = np.mean(forces)
        metrics['force_impulse'] = np.trapz(forces) if len(forces) > 1 else 0
    
    if 'current' in results and results['current']:
        currents = np.array(results['current'])
        metrics['peak_current'] = np.max(np.abs(currents))
        metrics['rms_current'] = np.sqrt(np.mean(currents**2))
    
    if 'energy_kinetic' in results and 'energy_capacitor' in results:
        E_kin = results['energy_kinetic']
        E_cap = results['energy_capacitor']
        if E_kin and E_cap:
            metrics['efficiency'] = E_kin[-1] / E_cap[0] if E_cap[0] > 0 else 0
            metrics['energy_transfer_ratio'] = np.max(E_kin) / E_cap[0] if E_cap[0] > 0 else 0
    
    return metrics


def export_data_summary(results, output_path):
    """
    Export a summary of simulation data to a file.
    
    Args:
        results: Simulation results dictionary
        output_path: Path to save summary
    """
    summary = {
        'data_overview': {},
        'performance_metrics': {},
        'data_quality': {}
    }
    
    # Data overview
    for key, value in results.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value)
            summary['data_overview'][key] = {
                'length': len(arr),
                'min': float(np.min(arr)) if len(arr) > 0 else None,
                'max': float(np.max(arr)) if len(arr) > 0 else None,
                'mean': float(np.mean(arr)) if len(arr) > 0 else None,
                'std': float(np.std(arr)) if len(arr) > 0 else None
            }
    
    # Performance metrics
    summary['performance_metrics'] = calculate_performance_metrics(results)
    
    # Data quality
    validation = validate_simulation_data(results)
    summary['data_quality'] = validation
    
    # Save summary
    try:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Data summary saved to: {output_path}")
    except Exception as e:
        print(f"Error saving data summary: {e}")


class DataProcessor:
    """Class for advanced data processing operations."""
    
    @staticmethod
    def smooth_data(data, window_size=5):
        """Apply moving average smoothing to data."""
        if len(data) < window_size:
            return data
        
        smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        # Pad to maintain original length
        pad_length = len(data) - len(smoothed)
        return np.pad(smoothed, (pad_length//2, pad_length - pad_length//2), mode='edge')
    
    @staticmethod
    def downsample_data(data, factor=2):
        """Downsample data by given factor."""
        return data[::factor]
    
    @staticmethod
    def normalize_data(data, method='minmax'):
        """Normalize data using specified method."""
        arr = np.array(data)
        
        if method == 'minmax':
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val > min_val:
                return (arr - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val, std_val = np.mean(arr), np.std(arr)
            if std_val > 0:
                return (arr - mean_val) / std_val
        
        return arr 


def load_actual_simulation_data(results_dir="simulation_results"):
    """
    Load actual simulation data from the results directory.
    
    Args:
        results_dir: Path to simulation results directory
        
    Returns:
        Dictionary with loaded simulation data
    """
    results_path = Path(results_dir)
    simulation_data = {}
    
    print(f"Loading actual simulation data from: {results_path}")
    
    # Load CSV time series data
    csv_file = results_path / "enhanced_single_stage_data.csv"
    if csv_file.exists():
        print(f"Loading CSV data: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Convert to dictionaries for compatibility
        simulation_data['time_series'] = {
            'time': df['time'].tolist(),
            'charge': df['charge'].tolist(),
            'current': df['current'].tolist(),
            'position': df['position'].tolist(),
            'velocity': df['velocity'].tolist(),
            'inductance': df['inductance'].tolist(),
            'force_total': df['force_total'].tolist(),
            'energy_kinetic': df['energy_kinetic'].tolist(),
            'energy_capacitor': df['energy_capacitor'].tolist()
        }
        
        # Add convenient aliases for backward compatibility
        simulation_data['time'] = df['time'].tolist()
        simulation_data['current'] = df['current'].tolist()
        simulation_data['force'] = df['force_total'].tolist()
        simulation_data['position'] = df['position'].tolist()
        simulation_data['velocity'] = df['velocity'].tolist()
        
        print(f"Loaded {len(df)} data points from CSV")
        print(f"Time range: {df['time'].min():.6f} to {df['time'].max():.6f} s")
        print(f"Current range: {df['current'].min():.2f} to {df['current'].max():.2f} A")
        print(f"Force range: {df['force_total'].min():.6f} to {df['force_total'].max():.2f} N")
        print(f"Position range: {df['position'].min():.6f} to {df['position'].max():.6f} m")
    
    # Load JSON results data
    json_file = results_path / "enhanced_single_stage_results.json"
    if json_file.exists():
        print(f"Loading JSON results: {json_file}")
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            simulation_data['detailed_results'] = json_data
            print(f"Loaded detailed JSON results with keys: {list(json_data.keys())}")
        except Exception as e:
            print(f"Warning: Could not load JSON results: {e}")
    
    # Load force component analysis
    force_file = results_path / "force_component_analysis.json"
    if force_file.exists():
        print(f"Loading force analysis: {force_file}")
        try:
            with open(force_file, 'r') as f:
                force_data = json.load(f)
            simulation_data['force_analysis'] = force_data
            print(f"Loaded {len(force_data)} force analysis data points")
            
            # Extract force components for plotting
            if force_data:
                simulation_data['force_components'] = {
                    'gradient_force': [d.get('gradient_force', 0) for d in force_data],
                    'reluctance_force': [d.get('reluctance_force', 0) for d in force_data],
                    'lorentz_force': [d.get('lorentz_force', 0) for d in force_data],
                    'eddy_current_force': [d.get('eddy_current_force', 0) for d in force_data],
                    'total_force': [d.get('total_force', 0) for d in force_data]
                }
        except Exception as e:
            print(f"Warning: Could not load force analysis: {e}")
    
    # Load configuration
    config_file = Path("coilgun_config_expert.json")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            simulation_data['config'] = config_data
            print(f"Loaded configuration: {config_file}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Add metadata
    simulation_data['data_source'] = 'actual_simulation'
    simulation_data['loaded_from'] = str(results_path)
    simulation_data['available_data'] = list(simulation_data.keys())
    
    if not simulation_data:
        print("Warning: No simulation data could be loaded!")
        return None
    
    print(f"Successfully loaded simulation data with components: {simulation_data['available_data']}")
    return simulation_data


def get_actual_current_profile(simulation_data):
    """
    Get the actual current profile from simulation data.
    
    Args:
        simulation_data: Loaded simulation data dictionary
        
    Returns:
        Tuple of (time_array, current_array) or None if not available
    """
    if not simulation_data or 'time_series' not in simulation_data:
        return None, None
    
    time_data = simulation_data['time_series']['time']
    current_data = simulation_data['time_series']['current']
    
    return np.array(time_data), np.array(current_data)


def get_actual_force_profile(simulation_data):
    """
    Get the actual force profile from simulation data.
    
    Args:
        simulation_data: Loaded simulation data dictionary
        
    Returns:
        Tuple of (time_array, force_array) or None if not available
    """
    if not simulation_data or 'time_series' not in simulation_data:
        return None, None
    
    time_data = simulation_data['time_series']['time']
    force_data = simulation_data['time_series']['force_total']
    
    return np.array(time_data), np.array(force_data)


def get_simulation_parameters(simulation_data):
    """
    Extract key simulation parameters from loaded data.
    
    Args:
        simulation_data: Loaded simulation data dictionary
        
    Returns:
        Dictionary with key parameters
    """
    if not simulation_data or 'config' not in simulation_data:
        return {}
    
    config = simulation_data['config']
    
    return {
        'coil_inner_radius': config['coil']['inner_diameter'] / 2,
        'coil_length': config['coil']['length'],
        'coil_turns': config['coil']['total_turns'],
        'projectile_diameter': config['projectile']['diameter'],
        'projectile_mass': config['projectile']['mass'],
        'capacitor_voltage': config['capacitor']['initial_voltage'],
        'capacitor_capacitance': config['capacitor']['capacitance'],
        'max_current': max(simulation_data.get('current', [0])),
        'max_force': max(simulation_data.get('force', [0])),
        'simulation_time': max(simulation_data.get('time', [0]))
    } 


class SimulationPhysicsEngine:
    """
    Simple physics engine mock that provides parameters from simulation configuration.
    This allows the field calculations to work with actual simulation parameters.
    """
    
    def __init__(self, config_data):
        """
        Initialize physics engine from configuration data.
        
        Args:
            config_data: Configuration dictionary from simulation
        """
        if config_data and 'coil' in config_data:
            coil_config = config_data['coil']
            projectile_config = config_data.get('projectile', {})
            
            # Coil parameters
            self.coil_inner_radius = coil_config['inner_diameter'] / 2
            self.coil_outer_radius = self.coil_inner_radius + 0.01  # Estimate outer radius
            self.coil_length = coil_config['length']
            self.num_turns = coil_config['total_turns']
            
            # Projectile parameters
            self.projectile_radius = projectile_config.get('diameter', 0.05) / 2
            self.projectile_mass = projectile_config.get('mass', 1.0)
            self.projectile_permeability = 1000  # Typical ferromagnetic material
            
            # Circuit parameters
            capacitor_config = config_data.get('capacitor', {})
            self.capacitance = capacitor_config.get('capacitance', 1e-3)
            self.initial_voltage = capacitor_config.get('initial_voltage', 400)
            self.resistance = capacitor_config.get('esr', 0.1)
            self.inductance = 1e-3  # Estimate
            
            print(f"Physics engine initialized:")
            print(f"  Coil: inner_r={self.coil_inner_radius*1000:.1f}mm, length={self.coil_length*1000:.1f}mm, turns={self.num_turns}")
            print(f"  Projectile: radius={self.projectile_radius*1000:.1f}mm, mass={self.projectile_mass:.3f}kg")
            print(f"  Circuit: C={self.capacitance:.3f}F, V0={self.initial_voltage}V")
        else:
            # Default parameters if no config available
            self.coil_inner_radius = 0.025
            self.coil_outer_radius = 0.035
            self.coil_length = 0.06
            self.num_turns = 100
            self.projectile_radius = 0.025
            self.projectile_mass = 1.0
            self.projectile_permeability = 1000
            self.capacitance = 1e-3
            self.initial_voltage = 400
            self.resistance = 0.1
            self.inductance = 1e-3
            print("Physics engine initialized with default parameters") 