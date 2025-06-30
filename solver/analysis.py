"""
Results Analysis Module

This module provides comprehensive analysis and processing of coilgun simulation results,
including energy analysis, force decomposition, and data export capabilities.
"""

import numpy as np
import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from physics import CoilgunPhysicsEngine


class ResultsAnalyzer:
    """
    Comprehensive results analysis and processing for coilgun simulations.
    """
    
    def __init__(self, physics_engine: CoilgunPhysicsEngine):
        """
        Initialize results analyzer.
        
        Args:
            physics_engine: Physics engine for detailed calculations
        """
        self.physics = physics_engine
    
    def analyze_solution(self, solution) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of simulation solution.
        
        Args:
            solution: ODE solution object
            
        Returns:
            Dictionary containing detailed analysis results
        """
        # Create detailed time grid for analysis
        t_detailed = np.linspace(solution.t[0], solution.t[-1], 
                                min(10000, len(solution.t) * 10))
        
        # Interpolate solution at detailed time points
        if solution.sol is not None:
            # Use interpolant if available
            y_detailed = solution.sol(t_detailed)
        else:
            # Fallback: use existing solution points directly
            print("Warning: Solution interpolant not available, using existing time points")
            t_detailed = solution.t
            y_detailed = solution.y
        
        # Extract basic state variables
        time = t_detailed
        charge = y_detailed[0] if len(y_detailed.shape) > 1 else y_detailed
        current = y_detailed[1] if len(y_detailed.shape) > 1 else np.zeros_like(time)
        position = y_detailed[2] if len(y_detailed.shape) > 1 else np.zeros_like(time)
        velocity = y_detailed[3] if len(y_detailed.shape) > 1 else np.zeros_like(time)
        
        # Calculate derived quantities
        results = {
            'time': time,
            'charge': charge,
            'current': current,
            'position': position,
            'velocity': velocity
        }
        
        # Add comprehensive analysis
        results.update(self._calculate_inductance_profile(position))
        results.update(self._calculate_force_analysis(current, position, velocity, time))
        results.update(self._calculate_power_analysis(results))
        results.update(self._calculate_energy_analysis(results))
        results.update(self._calculate_magnetic_field_analysis(position, current))
        results.update(self._calculate_eddy_current_analysis(current, position, velocity, time))
        results.update(self._extract_key_metrics(results))
        results.update(self._process_events(solution))
        
        return results
    
    def _calculate_inductance_profile(self, position: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate inductance profile along projectile trajectory."""
        inductance = np.array([
            self.physics.get_inductance(pos) for pos in position
        ])
        
        # Calculate inductance gradient
        inductance_gradient = np.gradient(inductance, position)
        
        return {
            'inductance': inductance,
            'inductance_gradient': inductance_gradient
        }
    
    def _calculate_force_analysis(self, current: np.ndarray, position: np.ndarray, 
                                velocity: np.ndarray, time: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate comprehensive force analysis."""
        force_calculations = []
        force_components = []
        
        for i, (I, x, v, t) in enumerate(zip(current, position, velocity, time)):
            # Get current and time history for enhanced analysis
            hist_length = min(50, i + 1)
            hist_start = max(0, i - hist_length + 1)
            current_history = current[hist_start:i+1] if i > 0 else None
            time_history = time[hist_start:i+1] if i > 0 else None
            
            try:
                # Use enhanced force calculation with circuit logic and timing optimization
                if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
                    force_result = self.physics.magnetic_force_with_circuit_logic(I, x, t, v)
                else:
                    # Fallback to enhanced magnetic force with history
                    force_result = self.physics.magnetic_force_ferromagnetic(
                        I, x, v, current_history, time_history
                    )
                
                # Unpack the tuple (force, eddy_power_loss) and use just the force
                if isinstance(force_result, tuple):
                    force = force_result[0]
                else:
                    force = force_result
                
                force_calculations.append(force)
                
                # Get detailed force analysis if available
                if hasattr(self.physics, 'force_analysis'):
                    force_components.append(self.physics.force_analysis.copy())
                else:
                    # Default components
                    force_components.append({
                        'force_gradient': force,
                        'force_reluctance': 0.0,
                        'force_lorentz': 0.0,
                        'force_maxwell': 0.0,
                        'force_eddy': 0.0,
                        'force_image': 0.0,
                        'power_loss_eddy': 0.0
                    })
                    
            except Exception as e:
                # Graceful fallback for compatibility
                try:
                    force_result = self.physics.magnetic_force_ferromagnetic(I, x, v)
                    if isinstance(force_result, tuple):
                        force = force_result[0]
                    else:
                        force = force_result
                    force_calculations.append(force)
                except:
                    force_calculations.append(0.0)  # Ultimate fallback
                
                force_components.append({
                    'force_gradient': 0.0,
                    'force_reluctance': 0.0,
                    'force_lorentz': 0.0,
                    'force_maxwell': 0.0,
                    'force_eddy': 0.0,
                    'force_image': 0.0,
                    'power_loss_eddy': 0.0
                })
        
        return {
            'force_total': np.array(force_calculations),
            'force_gradient': np.array([fc.get('force_gradient', 0) for fc in force_components]),
            'force_reluctance': np.array([fc.get('force_reluctance', 0) for fc in force_components]),
            'force_lorentz': np.array([fc.get('force_lorentz', 0) for fc in force_components]),
            'force_maxwell': np.array([fc.get('force_maxwell', 0) for fc in force_components]),
            'force_eddy': np.array([fc.get('force_eddy', 0) for fc in force_components]),
            'force_image': np.array([fc.get('force_image', 0) for fc in force_components]),
            'power_loss_eddy': np.array([fc.get('power_loss_eddy', 0) for fc in force_components])
        }
    
    def _calculate_power_analysis(self, results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate comprehensive power analysis."""
        time = results['time']
        charge = results['charge']
        current = results['current']
        velocity = results['velocity']
        inductance = results['inductance']
        force_total = results['force_total']
        
        # Calculate voltage components: V = Q/C - L*dI/dt
        dI_dt = np.gradient(current, time)
        voltage_capacitor = charge / self.physics.capacitance
        voltage_inductance = inductance * dI_dt
        voltage_total = voltage_capacitor - voltage_inductance
        
        # CORRECTED: Electrical power is V*I (total power drawn from circuit)
        power_electrical = voltage_total * current
        
        # Resistive losses are only the I²R component (heat dissipated in resistance)
        resistance = getattr(self.physics, 'total_resistance', 
                           getattr(self.physics.circuit_model, 'get_effective_resistance', lambda: 0.1)())
        power_loss_resistive = current**2 * resistance
        
        # Mechanical power (force doing work)
        power_mechanical = force_total * velocity
        
        return {
            'voltage_capacitor': voltage_capacitor,
            'voltage_inductance': voltage_inductance,
            'voltage_total': voltage_total,
            'power_electrical': power_electrical,
            'power_loss_resistive': power_loss_resistive,
            'power_mechanical': power_mechanical
        }
    
    def _calculate_energy_analysis(self, results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive energy analysis."""
        charge = results['charge']
        velocity = results['velocity']
        time = results['time']
        power_electrical = results['power_electrical']
        power_loss_resistive = results['power_loss_resistive']
        power_mechanical = results['power_mechanical']
        
        # Energy calculations
        energy_capacitor = 0.5 * charge**2 / self.physics.capacitance
        energy_kinetic = 0.5 * self.physics.proj_mass * velocity**2
        
        # Cumulative energy calculations
        energy_dissipated_resistive = np.trapz(np.abs(power_loss_resistive), time)
        energy_transferred_mechanical = np.trapz(np.abs(power_mechanical), time)
        
        # Initial and final energies
        initial_energy = energy_capacitor[0]
        final_kinetic_energy = energy_kinetic[-1]
        
        # FIXED: Correct efficiency calculation
        efficiency = final_kinetic_energy / initial_energy if initial_energy > 0 else 0.0
        efficiency = min(efficiency, 1.0)  # Cap at 100% (conservation of energy)
        
        return {
            'energy_capacitor': energy_capacitor,
            'energy_kinetic': energy_kinetic,
            'energy_dissipated_resistive': energy_dissipated_resistive,
            'energy_transferred_mechanical': energy_transferred_mechanical,
            'initial_energy': initial_energy,
            'final_kinetic_energy': final_kinetic_energy,
            'efficiency': efficiency,
            'energy_balance': initial_energy - final_kinetic_energy - energy_dissipated_resistive
        }
    
    def _calculate_magnetic_field_analysis(self, position: np.ndarray, 
                                         current: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate magnetic field analysis."""
        magnetic_field = []
        
        for pos, I in zip(position, current):
            try:
                if hasattr(self.physics, 'magnetic_field_solenoid_enhanced'):
                    field = self.physics.magnetic_field_solenoid_enhanced(pos, I)
                elif hasattr(self.physics, 'magnetic_field_solenoid_on_axis'):
                    field = self.physics.magnetic_field_solenoid_on_axis(pos, I)
                else:
                    field = 0.0
                magnetic_field.append(field)
            except Exception:
                magnetic_field.append(0.0)
        
        return {
            'magnetic_field': np.array(magnetic_field)
        }
    
    def _calculate_eddy_current_analysis(self, current: np.ndarray, position: np.ndarray,
                                       velocity: np.ndarray, time: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate eddy current analysis if available."""
        if not hasattr(self.physics, 'calculate_eddy_current_effects'):
            return {
                'skin_depth': np.full_like(current, np.inf),
                'eddy_current_magnitude': np.zeros_like(current),
                'eddy_current_resistance': np.full_like(current, np.inf),
                'eddy_induced_emf': np.zeros_like(current),
                'eddy_current_density': np.zeros_like(current),
                'frequency_content': np.zeros_like(current)
            }
        
        eddy_effects = []
        
        for i, (I, x, v) in enumerate(zip(current, position, velocity)):
            if abs(I) > 1e-6 and abs(v) > 1e-6:
                try:
                    # Get current and time history for frequency analysis
                    hist_length = min(50, i + 1)
                    hist_start = max(0, i - hist_length + 1)
                    current_hist = current[hist_start:i+1] if i > 0 else None
                    time_hist = time[hist_start:i+1] if i > 0 else None
                    
                    effects = self.physics.calculate_eddy_current_effects(
                        I, v, x, current_hist, time_hist
                    )
                    eddy_effects.append(effects)
                except Exception:
                    eddy_effects.append(self._default_eddy_effects())
            else:
                eddy_effects.append(self._default_eddy_effects())
        
        return {
            'skin_depth': np.array([ef.get('skin_depth', np.inf) for ef in eddy_effects]),
            'eddy_current_magnitude': np.array([ef.get('induced_current', 0) for ef in eddy_effects]),
            'eddy_current_resistance': np.array([ef.get('effective_resistance', np.inf) for ef in eddy_effects]),
            'eddy_induced_emf': np.array([ef.get('induced_emf', 0) for ef in eddy_effects]),
            'eddy_current_density': np.array([ef.get('current_density_peak', 0) for ef in eddy_effects]),
            'frequency_content': np.array([ef.get('frequency_effective', 0) for ef in eddy_effects])
        }
    
    def _default_eddy_effects(self) -> Dict[str, float]:
        """Return default eddy current effects."""
        return {
            'skin_depth': np.inf,
            'induced_current': 0,
            'opposing_force': 0,
            'power_loss': 0,
            'effective_resistance': np.inf,
            'induced_emf': 0,
            'current_density_peak': 0,
            'frequency_effective': 0
        }
    
    def _extract_key_metrics(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract key performance metrics from results."""
        time = results['time']
        current = results['current']
        position = results['position']
        velocity = results['velocity']
        force_total = results['force_total']
        
        return {
            'final_time': time[-1],
            'final_velocity': velocity[-1],
            'final_position': position[-1],
            'final_current': current[-1],
            'max_velocity': np.max(np.abs(velocity)),
            'max_current': np.max(np.abs(current)),
            'max_force': np.max(np.abs(force_total)),
            'time_to_center': self._find_time_at_position(time, position, 0.0),
            'velocity_at_center': self._find_velocity_at_position(time, position, velocity, 0.0),
            'exit_velocity': self._find_exit_velocity(time, position, velocity)
        }
    
    def _find_time_at_position(self, time: np.ndarray, position: np.ndarray, target_pos: float) -> float:
        """Find time when projectile reaches target position."""
        try:
            # Find closest approach to target position
            idx = np.argmin(np.abs(position - target_pos))
            return time[idx]
        except:
            return 0.0
    
    def _find_velocity_at_position(self, time: np.ndarray, position: np.ndarray, 
                                 velocity: np.ndarray, target_pos: float) -> float:
        """Find velocity when projectile reaches target position."""
        try:
            idx = np.argmin(np.abs(position - target_pos))
            return velocity[idx]
        except:
            return 0.0
    
    def _find_exit_velocity(self, time: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> float:
        """Find velocity when projectile exits coil."""
        try:
            coil_end = self.physics.coil_length / 2.0
            exit_pos = coil_end + self.physics.proj_length / 2.0
            
            # Find when projectile exits
            exit_indices = np.where(position >= exit_pos)[0]
            if len(exit_indices) > 0:
                return velocity[exit_indices[0]]
            else:
                return velocity[-1]  # Return final velocity if never exits
        except:
            return velocity[-1]
    
    def _process_events(self, solution) -> Dict[str, Any]:
        """Process events from solution."""
        events = {}
        
        if hasattr(solution, 't_events') and solution.t_events:
            for i, t_event in enumerate(solution.t_events):
                if len(t_event) > 0:
                    event_name = f"event_{i}"
                    events[event_name] = {
                        'times': t_event.tolist(),
                        'count': len(t_event)
                    }
        
        return {'events': events}
    
    def get_summary_results(self, solution) -> Dict[str, Any]:
        """Get summary of key simulation results."""
        results = self.analyze_solution(solution)
        
        # Extract key summary metrics
        summary = {
            'final_velocity': results.get('final_velocity', 0),
            'max_velocity': results.get('max_velocity', 0),
            'max_current': results.get('max_current', 0),
            'max_force': results.get('max_force', 0),
            'efficiency': results.get('energy_analysis', {}).get('efficiency', 0),
            'final_kinetic_energy': results.get('energy_analysis', {}).get('final_kinetic_energy', 0),
            'simulation_time': results.get('final_time', 0),
            'exit_velocity': results.get('exit_velocity', results.get('final_velocity', 0))
        }
        
        return summary
    
    def save_results_json(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def save_results_csv(self, results: Dict[str, Any], filename: str):
        """Save time-series results to CSV file."""
        if not HAS_PLOTTING:
            print("Warning: pandas not available, using fallback CSV export")
        
        # Prepare time-series data
        csv_data = {}
        time_series_keys = ['time', 'charge', 'current', 'position', 'velocity', 
                           'inductance', 'force_total', 'energy_kinetic', 'energy_capacitor']
        
        for key in time_series_keys:
            if key in results and isinstance(results[key], np.ndarray):
                csv_data[key] = results[key]
        
        if HAS_PLOTTING and csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
        else:
            # Fallback CSV writing
            with open(filename, 'w', newline='') as csvfile:
                if csv_data:
                    fieldnames = list(csv_data.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for i in range(len(csv_data[fieldnames[0]])):
                        row = {key: csv_data[key][i] for key in fieldnames}
                        writer.writerow(row)
    
    def _prepare_for_json(self, obj):
        """Recursively prepare object for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj
    
    def plot_results(self, solution, save_plots: bool = True, output_dir: str = "simulation_results"):
        """Plot comprehensive simulation results."""
        if not HAS_PLOTTING:
            print("Warning: matplotlib not available, plotting disabled")
            return
        
        results = self.analyze_solution(solution)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive plots
        self._plot_state_variables(results, save_plots, output_path)
        self._plot_force_analysis(results, save_plots, output_path)
        self._plot_energy_analysis(results, save_plots, output_path)
        self._plot_power_analysis(results, save_plots, output_path)
        
    def _plot_state_variables(self, results: Dict[str, Any], save: bool, output_path: Path):
        """Plot basic state variables."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Coilgun Simulation - State Variables')
        
        time = results['time']
        
        # Current vs time
        axes[0, 0].plot(time * 1000, results['current'])
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Current (A)')
        axes[0, 0].set_title('Current vs Time')
        axes[0, 0].grid(True)
        
        # Position vs time
        axes[0, 1].plot(time * 1000, results['position'] * 1000)
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Position (mm)')
        axes[0, 1].set_title('Position vs Time')
        axes[0, 1].grid(True)
        
        # Velocity vs time
        axes[1, 0].plot(time * 1000, results['velocity'])
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].set_title('Velocity vs Time')
        axes[1, 0].grid(True)
        
        # Force vs position
        axes[1, 1].plot(results['position'] * 1000, results['force_total'])
        axes[1, 1].set_xlabel('Position (mm)')
        axes[1, 1].set_ylabel('Force (N)')
        axes[1, 1].set_title('Force vs Position')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(output_path / 'state_variables.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_force_analysis(self, results: Dict[str, Any], save: bool, output_path: Path):
        """Plot force component analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Force Analysis')
        
        position = results['position'] * 1000  # Convert to mm
        
        # Total force
        axes[0, 0].plot(position, results['force_total'])
        axes[0, 0].set_xlabel('Position (mm)')
        axes[0, 0].set_ylabel('Force (N)')
        axes[0, 0].set_title('Total Force')
        axes[0, 0].grid(True)
        
        # Force components
        axes[0, 1].plot(position, results['force_gradient'], label='Gradient')
        axes[0, 1].plot(position, results['force_reluctance'], label='Reluctance')
        axes[0, 1].plot(position, results['force_lorentz'], label='Lorentz')
        axes[0, 1].plot(position, results['force_eddy'], label='Eddy')
        axes[0, 1].set_xlabel('Position (mm)')
        axes[0, 1].set_ylabel('Force (N)')
        axes[0, 1].set_title('Force Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Inductance profile
        axes[1, 0].plot(position, results['inductance'] * 1e6)
        axes[1, 0].set_xlabel('Position (mm)')
        axes[1, 0].set_ylabel('Inductance (µH)')
        axes[1, 0].set_title('Inductance vs Position')
        axes[1, 0].grid(True)
        
        # Magnetic field
        if 'magnetic_field' in results:
            axes[1, 1].plot(position, results['magnetic_field'])
            axes[1, 1].set_xlabel('Position (mm)')
            axes[1, 1].set_ylabel('Magnetic Field (T)')
            axes[1, 1].set_title('Magnetic Field vs Position')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(output_path / 'force_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_energy_analysis(self, results: Dict[str, Any], save: bool, output_path: Path):
        """Plot energy analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Energy Analysis')
        
        time = results['time'] * 1000  # Convert to ms
        
        # Energy vs time
        axes[0].plot(time, results['energy_capacitor'], label='Capacitor')
        axes[0].plot(time, results['energy_kinetic'], label='Kinetic')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Energy (J)')
        axes[0].set_title('Energy vs Time')
        axes[0].legend()
        axes[0].grid(True)
        
        # Energy efficiency
        efficiency = results.get('energy_analysis', {}).get('efficiency', 0) * 100
        initial_energy = results.get('energy_analysis', {}).get('initial_energy', 0)
        final_kinetic = results.get('energy_analysis', {}).get('final_kinetic_energy', 0)
        
        energy_labels = ['Initial\nStored', 'Final\nKinetic', 'Efficiency\n(%)']
        energy_values = [initial_energy, final_kinetic, efficiency]
        
        bars = axes[1].bar(energy_labels, energy_values)
        axes[1].set_title('Energy Summary')
        axes[1].set_ylabel('Energy (J) / Efficiency (%)')
        
        # Add value labels on bars
        for bar, value in zip(bars, energy_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(output_path / 'energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_power_analysis(self, results: Dict[str, Any], save: bool, output_path: Path):
        """Plot power analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Power Analysis')
        
        time = results['time'] * 1000  # Convert to ms
        
        # Power vs time
        axes[0].plot(time, results['power_electrical'], label='Electrical')
        axes[0].plot(time, results['power_mechanical'], label='Mechanical')
        axes[0].plot(time, results['power_loss_resistive'], label='Resistive Loss')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Power (W)')
        axes[0].set_title('Power vs Time')
        axes[0].legend()
        axes[0].grid(True)
        
        # Voltage components
        axes[1].plot(time, results['voltage_capacitor'], label='Capacitor')
        axes[1].plot(time, results['voltage_inductance'], label='Inductance')
        axes[1].plot(time, results['voltage_total'], label='Total')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Voltage (V)')
        axes[1].set_title('Voltage Components')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(output_path / 'power_analysis.png', dpi=300, bbox_inches='tight')
        plt.show() 