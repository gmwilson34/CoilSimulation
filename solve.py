# solve.py
"""
Advanced Coilgun Simulation Engine

This module orchestrates the complete coilgun simulation using Maxwell's equations,
advanced circuit modeling, and electromagnetic force calculations. It provides
comprehensive analysis and results output for engineering design.

Features:
- High-precision ODE integration with adaptive stepping
- Real-time progress monitoring and diagnostics
- Comprehensive result analysis and efficiency calculations
- Data export for visualization and further analysis
- Parametric studies and optimization support
- Multi-stage coilgun simulation with velocity transfer between stages
- Interactive progress bar for integration tracking
"""

import numpy as np
import json
import time
import sys
import threading
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from equations import CoilgunPhysicsEngine

class ProgressTracker:
    """
    Enhanced progress tracking class with physics diagnostics.
    """
    
    def __init__(self, t_span, update_interval=0.1, physics_engine=None):
        """
        Initialize enhanced progress tracker.
        
        Args:
            t_span: Time span tuple (t_start, t_end)
            update_interval: Update interval in seconds
            physics_engine: Physics engine for diagnostics
        """
        self.t_start, self.t_end = t_span
        self.t_duration = self.t_end - self.t_start
        self.update_interval = update_interval
        self.physics = physics_engine
        
        # Progress tracking
        self.current_time = self.t_start
        self.current_state = None
        self.step_count = 0
        self.start_real_time = time.time()
        self.last_update_time = self.start_real_time
        self.last_step_count = 0
        
        # Physics diagnostics
        self.max_current = 0
        self.max_force = 0
        self.max_velocity = 0
        self.current_position = 0
        self.physics_warnings = []
        
        # Progress bar settings
        self.bar_width = 50
        self.running = True
        
        # Start progress display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def update(self, t, y):
        """
        Enhanced update with physics diagnostics.
        
        Args:
            t: Current time
            y: Current state vector [Q, I, x, v]
        """
        self.current_time = t
        self.current_state = y
        self.step_count += 1
        
        # Update physics diagnostics
        if len(y) >= 4:
            Q, I, x, v = y
            self.max_current = max(self.max_current, abs(I))
            self.max_velocity = max(self.max_velocity, abs(v))
            self.current_position = x
            
            # Calculate current force for diagnostics
            if self.physics and abs(I) > 1e-6:
                try:
                    force = self.physics.magnetic_force_with_circuit_logic(I, x, t, v)
                    self.max_force = max(self.max_force, abs(force))
                except Exception as e:
                    if len(self.physics_warnings) < 5:  # Limit warnings
                        self.physics_warnings.append(f"Force calculation warning at t={t:.2e}s: {str(e)[:50]}")
    
    def _display_loop(self):
        """Display progress bar in a separate thread."""
        while self.running:
            self._draw_progress_bar()
            time.sleep(self.update_interval)
    
    def _draw_progress_bar(self):
        """Draw enhanced progress bar with physics diagnostics."""
        # Calculate progress percentage
        if self.t_duration > 0:
            progress = min(1.0, (self.current_time - self.t_start) / self.t_duration)
        else:
            progress = 0.0
        
        # Calculate integration rate
        current_real_time = time.time()
        real_time_elapsed = current_real_time - self.last_update_time
        
        if real_time_elapsed >= self.update_interval:
            steps_since_update = self.step_count - self.last_step_count
            integration_rate = steps_since_update / real_time_elapsed if real_time_elapsed > 0 else 0
            self.last_update_time = current_real_time
            self.last_step_count = self.step_count
        else:
            total_elapsed = current_real_time - self.start_real_time
            integration_rate = self.step_count / total_elapsed if total_elapsed > 0 else 0
        
        # Create progress bar
        filled = int(self.bar_width * progress)
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # Format time
        if self.current_time < 1e-3:
            time_str = f"{self.current_time*1e6:.1f}μs"
        elif self.current_time < 1:
            time_str = f"{self.current_time*1e3:.1f}ms"
        else:
            time_str = f"{self.current_time:.3f}s"
        
        if self.t_end < 1e-3:
            total_time_str = f"{self.t_end*1e6:.1f}μs"
        elif self.t_end < 1:
            total_time_str = f"{self.t_end*1e3:.1f}ms"
        else:
            total_time_str = f"{self.t_end:.3f}s"
        
        # Physics status indicators
        physics_status = ""
        if self.current_state is not None and len(self.current_state) >= 4:
            I, x, v = self.current_state[1], self.current_state[2], self.current_state[3]
            physics_status = f" | I:{I:.0f}A | x:{x*1000:.1f}mm | v:{v:.1f}m/s"
        
        # Create enhanced progress line
        progress_line = (f"\rSimulation: [{bar}] {progress*100:6.2f}% | "
                        f"Time: {time_str}/{total_time_str} | "
                        f"Steps: {self.step_count:,} | "
                        f"Rate: {integration_rate:.0f}/s{physics_status}")
        
        # Truncate if too long for terminal
        if len(progress_line) > 120:
            progress_line = progress_line[:117] + "..."
        
        # Write to terminal
        sys.stdout.write(progress_line)
        sys.stdout.flush()
    
    def stop(self):
        """Stop the progress tracker."""
        self.running = False
        if self.display_thread.is_alive():
            self.display_thread.join(timeout=0.5)
        
        # Clear the progress line and show final summary
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        
        if self.physics_warnings:
            print(f"Physics warnings encountered: {len(self.physics_warnings)}")
            for warning in self.physics_warnings[:3]:  # Show first 3
                print(f"  ⚠  {warning}")
        
        sys.stdout.flush()

class CoilgunSimulation:
    """
    Main simulation class that orchestrates the complete coilgun analysis.
    """
    
    def __init__(self, config_file):
        """
        Initialize the simulation with configuration file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize physics engine
        self.physics = CoilgunPhysicsEngine(config_file)
        
        # Progress tracker
        self.progress_tracker = None
        
        # Initialize results storage
        self.results = {
            # Basic state variables
            'time': None,
            'charge': None,
            'current': None,
            'position': None,
            'velocity': None,
            
            # Enhanced electromagnetic analysis
            'force_total': None,
            'force_gradient': None,
            'force_reluctance': None,
            'force_lorentz': None,
            'force_maxwell': None,
            'force_eddy': None,
            'inductance': None,
            'inductance_gradient': None,
            
            # Power and energy analysis
            'power_electrical': None,
            'power_mechanical': None,
            'power_loss_resistive': None,
            'power_loss_eddy': None,
            'energy_capacitor': None,
            'energy_kinetic': None,
            'energy_magnetic': None,
            
            # Advanced physics
            'magnetic_field': None,
            'permeability_effective': None,
            'saturation_factor': None,
            'eddy_current_magnitude': None,
            'skin_depth': None,
            'frequency_content': None,
            'temperature_rise': None,
            
            # Physics validation
            'field_accuracy': None,
            'force_consistency': None,
            'energy_conservation': None,
            
            # Backward compatibility
            'force': None,  # Alias for force_total
            'power': None   # Alias for power_electrical
        }
        
        # Simulation metadata
        self.simulation_info = {
            'config_file': config_file,
            'start_time': None,
            'end_time': None,
            'duration': None,
            'total_steps': None,
            'final_velocity': None,
            'efficiency': None,
            'max_current': None,
            'max_force': None,
            'exit_reason': None
        }
        
    def _create_progress_tracking_wrapper(self, original_func):
        """
        Create a wrapper for the ODE function that tracks progress.
        
        Args:
            original_func: Original ODE function
            
        Returns:
            Wrapped function that updates progress
        """
        def wrapped_func(t, y):
            if self.progress_tracker:
                self.progress_tracker.update(t, y)
            return original_func(t, y)
        return wrapped_func
        
    def run_simulation(self, save_data=True, verbose=True, show_progress=True):
        """
        Execute the complete coilgun simulation.
        
        Args:
            save_data: Whether to save detailed time-series data
            verbose: Whether to print progress and results
            show_progress: Whether to show integration progress bar
            
        Returns:
            dict: Simulation results and analysis
        """
        if verbose:
            print("=" * 60)
            print("ADVANCED COILGUN SIMULATION")
            print("=" * 60)
            self.physics.print_system_parameters()
            print("\nStarting simulation...")
        
        # Record start time
        self.simulation_info['start_time'] = time.time()
        
        # Get initial conditions
        y0 = self.physics.get_initial_conditions()
        
        # Simulation parameters
        sim_config = self.config['simulation']
        t_span = sim_config['time_span']
        max_step = sim_config.get('max_step', 1e-6)
        tolerance = sim_config.get('tolerance', 1e-9)
        method = sim_config.get('method', 'RK45')
        
        # Initialize enhanced progress tracker
        if show_progress and verbose:
            self.progress_tracker = ProgressTracker(t_span, physics_engine=self.physics)
            if verbose:
                print(f"Progress tracking enabled. Integration method: {method}")
        
        # Create progress-tracking wrapper for ODE function
        ode_func = self.physics.circuit_derivatives
        if self.progress_tracker:
            ode_func = self._create_progress_tracking_wrapper(ode_func)
        
        # Define events to stop simulation
        def projectile_at_center(t, y):
            """Event: projectile reaches coil center."""
            return y[2] - self.physics.coil_center
        
        def projectile_exits_coil(t, y):
            """Event: projectile completely exits coil."""
            return y[2] - (self.physics.coil_length + self.physics.proj_length)
        
        def current_reverses(t, y):
            """Event: current reverses direction."""
            return y[1]  # Current
        
        # Configure events
        projectile_at_center.terminal = True
        projectile_at_center.direction = 1
        
        projectile_exits_coil.terminal = False
        projectile_exits_coil.direction = 1
        
        current_reverses.terminal = False
        current_reverses.direction = -1
        
        events = [projectile_at_center, projectile_exits_coil, current_reverses]
        
        try:
            # Solve the ODE system
            if verbose:
                print(f"Integrating ODEs with {method} method...")
                if show_progress:
                    print("Integration progress will be shown below:")
            
            solution = solve_ivp(
                fun=ode_func,
                t_span=t_span,
                y0=y0,
                method=method,
                max_step=max_step,
                rtol=tolerance,
                atol=tolerance * 1e-3,
                events=events,
                dense_output=True
            )
            
            if not solution.success:
                raise RuntimeError(f"Integration failed: {solution.message}")
            
            # Store results
            self._process_results(solution, save_data)
            
            # Determine exit reason
            if solution.t_events[0].size > 0:  # Projectile reached center
                self.simulation_info['exit_reason'] = "Projectile reached coil center"
                final_time = solution.t_events[0][0]
                final_state = solution.sol(final_time)
            else:
                self.simulation_info['exit_reason'] = "Simulation time limit reached"
                final_time = solution.t[-1]
                final_state = solution.y[:, -1]
            
            # Extract final results
            final_velocity = final_state[3]
            self.simulation_info['final_velocity'] = final_velocity
            self.simulation_info['efficiency'] = self.physics.calculate_efficiency(final_velocity)
            
            # Record simulation metadata
            self.simulation_info['end_time'] = time.time()
            self.simulation_info['duration'] = (self.simulation_info['end_time'] - 
                                               self.simulation_info['start_time'])
            self.simulation_info['total_steps'] = len(solution.t)
            
            if save_data and self.results['current'] is not None:
                self.simulation_info['max_current'] = np.max(np.abs(self.results['current']))
                self.simulation_info['max_force'] = np.max(np.abs(self.results['force_total']))
            else:
                # Calculate max values from the solution data even if not saving detailed results
                if hasattr(solution, 'y') and solution.y.shape[1] > 0:
                    currents = solution.y[1, :]  # Current is the second state variable
                    self.simulation_info['max_current'] = np.max(np.abs(currents))
                    
                    # Calculate forces for max force
                    forces = []
                    for i, current in enumerate(currents):
                        position = solution.y[2, i]  # Position is third state variable
                        current_time = solution.t[i] if i < len(solution.t) else solution.t[-1]  # Time
                        force = self.physics.magnetic_force_with_circuit_logic(current, position, current_time)
                        forces.append(force)
                    self.simulation_info['max_force'] = np.max(np.abs(forces)) if forces else 0
                else:
                    self.simulation_info['max_current'] = 0
                    self.simulation_info['max_force'] = 0
            
            if verbose:
                self._print_results()
            
            return self._get_summary_results()
            
        except Exception as e:
            print(f"Simulation failed: {str(e)}")
            raise
        finally:
            # Always stop progress tracker
            if self.progress_tracker:
                self.progress_tracker.stop()
                if verbose and show_progress:
                    print("Integration completed.")

    def _process_results(self, solution, save_data):
        """
        Process and store simulation results.
        
        Args:
            solution: ODE solution object
            save_data: Whether to save time-series data
        """
        if not save_data:
            return
        
        # Time points for detailed analysis
        t_detailed = np.linspace(solution.t[0], solution.t[-1], 
                                min(10000, len(solution.t) * 10))
        
        # Interpolate solution at detailed time points
        y_detailed = solution.sol(t_detailed)
        
        # Store basic state variables
        self.results['time'] = t_detailed
        self.results['charge'] = y_detailed[0]
        self.results['current'] = y_detailed[1] 
        self.results['position'] = y_detailed[2]
        self.results['velocity'] = y_detailed[3]
        
        # Calculate derived quantities
        self.results['inductance'] = np.array([
            self.physics.get_inductance(pos) for pos in self.results['position']
        ])
        
        self.results['force_total'] = np.array([
            self.physics.magnetic_force_with_circuit_logic(I, x, current_time) 
            for I, x, current_time in zip(self.results['current'], self.results['position'], self.results['time'])
        ])
        
        # Power and energy analysis
        self.results['power_electrical'] = self.results['current'] * self.results['charge'] / self.physics.capacitance
        
        self.results['energy_capacitor'] = 0.5 * self.results['charge']**2 / self.physics.capacitance
        
        self.results['energy_kinetic'] = (0.5 * self.physics.proj_mass * 
                                         self.results['velocity']**2)
        
        # Enhanced physics analysis (with backward compatibility)
        try:
            # Force decomposition analysis if available
            if hasattr(self.physics, 'force_analysis'):
                force_components = []
                for I, x, v, t in zip(self.results['current'], self.results['position'], 
                                    self.results['velocity'], self.results['time']):
                    # Calculate force to get components stored in force_analysis
                    self.physics.magnetic_force_ferromagnetic(I, x, v)
                    force_components.append(self.physics.force_analysis.copy())
                
                # Extract force components
                self.results['force_gradient'] = np.array([fc.get('force_gradient', 0) for fc in force_components])
                self.results['force_reluctance'] = np.array([fc.get('force_reluctance', 0) for fc in force_components])
                self.results['force_lorentz'] = np.array([fc.get('force_lorentz', 0) for fc in force_components])
                self.results['force_maxwell'] = np.array([fc.get('force_maxwell', 0) for fc in force_components])
                self.results['force_eddy'] = np.array([fc.get('force_eddy', 0) for fc in force_components])
                self.results['power_loss_eddy'] = np.array([fc.get('power_loss_eddy', 0) for fc in force_components])
            
            # Advanced physics if enhanced methods available
            if hasattr(self.physics, 'calculate_eddy_current_effects'):
                eddy_effects = []
                for I, x, v in zip(self.results['current'], self.results['position'], self.results['velocity']):
                    if abs(I) > 1e-6 and abs(v) > 1e-6:
                        effects = self.physics.calculate_eddy_current_effects(I, v, x)
                        eddy_effects.append(effects)
                    else:
                        eddy_effects.append({'skin_depth': np.inf, 'induced_current': 0, 'opposing_force': 0})
                
                self.results['skin_depth'] = np.array([ef.get('skin_depth', np.inf) for ef in eddy_effects])
                self.results['eddy_current_magnitude'] = np.array([ef.get('induced_current', 0) for ef in eddy_effects])
                
            # Magnetic field analysis
            self.results['magnetic_field'] = np.array([
                self.physics.magnetic_field_solenoid_on_axis(pos, I) 
                for pos, I in zip(self.results['position'], self.results['current'])
            ])
            
            # Inductance gradient
            self.results['inductance_gradient'] = np.array([
                self.physics.get_inductance_gradient(pos) for pos in self.results['position']
            ])
            
            # Power decomposition
            self.results['power_mechanical'] = self.results['force_total'] * self.results['velocity']
            self.results['power_loss_resistive'] = self.results['current']**2 * self.physics.total_resistance
            
            # Magnetic energy
            self.results['energy_magnetic'] = 0.5 * self.results['inductance'] * self.results['current']**2
            
            # Physics validation if available
            if hasattr(self.physics, 'calculate_field_with_error_estimate'):
                field_validation = []
                for pos, I in zip(self.results['position'][:10], self.results['current'][:10]):  # Sample first 10 points
                    if abs(I) > 1e-6:
                        validation = self.physics.calculate_field_with_error_estimate(pos, I)
                        field_validation.append(validation.get('relative_error_estimate', 0))
                    else:
                        field_validation.append(0)
                self.results['field_accuracy'] = np.mean(field_validation) if field_validation else 0
            
        except Exception as e:
            print(f"Warning: Enhanced physics analysis failed: {e}")
            # Provide fallback values
            self.results['force_gradient'] = np.zeros_like(self.results['force_total'])
            self.results['force_reluctance'] = np.zeros_like(self.results['force_total'])
            self.results['force_lorentz'] = np.zeros_like(self.results['force_total'])
            self.results['force_maxwell'] = np.zeros_like(self.results['force_total'])
            self.results['force_eddy'] = np.zeros_like(self.results['force_total'])
            self.results['skin_depth'] = np.full_like(self.results['force_total'], np.inf)
            self.results['eddy_current_magnitude'] = np.zeros_like(self.results['force_total'])
            self.results['power_mechanical'] = self.results['force_total'] * self.results['velocity']
            self.results['power_loss_resistive'] = self.results['current']**2 * self.physics.total_resistance
            self.results['energy_magnetic'] = 0.5 * self.results['inductance'] * self.results['current']**2
        
        # Set backward compatibility aliases
        self.results['force'] = self.results['force_total']
        self.results['power'] = self.results['power_electrical']
        
    def _print_results(self):
        """Print comprehensive simulation results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        
        print(f"Exit reason: {self.simulation_info['exit_reason']}")
        print(f"Simulation time: {self.simulation_info['duration']:.3f} seconds")
        print(f"Integration steps: {self.simulation_info['total_steps']}")
        
        print(f"\nProjectile Performance:")
        print(f"  Final velocity: {self.simulation_info['final_velocity']:.2f} m/s")
        print(f"  Final kinetic energy: {0.5 * self.physics.proj_mass * self.simulation_info['final_velocity']**2:.3f} J")
        print(f"  Energy efficiency: {self.simulation_info['efficiency'] * 100:.2f}%")
        
        if 'max_current' in self.simulation_info:
            print(f"\nCircuit Performance:")
            max_current = self.simulation_info.get('max_current', 0)
            max_force = self.simulation_info.get('max_force', 0)
            print(f"  Maximum current: {max_current:.1f} A")
            print(f"  Maximum force: {max_force:.1f} N")
            
            # Calculate peak power
            if self.results['power_electrical'] is not None:
                max_power = np.max(self.results['power_electrical'])
                print(f"  Peak power: {max_power:.0f} W")
        
        # Energy analysis
        initial_energy = self.physics.initial_energy
        final_kinetic = 0.5 * self.physics.proj_mass * self.simulation_info['final_velocity']**2
        
        print(f"\nEnergy Analysis:")
        print(f"  Initial capacitor energy: {initial_energy:.3f} J")
        print(f"  Final kinetic energy: {final_kinetic:.3f} J")
        print(f"  Energy lost to resistance: {initial_energy - final_kinetic:.3f} J")
        print(f"  Resistive loss percentage: {((initial_energy - final_kinetic)/initial_energy)*100:.1f}%")
        
        # Performance metrics
        specific_energy = final_kinetic / self.physics.proj_mass  # J/kg
        momentum = self.physics.proj_mass * self.simulation_info['final_velocity']
        
        print(f"\nPerformance Metrics:")
        print(f"  Specific energy: {specific_energy:.0f} J/kg")
        print(f"  Momentum: {momentum*1000:.2f} g⋅m/s")
        print(f"  Muzzle energy: {final_kinetic:.3f} J")
        
        # Compare to theoretical maximum
        theoretical_max_velocity = np.sqrt(2 * initial_energy / self.physics.proj_mass)
        velocity_ratio = self.simulation_info['final_velocity'] / theoretical_max_velocity
        
        print(f"\nTheoretical Comparison:")
        print(f"  Theoretical max velocity: {theoretical_max_velocity:.2f} m/s")
        print(f"  Achieved fraction: {velocity_ratio:.3f}")
    
    def _get_summary_results(self):
        """
        Get summary results dictionary.
        
        Returns:
            dict: Summary of key simulation results
        """
        summary = {
            'final_velocity_ms': self.simulation_info['final_velocity'],
            'efficiency_percent': self.simulation_info['efficiency'] * 100,
            'final_kinetic_energy_J': 0.5 * self.physics.proj_mass * self.simulation_info['final_velocity']**2,
            'projectile_mass_g': self.physics.proj_mass * 1000,
            'initial_energy_J': self.physics.initial_energy,
            'simulation_time_s': self.simulation_info['duration'],
            'exit_reason': self.simulation_info['exit_reason']
        }
        
        if 'max_current' in self.simulation_info:
            summary.update({
                'max_current_A': self.simulation_info['max_current'],
                'max_force_N': self.simulation_info['max_force']
            })
        
        return summary
    
    def save_results(self, output_dir="simulation_results"):
        """
        Save simulation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save configuration
        config_file = output_path / "simulation_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Save summary results
        summary_file = output_path / "simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'simulation_info': self.simulation_info,
                'summary': self._get_summary_results()
            }, f, indent=4, default=str)
        
        # Save detailed time-series data if available
        if self.results['time'] is not None:
            data_file = output_path / "time_series_data.npz"
            np.savez_compressed(data_file, **self.results)
            
            # Also save as CSV for easy analysis
            csv_file = output_path / "time_series_data.csv"
            
            # Prepare data for CSV
            csv_data = {
                'time_s': self.results['time'],
                'charge_C': self.results['charge'],
                'current_A': self.results['current'],
                'position_m': self.results['position'],
                'velocity_ms': self.results['velocity'],
                'force_N': self.results['force_total'],
                'inductance_H': self.results['inductance'],
                'power_W': self.results['power_electrical'],
                'energy_capacitor_J': self.results['energy_capacitor'],
                'energy_kinetic_J': self.results['energy_kinetic']
            }
            
            # Create DataFrame-like structure and save
            try:
                import pandas as pd
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_file, index=False)
            except ImportError:
                # Fallback to manual CSV writing if pandas not available
                import csv
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_data.keys())
                    for i in range(len(self.results['time'])):
                        writer.writerow([csv_data[key][i] for key in csv_data.keys()])
        
        print(f"Results saved to: {output_path.absolute()}")
    
    def plot_results(self, save_plots=True, output_dir="simulation_results"):
        """
        Create comprehensive plots of simulation results.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        if self.results['time'] is None:
            print("No detailed results available for plotting.")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Coilgun Simulation Results', fontsize=16, fontweight='bold')
        
        t = self.results['time'] * 1000  # Convert to milliseconds
        
        # Current vs time
        axes[0, 0].plot(t, self.results['current'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Current (A)')
        axes[0, 0].set_title('Coil Current')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity vs time
        axes[0, 1].plot(t, self.results['velocity'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Projectile Velocity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Position vs time
        position_mm = self.results['position'] * 1000
        axes[1, 0].plot(t, position_mm, 'g-', linewidth=2)
        axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5, label='Coil entrance')
        axes[1, 0].axhline(self.physics.coil_length * 1000, color='k', linestyle='--', alpha=0.5, label='Coil exit')
        axes[1, 0].axhline(self.physics.coil_center * 1000, color='r', linestyle=':', alpha=0.7, label='Coil center')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Position (mm)')
        axes[1, 0].set_title('Projectile Position')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Force vs time
        axes[1, 1].plot(t, self.results['force_total'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Force (N)')
        axes[1, 1].set_title('Magnetic Force')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Energy vs time
        axes[2, 0].plot(t, self.results['energy_capacitor'], 'c-', linewidth=2, label='Capacitor')
        axes[2, 0].plot(t, self.results['energy_kinetic'], 'orange', linewidth=2, label='Kinetic')
        axes[2, 0].set_xlabel('Time (ms)')
        axes[2, 0].set_ylabel('Energy (J)')
        axes[2, 0].set_title('Energy Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Inductance vs position
        axes[2, 1].plot(position_mm, self.results['inductance'] * 1e6, 'purple', linewidth=2)
        axes[2, 1].set_xlabel('Position (mm)')
        axes[2, 1].set_ylabel('Inductance (µH)')
        axes[2, 1].set_title('Inductance vs Position')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            plot_file = output_path / "simulation_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_file}")
        
        plt.show()


class MultiStageCoilgunSimulation:
    """
    Multi-stage coilgun simulation class that handles sequential stages with velocity transfer.
    """
    
    def __init__(self, config_file):
        """
        Initialize multi-stage simulation with configuration file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config_file = config_file
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        if not self.config.get("multi_stage", {}).get("enabled", False):
            raise ValueError("Configuration file is not set up for multi-stage simulation")
        
        self.num_stages = self.config["multi_stage"]["num_stages"]
        self.shared_settings = self.config["multi_stage"]["shared_settings"]
        self.stage_groups = self.config["multi_stage"]["stage_groups"]
        
        # Initialize results storage for all stages
        self.stage_results = []
        self.aggregated_results = {
            'time': [],
            'charge': [],
            'current': [],
            'position': [],
            'velocity': [],
            'force': [],
            'inductance': [],
            'power': [],
            'energy_capacitor': [],
            'energy_kinetic': [],
            'stage_transitions': []  # Track when stages transition
        }
        
        # Overall simulation metadata
        self.simulation_info = {
            'config_file': config_file,
            'num_stages': self.num_stages,
            'start_time': None,
            'end_time': None,
            'duration': None,
            'total_steps': 0,
            'final_velocity': None,
            'overall_efficiency': None,
            'stage_efficiencies': [],
            'stage_final_velocities': [],
            'stage_durations': [],
            'total_initial_energy': 0,
            'total_final_kinetic_energy': 0
        }
    
    def create_stage_config(self, stage_num):
        """
        Create a temporary configuration file for a specific stage.
        
        Args:
            stage_num: Stage number (1-indexed)
            
        Returns:
            str: Path to temporary stage configuration file
        """
        stage_info = self.config["stages"][stage_num - 1]  # Convert to 0-indexed
        
        # Build single-stage config from multi-stage config
        stage_config = {}
        
        # Add stage-specific settings
        for key in ["coil", "capacitor", "simulation", "circuit_model", "magnetic_model", "output"]:
            if key in stage_info:
                stage_config[key] = stage_info[key]
            elif key in self.config.get("shared", {}):
                stage_config[key] = self.config["shared"][key]
            else:
                raise ValueError(f"Missing configuration for {key} in stage {stage_num}")
        
        # Add projectile (always shared, but may need position/velocity updates)
        stage_config["projectile"] = self.config["shared"]["projectile"].copy()
        
        # Add timing optimization configuration
        timing_config = {
            "enabled": True,
            "pre_charge": True,
            "optimal_force_timing": True,
            "charge_time_factor": 3.0,
            "optimal_force_position": 0.3,
            "turn_off_position": 0.7,
            "gradual_turn_on": False,
            "turn_on_duration": 1e-3
        }
        
        # Override with user-specified timing config if present
        if "timing_optimization" in self.config.get("shared", {}):
            timing_config.update(self.config["shared"]["timing_optimization"])
        
        stage_config["timing_optimization"] = timing_config
        
        # Update projectile initial conditions for this stage
        if stage_num > 1:
            # Get final velocity from previous stage
            prev_stage_results = self.stage_results[stage_num - 2]
            stage_config["projectile"]["initial_velocity"] = prev_stage_results["final_velocity"]
            
            # Reset position for the new stage (projectile starts before new coil)
            # Assume projectile starts at same relative position to each stage
            initial_pos_relative = self.config["shared"]["projectile"]["initial_position"]
            stage_config["projectile"]["initial_position"] = initial_pos_relative
        
        # Save temporary config file
        temp_config_file = f"temp_stage_{stage_num}_config.json"
        with open(temp_config_file, 'w') as f:
            json.dump(stage_config, f, indent=4)
        
        return temp_config_file
    
    def run_simulation(self, save_data=True, verbose=True, show_progress=True):
        """
        Execute the complete multi-stage coilgun simulation.
        
        Args:
            save_data: Whether to save detailed time-series data
            verbose: Whether to print progress and results
            show_progress: Whether to show integration progress bar for each stage
            
        Returns:
            dict: Aggregated simulation results and analysis
        """
        if verbose:
            print("=" * 70)
            print("MULTI-STAGE COILGUN SIMULATION")
            print("=" * 70)
            print(f"Number of stages: {self.num_stages}")
            print(f"Shared settings: {', '.join(self.shared_settings)}")
            print(f"Stage groups: {self.stage_groups}")
        
        # Record overall start time
        self.simulation_info['start_time'] = time.time()
        
        # Track cumulative time offset for aggregated results
        time_offset = 0.0
        
        try:
            # Run each stage sequentially
            for stage_num in range(1, self.num_stages + 1):
                if verbose:
                    print(f"\n" + "="*50)
                    print(f"RUNNING STAGE {stage_num}/{self.num_stages}")
                    print("="*50)
                
                # Create temporary config for this stage
                stage_config_file = self.create_stage_config(stage_num)
                
                try:
                    # Initialize and run single-stage simulation
                    stage_sim = CoilgunSimulation(stage_config_file)
                    
                    # Set previous stage velocity for timing optimization (stages 2+)
                    if stage_num > 1:
                        prev_velocity = self.stage_results[stage_num - 2]["final_velocity"]
                        stage_sim.physics.set_previous_stage_velocity(prev_velocity)
                        
                        if verbose:
                            print(f"  Timing optimization enabled:")
                            print(f"    Previous stage final velocity: {prev_velocity:.2f} m/s")
                            if hasattr(stage_sim.physics, 'timing_info'):
                                timing = stage_sim.physics.timing_info
                                print(f"    L/R time constant: {timing['time_constant']*1000:.1f} ms")
                                print(f"    Charge time needed: {timing['charge_time_needed']*1000:.1f} ms")
                                print(f"    Switch-on delay: {timing['switch_on_time']*1000:.1f} ms")
                                print(f"    Switch-off time: {timing['switch_off_time']*1000:.1f} ms")
                    
                    stage_results = stage_sim.run_simulation(save_data=save_data, verbose=verbose, show_progress=show_progress)
                    
                    # Store stage results
                    stage_results['stage_number'] = stage_num
                    stage_results['stage_duration'] = stage_sim.simulation_info['duration']
                    stage_results['stage_efficiency'] = stage_sim.simulation_info['efficiency']
                    stage_results['final_velocity'] = stage_sim.simulation_info['final_velocity']
                    stage_results['max_current'] = stage_sim.simulation_info.get('max_current', 0)
                    stage_results['max_force'] = stage_sim.simulation_info.get('max_force', 0)
                    
                    # Add stage simulation object for detailed data access
                    stage_results['simulation_object'] = stage_sim
                    
                    self.stage_results.append(stage_results)
                    
                    # Update simulation info
                    self.simulation_info['stage_final_velocities'].append(stage_results['final_velocity'])
                    self.simulation_info['stage_efficiencies'].append(stage_results['stage_efficiency'])
                    self.simulation_info['stage_durations'].append(stage_results['stage_duration'])
                    self.simulation_info['total_steps'] += stage_sim.simulation_info['total_steps']
                    
                    # Add to total initial energy
                    self.simulation_info['total_initial_energy'] += stage_results['initial_energy_J']
                    
                    # Aggregate time-series data if available
                    if save_data and stage_sim.results['time'] is not None:
                        # Adjust time to be cumulative across stages
                        adjusted_time = stage_sim.results['time'] + time_offset
                        
                        # Add stage transition marker
                        if stage_num > 1:
                            self.aggregated_results['stage_transitions'].append(time_offset)
                        
                        # Append data
                        self.aggregated_results['time'].extend(adjusted_time)
                        self.aggregated_results['charge'].extend(stage_sim.results['charge'])
                        self.aggregated_results['current'].extend(stage_sim.results['current'])
                        self.aggregated_results['position'].extend(stage_sim.results['position'])
                        self.aggregated_results['velocity'].extend(stage_sim.results['velocity'])
                        self.aggregated_results['force'].extend(stage_sim.results['force_total'])
                        self.aggregated_results['inductance'].extend(stage_sim.results['inductance'])
                        self.aggregated_results['power'].extend(stage_sim.results['power_electrical'])
                        self.aggregated_results['energy_capacitor'].extend(stage_sim.results['energy_capacitor'])
                        self.aggregated_results['energy_kinetic'].extend(stage_sim.results['energy_kinetic'])
                        
                        # Update time offset for next stage
                        time_offset = adjusted_time[-1]
                    
                    if verbose:
                        print(f"Stage {stage_num} completed:")
                        print(f"  Final velocity: {stage_results['final_velocity']:.2f} m/s")
                        print(f"  Efficiency: {stage_results['stage_efficiency']*100:.2f}%")
                        print(f"  Duration: {stage_results['stage_duration']:.3f} s")
                
                finally:
                    # Clean up temporary config file
                    if Path(stage_config_file).exists():
                        Path(stage_config_file).unlink()
            
            # Calculate overall results
            self.simulation_info['final_velocity'] = self.stage_results[-1]['final_velocity']
            
            # Calculate overall efficiency
            total_final_kinetic = 0.5 * self.stage_results[0]['simulation_object'].physics.proj_mass * self.simulation_info['final_velocity']**2
            self.simulation_info['total_final_kinetic_energy'] = total_final_kinetic
            self.simulation_info['overall_efficiency'] = total_final_kinetic / self.simulation_info['total_initial_energy']
            
            # Record end time
            self.simulation_info['end_time'] = time.time()
            self.simulation_info['duration'] = (self.simulation_info['end_time'] - 
                                               self.simulation_info['start_time'])
            
            if verbose:
                self._print_overall_results()
            
            return self._get_aggregated_summary_results()
            
        except Exception as e:
            print(f"Multi-stage simulation failed: {str(e)}")
            raise
    
    def _print_overall_results(self):
        """Print comprehensive multi-stage simulation results."""
        print("\n" + "=" * 70)
        print("MULTI-STAGE SIMULATION RESULTS")
        print("=" * 70)
        
        print(f"Total simulation time: {self.simulation_info['duration']:.3f} seconds")
        print(f"Total integration steps: {self.simulation_info['total_steps']}")
        
        print(f"\nOverall Performance:")
        print(f"  Final velocity: {self.simulation_info['final_velocity']:.2f} m/s")
        print(f"  Overall efficiency: {self.simulation_info['overall_efficiency'] * 100:.2f}%")
        print(f"  Total initial energy: {self.simulation_info['total_initial_energy']:.1f} J")
        print(f"  Final kinetic energy: {self.simulation_info['total_final_kinetic_energy']:.1f} J")
        
        print(f"\nStage-by-Stage Results:")
        for i, stage_result in enumerate(self.stage_results):
            stage_num = i + 1
            print(f"  Stage {stage_num}:")
            print(f"    Final velocity: {stage_result['final_velocity']:.2f} m/s")
            print(f"    Efficiency: {stage_result['stage_efficiency']*100:.2f}%")
            print(f"    Max current: {stage_result.get('max_current', 0):.1f} A")
            print(f"    Max force: {stage_result.get('max_force', 0):.1f} N")
            print(f"    Duration: {stage_result['stage_duration']:.3f} s")
        
        # Velocity progression
        print(f"\nVelocity Progression:")
        print(f"  Initial: 0.0 m/s")
        for i, velocity in enumerate(self.simulation_info['stage_final_velocities']):
            print(f"  After stage {i+1}: {velocity:.2f} m/s")
    
    def _get_aggregated_summary_results(self):
        """
        Get aggregated summary results dictionary.
        
        Returns:
            dict: Summary of key multi-stage simulation results
        """
        summary = {
            'multi_stage': True,
            'num_stages': self.num_stages,
            'final_velocity_ms': self.simulation_info['final_velocity'],
            'overall_efficiency_percent': self.simulation_info['overall_efficiency'] * 100,
            'total_initial_energy_J': self.simulation_info['total_initial_energy'],
            'final_kinetic_energy_J': self.simulation_info['total_final_kinetic_energy'],
            'simulation_time_s': self.simulation_info['duration'],
            'stage_final_velocities_ms': self.simulation_info['stage_final_velocities'],
            'stage_efficiencies_percent': [eff * 100 for eff in self.simulation_info['stage_efficiencies']],
            'stage_durations_s': self.simulation_info['stage_durations'],
            'projectile_mass_g': self.stage_results[0]['projectile_mass_g']
        }
        
        # Add max current and force across all stages
        max_current = max(stage.get('max_current', 0) for stage in self.stage_results)
        max_force = max(stage.get('max_force', 0) for stage in self.stage_results)
        summary['max_current_A'] = max_current
        summary['max_force_N'] = max_force
        
        return summary
    
    def save_results(self, output_dir="multistage_simulation_results"):
        """
        Save multi-stage simulation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save configuration
        config_file = output_path / "simulation_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Save overall summary results
        summary_file = output_path / "multistage_simulation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'simulation_info': self.simulation_info,
                'summary': self._get_aggregated_summary_results(),
                'stage_results': [
                    {k: v for k, v in stage.items() if k != 'simulation_object'}
                    for stage in self.stage_results
                ]
            }, f, indent=4, default=str)
        
        # Save aggregated time-series data if available
        if self.aggregated_results['time']:
            # Convert lists to numpy arrays
            aggregated_arrays = {}
            for key, data in self.aggregated_results.items():
                if key != 'stage_transitions' and data:
                    aggregated_arrays[key] = np.array(data)
            
            # Save as compressed numpy file
            data_file = output_path / "multistage_time_series_data.npz"
            np.savez_compressed(data_file, **aggregated_arrays, 
                               stage_transitions=np.array(self.aggregated_results['stage_transitions']))
            
            # Also save as CSV for easy analysis
            csv_file = output_path / "multistage_time_series_data.csv"
            
            # Prepare data for CSV
            csv_data = {
                'time_s': self.aggregated_results['time'],
                'charge_C': self.aggregated_results['charge'],
                'current_A': self.aggregated_results['current'],
                'position_m': self.aggregated_results['position'],
                'velocity_ms': self.aggregated_results['velocity'],
                'force_N': self.aggregated_results['force'],
                'inductance_H': self.aggregated_results['inductance'],
                'power_W': self.aggregated_results['power'],
                'energy_capacitor_J': self.aggregated_results['energy_capacitor'],
                'energy_kinetic_J': self.aggregated_results['energy_kinetic']
            }
            
            # Create DataFrame-like structure and save
            try:
                import pandas as pd
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_file, index=False)
            except ImportError:
                # Fallback to manual CSV writing if pandas not available
                import csv
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_data.keys())
                    for i in range(len(self.aggregated_results['time'])):
                        writer.writerow([csv_data[key][i] for key in csv_data.keys()])
        
        # Save individual stage results
        for i, stage_result in enumerate(self.stage_results):
            stage_dir = output_path / f"stage_{i+1}_results"
            if 'simulation_object' in stage_result:
                stage_result['simulation_object'].save_results(str(stage_dir))
        
        print(f"Multi-stage results saved to: {output_path.absolute()}")


def parametric_study(base_config_file, parameter_name, parameter_values, output_dir="parametric_study"):
    """
    Perform a parametric study by varying a single parameter.
    
    Args:
        base_config_file: Base configuration file
        parameter_name: Name of parameter to vary (e.g., 'capacitor.initial_voltage')
        parameter_values: List of values to test
        output_dir: Directory to save results
    """
    print(f"Starting parametric study: {parameter_name}")
    print(f"Testing {len(parameter_values)} values...")
    
    results = []
    
    for i, value in enumerate(parameter_values):
        print(f"  Run {i+1}/{len(parameter_values)}: {parameter_name} = {value}")
        
        # Load base configuration
        with open(base_config_file, 'r') as f:
            config = json.load(f)
        
        # Modify parameter
        keys = parameter_name.split('.')
        obj = config
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
        
        # Save modified config
        temp_config = f"temp_config_{i}.json"
        with open(temp_config, 'w') as f:
            json.dump(config, f)
        
        try:
            # Run simulation
            sim = CoilgunSimulation(temp_config)
            result = sim.run_simulation(save_data=False, verbose=False)
            result['parameter_value'] = value
            results.append(result)
            
        except Exception as e:
            print(f"    Simulation failed: {e}")
            results.append({'parameter_value': value, 'failed': True})
        
        # Clean up
        Path(temp_config).unlink()
    
    # Save parametric study results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / f"parametric_study_{parameter_name.replace('.', '_')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print(f"Parametric study completed. Results saved to: {results_file}")
    return results


def find_config_files():
    """Find all JSON configuration files in the project directory"""
    current_dir = Path(".")
    json_files = list(current_dir.glob("*.json"))
    
    # Filter to likely config files by checking content
    config_files = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Check if it looks like a coilgun config file
                # For single-stage configs: check top-level keys
                is_single_stage = any(key in data for key in ['coil', 'capacitor', 'projectile', 'simulation'])
                
                # For multi-stage configs: check for multi_stage key and nested structure
                is_multi_stage = (
                    'multi_stage' in data and 
                    data.get('multi_stage', {}).get('enabled', False) and
                    'stages' in data and 
                    'shared' in data
                )
                
                if is_single_stage or is_multi_stage:
                    config_files.append(json_file)
        except (json.JSONDecodeError, IOError):
            # Skip files that can't be read or aren't valid JSON
            continue
    
    return sorted(config_files)

def select_config_file():
    """Interactive selection of configuration file"""
    # Check if config file was provided as command line argument
    if len(sys.argv) >= 2:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            return config_file
        else:
            print(f"Warning: Specified config file '{config_file}' not found.")
            print("Searching for available config files...\n")
    
    # Find available config files
    config_files = find_config_files()
    
    if not config_files:
        print("No coilgun configuration files found in the current directory.")
        print("Please run 'python setup.py' first to create a configuration file.")
        sys.exit(1)
    
    # Present options to user
    print("Available coilgun configuration files:")
    print("-" * 40)
    for i, config_file in enumerate(config_files, 1):
        # Try to read description from config file
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
                description = data.get('description', 'No description available')
            print(f"{i}. {config_file.name}")
            print(f"   Description: {description}")
        except:
            print(f"{i}. {config_file.name}")
        print()
    
    # Get user selection
    while True:
        try:
            choice = input(f"Select configuration file (1-{len(config_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(config_files):
                selected_file = config_files[choice_num - 1]
                print(f"Selected: {selected_file.name}\n")
                return str(selected_file)
            else:
                print(f"Please enter a number between 1 and {len(config_files)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def main():
    """Main function to run coilgun simulation from command line"""
    
    try:
        config_file = select_config_file()
        
        print("=" * 60)
        print("COILGUN SIMULATION SOLVER")
        print("=" * 60)
        print(f"Configuration file: {config_file}")
        
        # Ask user if they want to proceed with simulation
        print(f"\nReady to run simulation with: {Path(config_file).name}")
        proceed = input("Do you want to proceed? (Y/n): ").strip().lower()
        if proceed in ['n', 'no', 'q', 'quit']:
            print("Simulation cancelled by user.")
            sys.exit(0)
        elif proceed == '' or proceed in ['y', 'yes']:
            pass  # Continue
        else:
            print("Invalid input. Proceeding with simulation...")
        
        print("\nStarting simulation...")
        
    except KeyboardInterrupt:
        print("\n\nSimulation cancelled by user (Ctrl+C)")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)
    
    try:
        # Check if this is a multi-stage configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        is_multi_stage = config.get("multi_stage", {}).get("enabled", False)
        
        if is_multi_stage:
            # Use multi-stage simulation
            print("Detected multi-stage configuration")
            sim = MultiStageCoilgunSimulation(config_file)
            results = sim.run_simulation(save_data=True, verbose=True, show_progress=True)
            
            # Create output directory based on config filename
            config_name = Path(config_file).stem
            output_dir = f"results_{config_name}"
            
            # Save detailed results
            print("\n" + "="*50)
            print("SAVING RESULTS")
            print("="*50)
            sim.save_results(output_dir)
            
            # Print summary
            print("\n" + "="*50)
            print("SIMULATION SUMMARY")
            print("="*50)
            print(f"Final velocity: {results['final_velocity_ms']:.1f} m/s")
            print(f"Overall efficiency: {results['overall_efficiency_percent']:.1f}%")
            print(f"Total initial energy: {results['total_initial_energy_J']:.1f} J")
            print(f"Final kinetic energy: {results['final_kinetic_energy_J']:.1f} J")
            print(f"Max current: {results.get('max_current_A', 0):.1f} A")
            print(f"Max force: {results.get('max_force_N', 0):.1f} N")
            print(f"Total simulation time: {results['simulation_time_s']:.3f} s")
            
            print(f"\nStage Performance:")
            for i, (velocity, efficiency) in enumerate(zip(results['stage_final_velocities_ms'], results['stage_efficiencies_percent'])):
                print(f"  Stage {i+1}: {velocity:.1f} m/s ({efficiency:.1f}% efficiency)")
            
            print(f"\nResults saved to directory: {output_dir}/")
            print("- multistage_simulation_summary.json (overall results)")
            print("- multistage_time_series_data.csv (aggregated time series)")
            print("- stage_X_results/ (individual stage results)")
            
        else:
            # Use single-stage simulation
            print("Detected single-stage configuration")
            sim = CoilgunSimulation(config_file)
            results = sim.run_simulation(save_data=True, verbose=True, show_progress=True)
            
            # Create output directory based on config filename
            config_name = Path(config_file).stem
            output_dir = f"results_{config_name}"
            
            # Save detailed results to CSV and JSON
            print("\n" + "="*50)
            print("SAVING RESULTS")
            print("="*50)
            sim.save_results(output_dir)
            
            # Print summary
            print("\n" + "="*50)
            print("SIMULATION SUMMARY")
            print("="*50)
            print(f"Final velocity: {results['final_velocity_ms']:.1f} m/s")
            print(f"Efficiency: {results['efficiency_percent']:.1f}%")
            print(f"Max current: {results.get('max_current_A', 0):.1f} A")
            print(f"Max force: {results.get('max_force_N', 0):.1f} N")
            print(f"Simulation time: {results['simulation_time_s']:.3f} s")
            print(f"Exit reason: {results['exit_reason']}")
            
            # Calculate key performance metrics
            initial_energy = results['initial_energy_J']
            final_kinetic_energy = 0.5 * sim.physics.proj_mass * results['final_velocity_ms']**2
            energy_transferred = final_kinetic_energy
            
            print(f"\nENERGY ANALYSIS:")
            print(f"Initial capacitor energy: {initial_energy:.1f} J")
            print(f"Final kinetic energy: {final_kinetic_energy:.1f} J")
            print(f"Energy transferred to projectile: {energy_transferred:.1f} J")
            
            print(f"\nResults saved to directory: {output_dir}/")
            print("- time_series_data.csv (detailed time series)")
            print("- simulation_summary.json (summary results)")
        
        print(f"\nTo view detailed visualizations, run:")
        print(f"python view.py {config_file}")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user (Ctrl+C)")
        print("Simulation results may be incomplete.")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nSimulation terminated due to error.")
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle signals gracefully"""
    print("\n\nReceived interrupt signal.")
    print("Cleaning up and exiting gracefully...")
    sys.exit(0)


if __name__ == '__main__':
    import os
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
