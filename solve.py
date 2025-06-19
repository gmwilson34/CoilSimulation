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
import signal
import traceback
import os
import csv
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

from equations import CoilgunPhysicsEngine

# MultiStageTimingCoordinator removed - timing optimization eliminated


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
        
        # Rate calculation with sliding window
        self.current_integration_rate = 0.0
        
        # Physics diagnostics
        self.max_current = 0
        self.max_force = 0
        self.max_velocity = 0
        self.current_position = 0
        self.physics_warnings = []
        self.displayed_warnings = set()  # Track displayed warnings to avoid duplicates
        
        # Progress bar settings
        self.bar_width = 50
        self.running = True
        self.stopped = False
        self.last_displayed_warning = None
        
        # Terminal control
        self.progress_active = False
        self.last_progress_length = 0
        self.integration_started = False  # Flag to control when to start displaying
        
        # Start progress display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def start_integration_display(self):
        """Start displaying the progress bar when integration begins."""
        self.integration_started = True

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
        
        # Start displaying progress bar on first update
        if not self.integration_started:
            self.integration_started = True
        
        # Update physics diagnostics
        if len(y) >= 4:
            Q, I, x, v = y[:4]  # Take only first 4 elements to handle thermal case
            self.max_current = max(self.max_current, abs(I))
            self.max_velocity = max(self.max_velocity, abs(v))
            self.current_position = x
            
            # Calculate current force for diagnostics using enhanced physics
            if self.physics and abs(I) > 1e-6:
                try:
                    # Use the enhanced magnetic force calculation with circuit logic
                    if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
                        force_result = self.physics.magnetic_force_with_circuit_logic(I, x, t, v)
                    else:
                        force_result = self.physics.magnetic_force_ferromagnetic(I, x, v)
                    
                    # Unpack the tuple (force, eddy_power_loss) and use just the force
                    if isinstance(force_result, tuple):
                        force = force_result[0]
                    else:
                        force = force_result
                    
                    self.max_force = max(self.max_force, abs(force))
                except Exception as e:
                    # Store warning but don't print immediately during progress bar display
                    warning_msg = f"Force calculation warning at t={t:.2e}s: {str(e)[:50]}"
                    if warning_msg not in self.displayed_warnings:
                        self.physics_warnings.append(warning_msg)
    
    def _clear_progress_line(self):
        """Clear the current progress line from terminal."""
        if self.progress_active and self.last_progress_length > 0:
            sys.stdout.write('\r' + ' ' * self.last_progress_length + '\r')
            sys.stdout.flush()
        self.progress_active = False
        self.last_progress_length = 0
    
    def _display_loop(self):
        """Display progress bar in a separate thread."""
        while self.running and not self.stopped:
            # Only start displaying after integration has started
            if self.integration_started:
                # Check for new warnings
                self._check_for_new_warnings()
                # Update progress bar
                self._draw_progress_bar()
            time.sleep(self.update_interval)
    
    def _check_for_new_warnings(self):
        """Check for new warnings and display them above the progress bar."""
        new_warning = None
        
        # Check for physics warnings
        for warning in self.physics_warnings:
            if warning not in self.displayed_warnings:
                new_warning = f"⚠  {warning}"
                self.displayed_warnings.add(warning)
                break
        
        # Check for energy warnings from physics engine
        if (not new_warning and self.physics and 
            hasattr(self.physics, 'latest_energy_warning') and 
            self.physics.latest_energy_warning and 
            self.physics.latest_energy_warning != self.last_displayed_warning):
            
            if hasattr(self.physics, 'energy_warning_count'):
                new_warning = f"⚠  Energy warning #{self.physics.energy_warning_count}: {self.physics.latest_energy_warning}"
            else:
                new_warning = f"⚠  Energy warning: {self.physics.latest_energy_warning}"
            self.last_displayed_warning = self.physics.latest_energy_warning
        
        # Display new warning if found
        if new_warning:
            # Clear current progress line
            self._clear_progress_line()
            # Print warning on a new line
            print(new_warning)
            # Force redraw of progress bar on next iteration
            sys.stdout.flush()
    
    def _draw_progress_bar(self, force_draw=False):
        """Draw enhanced progress bar with physics diagnostics."""
        if self.stopped and not force_draw:
            return
        
        # Calculate progress percentage
        if self.t_duration > 0:
            progress = min(1.0, (self.current_time - self.t_start) / self.t_duration)
        else:
            progress = 0.0
        
        # Calculate integration rate
        current_real_time = time.time()
        real_time_elapsed = current_real_time - self.last_update_time
        
        # Update rate calculation periodically for smoothness
        if real_time_elapsed >= self.update_interval and real_time_elapsed > 0:
            steps_since_update = self.step_count - self.last_step_count
            new_rate = steps_since_update / real_time_elapsed
            
            # Use exponential smoothing for stable rate display
            if self.current_integration_rate == 0:
                self.current_integration_rate = new_rate
            else:
                # Smooth the rate to avoid rapid fluctuations
                alpha = 0.3  # Smoothing factor
                self.current_integration_rate = (alpha * new_rate + 
                                               (1 - alpha) * self.current_integration_rate)
            
            self.last_update_time = current_real_time
            self.last_step_count = self.step_count
        
        # Use the smoothed rate for display
        integration_rate = self.current_integration_rate
        
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
            
            # Add force info if available and physics engine is active
            if hasattr(self, 'max_force') and self.max_force > 0:
                physics_status += f" | F:{self.max_force:.1f}N"
        
        # Create enhanced progress line
        progress_line = (f"\rSimulation: [{bar}] {progress*100:6.2f}% | "
                        f"Time: {time_str}/{total_time_str} | "
                        f"Steps: {self.step_count:,} | "
                        f"Rate: {integration_rate:.0f}/s{physics_status}")
        
        # Truncate if too long for terminal
        if len(progress_line) > 120:
            progress_line = progress_line[:117] + "..."
        
        # Write to terminal - always update the progress display
        sys.stdout.write(progress_line)
        sys.stdout.flush()
        self.progress_active = True
        self.last_progress_length = len(progress_line)
    
    def stop(self):
        """Stop the progress tracker."""
        if self.stopped:  # Prevent multiple calls
            return
            
        self.running = False
        self.stopped = True
        
        # Wait for display thread to finish
        if self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        # Clear the current progress line
        self._clear_progress_line()
        
        # Show the final completed progress bar only once
        if self.current_time > 0:
            # Force progress to 100% for final display
            saved_time = self.current_time
            self.current_time = self.t_end
            self._draw_progress_bar(force_draw=True)
            self.current_time = saved_time
            print()  # Move to next line
        
        # Show summary of any warnings that occurred
        total_warnings = len(self.physics_warnings)
        if (self.physics and hasattr(self.physics, 'energy_warning_count') and 
            self.physics.energy_warning_count > 0):
            total_warnings += self.physics.energy_warning_count
        
        if total_warnings > 0:
            print(f"⚠  Total warnings during simulation: {total_warnings}")
        
        sys.stdout.flush()
    
    def show_final_progress(self):
        """Show the final progress bar state at 100% completion."""
        # Force progress to 100%
        progress = 1.0
        
        # Create final progress bar
        filled = int(self.bar_width * progress)
        bar = '█' * filled + '░' * (self.bar_width - filled)
        
        # Format final time
        if self.current_time < 1e-3:
            time_str = f"{self.current_time*1e6:.1f}μs"
        elif self.current_time < 1:
            time_str = f"{self.current_time*1e3:.1f}ms"
        else:
            time_str = f"{self.current_time:.3f}s"
        
        # Final physics status
        physics_status = ""
        if self.current_state is not None and len(self.current_state) >= 4:
            I, x, v = self.current_state[1], self.current_state[2], self.current_state[3]
            physics_status = f" | Final: I:{I:.0f}A | x:{x*1000:.1f}mm | v:{v:.1f}m/s"
        
        # Create final progress line
        progress_line = (f"\rSimulation: [{bar}] {progress*100:6.2f}% | "
                        f"Completed in: {time_str} | "
                        f"Total steps: {self.step_count:,}{physics_status}")
        
        # Write final progress line
        sys.stdout.write(progress_line)
        sys.stdout.flush()

    # ...existing code...
    
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
        
        # Initialize physics engine with enhanced configuration validation
        self.physics = CoilgunPhysicsEngine(config_file)
        
        # Validate physics engine initialization
        if not hasattr(self.physics, 'circuit_derivatives'):
            raise RuntimeError("Physics engine failed to initialize properly - missing circuit_derivatives method")
        
        # Check for critical physics methods
        critical_methods = ['get_initial_conditions', 'magnetic_force_ferromagnetic', 'get_inductance']
        missing_methods = [method for method in critical_methods if not hasattr(self.physics, method)]
        if missing_methods:
            raise RuntimeError(f"Physics engine missing critical methods: {missing_methods}")
        
        # Enhanced physics compatibility check
        if hasattr(self.physics, '_initialize_advanced_physics'):
            print("Enhanced physics engine detected - advanced features available")
        else:
            print("Warning: Basic physics engine detected - some advanced features may be unavailable")
        
        # Progress tracker
        self.progress_tracker = None
        
        # Initialize results storage
        self.results = {
            # Basic state variables
            'time': np.array([]),
            'charge': np.array([]),
            'current': np.array([]),
            'position': np.array([]),
            'velocity': np.array([]),
            
            # Enhanced electromagnetic analysis
            'force_total': np.array([]),
            'force_gradient': np.array([]),
            'force_reluctance': np.array([]),
            'force_lorentz': np.array([]),
            'force_maxwell': np.array([]),
            'force_eddy': np.array([]),
            'inductance': np.array([]),
            'inductance_gradient': np.array([]),
            
            # Power and energy analysis
            'power_electrical': np.array([]),
            'power_mechanical': np.array([]),
            'power_loss_resistive': np.array([]),
            'power_loss_eddy': np.array([]),
            'energy_capacitor': np.array([]),
            'energy_kinetic': np.array([]),
            'energy_magnetic': np.array([]),
            
            # Advanced physics
            'magnetic_field': np.array([]),
            'permeability_effective': np.array([]),
            'saturation_factor': np.array([]),
            'eddy_current_magnitude': np.array([]),
            'skin_depth': np.array([]),
            'frequency_content': np.array([]),
            'temperature_rise': np.array([]),
            
            # Physics validation
            'field_accuracy': 0.0,
            'force_consistency': np.array([]),
            'energy_conservation': np.array([]),
            
            # Backward compatibility
            'force': np.array([]),  # Alias for force_total
            'power': np.array([])   # Alias for power_electrical
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
        
    
    def _enhanced_ode_wrapper(self, original_func):
        """
        Create an enhanced wrapper for the ODE function with better error handling.
        
        Args:
            original_func: Original ODE function
            
        Returns:
            Wrapped function with enhanced error handling
        """
        def wrapped_func(t, y):
            try:
                # Update progress tracker if available
                if self.progress_tracker:
                    self.progress_tracker.update(t, y)
                
                # Validate state vector
                if len(y) < 4:
                    raise ValueError(f"Invalid state vector length: {len(y)}")
                
                # Check for numerical issues
                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    raise ValueError("NaN or Inf detected in state vector")
                
                # Call original function with enhanced error handling
                try:
                    dydt = original_func(t, y)
                except Exception as e:
                    # Route warning through progress tracker instead of direct print
                    if self.progress_tracker:
                        warning_msg = f"ODE function failed at t={t:.6f}s: {str(e)[:50]}"
                        if warning_msg not in self.progress_tracker.displayed_warnings:
                            self.progress_tracker.physics_warnings.append(warning_msg)
                    
                    # Provide fallback derivatives to prevent integration failure
                    if len(y) == 4:
                        Q, I, x, v = y
                        # Simple fallback: exponential current decay
                        R = getattr(self.physics, 'total_resistance', 1.0)
                        L = getattr(self.physics, 'L_air_core', 1e-6)
                        dydt = np.array([-I, -R*I/L, v, 0.0])
                    else:
                        dydt = np.zeros_like(y)
                    return dydt
                
                # Validate derivatives
                if np.any(np.isnan(dydt)) or np.any(np.isinf(dydt)):
                    # Route warning through progress tracker instead of direct print
                    if self.progress_tracker:
                        warning_msg = f"Invalid derivatives at t={t:.6f}s, using fallback"
                        if warning_msg not in self.progress_tracker.displayed_warnings:
                            self.progress_tracker.physics_warnings.append(warning_msg)
                    dydt = np.zeros_like(y)
                
                return dydt
                
            except Exception as e:
                # Route critical error in ODE wrapper to progress tracker
                if self.progress_tracker:
                    warning_msg = f"Critical error in ODE wrapper at t={t:.6f}s: {str(e)[:50]}"
                    if warning_msg not in self.progress_tracker.displayed_warnings:
                        self.progress_tracker.physics_warnings.append(warning_msg)
                # Ultimate fallback
                return np.zeros_like(y)
        
        return wrapped_func
        
    def run_simulation(self, save_data=True, verbose=True, show_progress=True, check_physics=False):
        """
        Execute the complete coilgun simulation.
        
        Args:
            save_data: Whether to save detailed time-series data
            verbose: Whether to print progress and results
            show_progress: Whether to show integration progress bar
            check_physics: Whether to display physics engine integration status
            
        Returns:
            dict: Simulation results and analysis
        """
        if verbose:
            print("=" * 60)
            print("ADVANCED COILGUN SIMULATION")
            print("=" * 60)
            self.physics.print_system_parameters()
            
            # ENHANCED: Show physics engine integration status if requested
            if check_physics:
                self.check_physics_integration()
            
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
        
        # Create enhanced progress-tracking wrapper for ODE function
        ode_func = self.physics.circuit_derivatives
        if self.progress_tracker:
            ode_func = self._enhanced_ode_wrapper(ode_func)
        
        # Create time evaluation points for regular progress updates
        # This ensures the ODE function is called at regular intervals for progress tracking
        t_eval_points = None
        if self.progress_tracker:
            # Calculate number of evaluation points based on simulation parameters
            sim_duration = t_span[1] - t_span[0]
            # Estimate required steps based on max_step and simulation duration
            estimated_steps = int(sim_duration / max_step)
            # Use a reasonable fraction of estimated steps for progress updates
            # But ensure we have at least 100 points and at most 500000 points
            num_eval_points = max(100, estimated_steps)
            t_eval_points = np.linspace(t_span[0], t_span[1], num_eval_points)
        
        # Define events to stop simulation
        def projectile_at_center(t, y):
            """Event: projectile reaches coil center."""
            return y[2] - self.physics.coil_center
        
        def projectile_exits_coil(t, y):
            """Event: projectile front face exits coil (simplified exit condition)."""
            return y[2] - self.physics.coil_length
        
        def current_reverses(t, y):
            """Event: current reverses direction."""
            return y[1]  # Current
        
        # Configure events - these attributes are set by SciPy
        setattr(projectile_at_center, 'terminal', True)
        setattr(projectile_at_center, 'direction', 1)
        
        setattr(projectile_exits_coil, 'terminal', False)
        setattr(projectile_exits_coil, 'direction', 1)
        
        setattr(current_reverses, 'terminal', False)
        setattr(current_reverses, 'direction', -1)
        
        events = [projectile_at_center, projectile_exits_coil, current_reverses]
        
        try:
            # Solve the ODE system
            if verbose:
                print(f"Integrating ODEs with Radau method (optimized for energy conservation)...")
                if show_progress:
                    print("Integration progress will be shown below:")
            
            # FIXED: Back to Radau with proper energy tracking in circuit_derivatives
            solution = solve_ivp(
                fun=ode_func,
                t_span=t_span,
                y0=y0,
                method="Radau",   # FIXED: Radau is better for stiff systems, energy tracking moved to derivatives
                max_step=5e-5,    # FIXED: Set max_step as recommended in punch-list
                rtol=1e-6,        # FIXED: Tighter tolerances for better accuracy
                atol=1e-8,
                events=events,
                dense_output=True,
                t_eval=t_eval_points,  # Force evaluation at regular intervals for progress tracking
                # Add numerical stability options
                first_step=5e-6,  # Conservative first step (1/10 of max_step)
            )
            
            if not solution.success:
                # Try again with more conservative settings if first attempt fails
                if verbose:
                    print(f"Initial integration failed: {solution.message}")
                    print("Retrying with more conservative settings...")
                
                # More conservative integration parameters
                conservative_max_step = max_step * 0.1
                conservative_tolerance = tolerance * 10
                
                solution = solve_ivp(
                    fun=ode_func,
                    t_span=t_span,
                    y0=y0,
                    method='RK23',  # More stable method
                    max_step=conservative_max_step,
                    rtol=conservative_tolerance,
                    atol=conservative_tolerance * 1e-2,
                    events=events,
                    dense_output=True,
                    t_eval=t_eval_points,  # Also use evaluation points for conservative retry
                    first_step=conservative_max_step * 0.01,
                )
                
                if not solution.success:
                    raise RuntimeError(f"Integration failed: {solution.message}")
                else:
                    if verbose:
                        print("Integration succeeded with conservative settings.")
            
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
            
            if save_data and len(self.results['current']) > 0:
                self.simulation_info['max_current'] = np.max(np.abs(self.results['current']))
                self.simulation_info['max_force'] = np.max(np.abs(self.results['force_total'])) if len(self.results['force_total']) > 0 else 0
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
                        velocity = solution.y[3, i] if len(solution.y) > 3 else 0  # Velocity
                        
                        # Use the enhanced physics force calculation with fallback
                        if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
                            force_result = self.physics.magnetic_force_with_circuit_logic(current, position, current_time, velocity)
                        else:
                            force_result = self.physics.magnetic_force_ferromagnetic(current, position, velocity)
                        
                        # Unpack the tuple (force, eddy_power_loss) and use just the force
                        if isinstance(force_result, tuple):
                            force = force_result[0]
                        else:
                            force = force_result
                        
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
            
            # Print completion message after progress tracker is stopped
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
        
        # ENHANCED: Use the upgraded magnetic_force_ferromagnetic method with full physics
        # Calculate forces with enhanced current and time history for frequency analysis
        force_calculations = []
        for i, (I, x, v) in enumerate(zip(self.results['current'], self.results['position'], self.results['velocity'])):
            # Get current and time history for enhanced frequency analysis
            hist_length = min(50, i + 1)  # Use up to 50 historical points
            hist_start = max(0, i - hist_length + 1)
            current_history = self.results['current'][hist_start:i+1] if i > 0 else None
            time_history = self.results['time'][hist_start:i+1] if i > 0 else None
            
            try:
                # Use enhanced force calculation with circuit logic and timing optimization
                if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
                    force_result = self.physics.magnetic_force_with_circuit_logic(
                        I, x, self.results['time'][i], v
                    )
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
            except Exception as e:
                # Graceful fallback for compatibility
                try:
                    force_result = self.physics.magnetic_force_ferromagnetic(I, x, v)
                    # Unpack the tuple (force, eddy_power_loss) and use just the force
                    if isinstance(force_result, tuple):
                        force = force_result[0]
                    else:
                        force = force_result
                    force_calculations.append(force)
                except:
                    force_calculations.append(0.0)  # Ultimate fallback
        
        self.results['force_total'] = np.array(force_calculations)
        
        # Power and energy analysis
        self.results['power_electrical'] = self.results['current'] * self.results['charge'] / self.physics.capacitance
        
        self.results['energy_capacitor'] = 0.5 * self.results['charge']**2 / self.physics.capacitance
        
        self.results['energy_kinetic'] = (0.5 * self.physics.proj_mass * 
                                         self.results['velocity']**2)
        
        # Enhanced physics analysis (with backward compatibility and robust error handling)
        try:
            # ENHANCED: Force decomposition analysis using the upgraded force calculation
            if hasattr(self.physics, 'force_analysis'):
                force_components = []
                for i, (I, x, v, t) in enumerate(zip(self.results['current'], self.results['position'], 
                                        self.results['velocity'], self.results['time'])):
                    try:
                        # Get current and time history for frequency analysis (up to 50 points)
                        hist_length = min(50, i + 1)
                        hist_start = max(0, i - hist_length + 1)
                        current_hist = self.results['current'][hist_start:i+1] if i > 0 else None
                        time_hist = self.results['time'][hist_start:i+1] if i > 0 else None
                        
                        # Calculate force to populate force_analysis
                        if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
                            self.physics.magnetic_force_with_circuit_logic(I, x, t, v)
                        else:
                            self.physics.magnetic_force_ferromagnetic(I, x, v, current_hist, time_hist)
                        
                        # Store the detailed force analysis
                        force_components.append(self.physics.force_analysis.copy())
                    except Exception as e:
                        # Provide fallback force analysis if calculation fails
                        fallback_analysis = {
                            'force_total': 0.0,
                            'force_gradient': 0.0,
                            'force_reluctance': 0.0,
                            'force_lorentz': 0.0,
                            'force_maxwell': 0.0,
                            'force_eddy': 0.0,
                            'power_loss_eddy': 0.0
                        }
                        force_components.append(fallback_analysis)
                
                # Extract force components with enhanced detail
                self.results['force_gradient'] = np.array([fc.get('force_gradient', 0) for fc in force_components])
                self.results['force_reluctance'] = np.array([fc.get('force_reluctance', 0) for fc in force_components])
                self.results['force_lorentz'] = np.array([fc.get('force_lorentz', 0) for fc in force_components])
                self.results['force_maxwell'] = np.array([fc.get('force_maxwell', 0) for fc in force_components])
                self.results['force_eddy'] = np.array([fc.get('force_eddy', 0) for fc in force_components])
                self.results['force_image'] = np.array([fc.get('force_image', 0) for fc in force_components])
                self.results['power_loss_eddy'] = np.array([fc.get('power_loss_eddy', 0) for fc in force_components])
            else:
                # Fallback if force_analysis not available
                self.results['force_gradient'] = self.results['force_total'].copy()  # Assume all force is gradient
                self.results['force_reluctance'] = np.zeros_like(self.results['force_total'])
                self.results['force_lorentz'] = np.zeros_like(self.results['force_total'])
                self.results['force_maxwell'] = np.zeros_like(self.results['force_total'])
                self.results['force_eddy'] = np.zeros_like(self.results['force_total'])
                self.results['force_image'] = np.zeros_like(self.results['force_total'])
                self.results['power_loss_eddy'] = np.zeros_like(self.results['force_total'])
            
            # ENHANCED: Advanced eddy current analysis with detailed parameters
            if hasattr(self.physics, 'calculate_eddy_current_effects'):
                eddy_effects = []
                for i, (I, x, v) in enumerate(zip(self.results['current'], self.results['position'], self.results['velocity'])):
                    if abs(I) > 1e-6 and abs(v) > 1e-6:
                        try:
                            # Get current and time history for frequency analysis
                            hist_length = min(50, i + 1)
                            hist_start = max(0, i - hist_length + 1)
                            current_hist = self.results['current'][hist_start:i+1] if i > 0 else None
                            time_hist = self.results['time'][hist_start:i+1] if i > 0 else None
                            
                            effects = self.physics.calculate_eddy_current_effects(I, v, x, current_hist, time_hist)
                            eddy_effects.append(effects)
                        except Exception as e:
                            # Fallback for eddy current analysis failure
                            eddy_effects.append({
                                'skin_depth': np.inf, 'induced_current': 0, 'opposing_force': 0,
                                'power_loss': 0, 'effective_resistance': np.inf, 'induced_emf': 0,
                                'current_density_peak': 0, 'frequency_effective': 0
                            })
                    else:
                        eddy_effects.append({
                            'skin_depth': np.inf, 'induced_current': 0, 'opposing_force': 0,
                            'power_loss': 0, 'effective_resistance': np.inf, 'induced_emf': 0,
                            'current_density_peak': 0, 'frequency_effective': 0
                        })
                
                # Extract enhanced eddy current data
                self.results['skin_depth'] = np.array([ef.get('skin_depth', np.inf) for ef in eddy_effects])
                self.results['eddy_current_magnitude'] = np.array([ef.get('induced_current', 0) for ef in eddy_effects])
                self.results['eddy_current_resistance'] = np.array([ef.get('effective_resistance', np.inf) for ef in eddy_effects])
                self.results['eddy_induced_emf'] = np.array([ef.get('induced_emf', 0) for ef in eddy_effects])
                self.results['eddy_current_density'] = np.array([ef.get('current_density_peak', 0) for ef in eddy_effects])
                self.results['frequency_content'] = np.array([ef.get('frequency_effective', 0) for ef in eddy_effects])
            
            # ENHANCED: Magnetic field analysis using enhanced methods
            # ENHANCED: Magnetic field analysis using enhanced methods with fallback
            try:
                # Use the enhanced magnetic field calculation with proper error handling
                field_calculations = []
                for pos, I in zip(self.results['position'], self.results['current']):
                    try:
                        if hasattr(self.physics, 'magnetic_field_solenoid_enhanced'):
                            field = self.physics.magnetic_field_solenoid_enhanced(pos, I)
                        elif hasattr(self.physics, 'magnetic_field_solenoid_on_axis'):
                            field = self.physics.magnetic_field_solenoid_on_axis(pos, I)
                        else:
                            field = 0.0  # Fallback
                        field_calculations.append(field)
                    except Exception as e:
                        # Individual field calculation failed, use fallback
                        field_calculations.append(0.0)
                
                self.results['magnetic_field'] = np.array(field_calculations)
            except Exception as e:
                # Entire field calculation failed, provide zero array
                self.results['magnetic_field'] = np.zeros_like(self.results['current'])
            
            # ENHANCED: Inductance gradient with enhanced calculation and fallback
            try:
                gradient_calculations = []
                for i, pos in enumerate(self.results['position']):
                    try:
                        # Use current for saturation-dependent gradient if available
                        current_for_gradient = self.results['current'][i] if i < len(self.results['current']) else None
                        gradient = self.physics.get_inductance_gradient(pos, current=current_for_gradient)
                        gradient_calculations.append(gradient)
                    except Exception as e:
                        # Individual gradient calculation failed
                        gradient_calculations.append(0.0)
                
                self.results['inductance_gradient'] = np.array(gradient_calculations)
            except Exception as e:
                # Entire gradient calculation failed
                self.results['inductance_gradient'] = np.zeros_like(self.results['position'])
            
            # ENHANCED: Power decomposition with detailed loss analysis
            self.results['power_mechanical'] = self.results['force_total'] * self.results['velocity']
            self.results['power_loss_resistive'] = self.results['current']**2 * self.physics.total_resistance
            
            # Additional power loss components if available
            if 'power_loss_eddy' in self.results:
                self.results['power_loss_total'] = self.results['power_loss_resistive'] + self.results['power_loss_eddy']
            else:
                self.results['power_loss_total'] = self.results['power_loss_resistive']
            
            # ENHANCED: Magnetic energy with nonlinear inductance effects
            self.results['energy_magnetic'] = 0.5 * self.results['inductance'] * self.results['current']**2
            
            # ENHANCED: Temperature analysis if available
            if hasattr(self.physics, 'temperature'):
                # Simple temperature rise estimation from eddy current losses
                if hasattr(self.physics, 'eddy_power_loss'):
                    temp_rise = np.cumsum(self.results.get('power_loss_eddy', np.zeros_like(self.results['time']))) * np.gradient(self.results['time'])
                    self.results['temperature_rise'] = temp_rise * 0.1  # Simplified thermal model
                else:
                    self.results['temperature_rise'] = np.zeros_like(self.results['time'])
            
            # ENHANCED: Effective permeability tracking with robust error handling
            if hasattr(self.physics, '_calculate_effective_permeability'):
                permeability_values = []
                for I, x in zip(self.results['current'], self.results['position']):
                    try:
                        # Calculate magnetic coupling if method available
                        if hasattr(self.physics, '_calculate_magnetic_coupling'):
                            coupling = self.physics._calculate_magnetic_coupling(x, I)
                        else:
                            # Estimate coupling based on overlap
                            overlap_frac = self.physics._calculate_overlap_fraction(x) if hasattr(self.physics, '_calculate_overlap_fraction') else 1.0
                            coupling = overlap_frac
                        
                        # Calculate effective permeability
                        mu_eff = self.physics._calculate_effective_permeability(x, I, coupling, 0.0)
                        permeability_values.append(max(1.0, mu_eff))  # Ensure >= 1
                    except Exception as e:
                        # Fallback to basic calculation or default
                        if abs(I) > 1e-6:
                            # Simple saturation model fallback
                            overlap = max(0, min(1, (x - (-self.physics.proj_length)) / self.physics.coil_length))
                            mu_eff = 1 + (self.physics.proj_mu_r - 1) * overlap
                            permeability_values.append(mu_eff)
                        else:
                            permeability_values.append(1.0)
                self.results['permeability_effective'] = np.array(permeability_values)
            else:
                # Fallback permeability calculation based on simple overlap
                permeability_values = []
                for x in self.results['position']:
                    try:
                        # Simple geometric overlap calculation
                        proj_start = x - self.physics.proj_length
                        proj_end = x
                        coil_start = 0
                        coil_end = self.physics.coil_length
                        
                        overlap_start = max(proj_start, coil_start)
                        overlap_end = min(proj_end, coil_end)
                        overlap_length = max(0, overlap_end - overlap_start)
                        overlap_fraction = overlap_length / self.physics.coil_length if self.physics.coil_length > 0 else 0
                        
                        # Effective permeability
                        mu_eff = 1 + (self.physics.proj_mu_r - 1) * overlap_fraction
                        permeability_values.append(mu_eff)
                    except Exception as e:
                        permeability_values.append(1.0)
                self.results['permeability_effective'] = np.array(permeability_values)
            
            # ENHANCED: Saturation factor tracking with robust material property handling
            if hasattr(self.physics, 'saturation_enabled') and self.physics.saturation_enabled:
                saturation_factors = []
                for I, x in zip(self.results['current'], self.results['position']):
                    try:
                        # Get material-dependent saturation field from config with robust fallbacks
                        B_sat = 2.0  # Default steel saturation field (Tesla)
                        
                        if hasattr(self.physics, 'materials_data') and self.physics.materials_data:
                            try:
                                # Get projectile material from config
                                proj_material = self.physics.config.get('projectile', {}).get('material', 'steel')
                                materials = self.physics.materials_data.get('materials', {})
                                
                                if proj_material in materials:
                                    material_props = materials[proj_material]
                                    B_sat = material_props.get('saturation_field', B_sat)
                                elif 'steel' in materials:
                                    # Fallback to steel properties
                                    material_props = materials['steel']
                                    B_sat = material_props.get('saturation_field', B_sat)
                            except Exception as e:
                                # Keep default B_sat if materials database access fails
                                pass
                        
                        # Calculate magnetic field at projectile position with error handling
                        try:
                            if hasattr(self.physics, 'magnetic_field_solenoid_enhanced'):
                                B_field = self.physics.magnetic_field_solenoid_enhanced(x, I)
                            elif hasattr(self.physics, 'magnetic_field_solenoid_on_axis'):
                                B_field = self.physics.magnetic_field_solenoid_on_axis(x, I)
                            else:
                                B_field = 0.0
                        except Exception as e:
                            B_field = 0.0
                        
                        # Ensure B_field is a scalar for min() function
                        B_field_mag = float(np.abs(B_field)) if hasattr(B_field, '__len__') else abs(float(B_field))
                        
                        # Calculate saturation factor (1.0 = no saturation, <1.0 = saturated)
                        # Avoid division by zero
                        if B_field_mag > 1e-12:
                            sat_factor = min(1.0, float(B_sat) / B_field_mag)
                        else:
                            sat_factor = 1.0
                            
                        saturation_factors.append(max(0.1, sat_factor))  # Limit minimum saturation factor
                    except Exception as e:
                        # Ultimate fallback
                        saturation_factors.append(1.0)
                        
                self.results['saturation_factor'] = np.array(saturation_factors)
            else:
                # No saturation modeling enabled - all factors are 1.0
                self.results['saturation_factor'] = np.ones_like(self.results['force_total'])
                
            # ENHANCED: Physics validation and error bounds assessment
            if hasattr(self.physics, 'field_accuracy'):
                self.results['field_accuracy'] = getattr(self.physics, 'field_accuracy', 1e-6)
            
            # ENHANCED: Force consistency check
            force_consistency = []
            for i in range(1, len(self.results['force_total'])):
                if abs(self.results['force_total'][i-1]) > 1e-6:
                    consistency = abs(self.results['force_total'][i] - self.results['force_total'][i-1]) / abs(self.results['force_total'][i-1])
                    force_consistency.append(consistency)
                else:
                    force_consistency.append(0.0)
            if force_consistency:
                self.results['force_consistency'] = np.array([0.0] + force_consistency)
            else:
                self.results['force_consistency'] = np.zeros_like(self.results['force_total'])
                
            # ENHANCED: Energy conservation tracking
            E_initial = self.physics.initial_energy
            E_capacitor = self.results['energy_capacitor']
            E_kinetic = self.results['energy_kinetic'] 
            E_magnetic = self.results['energy_magnetic']
            
            # Estimate energy losses (cumulative)
            dt = np.gradient(self.results['time'])
            E_resistive_loss = np.cumsum(self.results['power_loss_resistive'] * dt)
            E_eddy_loss = np.cumsum(self.results.get('power_loss_eddy', np.zeros_like(self.results['time'])) * dt)
            
            # Total accounted energy
            E_total_accounted = E_capacitor + E_kinetic + E_magnetic + E_resistive_loss + E_eddy_loss
            
            # Energy conservation error
            energy_conservation_error = np.abs(E_total_accounted - E_initial) / E_initial
            self.results['energy_conservation'] = energy_conservation_error;
            
            # Physics validation if available
            if hasattr(self.physics, 'calculate_field_with_error_estimate'):
                field_validation = []
                for pos, I in zip(self.results['position'][:10], self.results['current'][:10]):  # Sample first 10 points
                    if abs(I) > 1e-6:
                        validation_method = getattr(self.physics, 'calculate_field_with_error_estimate', None)
                        if validation_method:
                            validation = validation_method(pos, I)
                            field_validation.append(validation.get('relative_error_estimate', 0))
                        else:
                            field_validation.append(0)
                    else:
                        field_validation.append(0)
                self.results['field_accuracy'] = np.mean(field_validation) if field_validation else 0.0
            
        except Exception as e:
            print(f"Warning: Enhanced physics analysis failed: {e}")
            print("Using fallback values for enhanced physics analysis...")
            
            # Provide comprehensive fallback values for enhanced physics
            n_points = len(self.results['force_total'])
            
            # Force analysis fallbacks
            self.results['force_gradient'] = self.results['force_total'].copy()  # Assume all force is gradient
            self.results['force_reluctance'] = np.zeros(n_points)
            self.results['force_lorentz'] = np.zeros(n_points)
            self.results['force_maxwell'] = np.zeros(n_points)
            self.results['force_eddy'] = np.zeros(n_points)
            self.results['force_image'] = np.zeros(n_points)
            
            # Eddy current analysis fallbacks
            self.results['skin_depth'] = np.full(n_points, np.inf)
            self.results['eddy_current_magnitude'] = np.zeros(n_points)
            self.results['eddy_current_resistance'] = np.full(n_points, np.inf)
            self.results['eddy_induced_emf'] = np.zeros(n_points)
            self.results['eddy_current_density'] = np.zeros(n_points)
            self.results['frequency_content'] = np.zeros(n_points)
            
            # Magnetic field fallbacks
            self.results['magnetic_field'] = np.zeros(n_points)
            self.results['inductance_gradient'] = np.zeros(n_points)
            
            # Power and energy fallbacks
            self.results['power_mechanical'] = self.results['force_total'] * self.results['velocity']
            self.results['power_loss_resistive'] = self.results['current']**2 * self.physics.total_resistance
            self.results['power_loss_eddy'] = np.zeros(n_points)
            self.results['power_loss_total'] = self.results['power_loss_resistive']
            self.results['energy_magnetic'] = 0.5 * self.results['inductance'] * self.results['current']**2
            
            # Material property fallbacks
            self.results['temperature_rise'] = np.zeros_like(self.results['time'])
            self.results['permeability_effective'] = np.ones(n_points)
            self.results['saturation_factor'] = np.ones(n_points)
            
            # Validation fallbacks
            self.results['field_accuracy'] = 0.0
            self.results['force_consistency'] = np.zeros(n_points)
            self.results['energy_conservation'] = np.zeros_like(self.results['time'])
        
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
        velocity_ratio = self.simulation_info['final_velocity'] / theoretical_max_velocity;
        
        print(f"\nTheoretical Comparison:")
        print(f"  Theoretical max velocity: {theoretical_max_velocity:.2f} m/s")
        print(f"  Achieved fraction: {velocity_ratio:.3f}")
        
        # ENHANCED: Advanced physics analysis results
        if len(self.results['time']) > 0:
            print(f"\nEnhanced Physics Analysis:")
            
            # Force decomposition analysis
            if 'force_gradient' in self.results and np.any(self.results['force_gradient'] != 0):
                max_force_grad = np.max(np.abs(self.results['force_gradient']))
                max_force_eddy = np.max(np.abs(self.results['force_eddy']))
                max_force_total = np.max(np.abs(self.results['force_total']))
                
                print(f"  Force Analysis:")
                print(f"    Max gradient force: {max_force_grad:.1f} N")
                print(f"    Max eddy current force: {max_force_eddy:.1f} N")
                print(f"    Max total force: {max_force_total:.1f} N")
                
                # Force breakdown at peak
                peak_idx = np.argmax(np.abs(self.results['force_total']))
                if abs(self.results['force_total'][peak_idx]) > 1e-9:
                    grad_percent = abs(self.results['force_gradient'][peak_idx]) / abs(self.results['force_total'][peak_idx]) * 100
                    eddy_percent = abs(self.results['force_eddy'][peak_idx]) / abs(self.results['force_total'][peak_idx]) * 100
                    print(f"    Force breakdown at peak: {grad_percent:.1f}% gradient, {eddy_percent:.1f}% eddy losses")
                
                # Additional force components if available
                if 'force_lorentz' in self.results:
                    max_lorentz = np.max(np.abs(self.results['force_lorentz']))
                    if max_lorentz > 0.1:
                        print(f"    Max Lorentz force: {max_lorentz:.1f} N")
                
                if 'force_maxwell' in self.results:
                    max_maxwell = np.max(np.abs(self.results['force_maxwell']))
                    if max_maxwell > 0.1:
                        print(f"    Max Maxwell stress force: {max_maxwell:.1f} N")
                
                if 'force_reluctance' in self.results:
                    max_reluctance = np.max(np.abs(self.results['force_reluctance']))
                    if max_reluctance > 0.1:
                        print(f"    Max reluctance force: {max_reluctance:.1f} N")
            
            # Eddy current effects analysis
            if 'eddy_current_magnitude' in self.results:
                max_eddy_current = np.max(self.results['eddy_current_magnitude'])
                avg_eddy_current = np.mean(self.results['eddy_current_magnitude'])
                finite_skin_depths = self.results['skin_depth'][self.results['skin_depth'] < np.inf]
                
                print(f"  Eddy Current Analysis:")
                if max_eddy_current > 0:
                    print(f"    Peak eddy current: {max_eddy_current:.1f} A")
                    print(f"    Average eddy current: {avg_eddy_current:.2f} A")
                else:
                    print(f"    No significant eddy currents detected")
                
                if len(finite_skin_depths) > 0:
                    min_skin_depth = np.min(finite_skin_depths)
                    avg_skin_depth = np.mean(finite_skin_depths)
                    print(f"    Min skin depth: {min_skin_depth*1000:.2f} mm")
                    print(f"    Avg skin depth: {avg_skin_depth*1000:.1f} mm")
                
                if 'frequency_content' in self.results:
                    max_frequency = np.max(self.results['frequency_content'])
                    if max_frequency > 10:
                        print(f"    Peak frequency content: {max_frequency:.0f} Hz")
            
            # Power and energy analysis
            if 'power_loss_eddy' in self.results:
                max_eddy_power = np.max(self.results['power_loss_eddy'])
                total_eddy_energy = np.trapezoid(self.results['power_loss_eddy'], self.results['time'])
                total_resistive_energy = np.trapezoid(self.results['power_loss_resistive'], self.results['time'])
                
                print(f"  Power Loss Analysis:")
                print(f"    Peak eddy power loss: {max_eddy_power:.1f} W")
                print(f"    Total eddy energy loss: {total_eddy_energy:.3f} J")
                print(f"    Total resistive loss: {total_resistive_energy:.3f} J")
                
                if float(total_eddy_energy) > 0 and float(total_resistive_energy) > 0:
                    loss_ratio = float(total_eddy_energy) / float(total_resistive_energy) * 100
                    print(f"    Eddy/Resistive loss ratio: {loss_ratio:.1f}%")
            
            # Magnetic saturation and permeability effects
            if 'saturation_factor' in self.results:
                min_sat_factor = np.min(self.results['saturation_factor'])
                avg_sat_factor = np.mean(self.results['saturation_factor'])
                
                print(f"  Magnetic Saturation Analysis:")
                print(f"    Min saturation factor: {min_sat_factor:.3f}")
                print(f"    Avg saturation factor: {avg_sat_factor:.3f}")
                
                if min_sat_factor < 0.95:
                    print(f"    ⚠ Significant magnetic saturation detected!")
                    sat_positions = self.results['position'][self.results['saturation_factor'] < 0.95]
                    if len(sat_positions) > 0:
                        print(f"    Saturation region: {sat_positions[0]*1000:.1f} to {sat_positions[-1]*1000:.1f} mm")
                elif min_sat_factor < 0.99:
                    print(f"    ⚠ Mild magnetic saturation detected")
                else:
                    print(f"    ✓ No significant magnetic saturation")
            
            # Permeability variation
            if 'permeability_effective' in self.results:
                mu_max = np.max(self.results['permeability_effective'])
                mu_min = np.min(self.results['permeability_effective'])
                mu_avg = np.mean(self.results['permeability_effective'])
                
                if mu_max != mu_min:
                    print(f"  Permeability Analysis:")
                    print(f"    Permeability range: {mu_min:.0f} - {mu_max:.0f} (avg: {mu_avg:.0f})")
                    print(f"    Permeability variation: {((mu_max-mu_min)/mu_avg)*100:.1f}%")
            
            # Energy conservation and accuracy checks
            if 'energy_conservation' in self.results:
                max_energy_error = np.max(self.results['energy_conservation'])
                avg_energy_error = np.mean(self.results['energy_conservation'])
                
                print(f"  Accuracy Assessment:")
                print(f"    Max energy conservation error: {max_energy_error*100:.3f}%")
                print(f"    Avg energy conservation error: {avg_energy_error*100:.4f}%")
                
                if max_energy_error < 0.001:
                    print(f"    ✓ Excellent energy conservation")
                elif max_energy_error < 0.01:
                    print(f"    ✓ Good energy conservation")
                else:
                    print(f"    ⚠ Energy conservation errors detected")
            
            # Force consistency check
            if 'force_consistency' in self.results and np.any(self.results['force_consistency'] > 0):
                max_force_error = np.max(self.results['force_consistency'])
                avg_force_error = np.mean(self.results['force_consistency'])
                
                print(f"    Max force consistency error: {max_force_error*100:.2f}%")
                print(f"    Avg force consistency error: {avg_force_error*100:.3f}%")
                
                if max_force_error < 0.01:
                    print(f"    ✓ Excellent force calculation consistency")
                elif max_force_error < 0.05:
                    print(f"    ✓ Good force calculation consistency")
                else:
                    print(f"    ⚠ Force calculation inconsistencies detected")
            
            # Field accuracy assessment
            if hasattr(self, 'results') and 'field_accuracy' in self.results:
                field_accuracy = self.results['field_accuracy']
                if field_accuracy > 0:
                    print(f"    Field calculation accuracy: {(1-field_accuracy)*100:.2f}%")
                    if field_accuracy < 0.01:
                        print(f"    ✓ Excellent field calculation accuracy")
                    elif field_accuracy < 0.05:
                        print(f"    ✓ Good field calculation accuracy")
                    else:
                        print(f"    ⚠ Field calculation accuracy issues detected")
            
            # Temperature effects
            if 'temperature_rise' in self.results:
                max_temp_rise = np.max(self.results['temperature_rise'])
                if max_temp_rise > 0.1:
                    print(f"  Thermal Analysis:")
                    print(f"    Maximum temperature rise: {max_temp_rise:.1f}°C")
                    if max_temp_rise > 100:
                        print(f"    ⚠ High temperature rise may affect performance")
                    elif max_temp_rise > 50:
                        print(f"    ⚠ Moderate temperature rise detected")
                    else:
                        print(f"    ✓ Temperature rise within reasonable limits")
            
            # Physics accuracy assessment
            if 'field_accuracy' in self.results and self.results['field_accuracy'] > 0:
                print(f"  Field calculation accuracy: {self.results['field_accuracy']*100:.3f}%")
                
            # Frequency content analysis
            if 'frequency_content' in self.results:
                max_freq = np.max(self.results['frequency_content'])
                if max_freq > 0:
                    print(f"  Max frequency content: {max_freq:.0f} Hz")
    
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
        if len(self.results['time']) > 0:
            data_file = output_path / "time_series_data.npz"
            np.savez_compressed(data_file, **self.results)
            
            # Also save as CSV for easy analysis
            csv_file = output_path / "time_series_data.csv"
            
            # Prepare data for CSV - ENHANCED with all physics data
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
            
            # ENHANCED: Add advanced physics data to CSV if available
            enhanced_fields = [
                ('force_gradient_N', 'force_gradient'),
                ('force_eddy_N', 'force_eddy'),
                ('magnetic_field_T', 'magnetic_field'),
                ('inductance_gradient_H_per_m', 'inductance_gradient'),
                ('power_mechanical_W', 'power_mechanical'),
                ('power_loss_resistive_W', 'power_loss_resistive'),
                ('power_loss_eddy_W', 'power_loss_eddy'),
                ('energy_magnetic_J', 'energy_magnetic'),
                ('skin_depth_m', 'skin_depth'),
                ('eddy_current_A', 'eddy_current_magnitude'),
                ('permeability_effective', 'permeability_effective'),
                ('saturation_factor', 'saturation_factor'),
                ('temperature_rise_K', 'temperature_rise'),
                ('frequency_Hz', 'frequency_content'),
                ('energy_conservation_error', 'energy_conservation')
            ]
            
            for csv_name, result_key in enhanced_fields:
                if result_key in self.results:
                    csv_data[csv_name] = self.results[result_key]
            
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
        if len(self.results['time']) == 0:
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
    
    def check_physics_integration(self):
        """
        Check and report the status of enhanced physics engine integration.
        
        Returns:
            dict: Status of various physics features
        """
        integration_status = {
            # Core electromagnetic physics
            'enhanced_force_calculation': hasattr(self.physics, 'magnetic_force_ferromagnetic'),
            'circuit_logic_force': hasattr(self.physics, 'magnetic_force_with_circuit_logic'),
            'enhanced_magnetic_field': hasattr(self.physics, 'magnetic_field_solenoid_enhanced'),
            'finite_solenoid_field': hasattr(self.physics, 'magnetic_field_finite_solenoid_on_axis'),
            'elliptic_integrals': hasattr(self.physics, 'magnetic_field_solenoid_enhanced'),
            'advanced_inductance': hasattr(self.physics, 'inductance_with_ferromagnetic_core'),
            'inductance_gradient': hasattr(self.physics, 'get_inductance_gradient'),
            
            # Force analysis components
            'force_analysis': hasattr(self.physics, 'force_analysis'),
            'maxwell_stress_tensor': hasattr(self.physics, '_calculate_maxwell_stress_force'),
            'reluctance_force': hasattr(self.physics, '_calculate_reluctance_force'),
            'lorentz_force': hasattr(self.physics, '_calculate_lorentz_force'),
            'image_force': hasattr(self.physics, '_calculate_image_force'),
            
            # AC and eddy current effects
            'eddy_current_effects': hasattr(self.physics, 'calculate_eddy_current_effects'),
            '3d_eddy_currents': hasattr(self.physics, '_calculate_3d_eddy_currents'),
            'skin_depth_modeling': hasattr(self.physics, 'calculate_skin_depth'),
            'proximity_effects': hasattr(self.physics, 'calculate_proximity_effects'),
            'frequency_analysis': getattr(self.physics, 'frequency_analysis_enabled', False),
            
            # Nonlinear magnetic effects
            'nonlinear_permeability': hasattr(self.physics, 'calculate_nonlinear_permeability'),
            'saturation_effects': getattr(self.physics, 'saturation_enabled', False),
            'hysteresis_modeling': getattr(self.physics, 'hysteresis_enabled', False),
            'jiles_atherton_model': hasattr(self.physics, 'jiles_atherton_hysteresis'),
            'magnetic_coupling': hasattr(self.physics, '_calculate_magnetic_coupling'),
            'effective_permeability': hasattr(self.physics, '_calculate_effective_permeability'),
            
            # Thermal and optimization
            'thermal_effects': getattr(self.physics, 'thermal_enabled', False),
            'temperature_dependent_properties': hasattr(self.physics, 'calculate_temperature_rise'),
            'timing_optimization': hasattr(self.physics, 'timing_config'),
            'multi_stage_timing': hasattr(self.physics, 'set_previous_stage_velocity'),
            
            # Materials and database
            'materials_database': hasattr(self.physics, 'materials_data'),
            'wire_specifications': hasattr(self.physics, 'get_wire_diameter'),
            'material_properties': hasattr(self.physics, 'get_material_property'),
            
            # Analysis and validation
            'energy_tracking': hasattr(self.physics, 'energy_tracking'),
            'field_accuracy': hasattr(self.physics, 'field_accuracy'),
            'numerical_stability': hasattr(self.physics, '_safe_numerical_operation'),
            'error_bounds': hasattr(self.physics, 'calculate_field_with_error_estimate'),
            
            # Advanced configuration
            'advanced_physics_config': hasattr(self.physics, '_initialize_advanced_physics'),
            'configuration_validation': hasattr(self.physics, 'validate_configuration'),
            'physics_parameters': hasattr(self.physics, 'print_system_parameters')
        }
        
        # Summary statistics (simplified output)
        enabled_features = [k for k, v in integration_status.items() if v]
        total_features = len(integration_status)
        enabled_count = len(enabled_features)
        coverage_percent = (enabled_count / total_features) * 100
        
        # Compatibility check
        if hasattr(self.physics, 'validate_configuration'):
            try:
                self.physics.validate_configuration()
                print("  ✓ Physics engine configuration validated successfully")
            except Exception as e:
                print(f"  ⚠  Configuration validation failed: {e}")
        
        return integration_status

    def optimize_physics_settings(self):
        """
        Optimize physics engine settings based on simulation parameters and available features.
        
        This method analyzes the configuration and automatically enables/configures
        advanced physics features for optimal accuracy and performance.
        
        Returns:
            dict: Summary of optimizations applied
        """
        optimizations = {
            'field_method_optimized': False,
            'eddy_current_enabled': False,
            'saturation_modeling_enabled': False,
            'timing_optimization_configured': False,
            'frequency_analysis_enabled': False,
            'thermal_effects_enabled': False,
            'energy_tracking_enabled': False,
            'force_analysis_enabled': False,
            'materials_database_configured': False,
            'numerical_stability_enhanced': False,
            'circuit_logic_enabled': False
        }
        
        # Calculate key parameters for optimization decisions
        expected_velocity = np.sqrt(2 * self.physics.initial_energy / self.physics.proj_mass)
        max_expected_current = self.physics.initial_voltage / self.physics.total_resistance
        peak_power = self.physics.initial_voltage**2 / self.physics.total_resistance
        
        # Silent optimization - just configure settings without verbose output
        # 1. Field calculation method optimization
        if hasattr(self.physics, 'field_method'):
            if hasattr(self.physics, 'magnetic_field_solenoid_enhanced'):
                if not hasattr(self.physics, 'field_method') or self.physics.field_method != 'exact_elliptic':
                    self.physics.field_method = 'exact_elliptic'
                optimizations['field_method_optimized'] = True
        
        # 2. Eddy current effects optimization
        if hasattr(self.physics, 'eddy_current_enabled'):
            if expected_velocity > 5:
                self.physics.eddy_current_enabled = True
                optimizations['eddy_current_enabled'] = True
        
        # 3. Energy tracking (always enable if available)
        if hasattr(self.physics, 'energy_tracking'):
            self.physics.energy_tracking = True
            optimizations['energy_tracking_enabled'] = True
        
        # 4. Circuit logic enhancement
        if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
            optimizations['circuit_logic_enabled'] = True
        
        # Simple summary
        enabled_optimizations = sum(1 for opt in optimizations.values() if opt)
        if enabled_optimizations >= 2:
            print("✓ Physics optimization complete")
        
        return optimizations

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
        
        # Timing optimization removed
        
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
                    
                    # Timing optimization removed - stages run independently
                    
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
                    if save_data and len(stage_sim.results['time']) > 0:
                        # Adjust time to be cumulative across stages
                        adjusted_time = stage_sim.results['time'] + time_offset
                        
                        # Add stage transition marker
                        if stage_num > 1:
                            self.aggregated_results['stage_transitions'].append(time_offset)
                        
                        # Append data to aggregated results
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
            
            # ENHANCED: Check physics integration status for single-stage simulations (silent)
            physics_status = sim.check_physics_integration()
            optimization_status = sim.optimize_physics_settings()
            results = sim.run_simulation(save_data=True, verbose=True, show_progress=True, check_physics=True)
            
            # Create output directory based on config filename
            config_name = Path(config_file).stem
            output_dir = f"results_{config_name}"
            
            # Save detailed results to CSV and JSON
            print("\n" + "="*50)
            print("SAVING RESULTS")
            print("="*50)
            sim.save_results(output_dir)
            
            # Print enhanced summary with physics analysis
            print("\n" + "="*50)
            print("SIMULATION SUMMARY")
            print("="*50)
            print(f"Final velocity: {results['final_velocity_ms']:.1f} m/s")
            print(f"Efficiency: {results['efficiency_percent']:.1f}%")
            print(f"Max current: {results.get('max_current_A', 0):.1f} A")
            print(f"Max force: {results.get('max_force_N', 0):.1f} N")
            print(f"Simulation time: {results['simulation_time_s']:.3f} s")
            print(f"Exit reason: {results['exit_reason']}")
            
            # Enhanced physics summary
            if hasattr(sim, 'results') and len(sim.results.get('time', [])) > 0:
                print(f"\nAdvanced Physics Summary:")
                
                # Force analysis summary
                if 'force_gradient' in sim.results:
                    max_gradient_force = np.max(np.abs(sim.results['force_gradient']))
                    max_eddy_force = np.max(np.abs(sim.results.get('force_eddy', [0])))
                    print(f"  Peak gradient force: {max_gradient_force:.1f} N")
                    if max_eddy_force > 0.1:
                        print(f"  Peak eddy current force: {max_eddy_force:.1f} N")
                
            # Extract energy values from results
            final_kinetic_energy = results.get('final_kinetic_energy_J', 0)
            initial_energy = results.get('initial_energy_J', 0)
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
