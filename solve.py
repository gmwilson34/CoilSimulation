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
"""

import numpy as np
import json
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from equations import CoilgunPhysicsEngine

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
        
        # Initialize results storage
        self.results = {
            'time': None,
            'charge': None,
            'current': None,
            'position': None,
            'velocity': None,
            'force': None,
            'inductance': None,
            'power': None,
            'energy_capacitor': None,
            'energy_kinetic': None
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
        
    def run_simulation(self, save_data=True, verbose=True):
        """
        Execute the complete coilgun simulation.
        
        Args:
            save_data: Whether to save detailed time-series data
            verbose: Whether to print progress and results
            
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
            
            solution = solve_ivp(
                fun=self.physics.circuit_derivatives,
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
                self.simulation_info['max_force'] = np.max(np.abs(self.results['force']))
            else:
                # Calculate max values from the solution data even if not saving detailed results
                if hasattr(solution, 'y') and solution.y.shape[1] > 0:
                    currents = solution.y[1, :]  # Current is the second state variable
                    self.simulation_info['max_current'] = np.max(np.abs(currents))
                    
                    # Calculate forces for max force
                    forces = []
                    for i, current in enumerate(currents):
                        position = solution.y[2, i]  # Position is third state variable
                        force = self.physics.magnetic_force_ferromagnetic(current, position)
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
        
        self.results['force'] = np.array([
            self.physics.magnetic_force_ferromagnetic(I, x) 
            for I, x in zip(self.results['current'], self.results['position'])
        ])
        
        # Power and energy analysis
        self.results['power'] = self.results['current'] * self.results['charge'] / self.physics.capacitance
        
        self.results['energy_capacitor'] = 0.5 * self.results['charge']**2 / self.physics.capacitance
        
        self.results['energy_kinetic'] = (0.5 * self.physics.proj_mass * 
                                         self.results['velocity']**2)
    
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
            if self.results['power'] is not None:
                max_power = np.max(self.results['power'])
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
                'force_N': self.results['force'],
                'inductance_H': self.results['inductance'],
                'power_W': self.results['power'],
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
        axes[1, 1].plot(t, self.results['force'], 'm-', linewidth=2)
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
                if any(key in data for key in ['coil', 'capacitor', 'projectile', 'simulation']):
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
    
    config_file = select_config_file()
    
    print("=" * 60)
    print("COILGUN SIMULATION SOLVER")
    print("=" * 60)
    print(f"Configuration file: {config_file}")
    
    try:
        # Initialize and run simulation
        sim = CoilgunSimulation(config_file)
        results = sim.run_simulation(save_data=True, verbose=True)
        
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
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import os
    main()
