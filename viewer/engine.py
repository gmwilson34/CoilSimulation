"""
Main visualization engine for coilgun systems.

This module provides the primary interface for the view package, combining
all visualization components into a unified, easy-to-use visualization engine.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from .core import BaseVisualizer, setup_signal_handling
from .fields import MagneticFieldCalculator, FieldLineTracer
from .plots import ContourPlotter, ForcePlotter, ProfilePlotter
from .plots3d import Plot3DVisualizer, GeometryRenderer
from .animations import AnimationEngine
from .analysis import PhysicsAnalyzer, ElectromagneticAnalyzer
from .multistage import MultistageVisualizer
from .utils import (extract_actual_current_data, find_results_directories,
                   select_results_directory, load_simulation_from_results,
                   create_output_directory, validate_simulation_data,
                   load_actual_simulation_data, get_actual_current_profile,
                   get_actual_force_profile, get_simulation_parameters,
                   SimulationPhysicsEngine)


class CoilgunVisualizationEngine(BaseVisualizer):
    """
    Main visualization engine for coilgun simulation visualization.
    
    This class provides a comprehensive interface for all visualization
    capabilities, integrating field calculations, plotting, animations,
    and analysis into a single convenient interface.
    """
    
    def __init__(self, physics_engine=None, auto_load_data=True):
        """
        Initialize the visualization engine.
        
        Args:
            physics_engine: CoilgunPhysicsEngine instance
            auto_load_data: Whether to automatically load simulation data
        """
        super().__init__(physics_engine)
        
        # Load actual simulation data if available
        self.simulation_data = None
        self.simulation_params = {}
        
        if auto_load_data:
            self.simulation_data = load_actual_simulation_data()
            if self.simulation_data:
                self.simulation_params = get_simulation_parameters(self.simulation_data)
                print(f"Loaded simulation parameters: {self.simulation_params}")
                
                # Create physics engine from simulation configuration
                if physics_engine is None and 'config' in self.simulation_data:
                    physics_engine = SimulationPhysicsEngine(self.simulation_data['config'])
                    self.physics = physics_engine
        
        # Initialize component modules with physics engine
        self.field_calculator = MagneticFieldCalculator(physics_engine)
        self.contour_plotter = ContourPlotter(physics_engine)
        self.force_plotter = ForcePlotter(physics_engine)
        self.profile_plotter = ProfilePlotter(physics_engine)
        self.plot3d = Plot3DVisualizer(physics_engine)
        self.animator = AnimationEngine(physics_engine)
        self.physics_analyzer = PhysicsAnalyzer(physics_engine)
        self.em_analyzer = ElectromagneticAnalyzer(physics_engine)
        self.multistage_viz = MultistageVisualizer(physics_engine)
        
        # Set up signal handling for graceful exit
        setup_signal_handling()
        
        if self.simulation_data:
            print("Coilgun Visualization Engine initialized with actual simulation data")
        else:
            print("Coilgun Visualization Engine initialized (no simulation data found)")
    
    def create_comprehensive_visualization_suite(self, config_file=None, 
                                               simulation_results=None,
                                               time_series_data=None,
                                               output_dir="comprehensive_visualizations"):
        """
        Create a comprehensive suite of all available visualizations using actual simulation data.
        
        Args:
            config_file: Configuration file path (optional, will use loaded config)
            simulation_results: Simulation results dictionary (optional, will use loaded data)
            time_series_data: Time series data dictionary (optional, will use loaded data)
            output_dir: Output directory name
        """
        output_path = create_output_directory(output_dir)
        
        print(f"Creating comprehensive visualization suite using actual simulation data in: {output_path}")
        
        # Use loaded data if not provided
        if simulation_results is None and self.simulation_data:
            simulation_results = self.simulation_data
        
        if time_series_data is None and self.simulation_data:
            time_series_data = self.simulation_data.get('time_series', {})
        
        # Validate input data
        if simulation_results:
            validation = validate_simulation_data(simulation_results)
            if not validation['valid']:
                print("Warning: Simulation data validation failed")
                for error in validation['errors']:
                    print(f"  Error: {error}")
        
        # 1. Actual simulation data analysis
        if simulation_results:
            self._create_actual_simulation_analysis(simulation_results, output_path)
        
        # 2. Magnetic field visualizations using actual currents
        self._create_field_visualizations_from_actual_data(output_path)
        
        # 3. Circuit and force analysis using real data
        if simulation_results:
            self._create_circuit_force_analysis(simulation_results, output_path)
        
        # 4. Force component analysis
        if simulation_results and 'force_analysis' in simulation_results:
            self._create_force_component_analysis(simulation_results, output_path)
        
        # 5. 3D visualizations with actual parameters
        self._create_3d_visualizations(output_path)
        
        # 6. Enhanced physics analysis
        if simulation_results:
            self._create_enhanced_physics_analysis(simulation_results, output_path)
        
        # 7. Animations using real data
        if simulation_results:
            self._create_animations(simulation_results, output_path)
        
        print(f"Comprehensive visualization suite completed: {output_path}")
        return output_path
    
    def create_field_visualization_suite(self, output_dir="field_visualizations"):
        """
        Create a suite of magnetic field visualizations using actual current values.
        
        Args:
            output_dir: Output directory name
        """
        output_path = create_output_directory(output_dir)
        
        print("Creating field visualization suite using actual simulation data...")
        
        # Get actual current values from simulation
        if self.simulation_data:
            time_array, current_array = get_actual_current_profile(self.simulation_data)
            if time_array is not None and current_array is not None:
                # Use actual current values at key time points
                max_current = np.max(current_array)
                current_values = [
                    max_current * 0.2,  # 20% of max
                    max_current * 0.4,  # 40% of max
                    max_current * 0.6,  # 60% of max
                    max_current * 0.8,  # 80% of max
                    max_current         # 100% of max
                ]
                print(f"Using actual current range: 0 - {max_current:.1f} A")
            else:
                current_values = [50, 100, 150, 200, 250, 300]
                print("Warning: Using default current values (no actual data available)")
        else:
            current_values = [50, 100, 150, 200, 250, 300]
            print("Warning: Using default current values (no simulation data loaded)")
        
        for current in current_values:
            print(f"Generating field visualizations for {current:.1f}A...")
            
            # 2D field contours
            field_data = self.field_calculator.calculate_bfield_map_2d(current)
            self.contour_plotter.plot_bfield_contours(
                field_data, 
                save_path=f"{output_path}/field_contours_{current:.1f}A.png"
            )
            
            # 3D field visualization
            self.plot3d.plot_3d_field_visualization(
                current,
                save_path=f"{output_path}/field_3d_{current:.1f}A.png",
                interactive=False
            )
        
        # On-axis field profiles using actual current range
        field_profile_data = self.field_calculator.calculate_onaxis_field_profile(current_values)
        self.contour_plotter.plot_onaxis_field_profile(
            field_profile_data,
            save_path=f"{output_path}/onaxis_field_profiles.png"
        )
        
        print(f"Field visualization suite completed: {output_path}")
        return output_path
    
    def create_quick_analysis(self, simulation_results=None, save_prefix="quick_analysis"):
        """
        Create a quick analysis visualization set using actual simulation data.
        
        Args:
            simulation_results: Simulation results dictionary (optional, will use loaded data)
            save_prefix: Prefix for saved files
        """
        # Use loaded data if not provided
        if simulation_results is None and self.simulation_data:
            simulation_results = self.simulation_data
        
        if not simulation_results:
            print("No simulation results available for quick analysis")
            return
        
        print("Creating quick analysis visualizations using actual simulation data...")
        
        # Force and current analysis using real data
        self.force_plotter.plot_force_current_map(
            simulation_results,
            save_path=f"{save_prefix}_force_current.png"
        )
        
        # Enhanced physics analysis using real data
        self.physics_analyzer.plot_enhanced_physics_analysis(
            simulation_results,
            save_path=f"{save_prefix}_physics_analysis.png"
        )
        
        print("Quick analysis completed")
    
    def create_interactive_3d_exploration(self, current=200):
        """
        Create interactive 3D field exploration.
        
        Args:
            current: Coil current for visualization
        """
        print(f"Creating interactive 3D exploration for {current}A...")
        print("Note: 3D field calculations may take 10-30 seconds...")
        print("Press Ctrl+C during calculation to cancel if needed.")
        
        try:
            self.plot3d.plot_3d_field_visualization(
                current,
                interactive=True,
                show_field_lines=True,
                show_coil=True
            )
        except KeyboardInterrupt:
            print("\n3D visualization cancelled by user.")
        except Exception as e:
            print(f"Error in 3D visualization: {e}")
            print("You may want to try reducing the current value or skipping 3D visualizations.")
    
    def animate_simulation(self, simulation_results, animation_type="motion", 
                          save_path=None):
        """
        Create animations of simulation results.
        
        Args:
            simulation_results: Simulation results dictionary
            animation_type: Type of animation ("motion", "field", "all")
            save_path: Path to save animation
        """
        print(f"Creating {animation_type} animation...")
        
        if animation_type == "motion" or animation_type == "all":
            self.animator.animate_projectile_motion(
                simulation_results, 
                save_path=save_path
            )
        
        if animation_type == "field" or animation_type == "all":
            self.animator.animate_field_evolution(
                simulation_results,
                save_path=save_path.replace('.gif', '_field.gif') if save_path else None
            )
    
    def analyze_multistage_performance(self, config_file, time_series_data, 
                                     summary_data, output_dir="multistage_analysis"):
        """
        Analyze multi-stage coilgun performance.
        
        Args:
            config_file: Configuration file path
            time_series_data: Time series data dictionary
            summary_data: Summary data dictionary
            output_dir: Output directory name
        """
        output_path = create_output_directory(output_dir)
        
        self.multistage_viz.create_multistage_visualizations(
            config_file, time_series_data, summary_data, output_path
        )
        
        return output_path
    
    def interactive_menu(self):
        """
        Launch an interactive menu for visualization selection.
        """
        print("\n=== Coilgun Visualization Engine ===")
        print("Interactive Visualization Menu")
        
        while True:
            print("\nAvailable options:")
            print("1. Load simulation results")
            print("2. Create field visualization suite")
            print("3. Interactive 3D field exploration (may be slow)")
            print("4. Create comprehensive visualization suite")
            print("5. Quick analysis from simulation results")
            print("6. Create animations")
            print("7. Multi-stage analysis")
            print("8. Fast 3D geometry view (no field calculation)")
            print("9. Exit")
            
            try:
                choice = input("\nSelect option (1-9): ").strip()
                
                if choice == "1":
                    self._menu_load_results()
                elif choice == "2":
                    print("Creating field visualization suite...")
                    result_path = self.create_field_visualization_suite()
                    print(f"Field visualizations saved to: {result_path}")
                elif choice == "3":
                    print("WARNING: 3D field calculations may take 10-30 seconds and use significant CPU.")
                    print("Consider using option 8 for faster 3D geometry visualization.")
                    confirm = input("Continue with full 3D field calculation? (y/N): ").strip().lower()
                    if confirm == 'y':
                        current = float(input("Enter current (A) [200]: ") or "200")
                        self.create_interactive_3d_exploration(current)
                    else:
                        print("3D field calculation cancelled.")
                elif choice == "4":
                    self._menu_comprehensive_suite()
                elif choice == "5":
                    self._menu_quick_analysis()
                elif choice == "6":
                    self._menu_animations()
                elif choice == "7":
                    self._menu_multistage_analysis()
                elif choice == "8":
                    current = float(input("Enter current (A) [200]: ") or "200")
                    print(f"Creating fast 3D geometry view for {current}A...")
                    self._create_fast_3d_view(current)
                elif choice == "9":
                    print("Exiting visualization engine...")
                    break
                else:
                    print("Invalid choice. Please select 1-9.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _create_actual_simulation_analysis(self, simulation_results, output_path):
        """Create analysis plots using actual simulation data."""
        print("Creating actual simulation data analysis...")
        
        if 'time_series' not in simulation_results:
            print("No time series data available for analysis")
            return
        
        time_data = np.array(simulation_results['time_series']['time'])
        current_data = np.array(simulation_results['time_series']['current'])
        force_data = np.array(simulation_results['time_series']['force_total'])
        position_data = np.array(simulation_results['time_series']['position'])
        velocity_data = np.array(simulation_results['time_series']['velocity'])
        
        # Create comprehensive simulation analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Actual Simulation Data Analysis', fontsize=16, fontweight='bold')
        
        # Convert time to milliseconds for better readability
        time_ms = time_data * 1000
        
        # 1. Current vs Time
        axes[0, 0].plot(time_ms, current_data, 'b-', linewidth=2)
        axes[0, 0].set_title('Current vs Time')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Current (A)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Force vs Time
        axes[0, 1].plot(time_ms, force_data, 'r-', linewidth=2)
        axes[0, 1].set_title('Force vs Time')
        axes[0, 1].set_xlabel('Time (ms)')
        axes[0, 1].set_ylabel('Force (N)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Position vs Time
        axes[0, 2].plot(time_ms, position_data * 1000, 'g-', linewidth=2)  # Convert to mm
        axes[0, 2].set_title('Position vs Time')
        axes[0, 2].set_xlabel('Time (ms)')
        axes[0, 2].set_ylabel('Position (mm)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Velocity vs Time
        axes[1, 0].plot(time_ms, velocity_data, 'm-', linewidth=2)
        axes[1, 0].set_title('Velocity vs Time')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Force vs Position
        axes[1, 1].plot(position_data * 1000, force_data, 'c-', linewidth=2)
        axes[1, 1].set_title('Force vs Position')
        axes[1, 1].set_xlabel('Position (mm)')
        axes[1, 1].set_ylabel('Force (N)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Current vs Force
        axes[1, 2].plot(current_data, force_data, 'orange', linewidth=2)
        axes[1, 2].set_title('Force vs Current')
        axes[1, 2].set_xlabel('Current (A)')
        axes[1, 2].set_ylabel('Force (N)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/actual_simulation_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_force_component_analysis(self, simulation_results, output_path):
        """Create force component analysis using actual data."""
        if 'force_analysis' not in simulation_results:
            print("No force component analysis data available")
            return
        
        print("Creating force component analysis...")
        
        force_data = simulation_results['force_analysis']
        time_data = np.array(simulation_results.get('time', []))
        
        if len(time_data) != len(force_data):
            # Create time array if lengths don't match
            time_data = np.linspace(0, self.simulation_params.get('simulation_time', 0.015), len(force_data))
        
        # Extract force components
        gradient_forces = [d.get('gradient_force', 0) for d in force_data]
        reluctance_forces = [d.get('reluctance_force', 0) for d in force_data]
        lorentz_forces = [d.get('lorentz_force', 0) for d in force_data]
        eddy_forces = [d.get('eddy_current_force', 0) for d in force_data]
        total_forces = [d.get('total_force', 0) for d in force_data]
        
        # Create force component plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('Force Component Analysis from Actual Simulation', fontsize=16, fontweight='bold')
        
        time_ms = time_data * 1000
        
        # Force components vs time
        ax1.plot(time_ms, gradient_forces, 'b-', linewidth=2, label='Gradient Force', alpha=0.8)
        ax1.plot(time_ms, reluctance_forces, 'r-', linewidth=2, label='Reluctance Force', alpha=0.8)
        ax1.plot(time_ms, lorentz_forces, 'g-', linewidth=2, label='Lorentz Force', alpha=0.8)
        ax1.plot(time_ms, eddy_forces, 'm-', linewidth=2, label='Eddy Current Force', alpha=0.8)
        ax1.plot(time_ms, total_forces, 'k--', linewidth=2, label='Total Force', alpha=0.8)
        
        ax1.set_title('Force Components vs Time')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Force (N)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Force ratios (stacked area plot)
        gradient_fractions = [d.get('force_ratios', {}).get('gradient_fraction', 0) for d in force_data]
        reluctance_fractions = [d.get('force_ratios', {}).get('reluctance_fraction', 0) for d in force_data]
        lorentz_fractions = [d.get('force_ratios', {}).get('lorentz_fraction', 0) for d in force_data]
        eddy_fractions = [d.get('force_ratios', {}).get('eddy_fraction', 0) for d in force_data]
        
        ax2.stackplot(time_ms, gradient_fractions, reluctance_fractions, 
                     lorentz_fractions, eddy_fractions,
                     labels=['Gradient', 'Reluctance', 'Lorentz', 'Eddy'],
                     colors=['blue', 'red', 'green', 'magenta'], alpha=0.7)
        
        ax2.set_title('Force Component Ratios vs Time')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Force Fraction')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/force_component_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_field_visualizations_from_actual_data(self, output_path):
        """Create field visualizations using actual current data."""
        print("Creating field visualizations from actual simulation data...")
        
        if self.simulation_data:
            time_array, current_array = get_actual_current_profile(self.simulation_data)
            if time_array is not None and current_array is not None:
                # Use specific time points for field visualization
                max_current_idx = np.argmax(current_array)
                quarter_time_idx = len(current_array) // 4
                half_time_idx = len(current_array) // 2
                
                time_indices = [quarter_time_idx, half_time_idx, max_current_idx]
                
                for i, idx in enumerate(time_indices):
                    current = current_array[idx]
                    time_point = time_array[idx]
                    
                    print(f"Creating field visualization for t={time_point*1000:.3f}ms, I={current:.1f}A")
                    
                    # 2D field contours at this current
                    field_data = self.field_calculator.calculate_bfield_map_2d(current)
                    self.contour_plotter.plot_bfield_contours(
                        field_data, 
                        save_path=f"{output_path}/field_actual_t{time_point*1000:.1f}ms_{current:.1f}A.png"
                    )
                
                # On-axis field profile using actual max current
                max_current = np.max(current_array)
                field_profile_data = self.field_calculator.calculate_onaxis_field_profile([max_current])
                self.contour_plotter.plot_onaxis_field_profile(
                    field_profile_data,
                    save_path=f"{output_path}/onaxis_field_actual_{max_current:.1f}A.png"
                )
            else:
                print("Warning: No actual current data available, skipping field visualizations")
        else:
            print("Warning: No simulation data loaded, skipping field visualizations")
    
    def _create_circuit_force_analysis(self, simulation_results, output_path):
        """Create circuit and force analysis using actual simulation data."""
        print("Creating circuit and force analysis from actual data...")
        
        # Use the enhanced force plotter that handles actual data
        self.force_plotter.plot_force_current_map(
            simulation_results,
            save_path=f"{output_path}/circuit_force_analysis.png"
        )
    
    def _create_3d_visualizations(self, output_path):
        """Create 3D visualizations using actual simulation parameters."""
        print("Creating 3D visualizations...")
        
        # Use actual max current if available
        if self.simulation_data:
            time_array, current_array = get_actual_current_profile(self.simulation_data)
            if time_array is not None and current_array is not None:
                current = np.max(current_array)
                print(f"Using actual max current {current:.1f}A for 3D visualization")
            else:
                current = 200
        else:
            current = 200
        
        # 3D field visualization
        self.plot3d.plot_3d_field_visualization(
            current,
            save_path=f"{output_path}/3d_field_visualization.png",
            interactive=False
        )
        
        # Geometry rendering
        self.plot3d.render_coil_geometry(
            save_path=f"{output_path}/3d_geometry.png"
        )
    
    def _create_enhanced_physics_analysis(self, simulation_results, output_path):
        """Create enhanced physics analysis using actual simulation data."""
        print("Creating enhanced physics analysis from actual data...")
        
        self.physics_analyzer.plot_enhanced_physics_analysis(
            simulation_results,
            save_path=f"{output_path}/enhanced_physics_analysis.png"
        )
    
    def _create_animations(self, simulation_results, output_path):
        """Create animations using actual simulation data."""
        print("Creating animations from actual simulation data...")
        
        # Projectile motion animation using actual data
        if 'time_series' in simulation_results:
            self.animator.animate_projectile_motion(
                simulation_results['time_series'],
                save_path=f"{output_path}/projectile_motion.gif"
            )
        
        # Field evolution animation using actual current profile
        if self.simulation_data:
            time_array, current_array = get_actual_current_profile(self.simulation_data)
            if time_array is not None and current_array is not None:
                self.animator.animate_field_evolution(
                    time_array, current_array,
                    save_path=f"{output_path}/field_evolution.gif"
                )
    
    def _create_multistage_analysis(self, config_file, time_series_data, output_path):
        """Create multi-stage analysis if applicable."""
        print("Creating multi-stage analysis...")
        
        # This would require summary data which might not be available
        # For now, create basic multi-stage visualizations
        try:
            self.multistage_viz.create_multistage_visualizations(
                config_file, time_series_data, {}, output_path
            )
        except Exception as e:
            print(f"Error creating multi-stage analysis: {e}")
    
    def _menu_load_results(self):
        """Menu option to load simulation results."""
        results_dir = select_results_directory()
        if results_dir:
            config, sim_results, ts_data = load_simulation_from_results(results_dir)
            # Store loaded data for use in other menu options
            self._loaded_config = config
            self._loaded_sim_results = sim_results
            self._loaded_ts_data = ts_data
            print("Results loaded successfully")
        else:
            print("No results loaded")
    
    def _menu_comprehensive_suite(self):
        """Menu option for comprehensive visualization suite."""
        config = getattr(self, '_loaded_config', None)
        sim_results = getattr(self, '_loaded_sim_results', None)
        ts_data = getattr(self, '_loaded_ts_data', None)
        
        output_dir = input("Output directory name [comprehensive_viz]: ") or "comprehensive_viz"
        
        self.create_comprehensive_visualization_suite(
            config, sim_results, ts_data, output_dir
        )
    
    def _menu_quick_analysis(self):
        """Menu option for quick analysis."""
        sim_results = getattr(self, '_loaded_sim_results', None)
        if sim_results:
            self.create_quick_analysis(sim_results)
        else:
            print("No simulation results loaded. Please load results first.")
    
    def _menu_animations(self):
        """Menu option for animations."""
        sim_results = getattr(self, '_loaded_sim_results', None)
        if sim_results:
            print("Animation types: motion, field, circuit")
            anim_type = input("Enter animation type [motion]: ") or "motion"
            save_name = input("Save filename (optional): ") or None
            
            self.animate_simulation(sim_results, anim_type, save_name)
        else:
            print("No simulation results loaded. Please load results first.")
    
    def _menu_multistage_analysis(self):
        """Menu option for multi-stage analysis."""
        config = getattr(self, '_loaded_config', None)
        ts_data = getattr(self, '_loaded_ts_data', None)
        
        if config and ts_data:
            self.analyze_multistage_performance(config, ts_data, {})
        else:
            print("Multi-stage analysis requires configuration and time series data.")
            print("Please load appropriate results first.")
    
    def _create_fast_3d_view(self, current):
        """Create a fast 3D view without field calculations."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # Only show geometry, no field calculations
            print("Creating fast 3D geometry view...")
            self.plot3d._create_simplified_3d_visualization(ax, current, show_coil=True, projectile_position=None)
            plt.show()
            
        except Exception as e:
            print(f"Error creating fast 3D view: {e}")


# Convenience functions for direct use
def create_field_visualization_suite(physics_engine=None, output_dir="field_visualizations"):
    """
    Convenience function to create field visualization suite.
    
    Args:
        physics_engine: CoilgunPhysicsEngine instance
        output_dir: Output directory name
    """
    viz_engine = CoilgunVisualizationEngine(physics_engine)
    return viz_engine.create_field_visualization_suite(output_dir)


def quick_visualization(simulation_results, physics_engine=None):
    """
    Convenience function for quick visualization of simulation results.
    
    Args:
        simulation_results: Simulation results dictionary
        physics_engine: CoilgunPhysicsEngine instance
    """
    viz_engine = CoilgunVisualizationEngine(physics_engine)
    viz_engine.create_quick_analysis(simulation_results)


def launch_interactive_visualization(physics_engine=None):
    """
    Convenience function to launch interactive visualization menu.
    
    Args:
        physics_engine: CoilgunPhysicsEngine instance
    """
    viz_engine = CoilgunVisualizationEngine(physics_engine)
    viz_engine.interactive_menu() 