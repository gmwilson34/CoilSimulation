"""
Backward compatibility wrapper for the refactored view package.

This module maintains backward compatibility with the original view.py interface
while leveraging the new modular structure in the view/ package.

The original CoilgunFieldVisualizer class and all functions are now available
through the new modular system but can still be imported from view.py.
"""

# Import everything from the new view package for backward compatibility
from viewer.engine import CoilgunVisualizationEngine

# Import the main classes and functions with their original names
from viewer.fields import MagneticFieldCalculator
from viewer.plots import ContourPlotter, ForcePlotter, ProfilePlotter
from viewer.plots3d import Plot3DVisualizer
from viewer.animations import AnimationEngine
from viewer.analysis import PhysicsAnalyzer
from viewer.multistage import MultistageVisualizer
from viewer.utils import *

# Create an alias for backward compatibility
# The original CoilgunFieldVisualizer class is now CoilgunVisualizationEngine
CoilgunFieldVisualizer = CoilgunVisualizationEngine

# Import all the original standalone functions
from viewer.utils import (
    extract_actual_current_data,
    find_results_directories, 
    select_results_directory,
    load_simulation_from_results,
    create_output_directory,
    validate_simulation_data,
    check_and_fix_empty_simulation_data
)

# Import the main creation functions that were originally in view.py
def create_enhanced_physics_visualizations(time_series_data=None, output_dir="physics_analysis", visualizer=None):
    """
    Create enhanced physics visualizations using actual simulation data.
    
    Args:
        time_series_data: Time series data dictionary (optional, will auto-load)
        output_dir: Output directory path
        visualizer: Visualizer instance (optional)
    """
    if visualizer is None:
        visualizer = CoilgunVisualizationEngine(auto_load_data=True)
    
    # Use actual data if available
    if visualizer.simulation_data and time_series_data is None:
        time_series_data = visualizer.simulation_data.get('time_series', {})
        print("Using actual simulation data for physics analysis")
    
    if hasattr(visualizer, 'physics_analyzer') and time_series_data:
        visualizer.physics_analyzer.plot_enhanced_physics_analysis(
            time_series_data, f"{output_dir}/enhanced_physics.png"
        )
    else:
        print("No time series data available for physics analysis")

def create_multistage_visualizations(config_file, time_series_data, summary_data, output_dir, visualizer=None):
    """
    Create multistage visualizations (backward compatibility function).
    
    Args:
        config_file: Configuration file path
        time_series_data: Time series data dictionary
        summary_data: Summary data dictionary
        output_dir: Output directory path
        visualizer: Visualizer instance (optional)
    """
    if visualizer is None:
        visualizer = CoilgunVisualizationEngine()
    
    if hasattr(visualizer, 'multistage_viz'):
        visualizer.multistage_viz.create_multistage_visualizations(
            config_file, time_series_data, summary_data, output_dir
        )
    else:
        ms_viz = MultistageVisualizer()
        ms_viz.create_multistage_visualizations(
            config_file, time_series_data, summary_data, output_dir
        )

def create_comprehensive_visualization_suite(config_file=None, simulation_results=None, 
                                           time_series_data=None,
                                           output_dir="comprehensive_visualizations"):
    """
    Create comprehensive visualization suite using actual simulation data.
    
    Args:
        config_file: Configuration file path (optional, will auto-load)
        simulation_results: Simulation results dictionary (optional, will auto-load)
        time_series_data: Time series data dictionary (optional, will auto-load)
        output_dir: Output directory name
    """
    print("Creating comprehensive visualization suite using actual simulation data...")
    
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("Using loaded actual simulation data")
        return viz_engine.create_comprehensive_visualization_suite(
            config_file, simulation_results, time_series_data, output_dir
        )
    else:
        print("Warning: No actual simulation data found. Creating basic visualizations...")
        viz_engine = CoilgunVisualizationEngine(auto_load_data=False)
        return viz_engine.create_comprehensive_visualization_suite(
            config_file, simulation_results, time_series_data, output_dir
        )

def create_field_visualization_suite(config_file=None, output_dir="field_visualizations"):
    """
    Create field visualization suite using actual simulation data.
    
    Args:
        config_file: Configuration file path (optional)
        output_dir: Output directory name
    """
    print("Creating field visualization suite using actual simulation data...")
    
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("Using actual current values from simulation data")
    else:
        print("Warning: No simulation data found, using default current values")
    
    return viz_engine.create_field_visualization_suite(output_dir)

# Main function for backward compatibility
def main():
    """
    Main function that launches the interactive visualization menu using actual simulation data.
    This maintains backward compatibility with the original view.py.
    """
    print("Launching Coilgun Visualization Engine...")
    print("Loading actual simulation data from simulation_results/...")
    
    # Create visualization engine with auto-loading of actual data
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("✓ Actual simulation data loaded successfully!")
        print(f"  - {len(viz_engine.simulation_data.get('time', []))} data points")
        print(f"  - Max current: {viz_engine.simulation_params.get('max_current', 0):.2f} A")
        print(f"  - Max force: {viz_engine.simulation_params.get('max_force', 0):.2f} N")
        print(f"  - Simulation time: {viz_engine.simulation_params.get('simulation_time', 0)*1000:.2f} ms")
        
        # Launch interactive menu without automatic quick analysis
        viz_engine.interactive_menu()
    else:
        print("⚠ No simulation data found. Please run a simulation first.")
        print("Expected files in simulation_results/:")
        print("  - enhanced_single_stage_data.csv")
        print("  - enhanced_single_stage_results.json") 
        print("  - force_component_analysis.json")

# Additional backward compatibility aliases
def signal_handler(signum, frame):
    """Signal handler for backward compatibility."""
    from viewer.core import setup_signal_handling
    setup_signal_handling()

def create_demo_visualization():
    """Create demo visualization using actual simulation data if available."""
    print("Creating demo visualization...")
    
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("Using actual simulation data for demo")
        viz_engine.create_comprehensive_visualization_suite(output_dir="demo_visualizations_actual")
    else:
        print("No actual data found, creating basic demo visualization")
        viz_engine.create_field_visualization_suite("demo_visualizations_basic")

def test_enhanced_visualizations(config_file="coilgun_config_expert.json"):
    """Test enhanced visualizations using actual simulation data."""
    print(f"Testing enhanced visualizations...")
    
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("✓ Actual simulation data available for testing")
        print("Creating comprehensive test visualization suite...")
        viz_engine.create_comprehensive_visualization_suite(output_dir="test_visualizations_actual")
    else:
        print("⚠ No actual simulation data found for testing")
        print("Please run a simulation with the expert config first")

def debug_enhanced_plotting():
    """Debug enhanced plotting using actual simulation data."""
    print("Debug mode: checking actual simulation data availability...")
    
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("✓ Simulation data loaded successfully")
        print(f"Available data keys: {viz_engine.simulation_data.get('available_data', [])}")
        
        # Create quick analysis for debugging
        viz_engine.create_quick_analysis(save_prefix="debug_analysis")
        
        # Launch interactive menu
        viz_engine.interactive_menu()
    else:
        print("✗ No simulation data found")
        print("Run the following to generate simulation data:")
        print("  python solve.py")
        
def launch_interactive_visualization():
    """Launch interactive visualization with actual simulation data."""
    print("Launching interactive visualization engine...")
    
    viz_engine = CoilgunVisualizationEngine(auto_load_data=True)
    
    if viz_engine.simulation_data:
        print("Ready with actual simulation data!")
    else:
        print("Warning: No simulation data loaded")
    
    viz_engine.interactive_menu()

# Ensure all original functionality is available
if __name__ == "__main__":
    main()
