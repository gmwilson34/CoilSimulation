#!/usr/bin/env python3
"""
Advanced Coilgun Simulation - Main Entry Point

This module provides backward compatibility with the original solve.py while
leveraging the new modular solver architecture. It maintains all original
class names and interfaces for seamless migration.

The original large monolithic solve.py has been refactored into a modular
structure located in the solver/ directory:

- solver.core: Base classes and configuration
- solver.single_stage: Single stage simulation
- solver.multi_stage: Multi-stage simulation  
- solver.analysis: Results analysis and processing
- solver.optimization: Parameter optimization and studies
- solver.progress: Progress tracking
- solver.utils: Utility functions
- solver.engine: Unified solver interface

This file now serves as a compatibility layer and main entry point.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import the new modular solver components
try:
    from solver import (
        EnhancedCoilgunSolver,
        CoilgunSolver,
        EnhancedSingleStageSimulation,
        SingleStageSimulation,
        EnhancedMultiStageSimulation,
        ParametricStudy,
        CoilgunOptimizer,
        SolverConfig,
        cleanup_temp_files
    )
    from solver.engine import (
        run_single_stage_simulation,
        run_multi_stage_simulation,
        run_parametric_study,
        main as solver_main
    )
    from solver.utils import (
        find_config_files,
        select_config_file,
        load_and_validate_config,
        print_config_summary,
        create_default_config
    )
    from solver.progress import ProgressTracker, SimpleProgressTracker
    from solver.analysis import ResultsAnalyzer
    
    SOLVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import new solver modules: {e}")
    print("Falling back to legacy mode...")
    SOLVER_AVAILABLE = False


# Backward compatibility aliases for original class names
if SOLVER_AVAILABLE:
    # Main simulation classes (enhanced by default, with fallback options)
    CoilgunSimulation = EnhancedSingleStageSimulation  # Enhanced by default
    MultiStageCoilgunSimulation = EnhancedMultiStageSimulation  # Enhanced by default
    
    # Legacy aliases for old code
    SingleStageSimulation = EnhancedSingleStageSimulation
    MultiStageSimulation = EnhancedMultiStageSimulation
    
    # Utility classes
    class LegacyProgressTracker(ProgressTracker):
        """Legacy progress tracker with original interface."""
        
        def __init__(self, t_span, **kwargs):
            # Convert old interface to new
            physics_engine = kwargs.pop('physics_engine', None)
            super().__init__(t_span, physics_engine, **kwargs)
    
    # Original utility functions
    def find_configuration_files(directory="."):
        """Original function name for finding config files."""
        return find_config_files(directory)
    
    def select_configuration_file():
        """Original function name for config file selection."""
        return select_config_file()


def print_header():
    """Print the simulation header."""
    print("\n" + "="*70)
    print("ADVANCED COILGUN SIMULATION")
    print("="*70)
    print("Refactored modular architecture with backward compatibility")
    print("Version: 2.0 (Modular)")
    print("="*70)


def main():
    """
    Main function - provides both new interface and legacy compatibility.
    """
    print_header()
    
    # Handle command line arguments for config file
    config_file = None
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return
    
    # Select config file if not provided
    if config_file is None and SOLVER_AVAILABLE:
        config_file = select_config_file()
        if config_file is None:
            print("❌ No configuration file selected")
            return
    
    # Run simulation directly if we have a config file
    if config_file and SOLVER_AVAILABLE:
        print(f"\n🚀 Starting simulation with: {config_file}")
        try:
            # Load config to determine simulation type
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Check for multi-stage configuration
            is_multistage = 'multi_stage' in config and config['multi_stage'].get('num_stages', 1) > 1
            
            # Run appropriate simulation
            if is_multistage:
                print("\n🔗 Running multi-stage simulation...")
                results = run_multi_stage_simulation(config_file, verbose=True)
            else:
                print("\n⚡ Running single stage simulation...")
                results = run_single_stage_simulation(config_file, verbose=True)
            
            # Print summary
            print("\n" + "="*60)
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print("="*60)
            
            if 'final_velocity' in results:
                print(f"Final velocity: {results['final_velocity']:.2f} m/s")
            if 'energy_analysis' in results:
                efficiency = results['energy_analysis'].get('efficiency', 0) * 100
                print(f"Efficiency: {efficiency:.1f}%")
            
            print("\n✓ Results saved in simulation output directory")
            
        except KeyboardInterrupt:
            print("\n⚠  Simulation interrupted by user")
        except Exception as e:
            print(f"\n❌ Simulation failed: {str(e)}")
            import traceback
            print("\nDetailed error:")
            traceback.print_exc()
    else:
        # Fallback to legacy mode or interactive mode
        if not SOLVER_AVAILABLE:
            print("\n❌ New modular solver not available, using legacy mode...")
        legacy_main()


def legacy_main():
    """
    Legacy main function for backward compatibility.
    """
    print("\n⚡ Legacy compatibility mode")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if not os.path.exists(config_file):
            print(f"❌ Configuration file not found: {config_file}")
            return
    else:
        # Interactive config selection
        if SOLVER_AVAILABLE:
            config_file = select_config_file()
        else:
            # Fallback for when solver modules not available
            config_files = [f for f in os.listdir('.') if f.endswith('.json')]
            if not config_files:
                print("❌ No JSON configuration files found in current directory")
                return
            
            print("\nAvailable configuration files:")
            for i, f in enumerate(config_files):
                print(f"  {i+1}. {f}")
            
            try:
                choice = int(input("\nSelect configuration file (number): ")) - 1
                if 0 <= choice < len(config_files):
                    config_file = config_files[choice]
                else:
                    print("❌ Invalid selection")
                    return
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Selection cancelled")
                return
    
    print(f"\n📁 Using configuration: {config_file}")
    
    # Load configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Check for multi-stage configuration
    is_multistage = 'multi_stage' in config and config['multi_stage'].get('num_stages', 1) > 1
    
    try:
        if SOLVER_AVAILABLE:
            # Use new modular solver
            if is_multistage:
                print("\n🔗 Running multi-stage simulation...")
                results = run_multi_stage_simulation(config_file, verbose=True)
            else:
                print("\n⚡ Running single stage simulation...")
                results = run_single_stage_simulation(config_file, verbose=True)
            
            # Print summary
            print("\n" + "="*60)
            print("SIMULATION COMPLETED SUCCESSFULLY")
            print("="*60)
            
            if 'final_velocity' in results:
                print(f"Final velocity: {results['final_velocity']:.2f} m/s")
            if 'energy_analysis' in results:
                efficiency = results['energy_analysis'].get('efficiency', 0) * 100
                print(f"Efficiency: {efficiency:.1f}%")
            
            print("\n✓ Results saved in simulation output directory")
            
        else:
            # Fallback message when new solver not available
            print("\n❌ New modular solver not available")
            print("Please ensure the solver directory and modules are properly installed")
            print("\nTo use the new solver:")
            print("1. Ensure all files in solver/ directory are present")
            print("2. Check that physics module is available")
            print("3. Verify all dependencies are installed")
            
    except Exception as e:
        print(f"\n❌ Simulation failed: {str(e)}")
        import traceback
        print("\nDetailed error:")
        traceback.print_exc()


# Legacy convenience functions for backward compatibility
def create_simulation(config_file: str, **kwargs):
    """
    Create a simulation object (legacy compatibility).
    
    Args:
        config_file: Configuration file path
        **kwargs: Additional options
        
    Returns:
        Simulation object
    """
    if not SOLVER_AVAILABLE:
        raise RuntimeError("New solver modules not available")
    
    # Load config to determine type
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if 'multi_stage' in config and config['multi_stage'].get('num_stages', 1) > 1:
        return MultiStageSimulation(config_file, **kwargs)
    else:
        return SingleStageSimulation(config_file, **kwargs)


def run_simulation(config_file: str, **kwargs) -> Dict[str, Any]:
    """
    Run a simulation (legacy compatibility).
    
    Args:
        config_file: Configuration file path
        **kwargs: Additional options
        
    Returns:
        Simulation results
    """
    if not SOLVER_AVAILABLE:
        raise RuntimeError("New solver modules not available")
    
    sim = create_simulation(config_file, **kwargs)
    return sim.run_simulation(**kwargs)


def parametric_study(config_file: str, parameter_name: str, 
                    parameter_values: List[float], **kwargs) -> Dict[str, Any]:
    """
    Run parametric study (legacy compatibility).
    
    Args:
        config_file: Configuration file path
        parameter_name: Parameter to vary
        parameter_values: Parameter values to test
        **kwargs: Additional options
        
    Returns:
        Study results
    """
    if not SOLVER_AVAILABLE:
        raise RuntimeError("New solver modules not available")
    
    return run_parametric_study(config_file, parameter_name, parameter_values, **kwargs)


# Make key classes and functions available at module level for imports
if SOLVER_AVAILABLE:
    # Export key classes with original names
    __all__ = [
        'CoilgunSimulation',  # Alias for SingleStageSimulation
        'MultiStageCoilgunSimulation',  # Alias for MultiStageSimulation
        'CoilgunSolver',
        'ParametricStudy',
        'CoilgunOptimizer', 
        'ProgressTracker',
        'LegacyProgressTracker',
        'ResultsAnalyzer',
        'SolverConfig',
        'create_simulation',
        'run_simulation',
        'parametric_study',
        'find_configuration_files',
        'select_configuration_file',
        'main'
    ]
else:
    __all__ = ['main', 'legacy_main']


# Enhanced error handling and helpful messages
def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
    
    try:
        import physics
    except ImportError:
        missing_deps.append('physics (custom physics engine module)')
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install missing dependencies and ensure physics module is available")
        return False
    
    return True


if __name__ == "__main__":
    # Check dependencies before running
    if not check_dependencies():
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Simulation interrupted by user")
        cleanup_temp_files()
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        cleanup_temp_files()
        sys.exit(1)
    finally:
        # Clean up any temporary files
        if SOLVER_AVAILABLE:
            cleanup_temp_files()
