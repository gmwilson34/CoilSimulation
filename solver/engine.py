"""
Main Solver Engine

This module provides the unified CoilgunSolver interface that integrates
all solver components while maintaining compatibility with existing code.
Enhanced to use the new physics engine capabilities.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .core import SolverConfig, SolverError
from .single_stage import EnhancedSingleStageSimulation, SingleStageSimulation
from .multi_stage import EnhancedMultiStageSimulation
from .optimization import ParametricStudy, CoilgunOptimizer
from .utils import (find_config_files, select_config_file, load_and_validate_config,
                   print_config_summary, create_results_report, cleanup_temp_files)


class EnhancedCoilgunSolver:
    """
    Enhanced unified coilgun solver interface that provides access to all solver capabilities
    through a single, easy-to-use class with full physics engine integration.
    """
    
    def __init__(self, config_file: Optional[str] = None, solver_config: Optional[SolverConfig] = None):
        """
        Initialize the enhanced coilgun solver.
        
        Args:
            config_file: Path to configuration file (if None, will prompt for selection)
            solver_config: Optional solver configuration
        """
        # Handle config file selection
        if config_file is None:
            config_file = select_config_file()
            if config_file is None:
                raise SolverError("No configuration file selected")
        
        self.config_file = config_file
        self.solver_config = solver_config or SolverConfig()
        
        # Load and validate configuration
        config, is_valid, errors = load_and_validate_config(config_file)
        if not is_valid:
            raise SolverError(f"Configuration validation failed: {errors}")
        
        self.config = config
        
        # Initialize enhanced simulation objects
        self.single_stage_sim = None
        self.multi_stage_sim = None
        self.optimizer = None
        self.parametric_study = None
        
        # Results storage
        self.last_results = None
        self.simulation_history = []
        
        # Enhanced physics settings
        self.enable_enhanced_physics = self.solver_config.get('physics.enable_enhanced', True)
        self.enable_advanced_analysis = self.solver_config.get('analysis.enable_advanced', True)
        
        print(f"✓ Enhanced CoilgunSolver initialized with: {config_file}")
        print(f"  - Enhanced Physics: {'Enabled' if self.enable_enhanced_physics else 'Disabled'}")
        print(f"  - Advanced Analysis: {'Enabled' if self.enable_advanced_analysis else 'Disabled'}")
        
        if self.solver_config.get('general.show_config_summary', True):
            print_config_summary(self.config)
    
    def run_single_stage(self, use_enhanced_solver: bool = None, **kwargs) -> Dict[str, Any]:
        """
        Run single stage simulation with enhanced physics.
        
        Args:
            use_enhanced_solver: Whether to use enhanced solver (auto-detected if None)
            **kwargs: Additional simulation options
            
        Returns:
            Simulation results dictionary
        """
        print("\n" + "="*60)
        print("ENHANCED SINGLE STAGE SIMULATION")
        print("="*60)
        
        # Auto-detect enhanced solver usage
        if use_enhanced_solver is None:
            use_enhanced_solver = self.enable_enhanced_physics
        
        # Initialize appropriate single stage simulation
        if use_enhanced_solver:
            self.single_stage_sim = EnhancedSingleStageSimulation(self.config_file, self.solver_config)
            print("Using Enhanced Single Stage Simulation with full physics integration")
        else:
            # Fallback to basic simulation for compatibility
            self.single_stage_sim = SingleStageSimulation(self.config_file, self.solver_config)
            print("Using Basic Single Stage Simulation")
        
        # Run simulation with enhanced options
        start_time = time.time()
        simulation_kwargs = kwargs.copy()
        if use_enhanced_solver:
            simulation_kwargs['enable_advanced_analysis'] = self.enable_advanced_analysis
        
        results = self.single_stage_sim.run_simulation(**simulation_kwargs)
        
        # Store enhanced results
        results['solver_type'] = 'enhanced_single_stage' if use_enhanced_solver else 'single_stage'
        results['total_solver_time'] = time.time() - start_time
        results['enhanced_physics_used'] = use_enhanced_solver
        results['advanced_analysis_enabled'] = self.enable_advanced_analysis
        
        self.last_results = results
        self.simulation_history.append(results)
        
        return results
    
    def run_multi_stage(self, use_enhanced_solver: bool = None, **kwargs) -> Dict[str, Any]:
        """
        Run enhanced multi-stage simulation.
        
        Args:
            use_enhanced_solver: Whether to use enhanced solver (auto-detected if None)
            **kwargs: Additional simulation options
            
        Returns:
            Simulation results dictionary
        """
        print("\n" + "="*60)
        print("ENHANCED MULTI-STAGE SIMULATION")
        print("="*60)
        
        # Check if multi-stage configuration is present
        if 'multi_stage' not in self.config:
            raise SolverError("Multi-stage configuration not found in config file")
        
        # Auto-detect enhanced solver usage
        if use_enhanced_solver is None:
            use_enhanced_solver = self.enable_enhanced_physics
        
        # Initialize enhanced multi-stage simulation
        if use_enhanced_solver:
            self.multi_stage_sim = EnhancedMultiStageSimulation(self.config_file, self.solver_config)
            print("Using Enhanced Multi-Stage Simulation with full physics integration")
        else:
            # Would need to create a basic multi-stage solver or fall back
            print("Enhanced multi-stage simulation is the only available option")
            self.multi_stage_sim = EnhancedMultiStageSimulation(self.config_file, self.solver_config)
        
        # Run simulation with enhanced options
        start_time = time.time()
        simulation_kwargs = kwargs.copy()
        simulation_kwargs['enable_advanced_analysis'] = self.enable_advanced_analysis
        
        results = self.multi_stage_sim.run_simulation(**simulation_kwargs)
        
        # Store enhanced results
        results['solver_type'] = 'enhanced_multi_stage'
        results['total_solver_time'] = time.time() - start_time
        results['enhanced_physics_used'] = use_enhanced_solver
        results['advanced_analysis_enabled'] = self.enable_advanced_analysis
        
        self.last_results = results
        self.simulation_history.append(results)
        
        # Clean up temporary files
        self.multi_stage_sim.cleanup_temp_files()
        
        return results
    
    def run_parametric_study(self, parameter_name: str, parameter_values: List[float],
                           output_dir: str = "parametric_study", **kwargs) -> Dict[str, Any]:
        """
        Run parametric study by varying a single parameter.
        
        Args:
            parameter_name: Name of parameter to vary (dot notation)
            parameter_values: List of parameter values to test
            output_dir: Output directory for results
            **kwargs: Additional study options
            
        Returns:
            Parametric study results
        """
        print("\n" + "="*60)
        print("RUNNING PARAMETRIC STUDY")
        print("="*60)
        
        # Initialize parametric study
        self.parametric_study = ParametricStudy(self.config_file, self.solver_config)
        
        # Run study
        start_time = time.time()
        results = self.parametric_study.run_parametric_study(
            parameter_name, parameter_values, output_dir, **kwargs
        )
        
        # Store results
        results['solver_type'] = 'parametric_study'
        results['total_solver_time'] = time.time() - start_time
        self.last_results = results
        self.simulation_history.append(results)
        
        return results
    
    def optimize_parameter(self, parameter_name: str, bounds: tuple, 
                          objective: str = 'velocity', method: str = 'golden') -> Dict[str, Any]:
        """
        Optimize a single parameter.
        
        Args:
            parameter_name: Parameter to optimize (dot notation)
            bounds: (min, max) bounds for parameter
            objective: Objective to optimize ('velocity', 'efficiency', 'force')
            method: Optimization method
            
        Returns:
            Optimization results
        """
        print("\n" + "="*60)
        print("RUNNING PARAMETER OPTIMIZATION")
        print("="*60)
        
        # Initialize optimizer
        if self.optimizer is None:
            self.optimizer = CoilgunOptimizer(self.config_file, self.solver_config)
        
        # Run optimization
        start_time = time.time()
        results = self.optimizer.optimize_single_parameter(
            parameter_name, bounds, objective, method
        )
        
        # Store results
        results['solver_type'] = 'optimization'
        results['total_solver_time'] = time.time() - start_time
        self.last_results = results
        self.simulation_history.append(results)
        
        return results
    
    def optimize_multiple_parameters(self, parameter_config: Dict[str, tuple],
                                   objective: str = 'velocity', 
                                   method: str = 'differential_evolution') -> Dict[str, Any]:
        """
        Optimize multiple parameters simultaneously.
        
        Args:
            parameter_config: Dictionary of {parameter_name: (min, max)} bounds
            objective: Objective to optimize
            method: Optimization method
            
        Returns:
            Optimization results
        """
        print("\n" + "="*60)
        print("RUNNING MULTI-PARAMETER OPTIMIZATION")
        print("="*60)
        
        # Initialize optimizer
        if self.optimizer is None:
            self.optimizer = CoilgunOptimizer(self.config_file, self.solver_config)
        
        # Run optimization
        start_time = time.time()
        results = self.optimizer.optimize_multiple_parameters(
            parameter_config, objective, method
        )
        
        # Store results
        results['solver_type'] = 'multi_optimization'
        results['total_solver_time'] = time.time() - start_time
        self.last_results = results
        self.simulation_history.append(results)
        
        return results
    
    def plot_results(self, save_plots: bool = True, output_dir: str = "solver_results"):
        """
        Plot results from the last simulation.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Output directory for plots
        """
        if self.last_results is None:
            print("❌ No results to plot. Run a simulation first.")
            return
        
        solver_type = self.last_results.get('solver_type', 'unknown')
        
        if solver_type == 'single_stage' and self.single_stage_sim:
            self.single_stage_sim.plot_results(save_plots, output_dir)
        elif solver_type == 'multi_stage' and self.multi_stage_sim:
            self.multi_stage_sim.plot_results(save_plots, output_dir)
        elif solver_type == 'parametric_study' and self.parametric_study:
            # Parametric study plotting is handled internally
            print("Parametric study plots saved during analysis")
        else:
            print(f"Plotting not implemented for solver type: {solver_type}")
    
    def save_results(self, output_dir: str = "solver_results", include_report: bool = True):
        """
        Save results from the last simulation.
        
        Args:
            output_dir: Output directory
            include_report: Whether to include a text report
        """
        if self.last_results is None:
            print("❌ No results to save. Run a simulation first.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results as JSON
        results_file = output_path / "solver_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(self.last_results)
            json.dump(json_results, f, indent=2)
        
        # Save configuration
        config_file = output_path / "configuration.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Create text report if requested
        if include_report:
            report_file = output_path / "simulation_report.txt"
            create_results_report(self.last_results, self.config, str(report_file))
        
        print(f"✓ Results saved to: {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last simulation results.
        
        Returns:
            Summary dictionary
        """
        if self.last_results is None:
            return {}
        
        solver_type = self.last_results.get('solver_type', 'unknown')
        
        if solver_type == 'single_stage' and self.single_stage_sim:
            return self.single_stage_sim.get_summary_results()
        elif solver_type == 'multi_stage' and self.multi_stage_sim:
            return self.multi_stage_sim.get_summary_results()
        elif solver_type in ['parametric_study', 'optimization', 'multi_optimization']:
            # Extract key metrics from optimization/parametric results
            return {
                'solver_type': solver_type,
                'success': self.last_results.get('success', True),
                'total_time': self.last_results.get('total_solver_time', 0)
            }
        else:
            return self.last_results
    
    def get_simulation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all simulations run with this solver.
        
        Returns:
            List of simulation results
        """
        return self.simulation_history.copy()
    
    def compare_simulations(self, indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compare multiple simulations from history.
        
        Args:
            indices: Indices of simulations to compare (if None, compare all)
            
        Returns:
            Comparison results
        """
        if not self.simulation_history:
            print("❌ No simulation history to compare")
            return {}
        
        if indices is None:
            simulations = self.simulation_history
        else:
            simulations = [self.simulation_history[i] for i in indices 
                         if 0 <= i < len(self.simulation_history)]
        
        if len(simulations) < 2:
            print("❌ Need at least 2 simulations to compare")
            return {}
        
        # Extract key metrics for comparison
        comparison = {
            'num_simulations': len(simulations),
            'simulation_types': [sim.get('solver_type', 'unknown') for sim in simulations],
            'comparison_metrics': {}
        }
        
        # Compare common metrics
        metrics = ['final_velocity', 'max_velocity', 'efficiency', 'max_current', 'max_force']
        
        for metric in metrics:
            values = []
            for sim in simulations:
                if metric in sim:
                    values.append(sim[metric])
                elif 'energy_analysis' in sim and metric == 'efficiency':
                    values.append(sim['energy_analysis'].get('efficiency', 0))
            
            if values:
                comparison['comparison_metrics'][metric] = {
                    'values': values,
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values) if len(values) > 1 else 0
                }
        
        return comparison
    
    def _prepare_for_json(self, obj):
        """Recursively prepare object for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        cleanup_temp_files()
        
        if self.multi_stage_sim:
            self.multi_stage_sim.cleanup_temp_files()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience functions for backward compatibility
def run_single_stage_simulation(config_file: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a single stage simulation.
    
    Args:
        config_file: Configuration file path
        **kwargs: Additional simulation options
        
    Returns:
        Simulation results
    """
    with EnhancedCoilgunSolver(config_file) as solver:
        return solver.run_single_stage(**kwargs)


def run_multi_stage_simulation(config_file: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a multi-stage simulation.
    
    Args:
        config_file: Configuration file path
        **kwargs: Additional simulation options
        
    Returns:
        Simulation results
    """
    with EnhancedCoilgunSolver(config_file) as solver:
        return solver.run_multi_stage(**kwargs)


def run_parametric_study(config_file: str, parameter_name: str, 
                        parameter_values: List[float], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run a parametric study.
    
    Args:
        config_file: Configuration file path
        parameter_name: Parameter to vary
        parameter_values: Parameter values to test
        **kwargs: Additional study options
        
    Returns:
        Study results
    """
    with EnhancedCoilgunSolver(config_file) as solver:
        return solver.run_parametric_study(parameter_name, parameter_values, **kwargs)


# Main function for command-line usage
def main():
    """Main function for command-line usage."""
    try:
        # Initialize solver (will prompt for config file)
        solver = EnhancedCoilgunSolver()
        
        # Interactive menu
        while True:
            print("\n" + "="*50)
            print("COILGUN SOLVER - INTERACTIVE MODE")
            print("="*50)
            print("1. Run single stage simulation")
            print("2. Run multi-stage simulation")
            print("3. Run parametric study")
            print("4. Optimize parameter")
            print("5. Plot last results")
            print("6. Save last results")
            print("7. Show simulation history")
            print("8. Exit")
            
            try:
                choice = input("\nSelect option (1-8): ").strip()
                
                if choice == '1':
                    solver.run_single_stage()
                
                elif choice == '2':
                    solver.run_multi_stage()
                
                elif choice == '3':
                    param_name = input("Parameter name (e.g., coil.total_turns): ")
                    value_range = input("Value range (e.g., 500,1000,1500): ")
                    values = [float(v.strip()) for v in value_range.split(',')]
                    solver.run_parametric_study(param_name, values)
                
                elif choice == '4':
                    param_name = input("Parameter name: ")
                    bounds_str = input("Bounds (min,max): ")
                    bounds = tuple(float(b.strip()) for b in bounds_str.split(','))
                    objective = input("Objective (velocity/efficiency/force): ") or 'velocity'
                    solver.optimize_parameter(param_name, bounds, objective)
                
                elif choice == '5':
                    solver.plot_results()
                
                elif choice == '6':
                    solver.save_results()
                
                elif choice == '7':
                    history = solver.get_simulation_history()
                    print(f"\nSimulation History ({len(history)} simulations):")
                    for i, sim in enumerate(history):
                        solver_type = sim.get('solver_type', 'unknown')
                        time_taken = sim.get('total_solver_time', 0)
                        print(f"  {i+1}. {solver_type} ({time_taken:.2f}s)")
                
                elif choice == '8':
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please select 1-8.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    
    finally:
        cleanup_temp_files()


# Backward compatibility aliases
CoilgunSolver = EnhancedCoilgunSolver  # Default to enhanced version

# Legacy function aliases for existing code
def run_single_stage_simulation_legacy(config_file: str, **kwargs) -> Dict[str, Any]:
    """Legacy single stage simulation function (redirects to enhanced version)."""
    return run_single_stage_simulation(config_file, **kwargs)

def run_multi_stage_simulation_legacy(config_file: str, **kwargs) -> Dict[str, Any]:
    """Legacy multi-stage simulation function (redirects to enhanced version)."""
    return run_multi_stage_simulation(config_file, **kwargs)


if __name__ == "__main__":
    main() 