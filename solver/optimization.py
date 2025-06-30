"""
Optimization and Parametric Study Module

This module provides optimization capabilities and parametric studies
for coilgun simulations.
"""

import numpy as np
import json
import copy
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from scipy.optimize import minimize_scalar, differential_evolution

from .single_stage import SingleStageSimulation
from .multi_stage import MultiStageSimulation
from .core import SolverConfig


class ParametricStudy:
    """
    Parametric study class for systematic parameter variation analysis.
    """
    
    def __init__(self, base_config_file: str, solver_config: Optional[SolverConfig] = None):
        """
        Initialize parametric study.
        
        Args:
            base_config_file: Base configuration file path
            solver_config: Optional solver configuration
        """
        self.base_config_file = base_config_file
        self.solver_config = solver_config or SolverConfig()
        
        # Load base configuration
        with open(base_config_file, 'r') as f:
            self.base_config = json.load(f)
        
        self.study_results = []
        self.parameter_name = None
        self.parameter_values = None
    
    def run_parametric_study(self, parameter_name: str, parameter_values: List[float],
                           output_dir: str = "parametric_study", 
                           use_multistage: bool = False,
                           save_individual_results: bool = False) -> Dict[str, Any]:
        """
        Run parametric study by varying a single parameter.
        
        Args:
            parameter_name: Name of parameter to vary (dot notation, e.g. 'coil.turns')
            parameter_values: List of parameter values to test
            output_dir: Output directory for results
            use_multistage: Whether to use multi-stage simulation
            save_individual_results: Whether to save individual simulation results
            
        Returns:
            Dictionary containing parametric study results
        """
        print(f"\n{'='*60}")
        print(f"PARAMETRIC STUDY: {parameter_name}")
        print(f"{'='*60}")
        print(f"Parameter values: {len(parameter_values)} points")
        print(f"Range: {min(parameter_values):.3f} to {max(parameter_values):.3f}")
        
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.study_results = []
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Run simulations for each parameter value
        for i, param_value in enumerate(parameter_values):
            print(f"\nRunning simulation {i+1}/{len(parameter_values)}: {parameter_name}={param_value:.3f}")
            
            try:
                # Create modified configuration
                config_file = self._create_parameter_config(param_value, output_path)
                
                # Run simulation
                if use_multistage:
                    sim = MultiStageSimulation(config_file, self.solver_config)
                else:
                    sim = SingleStageSimulation(config_file, self.solver_config)
                
                results = sim.run_simulation(
                    save_data=save_individual_results,
                    verbose=False,
                    show_progress=False
                )
                
                # Extract key metrics
                summary = sim.get_summary_results()
                summary['parameter_value'] = param_value
                summary['success'] = True
                summary['error_message'] = None
                
                self.study_results.append(summary)
                
                # Clean up temporary config file
                os.remove(config_file)
                
                print(f"  ✓ Success: final_velocity={summary.get('final_velocity', 0):.2f} m/s, "
                      f"efficiency={summary.get('efficiency', 0)*100:.1f}%")
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                
                # Record failure
                failed_result = {
                    'parameter_value': param_value,
                    'success': False,
                    'error_message': str(e),
                    'final_velocity': 0,
                    'efficiency': 0,
                    'max_current': 0,
                    'max_force': 0
                }
                self.study_results.append(failed_result)
        
        # Process and save results
        study_summary = self._process_parametric_results(output_path)
        
        print(f"\n✓ Parametric study completed")
        print(f"  Results saved to: {output_path}")
        print(f"  Successful simulations: {sum(1 for r in self.study_results if r.get('success', False))}/{len(parameter_values)}")
        
        return study_summary
    
    def _create_parameter_config(self, param_value: float, output_path: Path) -> str:
        """Create configuration file with modified parameter value."""
        # Create a copy of base configuration
        config = copy.deepcopy(self.base_config)
        
        # Set parameter value using dot notation
        self._set_nested_parameter(config, self.parameter_name, param_value)
        
        # Save temporary configuration file
        temp_config_file = output_path / f"temp_param_{param_value:.6f}.json"
        with open(temp_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(temp_config_file)
    
    def _set_nested_parameter(self, config: Dict[str, Any], param_path: str, value: float):
        """Set nested parameter value using dot notation."""
        keys = param_path.split('.')
        current = config
        
        # Navigate to the parent of the target parameter
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final parameter value
        current[keys[-1]] = value
    
    def _process_parametric_results(self, output_path: Path) -> Dict[str, Any]:
        """Process and analyze parametric study results."""
        if not self.study_results:
            return {}
        
        # Extract successful results
        successful_results = [r for r in self.study_results if r.get('success', False)]
        
        if not successful_results:
            print("⚠  No successful simulations in parametric study")
            return {'success': False, 'results': self.study_results}
        
        # Convert to numpy arrays for analysis
        param_values = np.array([r['parameter_value'] for r in successful_results])
        velocities = np.array([r.get('final_velocity', 0) for r in successful_results])
        efficiencies = np.array([r.get('efficiency', 0) for r in successful_results])
        max_currents = np.array([r.get('max_current', 0) for r in successful_results])
        max_forces = np.array([r.get('max_force', 0) for r in successful_results])
        
        # Find optimal points
        max_velocity_idx = np.argmax(velocities)
        max_efficiency_idx = np.argmax(efficiencies)
        
        # Calculate summary statistics
        summary = {
            'parameter_name': self.parameter_name,
            'num_simulations': len(self.study_results),
            'num_successful': len(successful_results),
            'success_rate': len(successful_results) / len(self.study_results),
            'parameter_range': [float(np.min(param_values)), float(np.max(param_values))],
            'velocity_range': [float(np.min(velocities)), float(np.max(velocities))],
            'efficiency_range': [float(np.min(efficiencies)), float(np.max(efficiencies))],
            'optimal_velocity': {
                'parameter_value': float(param_values[max_velocity_idx]),
                'velocity': float(velocities[max_velocity_idx]),
                'efficiency': float(efficiencies[max_velocity_idx])
            },
            'optimal_efficiency': {
                'parameter_value': float(param_values[max_efficiency_idx]),
                'velocity': float(velocities[max_efficiency_idx]),
                'efficiency': float(efficiencies[max_efficiency_idx])
            },
            'all_results': self.study_results
        }
        
        # Save results
        results_file = output_path / "parametric_study_results.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create summary CSV
        self._save_parametric_csv(successful_results, output_path)
        
        # Generate plots if matplotlib available
        try:
            self._plot_parametric_results(summary, output_path)
        except ImportError:
            print("Warning: matplotlib not available, plotting disabled")
        
        return summary
    
    def _save_parametric_csv(self, results: List[Dict[str, Any]], output_path: Path):
        """Save parametric results to CSV file."""
        import csv
        
        if not results:
            return
        
        csv_file = output_path / "parametric_study_data.csv"
        
        # Get all unique keys from results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    def _plot_parametric_results(self, summary: Dict[str, Any], output_path: Path):
        """Plot parametric study results."""
        import matplotlib.pyplot as plt
        
        results = summary['all_results']
        successful_results = [r for r in results if r.get('success', False)]
        
        if len(successful_results) < 2:
            return
        
        param_values = [r['parameter_value'] for r in successful_results]
        velocities = [r.get('final_velocity', 0) for r in successful_results]
        efficiencies = [r.get('efficiency', 0) * 100 for r in successful_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Parametric Study: {summary["parameter_name"]}')
        
        # Velocity vs parameter
        axes[0].plot(param_values, velocities, 'bo-')
        axes[0].set_xlabel(summary["parameter_name"])
        axes[0].set_ylabel('Final Velocity (m/s)')
        axes[0].set_title('Velocity vs Parameter')
        axes[0].grid(True)
        
        # Mark optimal point
        opt_vel = summary['optimal_velocity']
        axes[0].plot(opt_vel['parameter_value'], opt_vel['velocity'], 'ro', 
                    markersize=10, label=f'Max: {opt_vel["velocity"]:.2f} m/s')
        axes[0].legend()
        
        # Efficiency vs parameter
        axes[1].plot(param_values, efficiencies, 'go-')
        axes[1].set_xlabel(summary["parameter_name"])
        axes[1].set_ylabel('Efficiency (%)')
        axes[1].set_title('Efficiency vs Parameter')
        axes[1].grid(True)
        
        # Mark optimal point
        opt_eff = summary['optimal_efficiency']
        axes[1].plot(opt_eff['parameter_value'], opt_eff['efficiency']*100, 'ro',
                    markersize=10, label=f'Max: {opt_eff["efficiency"]*100:.1f}%')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'parametric_study_plots.png', dpi=300, bbox_inches='tight')
        plt.show()


class CoilgunOptimizer:
    """
    Advanced optimization class for coilgun design parameters.
    """
    
    def __init__(self, base_config_file: str, solver_config: Optional[SolverConfig] = None):
        """
        Initialize optimizer.
        
        Args:
            base_config_file: Base configuration file path
            solver_config: Optional solver configuration
        """
        self.base_config_file = base_config_file
        self.solver_config = solver_config or SolverConfig()
        
        with open(base_config_file, 'r') as f:
            self.base_config = json.load(f)
        
        self.optimization_history = []
        self.best_result = None
    
    def optimize_single_parameter(self, parameter_name: str, bounds: Tuple[float, float],
                                objective: str = 'velocity', method: str = 'golden') -> Dict[str, Any]:
        """
        Optimize a single parameter using scalar optimization.
        
        Args:
            parameter_name: Parameter to optimize (dot notation)
            bounds: (min, max) bounds for parameter
            objective: Objective to optimize ('velocity', 'efficiency', 'force')
            method: Optimization method ('golden', 'brent', 'bounded')
            
        Returns:
            Optimization results
        """
        print(f"\n{'='*60}")
        print(f"SINGLE PARAMETER OPTIMIZATION: {parameter_name}")
        print(f"{'='*60}")
        print(f"Bounds: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
        print(f"Objective: {objective}")
        print(f"Method: {method}")
        
        # Define objective function
        def objective_function(param_value):
            try:
                # Create temporary config
                config = copy.deepcopy(self.base_config)
                self._set_nested_parameter(config, parameter_name, param_value)
                
                # Save temporary config file
                temp_config = f"temp_opt_{param_value:.6f}.json"
                with open(temp_config, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Run simulation
                sim = SingleStageSimulation(temp_config, self.solver_config)
                results = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
                
                # Clean up
                os.remove(temp_config)
                
                # Extract objective value
                if objective == 'velocity':
                    obj_value = results.get('final_velocity', 0)
                elif objective == 'efficiency':
                    obj_value = results.get('energy_analysis', {}).get('efficiency', 0)
                elif objective == 'force':
                    obj_value = results.get('max_force', 0)
                else:
                    obj_value = results.get('final_velocity', 0)
                
                # Store in history
                self.optimization_history.append({
                    'parameter_value': param_value,
                    'objective_value': obj_value,
                    'success': True
                })
                
                print(f"  Evaluated: {parameter_name}={param_value:.4f}, {objective}={obj_value:.4f}")
                
                # Return negative for minimization (we want to maximize)
                return -obj_value
                
            except Exception as e:
                print(f"  Failed: {parameter_name}={param_value:.4f}, error: {str(e)}")
                self.optimization_history.append({
                    'parameter_value': param_value,
                    'objective_value': 0,
                    'success': False,
                    'error': str(e)
                })
                return 0  # Return 0 for failed evaluations
        
        # Run optimization
        result = minimize_scalar(
            objective_function,
            bounds=bounds,
            method=method
        )
        
        # Process results
        if result.success:
            optimal_value = result.x
            optimal_objective = -result.fun
            
            print(f"\n✓ Optimization completed successfully")
            print(f"  Optimal {parameter_name}: {optimal_value:.4f}")
            print(f"  Optimal {objective}: {optimal_objective:.4f}")
            
            # Run final simulation with optimal parameters
            final_config = copy.deepcopy(self.base_config)
            self._set_nested_parameter(final_config, parameter_name, optimal_value)
            
            temp_config = "temp_final_opt.json"
            with open(temp_config, 'w') as f:
                json.dump(final_config, f, indent=2)
            
            sim = SingleStageSimulation(temp_config, self.solver_config)
            final_results = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
            
            os.remove(temp_config)
            
            self.best_result = {
                'parameter_name': parameter_name,
                'optimal_value': optimal_value,
                'optimal_objective': optimal_objective,
                'objective_type': objective,
                'optimization_method': method,
                'scipy_result': result,
                'final_simulation': final_results,
                'optimization_history': self.optimization_history.copy()
            }
            
        else:
            print(f"\n❌ Optimization failed: {result.message}")
            self.best_result = {
                'success': False,
                'message': result.message,
                'optimization_history': self.optimization_history.copy()
            }
        
        return self.best_result
    
    def optimize_multiple_parameters(self, parameter_config: Dict[str, Tuple[float, float]],
                                   objective: str = 'velocity', method: str = 'differential_evolution',
                                   max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize multiple parameters simultaneously.
        
        Args:
            parameter_config: Dictionary of {parameter_name: (min, max)} bounds
            objective: Objective to optimize
            method: Optimization method
            max_iterations: Maximum iterations
            
        Returns:
            Optimization results
        """
        print(f"\n{'='*60}")
        print(f"MULTI-PARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Parameters: {list(parameter_config.keys())}")
        print(f"Objective: {objective}")
        print(f"Method: {method}")
        
        parameter_names = list(parameter_config.keys())
        bounds = [parameter_config[name] for name in parameter_names]
        
        # Define objective function
        def objective_function(param_vector):
            try:
                # Create configuration with parameter vector
                config = copy.deepcopy(self.base_config)
                
                for i, param_name in enumerate(parameter_names):
                    self._set_nested_parameter(config, param_name, param_vector[i])
                
                # Save temporary config
                temp_config = f"temp_multi_opt_{len(self.optimization_history)}.json"
                with open(temp_config, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Run simulation
                sim = SingleStageSimulation(temp_config, self.solver_config)
                results = sim.run_simulation(save_data=False, verbose=False, show_progress=False)
                
                # Clean up
                os.remove(temp_config)
                
                # Extract objective value
                if objective == 'velocity':
                    obj_value = results.get('final_velocity', 0)
                elif objective == 'efficiency':
                    obj_value = results.get('energy_analysis', {}).get('efficiency', 0)
                elif objective == 'force':
                    obj_value = results.get('max_force', 0)
                else:
                    obj_value = results.get('final_velocity', 0)
                
                # Store in history
                history_entry = {
                    'iteration': len(self.optimization_history),
                    'objective_value': obj_value,
                    'success': True
                }
                
                for i, param_name in enumerate(parameter_names):
                    history_entry[param_name] = param_vector[i]
                
                self.optimization_history.append(history_entry)
                
                if len(self.optimization_history) % 10 == 0:
                    print(f"  Iteration {len(self.optimization_history)}: {objective}={obj_value:.4f}")
                
                # Return negative for minimization
                return -obj_value
                
            except Exception as e:
                print(f"  Failed iteration {len(self.optimization_history)}: {str(e)}")
                
                history_entry = {
                    'iteration': len(self.optimization_history),
                    'objective_value': 0,
                    'success': False,
                    'error': str(e)
                }
                
                for i, param_name in enumerate(parameter_names):
                    history_entry[param_name] = param_vector[i]
                
                self.optimization_history.append(history_entry)
                return 0
        
        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=max_iterations,
                seed=42  # For reproducibility
            )
        else:
            raise ValueError(f"Unsupported multi-parameter optimization method: {method}")
        
        # Process results
        if result.success:
            optimal_params = result.x
            optimal_objective = -result.fun
            
            print(f"\n✓ Multi-parameter optimization completed")
            print(f"  Optimal {objective}: {optimal_objective:.4f}")
            
            for i, param_name in enumerate(parameter_names):
                print(f"  Optimal {param_name}: {optimal_params[i]:.4f}")
            
            # Create optimal configuration
            optimal_config = copy.deepcopy(self.base_config)
            for i, param_name in enumerate(parameter_names):
                self._set_nested_parameter(optimal_config, param_name, optimal_params[i])
            
            self.best_result = {
                'parameter_names': parameter_names,
                'optimal_values': optimal_params.tolist(),
                'optimal_objective': optimal_objective,
                'objective_type': objective,
                'optimization_method': method,
                'scipy_result': result,
                'optimal_config': optimal_config,
                'optimization_history': self.optimization_history.copy()
            }
            
        else:
            print(f"\n❌ Multi-parameter optimization failed: {result.message}")
            self.best_result = {
                'success': False,
                'message': result.message,
                'optimization_history': self.optimization_history.copy()
            }
        
        return self.best_result
    
    def _set_nested_parameter(self, config: Dict[str, Any], param_path: str, value: float):
        """Set nested parameter value using dot notation."""
        keys = param_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def save_optimization_results(self, output_dir: str = "optimization_results"):
        """Save optimization results to files."""
        if not self.best_result:
            print("No optimization results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results as JSON
        results_file = output_path / "optimization_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.best_result, f, indent=2, default=str)
        
        # Save optimization history as CSV
        if self.optimization_history:
            import csv
            
            csv_file = output_path / "optimization_history.csv"
            fieldnames = list(self.optimization_history[0].keys())
            
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.optimization_history)
        
        print(f"✓ Optimization results saved to {output_path}")


# Standalone functions for backward compatibility
def parametric_study(base_config_file: str, parameter_name: str, parameter_values: List[float],
                    output_dir: str = "parametric_study") -> Dict[str, Any]:
    """
    Run parametric study (standalone function for backward compatibility).
    
    Args:
        base_config_file: Base configuration file
        parameter_name: Parameter to vary
        parameter_values: List of parameter values
        output_dir: Output directory
        
    Returns:
        Parametric study results
    """
    study = ParametricStudy(base_config_file)
    return study.run_parametric_study(parameter_name, parameter_values, output_dir) 