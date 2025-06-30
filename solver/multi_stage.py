"""
Multi-Stage Coilgun Simulation

This module provides multi-stage coilgun simulation with velocity transfer
between stages and comprehensive results aggregation.
Enhanced to use the new physics engine capabilities.
"""

import numpy as np
import json
import copy
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import signal
import os
from datetime import datetime

from physics import CoilgunPhysicsEngine
from physics.forces import ForceAnalyzer
from physics.circuits import EnergyAnalyzer
from physics.materials import AdvancedMaterialProperties
from .core import BaseSolver, SolverConfig, SolverError, SimulationError
from .single_stage import EnhancedSingleStageSimulation
from .analysis import ResultsAnalyzer


class EnhancedMultiStageSimulation(BaseSolver):
    """
    Enhanced multi-stage coilgun simulation with advanced physics integration.
    Now leverages the enhanced single-stage simulation and comprehensive physics analysis.
    """
    
    def __init__(self, config_file: str, solver_config: Optional[SolverConfig] = None):
        """
        Initialize enhanced multi-stage simulation.
        
        Args:
            config_file: Path to configuration file
            solver_config: Optional solver configuration
        """
        super().__init__(config_file, solver_config)
        
        # Initialize enhanced physics analyzers
        self.force_analyzer = ForceAnalyzer(self.physics.forces)
        self.energy_analyzer = EnergyAnalyzer(self.physics.circuit_model)
        self.results_analyzer = ResultsAnalyzer(self.physics)
        
        # Advanced material properties integration
        self.material_properties = self.physics.materials
        
        # Multi-stage specific attributes with enhanced physics awareness
        multi_stage_config = self.physics_config.get('multi_stage', {})
        self.num_stages = multi_stage_config.get('num_stages', 1)
        self.stage_spacing = multi_stage_config.get('stage_spacing', 0.1)
        self.velocity_transfer_efficiency = multi_stage_config.get('velocity_transfer_efficiency', 1.0)
        
        # Enhanced physics settings
        self.enable_interstage_analysis = multi_stage_config.get('enable_interstage_analysis', True)
        self.enable_cumulative_energy_tracking = multi_stage_config.get('enable_cumulative_energy_tracking', True)
        self.stage_interaction_model = multi_stage_config.get('stage_interaction_model', 'independent')
        
        # Physics validation settings
        self.enable_physics_validation = self.solver_config.get('physics.enable_validation', True)
        self.enable_energy_conservation = self.solver_config.get('physics.energy_conservation', True)
        
        # Stage-specific results with enhanced analysis
        self.stage_results = []
        self.stage_simulations = []
        self.interstage_analysis = []
        
        # Overall results
        self.overall_results = {}
        self.cumulative_energy_tracking = {}
        
        # Initialize energy tracking
        if self.enable_cumulative_energy_tracking:
            self.energy_analyzer.initialize_energy_tracking()
        
        # Multi-stage specific initialization
        self.config_file = config_file
        
        # Load main configuration
        self.config = self.physics_config  # Use already loaded config from parent
        
        # Validate multi-stage configuration
        if 'multi_stage' not in self.config:
            raise SolverError("Multi-stage configuration not found in config file")
        
        self.num_stages = self.config['multi_stage']['num_stages']
        
        # Enhanced multi-stage settings
        self.velocity_transfer_efficiency = self.config.get('multi_stage', {}).get('velocity_transfer_efficiency', 0.95)
        self.enable_interstage_analysis = self.config.get('multi_stage', {}).get('enable_interstage_analysis', True)
        self.stage_separation_distance = self.config.get('multi_stage', {}).get('stage_separation_distance', 0.01)  # 1 cm
        
        # Enhanced tracking
        self.stage_results = []
        self.stage_simulations = []
        self.interstage_analysis = []
        self.overall_results = {}
        
        # Enhanced physics settings
        self.enable_enhanced_physics = self.solver_config.get('physics.enable_enhanced_physics', True)
        self.enable_advanced_analysis = self.solver_config.get('analysis.enable_advanced_analysis', True)
        
        # Signal handling for graceful interruption
        self._setup_signal_handlers()
        
        print(f"✓ Enhanced Multi-Stage Solver Initialized")
        print(f"  - Physics Engine: {type(self.physics).__name__}")
        print(f"  - Force Calculator: {type(self.physics.forces).__name__}")
        print(f"  - Material System: {type(self.material_properties).__name__}")
        print(f"  - Stages: {self.num_stages}")
        print(f"  - Stage spacing: {self.stage_spacing*1000:.1f}mm")
        print(f"  - Transfer efficiency: {self.velocity_transfer_efficiency*100:.1f}%")
        print(f"  - Interstage analysis: {'Enabled' if self.enable_interstage_analysis else 'Disabled'}")
        print(f"  - Energy tracking: {'Enabled' if self.enable_cumulative_energy_tracking else 'Disabled'}")
        print(f"  - Enhanced Physics: {'Enabled' if self.enable_enhanced_physics else 'Disabled'}")
        print(f"  - Advanced Analysis: {'Enabled' if self.enable_advanced_analysis else 'Disabled'}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption."""
        def signal_handler(signum, frame):
            print("\n⚠  Multi-stage simulation interrupted by user")
            
            # Dump current simulation results to file
            self._dump_interrupted_multistage_results()
            
            raise KeyboardInterrupt("Multi-stage simulation interrupted by user")
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def run_simulation(self, save_data: bool = True, verbose: bool = True, 
                      show_progress: bool = True, enable_advanced_analysis: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """
        Run the enhanced multi-stage coilgun simulation.
        
        Args:
            save_data: Whether to save simulation data
            verbose: Enable verbose output
            show_progress: Show progress bar during simulation
            enable_advanced_analysis: Enable comprehensive physics analysis
            **kwargs: Additional solver options
            
        Returns:
            Dictionary containing comprehensive aggregated simulation results
        """
        if verbose:
            print("\n" + "="*60)
            print("ENHANCED MULTI-STAGE COILGUN SIMULATION")
            print("="*60)
            self._print_enhanced_multistage_info()
        
        try:
            # Initialize enhanced tracking
            self.stage_results = []
            self.stage_simulations = []
            self.interstage_analysis = []
            
            # Initialize projectile state with enhanced tracking
            projectile_state = {
                'velocity': 0.0,
                'position': 0.0,
                'total_energy': 0.0,
                'cumulative_losses': 0.0,
                'stage_efficiencies': []
            }
            
            # Run each stage sequentially with enhanced analysis
            for stage_num in range(1, self.num_stages + 1):
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"ENHANCED STAGE {stage_num}/{self.num_stages}")
                    print(f"{'='*50}")
                    self._print_stage_initial_state(projectile_state, stage_num)
                
                # Perform interstage analysis if enabled
                if stage_num > 1 and self.enable_interstage_analysis:
                    interstage_data = self._perform_interstage_analysis(
                        projectile_state, stage_num, verbose
                    )
                    self.interstage_analysis.append(interstage_data)
                
                # Create enhanced stage configuration
                stage_config_file = self._create_enhanced_stage_config(stage_num, projectile_state)
                
                # Initialize enhanced stage simulation
                stage_sim = EnhancedSingleStageSimulation(stage_config_file, self.solver_config)
                
                # Modify initial conditions for enhanced velocity transfer
                if projectile_state['velocity'] > 0:
                    self._modify_enhanced_stage_initial_conditions(stage_sim, projectile_state)
                
                # Run enhanced stage simulation
                stage_results = stage_sim.run_simulation(
                    save_data=False,  # Save data only at the end
                    verbose=verbose,
                    show_progress=show_progress,
                    enable_advanced_analysis=enable_advanced_analysis,
                    **kwargs
                )
                
                # Store enhanced stage results
                self.stage_results.append(stage_results)
                self.stage_simulations.append(stage_sim)
                
                # Update projectile state with enhanced physics
                projectile_state = self._update_enhanced_projectile_state(
                    projectile_state, stage_results, stage_num
                )
                
                if verbose:
                    self._print_stage_completion_info(stage_results, projectile_state, stage_num)
                
                # Enhanced termination conditions
                if self._check_enhanced_termination_conditions(projectile_state, stage_num):
                    if verbose:
                        print(f"🛑 Enhanced termination conditions met after stage {stage_num}")
                    break
            
            # Process enhanced overall results
            self.overall_results = self._process_enhanced_overall_results(
                save_data, verbose, enable_advanced_analysis
            )
            
            if verbose:
                self._print_enhanced_overall_results()
            
            return self.overall_results
            
        except Exception as e:
            print(f"\n❌ Enhanced multi-stage simulation failed: {str(e)}")
            raise SimulationError(f"Enhanced multi-stage simulation failed: {str(e)}") from e
    
    def _create_enhanced_stage_config(self, stage_num: int, projectile_state: Dict[str, Any]) -> str:
        """
        Create configuration file for a specific stage.
        
        Args:
            stage_num: Stage number (1-indexed)
            projectile_state: Current state of the projectile
            
        Returns:
            Path to stage configuration file
        """
        # Create a copy of the original configuration
        stage_config = copy.deepcopy(self.physics_config)
        
        # Modify configuration for this stage
        if 'multi_stage' in stage_config:
            stage_settings = stage_config['multi_stage'].get('stage_settings', {})
            
            # Apply stage-specific settings if available
            stage_key = f'stage_{stage_num}'
            if stage_key in stage_settings:
                stage_specific = stage_settings[stage_key]
                
                # Update coil parameters
                if 'coil' in stage_specific:
                    stage_config['coil'].update(stage_specific['coil'])
                
                # Update capacitor parameters
                if 'capacitor' in stage_specific:
                    stage_config['capacitor'].update(stage_specific['capacitor'])
                
                # Update projectile parameters (usually just initial position)
                if 'projectile' in stage_specific:
                    stage_config['projectile'].update(stage_specific['projectile'])
        
        # Adjust projectile initial position for stage spacing
        if 'projectile' not in stage_config:
            stage_config['projectile'] = {}
        
        # Position projectile at entrance of this stage
        stage_offset = (stage_num - 1) * self.stage_spacing
        coil_start = -stage_config.get('coil', {}).get('length', 0.05) / 2.0
        initial_position = coil_start - stage_config.get('projectile', {}).get('length', 0.01) / 2.0
        
        stage_config['projectile']['initial_position'] = initial_position - stage_offset
        
        # Save stage configuration to temporary file
        stage_config_file = f"temp_stage_{stage_num}_config.json"
        with open(stage_config_file, 'w') as f:
            json.dump(stage_config, f, indent=2)
        
        return stage_config_file
    
    def _modify_enhanced_stage_initial_conditions(self, stage_sim: EnhancedSingleStageSimulation, projectile_state: Dict[str, Any]):
        """
        Modify stage simulation to start with transferred velocity.
        
        Args:
            stage_sim: Stage simulation object
            projectile_state: Current state of the projectile
        """
        # Store the original get_initial_conditions method
        original_get_initial_conditions = stage_sim.get_initial_conditions
        
        # Create a modified version that includes initial velocity
        def modified_get_initial_conditions():
            Q0, I0, x0, v0 = original_get_initial_conditions()
            return Q0, I0, x0, projectile_state['velocity']
        
        # Replace the method
        stage_sim.get_initial_conditions = modified_get_initial_conditions
    
    def _perform_interstage_analysis(self, projectile_state: Dict[str, Any], stage_num: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Perform analysis between stages.
        
        Args:
            projectile_state: Current state of the projectile
            stage_num: Current stage number
            verbose: Enable verbose output
            
        Returns:
            Interstage analysis data
        """
        if verbose:
            print(f"🔬 Performing interstage analysis before stage {stage_num}")
        
        interstage_data = {
            'stage_number': stage_num,
            'entry_velocity': projectile_state['velocity'],
            'entry_position': projectile_state['position'],
            'cumulative_energy': projectile_state['total_energy'],
            'cumulative_losses': projectile_state['cumulative_losses'],
            'velocity_transfer_loss': (1.0 - self.velocity_transfer_efficiency) * projectile_state['velocity'],
            'stage_efficiencies': projectile_state['stage_efficiencies'].copy()
        }
        
        # Calculate velocity transfer losses
        if projectile_state['velocity'] > 0:
            transfer_loss = projectile_state['velocity'] * (1.0 - self.velocity_transfer_efficiency)
            interstage_data['velocity_transfer_loss'] = transfer_loss
            
            if verbose:
                print(f"  Entry velocity: {projectile_state['velocity']:.3f} m/s")
                print(f"  Transfer efficiency: {self.velocity_transfer_efficiency*100:.1f}%")
                print(f"  Velocity loss: {transfer_loss:.3f} m/s")
        
        # Add thermal analysis if enabled
        if self.enable_enhanced_physics:
            interstage_data['thermal_analysis'] = self._analyze_interstage_thermal_effects(projectile_state)
        
        return interstage_data
    
    def _analyze_interstage_thermal_effects(self, projectile_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze thermal effects between stages."""
        # Placeholder for thermal analysis
        return {
            'temperature_rise': 0.0,
            'thermal_expansion': 0.0,
            'resistance_change': 0.0
        }
    
    def _update_enhanced_projectile_state(self, projectile_state: Dict[str, Any], stage_results: Dict[str, Any], stage_num: int) -> Dict[str, Any]:
        """
        Update projectile state with results from the current stage.
        
        Args:
            projectile_state: Current state of the projectile
            stage_results: Results from the current stage simulation
            stage_num: Current stage number
            
        Returns:
            Updated projectile state
        """
        exit_velocity = stage_results.get('exit_velocity', stage_results.get('final_velocity', 0))
        energy_loss = stage_results.get('energy_loss', 0)
        
        # Update velocity
        projectile_state['velocity'] = exit_velocity * self.velocity_transfer_efficiency
        
        # Update position
        projectile_state['position'] += exit_velocity * stage_results.get('time', 0)
        
        # Update total energy
        projectile_state['total_energy'] += stage_results.get('energy_analysis', {}).get('final_kinetic_energy', 0)
        
        # Update cumulative losses
        projectile_state['cumulative_losses'] += energy_loss
        
        # Update stage efficiency
        projectile_state['stage_efficiencies'].append(stage_results.get('energy_analysis', {}).get('efficiency', 0))
        
        return projectile_state
    
    def _check_enhanced_termination_conditions(self, projectile_state: Dict[str, Any], stage_num: int) -> bool:
        """
        Check termination conditions for the enhanced multi-stage simulation.
        
        Args:
            projectile_state: Current state of the projectile
            stage_num: Current stage number
            
        Returns:
            True if termination conditions are met, False otherwise
        """
        # Implement termination conditions based on the current state of the projectile
        # This is a placeholder and should be replaced with the actual logic
        return False
    
    def _process_enhanced_overall_results(self, save_data: bool, verbose: bool, enable_advanced_analysis: bool) -> Dict[str, Any]:
        """Process and aggregate results from all stages."""
        if not self.stage_results:
            return {}
        
        # Aggregate key metrics
        overall_results = {
            'num_stages_completed': len(self.stage_results),
            'stage_results': self.stage_results,
            'aggregated_metrics': self._get_aggregated_summary_results(enable_advanced_analysis)
        }
        
        # Add timing information
        total_simulation_time = sum(
            stage_result.get('simulation_time', 0) for stage_result in self.stage_results
        )
        overall_results['total_simulation_time'] = total_simulation_time
        
        # Calculate cumulative trajectory
        overall_results.update(self._calculate_cumulative_trajectory())
        
        # Save data if requested
        if save_data:
            self._save_multistage_data(overall_results)
        
        return overall_results
    
    def _get_aggregated_summary_results(self, enable_advanced_analysis: bool) -> Dict[str, Any]:
        """Get aggregated summary results across all stages."""
        if not self.stage_results:
            return {}
        
        # Extract key metrics from each stage
        stage_velocities = []
        stage_energies = []
        stage_efficiencies = []
        
        for i, stage_result in enumerate(self.stage_results):
            exit_vel = stage_result.get('exit_velocity', stage_result.get('final_velocity', 0))
            stage_velocities.append(exit_vel)
            
            energy_analysis = stage_result.get('energy_analysis', {})
            stage_energies.append(energy_analysis.get('final_kinetic_energy', 0))
            stage_efficiencies.append(energy_analysis.get('efficiency', 0))
        
        # Calculate overall metrics
        final_velocity = stage_velocities[-1] if stage_velocities else 0
        max_velocity = max(stage_velocities) if stage_velocities else 0
        
        # Calculate overall efficiency (final kinetic energy / total initial energy)
        total_initial_energy = sum(
            stage_result.get('energy_analysis', {}).get('initial_energy', 0)
            for stage_result in self.stage_results
        )
        final_kinetic_energy = stage_energies[-1] if stage_energies else 0
        overall_efficiency = final_kinetic_energy / total_initial_energy if total_initial_energy > 0 else 0
        
        # Aggregate force and current maximums
        max_current = max(
            stage_result.get('max_current', 0) for stage_result in self.stage_results
        )
        max_force = max(
            stage_result.get('max_force', 0) for stage_result in self.stage_results
        )
        
        return {
            'final_velocity': final_velocity,
            'max_velocity': max_velocity,
            'max_current': max_current,
            'max_force': max_force,
            'overall_efficiency': overall_efficiency,
            'final_kinetic_energy': final_kinetic_energy,
            'total_initial_energy': total_initial_energy,
            'stage_velocities': stage_velocities,
            'stage_energies': stage_energies,
            'stage_efficiencies': stage_efficiencies,
            'velocity_gain_per_stage': np.diff([0] + stage_velocities).tolist()
        }
    
    def _calculate_cumulative_trajectory(self) -> Dict[str, Any]:
        """Calculate cumulative trajectory across all stages."""
        if not self.stage_results:
            return {}
        
        # Combine time series data from all stages
        cumulative_time = []
        cumulative_position = []
        cumulative_velocity = []
        cumulative_current = []
        cumulative_force = []
        
        time_offset = 0.0
        position_offset = 0.0
        
        for i, stage_result in enumerate(self.stage_results):
            stage_time = stage_result.get('time', np.array([]))
            stage_position = stage_result.get('position', np.array([]))
            stage_velocity = stage_result.get('velocity', np.array([]))
            stage_current = stage_result.get('current', np.array([]))
            stage_force = stage_result.get('force_total', np.array([]))
            
            if len(stage_time) > 0:
                # Add time offset for sequential stages
                adj_time = stage_time + time_offset
                cumulative_time.extend(adj_time)
                
                # Add position offset for stage spacing
                adj_position = stage_position + position_offset
                cumulative_position.extend(adj_position)
                
                # Velocity, current, and force don't need offset
                cumulative_velocity.extend(stage_velocity)
                cumulative_current.extend(stage_current)
                cumulative_force.extend(stage_force)
                
                # Update offsets for next stage
                time_offset = adj_time[-1] + 0.001  # Small gap between stages
                position_offset += self.stage_spacing
        
        return {
            'cumulative_trajectory': {
                'time': np.array(cumulative_time),
                'position': np.array(cumulative_position),
                'velocity': np.array(cumulative_velocity),
                'current': np.array(cumulative_current),
                'force_total': np.array(cumulative_force)
            }
        }
    
    def _save_multistage_data(self, results: Dict[str, Any]):
        """Save multi-stage simulation data."""
        # Create output directory
        output_dir = Path("multistage_simulation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save overall results as JSON
        json_file = output_dir / "multistage_results.json"
        self.results_analyzer.save_results_json(results, str(json_file))
        
        # Save individual stage results
        for i, stage_result in enumerate(self.stage_results):
            stage_json = output_dir / f"stage_{i+1}_results.json"
            self.results_analyzer.save_results_json(stage_result, str(stage_json))
        
        # Save cumulative data as CSV
        if 'cumulative_trajectory' in results:
            csv_file = output_dir / "cumulative_trajectory.csv"
            self.results_analyzer.save_results_csv(results['cumulative_trajectory'], str(csv_file))
        
        print(f"✓ Multi-stage results saved to {output_dir}")
    
    def _print_enhanced_multistage_info(self):
        """Print information about the enhanced multi-stage simulation."""
        print(f"Number of stages: {self.num_stages}")
        print(f"Stage spacing: {self.stage_spacing*1000:.1f}mm")
        print(f"Transfer efficiency: {self.velocity_transfer_efficiency*100:.1f}%")
        print(f"Interstage analysis: {'Enabled' if self.enable_interstage_analysis else 'Disabled'}")
        print(f"Energy tracking: {'Enabled' if self.enable_cumulative_energy_tracking else 'Disabled'}")
        print(f"Enhanced Physics: {'Enabled' if self.enable_enhanced_physics else 'Disabled'}")
        print(f"Advanced Analysis: {'Enabled' if self.enable_advanced_analysis else 'Disabled'}")
    
    def _print_stage_initial_state(self, projectile_state: Dict[str, Any], stage_num: int):
        """Print initial state of the projectile at the start of a stage."""
        print(f"Initial velocity: {projectile_state['velocity']:.2f} m/s")
        print(f"Initial position: {projectile_state['position']:.2f} m")
        print(f"Initial total energy: {projectile_state['total_energy']:.2f} J")
        print(f"Initial cumulative losses: {projectile_state['cumulative_losses']:.2f} J")
    
    def _print_stage_completion_info(self, stage_results: Dict[str, Any], projectile_state: Dict[str, Any], stage_num: int):
        """Print information about the completion of a stage."""
        exit_velocity = stage_results.get('exit_velocity', stage_results.get('final_velocity', 0))
        energy_loss = stage_results.get('energy_loss', 0)
        
        print(f"Stage {stage_num} exit velocity: {exit_velocity:.2f} m/s")
        print(f"Transferred to next stage: {projectile_state['velocity']:.2f} m/s")
        print(f"Energy loss: {energy_loss:.2f} J")
    
    def _print_enhanced_overall_results(self):
        """Print comprehensive multi-stage simulation results."""
        print("\n" + "="*60)
        print("MULTI-STAGE SIMULATION RESULTS")
        print("="*60)
        
        aggregated = self.overall_results.get('aggregated_metrics', {})
        
        # Overall performance
        print(f"Stages completed: {self.overall_results.get('num_stages_completed', 0)}/{self.num_stages}")
        print(f"Final velocity: {aggregated.get('final_velocity', 0):.2f} m/s")
        print(f"Maximum velocity: {aggregated.get('max_velocity', 0):.2f} m/s")
        print(f"Overall efficiency: {aggregated.get('overall_efficiency', 0)*100:.1f}%")
        
        # Stage-by-stage breakdown
        stage_velocities = aggregated.get('stage_velocities', [])
        stage_efficiencies = aggregated.get('stage_efficiencies', [])
        velocity_gains = aggregated.get('velocity_gain_per_stage', [])
        
        print(f"\nStage-by-Stage Results:")
        print(f"{'Stage':<6} {'Exit Vel':<10} {'Efficiency':<12} {'Vel Gain':<10}")
        print(f"{'':=<6} {'':=<10} {'':=<12} {'':=<10}")
        
        for i in range(len(stage_velocities)):
            stage_num = i + 1
            exit_vel = stage_velocities[i]
            efficiency = stage_efficiencies[i] * 100 if i < len(stage_efficiencies) else 0
            vel_gain = velocity_gains[i] if i < len(velocity_gains) else 0
            
            print(f"{stage_num:<6} {exit_vel:<10.2f} {efficiency:<12.1f} {vel_gain:<10.2f}")
        
        # Energy summary
        total_energy = aggregated.get('total_initial_energy', 0)
        final_kinetic = aggregated.get('final_kinetic_energy', 0)
        
        print(f"\nEnergy Summary:")
        print(f"Total initial energy: {total_energy:.1f} J")
        print(f"Final kinetic energy: {final_kinetic:.3f} J")
        print(f"Overall efficiency: {aggregated.get('overall_efficiency', 0)*100:.1f}%")
        
        # Performance metrics
        max_current = aggregated.get('max_current', 0)
        max_force = aggregated.get('max_force', 0)
        
        print(f"\nMaximum Values (across all stages):")
        print(f"Current: {max_current:.1f} A")
        print(f"Force: {max_force:.1f} N")
        
        # Material property cache performance (from any stage)
        if self.stage_simulations and hasattr(self.stage_simulations[0], 'physics'):
            physics_engine = self.stage_simulations[0].physics
            if hasattr(physics_engine, 'permeability_model'):
                perm_model = physics_engine.permeability_model
                if hasattr(perm_model, 'print_cache_statistics'):
                    perm_model.print_cache_statistics()
        
        print("="*60)
    
    def plot_results(self, save_plots: bool = True, output_dir: str = "multistage_simulation_results"):
        """Plot multi-stage simulation results."""
        if not self.overall_results:
            print("❌ No simulation results to plot. Run simulation first.")
            return
        
        # Plot individual stage results
        for i, stage_sim in enumerate(self.stage_simulations):
            stage_output_dir = f"{output_dir}/stage_{i+1}"
            stage_sim.plot_results(save_plots, stage_output_dir)
        
        # Plot cumulative trajectory if available
        if 'cumulative_trajectory' in self.overall_results:
            self._plot_cumulative_trajectory(save_plots, output_dir)
    
    def _plot_cumulative_trajectory(self, save_plots: bool, output_dir: str):
        """Plot cumulative trajectory across all stages."""
        try:
            import matplotlib.pyplot as plt
            
            trajectory = self.overall_results['cumulative_trajectory']
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Multi-Stage Cumulative Trajectory')
            
            time_ms = trajectory['time'] * 1000
            
            # Velocity vs time
            axes[0, 0].plot(time_ms, trajectory['velocity'])
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Velocity (m/s)')
            axes[0, 0].set_title('Cumulative Velocity')
            axes[0, 0].grid(True)
            
            # Position vs time
            axes[0, 1].plot(time_ms, trajectory['position'] * 1000)
            axes[0, 1].set_xlabel('Time (ms)')
            axes[0, 1].set_ylabel('Position (mm)')
            axes[0, 1].set_title('Cumulative Position')
            axes[0, 1].grid(True)
            
            # Current vs time
            axes[1, 0].plot(time_ms, trajectory['current'])
            axes[1, 0].set_xlabel('Time (ms)')
            axes[1, 0].set_ylabel('Current (A)')
            axes[1, 0].set_title('Cumulative Current')
            axes[1, 0].grid(True)
            
            # Force vs position
            axes[1, 1].plot(trajectory['position'] * 1000, trajectory['force_total'])
            axes[1, 1].set_xlabel('Position (mm)')
            axes[1, 1].set_ylabel('Force (N)')
            axes[1, 1].set_title('Cumulative Force')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(output_path / 'cumulative_trajectory.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Warning: matplotlib not available, plotting disabled")
    
    def get_summary_results(self) -> Dict[str, Any]:
        """Get summary of multi-stage simulation results."""
        if not self.overall_results:
            return {}
        
        return self.overall_results.get('aggregated_metrics', {})
    
    def cleanup_temp_files(self):
        """Clean up temporary stage configuration files."""
        for stage_num in range(1, self.num_stages + 1):
            temp_file = f"temp_stage_{stage_num}_config.json"
            try:
                Path(temp_file).unlink(missing_ok=True)
            except Exception:
                pass

    def _dump_interrupted_multistage_results(self):
        """Dump current multi-stage simulation state when interrupted."""
        try:
            # Create interrupted results summary
            interrupted_results = {
                'interruption_info': {
                    'timestamp': datetime.now().isoformat(),
                    'interruption_type': 'user_ctrl_c',
                    'simulation_type': 'enhanced_multi_stage'
                },
                'multi_stage_progress': {
                    'total_stages': self.num_stages,
                    'completed_stages': len(self.stage_results),
                    'velocity_transfer_efficiency': self.velocity_transfer_efficiency,
                    'enable_interstage_analysis': self.enable_interstage_analysis
                },
                'stage_results': [],
                'system_configuration': {
                    'num_stages': self.num_stages,
                    'enhanced_physics_enabled': self.enable_enhanced_physics,
                    'advanced_analysis_enabled': self.enable_advanced_analysis
                }
            }
            
            # Add completed stage results
            for i, stage_result in enumerate(self.stage_results):
                stage_summary = {
                    'stage_number': i + 1,
                    'final_velocity': stage_result.get('final_velocity', 0),
                    'max_velocity': stage_result.get('max_velocity', 0),
                    'max_current': stage_result.get('max_current', 0),
                    'max_force': stage_result.get('max_force', 0),
                    'efficiency': stage_result.get('energy_analysis', {}).get('efficiency', 0),
                    'simulation_time': stage_result.get('simulation_time', 0)
                }
                interrupted_results['stage_results'].append(stage_summary)
            
            # Add overall metrics if any stages were completed
            if self.stage_results:
                final_velocities = [stage.get('final_velocity', 0) for stage in self.stage_results]
                interrupted_results['partial_overall_metrics'] = {
                    'stages_completed': len(self.stage_results),
                    'last_stage_velocity': final_velocities[-1] if final_velocities else 0,
                    'max_velocity_achieved': max(stage.get('max_velocity', 0) for stage in self.stage_results),
                    'cumulative_velocity_gain': sum(final_velocities),
                    'average_stage_efficiency': np.mean([
                        stage.get('energy_analysis', {}).get('efficiency', 0) 
                        for stage in self.stage_results
                    ]) if self.stage_results else 0
                }
            
            # Create output directory if it doesn't exist
            output_dir = "simulation_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interrupted_multistage_simulation_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save interrupted results
            with open(filepath, 'w') as f:
                json.dump(interrupted_results, f, indent=2)
            
            print(f"💾 Multi-stage simulation state dumped to: {filepath}")
            print(f"   - Completed stages: {len(self.stage_results)}/{self.num_stages}")
            if self.stage_results:
                last_velocity = interrupted_results['partial_overall_metrics']['last_stage_velocity']
                max_velocity = interrupted_results['partial_overall_metrics']['max_velocity_achieved']
                print(f"   - Last stage exit velocity: {last_velocity:.3f} m/s")
                print(f"   - Maximum velocity achieved: {max_velocity:.3f} m/s")
            
        except Exception as e:
            print(f"⚠  Failed to dump multi-stage simulation results: {e}")
            print("   Simulation state could not be saved.")


# Backward compatibility aliases
MultiStageSimulation = EnhancedMultiStageSimulation
MultiStageCoilgunSimulation = EnhancedMultiStageSimulation 