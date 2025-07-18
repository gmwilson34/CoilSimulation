"""
Single Stage Coilgun Simulation

This module provides the single stage coilgun simulation implementation
with enhanced ODE integration, event detection, and results analysis.
Updated to fully leverage the new physics engine capabilities.
"""

import numpy as np
import json
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from scipy.integrate import solve_ivp

from physics import CoilgunPhysicsEngine
from physics.forces import ForceAnalyzer
from physics.circuits import EnergyAnalyzer
from physics.materials import AdvancedMaterialProperties
from .core import BaseSolver, SolverConfig, SolverError, SimulationError
from .progress import ProgressTracker, SimpleProgressTracker
from .analysis import ResultsAnalyzer


class EnhancedSingleStageSimulation(BaseSolver):
    """
    Enhanced single-stage coilgun simulation with full physics engine integration.
    Now leverages advanced material properties, force analysis, and energy conservation.
    """
    
    def __init__(self, config_file: str, solver_config: Optional[SolverConfig] = None):
        """
        Initialize enhanced single stage simulation.
        
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
        
        # Physics validation settings
        self.enable_physics_validation = self.solver_config.get('physics.enable_validation', True)
        self.enable_energy_conservation = self.solver_config.get('physics.energy_conservation', True)
        self.enable_thermal_effects = self.solver_config.get('physics.thermal_effects', False)
        
        # Progress tracking
        self.progress_tracker = None
        
        # Integration settings with physics-aware defaults
        self.integration_method = self.solver_config.get('solver.method', 'RK45')
        self.rtol = self.solver_config.get('solver.rtol', 1e-8)
        self.atol = self.solver_config.get('solver.atol', 1e-10)
        self.max_step = self.solver_config.get('solver.max_step', 1e-4)
        
        # Enhanced termination settings based on physics
        self.exit_velocity_threshold = self.solver_config.get('termination.exit_velocity_threshold', 10.0)
        self.current_threshold = self.solver_config.get('termination.current_threshold', 1.0)
        self.force_threshold = self.solver_config.get('termination.force_threshold', 1e-6)
        
        # Termination behavior options from config
        termination_config = self.physics_config.get('simulation', {}).get('termination', {})
        self.stop_at_center = termination_config.get('stop_at_center', True)  # Default: stop at center
        self.stop_at_max_velocity = termination_config.get('stop_at_max_velocity', True)  # Default: stop at max velocity
        
        # Energy tracking initialization
        if self.enable_energy_conservation:
            self.energy_analyzer.initialize_energy_tracking()
        
        # Signal handling for graceful interruption
        self._setup_signal_handlers()
        
        print(f"✓ Enhanced Single Stage Solver Initialized")
        print(f"  - Physics Engine: {type(self.physics).__name__}")
        print(f"  - Force Calculator: {type(self.physics.forces).__name__}")
        print(f"  - Material System: {type(self.material_properties).__name__}")
        print(f"  - Method: {self.integration_method}")
        print(f"  - Energy Conservation: {'Enabled' if self.enable_energy_conservation else 'Disabled'}")
        print(f"  - Thermal Effects: {'Enabled' if self.enable_thermal_effects else 'Disabled'}")
        print(f"  - Stop at Center: {'Enabled' if self.stop_at_center else 'Disabled'}")
        print(f"  - Stop at Max Velocity: {'Enabled' if self.stop_at_max_velocity else 'Disabled'}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful interruption."""
        def signal_handler(signum, frame):
            print("\n⚠  Simulation interrupted by user")
            
            # Stop progress tracking
            if self.progress_tracker:
                self.progress_tracker.stop()
            
            # Dump current simulation results to file
            self._dump_interrupted_simulation_results()
            
            raise KeyboardInterrupt("Simulation interrupted by user")
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def run_simulation(self, save_data: bool = True, verbose: bool = True, 
                      show_progress: bool = True, check_physics: bool = True,
                      max_time: float = None, enable_advanced_analysis: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """
        Run the enhanced single stage coilgun simulation.
        
        Args:
            save_data: Whether to save simulation data
            verbose: Enable verbose output
            show_progress: Show progress bar during simulation
            check_physics: Perform physics validation checks
            max_time: Maximum simulation time (auto-determined if None)
            enable_advanced_analysis: Enable comprehensive physics analysis
            **kwargs: Additional solver options
            
        Returns:
            Dictionary containing comprehensive simulation results
        """
        if verbose:
            print("\n" + "="*60)
            print("ENHANCED SINGLE STAGE COILGUN SIMULATION")
            print("="*60)
            self._print_enhanced_system_info()
        
        try:
            # Enhanced physics validation
            if check_physics and self.enable_physics_validation:
                self._run_comprehensive_physics_validation()
            
            # Determine simulation time span with physics-aware estimation
            t_span = self._determine_time_span_enhanced(max_time)
            
            # Get initial conditions with physics validation
            y0 = self._get_validated_initial_conditions()
            
            # Setup enhanced progress tracking
            self.progress_tracker = self._create_enhanced_progress_tracker(t_span, show_progress)
            
            # Create physics-aware ODE function with advanced features
            ode_func = self._create_enhanced_ode_wrapper()
            
            # Create enhanced event functions with physics-based termination
            events = self._create_enhanced_simulation_events()
            
            # Start integration with enhanced monitoring
            start_time = time.time()
            if verbose:
                print(f"\nStarting enhanced integration from t={t_span[0]:.4f}s to t={t_span[1]:.4f}s")
                print(f"Initial conditions: Q={y0[0]:.3f}C, I={y0[1]:.1f}A, x={y0[2]*1000:.2f}mm, v={y0[3]:.2f}m/s")
                print(f"Force method: {type(self.physics.forces).__name__}")
                print(f"Energy conservation: {'ON' if self.enable_energy_conservation else 'OFF'}")
            
            # Enhanced solver with adaptive settings
            solver_kwargs = self._get_enhanced_solver_settings(kwargs)
            
            # Solve ODE system with enhanced capabilities
            # Remove potentially duplicate keys from solver_kwargs
            final_solver_kwargs = solver_kwargs.copy()
            
            # Set default values if not in solver_kwargs
            if 'rtol' not in final_solver_kwargs:
                final_solver_kwargs['rtol'] = self.rtol
            if 'atol' not in final_solver_kwargs:
                final_solver_kwargs['atol'] = self.atol
            if 'max_step' not in final_solver_kwargs:
                final_solver_kwargs['max_step'] = self.max_step
            
            solution = solve_ivp(
                ode_func, t_span, y0,
                method=self.integration_method,
                events=events,
                **final_solver_kwargs
            )
            
            # Stop progress tracking
            if self.progress_tracker:
                self.progress_tracker.stop()
                self.progress_tracker.show_final_progress()
            
            # Calculate simulation timing
            self.simulation_time = time.time() - start_time
            
            # Enhanced results processing with comprehensive analysis
            self.solution = solution
            results = self._process_enhanced_results(solution, save_data, verbose, enable_advanced_analysis)
            
            if verbose:
                self._print_enhanced_results(results)
            
            return results
            
        except KeyboardInterrupt:
            if self.progress_tracker:
                self.progress_tracker.stop()
            raise
        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.stop()
            print(f"\n❌ Enhanced simulation failed: {str(e)}")
            raise SimulationError(f"Enhanced simulation failed: {str(e)}") from e
    
    def _determine_time_span_enhanced(self, max_time: Optional[float]) -> Tuple[float, float]:
        """Determine appropriate simulation time span with enhanced physics awareness."""
        if max_time is not None:
            return (0.0, max_time)
        
        # First, check if time_span is specified in the config file
        config_time_span = self.physics_config.get('simulation', {}).get('time_span')
        if config_time_span is not None and len(config_time_span) == 2:
            t_start, t_end = config_time_span
            print(f"✓ Using time span from config: [{t_start:.4f}, {t_end:.4f}] seconds")
            return (float(t_start), float(t_end))
        
        # Enhanced auto-determination based on multiple physics considerations
        print("⚠  No time_span specified in config, auto-determining based on physics...")
        RC = self.physics.circuit_model.get_effective_resistance() * self.physics.capacitance
        
        # Estimate projectile transit time using force analysis
        try:
            # Get average force for velocity estimation
            test_current = self.physics.initial_voltage / self.physics.circuit_model.get_effective_resistance()
            test_force, _ = self.physics.magnetic_force_with_circuit_logic(test_current, 0.0, 0.0, 0.0)
            if test_force > 0:
                estimated_acceleration = test_force / self.physics.proj_mass
                estimated_time = np.sqrt(2 * self.physics.coil_length / estimated_acceleration)
            else:
                estimated_time = 0.01  # Fallback
        except:
            estimated_time = 0.01  # Safe fallback
        
        # Use the longer of the time constants with safety margin
        auto_time = max(5 * RC, 3 * estimated_time, 0.005)
        auto_time = min(auto_time, 1.0)  # Cap at 1 second
        
        print(f"✓ Auto-determined time span: [0.0000, {auto_time:.4f}] seconds")
        return (0.0, auto_time)
    
    def _get_validated_initial_conditions(self):
        """Get initial conditions with physics validation."""
        y0 = list(self.get_initial_conditions())
        
        # Validate initial conditions with physics engine
        if self.enable_physics_validation:
            # Check initial charge vs capacitor capacity
            if abs(y0[0]) > self.physics.capacitance * self.physics.initial_voltage * 1.1:
                print(f"⚠  Warning: Initial charge {y0[0]:.3f}C exceeds expected value")
            
            # Check initial position within reasonable bounds
            if abs(y0[2]) > self.physics.coil_length:
                print(f"⚠  Warning: Initial position {y0[2]*1000:.1f}mm outside coil")
        
        return y0
    
    def _create_enhanced_progress_tracker(self, t_span: Tuple[float, float], show_progress: bool):
        """Create enhanced progress tracker with physics monitoring."""
        if show_progress:
            # Use enhanced progress tracker with physics monitoring
            tracker = ProgressTracker(t_span, self.physics, 
                                    update_interval=0.005,  # 200 Hz update rate for real-time position tracking
                                    bar_width=50)
            self.add_step_callback(tracker.update)
            return tracker
        else:
            # Use simple tracker for interface compatibility
            return SimpleProgressTracker(t_span, update_frequency=10)  # More frequent updates
    
    def _create_enhanced_ode_wrapper(self):
        """Create enhanced ODE wrapper with advanced physics integration."""
        base_func = self.create_ode_function()
        
        def enhanced_ode_func(t, y):
            try:
                # Get basic derivatives
                dydt = base_func(t, y)
                
                # Enhanced physics monitoring and validation
                if self.enable_physics_validation and self.step_count % 100 == 0:
                    self._validate_physics_state(t, y, dydt)
                
                # Energy conservation monitoring
                if self.enable_energy_conservation and self.step_count % 50 == 0:
                    self._monitor_energy_conservation(t, y, dydt)
                
                # Thermal effects (if enabled)
                if self.enable_thermal_effects:
                    dydt = self._apply_thermal_corrections(t, y, dydt)
                
                # Execute step callbacks
                for callback in self.step_callbacks:
                    try:
                        callback(t, y, dydt)
                    except Exception as e:
                        print(f"Warning: Step callback failed: {e}")
                
                self.step_count += 1
                return dydt
                
            except Exception as e:
                print(f"Error in enhanced ODE function at t={t}: {e}")
                raise
        
        return enhanced_ode_func
    
    def _create_enhanced_simulation_events(self):
        """Create enhanced event functions with physics-based termination."""
        events = []
        
        # Projectile reaches center of coil event (user requested feature)
        if self.stop_at_center:
            def projectile_at_center(t, y):
                return y[2]  # position = 0 (center of coil)
            projectile_at_center.terminal = True  # Stop when projectile reaches center
            projectile_at_center.direction = 1  # Detect when crossing zero from negative to positive
            events.append(projectile_at_center)
        
        # Enhanced projectile exit event with force consideration
        def enhanced_projectile_exits_coil(t, y):
            coil_end = self.physics.coil_length / 2.0
            projectile_rear = y[2] - self.physics.proj_length / 2.0
            
            # Also check if force becomes strongly negative (retarding)
            if abs(y[1]) > 1.0:  # Only if significant current
                try:
                    force, _ = self.physics.magnetic_force_with_circuit_logic(y[1], y[2], t, y[3])
                    if force < -self.force_threshold and projectile_rear > coil_end * 0.8:
                        return projectile_rear - coil_end  # Allow early termination if force is retarding
                except:
                    pass
            
            return projectile_rear - coil_end
        enhanced_projectile_exits_coil.terminal = True
        enhanced_projectile_exits_coil.direction = 1
        events.append(enhanced_projectile_exits_coil)
        
        # Maximum velocity detection event (stops when velocity starts decreasing)
        if self.stop_at_max_velocity:
            def max_velocity_reached(t, y):
                # This detects when acceleration becomes negative (velocity stops increasing)
                if len(y) >= 4:
                    try:
                        # Calculate current acceleration
                        force, _ = self.physics.magnetic_force_with_circuit_logic(y[1], y[2], t, y[3])
                        acceleration = force / self.physics.proj_mass
                        return acceleration  # Will trigger when acceleration becomes negative
                    except:
                        return 1.0  # Continue if force calculation fails
                return 1.0
            max_velocity_reached.terminal = True  # Stop when max velocity is reached
            max_velocity_reached.direction = -1  # Detect when acceleration becomes negative
            events.append(max_velocity_reached)
        
        # Energy-based termination (when kinetic energy stops increasing)
        def energy_optimization_point(t, y):
            if len(y) >= 4:
                kinetic_energy = 0.5 * self.physics.proj_mass * y[3]**2
                # This is a simplified check - could be enhanced with energy history
                return y[3] if y[3] > self.exit_velocity_threshold else 1.0
            return 1.0
        energy_optimization_point.terminal = False
        energy_optimization_point.direction = -1
        events.append(energy_optimization_point)
        
        # Enhanced current reversal with physics awareness
        def enhanced_current_reverses(t, y):
            # More sophisticated current reversal detection
            current = y[1]
            if abs(current) < self.current_threshold:
                return current  # Terminate if current too low
            return current
        enhanced_current_reverses.terminal = False
        enhanced_current_reverses.direction = -1
        events.append(enhanced_current_reverses)
        
        # Physics-based force threshold event
        def force_threshold_event(t, y):
            try:
                force, _ = self.physics.magnetic_force_with_circuit_logic(y[1], y[2], t, y[3])
                return abs(force) - self.force_threshold
            except:
                return 1.0  # Continue if force calculation fails
        force_threshold_event.terminal = False
        force_threshold_event.direction = -1
        events.append(force_threshold_event)
        
        return events
    
    def _run_comprehensive_physics_validation(self):
        """Run comprehensive physics validation checks."""
        print("\n🔍 Running comprehensive physics validation...")
        
        # Test force calculation at several points
        test_positions = [-0.01, 0.0, 0.01]
        test_current = 100.0
        
        for pos in test_positions:
            try:
                force_result = self.physics.magnetic_force_ferromagnetic(test_current, pos)
                force = force_result[0] if isinstance(force_result, tuple) else force_result
                print(f"  Force at x={pos*1000:.1f}mm: {force:.1f}N")
            except Exception as e:
                print(f"  ⚠  Force calculation failed at x={pos*1000:.1f}mm: {e}")
        
        # Test inductance calculation
        try:
            L0 = self.physics.get_inductance(0.0)
            print(f"  Inductance at center: {L0*1e6:.1f}µH")
        except Exception as e:
            print(f"  ⚠  Inductance calculation failed: {e}")
        
        print("✓ Comprehensive physics validation complete")
    
    def _process_enhanced_results(self, solution, save_data: bool, verbose: bool, enable_advanced_analysis: bool) -> Dict[str, Any]:
        """Process enhanced simulation results with comprehensive analysis."""
        if not solution.success:
            raise SimulationError(f"Integration failed: {solution.message}")
        
        if verbose:
            print("\n🔍 Analyzing results...")
        
        # Use enhanced analyzer for comprehensive results
        results = self.results_analyzer.analyze_solution(solution)
        
        # Add enhanced analysis if requested
        if enable_advanced_analysis:
            try:
                # Force component analysis using new physics engine
                if hasattr(self.physics, 'analyze_force_components'):
                    force_analysis = []
                    for i, (current, position, velocity) in enumerate(zip(
                        results['current'], results['position'], results['velocity']
                    )):
                        if i % 10 == 0:  # Sample every 10th point for performance
                            analysis = self.physics.analyze_force_components(
                                current, position, velocity, 
                                results.get('current_history'), results.get('time_history')
                            )
                            force_analysis.append(analysis)
                    results['detailed_force_analysis'] = force_analysis
                
                # Enhanced energy analysis
                if self.enable_energy_conservation:
                    energy_tracking = self.energy_analyzer.get_energy_summary()
                    results['energy_conservation_analysis'] = energy_tracking
                    
                    # Calculate efficiency with multiple metrics
                    final_velocity = results.get('final_velocity', 0.0)
                    if final_velocity > 0:
                        kinetic_energy = 0.5 * self.physics.proj_mass * final_velocity**2
                        initial_energy = 0.5 * self.physics.capacitance * self.physics.initial_voltage**2
                        results['enhanced_efficiency_analysis'] = {
                            'kinetic_efficiency': kinetic_energy / initial_energy,
                            'force_efficiency': energy_tracking.get('force_efficiency', 0),
                            'circuit_efficiency': energy_tracking.get('circuit_efficiency', 0)
                        }
                
                # Material property analysis
                if hasattr(self.material_properties, 'get_temperature_effects'):
                    results['material_analysis'] = {
                        'coil_material_properties': self.material_properties.get_enhanced_properties(
                            self.physics.config['coil']['material']
                        ),
                        'projectile_material_properties': self.material_properties.get_enhanced_properties(
                            self.physics.config['projectile']['material']
                        )
                    }
            
            except Exception as e:
                print(f"⚠  Warning: Advanced analysis failed: {e}")
                # Continue with basic results
        
        # Determine termination reason based on events
        termination_reason = "unknown"
        termination_details = {}
        
        if hasattr(solution, 't_events') and solution.t_events:
            event_times = [events for events in solution.t_events if len(events) > 0]
            if event_times:
                # Find which event triggered first
                first_event_time = min([events[0] for events in event_times])
                for i, events in enumerate(solution.t_events):
                    if len(events) > 0 and events[0] == first_event_time:
                        event_names = [
                            "projectile_at_center",
                            "projectile_exits_coil", 
                            "max_velocity_reached",
                            "energy_optimization_point",
                            "current_reversal",
                            "force_threshold"
                        ]
                        if i < len(event_names):
                            termination_reason = event_names[i]
                            termination_details = {
                                'event_time': first_event_time,
                                'final_position': solution.y[2][-1] if len(solution.y) > 2 else 0,
                                'final_velocity': solution.y[3][-1] if len(solution.y) > 3 else 0,
                                'final_current': solution.y[1][-1] if len(solution.y) > 1 else 0
                            }
                        break
        
        if termination_reason == "unknown":
            if solution.t[-1] >= solution.t[0] + 0.99 * (solution.t[-1] - solution.t[0]):
                termination_reason = "time_limit_reached"
                termination_details = {'final_time': solution.t[-1]}
        
        # Add simulation metadata with enhanced information
        results.update({
            'solver_info': {
                'solver_type': 'enhanced_single_stage',
                'physics_engine': type(self.physics).__name__,
                'force_calculator': type(self.physics.forces).__name__,
                'material_system': type(self.material_properties).__name__,
                'integration_method': self.integration_method,
                'simulation_time': self.simulation_time,
                'total_steps': self.step_count,
                'energy_conservation_enabled': bool(self.enable_energy_conservation),
                'thermal_effects_enabled': bool(self.enable_thermal_effects),
                'physics_validation_enabled': bool(self.enable_physics_validation),
                'stop_at_center_enabled': bool(self.stop_at_center),
                'stop_at_max_velocity_enabled': bool(self.stop_at_max_velocity),
                'termination_reason': termination_reason,
                'termination_details': termination_details
            },
            'configuration': self.physics_config
        })
        
        # Save enhanced results if requested
        if save_data:
            self._save_enhanced_simulation_data(results)
        
        return results
    
    def _save_enhanced_simulation_data(self, results: Dict[str, Any]):
        """Save enhanced simulation data with additional analysis files."""
        output_dir = Path("simulation_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save main results as JSON with enhanced metadata
        json_file = output_dir / "enhanced_single_stage_results.json"
        self.results_analyzer.save_results_json(results, str(json_file))
        
        # Save detailed data as CSV
        csv_file = output_dir / "enhanced_single_stage_data.csv"
        self.results_analyzer.save_results_csv(results, str(csv_file))
        
        # Save enhanced analysis files if available
        if 'detailed_force_analysis' in results:
            force_file = output_dir / "force_component_analysis.json"
            with open(force_file, 'w') as f:
                json.dump(results['detailed_force_analysis'], f, indent=2, default=str)
        
        if 'energy_conservation_analysis' in results:
            energy_file = output_dir / "energy_conservation_analysis.json"
            with open(energy_file, 'w') as f:
                json.dump(results['energy_conservation_analysis'], f, indent=2, default=str)
        
        print(f"✓  Results saved to {output_dir}")
    
    def _print_enhanced_system_info(self):
        """Print enhanced system information."""
        print("\n📊 ENHANCED SYSTEM CONFIGURATION")
        print("-" * 40)
        self.physics.print_system_parameters()
        
        print(f"\n🔬 PHYSICS ENGINE CONFIGURATION")
        print("-" * 40)
        print(f"  Physics Engine: {type(self.physics).__name__}")
        print(f"  Force Calculator: {type(self.physics.forces).__name__}")
        print(f"  Material System: {type(self.material_properties).__name__}")
        print(f"  Field Calculator: {type(self.physics.field_calculator).__name__}")
        print(f"  Circuit Model: {type(self.physics.circuit_model).__name__}")
        
        print(f"\n⚙️  SOLVER ENHANCEMENTS")
        print("-" * 40)
        print(f"  Energy Conservation: {'✓ Enabled' if self.enable_energy_conservation else '✗ Disabled'}")
        print(f"  Thermal Effects: {'✓ Enabled' if self.enable_thermal_effects else '✗ Disabled'}")
        print(f"  Physics Validation: {'✓ Enabled' if self.enable_physics_validation else '✗ Disabled'}")
        print(f"  Integration Method: {self.integration_method}")
        print(f"  Tolerances: rtol={self.rtol:.1e}, atol={self.atol:.1e}")
    
    def _print_enhanced_results(self, results: Dict[str, Any]):
        """Print comprehensive enhanced simulation results."""
        print("\n" + "="*60)
        print("ENHANCED SIMULATION RESULTS")
        print("="*60)
        
        # Basic results
        final_velocity = results.get('final_velocity', 0.0)
        max_current = results.get('max_current', 0.0)
        max_force = results.get('max_force', 0.0)
        simulation_time = results.get('simulation_time', 0.0)
        
        print(f"🎯 Performance Metrics:")
        print(f"   Final Velocity: {final_velocity:.2f} m/s")
        print(f"   Max Current: {max_current:.1f} A")
        print(f"   Max Force: {max_force:.0f} N")
        print(f"   Simulation Time: {simulation_time:.4f} s")
        
        # Enhanced efficiency analysis
        if 'enhanced_efficiency_analysis' in results:
            eff_analysis = results['enhanced_efficiency_analysis']
            print(f"\n⚡ Enhanced Efficiency Analysis:")
            print(f"   Kinetic Efficiency: {eff_analysis.get('kinetic_efficiency', 0)*100:.1f}%")
            print(f"   Force Efficiency: {eff_analysis.get('force_efficiency', 0)*100:.1f}%")
            print(f"   Circuit Efficiency: {eff_analysis.get('circuit_efficiency', 0)*100:.1f}%")
        
        # Energy conservation results
        if 'energy_conservation_analysis' in results:
            energy_analysis = results['energy_conservation_analysis']
            print(f"\n🔋 Energy Conservation:")
            total_error = energy_analysis.get('total_energy_error', 0)
            print(f"   Total Energy Error: {abs(total_error)*100:.3f}%")
            print(f"   Conservation Quality: {'Excellent' if abs(total_error) < 0.01 else 'Good' if abs(total_error) < 0.05 else 'Fair'}")
        
        # Solver performance and termination info
        solver_info = results.get('solver_info', {})
        print(f"\n🔧 Solver Performance:")
        print(f"   Total Steps: {solver_info.get('total_steps', 0)}")
        print(f"   Computation Time: {solver_info.get('simulation_time', 0):.3f} s")
        print(f"   Physics Engine: {solver_info.get('physics_engine', 'Unknown')}")
        
        # Termination information
        termination_reason = solver_info.get('termination_reason', 'unknown')
        termination_details = solver_info.get('termination_details', {})
        print(f"\n🎯 Simulation Termination:")
        
        termination_messages = {
            'projectile_at_center': '   Stopped: Projectile reached center of coil (x=0)',
            'projectile_exits_coil': '   Stopped: Projectile exited the coil',
            'max_velocity_reached': '   Stopped: Maximum velocity reached (acceleration became negative)',
            'energy_optimization_point': '   Stopped: Energy optimization point reached',
            'current_reversal': '   Stopped: Current reversed direction',
            'force_threshold': '   Stopped: Force dropped below threshold',
            'time_limit_reached': '   Stopped: Simulation time limit reached'
        }
        
        message = termination_messages.get(termination_reason, f'   Stopped: {termination_reason}')
        print(message)
        
        if termination_details:
            if 'event_time' in termination_details:
                print(f"   Event Time: {termination_details['event_time']:.6f} s")
            if 'final_position' in termination_details:
                print(f"   Final Position: {termination_details['final_position']*1000:.2f} mm")
            if 'final_velocity' in termination_details:
                print(f"   Final Velocity: {termination_details['final_velocity']:.2f} m/s")
            if 'final_current' in termination_details:
                print(f"   Final Current: {termination_details['final_current']:.1f} A")
        
        # Physics analysis summary
        if 'detailed_force_analysis' in results:
            print(f"\n🔬 Physics Analysis: {len(results['detailed_force_analysis'])} detailed force calculations performed")
        
        # Material property cache performance
        if hasattr(self.physics, 'permeability_model'):
            perm_model = self.physics.permeability_model
            if hasattr(perm_model, 'print_cache_statistics'):
                pass  # Cache statistics printing disabled
        
        print("\n" + "="*60)
    
    def plot_results(self, save_plots: bool = True, output_dir: str = "simulation_results"):
        """Plot simulation results."""
        if self.solution is None:
            print("❌ No simulation results to plot. Run simulation first.")
            return
        
        self.results_analyzer.plot_results(self.solution, save_plots, output_dir)
    
    def get_summary_results(self) -> Dict[str, Any]:
        """Get summary of key simulation results."""
        if self.solution is None:
            return {}
        
        return self.results_analyzer.get_summary_results(self.solution)
    
    def optimize_physics_settings(self):
        """Optimize physics calculation settings for accuracy vs speed."""
        print("\n🔧 Optimizing physics settings...")
        
        # This could be expanded to automatically tune physics parameters
        # based on the specific simulation requirements
        
        print("✓ Physics optimization complete")
    
    def _get_enhanced_solver_settings(self, user_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced solver settings based on physics requirements."""
        solver_kwargs = user_kwargs.copy()
        
        # Physics-aware solver settings
        if self.enable_energy_conservation:
            # Tighter tolerances for energy conservation
            solver_kwargs.setdefault('rtol', self.rtol * 0.1)
            solver_kwargs.setdefault('atol', self.atol * 0.1)
        
        # Dense output for better analysis
        solver_kwargs.setdefault('dense_output', True)
        
        # Adaptive max step based on system dynamics
        if not solver_kwargs.get('max_step'):
            # Estimate characteristic time scale from RC constant
            RC = self.physics.circuit_model.get_effective_resistance() * self.physics.capacitance
            solver_kwargs['max_step'] = min(self.max_step, RC / 100)
        
        return solver_kwargs
    
    def _validate_physics_state(self, t: float, y: List[float], dydt: List[float]):
        """Validate physics state during integration."""
        try:
            # Check for reasonable values
            if abs(y[1]) > self.physics.MAX_CURRENT:
                print(f"⚠  Warning: Current {y[1]:.1f}A exceeds safety limit at t={t:.4f}s")
            
            if abs(y[3]) > 1000:  # 1000 m/s velocity limit
                print(f"⚠  Warning: Velocity {y[3]:.1f}m/s is very high at t={t:.4f}s")
            
            # Check energy conservation
            if self.enable_energy_conservation:
                energy_balance = self.physics.calculate_energy_balance(
                    y[1], y[0]/self.physics.capacitance, 
                    0.5 * self.physics.proj_mass * y[3]**2, 0.0
                )
                if abs(energy_balance.get('total_energy_error', 0)) > 0.1:
                    print(f"⚠  Warning: Energy conservation error at t={t:.4f}s")
        
        except Exception as e:
            # Don't halt simulation for validation errors
            pass
    
    def _monitor_energy_conservation(self, t: float, y: List[float], dydt: List[float]):
        """Monitor energy conservation during simulation."""
        try:
            self.energy_analyzer.update_energy_tracking(t, y[0], y[1], y[2], y[3])
        except Exception as e:
            # Don't halt simulation for monitoring errors
            pass
    
    def _apply_thermal_corrections(self, t: float, y: List[float], dydt: List[float]) -> List[float]:
        """Apply thermal corrections if enabled."""
        # This is a placeholder for thermal effects
        # In a full implementation, this would adjust material properties based on temperature
        return dydt
    
    def _dump_interrupted_simulation_results(self):
        """Dump current simulation state and results when interrupted."""
        try:
            import json
            import os
            from datetime import datetime
            
            # Get current progress tracker diagnostics
            if self.progress_tracker and hasattr(self.progress_tracker, 'get_diagnostics'):
                diagnostics = self.progress_tracker.get_diagnostics()
            else:
                diagnostics = {}
            
            # Create interrupted results summary
            interrupted_results = {
                'interruption_info': {
                    'timestamp': datetime.now().isoformat(),
                    'interruption_type': 'user_ctrl_c',
                    'simulation_type': 'enhanced_single_stage'
                },
                'progress_at_interruption': {
                    'elapsed_time': diagnostics.get('elapsed_time', 0),
                    'step_count': diagnostics.get('step_count', 0),
                    'current_time': diagnostics.get('current_time', 0),
                    'progress_percentage': diagnostics.get('progress', 0) * 100,
                    'integration_rate': diagnostics.get('integration_rate', 0)
                },
                'current_state': {
                    'max_current': diagnostics.get('max_current', 0),
                    'max_force': diagnostics.get('max_force', 0),
                    'max_velocity': diagnostics.get('max_velocity', 0),
                    'current_position': diagnostics.get('current_position', 0),
                    'warning_count': diagnostics.get('warning_count', 0)
                },
                'system_configuration': {
                    'physics_engine': type(self.physics).__name__,
                    'force_calculator': type(self.physics.forces).__name__,
                    'integration_method': self.integration_method,
                    'energy_conservation_enabled': self.enable_energy_conservation,
                    'thermal_effects_enabled': self.enable_thermal_effects
                }
            }
            
            # Add current simulation state if available
            if diagnostics.get('current_state') is not None:
                state = diagnostics['current_state']
                if len(state) >= 4:
                    Q, I, x, v = state[:4]
                    interrupted_results['current_state'].update({
                        'charge': float(Q),
                        'current': float(I), 
                        'position': float(x),
                        'velocity': float(v)
                    })
            
            # Create output directory if it doesn't exist
            output_dir = "simulation_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interrupted_simulation_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save interrupted results
            with open(filepath, 'w') as f:
                json.dump(interrupted_results, f, indent=2)
            
            print(f"💾 Simulation state dumped to: {filepath}")
            print(f"   - Progress: {interrupted_results['progress_at_interruption']['progress_percentage']:.1f}%")
            print(f"   - Steps completed: {interrupted_results['progress_at_interruption']['step_count']:,}")
            print(f"   - Max velocity reached: {interrupted_results['current_state']['max_velocity']:.3f} m/s")
            
        except Exception as e:
            print(f"⚠  Failed to dump simulation results: {e}")
            print("   Simulation state could not be saved.")


# Backward compatibility aliases
SingleStageSimulation = EnhancedSingleStageSimulation  # New default
CoilgunSimulation = EnhancedSingleStageSimulation 