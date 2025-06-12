"""
Advanced 3D Coilgun Animation and Visualization

This module creates 3D animations of coilgun simulations showing:
- Detailed coil geometry with helical copper windings (similar to view.py)
- Vertically stacked multi-stage coils with spacing
- Real-time current intensity and timing
- Correctly dimensioned projectile with proper 3D rendering and material properties
- Magnetic field visualization
- Real-time vs simulation time tracking
- Coil array dimensions and specifications
- Projectile altitude tracking
- Professional-quality animations for analysis and presentation

Usage:
    # Interactive mode - select from available results
    python animate.py
    
    # Use specific results directory
    python animate.py results_test_config
    
    # Save animation as GIF
    python animate.py --save coilgun_animation.gif
    
    # Custom duration and frame rate
    python animate.py --duration 15 --fps 24 --save high_quality.mp4

Requirements:
    - numpy, matplotlib, scipy
    - Simulation results from solve.py
    - Sufficient memory for high-quality rendering
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import json
import sys
from pathlib import Path
from scipy.interpolate import interp1d
import time

# Import our simulation modules
try:
    from equations import CoilgunPhysicsEngine
    from solve import CoilgunSimulation, MultiStageCoilgunSimulation
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure equations.py and solve.py are in the same directory.")
    sys.exit(1)
from view import CoilgunFieldVisualizer

class Arrow3D(FancyArrowPatch):
    """3D arrow patch for better arrow rendering in 3D."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self):
        """Project the 3D arrow and return minimum Z for depth ordering."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class Advanced3DCoilgunVisualizer:
    """
    Advanced 3D visualizer for coilgun simulations with realistic physics representation.
    Renders coils exactly like view.py with proper vertical stacking and correct projectile dimensions.
    """
    
    def __init__(self, simulation_results_dir=None, config_file=None):
        """
        Initialize the 3D visualizer.
        
        Args:
            simulation_results_dir: Directory containing simulation results
            config_file: Configuration file path (if running new simulation)
        """
        self.simulation_results_dir = simulation_results_dir
        self.config_file = config_file
        
        # Animation parameters - slower and shorter for better visualization
        self.animation_duration = 15.0  # seconds - longer for slower animation
        self.fps = 30
        self.total_frames = int(self.animation_duration * self.fps)
        
        # 3D visualization settings
        self.coil_visual_turns = 8  # Number of visual turns per coil (like view.py)
        self.field_line_density = 16  # number of field lines
        self.extended_trajectory_time = 0.01  # Very short time after exit (1cm more travel)
        self.stage_spacing = 0.12  # 12cm spacing between stages for better visualization
        
        # Color schemes - copper color like view.py
        self.copper_colormap = plt.cm.copper
        self.field_colormap = plt.cm.viridis
        
        # Data storage
        self.simulation_data = None
        self.physics_engine = None
        self.is_multi_stage = False
        self.stage_configs = []
        
        # 3D geometry cache
        self.coil_geometry_cache = {}
        self.field_line_cache = {}
        
        # Material colors for projectile rendering
        self.projectile_materials = {
            'Pure_Iron': {'color': [0.7, 0.7, 0.7], 'metallic': True},
            'Low_Carbon_Steel': {'color': [0.4, 0.4, 0.5], 'metallic': True},
            'Silicon_Steel': {'color': [0.5, 0.5, 0.6], 'metallic': True},
            'Aluminum': {'color': [0.8, 0.8, 0.9], 'metallic': True},
            'Copper': {'color': [0.7, 0.4, 0.2], 'metallic': True}
        }
        
        # Animation timing
        self.real_time_start = None
        
    def load_simulation_data(self, results_dir=None):
        """
        Load simulation data from results directory.
        
        Args:
            results_dir: Path to results directory
        """
        if results_dir is None:
            results_dir = self.simulation_results_dir
        
        if results_dir is None:
            raise ValueError("No results directory specified")
        
        results_path = Path(results_dir)
        if not results_path.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        print(f"Loading simulation data from: {results_path}")
        
        # Load configuration
        config_file = results_path / "simulation_config.json"
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Check if multi-stage
        self.is_multi_stage = self.config.get("multi_stage", {}).get("enabled", False)
        
        if self.is_multi_stage:
            print("Loading multi-stage simulation data...")
            self._load_multistage_data(results_path)
        else:
            print("Loading single-stage simulation data...")
            self._load_single_stage_data(results_path)
        
        print(f"Data loaded successfully. Animation duration: {self.animation_duration:.1f}s")
    
    def _load_single_stage_data(self, results_path):
        """Load single-stage simulation data."""
        # Load summary
        summary_file = results_path / "simulation_summary.json"
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Load time series data
        npz_file = results_path / "time_series_data.npz"
        csv_file = results_path / "time_series_data.csv"
        
        if npz_file.exists():
            with np.load(npz_file) as data:
                time_series = {key: data[key] for key in data.files}
        elif csv_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                time_series = {col: df[col].values for col in df.columns}
            except ImportError:
                raise ImportError("pandas required for CSV loading when npz not available")
        else:
            raise FileNotFoundError("No time series data file found")
        
        # Store simulation data
        self.simulation_data = {
            'time': time_series.get('time_s', time_series.get('time')),
            'current': time_series.get('current_A', time_series.get('current')),
            'position': time_series.get('position_m', time_series.get('position')),
            'velocity': time_series.get('velocity_ms', time_series.get('velocity')),
            'force': time_series.get('force_N', time_series.get('force')),
            'summary': summary_data['summary']
        }
        
        # Initialize physics engine
        self.physics_engine = CoilgunPhysicsEngine(str(results_path / "simulation_config.json"))
        
        # Set animation duration based on simulation time
        sim_duration = self.simulation_data['time'][-1]
        self.animation_duration = min(10.0, max(3.0, sim_duration * 2))
    
    def _load_multistage_data(self, results_path):
        """Load multi-stage simulation data."""
        # Load summary
        summary_file = results_path / "multistage_simulation_summary.json"
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Load aggregated time series data
        npz_file = results_path / "multistage_time_series_data.npz"
        csv_file = results_path / "multistage_time_series_data.csv"
        
        if npz_file.exists():
            with np.load(npz_file) as data:
                time_series = {key: data[key] for key in data.files}
        elif csv_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                time_series = {col: df[col].values for col in df.columns}
            except ImportError:
                raise ImportError("pandas required for CSV loading when npz not available")
        else:
            raise FileNotFoundError("No time series data file found")
        
        # Store simulation data with validation
        time_data = time_series.get('time_s', time_series.get('time'))
        position_data = time_series.get('position_m', time_series.get('position'))
        
        # Debug position data to check for resets
        if len(position_data) > 100:
            pos_diffs = np.diff(position_data)
            negative_jumps = pos_diffs < -0.05  # Detect position resets (>5cm backward)
            if np.any(negative_jumps):
                reset_indices = np.where(negative_jumps)[0]
                print(f"Warning: Detected {len(reset_indices)} position resets in multi-stage data")
                print(f"Reset locations: {reset_indices}")
                
                # Fix position resets by making positions cumulative
                print("Fixing position data to be cumulative...")
                fixed_position = position_data.copy()
                for reset_idx in reset_indices:
                    # Get the position offset at reset
                    offset = fixed_position[reset_idx] - fixed_position[reset_idx + 1]
                    # Apply offset to all subsequent positions
                    fixed_position[reset_idx + 1:] += offset
                
                position_data = fixed_position
                print("Position data corrected to be cumulative across stages")
        
        self.simulation_data = {
            'time': time_data,
            'current': time_series.get('current_A', time_series.get('current')),
            'position': position_data,
            'velocity': time_series.get('velocity_ms', time_series.get('velocity')),
            'force': time_series.get('force_N', time_series.get('force')),
            'stage_transitions': time_series.get('stage_transitions', []),
            'summary': summary_data['summary'],
            'num_stages': summary_data['simulation_info']['num_stages']
        }
        
        # Load individual stage configurations
        self._load_stage_configurations(results_path)
        
        # Set animation duration to focus on coil passage - much slower for visibility
        sim_duration = self.simulation_data['time'][-1]
        self.animation_duration = min(25.0, max(12.0, sim_duration * 8))  # 8x slower for coil visibility
    
    def _load_stage_configurations(self, results_path):
        """Load individual stage configurations for multi-stage visualization."""
        self.stage_configs = []
        num_stages = self.simulation_data['num_stages']
        
        for stage_num in range(1, num_stages + 1):
            stage_dir = results_path / f"stage_{stage_num}_results"
            if stage_dir.exists():
                stage_config_file = stage_dir / "simulation_config.json"
                if stage_config_file.exists():
                    with open(stage_config_file, 'r') as f:
                        stage_config = json.load(f)
                    
                    # Create physics engine for this stage
                    physics_engine = CoilgunPhysicsEngine(str(stage_config_file))
                    
                    self.stage_configs.append({
                        'config': stage_config,
                        'physics': physics_engine,
                        'stage_num': stage_num
                    })
        
        # Use first stage as primary physics engine
        if self.stage_configs:
            self.physics_engine = self.stage_configs[0]['physics']
        
        # Load individual stage current data for realistic rendering
        self._load_stage_current_data(results_path)
    
    def _load_stage_current_data(self, results_path):
        """Load individual stage current data for realistic current visualization."""
        self.stage_current_data = {}
        
        for stage_config in self.stage_configs:
            stage_num = stage_config['stage_num']
            stage_dir = results_path / f"stage_{stage_num}_results"
            
            # Try to load stage-specific time series data
            npz_file = stage_dir / "time_series_data.npz"
            csv_file = stage_dir / "time_series_data.csv"
            
            if npz_file.exists():
                with np.load(npz_file) as data:
                    stage_time_series = {key: data[key] for key in data.files}
            elif csv_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    stage_time_series = {col: df[col].values for col in df.columns}
                except ImportError:
                    print(f"Warning: Could not load stage {stage_num} current data (pandas required)")
                    continue
            else:
                print(f"Warning: No time series data found for stage {stage_num}")
                continue
            
            # Store stage current data
            self.stage_current_data[stage_num] = {
                'time': stage_time_series.get('time_s', stage_time_series.get('time', [])),
                'current': stage_time_series.get('current_A', stage_time_series.get('current', []))
            }
            
            print(f"Loaded current data for stage {stage_num}: {len(self.stage_current_data[stage_num]['time'])} points")
    
    def get_stage_current_at_time(self, stage_num, sim_time, projectile_pos):
        """Get the current for a specific stage with strict position-based firing to prevent overlap."""
        if stage_num not in self.stage_current_data:
            return 0.0
        
        stage_data = self.stage_current_data[stage_num]
        if len(stage_data['time']) == 0 or len(stage_data['current']) == 0:
            return 0.0
        
        # Get stage transition times from simulation
        stage_transitions = self.simulation_data.get('stage_transitions', [])
        
        # Get the actual coil position for this stage
        coil_positions = self._get_actual_coil_positions()
        if stage_num > len(coil_positions):
            return 0.0
        
        coil_start = coil_positions[stage_num - 1]
        
        # Get coil physics for timing calculations
        if stage_num <= len(self.stage_configs):
            coil_physics = self.stage_configs[stage_num - 1]['physics']
            coil_length = coil_physics.coil_length
        else:
            return 0.0
        
        coil_center = coil_start + coil_length / 2
        coil_end = coil_start + coil_length
        
        # STRICT POSITION-BASED FIRING LOGIC
        # Only fire if this is the closest coil to the projectile
        
        # Calculate distances to this coil
        distance_to_coil_start = coil_start - projectile_pos
        distance_to_center = coil_center - projectile_pos
        distance_past_coil_end = projectile_pos - coil_end
        
        # Find which coil the projectile is closest to
        closest_coil_stage = self._find_closest_coil_to_projectile(projectile_pos)
        
        # Only allow firing if this is the closest coil or the next coil in line
        if stage_num != closest_coil_stage and stage_num != (closest_coil_stage + 1):
            return 0.0  # Not the active coil
        
        # Determine base firing time from simulation (with delays)
        if stage_num == 1:
            base_start_time = 0.0
        elif stage_num - 2 < len(stage_transitions):
            original_start_time = stage_transitions[stage_num - 2]
            # Apply any spacing-based firing delay
            if hasattr(self, '_firing_delays') and stage_num in self._firing_delays:
                firing_delay = self._firing_delays[stage_num]
                base_start_time = original_start_time + firing_delay
            else:
                base_start_time = original_start_time
        else:
            return 0.0
        
        # Calculate stage local time
        stage_local_time = sim_time - base_start_time
        
        # More restrictive timing - only fire when projectile is actually approaching this coil
        if stage_num == closest_coil_stage:
            # This is the closest coil - fire if projectile is approaching or in coil
            max_distance_before = 0.15  # 15cm before coil
            max_distance_after = 0.05   # 5cm after coil
            
            if distance_to_coil_start > max_distance_before:
                return 0.0  # Too far before
            if distance_past_coil_end > max_distance_after:
                return 0.0  # Too far after
                
        elif stage_num == (closest_coil_stage + 1):
            # This is the next coil - only start firing when projectile is very close to this coil
            max_distance_before = 0.08  # Only 8cm before coil (much more restrictive)
            
            if distance_to_coil_start > max_distance_before:
                return 0.0  # Wait until projectile is closer
            if distance_past_coil_end > 0.0:
                return 0.0  # Don't fire if projectile already passed
        else:
            return 0.0  # Not an active coil
        
        # Get base current from timing data
        try:
            from scipy.interpolate import interp1d
            
            # Use stage local time, but clamp to valid range
            lookup_time = max(0.0, min(stage_local_time, stage_data['time'][-1]))
            
            if len(stage_data['time']) > 1:
                interp_func = interp1d(stage_data['time'], stage_data['current'], 
                                     kind='linear', bounds_error=False, fill_value=0.0)
                base_current = float(interp_func(lookup_time))
            else:
                base_current = stage_data['current'][0] if len(stage_data['current']) > 0 else 0.0
            
            # Apply position-based scaling for optimal force delivery
            if distance_to_center > 0:
                # Before center - ramp up current as projectile approaches
                if distance_to_coil_start > 0:
                    # Ramp based on approach distance
                    ramp_distance = 0.1  # 10cm ramp distance
                    position_scale = 1.0 - (distance_to_coil_start / ramp_distance)
                    position_scale = max(0.2, min(1.0, position_scale))
                else:
                    position_scale = 1.0  # Projectile is at or past coil start
            else:
                # At or past center
                if distance_past_coil_end <= 0:
                    # Still in coil - maintain good current
                    position_scale = 1.0
                else:
                    # Past coil - quickly reduce current
                    fade_distance = 0.03  # 3cm fade distance
                    position_scale = max(0.0, 1.0 - (distance_past_coil_end / fade_distance))
            
            final_current = base_current * position_scale
            
            # Debug output for active stages
            if final_current > 1.0 and sim_time < 0.1:
                print(f"FIRING Stage {stage_num}: closest={closest_coil_stage}, "
                      f"proj_pos={projectile_pos:.3f}m, coil_start={coil_start:.3f}m, "
                      f"current={final_current:.1f}A")
            
            return final_current
            
        except Exception as e:
            print(f"Warning: Current calculation failed for stage {stage_num}: {e}")
            return 0.0
    
    def _find_closest_coil_to_projectile(self, projectile_pos):
        """
        Find which coil stage the projectile is closest to.
        
        Args:
            projectile_pos: Current projectile position
            
        Returns:
            Stage number of closest coil (1-indexed)
        """
        if not hasattr(self, '_actual_coil_positions') or not self.stage_configs:
            return 1
        
        coil_positions = self._actual_coil_positions
        min_distance = float('inf')
        closest_stage = 1
        
        for i, coil_start in enumerate(coil_positions):
            stage_num = i + 1
            if stage_num <= len(self.stage_configs):
                coil_physics = self.stage_configs[i]['physics']
                coil_center = coil_start + coil_physics.coil_length / 2
                
                # Distance to coil center
                distance = abs(projectile_pos - coil_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_stage = stage_num
        
        return closest_stage
    
    def _get_actual_coil_positions(self):
        """
        Get coil positions with enforced minimum spacing and calculate firing delays.
        This prevents visual overlap while maintaining correct firing timing.
        
        Returns:
            List of coil start positions with proper spacing
        """
        if hasattr(self, '_actual_coil_positions'):
            return self._actual_coil_positions
        
        stage_positions = []
        
        if not (self.is_multi_stage and 'stage_transitions' in self.simulation_data):
            # Single stage or no transition data
            stage_positions.append(0.0)
            self._actual_coil_positions = stage_positions
            self._firing_delays = {1: 0.0}  # No delay for single stage
            return stage_positions
        
        # Get stage transition times and simulation data
        stage_transitions = self.simulation_data.get('stage_transitions', [])
        sim_time = self.simulation_data['time']
        sim_position = self.simulation_data['position']
        sim_velocity = self.simulation_data['velocity']
        
        print("Calculating coil positions with spacing enforcement and firing delays...")
        
        # Initialize firing delays storage
        self._firing_delays = {}
        
        try:
            # Interpolate position and velocity vs time
            from scipy.interpolate import interp1d
            if len(sim_time) > 1 and len(sim_position) > 1:
                position_interp = interp1d(sim_time, sim_position, 
                                         kind='linear', bounds_error=False, 
                                         fill_value=(sim_position[0], sim_position[-1]))
                velocity_interp = interp1d(sim_time, sim_velocity,
                                         kind='linear', bounds_error=False,
                                         fill_value=(sim_velocity[0], sim_velocity[-1]))
                
                # First stage always starts at 0 with no delay
                stage_positions.append(0.0)
                self._firing_delays[1] = 0.0
                print(f"Stage 1: position=0.000m, delay=0.000s")
                
                # Calculate positions with enforced minimum spacing and firing delays
                for i, transition_time in enumerate(stage_transitions):
                    stage_num = i + 2  # Stage numbers are 1-indexed, transitions start at stage 2
                    
                    if i < len(self.stage_configs) - 1:  # Ensure we have stage config
                        prev_stage_physics = self.stage_configs[i]['physics']
                        current_stage_physics = self.stage_configs[i + 1]['physics']
                        
                        # Calculate minimum position based on previous coil
                        min_position = stage_positions[-1] + prev_stage_physics.coil_length + self.stage_spacing
                        
                        # Get timing-based position as reference
                        if transition_time > 0 and transition_time < sim_time[-1]:
                            timing_position = float(position_interp(transition_time))
                            transition_velocity = float(velocity_interp(transition_time))
                        else:
                            timing_position = None
                            transition_velocity = sim_velocity[-1]  # Use final velocity as fallback
                        
                        # Use the larger of timing-based or minimum spacing position
                        if timing_position is not None and timing_position > min_position:
                            final_position = timing_position
                            firing_delay = 0.0  # No delay needed, timing matches position
                            reason = f"timing-based at t={transition_time:.4f}s"
                        else:
                            final_position = min_position
                            # Calculate firing delay based on projectile travel time
                            if timing_position is not None and transition_velocity > 0:
                                # Distance projectile needs to travel to reach this coil
                                extra_distance = final_position - timing_position
                                # Time delay based on projectile velocity at transition
                                firing_delay = extra_distance / transition_velocity
                            else:
                                firing_delay = 0.0  # Fallback if no velocity data
                            
                            reason = f"spaced at {final_position:.3f}m, delay={firing_delay:.4f}s"
                        
                        stage_positions.append(final_position)
                        self._firing_delays[stage_num] = firing_delay
                        print(f"Stage {stage_num}: position={final_position:.3f}m, delay={firing_delay:.4f}s ({reason})")
                    else:
                        # Fallback for missing stage config
                        fallback_pos = stage_positions[-1] + 0.15  # 15cm default spacing
                        stage_positions.append(fallback_pos)
                        self._firing_delays[stage_num] = 0.0
                        print(f"Stage {stage_num}: position={fallback_pos:.3f}m, delay=0.000s (fallback)")
                        
        except Exception as e:
            print(f"Warning: Could not calculate positions with delays: {e}")
            print("Using fallback spacing...")
            # Fallback to simple spacing
            stage_positions = [0.0]
            self._firing_delays = {1: 0.0}
            if self.is_multi_stage and self.stage_configs:
                for i in range(len(self.stage_configs) - 1):
                    prev_physics = self.stage_configs[i]['physics']
                    next_pos = stage_positions[-1] + prev_physics.coil_length + self.stage_spacing
                    stage_positions.append(next_pos)
                    self._firing_delays[i + 2] = 0.0
        
        # Cache the calculated positions
        self._actual_coil_positions = stage_positions
        print(f"Final coil positions: {[f'{pos:.3f}m' for pos in stage_positions]}")
        print(f"Firing delays: {[(f'Stage {k}: {v:.4f}s') for k, v in self._firing_delays.items()]}")
        
        return stage_positions
    
    def create_3d_coil_geometry(self, physics_engine, z_offset=0.0):
        """
        Create 3D coil geometry exactly like view.py.
        
        Args:
            physics_engine: Physics engine for coil parameters
            z_offset: Vertical offset for multi-stage stacking
            
        Returns:
            List of coil line coordinates for 3D plotting
        """
        # Create helical coil path - exactly like view.py
        turns = np.linspace(0, self.coil_visual_turns, 1000)
        theta = 2 * np.pi * turns
        
        # Axial position with offset for vertical stacking
        z_coil = (turns / self.coil_visual_turns) * physics_engine.coil_length + z_offset
        
        # Debug: Print z-range for this coil
        print(f"Creating coil at z_offset={z_offset:.3f}m, z_range=[{z_coil[0]:.3f}, {z_coil[-1]:.3f}]")
        
        # Create multiple layers
        coil_lines = []
        
        for layer in range(physics_engine.num_layers):
            # Radius for this layer - exactly like view.py
            layer_radius = (physics_engine.coil_inner_radius + 
                           layer * (physics_engine.coil_outer_radius - physics_engine.coil_inner_radius) / physics_engine.num_layers)
            
            # Coordinates for this layer
            x_coil = layer_radius * np.cos(theta)
            y_coil = layer_radius * np.sin(theta)
            
            coil_lines.append({
                'x': x_coil,
                'y': y_coil, 
                'z': z_coil,  # This should contain the z_offset
                'layer': layer,
                'radius': layer_radius
            })
        
        return coil_lines
    
    def create_3d_projectile_geometry(self, position, physics_engine):
        """
        Create enhanced 3D projectile geometry with material properties and better rendering.
        
        Args:
            position: Axial position of projectile
            physics_engine: Physics engine for projectile parameters
            
        Returns:
            Dict with projectile mesh coordinates and material properties
        """
        # Create high-resolution cylindrical projectile
        theta = np.linspace(0, 2*np.pi, 32)  # Higher resolution
        z_proj = np.array([position - physics_engine.proj_length, position])
        
        # Create surface coordinates
        theta_mesh, z_mesh = np.meshgrid(theta, z_proj)
        x_mesh = physics_engine.proj_radius * np.cos(theta_mesh)
        y_mesh = physics_engine.proj_radius * np.sin(theta_mesh)
        
        # Get material properties
        material_name = getattr(physics_engine, 'config', {}).get('projectile', {}).get('material', 'Low_Carbon_Steel')
        material_props = self.projectile_materials.get(material_name, self.projectile_materials['Low_Carbon_Steel'])
        
        # Create end caps for better 3D appearance
        theta_cap = np.linspace(0, 2*np.pi, 32)
        r_cap = np.linspace(0, physics_engine.proj_radius, 8)
        theta_cap_mesh, r_cap_mesh = np.meshgrid(theta_cap, r_cap)
        
        # Front cap (at current position)
        x_cap_front = r_cap_mesh * np.cos(theta_cap_mesh)
        y_cap_front = r_cap_mesh * np.sin(theta_cap_mesh)
        z_cap_front = np.full_like(x_cap_front, position)
        
        # Back cap (at position - length)
        x_cap_back = r_cap_mesh * np.cos(theta_cap_mesh)
        y_cap_back = r_cap_mesh * np.sin(theta_cap_mesh)
        z_cap_back = np.full_like(x_cap_back, position - physics_engine.proj_length)
        
        return {
            'body': {'x': x_mesh, 'y': y_mesh, 'z': z_mesh},
            'front_cap': {'x': x_cap_front, 'y': y_cap_front, 'z': z_cap_front},
            'back_cap': {'x': x_cap_back, 'y': y_cap_back, 'z': z_cap_back},
            'material': material_props,
            'center_position': position - physics_engine.proj_length / 2,
            'dimensions': {
                'diameter': physics_engine.proj_diameter * 1000,  # Convert to mm
                'length': physics_engine.proj_length * 1000,     # Convert to mm
                'mass': physics_engine.proj_mass * 1000         # Convert to grams
            }
        }
    
    def create_magnetic_field_lines(self, physics_engine, current, z_offset=0.0):
        """
        Create magnetic field line visualization for a coil stage.
        
        Args:
            physics_engine: Physics engine for field calculation
            current: Current in the coil
            z_offset: Vertical offset for multi-stage stacking
            
        Returns:
            List of field line coordinates
        """
        if abs(current) < 1.0:
            return []
        
        field_lines = []
        
        # Create radial field lines from coil - similar to view.py approach
        num_lines = 8
        theta_start = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
        
        # Multiple starting positions along coil
        z_starts = [
            z_offset + physics_engine.coil_length * 0.2,
            z_offset + physics_engine.coil_length * 0.5,
            z_offset + physics_engine.coil_length * 0.8
        ]
        
        for z_start in z_starts:
            for theta in theta_start:
                # Start from coil inner radius
                r_start = physics_engine.coil_inner_radius * 1.1
                x_start = r_start * np.cos(theta)
                y_start = r_start * np.sin(theta)
                
                # Create field line using solenoid field approximation
                t = np.linspace(0, 1, 50)
                
                # Field line geometry based on solenoid field
                r_expand = r_start + (physics_engine.coil_outer_radius * 1.5 - r_start) * t**2
                z_curve = z_start + 0.2 * (t - 0.5) * physics_engine.coil_length * np.sign(current)
                
                x_line = r_expand * np.cos(theta + 0.1 * t * np.sign(current))
                y_line = r_expand * np.sin(theta + 0.1 * t * np.sign(current))
                z_line = z_curve
                
                # Only add field lines that stay within reasonable bounds
                if np.max(r_expand) < physics_engine.coil_outer_radius * 2:
                    field_lines.append(np.column_stack([x_line, y_line, z_line]))
        
        return field_lines
    
    def get_copper_color_with_current(self, current, max_current, layer_index, total_layers):
        """
        Get copper color with current-dependent intensity, exactly like view.py.
        
        Args:
            current: Current in the coil
            max_current: Maximum current for normalization
            layer_index: Layer index for color variation
            total_layers: Total number of layers
            
        Returns:
            RGB color array
        """
        if max_current == 0:
            intensity = 0.5
        else:
            intensity = abs(current) / max_current
            intensity = np.clip(intensity, 0.1, 1.0)  # Keep some minimum visibility
        
        # Base copper color from layer position (like view.py)
        layer_fraction = layer_index / max(1, total_layers - 1)
        base_copper = np.array(self.copper_colormap(layer_fraction)[:3])  # Convert to numpy array
        
        # Apply current intensity
        copper_color = base_copper * (0.3 + 0.7 * intensity)
        
        # Add glow effect for high currents
        if intensity > 0.5:
            glow_factor = (intensity - 0.5) * 2.0
            copper_color = copper_color + glow_factor * np.array([0.3, 0.2, 0.0])
        
        # Ensure values stay in valid range
        copper_color = np.clip(copper_color, 0, 1)
        
        return copper_color
    
    def get_coil_array_specifications(self):
        """
        Extract coil array specifications from configuration for display.
        
        Returns:
            Dict with coil array specifications
        """
        specs = {
            'total_stages': 1,
            'total_height': 0,
            'total_mass_estimate': 0,
            'inner_diameter': 0,
            'outer_diameter': 0,
            'stage_specs': []
        }
        
        if self.is_multi_stage and self.stage_configs:
            specs['total_stages'] = len(self.stage_configs)
            for i, stage_config in enumerate(self.stage_configs):
                stage_physics = stage_config['physics']
                
                # Calculate copper mass estimate for this stage
                wire_volume = stage_physics.wire_length * stage_physics.wire_area
                copper_density = 8960  # kg/m³
                wire_mass = wire_volume * copper_density
                
                stage_spec = {
                    'stage_num': stage_config['stage_num'],
                    'inner_diameter_mm': stage_physics.coil_inner_radius * 2 * 1000,
                    'outer_diameter_mm': stage_physics.coil_outer_radius * 2 * 1000,
                    'length_mm': stage_physics.coil_length * 1000,
                    'turns': int(stage_physics.total_turns),
                    'wire_gauge': stage_physics.config['coil']['wire_gauge_awg'],
                    'wire_length_m': stage_physics.wire_length,
                    'resistance_ohm': stage_physics.coil_resistance,
                    'mass_estimate_g': wire_mass * 1000,
                    'capacitance_mf': stage_physics.capacitance * 1000,
                    'max_voltage_v': stage_physics.initial_voltage
                }
                specs['stage_specs'].append(stage_spec)
                
                # Update totals
                specs['total_height'] += stage_physics.coil_length
                specs['total_mass_estimate'] += wire_mass * 1000
                
                # Use first stage for common dimensions
                if i == 0:
                    specs['inner_diameter'] = stage_spec['inner_diameter_mm']
                    specs['outer_diameter'] = stage_spec['outer_diameter_mm']
        else:
            # Single stage
            stage_physics = self.physics_engine
            wire_volume = stage_physics.wire_length * stage_physics.wire_area
            copper_density = 8960  # kg/m³
            wire_mass = wire_volume * copper_density
            
            specs['inner_diameter'] = stage_physics.coil_inner_radius * 2 * 1000
            specs['outer_diameter'] = stage_physics.coil_outer_radius * 2 * 1000
            specs['total_height'] = stage_physics.coil_length * 1000
            specs['total_mass_estimate'] = wire_mass * 1000
            
            stage_spec = {
                'stage_num': 1,
                'inner_diameter_mm': specs['inner_diameter'],
                'outer_diameter_mm': specs['outer_diameter'],
                'length_mm': specs['total_height'],
                'turns': int(stage_physics.total_turns),
                'wire_gauge': stage_physics.config['coil']['wire_gauge_awg'],
                'wire_length_m': stage_physics.wire_length,
                'resistance_ohm': stage_physics.coil_resistance,
                'mass_estimate_g': specs['total_mass_estimate'],
                'capacitance_mf': stage_physics.capacitance * 1000,
                'max_voltage_v': stage_physics.initial_voltage
            }
            specs['stage_specs'] = [stage_spec]
        
        # Add spacing between stages for multi-stage
        if specs['total_stages'] > 1:
            specs['total_height'] += (specs['total_stages'] - 1) * self.stage_spacing * 1000
        
        return specs
    
    def format_altitude_display(self, position, coil_specs):
        """
        Format projectile altitude relative to coil array.
        
        Args:
            position: Current projectile position (m)
            coil_specs: Coil array specifications
            
        Returns:
            Formatted altitude string
        """
        position_mm = position * 1000  # Convert to mm
        
        if self.is_multi_stage and self.stage_configs:
            # Find which stage the projectile is in/near using actual coil positions
            coil_positions = self._get_actual_coil_positions()
            
            for i, stage_config in enumerate(self.stage_configs):
                stage_physics = stage_config['physics']
                
                if i < len(coil_positions):
                    stage_start_m = coil_positions[i]
                else:
                    stage_start_m = i * 0.15  # Fallback spacing
                
                stage_start = stage_start_m * 1000
                stage_end = (stage_start_m + stage_physics.coil_length) * 1000
                
                if stage_start <= position_mm <= stage_end:
                    current_stage = i + 1
                    relative_pos = position_mm - stage_start
                    return f"Alt: {position_mm:.1f}mm\nStage {current_stage}: +{relative_pos:.1f}mm"
                elif position_mm < stage_start:
                    distance_to_stage = stage_start - position_mm
                    return f"Alt: {position_mm:.1f}mm\n→ Stage {i+1}: {distance_to_stage:.1f}mm"
            
            # Past all stages - calculate total coil array length
            if len(coil_positions) > 0:
                last_coil_end = coil_positions[-1] + self.stage_configs[-1]['physics'].coil_length
                past_distance = position_mm - (last_coil_end * 1000)
            else:
                past_distance = position_mm
            return f"Alt: {position_mm:.1f}mm\nPost-Launch: +{past_distance:.1f}mm"
        else:
            # Single stage
            coil_length_mm = coil_specs['total_height']
            if 0 <= position_mm <= coil_length_mm:
                return f"Alt: {position_mm:.1f}mm\nIn Coil: {(position_mm/coil_length_mm)*100:.0f}%"
            elif position_mm < 0:
                return f"Alt: {position_mm:.1f}mm\nPre-Launch: {abs(position_mm):.1f}mm"
            else:
                return f"Alt: {position_mm:.1f}mm\nPost-Launch: +{position_mm - coil_length_mm:.1f}mm"
    
    def create_animation(self, save_path=None, show_animation=True):
        """Create comprehensive 3D animation with proper vertical stacking and realistic rendering."""
        print("Creating 3D coilgun animation with vertical stacking...")
        
        # Prepare data
        sim_time = self.simulation_data['time']
        sim_current = self.simulation_data['current']
        sim_position = self.simulation_data['position']
        sim_velocity = self.simulation_data['velocity']
        
        # Calculate extended trajectory
        final_time = sim_time[-1]
        final_position = sim_position[-1]
        final_velocity = sim_velocity[-1]
        
        # Extended trajectory (ballistic)
        ext_time_points = np.linspace(final_time, final_time + self.extended_trajectory_time, 50)
        ext_positions = final_position + final_velocity * (ext_time_points - final_time)
        
        # Combine data
        full_times = np.concatenate([sim_time, ext_time_points])
        full_positions = np.concatenate([sim_position, ext_positions])
        full_currents = np.concatenate([sim_current, np.zeros(len(ext_time_points))])
        full_velocities = np.concatenate([sim_velocity, np.full(len(ext_time_points), final_velocity)])
        
        # Animation time points
        total_frames = int(self.animation_duration * self.fps)
        animation_times = np.linspace(0, self.animation_duration, total_frames)
        
        # Map to simulation time with better scaling to reduce stuttering
        sim_time_scale = full_times[-1] / self.animation_duration
        scaled_times = animation_times * sim_time_scale
        
        # Ensure monotonic time arrays for better interpolation
        if len(full_times) > 1:
            time_diffs = np.diff(full_times)
            if np.any(time_diffs <= 0):
                print("Warning: Non-monotonic time data detected, fixing...")
                # Remove duplicates and ensure monotonic increase
                unique_mask = np.concatenate([[True], time_diffs > 1e-10])
                full_times = full_times[unique_mask]
                full_positions = full_positions[unique_mask]
                full_currents = full_currents[unique_mask]
                full_velocities = full_velocities[unique_mask]
        
        # Interpolate data with smoother interpolation to reduce stuttering
        try:
            interp_position = interp1d(full_times, full_positions, kind='cubic', bounds_error=False, fill_value='extrapolate')
            interp_velocity = interp1d(full_times, full_velocities, kind='cubic', bounds_error=False, fill_value=final_velocity)
        except ValueError:
            # Fallback to linear if cubic fails
            print("Warning: Cubic interpolation failed, using linear...")
            interp_position = interp1d(full_times, full_positions, kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_velocity = interp1d(full_times, full_velocities, kind='linear', bounds_error=False, fill_value=final_velocity)
        
        interp_current = interp1d(full_times, full_currents, kind='linear', bounds_error=False, fill_value=0)
        
        anim_positions = interp_position(scaled_times)
        anim_currents = interp_current(scaled_times)
        anim_velocities = interp_velocity(scaled_times)
        
        max_current = np.max(np.abs(sim_current))
        
        # Get coil specifications for display
        coil_specs = self.get_coil_array_specifications()
        
        # Create figure with larger layout for maximum detail viewing
        fig = plt.figure(figsize=(26, 14))
        
        # Main 3D plot - much larger for enhanced detail
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        # Information panel - maintains proportion with larger figure
        info_ax = fig.add_subplot(1, 2, 2)
        info_ax.axis('off')
        
        # Initialize real-time tracking
        self.real_time_start = time.time()
        
        # Pre-generate vertically stacked coil geometries with timing-based spacing
        coil_geometries = []
        stage_physics_list = []
        
        if self.is_multi_stage and self.stage_configs:
            print("Pre-generating vertically stacked multi-stage coil geometries...")
            print(f"Number of stage configs loaded: {len(self.stage_configs)}")
            
            # Get actual coil positions from simulation data
            stage_positions = self._get_actual_coil_positions()
            
            for i, stage_config in enumerate(self.stage_configs):
                stage_physics = stage_config['physics']
                stage_num = stage_config['stage_num']
                
                # Use actual position from simulation data
                if stage_num <= len(stage_positions):
                    z_offset = stage_positions[stage_num - 1]
                else:
                    # Fallback to spacing-based position
                    if i == 0:
                        z_offset = 0.0
                    else:
                        prev_geom = coil_geometries[i-1]
                        z_offset = prev_geom['z_offset'] + prev_geom['length'] + 0.05  # 5cm spacing
                
                # Create coil geometry at calculated offset
                coil_lines = self.create_3d_coil_geometry(stage_physics, z_offset)
                
                print(f"Stage {stage_num}: z_offset={z_offset:.3f}m, length={stage_physics.coil_length:.3f}m")
                
                coil_geometries.append({
                    'coil_lines': coil_lines,
                    'z_offset': z_offset,
                    'length': stage_physics.coil_length,
                    'inner_radius': stage_physics.coil_inner_radius,
                    'outer_radius': stage_physics.coil_outer_radius,
                    'stage_num': stage_num
                })
                
                stage_physics_list.append(stage_physics)
            
            total_span = max(geom['z_offset'] + geom['length'] for geom in coil_geometries)
            print(f"Total vertical span: {total_span:.3f}m")
            print(f"Generated {len(coil_geometries)} coil geometries with timing-based spacing")
        else:
            # Single stage
            coil_lines = self.create_3d_coil_geometry(self.physics_engine, 0.0)
            coil_geometries = [{
                'coil_lines': coil_lines,
                'z_offset': 0.0,
                'length': self.physics_engine.coil_length,
                'inner_radius': self.physics_engine.coil_inner_radius,
                'outer_radius': self.physics_engine.coil_outer_radius,
                'stage_num': 1
            }]
            stage_physics_list = [self.physics_engine]
        
        # Set axis limits based on all coils
        max_radius = max(geom['outer_radius'] for geom in coil_geometries)
        total_height = sum(geom['length'] for geom in coil_geometries) + (len(coil_geometries) - 1) * self.stage_spacing
        max_z = total_height + final_velocity * self.extended_trajectory_time
        
        print(f"Animation limits: max_radius={max_radius:.3f}m, total_height={total_height:.3f}m, max_z={max_z:.3f}m")
        print(f"Coil z-ranges: {[(g['z_offset'], g['z_offset']+g['length']) for g in coil_geometries]}")
        
        def animate_frame(frame):
            ax.clear()
            
            # Current frame data
            current_time = animation_times[frame]
            current_pos = anim_positions[frame]
            current_current = anim_currents[frame]
            current_velocity = anim_velocities[frame]
            
            # Calculate simulation time for current lookup
            sim_time_current = current_time * sim_time_scale
            
            # Define enhanced vertical stretch factor for maximum coil separation
            z_stretch_factor = 2.4
            
            # Render all coil stages with proper copper coloring - ALWAYS SHOW ALL STAGES
            for stage_idx, coil_geom in enumerate(coil_geometries):
                stage_physics = stage_physics_list[stage_idx]
                stage_num = coil_geom['stage_num']
                
                # Get real current for this specific stage at this time
                if self.is_multi_stage:
                    stage_current = self.get_stage_current_at_time(stage_num, sim_time_current, current_pos)
                else:
                    stage_current = current_current
                
                # Debug: Print which stages are being rendered (only on first frame)
                if frame == 0:
                    print(f"Rendering stage {stage_num} at z_offset={coil_geom['z_offset']:.3f}m, current={stage_current:.1f}A")
                
                # Show stage current info in animation (lowered threshold for better visibility)
                if abs(stage_current) > 1.0:  # Show current info for active stages (lowered from 10A)
                    # Add current indicator text near coil with better positioning and vertical stretch
                    center_z = (coil_geom['z_offset'] + coil_geom['length'] / 2) * z_stretch_factor
                    # Use different colors for different current levels
                    if abs(stage_current) > 100:
                        current_color = 'red'
                        current_size = 12
                    elif abs(stage_current) > 10:
                        current_color = 'orange'
                        current_size = 10
                    else:
                        current_color = 'yellow'
                        current_size = 8
                    
                    # Show timing and position info for debugging
                    debug_text = ""
                    coil_center = coil_geom['z_offset'] + coil_geom['length'] / 2
                    distance_to_center = coil_center - current_pos
                    
                    if hasattr(self, '_firing_delays') and stage_num in self._firing_delays:
                        delay = self._firing_delays[stage_num]
                        if delay > 0.0001:  # Show delay if significant
                            debug_text += f"\nDelay: {delay*1000:.1f}ms"
                    
                    # Show distance to coil center
                    if abs(distance_to_center) < 0.3:  # Only show when projectile is nearby
                        debug_text += f"\nDist: {distance_to_center*1000:.0f}mm"
                    
                    ax.text(coil_geom['outer_radius'] * 1.8, 0, center_z, 
                           f'Stage {stage_num}\n{stage_current:.0f}A{debug_text}', 
                           color=current_color, fontsize=current_size, fontweight='bold', 
                           ha='center', va='center')
                
                # Render coil windings with proper copper coloring based on current and vertical stretch
                for layer_idx, coil_line in enumerate(coil_geom['coil_lines']):
                    if abs(stage_current) > 0.1 * max_current:
                        # Active stage - bright copper with current intensity
                        copper_color = self.get_copper_color_with_current(
                            stage_current, max_current, layer_idx, len(coil_geom['coil_lines'])
                        )
                        linewidth = 4
                        alpha = 0.95
                    else:
                        # Inactive stage - dim copper color but visible for context
                        layer_fraction = layer_idx / max(1, len(coil_geom['coil_lines']) - 1)
                        base_copper = np.array(self.copper_colormap(layer_fraction)[:3])
                        copper_color = base_copper * 0.4  # Very dim copper
                        linewidth = 2
                        alpha = 0.5
                    
                    # Apply vertical stretch to coil geometry
                    stretched_z = coil_line['z'] * z_stretch_factor
                    ax.plot(coil_line['x'], coil_line['y'], stretched_z, 
                           color=copper_color, linewidth=linewidth, alpha=alpha)
                
                # Render magnetic field lines only for active stages with vertical stretch
                if abs(stage_current) > 0.1 * max_current:
                    field_lines = self.create_magnetic_field_lines(
                        stage_physics, stage_current, coil_geom['z_offset']
                    )
                    
                    field_intensity = abs(stage_current) / max_current
                    field_color = self.field_colormap(field_intensity)[:3]
                    
                    for field_line in field_lines:
                        # Apply vertical stretch to field lines
                        stretched_field_z = field_line[:, 2] * z_stretch_factor
                        ax.plot(field_line[:, 0], field_line[:, 1], stretched_field_z,
                               color=field_color, alpha=0.5, linewidth=1)
            
            # Render enhanced projectile with material properties and vertical stretch
            if current_pos > -0.1:  # Show when approaching or past coil
                # Use the primary physics engine for projectile dimensions
                primary_physics = stage_physics_list[0]
                projectile_data = self.create_3d_projectile_geometry(current_pos, primary_physics)
                
                # Render projectile body with material color and vertical stretch
                material_color = projectile_data['material']['color']
                metallic_alpha = 0.9 if projectile_data['material']['metallic'] else 0.7
                
                # Apply vertical stretch to projectile geometry
                stretched_body_z = projectile_data['body']['z'] * z_stretch_factor
                stretched_front_z = projectile_data['front_cap']['z'] * z_stretch_factor
                stretched_back_z = projectile_data['back_cap']['z'] * z_stretch_factor
                
                # Main body
                ax.plot_surface(projectile_data['body']['x'], 
                               projectile_data['body']['y'], 
                               stretched_body_z, 
                               color=material_color, alpha=metallic_alpha)
                
                # End caps for better 3D appearance
                ax.plot_surface(projectile_data['front_cap']['x'], 
                               projectile_data['front_cap']['y'], 
                               stretched_front_z, 
                               color=material_color, alpha=metallic_alpha)
                ax.plot_surface(projectile_data['back_cap']['x'], 
                               projectile_data['back_cap']['y'], 
                               stretched_back_z, 
                               color=material_color, alpha=metallic_alpha)
                
                # Projectile altitude display - positioned on left side with vertical stretch
                altitude_text = self.format_altitude_display(current_pos, coil_specs)
                stretched_text_z = (current_pos + 0.01) * z_stretch_factor
                ax.text(-primary_physics.proj_radius * 3.0, primary_physics.proj_radius * 1.5, stretched_text_z, 
                       altitude_text, 
                       color='cyan', fontsize=12, fontweight='bold', 
                       ha='right', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                
                # Velocity arrow (pretty 3D arrow) with vertical stretch
                if current_velocity > 0:
                    vector_length = min(0.08, max(0.02, current_velocity * 0.0005)) * z_stretch_factor
                    arrow_start_z = (current_pos + primary_physics.proj_length/2) * z_stretch_factor
                    # Create pretty arrow using Arrow3D
                    arrow = Arrow3D(
                        [0, 0], [0, 0],
                        [arrow_start_z, arrow_start_z + vector_length],
                        mutation_scale=20, lw=3, arrowstyle='-|>', color='blue', alpha=0.9
                    )
                    ax.add_artist(arrow)
                
                # Add velocity trail (show recent positions) with vertical stretch
                if frame > 15:
                    trail_length = min(frame, 20)
                    trail_positions = anim_positions[frame-trail_length:frame]
                    trail_colors = np.linspace(0.3, 1.0, len(trail_positions))
                    trail_sizes = np.linspace(5, 20, len(trail_positions))
                    
                    for i, (pos, alpha, size) in enumerate(zip(trail_positions[:-1], trail_colors[:-1], trail_sizes[:-1])):
                        if pos > -0.1:  # Only show trail when projectile is visible
                            stretched_trail_z = pos * z_stretch_factor
                            ax.scatter([0], [0], [stretched_trail_z], color='orange', alpha=alpha, s=size)
            
            # Add stage indicators for multi-stage with current-based activation
            if self.is_multi_stage and len(coil_geometries) > 1:
                for i, coil_geom in enumerate(coil_geometries):
                    stage_num = coil_geom['stage_num']
                    stage_current = self.get_stage_current_at_time(stage_num, sim_time_current, current_pos)
                    
                    # Stage is "active" if it has any significant current (lowered threshold)
                    stage_active = abs(stage_current) > 1.0  # Lowered from 10A
                    
                    # More detailed status display
                    if abs(stage_current) > 100:
                        indicator_color = 'red'
                        status = 'HIGH'
                    elif abs(stage_current) > 10:
                        indicator_color = 'orange'
                        status = 'MED'
                    elif abs(stage_current) > 1:
                        indicator_color = 'yellow'
                        status = 'LOW'
                    else:
                        indicator_color = 'gray'
                        status = 'OFF'
                    
                    indicator_alpha = 1.0 if stage_active else 0.4
                    
                    # Stage status label - positioned on the opposite side with vertical stretch
                    center_z = (coil_geom['z_offset'] + coil_geom['length'] / 2) * z_stretch_factor
                    label_radius = coil_geom['outer_radius'] * -1.8  # Negative for opposite side
                    ax.text(label_radius, 0, center_z, 
                           f'S{stage_num}: {status}', 
                           color=indicator_color, alpha=indicator_alpha,
                           fontsize=10, fontweight='bold', ha='center')
            
            # Set labels and title
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            
            # Real-time tracking and information panel
            current_real_time = time.time() - self.real_time_start
            
            # Clear info panel and add comprehensive information
            info_ax.clear()
            info_ax.axis('off')
            
            # Information panel content
            info_text = []
            
            # Timing information - swapped sim and real time positions
            info_text.append("=== TIMING INFORMATION ===")
            info_text.append(f"Sim Time: {sim_time_current:.4f}s")
            info_text.append(f"Real Time: {current_real_time:.2f}s")
            info_text.append(f"Time Scale: {sim_time_current/max(current_real_time, 0.001):.1f}x")
            info_text.append("")
            
            # Coil Array Specifications
            info_text.append("=== COIL ARRAY SPECS ===")
            info_text.append(f"Stages: {coil_specs['total_stages']}")
            info_text.append(f"Inner Ø: {coil_specs['inner_diameter']:.1f}mm")
            info_text.append(f"Outer Ø: {coil_specs['outer_diameter']:.1f}mm")
            info_text.append(f"Total Height: {coil_specs['total_height']:.1f}mm")
            info_text.append(f"Est. Mass: {coil_specs['total_mass_estimate']:.0f}g")
            info_text.append("")
            
            # Current projectile status
            if current_pos > -0.1:
                projectile_data = self.create_3d_projectile_geometry(current_pos, stage_physics_list[0])
                info_text.append("=== PROJECTILE STATUS ===")
                material_name = getattr(stage_physics_list[0], 'config', {}).get('projectile', {}).get('material', 'Unknown')
                info_text.append(f"Material: {material_name}")
                info_text.append(f"Ø: {projectile_data['dimensions']['diameter']:.1f}mm")
                info_text.append(f"Length: {projectile_data['dimensions']['length']:.1f}mm")
                info_text.append(f"Mass: {projectile_data['dimensions']['mass']:.1f}g")
                info_text.append(f"Velocity: {current_velocity:.1f} m/s")
                
                # Calculate kinetic energy
                kinetic_energy = 0.5 * (projectile_data['dimensions']['mass']/1000) * current_velocity**2
                info_text.append(f"KE: {kinetic_energy:.2f} J")
                info_text.append("")
            
            # Stage details for multi-stage - show all stages
            if self.is_multi_stage and len(coil_specs['stage_specs']) > 1:
                info_text.append("=== STAGE DETAILS ===")
                for stage_spec in coil_specs['stage_specs']:  # Show all stages
                    stage_current = self.get_stage_current_at_time(stage_spec['stage_num'], sim_time_current, current_pos)
                    status = "ACTIVE" if abs(stage_current) > 10 else "STANDBY"
                    info_text.append(f"Stage {stage_spec['stage_num']}: {status}")
                    info_text.append(f"  {stage_spec['length_mm']:.0f}mm, AWG{stage_spec['wire_gauge']}")
                    info_text.append(f"  {stage_current:.0f}A / {stage_spec['max_voltage_v']}V")
            
            # Display info text
            info_ax.text(0.05, 0.95, '\n'.join(info_text), 
                        transform=info_ax.transAxes, 
                        verticalalignment='top', horizontalalignment='left',
                        fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            # Dynamic title with current-based active stage detection
            if self.is_multi_stage:
                # Find which stage has the highest current (most active)
                max_current_stage = 1
                max_current_value = 0
                for i, coil_geom in enumerate(coil_geometries):
                    stage_num = coil_geom['stage_num']
                    stage_current = self.get_stage_current_at_time(stage_num, sim_time_current, current_pos)
                    if abs(stage_current) > max_current_value:
                        max_current_value = abs(stage_current)
                        max_current_stage = stage_num
                
                if max_current_value > 10:
                    title = f'Multi-Stage Coilgun Animation - Active: Stage {max_current_stage} ({max_current_value:.0f}A)'
                else:
                    title = 'Multi-Stage Coilgun Animation - No Active Stage'
            else:
                title = '3D Coilgun Animation'
            
            ax.set_title(f'{title}\nSim: {sim_time_current:.4f}s | Real: {current_real_time:.2f}s | Velocity: {current_velocity:.1f} m/s',
                        fontsize=12, fontweight='bold')
            
            # Set axis limits with vertical stretch for better coil separation
            zoom_factor = 0.8  # Balanced zoom for good grid coverage
            ax.set_xlim([-max_radius*zoom_factor, max_radius*zoom_factor])
            ax.set_ylim([-max_radius*zoom_factor, max_radius*zoom_factor])
            # Vertically stretched Z axis for better coil stage separation
            z_margin = 0.06  # Base margin  
            z_min = -z_margin * z_stretch_factor  # Stretched lower bound
            z_max = (total_height + z_margin) * z_stretch_factor  # Stretched upper bound
            ax.set_zlim([z_min, z_max])
            
            # Set aspect ratio to maintain proportional stretching
            ax.set_box_aspect([1, 1, z_stretch_factor])
            
            # Debug axis limits on first frame
            if frame == 0:
                print(f"Axis limits: Z=[{z_min:.3f}, {z_max:.3f}], total_height={total_height:.3f}")
                print(f"Enhanced vertical stretch factor: {z_stretch_factor}x for maximum coil separation")
                print(f"This focuses on coil region with vertical stretching (max_z={max_z:.1f})")
            
            # Dynamic camera view with side profile for better coil cross-section viewing
            base_azim = 35 + frame * 0.12  # Keep original azimuth rotation
            elev = 20 + 2 * np.sin(frame * 0.025)  # 30% more normal to coil (lower for side profile)
            ax.view_init(elev=elev, azim=base_azim)
            
            # Closer camera distance for better detail
            ax.dist = 6  # Closer camera distance for good detail
            
            # Progress indicator
            if frame % 30 == 0:
                progress = (frame / total_frames) * 100
                print(f"Animation progress: {progress:.1f}%")
        
        # Create animation
        print("Generating animation frames...")
        anim = FuncAnimation(fig, animate_frame, frames=total_frames, 
                           interval=1000/self.fps, blit=False, repeat=True)
        
        # Save if requested
        if save_path:
            print(f"Saving animation to: {save_path}")
            if save_path.endswith('.gif'):
                writer = PillowWriter(fps=self.fps)
                anim.save(save_path, writer=writer, dpi=300)
            else:
                anim.save(save_path, fps=self.fps, dpi=300)
            print("Animation saved!")
        
        # Show if requested
        if show_animation:
            plt.tight_layout()
            # Adjust subplot spacing to give more room to 3D plot
            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.15)
            plt.show()
        
        return anim


def find_results_directories():
    """Find simulation results directories."""
    current_dir = Path(".")
    result_dirs = []
    
    for item in current_dir.iterdir():
        if item.is_dir() and item.name.startswith('results_'):
            config_file = item / "simulation_config.json"
            if config_file.exists():
                result_dirs.append(item)
    
    return sorted(result_dirs)


def select_results_directory():
    """Interactive selection of results directory."""
    result_dirs = find_results_directories()
    
    if not result_dirs:
        print("No simulation results found. Run 'python solve.py' first.")
        sys.exit(1)
    
    print("Available simulation results:")
    print("-" * 40)
    
    for i, results_dir in enumerate(result_dirs, 1):
        print(f"{i}. {results_dir.name}")
    
    while True:
        try:
            choice = input(f"Select results (1-{len(result_dirs)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(result_dirs):
                return str(result_dirs[choice_num - 1])
            else:
                print(f"Please enter 1-{len(result_dirs)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


def create_demo_animation():
    """Create a demo animation with sample data if no results are available."""
    print("Creating demo 3D coilgun animation...")
    
    # This would create a sample animation - for now just inform user
    print("Demo mode: Please run 'python solve.py' first to generate simulation results.")
    print("Then run 'python animate.py' to create 3D animations from real data.")
    return None


def main():
    """Main function for 3D animation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced 3D Coilgun Animation - Realistic copper coils with vertical stacking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python animate.py                          # Interactive mode - select from available results
  python animate.py results_test_config      # Use specific results directory
  python animate.py --save animation.gif     # Save animation as GIF
  python animate.py --duration 15 --fps 20   # Custom duration and frame rate
  
Features:
  - Realistic 3D helical copper coil rendering (exactly like view.py)
  - Vertically stacked multi-stage coils with proper 3cm spacing
  - Current-dependent copper color intensity and glow effects
  - Correctly dimensioned cylindrical projectile
  - Magnetic field lines synchronized with current
  - Extended trajectory showing final projectile velocity
  - Dynamic camera movement and stage indicators
        """
    )
    
    parser.add_argument('results_dir', nargs='?', help='Results directory path')
    parser.add_argument('--save', type=str, help='Save animation file (supports .gif, .mp4)')
    parser.add_argument('--duration', type=float, default=15.0, 
                       help='Animation duration in seconds (default: 15.0 - slower for better visibility)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Frames per second (default: 30, higher = smoother but larger files)')
    parser.add_argument('--demo', action='store_true', 
                       help='Create demo animation (for testing without results)')
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("ADVANCED 3D COILGUN ANIMATION SUITE")
        print("=" * 70)
        print("Features: Realistic Copper Coils | Vertical Stacking | Correct Projectile Dimensions")
        print()
        
        # Demo mode
        if args.demo:
            create_demo_animation()
            return
        
        # Select results directory
        if args.results_dir:
            results_dir = args.results_dir
            if not Path(results_dir).exists():
                print(f"Error: Results directory '{results_dir}' not found.")
                print("Available results directories:")
                for dir_name in find_results_directories():
                    print(f"  - {dir_name.name}")
                sys.exit(1)
        else:
            results_dir = select_results_directory()
        
        print(f"Selected results: {Path(results_dir).name}")
        
        # Show animation configuration
        print(f"\nAnimation Configuration:")
        print(f"  Duration: {args.duration:.1f} seconds")
        print(f"  Frame rate: {args.fps} FPS")
        print(f"  Total frames: {int(args.duration * args.fps)}")
        if args.save:
            print(f"  Output file: {args.save}")
        else:
            print(f"  Output: Interactive display only")
        
        # Confirm before proceeding
        print(f"\nThis will create a detailed 3D animation showing:")
        print(f"  ✓ Realistic helical copper coil windings (like view.py)")
        print(f"  ✓ Vertically stacked multi-stage coils with 3cm spacing")
        print(f"  ✓ Current-dependent copper color intensity and glow")
        print(f"  ✓ Correctly dimensioned cylindrical projectile")
        print(f"  ✓ Magnetic field lines synchronized with current flow")
        print(f"  ✓ Extended projectile trajectory beyond coil exit")
        
        proceed = input("\nProceed with animation creation? (Y/n): ").strip().lower()
        if proceed in ['n', 'no', 'q', 'quit']:
            print("Animation cancelled.")
            return
        
        # Create visualizer
        print("\nInitializing 3D visualizer...")
        visualizer = Advanced3DCoilgunVisualizer()
        visualizer.animation_duration = args.duration
        visualizer.fps = args.fps
        
        # Load data and create animation
        print("Loading simulation data...")
        visualizer.load_simulation_data(results_dir)
        
        print("Creating 3D animation...")
        anim = visualizer.create_animation(save_path=args.save, show_animation=True)
        
        print("\n" + "="*50)
        print("3D ANIMATION COMPLETE!")
        print("="*50)
        
        if args.save:
            print(f"Animation saved to: {args.save}")
            if Path(args.save).exists():
                file_size_mb = Path(args.save).stat().st_size / (1024*1024)
                print(f"File size: {file_size_mb:.1f} MB")
        
        print("\nFor more options, run: python animate.py --help")
        
    except KeyboardInterrupt:
        print("\n\nAnimation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAnimation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure simulation results exist (run 'python solve.py' first)")
        print("2. Check that the results directory contains time series data")
        print("3. For multi-stage simulations, ensure all stage directories exist")
        sys.exit(1)


if __name__ == "__main__":
    main() 