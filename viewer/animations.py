"""
Animation functions for coilgun visualization.

This module provides animation capabilities for projectile motion,
field evolution, and dynamic simulation visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Optional, Dict, Any

from .core import BaseVisualizer, CoilGeometry, VISUALIZATION_CONSTANTS
from .fields import MagneticFieldCalculator
from .plots import ContourPlotter


class AnimationEngine(BaseVisualizer):
    """Class for creating animated visualizations."""
    
    def animate_3d_projectile_motion(self, simulation_results, save_path=None, 
                                   interval=100, show_field=True):
        """
        Create 3D animation of projectile motion with magnetic field.
        
        Args:
            simulation_results: Simulation results dictionary
            save_path: Path to save animation
            interval: Animation interval in ms
            show_field: Whether to show magnetic field
        """
        if 'time' not in simulation_results or 'position' not in simulation_results:
            print("Insufficient data for projectile motion animation")
            return
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract simulation data
        time_data = simulation_results['time']
        position_data = simulation_results['position']
        
        if 'current' in simulation_results:
            current_data = simulation_results['current']
        else:
            current_data = [100] * len(time_data)  # Default current
        
        # Set up initial plot
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Projectile Motion Animation')
        
        # Set plot limits
        max_pos = max(position_data) if position_data else 0.1
        ax.set_xlim(-0.05, 0.05)
        ax.set_ylim(-0.05, 0.05)
        ax.set_zlim(-0.02, max_pos * 1.2)
        
        # Animation function
        def animate(frame_idx):
            ax.clear()
            
            if frame_idx >= len(position_data):
                return
            
            # Current frame data
            current_time = time_data[frame_idx]
            current_position = position_data[frame_idx]
            current_current = current_data[frame_idx] if frame_idx < len(current_data) else 0
            
            # Add 3D coil geometry
            self._add_coil_geometry_3d(ax)
            
            # Add projectile at current position
            self._add_projectile_geometry_3d(ax, current_position)
            
            # Add magnetic field visualization if requested
            if show_field and self.physics:
                self._add_simplified_field_3d(ax, current_current)
            
            # Add trajectory trail
            trail_length = min(50, frame_idx)
            if trail_length > 1:
                trail_positions = position_data[max(0, frame_idx-trail_length):frame_idx+1]
                trail_x = [0] * len(trail_positions)  # On axis
                trail_y = [0] * len(trail_positions)  # On axis
                ax.plot(trail_x, trail_y, trail_positions, 'r--', alpha=0.6, linewidth=2)
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Projectile Motion - t={current_time*1000:.1f}ms, I={current_current:.1f}A')
            
            # Set consistent limits
            ax.set_xlim(-0.05, 0.05)
            ax.set_ylim(-0.05, 0.05)
            ax.set_zlim(-0.02, max_pos * 1.2)
        
        # Create animation
        frames = len(position_data)
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, repeat=True)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=10)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def animate_field_evolution(self, simulation_results, save_path=None, 
                              interval=50, show_contours=True):
        """
        Create 2D animation of magnetic field evolution during simulation.
        
        Args:
            simulation_results: Simulation results dictionary
            save_path: Path to save animation
            interval: Animation interval in ms
            show_contours: Whether to show field contours
        """
        if 'time' not in simulation_results or 'current' not in simulation_results:
            print("Insufficient data for field evolution animation")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Magnetic Field Evolution Animation', fontsize=14)
        
        # Extract simulation data
        time_data = simulation_results['time']
        current_data = simulation_results['current']
        position_data = simulation_results.get('position', [0] * len(time_data))
        
        # Pre-calculate field data for different currents
        field_calculator = MagneticFieldCalculator(self.physics)
        
        # Create coordinate grid
        z_range = VISUALIZATION_CONSTANTS['DEFAULT_Z_RANGE']
        r_range = VISUALIZATION_CONSTANTS['DEFAULT_R_RANGE']
        
        def create_safe_levels(data, num_levels=15):
            data_min, data_max = np.min(data), np.max(data)
            if np.abs(data_max - data_min) < 1e-15:
                if np.abs(data_max) < 1e-15:
                    levels = np.linspace(-1e-10, 1e-10, num_levels)
                else:
                    eps = np.abs(data_max) * 1e-6
                    levels = np.linspace(data_max - eps, data_max + eps, num_levels)
            else:
                levels = np.linspace(data_min, data_max, num_levels)
            return levels
        
        def animate(frame_idx):
            if frame_idx >= len(current_data):
                return
            
            ax1.clear()
            ax2.clear()
            
            # Current frame data
            current_time = time_data[frame_idx]
            current_current = current_data[frame_idx]
            current_position = position_data[frame_idx] if frame_idx < len(position_data) else 0
            
            # Calculate field for current frame
            field_data = field_calculator.calculate_bfield_map_2d(
                current_current, z_range, r_range, num_z=50, num_r=30
            )
            
            Z, R = field_data['Z'], field_data['R']
            Bz = field_data['Bz']
            B_magnitude = field_data['B_magnitude']
            
            # Convert to mm for display
            Z_mm = Z * 1000
            R_mm = R * 1000
            
            # Plot 1: Axial field component
            if show_contours:
                levels = create_safe_levels(Bz, num_levels=15)
                im1 = ax1.contourf(Z_mm, R_mm, Bz, levels=levels, cmap='RdBu_r')
                ax1.contour(Z_mm, R_mm, Bz, levels=levels, colors='black', 
                           linewidths=0.5, alpha=0.3)
            else:
                im1 = ax1.imshow(Bz, extent=[Z_mm.min(), Z_mm.max(), R_mm.min(), R_mm.max()],
                               aspect='auto', cmap='RdBu_r', origin='lower')
            
            ax1.set_xlabel('Z position (mm)')
            ax1.set_ylabel('R position (mm)')
            ax1.set_title(f'Axial Field (Bz) - I={current_current:.1f}A')
            
            # Plot 2: Field magnitude
            if show_contours:
                levels_mag = create_safe_levels(B_magnitude, num_levels=15)
                im2 = ax2.contourf(Z_mm, R_mm, B_magnitude, levels=levels_mag, cmap='plasma')
                ax2.contour(Z_mm, R_mm, B_magnitude, levels=levels_mag, colors='black',
                           linewidths=0.5, alpha=0.3)
            else:
                im2 = ax2.imshow(B_magnitude, extent=[Z_mm.min(), Z_mm.max(), R_mm.min(), R_mm.max()],
                               aspect='auto', cmap='plasma', origin='lower')
            
            ax2.set_xlabel('Z position (mm)')
            ax2.set_ylabel('R position (mm)')
            ax2.set_title(f'Field Magnitude |B| - t={current_time*1000:.1f}ms')
            
            # Add coil and projectile to both plots
            for ax in [ax1, ax2]:
                CoilGeometry.add_coil_boundaries(ax, self.physics)
                CoilGeometry.add_projectile_marker(ax, current_position * 1000)
                ax.grid(True, alpha=0.3)
        
        # Create animation
        frames = len(current_data)
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, repeat=True)
        
        if save_path:
            print(f"Saving field evolution animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def animate_circuit_response(self, simulation_results, save_path=None, interval=50):
        """
        Create animation of circuit response (current, voltage, power).
        
        Args:
            simulation_results: Simulation results dictionary
            save_path: Path to save animation
            interval: Animation interval in ms
        """
        if 'time' not in simulation_results:
            print("No time data for circuit response animation")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Circuit Response Animation', fontsize=14)
        
        time_data = simulation_results['time']
        current_data = simulation_results.get('current', [])
        voltage_data = simulation_results.get('voltage', [])
        position_data = simulation_results.get('position', [])
        velocity_data = simulation_results.get('velocity', [])
        
        def animate(frame_idx):
            if frame_idx >= len(time_data):
                return
            
            # Clear all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Current frame slice
            end_idx = frame_idx + 1
            t_slice = np.array(time_data[:end_idx]) * 1000  # Convert to ms
            
            # Plot 1: Current vs Time
            if current_data:
                i_slice = current_data[:end_idx]
                ax1.plot(t_slice, i_slice, 'b-', linewidth=2)
                ax1.set_ylabel('Current (A)')
                ax1.set_title('Current Response')
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Voltage vs Time
            if voltage_data:
                v_slice = voltage_data[:end_idx]
                ax2.plot(t_slice, v_slice, 'r-', linewidth=2)
                ax2.set_ylabel('Voltage (V)')
                ax2.set_title('Voltage Response')
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Position vs Time
            if position_data:
                pos_slice = np.array(position_data[:end_idx]) * 1000  # Convert to mm
                ax3.plot(t_slice, pos_slice, 'g-', linewidth=2)
                ax3.set_xlabel('Time (ms)')
                ax3.set_ylabel('Position (mm)')
                ax3.set_title('Projectile Position')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Velocity vs Time
            if velocity_data:
                vel_slice = velocity_data[:end_idx]
                ax4.plot(t_slice, vel_slice, 'm-', linewidth=2)
                ax4.set_xlabel('Time (ms)')
                ax4.set_ylabel('Velocity (m/s)')
                ax4.set_title('Projectile Velocity')
                ax4.grid(True, alpha=0.3)
            
            # Add current time marker
            current_time = time_data[frame_idx] * 1000
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(current_time, color='red', linestyle='--', alpha=0.7)
        
        # Create animation
        frames = len(time_data)
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, repeat=True)
        
        if save_path:
            print(f"Saving circuit animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _add_coil_geometry_3d(self, ax):
        """Add simplified 3D coil geometry for animation."""
        if not self.physics:
            return
        
        inner_radius = self.physics.coil_inner_radius
        outer_radius = self.physics.coil_outer_radius
        coil_length = self.physics.coil_length
        
        # Create coil cylinder representation
        theta = np.linspace(0, 2*np.pi, 20)
        z_coil = [0, coil_length]
        
        for z in z_coil:
            x_inner = inner_radius * np.cos(theta)
            y_inner = inner_radius * np.sin(theta)
            z_inner = [z] * len(theta)
            
            x_outer = outer_radius * np.cos(theta)
            y_outer = outer_radius * np.sin(theta)
            z_outer = [z] * len(theta)
            
            ax.plot(x_inner, y_inner, z_inner, 'b-', alpha=0.5)
            ax.plot(x_outer, y_outer, z_outer, 'b-', alpha=0.5)
        
        # Connect inner and outer
        for i in range(0, len(theta), 4):
            x_connect = [inner_radius * np.cos(theta[i]), outer_radius * np.cos(theta[i])]
            y_connect = [inner_radius * np.sin(theta[i]), outer_radius * np.sin(theta[i])]
            z_connect = [0, 0]
            ax.plot(x_connect, y_connect, z_connect, 'b-', alpha=0.3)
    
    def _add_projectile_geometry_3d(self, ax, position):
        """Add simplified 3D projectile geometry for animation."""
        if not self.physics:
            return
        
        projectile_radius = getattr(self.physics, 'projectile_radius', 0.002)
        projectile_length = getattr(self.physics, 'projectile_length', 0.01)
        
        # Create simple cylindrical projectile
        theta = np.linspace(0, 2*np.pi, 12)
        x_proj = projectile_radius * np.cos(theta)
        y_proj = projectile_radius * np.sin(theta)
        
        # Front and back circles
        z_front = [position] * len(theta)
        z_back = [position + projectile_length] * len(theta)
        
        ax.plot(x_proj, y_proj, z_front, 'r-', linewidth=2)
        ax.plot(x_proj, y_proj, z_back, 'r-', linewidth=2)
        
        # Connect front and back
        for i in range(0, len(theta), 3):
            x_connect = [x_proj[i], x_proj[i]]
            y_connect = [y_proj[i], y_proj[i]]
            z_connect = [position, position + projectile_length]
            ax.plot(x_connect, y_connect, z_connect, 'r-', linewidth=1)
    
    def _add_simplified_field_3d(self, ax, current):
        """Add simplified 3D field visualization for animation."""
        if not self.physics or current <= 0:
            return
        
        # Create simple field line representation
        coil_radius = (self.physics.coil_inner_radius + self.physics.coil_outer_radius) / 2
        
        # Simple axial field lines
        z_line = np.linspace(-0.02, self.physics.coil_length + 0.05, 20)
        
        for r_frac in [0.5, 0.8, 1.2]:
            r = coil_radius * r_frac
            x_line = [r] * len(z_line)
            y_line = [0] * len(z_line)
            
            # Scale alpha with current
            alpha = min(0.8, current / 200)
            ax.plot(x_line, y_line, z_line, 'orange', alpha=alpha, linewidth=1) 