"""
2D plotting functions for coilgun visualization.

This module provides 2D plotting capabilities including magnetic field contours,
force maps, field profiles, and other 2D visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize
from typing import Optional, Dict, Any, List

from .core import BaseVisualizer, CoilGeometry
from .fields import MagneticFieldCalculator


class ContourPlotter(BaseVisualizer):
    """Class for creating 2D contour plots of magnetic fields."""
    
    def plot_bfield_contours(self, field_data, save_path=None, show_coil=True, 
                            show_projectile=True, projectile_position=None):
        """
        Create magnetic field contour plots.
        
        Args:
            field_data: Field data dictionary from calculate_bfield_map_2d
            save_path: Path to save the plot
            show_coil: Whether to show coil boundaries
            show_projectile: Whether to show projectile marker
            projectile_position: Projectile position (m)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle('Magnetic Field Distribution', fontsize=16, fontweight='bold')
        
        Z, R = field_data['Z'], field_data['R']
        Bz, Br = field_data['Bz'], field_data['Br']
        B_magnitude = field_data['B_magnitude']
        
        # Convert to millimeters for display
        Z_mm = Z * 1000
        R_mm = R * 1000
        
        # Function to create safe contour levels
        def create_safe_levels(data, num_levels=20):
            """Create contour levels that are guaranteed to be increasing."""
            data_min, data_max = np.min(data), np.max(data)
            
            # If data is essentially constant, create small artificial range
            if np.abs(data_max - data_min) < 1e-15:
                if np.abs(data_max) < 1e-15:
                    # Data is all zeros
                    levels = np.linspace(-1e-10, 1e-10, num_levels)
                else:
                    # Data is constant but non-zero
                    eps = np.abs(data_max) * 1e-6
                    levels = np.linspace(data_max - eps, data_max + eps, num_levels)
            else:
                # Normal case with varying data
                levels = np.linspace(data_min, data_max, num_levels)
            
            return levels
        
        # 1. Axial field component (Bz)
        levels_z = create_safe_levels(Bz)
        im1 = ax1.contourf(Z_mm, R_mm, Bz, levels=levels_z, cmap='RdBu_r')
        ax1.contour(Z_mm, R_mm, Bz, levels=levels_z, colors='black', linewidths=0.5, alpha=0.3)
        plt.colorbar(im1, ax=ax1, label='Bz (T)')
        self.apply_common_styling(ax1, 'Axial Field Component (Bz)', 'Z position (mm)', 'R position (mm)')
        
        # Add field strength info
        bz_max = np.max(np.abs(Bz))
        ax1.text(0.02, 0.98, f'Max |Bz|: {bz_max:.2e} T', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        # 2. Radial field component (Br)
        levels_r = create_safe_levels(Br)
        im2 = ax2.contourf(Z_mm, R_mm, Br, levels=levels_r, cmap='RdBu_r')
        ax2.contour(Z_mm, R_mm, Br, levels=levels_r, colors='black', linewidths=0.5, alpha=0.3)
        plt.colorbar(im2, ax=ax2, label='Br (T)')
        self.apply_common_styling(ax2, 'Radial Field Component (Br)', 'Z position (mm)', 'R position (mm)')
        
        # Add field strength info
        br_max = np.max(np.abs(Br))
        ax2.text(0.02, 0.98, f'Max |Br|: {br_max:.2e} T', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        # 3. Field magnitude
        levels_mag = create_safe_levels(B_magnitude)
        im3 = ax3.contourf(Z_mm, R_mm, B_magnitude, levels=levels_mag, cmap='plasma')
        ax3.contour(Z_mm, R_mm, B_magnitude, levels=levels_mag, colors='black', linewidths=0.5, alpha=0.3)
        plt.colorbar(im3, ax=ax3, label='|B| (T)')
        self.apply_common_styling(ax3, 'Field Magnitude |B|', 'Z position (mm)', 'R position (mm)')
        
        # Add field strength info
        b_max = np.max(B_magnitude)
        ax3.text(0.02, 0.98, f'Max |B|: {b_max:.2e} T', transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        # 4. Field vectors
        # Subsample for cleaner vector plot
        step = max(1, Z.shape[1] // 15)
        Z_sub, R_sub = Z[::step, ::step], R[::step, ::step]
        Bz_sub, Br_sub = Bz[::step, ::step], Br[::step, ::step]
        B_mag_sub = B_magnitude[::step, ::step]
        
        # Only plot vectors where field is significant
        mask = B_mag_sub > np.max(B_magnitude) * 0.01
        
        if np.any(mask):
            ax4.quiver(Z_sub[mask] * 1000, R_sub[mask] * 1000, 
                      Bz_sub[mask], Br_sub[mask], 
                      B_mag_sub[mask], cmap='viridis', scale=None, width=0.003)
        
        self.apply_common_styling(ax4, 'Field Vector Plot', 'Z position (mm)', 'R position (mm)')
        
        # Add coil and projectile to all plots
        for ax in [ax1, ax2, ax3, ax4]:
            if show_coil:
                CoilGeometry.add_coil_boundaries(ax, self.physics)
            if show_projectile and projectile_position is not None:
                CoilGeometry.add_projectile_marker(ax, projectile_position * 1000)  # Convert to mm
        
        plt.tight_layout()
        self.save_figure(fig, save_path)
        plt.show()
    
    def plot_onaxis_field_profile(self, field_data=None, current_values=None, save_path=None):
        """
        Plot magnetic field profile along the coil axis.
        
        Args:
            field_data: Field profile data from calculate_onaxis_field_profile
            current_values: Current values for multiple profiles
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.fig_size)
        fig.suptitle('On-Axis Magnetic Field Profile', fontsize=16, fontweight='bold')
        
        if field_data is None:
            # Calculate field data
            calculator = MagneticFieldCalculator(self.physics)
            field_data = calculator.calculate_onaxis_field_profile(current_values)
        
        z_vals = field_data['z_vals'] * 1000  # Convert to mm
        field_profiles = field_data['field_profiles']
        current_values = field_data['current_values']
        
        # Plot multiple current profiles
        for current in current_values:
            bz_profile = field_profiles[current]
            ax1.plot(z_vals, bz_profile, linewidth=2, label=f'{current} A', alpha=0.8)
        
        self.apply_common_styling(ax1, 'Axial Field vs Position', 'Z position (mm)', 'Bz (T)')
        ax1.legend()
        
        # Plot field gradient (force indicator)
        if len(current_values) > 0:
            current = current_values[-1]  # Use highest current for gradient
            bz_profile = field_profiles[current]
            z_vals_gradient = z_vals[1:-1]  # Remove endpoints for gradient calc
            gradient = np.gradient(bz_profile)[1:-1]  # Field gradient
            
            ax2.plot(z_vals_gradient, gradient, 'r-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax2, f'Field Gradient (Force Indicator) at {current} A', 
                                    'Z position (mm)', 'dBz/dz (T/m)')
        
        # Add coil boundaries
        for ax in [ax1, ax2]:
            if self.physics:
                coil_start = 0
                coil_end = self.physics.coil_length * 1000  # Convert to mm
                ax.axvspan(coil_start, coil_end, alpha=0.2, color='lightblue', label='Coil')
        
        plt.tight_layout()
        self.save_figure(fig, save_path)
        plt.show()


class ForcePlotter(BaseVisualizer):
    """Class for plotting force maps and force-related visualizations."""
    
    def plot_force_current_map(self, simulation_results, save_path=None):
        """
        Create enhanced force vs current/position map using actual simulation data.
        
        Args:
            simulation_results: Simulation results dictionary with actual data
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Force Analysis from Actual Simulation Data', fontsize=16, fontweight='bold')
        
        # Handle both direct data and time_series structure
        if 'time_series' in simulation_results:
            time_data = np.array(simulation_results['time_series']['time'])
            current_data = np.array(simulation_results['time_series']['current'])
            force_data = np.array(simulation_results['time_series']['force_total'])
            position_data = np.array(simulation_results['time_series']['position'])
            velocity_data = np.array(simulation_results['time_series']['velocity'])
            energy_data = np.array(simulation_results['time_series']['energy_capacitor'])
        elif 'time' in simulation_results:
            time_data = np.array(simulation_results['time'])
            current_data = np.array(simulation_results.get('current', []))
            force_data = np.array(simulation_results.get('force', []))
            position_data = np.array(simulation_results.get('position', []))
            velocity_data = np.array(simulation_results.get('velocity', []))
            energy_data = np.array(simulation_results.get('energy_capacitor', []))
        else:
            print("No time data available for force mapping")
            return
        
        if len(time_data) == 0:
            print("Empty time data array")
            return
        
        t_ms = time_data * 1000  # Convert to milliseconds
        
        # 1. Force vs Time
        if len(force_data) > 0:
            ax1.plot(t_ms, force_data, 'b-', linewidth=2, alpha=0.8, label=f'Max: {np.max(force_data):.2f} N')
            self.apply_common_styling(ax1, 'Force vs Time (Actual Data)', 'Time (ms)', 'Force (N)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Current vs Time
        if len(current_data) > 0:
            ax2.plot(t_ms, current_data, 'r-', linewidth=2, alpha=0.8, label=f'Max: {np.max(current_data):.2f} A')
            self.apply_common_styling(ax2, 'Current vs Time (Actual Data)', 'Time (ms)', 'Current (A)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Force vs Position
        if len(force_data) > 0 and len(position_data) > 0:
            position_mm = position_data * 1000  # Convert to mm
            ax3.plot(position_mm, force_data, 'g-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax3, 'Force vs Position (Actual Data)', 'Position (mm)', 'Force (N)')
            ax3.grid(True, alpha=0.3)
            
            # Add position range info
            pos_range = f'Range: {np.min(position_mm):.2f} to {np.max(position_mm):.2f} mm' if len(position_mm) > 0 else 'Range: N/A'
            ax3.text(0.02, 0.98, pos_range, transform=ax3.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                    verticalalignment='top')
        
        # 4. Energy vs Time
        if len(energy_data) > 0:
            ax4.plot(t_ms, energy_data, 'm-', linewidth=2, alpha=0.8, label=f'Start: {energy_data[0]:.1f} J')
            self.apply_common_styling(ax4, 'Capacitor Energy vs Time (Actual Data)', 'Time (ms)', 'Energy (J)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        elif len(current_data) > 0 and len(force_data) > 0:
            # Calculate instantaneous power as alternative
            power = current_data * force_data * velocity_data if len(velocity_data) > 0 else current_data * force_data
            ax4.plot(t_ms, power, 'm-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax4, 'Power vs Time (Calculated)', 'Time (ms)', 'Power (W)')
            ax4.grid(True, alpha=0.3)
        
        # Add simulation statistics as text
        stats_text = f"""Simulation Statistics:
Duration: {np.max(time_data)*1000:.2f} ms if len(time_data) > 0 else 0:.2f
Max Current: {np.max(current_data):.2f} A if len(current_data) > 0 else 0:.2f
Max Force: {np.max(force_data):.2f} N if len(force_data) > 0 else 0:.2f
Max Velocity: {np.max(velocity_data):.2f} m/s if len(velocity_data) > 0 else 0:.2f
Data Points: {len(time_data)}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for stats
        self.save_figure(fig, save_path)
        plt.show()


class ProfilePlotter(BaseVisualizer):
    """Class for plotting various physics profiles and comparisons."""
    
    def plot_velocity_progression(self, summary_data, save_path=None):
        """
        Create velocity progression plot for multi-stage analysis.
        
        Args:
            summary_data: Summary data from multi-stage simulation
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)
        fig.suptitle('Velocity Progression Analysis', fontsize=16, fontweight='bold')
        
        if not summary_data:
            print("No summary data available for velocity progression")
            return
        
        stages = list(summary_data.keys())
        initial_velocities = [summary_data[stage].get('initial_velocity', 0) for stage in stages]
        final_velocities = [summary_data[stage].get('final_velocity', 0) for stage in stages]
        velocity_gains = [final - initial for initial, final in zip(initial_velocities, final_velocities)]
        
        # 1. Velocity progression bar chart
        x_pos = np.arange(len(stages))
        width = 0.35
        
        ax1.bar(x_pos - width/2, initial_velocities, width, label='Initial Velocity', alpha=0.7)
        ax1.bar(x_pos + width/2, final_velocities, width, label='Final Velocity', alpha=0.7)
        
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Velocity by Stage')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stages)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Velocity gain per stage
        ax2.bar(stages, velocity_gains, alpha=0.7, color='green')
        self.apply_common_styling(ax2, 'Velocity Gain per Stage', 'Stage', 'Velocity Gain (m/s)')
        
        plt.tight_layout()
        self.save_figure(fig, save_path)
        plt.show()
    
    def plot_efficiency_comparison(self, summary_data, save_path=None):
        """
        Create efficiency comparison plot.
        
        Args:
            summary_data: Summary data from simulation
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)
        fig.suptitle('Efficiency Analysis', fontsize=16, fontweight='bold')
        
        if not summary_data:
            print("No summary data available for efficiency analysis")
            return
        
        stages = list(summary_data.keys())
        efficiencies = [summary_data[stage].get('efficiency', 0) * 100 for stage in stages]
        energy_consumed = [summary_data[stage].get('energy_consumed', 0) for stage in stages]
        
        # 1. Efficiency by stage
        ax1.bar(stages, efficiencies, alpha=0.7, color='blue')
        ax1.set_ylabel('Efficiency (%)')
        ax1.set_title('Efficiency by Stage')
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy consumption
        ax2.bar(stages, energy_consumed, alpha=0.7, color='red')
        ax2.set_ylabel('Energy Consumed (J)')
        ax2.set_title('Energy Consumption by Stage')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, save_path)
        plt.show() 