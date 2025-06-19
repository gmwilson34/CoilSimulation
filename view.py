"""
Advanced Magnetic Field Visualization for Coilgun Simulation

This module provides comprehensive visualization of magnetic fields, force maps,
and dynamic simulation results using the physics equations from the main engine.

Features:
- 2D magnetic field contour plots
- 3D field surface plots
- 3D field line visualization
- 3D coil geometry and projectile rendering
- Force and inductance mapping
- Animation of field evolution during simulation
- Interactive 3D field exploration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import json
import sys
import signal
from scipy.integrate import odeint
from scipy.interpolate import griddata
from typing import Optional, Tuple, List, Any, Union

from equations import CoilgunPhysicsEngine
from solve import CoilgunSimulation

# Set up plotting style
plt.style.use('default')

# Optional seaborn import for enhanced styling
try:
    import seaborn as sns
    sns.set_palette("viridis")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class CoilgunFieldVisualizer:
    """
    Advanced magnetic field and force visualization for coilgun systems.
    Enhanced with new physics data visualization capabilities.
    """
    
    def __init__(self, physics_engine):
        """
        Initialize the visualizer with a physics engine.
        
        Args:
            physics_engine: CoilgunPhysicsEngine instance
        """
        self.physics = physics_engine
        self.fig_size = (15, 10)
    
    def plot_enhanced_physics_analysis(self, results, save_path=None):
        """
        Create comprehensive visualization of enhanced physics analysis results.
        
        Args:
            results: Results dictionary from enhanced simulation
            save_path: Path to save the plot
        """
        # Create figure with multiple subplots for comprehensive analysis
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Enhanced Electromagnetic Physics Analysis', fontsize=16, fontweight='bold')
        
        # Check if we have time data
        if 'time' not in results or len(results['time']) == 0:
            print("No time data available for enhanced physics analysis")
            plt.close()
            return
        
        t_ms = np.array(results['time']) * 1000  # Convert to ms
        
        # 1. Force decomposition plot
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Plot force components
        if 'force_gradient' in results:
            ax1.plot(t_ms, results['force_gradient'], 'b-', linewidth=2, label='Gradient Force', alpha=0.8)
        if 'force_reluctance' in results:
            ax1.plot(t_ms, results['force_reluctance'], 'g--', linewidth=2, label='Reluctance Force', alpha=0.8)
        if 'force_eddy' in results:
            ax1.plot(t_ms, results['force_eddy'], 'r:', linewidth=2, label='Eddy Current Force', alpha=0.8)
        if 'force_lorentz' in results:
            ax1.plot(t_ms, results['force_lorentz'], 'm-.', linewidth=1.5, label='Lorentz Force', alpha=0.8)
        if 'force_maxwell' in results:
            ax1.plot(t_ms, results['force_maxwell'], 'c--', linewidth=1.5, label='Maxwell Stress', alpha=0.8)
        if 'force_total' in results:
            ax1.plot(t_ms, results['force_total'], 'k-', linewidth=3, label='Total Force', alpha=0.9)
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Force Component Decomposition')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Eddy current analysis
        ax2 = fig.add_subplot(gs[0, 1])
        if 'eddy_current_magnitude' in results and np.any(results['eddy_current_magnitude'] > 0):
            ax2.plot(t_ms, results['eddy_current_magnitude'], 'r-', linewidth=2, label='Eddy Current')
            ax2_twin = ax2.twinx()
            if 'skin_depth' in results:
                finite_skin = np.where(results['skin_depth'] < 1e6, results['skin_depth'] * 1000, np.nan)
                ax2_twin.plot(t_ms, finite_skin, 'b--', linewidth=1.5, label='Skin Depth', alpha=0.7)
                ax2_twin.set_ylabel('Skin Depth (mm)', color='b')
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Eddy Current (A)', color='r')
        ax2.set_title('Eddy Current Effects')
        ax2.grid(True, alpha=0.3)
        
        # 3. Power loss analysis
        ax3 = fig.add_subplot(gs[0, 2])
        if 'power_loss_resistive' in results:
            ax3.plot(t_ms, results['power_loss_resistive'], 'g-', linewidth=2, label='Resistive Loss', alpha=0.8)
        if 'power_loss_eddy' in results:
            ax3.plot(t_ms, results['power_loss_eddy'], 'r-', linewidth=2, label='Eddy Loss', alpha=0.8)
        if 'power_mechanical' in results:
            ax3.plot(t_ms, results['power_mechanical'], 'b-', linewidth=2, label='Mechanical Power', alpha=0.8)
        
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power Analysis')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy conservation tracking
        ax4 = fig.add_subplot(gs[1, 0])
        if 'energy_capacitor' in results and 'energy_kinetic' in results:
            E_cap = results['energy_capacitor']
            E_kin = results['energy_kinetic']
            E_total = E_cap + E_kin
            
            ax4.plot(t_ms, E_cap, 'b-', linewidth=2, label='Capacitor Energy', alpha=0.8)
            ax4.plot(t_ms, E_kin, 'r-', linewidth=2, label='Kinetic Energy', alpha=0.8)
            ax4.plot(t_ms, E_total, 'k--', linewidth=2, label='Total Energy', alpha=0.8)
            
            # Add energy conservation error if available
            if 'energy_conservation' in results:
                ax4_twin = ax4.twinx()
                ax4_twin.plot(t_ms, results['energy_conservation'] * 100, 'orange', linewidth=1, alpha=0.6)
                ax4_twin.set_ylabel('Energy Error (%)', color='orange')
        
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Energy (J)')
        ax4.set_title('Energy Conservation')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Magnetic field and saturation analysis
        ax5 = fig.add_subplot(gs[1, 1])
        if 'magnetic_field' in results:
            ax5.plot(t_ms, results['magnetic_field'], 'purple', linewidth=2, label='B-field', alpha=0.8)
            
            # Add saturation analysis if available
            if 'saturation_factor' in results:
                ax5_twin = ax5.twinx()
                ax5_twin.plot(t_ms, results['saturation_factor'], 'orange', linewidth=1.5, label='Saturation Factor', alpha=0.7)
                ax5_twin.set_ylabel('Saturation Factor', color='orange')
                ax5_twin.set_ylim(0, 1.1)
        
        ax5.set_xlabel('Time (ms)')
        ax5.set_ylabel('Magnetic Field (T)')
        ax5.set_title('Magnetic Field & Saturation')
        ax5.grid(True, alpha=0.3)
        
        # 6. Frequency analysis
        ax6 = fig.add_subplot(gs[1, 2])
        if 'frequency_content' in results and np.any(results['frequency_content'] > 0):
            ax6.plot(t_ms, results['frequency_content'], 'cyan', linewidth=2, label='Peak Frequency', alpha=0.8)
            ax6.set_ylabel('Frequency (Hz)')
            ax6.set_title('Frequency Content Analysis')
        else:
            # Current frequency spectrum
            if 'current' in results and len(results['current']) > 100:
                current_data = results['current']
                dt = results['time'][1] - results['time'][0]
                freqs = np.fft.fftfreq(len(current_data), dt)
                fft_current = np.abs(np.fft.fft(current_data))
                
                # Plot positive frequencies only
                pos_freqs = freqs[freqs > 0][:len(freqs)//4]  # Show first quarter
                pos_fft = fft_current[freqs > 0][:len(freqs)//4]
                
                ax6.semilogy(pos_freqs, pos_fft, 'cyan', linewidth=1.5, alpha=0.8)
                ax6.set_ylabel('Current FFT Magnitude')
                ax6.set_title('Current Frequency Spectrum')
        
        ax6.set_xlabel('Time (ms)' if 'frequency_content' in results else 'Frequency (Hz)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Inductance and gradient analysis
        ax7 = fig.add_subplot(gs[2, 0])
        if 'inductance' in results:
            ax7.plot(t_ms, results['inductance'] * 1000, 'green', linewidth=2, label='Inductance', alpha=0.8)
            
            if 'inductance_gradient' in results:
                ax7_twin = ax7.twinx()
                ax7_twin.plot(t_ms, results['inductance_gradient'] * 1000, 'brown', linewidth=1.5, label='dL/dx', alpha=0.7)
                ax7_twin.set_ylabel('dL/dx (mH/m)', color='brown')
        
        ax7.set_xlabel('Time (ms)')
        ax7.set_ylabel('Inductance (mH)')
        ax7.set_title('Inductance & Gradient')
        ax7.grid(True, alpha=0.3)
        
        # 8. Force vs position analysis
        ax8 = fig.add_subplot(gs[2, 1])
        if 'position' in results and 'force_total' in results:
            pos_mm = np.array(results['position']) * 1000  # Convert to mm
            ax8.plot(pos_mm, results['force_total'], 'k-', linewidth=2, alpha=0.8)
            ax8.set_xlabel('Position (mm)')
            ax8.set_ylabel('Force (N)')
            ax8.set_title('Force vs Position')
            ax8.grid(True, alpha=0.3)
        
        # 9. Efficiency and performance metrics
        ax9 = fig.add_subplot(gs[2, 2])
        if 'energy_capacitor' in results and 'energy_kinetic' in results:
            E_cap = results['energy_capacitor']
            E_kin = results['energy_kinetic']
            
            # Calculate efficiency over time
            efficiency = np.where(E_cap > 0, E_kin / (E_cap[0] - E_cap + E_kin) * 100, 0)
            ax9.plot(t_ms, efficiency, 'purple', linewidth=2, alpha=0.8)
            ax9.set_xlabel('Time (ms)')
            ax9.set_ylabel('Efficiency (%)')
            ax9.set_title('Instantaneous Efficiency')
            ax9.grid(True, alpha=0.3)
        
        # 10. Temperature analysis if available
        ax10 = fig.add_subplot(gs[3, 0])
        if 'temperature_rise' in results and np.any(results['temperature_rise'] > 0):
            ax10.plot(t_ms, results['temperature_rise'], 'red', linewidth=2, alpha=0.8)
            ax10.set_xlabel('Time (ms)')
            ax10.set_ylabel('Temperature Rise (°C)')
            ax10.set_title('Temperature Analysis')
        else:
            # Power loss-based temperature estimate
            if 'power_loss_resistive' in results:
                # Simple thermal model: ΔT ∝ ∫P dt
                thermal_capacity = 1000  # J/K (rough estimate)
                temp_rise = np.cumsum(results['power_loss_resistive']) * (t_ms[1] - t_ms[0]) / 1000 / thermal_capacity
                ax10.plot(t_ms, temp_rise, 'red', linewidth=2, alpha=0.8)
                ax10.set_xlabel('Time (ms)')
                ax10.set_ylabel('Est. Temperature Rise (°C)')
                ax10.set_title('Estimated Temperature Rise')
        
        ax10.grid(True, alpha=0.3)
        
        # 11. Velocity and acceleration analysis
        ax11 = fig.add_subplot(gs[3, 1])
        if 'velocity' in results:
            ax11.plot(t_ms, results['velocity'], 'blue', linewidth=2, label='Velocity', alpha=0.8)
            
            # Calculate acceleration from velocity
            if len(results['velocity']) > 1:
                dt = results['time'][1] - results['time'][0]
                acceleration = np.gradient(results['velocity'], dt)
                ax11_twin = ax11.twinx()
                ax11_twin.plot(t_ms, acceleration, 'red', linewidth=1.5, label='Acceleration', alpha=0.7)
                ax11_twin.set_ylabel('Acceleration (m/s²)', color='red')
        
        ax11.set_xlabel('Time (ms)')
        ax11.set_ylabel('Velocity (m/s)', color='blue')
        ax11.set_title('Velocity & Acceleration')
        ax11.grid(True, alpha=0.3)
        
        # 12. Current and charge analysis
        ax12 = fig.add_subplot(gs[3, 2])
        if 'current' in results:
            ax12.plot(t_ms, results['current'], 'blue', linewidth=2, label='Current', alpha=0.8)
            
            if 'charge' in results:
                ax12_twin = ax12.twinx()
                ax12_twin.plot(t_ms, results['charge'], 'green', linewidth=1.5, label='Charge', alpha=0.7)
                ax12_twin.set_ylabel('Charge (C)', color='green')
        
        ax12.set_xlabel('Time (ms)')
        ax12.set_ylabel('Current (A)', color='blue')
        ax12.set_title('Current & Charge')
        ax12.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced physics analysis plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_force_decomposition_detailed(self, results, save_path=None):
        """
        Create detailed force decomposition visualization.
        
        Args:
            results: Results dictionary with force component data
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Force Decomposition Analysis', fontsize=16, fontweight='bold')
        
        if 'time' not in results or len(results['time']) == 0:
            print("No time data available for force decomposition plot")
            return
        
        t_ms = np.array(results['time']) * 1000
        
        # 1. Force components over time
        ax1 = axes[0, 0]
        force_components = ['force_gradient', 'force_reluctance', 'force_lorentz', 'force_maxwell', 'force_eddy']
        colors = ['blue', 'green', 'red', 'magenta', 'orange']
        labels = ['Gradient', 'Reluctance', 'Lorentz', 'Maxwell Stress', 'Eddy Current']
        
        for component, color, label in zip(force_components, colors, labels):
            if component in results and np.any(np.abs(results[component]) > 1e-9):
                ax1.plot(t_ms, results[component], color=color, linewidth=2, label=label, alpha=0.8)
        
        if 'force_total' in results:
            ax1.plot(t_ms, results['force_total'], 'k-', linewidth=3, label='Total', alpha=0.9)
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Force Components vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Force components vs position
        ax2 = axes[0, 1]
        if 'position' in results:
            pos_mm = np.array(results['position']) * 1000
            
            for component, color, label in zip(force_components, colors, labels):
                if component in results and np.any(np.abs(results[component]) > 1e-9):
                    ax2.plot(pos_mm, results[component], color=color, linewidth=2, label=label, alpha=0.8)
            
            if 'force_total' in results:
                ax2.plot(pos_mm, results['force_total'], 'k-', linewidth=3, label='Total', alpha=0.9)
        
        ax2.set_xlabel('Position (mm)')
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Force Components vs Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Relative force contribution
        ax3 = axes[1, 0]
        if 'force_total' in results and np.any(np.abs(results['force_total']) > 1e-9):
            total_force = np.abs(results['force_total'])
            max_total_idx = np.argmax(total_force)
            
            if max_total_idx < len(total_force) and total_force[max_total_idx] > 1e-9:
                contributions = []
                component_labels = []
                
                for component, label in zip(force_components, labels):
                    if component in results:
                        contribution = np.abs(results[component][max_total_idx]) / total_force[max_total_idx] * 100
                        if contribution > 0.1:  # Show only significant contributions
                            contributions.append(contribution)
                            component_labels.append(label)
                
                if contributions:
                    ax3.pie(contributions, labels=component_labels, autopct='%1.1f%%', startangle=90)
                    ax3.set_title(f'Force Contribution at Peak Force\n(t = {t_ms[max_total_idx]:.2f} ms)')
        
        # 4. Force efficiency over time
        ax4 = axes[1, 1]
        if 'force_total' in results and 'current' in results:
            current = results['current']
            force = results['force_total']
            
            # Force per unit current (force efficiency)
            force_efficiency = np.where(np.abs(current) > 1e-6, force / current, 0)
            
            ax4.plot(t_ms, force_efficiency, 'purple', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Force/Current (N/A)')
            ax4.set_title('Force Efficiency (Force per Current)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Force decomposition plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_electromagnetic_field_analysis(self, results, save_path=None):
        """
        Create comprehensive electromagnetic field analysis visualization.
        
        Args:
            results: Results dictionary with field data
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Electromagnetic Field Analysis', fontsize=16, fontweight='bold')
        
        if 'time' not in results or len(results['time']) == 0:
            print("No time data available for field analysis plot")
            return
        
        t_ms = np.array(results['time']) * 1000
        
        # 1. Magnetic field strength
        ax1 = axes[0, 0]
        if 'magnetic_field' in results:
            ax1.plot(t_ms, results['magnetic_field'], 'purple', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Magnetic Field (T)')
            ax1.set_title('Magnetic Field Strength')
            ax1.grid(True, alpha=0.3)
        
        # 2. Field gradient
        ax2 = axes[0, 1]
        if 'magnetic_field' in results and 'position' in results:
            # Calculate field gradient numerically
            B = results['magnetic_field']
            pos = results['position']
            
            if len(B) > 1 and len(pos) > 1:
                dB_dx = np.gradient(B, pos)
                ax2.plot(t_ms, dB_dx, 'brown', linewidth=2, alpha=0.8)
                ax2.set_xlabel('Time (ms)')
                ax2.set_ylabel('dB/dx (T/m)')
                ax2.set_title('Magnetic Field Gradient')
                ax2.grid(True, alpha=0.3)
        
        # 3. Inductance variation
        ax3 = axes[0, 2]
        if 'inductance' in results:
            L_mH = np.array(results['inductance']) * 1000
            ax3.plot(t_ms, L_mH, 'green', linewidth=2, alpha=0.8)
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Inductance (mH)')
            ax3.set_title('Coil Inductance')
            ax3.grid(True, alpha=0.3)
        
        # 4. Magnetic energy
        ax4 = axes[1, 0]
        if 'inductance' in results and 'current' in results:
            L = results['inductance']
            I = results['current']
            E_magnetic = 0.5 * L * I**2
            
            ax4.plot(t_ms, E_magnetic, 'red', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Magnetic Energy (J)')
            ax4.set_title('Magnetic Field Energy')
            ax4.grid(True, alpha=0.3)
        
        # 5. Permeability effects
        ax5 = axes[1, 1]
        if 'permeability_effective' in results:
            mu_eff = results['permeability_effective']
            ax5.plot(t_ms, mu_eff, 'orange', linewidth=2, alpha=0.8)
            ax5.set_xlabel('Time (ms)')
            ax5.set_ylabel('Effective Permeability')
            ax5.set_title('Material Permeability')
            ax5.grid(True, alpha=0.3)
        
        # 6. Saturation analysis
        ax6 = axes[1, 2]
        if 'saturation_factor' in results:
            saturation = results['saturation_factor']
            ax6.plot(t_ms, saturation, 'cyan', linewidth=2, alpha=0.8)
            ax6.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Saturation Threshold')
            ax6.set_xlabel('Time (ms)')
            ax6.set_ylabel('Saturation Factor')
            ax6.set_title('Magnetic Saturation')
            ax6.set_ylim(0, 1.1)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Electromagnetic field analysis plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    def calculate_bfield_map_2d(self, current, z_range=None, r_range=None, 
                                num_z=100, num_r=50, include_projectile=True, 
                                projectile_position=None):
        """
        Calculate detailed 2D magnetic field map using Biot-Savart law.
        
        Args:
            current: Current in the coil (A)
            z_range: [z_min, z_max] axial range (m)
            r_range: [r_min, r_max] radial range (m)
            num_z: Number of axial grid points
            num_r: Number of radial grid points
            include_projectile: Whether to include projectile effects
            projectile_position: Position of projectile (if included)
            
        Returns:
            dict: Contains Z, R meshgrids and Bz, Br field components
        """
        # Set default ranges if not provided
        if z_range is None:
            z_range = [-self.physics.coil_length, 2*self.physics.coil_length]
        if r_range is None:
            r_range = [0, 3*self.physics.coil_outer_radius]
        
        # Create coordinate grids
        z_points = np.linspace(z_range[0], z_range[1], num_z, retstep=False)
        r_points = np.linspace(r_range[0], r_range[1], num_r, retstep=False)
        Z, R = np.meshgrid(z_points, r_points)
        
        # Initialize field arrays
        Bz = np.zeros_like(Z)
        Br = np.zeros_like(Z)
        B_magnitude = np.zeros_like(Z)
        
        print(f"Calculating B-field on {num_z}×{num_r} grid...")
        
        # Calculate field using superposition of current loops
        # Discretize coil into current loops
        num_loops = max(50, int(self.physics.total_turns / 5))
        loop_positions = np.linspace(0, self.physics.coil_length, num_loops, retstep=False)
        current_per_loop = current * self.physics.total_turns / num_loops
        
        for i, z in enumerate(z_points):
            for j, r in enumerate(r_points):
                bz_total = 0
                br_total = 0
                
                # Sum contributions from all current loops
                for loop_z in loop_positions:
                    # Use exact Biot-Savart solution for circular loop
                    bz_loop, br_loop = self._biot_savart_circular_loop(
                        z, r, loop_z, self.physics.avg_coil_radius, current_per_loop
                    )
                    bz_total += bz_loop
                    br_total += br_loop
                
                Bz[j, i] = bz_total
                Br[j, i] = br_total
                B_magnitude[j, i] = np.sqrt(bz_total**2 + br_total**2)
        
        print("B-field calculation complete.")
        
        return {
            'Z': Z,
            'R': R, 
            'Bz': Bz,
            'Br': Br,
            'B_magnitude': B_magnitude,
            'current': current,
            'z_range': z_range,
            'r_range': r_range
        }
    
    def calculate_bfield_3d(self, current, z_range=None, x_range=None, y_range=None,
                           num_z=50, num_x=30, num_y=30):
        """
        Calculate 3D magnetic field by rotating 2D axisymmetric solution.
        
        Args:
            current: Current in the coil (A)
            z_range: [z_min, z_max] axial range (m)
            x_range: [x_min, x_max] range (m)
            y_range: [y_min, y_max] range (m)
            num_z, num_x, num_y: Grid discretization
            
        Returns:
            dict: 3D field data
        """
        # Set default ranges
        if z_range is None:
            z_range = [-self.physics.coil_length * 0.5, self.physics.coil_length * 1.5]
        if x_range is None:
            max_r = 2 * self.physics.coil_outer_radius
            x_range = [-max_r, max_r]
        if y_range is None:
            max_r = 2 * self.physics.coil_outer_radius
            y_range = [-max_r, max_r]
        
        # Create 3D coordinate grids
        z_points = np.linspace(z_range[0], z_range[1], num_z, retstep=False)
        x_points = np.linspace(x_range[0], x_range[1], num_x, retstep=False)
        y_points = np.linspace(y_range[0], y_range[1], num_y, retstep=False)
        
        Z, X, Y = np.meshgrid(z_points, x_points, y_points, indexing='ij')
        
        # Convert Cartesian to cylindrical coordinates
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        
        # Initialize 3D field arrays
        Bx = np.zeros_like(Z)
        By = np.zeros_like(Z)
        Bz = np.zeros_like(Z)
        
        print(f"Calculating 3D B-field on {num_z}×{num_x}×{num_y} grid...")
        
        # Calculate 2D field components for each point
        for i in range(num_z):
            for j in range(num_x):
                for k in range(num_y):
                    z = Z[i, j, k]
                    r = R[i, j, k]
                    phi = Phi[i, j, k]
                    
                    if r < 1e-12:  # On axis
                        bz_cyl = self.physics.magnetic_field_solenoid_on_axis(z, current)
                        br_cyl = 0
                    else:
                        # Calculate field using 2D solution
                        bz_cyl, br_cyl = self._biot_savart_total_field(z, r, current)
                    
                    # Convert cylindrical field components to Cartesian
                    Bz[i, j, k] = bz_cyl
                    Bx[i, j, k] = br_cyl * np.cos(phi)
                    By[i, j, k] = br_cyl * np.sin(phi)
        
        print("3D B-field calculation complete.")
        
        return {
            'X': X, 'Y': Y, 'Z': Z,
            'Bx': Bx, 'By': By, 'Bz': Bz,
            'B_magnitude': np.sqrt(Bx**2 + By**2 + Bz**2),
            'current': current
        }
    
    def _biot_savart_total_field(self, z, r, current):
        """Calculate total field at (z,r) using superposition of current loops."""
        num_loops = max(50, int(self.physics.total_turns / 5))
        loop_positions = np.linspace(0, self.physics.coil_length, num_loops, retstep=False)
        current_per_loop = current * self.physics.total_turns / num_loops
        
        bz_total = 0
        br_total = 0
        
        for loop_z in loop_positions:
            bz_loop, br_loop = self._biot_savart_circular_loop(
                z, r, loop_z, self.physics.avg_coil_radius, current_per_loop
            )
            bz_total += bz_loop
            br_total += br_loop
        
        return bz_total, br_total

    def _biot_savart_circular_loop(self, z, r, loop_z, loop_radius, current):
        """
        Calculate magnetic field from a circular current loop using exact Biot-Savart law.
        
        Args:
            z, r: Field point coordinates (m)
            loop_z: Axial position of the loop (m)
            loop_radius: Radius of the current loop (m)
            current: Current in the loop (A)
            
        Returns:
            Bz, Br: Axial and radial field components (T)
        """
        mu0 = self.physics.mu0
        
        # Distance from loop to field point
        dz = z - loop_z
        
        # Handle on-axis case (r = 0)
        if r < 1e-12:
            # On-axis formula
            distance_cubed = (loop_radius**2 + dz**2)**(3/2)
            if distance_cubed > 1e-12:
                Bz = mu0 * current * loop_radius**2 / (2 * distance_cubed)
            else:
                Bz = 0
            Br = 0
            return Bz, Br
        
        # Off-axis calculation using elliptic integrals (simplified)
        # For computational efficiency, use dipole approximation for far field
        distance = np.sqrt(dz**2 + r**2)
        
        if distance > 3 * loop_radius:
            # Far field dipole approximation
            magnetic_moment = np.pi * loop_radius**2 * current
            
            cos_theta = dz / distance
            sin_theta = r / distance
            
            B_parallel = (mu0 * magnetic_moment / (4 * np.pi * distance**3)) * 2 * cos_theta
            B_perpendicular = (mu0 * magnetic_moment / (4 * np.pi * distance**3)) * sin_theta
            
            Bz = B_parallel * cos_theta - B_perpendicular * sin_theta
            Br = B_parallel * sin_theta + B_perpendicular * cos_theta
            
        else:
            # Near field - use more accurate calculation
            # Simplified version of elliptic integral solution
            k_squared = 4 * loop_radius * r / ((loop_radius + r)**2 + dz**2)
            
            if k_squared < 1e-12:
                Bz = 0
                Br = 0
            else:
                # Approximate elliptic integrals for computational efficiency
                k = np.sqrt(k_squared)
                
                # Complete elliptic integrals (approximated)
                if k < 0.9:
                    K_k = np.pi/2 * (1 + k**2/4 + 9*k**4/64)  # Approximation for K(k)
                    E_k = np.pi/2 * (1 - k**2/4 - 3*k**4/64)  # Approximation for E(k)
                else:
                    # High k approximation
                    K_k = np.log(4/np.sqrt(1-k**2))
                    E_k = 1
                
                # Field components
                C = mu0 * current / (4 * np.pi * np.sqrt((loop_radius + r)**2 + dz**2))
                
                Bz = C * (((loop_radius**2 - r**2 - dz**2) * E_k + (loop_radius + r)**2 * K_k + dz**2) / (loop_radius - r)**2)
                Br = C * dz / r * (((loop_radius**2 + r**2 + dz**2) * E_k - (loop_radius - r)**2 * K_k) / (loop_radius - r)**2)
        
        return Bz, Br
    
    def trace_field_lines_3d(self, field_data_3d, start_points, max_length=0.2, step_size=0.001):
        """
        Trace 3D magnetic field lines from starting points.
        
        Args:
            field_data_3d: 3D field data from calculate_bfield_3d
            start_points: List of (x, y, z) starting points
            max_length: Maximum length of field line
            step_size: Integration step size
            
        Returns:
            List of field line coordinates
        """
        from scipy.interpolate import RegularGridInterpolator
        
        X, Y, Z = field_data_3d['X'], field_data_3d['Y'], field_data_3d['Z']
        Bx, By, Bz = field_data_3d['Bx'], field_data_3d['By'], field_data_3d['Bz']
        
        # Create interpolators for field components
        z_points = Z[:, 0, 0]
        x_points = X[0, :, 0]
        y_points = Y[0, 0, :]
        
        interp_bx = RegularGridInterpolator((z_points, x_points, y_points), Bx, 
                                           bounds_error=False, fill_value=0)
        interp_by = RegularGridInterpolator((z_points, x_points, y_points), By, 
                                           bounds_error=False, fill_value=0)
        interp_bz = RegularGridInterpolator((z_points, x_points, y_points), Bz, 
                                           bounds_error=False, fill_value=0)
        
        def field_func(pos):
            """Field function for integration."""
            z, x, y = pos
            bx = interp_bx([z, x, y])[0]
            by = interp_by([z, x, y])[0]
            bz = interp_bz([z, x, y])[0]
            
            # Normalize to unit vector
            b_mag = np.sqrt(bx**2 + by**2 + bz**2)
            if b_mag > 1e-12:
                return np.array([bz, bx, by]) / b_mag
            else:
                return np.array([0, 0, 0])
        
        field_lines = []
        
        for start_point in start_points:
            x0, y0, z0 = start_point
            
            # Trace forward
            t = np.arange(0, max_length, step_size)
            try:
                line_forward = odeint(lambda pos, t: field_func(pos), [z0, x0, y0], t)
                
                # Trace backward
                t_back = np.arange(0, -max_length, -step_size)
                line_backward = odeint(lambda pos, t: -field_func(pos), [z0, x0, y0], t_back)
                
                # Combine and reorder
                line_full = np.vstack([line_backward[::-1][:-1], line_forward])
                
                # Convert back to x, y, z order
                field_line = np.column_stack([line_full[:, 1], line_full[:, 2], line_full[:, 0]])
                field_lines.append(field_line)
                
            except Exception as e:
                print(f"Warning: Field line tracing failed from {start_point}: {e}")
                continue
        
        return field_lines
    
    def create_3d_coil_geometry(self, num_turns_visual=20):
        """
        Create 3D coil geometry for visualization.
        
        Args:
            num_turns_visual: Number of turns to show (for visual clarity)
            
        Returns:
            Coil coordinates for 3D plotting
        """
        # Create helical coil path
        turns = np.linspace(0, num_turns_visual, 1000, retstep=False)
        theta = 2 * np.pi * turns
        
        # Axial position
        z_coil = (turns / num_turns_visual) * self.physics.coil_length
        
        # Create multiple layers
        coil_lines = []
        
        for layer in range(self.physics.num_layers):
            # Radius for this layer
            layer_radius = (self.physics.coil_inner_radius + 
                           layer * (self.physics.coil_outer_radius - self.physics.coil_inner_radius) / self.physics.num_layers)
            
            # Coordinates for this layer
            x_coil = layer_radius * np.cos(theta)
            y_coil = layer_radius * np.sin(theta)
            
            coil_lines.append(np.column_stack([x_coil, y_coil, z_coil]))
        
        return coil_lines
    
    def create_3d_projectile_geometry(self, position):
        """
        Create 3D projectile geometry at given position.
        
        Args:
            position: Axial position of projectile
            
        Returns:
            Projectile mesh coordinates
        """
        # Create cylindrical projectile
        theta = np.linspace(0, 2*np.pi, 20, retstep=False)
        z_proj = np.array([position - self.physics.proj_length, position])
        
        # Create surface coordinates
        theta_mesh, z_mesh = np.meshgrid(theta, z_proj)
        x_mesh = self.physics.proj_radius * np.cos(theta_mesh)
        y_mesh = self.physics.proj_radius * np.sin(theta_mesh)
        
        return x_mesh, y_mesh, z_mesh
    
    def plot_3d_field_visualization(self, current, save_path=None, interactive=True,
                                   show_field_lines=True, show_coil=True, 
                                   projectile_position=None):
        """
        Create comprehensive 3D visualization of magnetic field and coil geometry.
        
        Args:
            current: Current for field calculation
            save_path: Path to save plot
            interactive: Whether to create interactive plot
            show_field_lines: Whether to show 3D field lines
            show_coil: Whether to show coil geometry
            projectile_position: Position of projectile
        """
        print("Creating 3D field visualization...")
        
        # Calculate 3D field data
        field_data_3d = self.calculate_bfield_3d(current, num_z=40, num_x=25, num_y=25)
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        
        # Plot field magnitude as volume rendering (simplified with scatter)
        if True:  # Volume rendering
            X, Y, Z = field_data_3d['X'], field_data_3d['Y'], field_data_3d['Z']
            B_mag = field_data_3d['B_magnitude']
            
            # Sample points for visualization (reduce density)
            skip = 2
            x_sample = X[::skip, ::skip, ::skip].flatten()
            y_sample = Y[::skip, ::skip, ::skip].flatten()
            z_sample = Z[::skip, ::skip, ::skip].flatten()
            b_sample = B_mag[::skip, ::skip, ::skip].flatten()
            
            # Only plot points with significant field
            threshold = np.percentile(b_sample, 70)
            mask = b_sample > threshold
            
            scatter = ax.scatter(x_sample[mask] * 1000, y_sample[mask] * 1000, z_sample[mask] * 1000,
                               c=b_sample[mask] * 1000, cmap='plasma', alpha=0.3, s=10)  # type: ignore
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
            cbar.set_label('|B| (mT)', fontsize=12)
        
        # Plot 3D field lines
        if show_field_lines:
            print("Tracing 3D field lines...")
            
            # Create starting points for field lines
            start_points = []
            
            # Field lines from coil inner radius
            num_lines = 12
            theta_start = np.linspace(0, 2*np.pi, num_lines, endpoint=False, retstep=False)
            
            for theta in theta_start:
                for z_start in [0.01, self.physics.coil_length/2, self.physics.coil_length - 0.01]:
                    r_start = self.physics.coil_inner_radius * 1.1
                    x_start = r_start * np.cos(theta)
                    y_start = r_start * np.sin(theta)
                    start_points.append([x_start, y_start, z_start])
            
            # Trace field lines
            field_lines = self.trace_field_lines_3d(field_data_3d, start_points)
            
            # Plot field lines
            for i, line in enumerate(field_lines):
                if len(line) > 10:  # Only plot substantial field lines
                    ax.plot(line[:, 0] * 1000, line[:, 1] * 1000, line[:, 2] * 1000,
                           'blue', alpha=0.7, linewidth=1.5)
        
        # Plot 3D coil geometry
        if show_coil:
            print("Rendering 3D coil geometry...")
            coil_lines = self.create_3d_coil_geometry(num_turns_visual=8)
            
            for i, coil_line in enumerate(coil_lines):
                # Use available matplotlib colormap
                color = plt.cm.get_cmap('copper')(i / len(coil_lines))
                ax.plot(coil_line[:, 0] * 1000, coil_line[:, 1] * 1000, coil_line[:, 2] * 1000,
                       color=color, linewidth=3, alpha=0.8)
        
        # Plot projectile
        if projectile_position is not None:
            print("Adding projectile geometry...")
            x_proj, y_proj, z_proj = self.create_3d_projectile_geometry(projectile_position)
            
            ax.plot_surface(x_proj * 1000, y_proj * 1000, z_proj * 1000,  # type: ignore
                           color='red', alpha=0.8, linewidth=0)
        
        # Set labels and title
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)  # type: ignore
        ax.set_title(f'3D Coilgun Magnetic Field Visualization (I = {current:.0f} A)', 
                     fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = max(
            self.physics.coil_outer_radius * 1000,
            self.physics.coil_length * 1000
        )
        ax.set_xlim(-max_range*1.2, max_range*1.2)
        ax.set_ylim(-max_range*1.2, max_range*1.2)
        ax.set_zlim(-max_range*0.3, max_range*1.8)  # type: ignore
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=45)  # type: ignore
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to: {save_path}")
        
        if interactive:
            plt.show()
        
        return fig, ax

    def plot_bfield_contours(self, field_data, save_path=None, show_coil=True, 
                            show_projectile=True, projectile_position=None):
        """
        Create detailed magnetic field contour plots.
        
        Args:
            field_data: Field data from calculate_bfield_map_2d
            save_path: Path to save the plot (optional)
            show_coil: Whether to show coil geometry
            show_projectile: Whether to show projectile
            projectile_position: Position of projectile
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle(f'Coilgun Magnetic Field Analysis (I = {field_data["current"]:.0f} A)', 
                     fontsize=16, fontweight='bold')
        
        Z = field_data['Z'] * 1000  # Convert to mm
        R = field_data['R'] * 1000
        Bz = field_data['Bz'] * 1000  # Convert to mT
        Br = field_data['Br'] * 1000
        B_mag = field_data['B_magnitude'] * 1000
        
        # Plot 1: Axial field component (Bz)
        contour1 = ax1.contourf(Z, R, Bz, levels=50, cmap='RdBu_r')
        ax1.contour(Z, R, Bz, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        fig.colorbar(contour1, ax=ax1, label='Bz (mT)')
        ax1.set_title('Axial Magnetic Field (Bz)')
        ax1.set_xlabel('Axial Position (mm)')
        ax1.set_ylabel('Radial Position (mm)')
        
        # Plot 2: Radial field component (Br)
        contour2 = ax2.contourf(Z, R, Br, levels=50, cmap='RdBu_r')
        ax2.contour(Z, R, Br, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        fig.colorbar(contour2, ax=ax2, label='Br (mT)')
        ax2.set_title('Radial Magnetic Field (Br)')
        ax2.set_xlabel('Axial Position (mm)')
        ax2.set_ylabel('Radial Position (mm)')
        
        # Plot 3: Field magnitude
        contour3 = ax3.contourf(Z, R, B_mag, levels=50, cmap='plasma')
        ax3.contour(Z, R, B_mag, levels=20, colors='white', alpha=0.5, linewidths=0.5)
        fig.colorbar(contour3, ax=ax3, label='|B| (mT)')
        ax3.set_title('Magnetic Field Magnitude')
        ax3.set_xlabel('Axial Position (mm)')
        ax3.set_ylabel('Radial Position (mm)')
        
        # Plot 4: Field lines (streamplot)
        # Subsample for cleaner streamlines
        skip = 3
        ax4.streamplot(Z[::skip, ::skip], R[::skip, ::skip], 
                      Bz[::skip, ::skip], Br[::skip, ::skip],
                      color=B_mag[::skip, ::skip], cmap='viridis',
                      density=1.5, arrowsize=1.2)
        ax4.set_title('Magnetic Field Lines')
        ax4.set_xlabel('Axial Position (mm)')
        ax4.set_ylabel('Radial Position (mm)')
        
        # Add coil geometry to all plots
        if show_coil:
            for ax in [ax1, ax2, ax3, ax4]:
                self._add_coil_geometry(ax)
        
        # Add projectile if specified
        if show_projectile and projectile_position is not None:
            for ax in [ax1, ax2, ax3, ax4]:
                self._add_projectile_geometry(ax, projectile_position)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"B-field contour plot saved to: {save_path}")
        
        plt.show()
    
    def plot_bfield_3d(self, field_data, save_path=None):
        """
        Create 3D surface plot of magnetic field magnitude.
        
        Args:
            field_data: Field data from calculate_bfield_map_2d
            save_path: Path to save the plot (optional)
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        
        Z = field_data['Z'] * 1000  # Convert to mm
        R = field_data['R'] * 1000
        B_mag = field_data['B_magnitude'] * 1000  # Convert to mT
        
        # Create 3D surface plot
        surf = ax.plot_surface(Z, R, B_mag, cmap='plasma', alpha=0.8,   # type: ignore
                              linewidth=0, antialiased=True)
        
        # Add contour lines at the base
        ax.contour(Z, R, B_mag, zdir='z', offset=0, levels=20, cmap='plasma', alpha=0.5)
        
        ax.set_xlabel('Axial Position (mm)')
        ax.set_ylabel('Radial Position (mm)')
        ax.set_zlabel('Magnetic Field Magnitude (mT)')  # type: ignore
        ax.set_title(f'3D Magnetic Field Distribution (I = {field_data["current"]:.0f} A)')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='|B| (mT)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D B-field plot saved to: {save_path}")
        
        plt.show()
    
    def plot_onaxis_field_profile(self, current_values=None, save_path=None):
        """
        Plot magnetic field and force along the coil axis for different currents.
        
        Args:
            current_values: List of current values to plot (A). If None, uses circuit-based estimation.
            save_path: Path to save the plot (optional)
        """
        if current_values is None:
            # Use circuit-based estimation instead of arbitrary defaults
            initial_voltage = self.physics.initial_voltage
            estimated_max_current = initial_voltage / self.physics.total_resistance
            current_values = [
                estimated_max_current * 0.3,
                estimated_max_current * 0.7,
                estimated_max_current
            ]
            print(f"Using circuit-based current estimates for field profile: {[f'{c:.0f}A' for c in current_values]}")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Create color map for different currents
        if len(current_values) > 1:
            colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(current_values), retstep=False))
        else:
            colors = ['blue']
        
        # Axial positions for field calculation
        z_points = np.linspace(-self.physics.coil_length, 2*self.physics.coil_length, 300, retstep=False)
        z_mm = z_points * 1000  # Convert to mm
        
        # Positions for force calculation (projectile range)
        positions = np.linspace(-0.05, self.physics.coil_length + 0.05, 200, retstep=False)
        positions_mm = positions * 1000
        
        # Plot field and force for each current value
        for i, current in enumerate(current_values):
            color = colors[i] if len(current_values) > 1 else colors[0]
            
            # Calculate magnetic field along axis
            bz_values = []
            for z in z_points:
                bz = self.physics.magnetic_field_solenoid_on_axis(z, current)
                bz_values.append(bz * 1000)  # Convert to mT
            
            # Plot field profile
            stage_label = f'Stage {i+1}' if len(current_values) > 3 else f'{current:.0f}A'
            ax1.plot(z_mm, bz_values, linewidth=2, color=color, label=f'{stage_label} ({current:.0f}A)')
            
            # Calculate force profile
            forces = []
            for pos in positions:
                force = self.physics.magnetic_force_with_circuit_logic(current, pos)
                forces.append(force)
            
            # Plot force profile with same color
            ax2.plot(positions_mm, forces, linewidth=2, color=color, label=f'{stage_label} ({current:.0f}A)')
        
        # Configure field plot
        ax1.set_xlabel('Axial Position (mm)')
        ax1.set_ylabel('Magnetic Field Bz (mT)')
        ax1.set_title('On-Axis Magnetic Field Profile (All Stages)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add coil boundaries to field plot
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.axvline(self.physics.coil_length * 1000, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.axvline(self.physics.coil_center * 1000, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        # Add text labels for coil boundaries
        ax1.text(0, ax1.get_ylim()[1] * 0.95, 'Coil Start', rotation=90, ha='right', va='top', alpha=0.7)
        ax1.text(self.physics.coil_length * 1000, ax1.get_ylim()[1] * 0.95, 'Coil End', rotation=90, ha='right', va='top', alpha=0.7)
        ax1.text(self.physics.coil_center * 1000, ax1.get_ylim()[1] * 0.95, 'Coil Center', rotation=90, ha='right', va='top', alpha=0.7, color='orange')
        
        # Configure force plot
        ax2.set_xlabel('Projectile Position (mm)')
        ax2.set_ylabel('Magnetic Force (N)')
        ax2.set_title('Magnetic Force vs Position (All Stages)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add reference lines to force plot
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax2.axvline(self.physics.coil_length * 1000, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax2.axvline(self.physics.coil_center * 1000, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        # Add text labels for coil boundaries in force plot
        force_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
        ax2.text(0, ax2.get_ylim()[1] - force_range * 0.05, 'Coil Start', rotation=90, ha='right', va='top', alpha=0.7)
        ax2.text(self.physics.coil_length * 1000, ax2.get_ylim()[1] - force_range * 0.05, 'Coil End', rotation=90, ha='right', va='top', alpha=0.7)
        ax2.text(self.physics.coil_center * 1000, ax2.get_ylim()[1] - force_range * 0.05, 'Coil Center', rotation=90, ha='right', va='top', alpha=0.7, color='orange')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"On-axis field and force profiles saved to: {save_path}")
        
        plt.show()

    def animate_3d_projectile_motion(self, simulation_results, save_path=None, interval=100):
        """
        Create 3D animation of projectile motion with magnetic field visualization.
        
        Args:
            simulation_results: Results from CoilgunSimulation
            save_path: Path to save animation
            interval: Animation interval in ms
        """
        if simulation_results.results['time'] is None:
            print("No detailed simulation results available for animation.")
            return
        
        # Extract data
        time_data = simulation_results.results['time']
        current_data = simulation_results.results['current']
        position_data = simulation_results.results['position']
        
        # Select frames for animation
        num_frames = min(50, len(time_data) // 20)  # Reduce frames for 3D
        frame_indices = np.linspace(0, len(time_data)-1, num_frames, dtype=int, retstep=False)
        
        print(f"Creating 3D animation with {num_frames} frames...")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')  # type: ignore
        
        # Pre-render coil geometry
        coil_lines = self.create_3d_coil_geometry(num_turns_visual=6)
        
        def animate(frame_idx):
            ax.clear()
            
            idx = frame_indices[frame_idx]
            current = current_data[idx]
            position = position_data[idx]
            time = time_data[idx]
            
            artists = []
            
            # Plot coil
            for i, coil_line in enumerate(coil_lines):
                color = plt.cm.get_cmap('copper')(i / len(coil_lines))
                line = ax.plot(coil_line[:, 0] * 1000, coil_line[:, 1] * 1000, coil_line[:, 2] * 1000,
                              color=color, linewidth=2, alpha=0.7)
                artists.extend(line)
            
            # Plot projectile
            x_proj, y_proj, z_proj = self.create_3d_projectile_geometry(position)
            surf = ax.plot_surface(x_proj * 1000, y_proj * 1000, z_proj * 1000,  # type: ignore
                                  color='red', alpha=0.9, linewidth=0)
            artists.append(surf)
            
            # Add some field lines around projectile
            if current > 10:  # Only show field lines when current is significant
                # Simple field visualization - radial lines from coil center
                theta_lines = np.linspace(0, 2*np.pi, 8, endpoint=False, retstep=False)
                for theta in theta_lines:
                    r_line = np.linspace(0, self.physics.coil_outer_radius * 1.5, 20, retstep=False)
                    x_line = r_line * np.cos(theta) * 1000
                    y_line = r_line * np.sin(theta) * 1000
                    z_line = np.full_like(r_line, self.physics.coil_center * 1000)
                    
                    # Color by field strength (approximate)
                    colors = plt.cm.get_cmap('viridis')(r_line / (self.physics.coil_outer_radius * 1.5))
                    for i in range(len(r_line)-1):
                        line = ax.plot([x_line[i], x_line[i+1]], [y_line[i], y_line[i+1]], 
                                      [z_line[i], z_line[i+1]], color=colors[i], alpha=0.6)
                        artists.extend(line)
            
            # Set labels and title
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')  # type: ignore
            ax.set_title(f'3D Coilgun Animation - t={time*1000:.1f}ms, I={current:.0f}A, v={simulation_results.results["velocity"][idx]:.1f}m/s')
            
            # Set consistent axis limits
            max_range = max(self.physics.coil_outer_radius * 1000, self.physics.coil_length * 1000)
            ax.set_xlim(-max_range*1.2, max_range*1.2)
            ax.set_ylim(-max_range*1.2, max_range*1.2)
            ax.set_zlim(-max_range*0.3, max_range*1.8)  # type: ignore
            
            ax.view_init(elev=15, azim=frame_idx * 2)  # type: ignore  # Slowly rotate view
            
            return artists
        
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False)  # type: ignore
        
        if save_path:
            print("Saving 3D animation (this may take a while)...")
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"3D animation saved to: {save_path}")
        
        plt.show()
        return anim

    def animate_field_evolution(self, simulation_results, save_path=None, interval=50):
        """
        Create animation of magnetic field evolution during projectile motion.
        
        Args:
            simulation_results: Results from CoilgunSimulation
            save_path: Path to save animation (optional)
            interval: Animation interval in ms
        """
        if simulation_results.results['time'] is None:
            print("No detailed simulation results available for animation.")
            return
        
        # Subsample time points for animation
        time_data = simulation_results.results['time']
        current_data = simulation_results.results['current']
        position_data = simulation_results.results['position']
        
        # Select frames for animation
        if len(time_data) < 10:
            print("Insufficient time data for animation (need at least 10 points)")
            return None
        
        num_frames = min(100, len(time_data) // 10)  # Limit to 100 frames
        frame_indices = np.linspace(0, len(time_data)-1, num_frames, dtype=int, retstep=False)
        
        # Pre-calculate field data for efficiency
        print("Pre-calculating field frames for animation...")
        field_frames = []
        
        for i, idx in enumerate(frame_indices):
            current = current_data[idx]
            position = position_data[idx]
            
            if i % 20 == 0:
                print(f"Calculating frame {i+1}/{num_frames}")
            
            # Calculate field with smaller grid for speed
            field_data = self.calculate_bfield_map_2d(
                current, num_z=50, num_r=30, 
                projectile_position=position
            )
            field_frames.append({
                'field': field_data,
                'time': time_data[idx],
                'current': current,
                'position': position
            })
        
        # Create animation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        def animate(frame_idx):
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            frame = field_frames[frame_idx]
            field_data = frame['field']
            
            Z = field_data['Z'] * 1000
            R = field_data['R'] * 1000
            Bz = field_data['Bz'] * 1000
            B_mag = field_data['B_magnitude'] * 1000
            
            artists = []
            
            # Plot field magnitude
            im1 = ax1.contourf(Z, R, B_mag, levels=30, cmap='plasma')
            ax1.set_title(f'|B| Field (t = {frame["time"]*1000:.1f} ms)')
            ax1.set_xlabel('Position (mm)')
            ax1.set_ylabel('Radius (mm)')
            artists.extend(im1.collections)
            
            # Plot axial field
            im2 = ax2.contourf(Z, R, Bz, levels=30, cmap='RdBu_r')
            ax2.set_title(f'Bz Field (I = {frame["current"]:.0f} A)')
            ax2.set_xlabel('Position (mm)')
            ax2.set_ylabel('Radius (mm)')
            artists.extend(im2.collections)
            
            # Add coil and projectile geometry
            for ax in [ax1, ax2]:
                self._add_coil_geometry(ax)
                self._add_projectile_geometry(ax, frame['position'])
            
            # Plot current vs time
            current_history = [f['current'] for f in field_frames[:frame_idx+1]]
            time_history = [f['time']*1000 for f in field_frames[:frame_idx+1]]
            
            line1 = ax3.plot(time_history, current_history, 'b-', linewidth=2)
            line1_marker = ax3.axvline(frame['time']*1000, color='red', linestyle='--')
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Current (A)')
            ax3.set_title('Current vs Time')
            ax3.grid(True, alpha=0.3)
            artists.extend(line1)
            artists.append(line1_marker)
            
            # Plot position vs time
            position_history = [f['position']*1000 for f in field_frames[:frame_idx+1]]
            
            line2 = ax4.plot(time_history, position_history, 'g-', linewidth=2)
            line2_marker = ax4.axvline(frame['time']*1000, color='red', linestyle='--')
            line2_ref1 = ax4.axhline(0, color='black', linestyle=':', alpha=0.5, label='Coil entrance')
            line2_ref2 = ax4.axhline(self.physics.coil_center*1000, color='orange', linestyle=':', alpha=0.5, label='Coil center')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Position (mm)')
            ax4.set_title('Projectile Position vs Time')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            artists.extend(line2)
            artists.extend([line2_marker, line2_ref1, line2_ref2])
            
            plt.tight_layout()
            return artists
        
        if len(field_frames) == 0:
            print("No field frames calculated - cannot create animation")
            return None
        
        anim = FuncAnimation(fig, animate, frames=len(field_frames), 
                           interval=interval, blit=False, repeat=True)  # type: ignore
        
        if save_path:
            try:
                anim.save(save_path, writer='pillow', fps=1000//interval)
                print(f"Animation saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")
        
        plt.show()
        return anim
    
    def _add_coil_geometry(self, ax):
        """Add coil geometry visualization to a plot."""
        # Coil boundaries
        coil_inner = patches.Rectangle(
            (0, self.physics.coil_inner_radius * 1000), 
            self.physics.coil_length * 1000,
            (self.physics.coil_outer_radius - self.physics.coil_inner_radius) * 1000,
            linewidth=2, edgecolor='brown', facecolor='brown', alpha=0.3,
            label='Coil'
        )
        ax.add_patch(coil_inner)
        
        # Center line
        ax.axvline(self.physics.coil_center * 1000, color='orange', 
                  linestyle=':', alpha=0.7, linewidth=2, label='Coil center')
    
    def _add_projectile_geometry(self, ax, position):
        """Add projectile geometry visualization to a plot."""
        position_mm = position * 1000
        proj_length_mm = self.physics.proj_length * 1000
        proj_radius_mm = self.physics.proj_radius * 1000
        
        # Projectile rectangle
        projectile = patches.Rectangle(
            (position_mm - proj_length_mm, 0),
            proj_length_mm, proj_radius_mm,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.7,
            label='Projectile'
        )
        ax.add_patch(projectile)


def extract_actual_current_data(time_series_data=None, simulation_results=None, physics_engine=None):
    """
    Extract actual current data from multiple possible sources.
    
    Args:
        time_series_data: Time series data dict (optional)
        simulation_results: CoilgunSimulation results object (optional) 
        physics_engine: Physics engine for circuit-based fallback (optional)
        
    Returns:
        tuple: (current_values_list, data_source_description, max_current, avg_current)
    """
    # Try to get current data from multiple sources
    current_data = None
    data_source = None
    
    # Source 1: simulation_results object
    if simulation_results is not None and hasattr(simulation_results, 'results'):
        if simulation_results.results.get('current') is not None:
            current_data = simulation_results.results['current']
            data_source = "simulation_results.results"
    
    # Source 2: time_series_data dict
    if current_data is None and time_series_data is not None:
        # Try different key formats
        for key in ['current_A', 'current', 'Current']:
            if key in time_series_data and time_series_data[key] is not None:
                current_data = time_series_data[key]
                data_source = f"time_series_data['{key}']"
                break
    
    # Source 3: simulation_results simulation_info
    if current_data is None and simulation_results is not None:
        if hasattr(simulation_results, 'simulation_info'):
            max_current = simulation_results.simulation_info.get('max_current', None)
            if max_current is not None and max_current > 0:
                # Create a simple current profile based on max current
                current_data = [0, max_current * 0.5, max_current, max_current * 0.3, 0]
                data_source = f"simulation_info.max_current ({max_current:.1f}A)"
    
    if current_data is not None and len(current_data) > 0:
        current_array = np.array(current_data)
        nonzero_currents = current_array[current_array != 0]
        
        if len(nonzero_currents) > 0:
            max_current = np.max(np.abs(nonzero_currents))
            avg_current = np.mean(np.abs(nonzero_currents))
            
            # Create meaningful current values based on actual data
            if max_current > avg_current * 1.2:
                # Good dynamic range in the data
                currents = [
                    avg_current * 0.3,  # Low current
                    avg_current,        # Average current  
                    max_current         # Peak current
                ]
            else:
                # Limited dynamic range, use peak current as basis
                currents = [
                    max_current * 0.3,  # 30% of peak
                    max_current * 0.7,  # 70% of peak
                    max_current         # Peak current
                ]
            
            return currents, data_source, max_current, avg_current
    
    # Fallback to circuit-based estimation if no actual data
    if physics_engine is not None:
        initial_voltage = physics_engine.initial_voltage
        estimated_max_current = initial_voltage / physics_engine.total_resistance
        
        currents = [
            estimated_max_current * 0.3,
            estimated_max_current * 0.7,
            estimated_max_current
        ]
        data_source = f"circuit-based estimate (V={initial_voltage}V, R={physics_engine.total_resistance:.3f}Ω)"
        return currents, data_source, estimated_max_current, estimated_max_current * 0.7
    
    # Ultimate fallback - but this should be avoided
    return [50, 100, 200], "ultimate_fallback", 200, 100


def create_enhanced_physics_visualizations(time_series_data, output_dir, visualizer):
    """
    Create enhanced physics visualization plots for new data structure.
    
    Args:
        time_series_data: Enhanced time series data with physics components
        output_dir: Directory to save visualizations
        visualizer: CoilgunFieldVisualizer instance
    """
    print("Creating enhanced physics visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create comprehensive enhanced physics analysis plot
    if time_series_data and len(time_series_data.get('time', [])) > 0:
        print("  Creating comprehensive physics analysis...")
        try:
            visualizer.plot_enhanced_physics_analysis(
                time_series_data, 
                save_path=output_path / "enhanced_physics_analysis.png"
            )
        except Exception as e:
            print(f"  Warning: Enhanced physics analysis failed: {e}")
        
        # Create detailed force decomposition plot
        print("  Creating force decomposition analysis...")
        try:
            visualizer.plot_force_decomposition_detailed(
                time_series_data,
                save_path=output_path / "force_decomposition_detailed.png"
            )
        except Exception as e:
            print(f"  Warning: Force decomposition analysis failed: {e}")
        
        # Create electromagnetic field analysis
        print("  Creating electromagnetic field analysis...")
        try:
            visualizer.plot_electromagnetic_field_analysis(
                time_series_data,
                save_path=output_path / "electromagnetic_field_analysis.png"
            )
        except Exception as e:
            print(f"  Warning: Electromagnetic field analysis failed: {e}")
        
        print("  Enhanced physics visualizations completed!")
    else:
        print("  No time series data available for enhanced physics visualization")

def create_multistage_visualizations(config_file, time_series_data, summary_data, output_dir, visualizer):
    """
    Create specialized visualizations for multi-stage coilgun results.
    Enhanced with new physics data visualization.
    
    Args:
        config_file: Path to multi-stage configuration file
        time_series_data: Aggregated time series data from all stages
        summary_data: Summary data from multi-stage simulation
        output_dir: Directory to save visualizations
        visualizer: CoilgunFieldVisualizer instance
    """
    print("Creating multi-stage specific visualizations...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Extract stage information
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    num_stages = config["multi_stage"]["num_stages"]
    stage_transitions = time_series_data.get('stage_transitions', [])
    
    # NEW: Create enhanced physics visualizations first
    print("Creating enhanced physics analysis plots...")
    create_enhanced_physics_visualizations(time_series_data, output_path, visualizer)
    
    # 1. Create aggregated time series plots with stage markers
    print("Creating multi-stage time series plots...")
    create_multistage_time_series_plots(time_series_data, summary_data, stage_transitions, output_path)
    
    # 2. Create velocity progression plot
    print("Creating velocity progression visualization...")
    create_velocity_progression_plot(summary_data, output_path)
    
    # 3. Create efficiency comparison plot
    print("Creating efficiency comparison...")
    create_efficiency_comparison_plot(summary_data, output_path)
    
    # 4. Create stage-by-stage performance summary
    print("Creating stage performance summary...")
    create_stage_performance_summary(summary_data, output_path)
    
    # 5. Extract stage-specific current data from individual stages
    stage_currents = []
    stage_labels = []
    
    # Get stage-specific current data from summary
    stage_results = summary_data.get('stage_results', [])
    if stage_results:
        print(f"Extracting current data for {len(stage_results)} stages...")
        for i, stage_data in enumerate(stage_results):
            stage_num = i + 1
            max_current = stage_data.get('max_current_A', 0)
            
            if max_current > 0:
                # Use the actual max current for each stage
                stage_currents.append(max_current)
                stage_labels.append(f"stage_{stage_num}_{max_current:.0f}A")
                print(f"  Stage {stage_num}: {max_current:.0f}A")
            else:
                # Fallback for missing stage data
                estimated_current = 200 + (stage_num * 100)  # Rough estimate
                stage_currents.append(estimated_current)
                stage_labels.append(f"stage_{stage_num}_{estimated_current:.0f}A_estimated")
                print(f"  Stage {stage_num}: {estimated_current:.0f}A (estimated)")
    else:
        # Fallback to aggregated data approach
        print("No individual stage data available, using aggregated approach...")
        if time_series_data is not None:
            # Get current data (handle both CSV and numpy formats)
            if 'current_A' in time_series_data:
                current_data = np.array(time_series_data['current_A'])
            elif 'current' in time_series_data:
                current_data = np.array(time_series_data['current'])
            else:
                current_data = None
            
            if current_data is not None and len(current_data) > 0:
                # Extract meaningful current values
                max_current = np.max(np.abs(current_data))
                avg_current = np.mean(np.abs(current_data[current_data != 0]))
                
                # Create stages based on estimated values
                for stage_num in range(1, num_stages + 1):
                    current_factor = stage_num / num_stages  # Scale by stage number
                    stage_current = avg_current * (0.5 + current_factor)  # Range from 50% to 150% of avg
                    stage_currents.append(stage_current)
                    stage_labels.append(f"stage_{stage_num}_{stage_current:.0f}A_estimated")
            else:
                # Ultimate fallback
                for stage_num in range(1, num_stages + 1):
                    current = 100 + (stage_num * 100)
                    stage_currents.append(current)
                    stage_labels.append(f"stage_{stage_num}_{current}A_default")
        else:
            # No data at all
            for stage_num in range(1, num_stages + 1):
                current = 100 + (stage_num * 100)
                stage_currents.append(current)
                stage_labels.append(f"stage_{stage_num}_{current}A_default")
    
    # 6. Create comprehensive field profile comparison across all stages
    print(f"Creating field visualization for all {len(stage_currents)} stages...")
    visualizer.plot_onaxis_field_profile(
        current_values=stage_currents,
        save_path=output_path / "multistage_all_stages_field_profile.png"
    )
    
    # 7. Create individual stage visualizations
    print("Creating comprehensive field visualizations for each stage...")
    try:
        # Create stage-specific subdirectories and visualizations
        for stage_num, (current, label) in enumerate(zip(stage_currents, stage_labels), 1):
            stage_dir = output_path / f"stage_{stage_num}_field_visualizations"
            stage_dir.mkdir(exist_ok=True)
            
            print(f"   Creating visualizations for Stage {stage_num} ({current:.0f}A)...")
            
            # 3D field visualization for this stage
            visualizer.plot_3d_field_visualization(
                current, 
                save_path=stage_dir / f"3d_field_visualization_{label}.png",
                interactive=False,
                show_field_lines=True,
                show_coil=True,
                projectile_position=visualizer.physics.initial_position
            )
            
            # 2D field contour plots for this stage
            field_data = visualizer.calculate_bfield_map_2d(current, num_z=60, num_r=30)
            visualizer.plot_bfield_contours(
                field_data, 
                save_path=stage_dir / f"bfield_contours_{label}.png",
                show_projectile=True,
                projectile_position=visualizer.physics.initial_position
            )
            
            # 3D surface plot for this stage
            visualizer.plot_bfield_3d(
                field_data,
                save_path=stage_dir / f"bfield_3d_surface_{label}.png"
            )
            
            # Stage-specific on-axis field profile
            visualizer.plot_onaxis_field_profile(
                current_values=[current],
                save_path=stage_dir / f"onaxis_field_profile_{label}.png"
            )
        
        # 8. Create comparison visualizations in main directory
        print("Creating stage comparison visualizations...")
        
        # Combined 3D visualization with representative currents (max 4 to avoid clutter)
        comparison_currents = stage_currents
        comparison_labels = stage_labels
        
        if len(stage_currents) > 4:
            # Select representative stages: first, middle, second-to-last, last
            indices = [0, len(stage_currents)//2, len(stage_currents)-2, len(stage_currents)-1]
            comparison_currents = [stage_currents[i] for i in indices]
            comparison_labels = [f"comparison_{stage_labels[i]}" for i in indices]
        
        for current, label in zip(comparison_currents, comparison_labels):
            print(f"   Creating comparison visualization for {current:.0f}A...")
            visualizer.plot_3d_field_visualization(
                current, 
                save_path=output_path / f"3d_field_visualization_{label}.png",
                interactive=False,
                show_field_lines=True,
                show_coil=True,
                projectile_position=visualizer.physics.initial_position
            )
        
        print(f"Field visualizations complete for all {len(stage_currents)} stages!")
    except Exception as e:
        print(f"Warning: 3D visualization creation failed: {e}")
    
    # 9. Create comprehensive stage comparison plot
    print("Creating stage field comparison plot...")
    try:
        create_stage_comparison_field_plot(stage_currents, stage_labels, output_path, visualizer)
    except Exception as e:
        print(f"Warning: Stage comparison plot creation failed: {e}")
    
    print(f"Multi-stage visualizations saved to: {output_path}")


def create_multistage_time_series_plots(time_series_data, summary_data, stage_transitions, output_path):
    """Create time series plots with stage transition markers."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Multi-Stage Coilgun Simulation Results', fontsize=16, fontweight='bold')
    
    # Check if we have CSV-style keys or numpy-style keys
    if 'time_s' in time_series_data:
        # CSV format
        t = np.array(time_series_data['time_s']) * 1000  # Convert to milliseconds
        current = time_series_data['current_A']
        velocity = time_series_data['velocity_ms']
        position = np.array(time_series_data['position_m']) * 1000  # Convert to mm
        force = time_series_data['force_N']
        energy_cap = time_series_data['energy_capacitor_J']
        energy_kin = time_series_data['energy_kinetic_J']
        power = time_series_data['power_W']
    else:
        # Numpy format (multistage)
        t = np.array(time_series_data['time']) * 1000  # Convert to milliseconds
        current = time_series_data['current']
        velocity = time_series_data['velocity']
        position = np.array(time_series_data['position']) * 1000  # Convert to mm
        force = time_series_data['force']
        energy_cap = time_series_data['energy_capacitor']
        energy_kin = time_series_data['energy_kinetic']
        power = time_series_data['power']
    
    # Current vs time
    axes[0, 0].plot(t, current, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Current (A)')
    axes[0, 0].set_title('Current vs Time (All Stages)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity vs time
    axes[0, 1].plot(t, velocity, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity vs Time (All Stages)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Position vs time
    axes[1, 0].plot(t, position, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Position (mm)')
    axes[1, 0].set_title('Position vs Time (All Stages)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Force vs time
    axes[1, 1].plot(t, force, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time (ms)')
    axes[1, 1].set_ylabel('Force (N)')
    axes[1, 1].set_title('Magnetic Force vs Time (All Stages)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Energy vs time
    axes[2, 0].plot(t, energy_cap, 'c-', linewidth=2, label='Capacitor')
    axes[2, 0].plot(t, energy_kin, 'orange', linewidth=2, label='Kinetic')
    axes[2, 0].set_xlabel('Time (ms)')
    axes[2, 0].set_ylabel('Energy (J)')
    axes[2, 0].set_title('Energy vs Time (All Stages)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Power vs time
    axes[2, 1].plot(t, power, 'purple', linewidth=2)
    axes[2, 1].set_xlabel('Time (ms)')
    axes[2, 1].set_ylabel('Power (W)')
    axes[2, 1].set_title('Power vs Time (All Stages)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Add stage transition markers to all plots
    for transition_time in stage_transitions:
        transition_ms = transition_time * 1000
        for ax in axes.flat:
            ax.axvline(transition_ms, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Add stage labels
    if len(stage_transitions) > 0:
        stage_times = [0] + list(stage_transitions) + [t[-1]]
        for i, (start_time, end_time) in enumerate(zip(stage_times[:-1], stage_times[1:])):
            mid_time = (start_time + end_time) / 2
            axes[0, 0].text(mid_time, axes[0, 0].get_ylim()[1] * 0.9, f'Stage {i+1}', 
                           ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / "multistage_time_series.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_velocity_progression_plot(summary_data, output_path):
    """Create velocity progression visualization."""
    summary = summary_data.get('summary', {})
    stage_velocities = summary.get('stage_final_velocities_ms', [])
    
    if len(stage_velocities) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = list(range(1, len(stage_velocities) + 1))
    cumulative_velocities = [0] + stage_velocities
    
    # Bar plot of velocity gains
    velocity_gains = [cumulative_velocities[i+1] - cumulative_velocities[i] for i in range(len(stage_velocities))]
    bars = ax.bar(stages, velocity_gains, alpha=0.7, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'][:len(stages)])
    
    # Line plot of cumulative velocity
    ax2 = ax.twinx()
    ax2.plot(stages, stage_velocities, 'ro-', linewidth=3, markersize=8, label='Cumulative Velocity')
    
    ax.set_xlabel('Stage Number')
    ax.set_ylabel('Velocity Gain (m/s)')
    ax2.set_ylabel('Cumulative Velocity (m/s)')
    ax.set_title('Velocity Progression Through Stages')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, gain) in enumerate(zip(bars, velocity_gains)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{gain:.1f} m/s', ha='center', va='bottom')
    
    # Add cumulative velocity labels
    for i, velocity in enumerate(stage_velocities):
        ax2.text(i+1, velocity + 2, f'{velocity:.1f} m/s', ha='center', va='bottom')
    
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_path / "velocity_progression.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_efficiency_comparison_plot(summary_data, output_path):
    """Create efficiency comparison visualization."""
    summary = summary_data.get('summary', {})
    stage_efficiencies = summary.get('stage_efficiencies_percent', [])
    overall_efficiency = summary.get('overall_efficiency_percent', 0)
    
    if len(stage_efficiencies) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stages = list(range(1, len(stage_efficiencies) + 1))
    
    # Bar plot of individual stage efficiencies
    bars = ax.bar(stages, stage_efficiencies, alpha=0.7, color=['lightblue', 'lightgreen', 'orange', 'pink', 'lightcoral'][:len(stages)])
    
    # Add overall efficiency line
    ax.axhline(overall_efficiency, color='red', linestyle='--', linewidth=2, label=f'Overall Efficiency ({overall_efficiency:.1f}%)')
    
    ax.set_xlabel('Stage Number')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Efficiency Comparison by Stage')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for bar, efficiency in zip(bars, stage_efficiencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{efficiency:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / "efficiency_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_stage_performance_summary(summary_data, output_path):
    """Create a comprehensive stage performance summary table."""
    summary = summary_data.get('summary', {})
    
    stage_velocities = summary.get('stage_final_velocities_ms', [])
    stage_efficiencies = summary.get('stage_efficiencies_percent', [])
    stage_durations = summary.get('stage_durations_s', [])
    
    if len(stage_velocities) == 0 or len(stage_efficiencies) == 0 or len(stage_durations) == 0:
        return
    
    # Create performance summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    velocity_gains = [stage_velocities[0]] + [stage_velocities[i] - stage_velocities[i-1] for i in range(1, len(stage_velocities))]
    
    for i in range(len(stage_velocities)):
        table_data.append([
            f"Stage {i+1}",
            f"{velocity_gains[i]:.1f} m/s",
            f"{stage_velocities[i]:.1f} m/s",
            f"{stage_efficiencies[i]:.1f}%",
            f"{stage_durations[i]*1000:.1f} ms"
        ])
    
    # Add totals row
    total_duration = sum(stage_durations)
    final_velocity = stage_velocities[-1]
    overall_efficiency = summary.get('overall_efficiency_percent', 0)
    
    table_data.append([
        "TOTAL",
        f"{final_velocity:.1f} m/s",
        f"{final_velocity:.1f} m/s",
        f"{overall_efficiency:.1f}%",
        f"{total_duration*1000:.1f} ms"
    ])
    
    headers = ["Stage", "Velocity Gain", "Cumulative Velocity", "Efficiency", "Duration"]
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    table[(len(table_data), 0)].set_facecolor('#40466e')
    table[(len(table_data), 1)].set_facecolor('#40466e')
    table[(len(table_data), 2)].set_facecolor('#40466e')
    table[(len(table_data), 3)].set_facecolor('#40466e')
    table[(len(table_data), 4)].set_facecolor('#40466e')
    
    # Header styling
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Make the totals row bold
    for i in range(len(headers)):
        table[(len(table_data), i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Multi-Stage Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(output_path / "stage_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_visualization_suite(config_file, simulation_results=None, 
                                           time_series_data=None,
                                           output_dir="comprehensive_visualizations"):
    """
    Create a comprehensive suite of visualizations including enhanced physics analysis.
    
    Args:
        config_file: Path to coilgun configuration file
        simulation_results: CoilgunSimulation results (optional)
        time_series_data: Time series data dict (optional)
        output_dir: Directory to save visualization files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creating comprehensive visualization suite...")
    
    # Initialize physics engine and visualizer
    physics = CoilgunPhysicsEngine(config_file)
    visualizer = CoilgunFieldVisualizer(physics)
    
    # NEW: Create enhanced physics visualizations if data available
    physics_data = None
    if simulation_results is not None and hasattr(simulation_results, 'results'):
        physics_data = simulation_results.results
    elif time_series_data is not None:
        physics_data = time_series_data
    
    if physics_data is not None:
        print("\n0. Creating enhanced physics analysis...")
        create_enhanced_physics_visualizations(physics_data, output_path, visualizer)
    
    # Extract actual current values using robust helper function
    currents, data_source, max_current, avg_current = extract_actual_current_data(
        time_series_data=time_series_data,
        simulation_results=simulation_results,
        physics_engine=physics
    )
    
    current_labels = [f"{c:.0f}A" for c in currents]
    
    if "circuit-based estimate" in data_source:
        print(f"Using circuit-based current estimates: {current_labels}")
        print(f"  {data_source}")
    elif "ultimate_fallback" in data_source:
        print(f"⚠️  Using fallback current values: {current_labels}")
        print(f"  No simulation data available")
    else:
        print(f"✓ Using actual simulation currents: {current_labels}")
        print(f"  Data source: {data_source}")
        print(f"  Peak current: {max_current:.1f}A, Average: {avg_current:.1f}A")
    
    # 1. 3D Field Visualization
    print("\n1. Creating 3D magnetic field visualization...")
    
    for current, label in zip(currents, current_labels):
        print(f"   Creating 3D visualization for {current:.0f}A...")
        visualizer.plot_3d_field_visualization(
            current, 
            save_path=output_path / f"3d_field_visualization_{label}.png",
            interactive=False,
            projectile_position=physics.initial_position
        )
    
    # 2. Traditional 2D field analysis
    print("\n2. Creating 2D field analysis...")
    for current, label in zip(currents, current_labels):
        print(f"   Calculating 2D field for {current:.0f}A...")
        field_data = visualizer.calculate_bfield_map_2d(current, num_z=80, num_r=40)
        
        # 2D contour plots
        visualizer.plot_bfield_contours(
            field_data, 
            save_path=output_path / f"bfield_contours_{label}.png",
            show_projectile=True,
            projectile_position=physics.initial_position
        )
        
        # 3D surface plot
        visualizer.plot_bfield_3d(
            field_data,
            save_path=output_path / f"bfield_3d_surface_{label}.png"
        )
    
    # 3. On-axis field profiles
    print("\n3. Creating on-axis field profiles...")
    visualizer.plot_onaxis_field_profile(
        current_values=currents,
        save_path=output_path / "onaxis_field_profiles.png"
    )
    
    # 4. If simulation results provided, create animations
    if simulation_results is not None:
        print("\n4. Creating field evolution animations...")
        
        # 2D field evolution animation
        try:
            visualizer.animate_field_evolution(
                simulation_results,
                save_path=output_path / "field_evolution_2d.gif",
                interval=100
            )
        except Exception as e:
            print(f"   Warning: Field evolution animation failed: {e}")
        
        # 3D projectile motion animation
        print("\n5. Creating 3D projectile motion animation...")
        try:
            visualizer.animate_3d_projectile_motion(
                simulation_results,
                save_path=output_path / "projectile_motion_3d.gif",
                interval=150
            )
        except Exception as e:
            print(f"   Warning: 3D projectile animation failed: {e}")
    
    print(f"\nComprehensive visualization suite complete!")
    print(f"Files saved to: {output_path.absolute()}")
    
    return visualizer


def create_field_visualization_suite(config_file, output_dir="field_visualizations"):
    """
    Create a complete suite of magnetic field visualizations.
    
    Args:
        config_file: Path to coilgun configuration file
        output_dir: Directory to save visualization files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creating comprehensive magnetic field visualization suite...")
    
    # Initialize physics engine and visualizer
    physics = CoilgunPhysicsEngine(config_file)
    visualizer = CoilgunFieldVisualizer(physics)
    
    # 1. Static field analysis at different currents
    print("\n1. Creating static field analysis...")
    
    # Use reasonable current values based on the capacitor energy
    initial_voltage = physics.initial_voltage
    estimated_max_current = initial_voltage / physics.total_resistance  # Approximate peak current
    
    currents = [
        estimated_max_current * 0.3,  # 30% of estimated max
        estimated_max_current * 0.7,  # 70% of estimated max
        estimated_max_current         # Estimated max current
    ]
    current_labels = [f"{c:.0f}A" for c in currents]
    print(f"Using estimated currents based on circuit parameters: {current_labels}")
    
    for current, label in zip(currents, current_labels):
        print(f"   Calculating field for {current:.0f}A...")
        field_data = visualizer.calculate_bfield_map_2d(current, num_z=80, num_r=40)
        
        # 2D contour plots
        visualizer.plot_bfield_contours(
            field_data, 
            save_path=output_path / f"bfield_contours_{label}.png",
            show_projectile=True,
            projectile_position=physics.initial_position
        )
        
        # 3D surface plot
        visualizer.plot_bfield_3d(
            field_data,
            save_path=output_path / f"bfield_3d_{label}.png"
        )
    
    # 2. On-axis field profiles
    print("\n2. Creating on-axis field profiles...")
    visualizer.plot_onaxis_field_profile(
        current_values=currents,
        save_path=output_path / "onaxis_field_profiles.png"
    )
    
    # 3. Dynamic simulation with field animation
    print("\n3. Running simulation for field animation...")
    sim = CoilgunSimulation(config_file)
    results = sim.run_simulation(save_data=True, verbose=False)
    
    print("\n4. Creating field evolution animation...")
    anim = visualizer.animate_field_evolution(
        sim,
        save_path=output_path / "field_evolution.gif",
        interval=100
    )
    
    print(f"\nVisualization suite complete! Files saved to: {output_path.absolute()}")
    
    return visualizer, sim, results


def find_results_directories():
    """Find all results directories in the project directory (single and multi-stage)"""
    current_dir = Path(".")
    result_dirs = []
    
    for item in current_dir.iterdir():
        if item.is_dir():
            # Check if it's a results directory by looking for expected files
            config_file = item / "simulation_config.json"
            single_stage_summary = item / "simulation_summary.json"
            multi_stage_summary = item / "multistage_simulation_summary.json"
            
            # Accept directory if it has config and either type of summary
            if config_file.exists() and (single_stage_summary.exists() or multi_stage_summary.exists()):
                result_dirs.append(item)
    
    return sorted(result_dirs)

def select_results_directory():
    """Interactive selection of results directory"""
    import sys
    
    # Check if results directory was provided as command line argument
    if len(sys.argv) >= 2:
        results_dir = sys.argv[1]
        if Path(results_dir).exists():
            # Check if it's a valid results directory
            config_file = Path(results_dir) / "simulation_config.json"
            if config_file.exists():
                return results_dir
            else:
                print(f"Warning: '{results_dir}' doesn't appear to be a results directory.")
                print("Searching for available results directories...\n")
        else:
            print(f"Warning: Specified directory '{results_dir}' not found.")
            print("Searching for available results directories...\n")
    
    # Find available results directories
    result_dirs = find_results_directories()
    
    if not result_dirs:
        print("No simulation results directories found in the current directory.")
        print("Please run a simulation with 'python solve.py' first to generate results.")
        sys.exit(1)
    
    # Present options to user
    print("Available simulation results directories:")
    print("-" * 50)
    for i, results_dir in enumerate(result_dirs, 1):
        # Try to read summary from results directory
        try:
            # Check for multi-stage first
            multi_stage_summary = results_dir / "multistage_simulation_summary.json"
            single_stage_summary = results_dir / "simulation_summary.json"
            
            if multi_stage_summary.exists():
                # Multi-stage results
                with open(multi_stage_summary, 'r') as f:
                    data = json.load(f)
                    summary = data.get('summary', {})
                    final_velocity = summary.get('final_velocity_ms', 'N/A')
                    efficiency = summary.get('overall_efficiency_percent', 'N/A')
                    num_stages = summary.get('num_stages', 'N/A')
                
                print(f"{i}. {results_dir.name} (Multi-stage)")
                print(f"   Stages: {num_stages}")
                print(f"   Final velocity: {final_velocity} m/s")
                print(f"   Overall efficiency: {efficiency}%")
                
                # Show stage progression
                if 'stage_final_velocities_ms' in summary:
                    velocities = summary['stage_final_velocities_ms']
                    velocity_str = " → ".join([f"{v:.1f}" for v in velocities])
                    print(f"   Velocity progression: {velocity_str} m/s")
                    
            elif single_stage_summary.exists():
                # Single-stage results
                with open(single_stage_summary, 'r') as f:
                    data = json.load(f)
                    summary = data.get('summary', {})
                    final_velocity = summary.get('final_velocity_ms', 'N/A')
                    efficiency = summary.get('efficiency_percent', 'N/A')
                    exit_reason = data.get('simulation_info', {}).get('exit_reason', 'N/A')
                
                print(f"{i}. {results_dir.name}")
                print(f"   Final velocity: {final_velocity} m/s")
                print(f"   Efficiency: {efficiency}%")
                print(f"   Exit reason: {exit_reason}")
            else:
                print(f"{i}. {results_dir.name}")
                print(f"   (Unable to read summary)")
        except Exception as e:
            print(f"{i}. {results_dir.name}")
            print(f"   (Error reading summary: {e})")
        print()
    
    # Get user selection
    while True:
        try:
            choice = input(f"Select results directory (1-{len(result_dirs)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(result_dirs):
                selected_dir = result_dirs[choice_num - 1]
                print(f"Selected: {selected_dir.name}\n")
                return str(selected_dir)
            else:
                print(f"Please enter a number between 1 and {len(result_dirs)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def load_simulation_from_results(results_dir):
    """
    Load simulation data from a results directory (single or multi-stage).
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        tuple: (config_file_path, time_series_data, summary_data, is_multi_stage)
    """
    results_path = Path(results_dir)
    
    # Load configuration
    config_file = results_path / "simulation_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found in {results_dir}")
    
    # Check if this is multi-stage
    with open(config_file, 'r') as f:
        config = json.load(f)
    is_multi_stage = config.get("multi_stage", {}).get("enabled", False)
    
    # Load summary (different files for single vs multi-stage)
    if is_multi_stage:
        summary_file = results_path / "multistage_simulation_summary.json"
    else:
        summary_file = results_path / "simulation_summary.json"
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Load time series data (different files for single vs multi-stage)
    time_series_data = None
    
    if is_multi_stage:
        npz_file = results_path / "multistage_time_series_data.npz"
        csv_file = results_path / "multistage_time_series_data.csv"
    else:
        npz_file = results_path / "time_series_data.npz"
        csv_file = results_path / "time_series_data.csv"
    
    if npz_file.exists():
        # Load from compressed numpy file
        with np.load(npz_file) as data:
            time_series_data = {key: data[key] for key in data.files}
    elif csv_file.exists():
        # Load from CSV file
        try:
            import pandas as pd
            df = pd.read_csv(csv_file)
            time_series_data = {col: df[col].values for col in df.columns}
        except ImportError:
            # Fallback CSV reading without pandas
            import csv
            time_series_data = {}
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                if fieldnames is not None:
                    data_lists = {fieldname: [] for fieldname in fieldnames}
                    for row in reader:
                        for key, value in row.items():
                            data_lists[key].append(float(value))
                    time_series_data = {key: np.array(values) for key, values in data_lists.items()}
    
    return str(config_file), time_series_data, summary_data, is_multi_stage

def main():
    """Main function to create visualizations from results directory"""
    import sys
    import json
    
    try:
        # Check command line arguments for special modes
        if len(sys.argv) >= 2:
            if sys.argv[1] == '--3d':
                # Special 3D visualization mode
                print("=" * 60)
                print("3D COILGUN VISUALIZATION MODE")
                print("=" * 60)
                
                if len(sys.argv) >= 3:
                    config_file = sys.argv[2]
                else:
                    try:
                        config_file = input("Enter config file path: ").strip()
                    except KeyboardInterrupt:
                        print("\nOperation cancelled by user.")
                        sys.exit(0)
                
                if not Path(config_file).exists():
                    print(f"Error: Config file '{config_file}' not found.")
                    sys.exit(1)
                
                # Ask user if they want to proceed with 3D visualization
                print(f"\nReady to create 3D visualizations for: {Path(config_file).name}")
                try:
                    proceed = input("This may take several minutes. Do you want to proceed? (Y/n): ").strip().lower()
                    if proceed in ['n', 'no', 'q', 'quit']:
                        print("3D visualization cancelled by user.")
                        sys.exit(0)
                    elif proceed == '' or proceed in ['y', 'yes']:
                        pass  # Continue
                    else:
                        print("Invalid input. Proceeding with visualization...")
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    sys.exit(0)
                
                # Run simulation and create comprehensive 3D visualizations
                print("Running simulation for 3D visualization...")
                sim = CoilgunSimulation(config_file)
                results = sim.run_simulation(save_data=True, verbose=True)
                
                # Create comprehensive 3D visualization suite
                output_dir = f"3d_visualizations_{Path(config_file).stem}"
                visualizer = create_comprehensive_visualization_suite(
                    config_file, 
                    simulation_results=sim,
                    output_dir=output_dir
                )
                
                print(f"\n3D visualizations complete! Check: {output_dir}/")
                return
        
        # Standard mode - select results directory
        results_dir = select_results_directory()
        
        print("=" * 60)
        print("COILGUN VISUALIZATION SUITE")
        print("=" * 60)
        print(f"Results directory: {results_dir}")
        
        # Ask user if they want to proceed with visualization
        print(f"\nReady to create visualizations for: {Path(results_dir).name}")
        try:
            proceed = input("Do you want to proceed? (Y/n): ").strip().lower()
            if proceed in ['n', 'no', 'q', 'quit']:
                print("Visualization cancelled by user.")
                sys.exit(0)
            elif proceed == '' or proceed in ['y', 'yes']:
                pass  # Continue
            else:
                print("Invalid input. Proceeding with visualization...")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)
        
        print("\nStarting visualization...")
        
    except KeyboardInterrupt:
        print("\n\nVisualization cancelled by user (Ctrl+C)")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)
    
    try:
        # Load simulation data from results directory
        print("Loading simulation data...")
        config_file, time_series_data, summary_data, is_multi_stage = load_simulation_from_results(results_dir)
        
        # Create output directory based on results directory name
        results_name = Path(results_dir).name
        output_dir = f"visualizations_{results_name}"
        
        # Print simulation summary
        summary = summary_data.get('summary', {})
        
        if is_multi_stage:
            print(f"Multi-stage configuration detected")
            print(f"Number of stages: {summary.get('num_stages', 'N/A')}")
            print(f"Final velocity: {summary.get('final_velocity_ms', 'N/A')} m/s")
            print(f"Overall efficiency: {summary.get('overall_efficiency_percent', 'N/A')}%")
            print(f"Total initial energy: {summary.get('total_initial_energy_J', 'N/A')} J")
            print(f"Max current: {summary.get('max_current_A', 'N/A')} A")
            
            # Print stage progression
            if 'stage_final_velocities_ms' in summary:
                print(f"Velocity progression:")
                for i, velocity in enumerate(summary['stage_final_velocities_ms']):
                    print(f"  After stage {i+1}: {velocity:.1f} m/s")
        else:
            print(f"Single-stage configuration detected")
            print(f"Final velocity: {summary.get('final_velocity_ms', 'N/A')} m/s")
            print(f"Efficiency: {summary.get('efficiency_percent', 'N/A')}%")
            print(f"Max current: {summary.get('max_current_A', 'N/A')} A")
        
        # Initialize visualizer with first stage config (or single stage)
        if is_multi_stage:
            # For multi-stage, we need to create a temporary single-stage config for the visualizer
            # We'll use the first stage configuration for field visualizations
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            first_stage = config["stages"][0]
            temp_config = {}
            
            # Build single-stage config from first stage
            for key in ["coil", "capacitor", "simulation", "circuit_model", "magnetic_model", "output"]:
                if key in first_stage:
                    temp_config[key] = first_stage[key]
                elif key in config.get("shared", {}):
                    temp_config[key] = config["shared"][key]
            
            temp_config["projectile"] = config["shared"]["projectile"]
            
            # Save temporary config
            temp_config_file = "temp_visualization_config.json"
            with open(temp_config_file, 'w') as f:
                json.dump(temp_config, f, indent=4)
            
            physics = CoilgunPhysicsEngine(temp_config_file)
            
            # Clean up temp file
            Path(temp_config_file).unlink()
        else:
            physics = CoilgunPhysicsEngine(config_file)
        
        visualizer = CoilgunFieldVisualizer(physics)
        
        if time_series_data is not None:
            print("Creating visualizations with time series data...")
            
            if is_multi_stage:
                # Create multi-stage visualizations
                create_multistage_visualizations(
                    config_file, time_series_data, summary_data, output_dir, visualizer
                )
            else:
                # Create single-stage visualizations (original behavior)
                # Create simulation object to use existing plotting methods
                sim = CoilgunSimulation(config_file)
                
                # Map CSV column names to expected result keys (enhanced with new physics data)
                csv_to_result_mapping = {
                    'time_s': 'time',
                    'charge_C': 'charge', 
                    'current_A': 'current',
                    'position_m': 'position',
                    'velocity_ms': 'velocity',
                    'force_N': 'force',
                    'inductance_H': 'inductance',
                    'power_W': 'power',
                    'energy_capacitor_J': 'energy_capacitor',
                    'energy_kinetic_J': 'energy_kinetic',
                    # Enhanced physics data mapping
                    'force_gradient': 'force_gradient',
                    'force_reluctance': 'force_reluctance', 
                    'force_lorentz': 'force_lorentz',
                    'force_maxwell': 'force_maxwell',
                    'force_eddy': 'force_eddy',
                    'force_total': 'force_total',
                    'eddy_current_magnitude': 'eddy_current_magnitude',
                    'skin_depth': 'skin_depth',
                    'power_loss_resistive': 'power_loss_resistive',
                    'power_loss_eddy': 'power_loss_eddy',
                    'power_mechanical': 'power_mechanical',
                    'magnetic_field': 'magnetic_field',
                    'inductance_gradient': 'inductance_gradient',
                    'saturation_factor': 'saturation_factor',
                    'permeability_effective': 'permeability_effective',
                    'energy_conservation': 'energy_conservation',
                    'frequency_content': 'frequency_content',
                    'temperature_rise': 'temperature_rise'
                }
                
                # Populate results with loaded data
                for csv_key, result_key in csv_to_result_mapping.items():
                    if csv_key in time_series_data:
                        sim.results[result_key] = time_series_data[csv_key]
                
                # Also handle direct numpy array format (from .npz files) 
                # that may have different key names
                for key in time_series_data.keys():
                    if key not in csv_to_result_mapping and key not in sim.results:
                        sim.results[key] = time_series_data[key]
                
                # Create detailed plots using existing methods
                print(f"Creating detailed plots in: {output_dir}/")
                sim.plot_results(save_plots=True, output_dir=output_dir)
                
                # Create comprehensive 3D visualizations
                print("Creating comprehensive 3D magnetic field visualizations...")
                create_comprehensive_visualization_suite(
                    config_file, 
                    simulation_results=sim,
                    time_series_data=time_series_data,
                    output_dir=output_dir
                )
                
                # Create individual 3D field visualization using actual simulation data
                print("Creating static 3D field visualization...")
                
                # Extract actual current for visualization
                max_current = 300  # fallback
                if 'current' in sim.results and sim.results['current'] is not None:
                    current_data = np.array(sim.results['current'])
                    if len(current_data) > 0:
                        max_current = np.max(np.abs(current_data))
                        print(f"Using actual peak current from simulation: {max_current:.1f}A")
                    else:
                        print("No current data in simulation results, using fallback")
                else:
                    print("No current data available, using fallback")
                
                initial_position = sim.results['position'][0] if sim.results.get('position') is not None else physics.initial_position
                
                visualizer.plot_3d_field_visualization(
                    current=max_current,
                    save_path=Path(output_dir) / "3d_field_comprehensive.png",
                    interactive=False,
                    show_field_lines=True,
                    show_coil=True,
                    projectile_position=initial_position
                )
            
        else:
            print("No time series data available. Creating basic field visualizations...")
            # Create field visualization suite without animation
            create_field_visualization_suite(config_file, output_dir)
            
            # Create 3D field visualization using circuit-based estimation
            print("Creating 3D field visualization...")
            estimated_current = physics.initial_voltage / physics.total_resistance
            print(f"Using circuit-based current estimate: {estimated_current:.0f}A")
            visualizer.plot_3d_field_visualization(
                current=estimated_current,
                save_path=Path(output_dir) / "3d_field_static.png",
                interactive=False,
                projectile_position=physics.initial_position
            )
        
        print("\n" + "="*50)
        print("VISUALIZATION COMPLETE")
        print("="*50)
        print(f"Files saved to: {output_dir}/")
        print("Generated visualizations:")
        
        if is_multi_stage:
            print("- Enhanced electromagnetic physics analysis plots")
            print("- Detailed force decomposition analysis")
            print("- Electromagnetic field analysis with saturation tracking")
            print("- Multi-stage time series plots with stage transitions")
            print("- Velocity progression through stages")
            print("- Efficiency comparison by stage")
            print("- Stage performance summary table")
            print(f"- Comprehensive field and force profiles for all {len(summary_data.get('stage_results', []))} stages")
            print("- Individual stage field visualizations in stage_X_field_visualizations/ subdirectories")
            print("- Combined stage comparison plots showing all stages together")
            print("- 3D field visualizations for each stage")
            print("- 2D field contour plots for each stage")
            print("- Stage-specific on-axis field and force profiles")
            if time_series_data is None:
                print("- Basic field visualizations")
        else:
            print("- Enhanced electromagnetic physics analysis plots")
            print("- Detailed force decomposition with component breakdown")
            print("- Electromagnetic field analysis with saturation and permeability tracking")
            print("- Energy conservation and efficiency analysis")
            print("- Eddy current effects and skin depth analysis")
            print("- Power loss decomposition (resistive vs eddy)")
            print("- Frequency content analysis")
            print("- Temperature rise estimation")
            print("- Simulation result plots (if time series data available)")
            print("- 2D magnetic field contour plots")
            print("- 3D field surface plots") 
            print("- 3D comprehensive field visualization with field lines")
            print("- On-axis field profiles")
            print("- Field evolution animations (if time series data available)")
            print("- 3D projectile motion animation (if time series data available)")
        
        print("\nFor advanced 3D-only mode, run:")
        print("python view.py --3d <config_file>")
        print("\nCheck the output directory for all visualization files!")
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user (Ctrl+C)")
        print("Visualization results may be incomplete.")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nVisualization terminated due to error.")
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle signals gracefully"""
    print("\n\nReceived interrupt signal.")
    print("Cleaning up and exiting gracefully...")
    sys.exit(0)


def create_demo_visualization():
    """
    Create a demonstration visualization using a default configuration.
    """
    print("Creating demonstration coilgun visualization...")
    
    # Create a demo configuration
    demo_config = {
        "coil": {
            "inner_diameter": 0.015,
            "length": 0.075,
            "wire_gauge_awg": 16,
            "num_layers": 6,
            "wire_material": "Copper",
            "packing_factor": 0.85,
            "insulation_thickness": 5e-5
        },
        "projectile": {
            "diameter": 0.012,
            "length": 0.025,
            "material": "Low_Carbon_Steel",
            "initial_position": -0.05,
            "initial_velocity": 0.0
        },
        "capacitor": {
            "capacitance": 0.003,
            "initial_voltage": 400,
            "esr": 0.01,
            "esl": 5e-8
        },
        "simulation": {
            "time_span": [0, 0.02],
            "max_step": 1e-6,
            "tolerance": 1e-9,
            "method": "RK45"
        },
        "circuit_model": {
            "switch_resistance": 0.001,
            "switch_inductance": 1e-8,
            "parasitic_capacitance": 1e-11,
            "include_skin_effect": False,
            "include_proximity_effect": False
        },
        "magnetic_model": {
            "calculation_method": "biot_savart",
            "axial_discretization": 1000,
            "radial_discretization": 100,
            "include_saturation": False,
            "include_hysteresis": False
        },
        "output": {
            "save_trajectory": True,
            "save_current_profile": True,
            "save_field_data": False,
            "print_progress": True,
            "save_interval": 100
        }
    }
    
    # Save demo config
    demo_config_file = "demo_coilgun_config.json"
    with open(demo_config_file, 'w') as f:
        json.dump(demo_config, f, indent=4)
    
    print(f"Demo configuration saved to: {demo_config_file}")
    
    # Run simulation
    print("Running demonstration simulation...")
    sim = CoilgunSimulation(demo_config_file)
    results = sim.run_simulation(save_data=True, verbose=True)
    
    # Create comprehensive visualizations
    print("Creating comprehensive demonstration visualizations...")
    output_dir = "demo_visualizations"
    visualizer = create_comprehensive_visualization_suite(
        demo_config_file,
        simulation_results=sim,
        output_dir=output_dir
    )
    
    print(f"\nDemo visualization complete!")
    print(f"Check the '{output_dir}' directory for all visualization files.")
    print(f"Config file: {demo_config_file}")
    
    return visualizer, sim


def create_stage_comparison_field_plot(stage_currents, stage_labels, output_path, visualizer):
    """Create a comprehensive comparison of field profiles across all stages."""
    if not stage_currents:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Multi-Stage Magnetic Field Comparison', fontsize=16, fontweight='bold')
    
    # Axial positions for field calculation
    z_points = np.linspace(-visualizer.physics.coil_length * 0.5, 
                          visualizer.physics.coil_length * 1.5, 300, retstep=False)
    z_mm = z_points * 1000  # Convert to mm
    
    # Color map for stages
    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(stage_currents), retstep=False))
    
    # Plot 1: On-axis magnetic field for each stage
    max_field = 0
    for i, (current, label) in enumerate(zip(stage_currents, stage_labels)):
        stage_num = i + 1
        
        # Calculate field along axis
        bz_values = []
        for z in z_points:
            bz = visualizer.physics.magnetic_field_solenoid_on_axis(z, current)
            bz_values.append(bz * 1000)  # Convert to mT
        
        max_field = max(max_field, max(bz_values))
        
        # Plot field profile
        ax1.plot(z_mm, bz_values, linewidth=2, color=colors[i], 
                label=f'Stage {stage_num} ({current:.0f}A)')
    
    ax1.set_xlabel('Axial Position (mm)')
    ax1.set_ylabel('Magnetic Field Bz (mT)')
    ax1.set_title('On-Axis Magnetic Field by Stage')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add coil boundaries
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Coil boundaries')
    ax1.axvline(visualizer.physics.coil_length * 1000, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(visualizer.physics.coil_center * 1000, color='orange', linestyle=':', alpha=0.7, label='Coil center')
    
    # Plot 2: Force comparison for each stage
    positions = np.linspace(-0.05, visualizer.physics.coil_length + 0.05, 200, retstep=False)
    positions_mm = positions * 1000
    
    max_force = 0
    for i, (current, label) in enumerate(zip(stage_currents, stage_labels)):
        stage_num = i + 1
        
        # Calculate force profile
        forces = []
        for pos in positions:
            force = visualizer.physics.magnetic_force_with_circuit_logic(current, pos)
            forces.append(force)
        
        max_force = max(max_force, max(forces) if forces else 0)
        
        # Plot force profile
        ax2.plot(positions_mm, forces, linewidth=2, color=colors[i], 
                label=f'Stage {stage_num} ({current:.0f}A)')
    
    ax2.set_xlabel('Projectile Position (mm)')
    ax2.set_ylabel('Magnetic Force (N)')
    ax2.set_title('Magnetic Force vs Position by Stage')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add reference lines
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(visualizer.physics.coil_length * 1000, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(visualizer.physics.coil_center * 1000, color='orange', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / "stage_field_force_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stage comparison plot saved with {len(stage_currents)} stages")


if __name__ == "__main__":
    import os
    import signal
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check for demo mode
        if len(sys.argv) >= 2 and sys.argv[1] == '--demo':
            create_demo_visualization()
        else:
            main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        print("Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnhandled error: {e}")
        sys.exit(1)
