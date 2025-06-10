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
from scipy.integrate import odeint
from scipy.interpolate import griddata

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
    """
    
    def __init__(self, physics_engine):
        """
        Initialize the visualizer with a physics engine.
        
        Args:
            physics_engine: CoilgunPhysicsEngine instance
        """
        self.physics = physics_engine
        self.fig_size = (15, 10)
        
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
        z_points = np.linspace(z_range[0], z_range[1], num_z)
        r_points = np.linspace(r_range[0], r_range[1], num_r)
        Z, R = np.meshgrid(z_points, r_points)
        
        # Initialize field arrays
        Bz = np.zeros_like(Z)
        Br = np.zeros_like(Z)
        B_magnitude = np.zeros_like(Z)
        
        print(f"Calculating B-field on {num_z}×{num_r} grid...")
        
        # Calculate field using superposition of current loops
        # Discretize coil into current loops
        num_loops = max(50, int(self.physics.total_turns / 5))
        loop_positions = np.linspace(0, self.physics.coil_length, num_loops)
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
        z_points = np.linspace(z_range[0], z_range[1], num_z)
        x_points = np.linspace(x_range[0], x_range[1], num_x)
        y_points = np.linspace(y_range[0], y_range[1], num_y)
        
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
        loop_positions = np.linspace(0, self.physics.coil_length, num_loops)
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
        turns = np.linspace(0, num_turns_visual, 1000)
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
        theta = np.linspace(0, 2*np.pi, 20)
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
        ax = fig.add_subplot(111, projection='3d')
        
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
                               c=b_sample[mask] * 1000, cmap='plasma', alpha=0.3, s=10)
            
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
            theta_start = np.linspace(0, 2*np.pi, num_lines, endpoint=False)
            
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
                color = plt.cm.copper(i / len(coil_lines))
                ax.plot(coil_line[:, 0] * 1000, coil_line[:, 1] * 1000, coil_line[:, 2] * 1000,
                       color=color, linewidth=3, alpha=0.8)
        
        # Plot projectile
        if projectile_position is not None:
            print("Adding projectile geometry...")
            x_proj, y_proj, z_proj = self.create_3d_projectile_geometry(projectile_position)
            
            ax.plot_surface(x_proj * 1000, y_proj * 1000, z_proj * 1000,
                           color='red', alpha=0.8, linewidth=0)
        
        # Set labels and title
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title(f'3D Coilgun Magnetic Field Visualization (I = {current:.0f} A)', 
                     fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_range = max(
            self.physics.coil_outer_radius * 1000,
            self.physics.coil_length * 1000
        )
        ax.set_xlim([-max_range*1.2, max_range*1.2])
        ax.set_ylim([-max_range*1.2, max_range*1.2])
        ax.set_zlim([-max_range*0.3, max_range*1.8])
        
        # Improve viewing angle
        ax.view_init(elev=20, azim=45)
        
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
        ax = fig.add_subplot(111, projection='3d')
        
        Z = field_data['Z'] * 1000  # Convert to mm
        R = field_data['R'] * 1000
        B_mag = field_data['B_magnitude'] * 1000  # Convert to mT
        
        # Create 3D surface plot
        surf = ax.plot_surface(Z, R, B_mag, cmap='plasma', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour lines at the base
        ax.contour(Z, R, B_mag, zdir='z', offset=0, levels=20, cmap='plasma', alpha=0.5)
        
        ax.set_xlabel('Axial Position (mm)')
        ax.set_ylabel('Radial Position (mm)')
        ax.set_zlabel('Magnetic Field Magnitude (mT)')
        ax.set_title(f'3D Magnetic Field Distribution (I = {field_data["current"]:.0f} A)')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='|B| (mT)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D B-field plot saved to: {save_path}")
        
        plt.show()
    
    def plot_onaxis_field_profile(self, current_values=[100, 300, 500], save_path=None):
        """
        Plot magnetic field along the coil axis for different currents.
        
        Args:
            current_values: List of current values to plot (A)
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Axial positions
        z_points = np.linspace(-self.physics.coil_length, 2*self.physics.coil_length, 300)
        z_mm = z_points * 1000  # Convert to mm
        
        # Plot field for different currents
        for current in current_values:
            bz_values = []
            for z in z_points:
                bz = self.physics.magnetic_field_solenoid_on_axis(z, current)
                bz_values.append(bz * 1000)  # Convert to mT
            
            ax1.plot(z_mm, bz_values, linewidth=2, label=f'{current} A')
        
        ax1.set_xlabel('Axial Position (mm)')
        ax1.set_ylabel('Magnetic Field Bz (mT)')
        ax1.set_title('On-Axis Magnetic Field Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add coil boundaries
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7, label='Coil start')
        ax1.axvline(self.physics.coil_length * 1000, color='red', linestyle='--', alpha=0.7, label='Coil end')
        ax1.axvline(self.physics.coil_center * 1000, color='orange', linestyle=':', alpha=0.7, label='Coil center')
        
        # Plot force profile
        current = max(current_values)
        positions = np.linspace(-0.05, self.physics.coil_length + 0.05, 200)
        positions_mm = positions * 1000
        forces = []
        
        for pos in positions:
            force = self.physics.magnetic_force_ferromagnetic(current, pos)
            forces.append(force)
        
        ax2.plot(positions_mm, forces, 'purple', linewidth=2, label=f'Force @ {current} A')
        ax2.set_xlabel('Projectile Position (mm)')
        ax2.set_ylabel('Magnetic Force (N)')
        ax2.set_title('Magnetic Force vs Position')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add reference lines
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(self.physics.coil_length * 1000, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(self.physics.coil_center * 1000, color='orange', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"On-axis field profile saved to: {save_path}")
        
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
        frame_indices = np.linspace(0, len(time_data)-1, num_frames, dtype=int)
        
        print(f"Creating 3D animation with {num_frames} frames...")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Pre-render coil geometry
        coil_lines = self.create_3d_coil_geometry(num_turns_visual=6)
        
        def animate(frame_idx):
            ax.clear()
            
            idx = frame_indices[frame_idx]
            current = current_data[idx]
            position = position_data[idx]
            time = time_data[idx]
            
            # Plot coil
            for i, coil_line in enumerate(coil_lines):
                color = plt.cm.copper(i / len(coil_lines))
                ax.plot(coil_line[:, 0] * 1000, coil_line[:, 1] * 1000, coil_line[:, 2] * 1000,
                       color=color, linewidth=2, alpha=0.7)
            
            # Plot projectile
            x_proj, y_proj, z_proj = self.create_3d_projectile_geometry(position)
            ax.plot_surface(x_proj * 1000, y_proj * 1000, z_proj * 1000,
                           color='red', alpha=0.9, linewidth=0)
            
            # Add some field lines around projectile
            if current > 10:  # Only show field lines when current is significant
                # Simple field visualization - radial lines from coil center
                theta_lines = np.linspace(0, 2*np.pi, 8, endpoint=False)
                for theta in theta_lines:
                    r_line = np.linspace(0, self.physics.coil_outer_radius * 1.5, 20)
                    x_line = r_line * np.cos(theta) * 1000
                    y_line = r_line * np.sin(theta) * 1000
                    z_line = np.full_like(r_line, self.physics.coil_center * 1000)
                    
                    # Color by field strength (approximate)
                    colors = plt.cm.viridis(r_line / (self.physics.coil_outer_radius * 1.5))
                    for i in range(len(r_line)-1):
                        ax.plot([x_line[i], x_line[i+1]], [y_line[i], y_line[i+1]], 
                               [z_line[i], z_line[i+1]], color=colors[i], alpha=0.6)
            
            # Set labels and title
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(f'3D Coilgun Animation - t={time*1000:.1f}ms, I={current:.0f}A, v={simulation_results.results["velocity"][idx]:.1f}m/s')
            
            # Set consistent axis limits
            max_range = max(self.physics.coil_outer_radius * 1000, self.physics.coil_length * 1000)
            ax.set_xlim([-max_range*1.2, max_range*1.2])
            ax.set_ylim([-max_range*1.2, max_range*1.2])
            ax.set_zlim([-max_range*0.3, max_range*1.8])
            
            ax.view_init(elev=15, azim=frame_idx * 2)  # Slowly rotate view
        
        anim = FuncAnimation(fig, animate, frames=num_frames, interval=interval, blit=False)
        
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
        num_frames = min(100, len(time_data) // 10)  # Limit to 100 frames
        frame_indices = np.linspace(0, len(time_data)-1, num_frames, dtype=int)
        
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
            
            # Plot field magnitude
            im1 = ax1.contourf(Z, R, B_mag, levels=30, cmap='plasma')
            ax1.set_title(f'|B| Field (t = {frame["time"]*1000:.1f} ms)')
            ax1.set_xlabel('Position (mm)')
            ax1.set_ylabel('Radius (mm)')
            
            # Plot axial field
            im2 = ax2.contourf(Z, R, Bz, levels=30, cmap='RdBu_r')
            ax2.set_title(f'Bz Field (I = {frame["current"]:.0f} A)')
            ax2.set_xlabel('Position (mm)')
            ax2.set_ylabel('Radius (mm)')
            
            # Add coil and projectile geometry
            for ax in [ax1, ax2]:
                self._add_coil_geometry(ax)
                self._add_projectile_geometry(ax, frame['position'])
            
            # Plot current vs time
            current_history = [f['current'] for f in field_frames[:frame_idx+1]]
            time_history = [f['time']*1000 for f in field_frames[:frame_idx+1]]
            
            ax3.plot(time_history, current_history, 'b-', linewidth=2)
            ax3.axvline(frame['time']*1000, color='red', linestyle='--')
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('Current (A)')
            ax3.set_title('Current vs Time')
            ax3.grid(True, alpha=0.3)
            
            # Plot position vs time
            position_history = [f['position']*1000 for f in field_frames[:frame_idx+1]]
            
            ax4.plot(time_history, position_history, 'g-', linewidth=2)
            ax4.axvline(frame['time']*1000, color='red', linestyle='--')
            ax4.axhline(0, color='black', linestyle=':', alpha=0.5, label='Coil entrance')
            ax4.axhline(self.physics.coil_center*1000, color='orange', linestyle=':', alpha=0.5, label='Coil center')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Position (mm)')
            ax4.set_title('Projectile Position vs Time')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, animate, frames=len(field_frames), 
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"Animation saved to: {save_path}")
        
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


def create_comprehensive_visualization_suite(config_file, simulation_results=None, 
                                           output_dir="comprehensive_visualizations"):
    """
    Create a comprehensive suite of visualizations including 3D field lines and projectile motion.
    
    Args:
        config_file: Path to coilgun configuration file
        simulation_results: CoilgunSimulation results (optional)
        output_dir: Directory to save visualization files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Creating comprehensive visualization suite...")
    
    # Initialize physics engine and visualizer
    physics = CoilgunPhysicsEngine(config_file)
    visualizer = CoilgunFieldVisualizer(physics)
    
    # 1. 3D Field Visualization
    print("\n1. Creating 3D magnetic field visualization...")
    currents = [100, 300, 500]
    
    for current in currents:
        print(f"   Creating 3D visualization for {current}A...")
        visualizer.plot_3d_field_visualization(
            current, 
            save_path=output_path / f"3d_field_visualization_{current}A.png",
            interactive=False,
            projectile_position=physics.initial_position
        )
    
    # 2. Traditional 2D field analysis
    print("\n2. Creating 2D field analysis...")
    for current in currents:
        print(f"   Calculating 2D field for {current}A...")
        field_data = visualizer.calculate_bfield_map_2d(current, num_z=80, num_r=40)
        
        # 2D contour plots
        visualizer.plot_bfield_contours(
            field_data, 
            save_path=output_path / f"bfield_contours_{current}A.png",
            show_projectile=True,
            projectile_position=physics.initial_position
        )
        
        # 3D surface plot
        visualizer.plot_bfield_3d(
            field_data,
            save_path=output_path / f"bfield_3d_surface_{current}A.png"
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
        visualizer.animate_field_evolution(
            simulation_results,
            save_path=output_path / "field_evolution_2d.gif",
            interval=100
        )
        
        # 3D projectile motion animation
        print("\n5. Creating 3D projectile motion animation...")
        visualizer.animate_3d_projectile_motion(
            simulation_results,
            save_path=output_path / "projectile_motion_3d.gif",
            interval=150
        )
    
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
    currents = [100, 300, 500]
    
    for current in currents:
        print(f"   Calculating field for {current}A...")
        field_data = visualizer.calculate_bfield_map_2d(current, num_z=80, num_r=40)
        
        # 2D contour plots
        visualizer.plot_bfield_contours(
            field_data, 
            save_path=output_path / f"bfield_contours_{current}A.png",
            show_projectile=True,
            projectile_position=physics.initial_position
        )
        
        # 3D surface plot
        visualizer.plot_bfield_3d(
            field_data,
            save_path=output_path / f"bfield_3d_{current}A.png"
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
    """Find all results directories in the project directory"""
    current_dir = Path(".")
    result_dirs = []
    
    for item in current_dir.iterdir():
        if item.is_dir():
            # Check if it's a results directory by looking for expected files
            config_file = item / "simulation_config.json"
            summary_file = item / "simulation_summary.json"
            
            if config_file.exists() and summary_file.exists():
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
            summary_file = results_dir / "simulation_summary.json"
            with open(summary_file, 'r') as f:
                data = json.load(f)
                summary = data.get('summary', {})
                final_velocity = summary.get('final_velocity_ms', 'N/A')
                efficiency = summary.get('efficiency_percent', 'N/A')
                exit_reason = data.get('simulation_info', {}).get('exit_reason', 'N/A')
            
            print(f"{i}. {results_dir.name}")
            print(f"   Final velocity: {final_velocity} m/s")
            print(f"   Efficiency: {efficiency}%")
            print(f"   Exit reason: {exit_reason}")
        except:
            print(f"{i}. {results_dir.name}")
            print(f"   (Unable to read summary)")
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
    Load simulation data from a results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        tuple: (config_file_path, time_series_data, summary_data)
    """
    results_path = Path(results_dir)
    
    # Load configuration
    config_file = results_path / "simulation_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found in {results_dir}")
    
    # Load summary
    summary_file = results_path / "simulation_summary.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Load time series data
    time_series_data = None
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
                data_lists = {fieldname: [] for fieldname in reader.fieldnames}
                for row in reader:
                    for key, value in row.items():
                        data_lists[key].append(float(value))
                time_series_data = {key: np.array(values) for key, values in data_lists.items()}
    
    return str(config_file), time_series_data, summary_data

def main():
    """Main function to create visualizations from results directory"""
    import sys
    import json
    
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
                config_file = input("Enter config file path: ").strip()
            
            if not Path(config_file).exists():
                print(f"Error: Config file '{config_file}' not found.")
                sys.exit(1)
            
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
    
    try:
        # Load simulation data from results directory
        print("Loading simulation data...")
        config_file, time_series_data, summary_data = load_simulation_from_results(results_dir)
        
        # Create output directory based on results directory name
        results_name = Path(results_dir).name
        output_dir = f"visualizations_{results_name}"
        
        # Print simulation summary
        summary = summary_data.get('summary', {})
        print(f"Final velocity: {summary.get('final_velocity_ms', 'N/A')} m/s")
        print(f"Efficiency: {summary.get('efficiency_percent', 'N/A')}%")
        print(f"Max current: {summary.get('max_current_A', 'N/A')} A")
        
        # Initialize visualizer
        physics = CoilgunPhysicsEngine(config_file)
        visualizer = CoilgunFieldVisualizer(physics)
        
        if time_series_data is not None:
            print("Creating visualizations with time series data...")
            
            # Create simulation object to use existing plotting methods
            sim = CoilgunSimulation(config_file)
            
            # Map CSV column names to expected result keys
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
                'energy_kinetic_J': 'energy_kinetic'
            }
            
            # Populate results with loaded data
            for csv_key, result_key in csv_to_result_mapping.items():
                if csv_key in time_series_data:
                    sim.results[result_key] = time_series_data[csv_key]
            
            # Create detailed plots using existing methods
            print(f"Creating detailed plots in: {output_dir}/")
            sim.plot_results(save_plots=True, output_dir=output_dir)
            
            # Create comprehensive 3D visualizations
            print("Creating comprehensive 3D magnetic field visualizations...")
            create_comprehensive_visualization_suite(
                config_file, 
                simulation_results=sim,
                output_dir=output_dir
            )
            
            # Create individual 3D field visualization
            print("Creating static 3D field visualization...")
            max_current = np.max(sim.results['current']) if sim.results['current'] is not None else 300
            initial_position = sim.results['position'][0] if sim.results['position'] is not None else physics.initial_position
            
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
            
            # Create 3D field visualization
            print("Creating 3D field visualization...")
            visualizer.plot_3d_field_visualization(
                current=300,  # Default current
                save_path=Path(output_dir) / "3d_field_static.png",
                interactive=False,
                projectile_position=physics.initial_position
            )
        
        print("\n" + "="*50)
        print("VISUALIZATION COMPLETE")
        print("="*50)
        print(f"Files saved to: {output_dir}/")
        print("Generated visualizations:")
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
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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


if __name__ == "__main__":
    import os
    
    # Check for demo mode
    if len(sys.argv) >= 2 and sys.argv[1] == '--demo':
        create_demo_visualization()
    else:
        main()
