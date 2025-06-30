"""
3D plotting and visualization for coilgun systems.

This module provides 3D visualization capabilities including magnetic field
surfaces, field lines, coil geometry, and interactive 3D plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple, Dict, Any
import time

from .core import BaseVisualizer
from .fields import FieldLineTracer


class Plot3DVisualizer(BaseVisualizer):
    """Class for 3D plotting and visualization."""
    
    def plot_3d_field_visualization(self, current, save_path=None, interactive=True,
                                   show_field_lines=True, show_coil=True, 
                                   projectile_position=None):
        """
        Create comprehensive 3D magnetic field visualization.
        
        Args:
            current: Coil current (A)
            save_path: Path to save the plot
            interactive: Whether to show interactive plot
            show_field_lines: Whether to show magnetic field lines
            show_coil: Whether to show 3D coil geometry
            projectile_position: Position of projectile (m)
        """
        from .fields import MagneticFieldCalculator
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        print("Calculating 3D magnetic field distribution...")
        
        # Use much smaller grid for interactive mode to prevent freezing
        if interactive:
            # Very small grid for interactive use (fast)
            num_z, num_x, num_y = 8, 6, 6  # Only 288 points instead of 4500
            timeout_seconds = 15  # Maximum time allowed
        else:
            # Larger grid for saved plots (slower but better quality)
            num_z, num_x, num_y = 12, 10, 10  # 1200 points
            timeout_seconds = 60
            
        start_time = time.time()
        
        try:
            # Calculate 3D field data with timeout protection
            calculator = MagneticFieldCalculator(self.physics)
            field_data_3d = calculator.calculate_bfield_3d(
                current, num_z=num_z, num_x=num_x, num_y=num_y
            )
            
            calculation_time = time.time() - start_time
            print(f"Field calculation completed in {calculation_time:.1f}s")
            
            if calculation_time > timeout_seconds:
                print(f"Warning: Calculation took {calculation_time:.1f}s (longer than {timeout_seconds}s timeout)")
            
        except Exception as e:
            print(f"Error in 3D field calculation: {e}")
            print("Creating simplified visualization without field data...")
            self._create_simplified_3d_visualization(ax, current, show_coil, projectile_position)
            self.save_figure(fig, save_path)
            if interactive:
                plt.show()
            else:
                plt.close()
            return
        
        # Extract field data
        X, Y, Z = field_data_3d['X'], field_data_3d['Y'], field_data_3d['Z']
        B_magnitude = field_data_3d['B_magnitude']
        Bx, By, Bz = field_data_3d['Bx'], field_data_3d['By'], field_data_3d['Bz']
        
        # Create field magnitude visualization using cross-sections
        print("Creating field magnitude cross-sections...")
        self._plot_field_cross_sections(ax, X, Y, Z, B_magnitude)
        
        # Add magnetic field vectors (simplified) - only for non-interactive to save time
        if show_field_lines and not interactive:
            print("Adding magnetic field vectors...")
            self._add_field_vectors_3d(ax, X, Y, Z, Bx, By, Bz, B_magnitude)
        elif show_field_lines and interactive:
            print("Skipping field vectors in interactive mode for better performance")
        
        # Add 3D coil geometry (simplified)
        if show_coil:
            print("Adding 3D coil geometry...")
            self._add_simple_coil_geometry(ax)
        
        # Add projectile if specified
        if projectile_position is not None:
            print("Adding projectile geometry...")
            self._add_simple_projectile_geometry(ax, projectile_position)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Magnetic Field Visualization (I = {current:.1f} A)', fontsize=14)
        
        # Set reasonable limits
        max_dim = 0.04  # 40mm
        ax.set_xlim([-max_dim, max_dim])
        ax.set_ylim([-max_dim, max_dim])
        ax.set_zlim([0, 0.08])  # 80mm
        
        # Add field strength information
        max_field = np.max(B_magnitude)
        ax.text2D(0.02, 0.98, f'Max Field: {max_field:.2e} T', transform=ax.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 verticalalignment='top')
        
        # Add performance info for debugging
        if interactive:
            ax.text2D(0.02, 0.02, f'Grid: {num_x}×{num_y}×{num_z}, Time: {calculation_time:.1f}s', 
                     transform=ax.transAxes,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                     verticalalignment='bottom', fontsize=10)
        
        self.save_figure(fig, save_path)
        
        if interactive:
            plt.show()
        else:
            plt.close()
    
    def _create_simplified_3d_visualization(self, ax, current, show_coil=True, projectile_position=None):
        """
        Create a simplified 3D visualization when full field calculation fails.
        
        Args:
            ax: 3D axes object
            current: Coil current (A)
            show_coil: Whether to show coil geometry
            projectile_position: Position of projectile (m)
        """
        print("Creating simplified 3D visualization...")
        
        # Add coil geometry
        if show_coil:
            self._add_simple_coil_geometry(ax)
        
        # Add projectile if specified
        if projectile_position is not None:
            self._add_simple_projectile_geometry(ax, projectile_position)
        
        # Add some representative field lines (simplified analytical model)
        self._add_analytical_field_representation(ax, current)
        
        # Set labels and title
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Coil Visualization (I = {current:.1f} A) - Simplified', fontsize=14)
        
        # Set reasonable limits
        max_dim = 0.04  # 40mm
        ax.set_xlim([-max_dim, max_dim])
        ax.set_ylim([-max_dim, max_dim])
        ax.set_zlim([0, 0.08])  # 80mm
        
        ax.text2D(0.02, 0.98, 'Simplified visualization (field calculation skipped)', 
                 transform=ax.transAxes,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                 verticalalignment='top')
    
    def _add_analytical_field_representation(self, ax, current):
        """
        Add a simplified analytical representation of magnetic field lines.
        
        Args:
            ax: 3D axes object
            current: Coil current (A)
        """
        if not self.physics:
            return
            
        try:
            # Simple dipole-like field lines
            coil_radius = (self.physics.coil_inner_radius + self.physics.coil_outer_radius) / 2
            coil_center_z = self.physics.coil_length / 2
            
            # Create simple field line representation
            n_lines = 8
            for i in range(n_lines):
                angle = 2 * np.pi * i / n_lines
                
                # Create curved lines representing field
                t = np.linspace(0, 1, 20)
                
                # Inside coil - axial field
                x_inner = coil_radius * 0.5 * np.cos(angle) * np.ones_like(t)
                y_inner = coil_radius * 0.5 * np.sin(angle) * np.ones_like(t)
                z_inner = t * self.physics.coil_length
                
                # Outside coil - curved field
                expansion_factor = 1 + 2 * t
                x_outer = coil_radius * expansion_factor * np.cos(angle) 
                y_outer = coil_radius * expansion_factor * np.sin(angle)
                z_outer = coil_center_z + (t - 0.5) * self.physics.coil_length * 2
                
                # Plot field lines
                ax.plot(x_inner * 1000, y_inner * 1000, z_inner * 1000, 
                       'b-', alpha=0.6, linewidth=2)
                ax.plot(x_outer * 1000, y_outer * 1000, z_outer * 1000, 
                       'r--', alpha=0.4, linewidth=1)
                       
        except Exception as e:
            print(f"Warning: Could not add analytical field representation: {e}")
    
    def _plot_field_cross_sections(self, ax, X, Y, Z, B_magnitude):
        """
        Plot magnetic field cross-sections instead of problematic isosurfaces.
        
        Args:
            ax: 3D axes object
            X, Y, Z: Coordinate arrays
            B_magnitude: Field magnitude array
        """
        max_field = np.max(B_magnitude)
        print(f"Debug: Maximum field magnitude = {max_field:.6e} T")
        
        if max_field < 1e-10:  # Much lower threshold for realistic coilgun fields
            print("Warning: Field magnitude is too small for visualization")
            return
        
        # Get shape information
        nx, ny, nz = X.shape
        
        # Create cross-sections at different Z planes
        z_indices = [nz//4, nz//2, 3*nz//4]  # 25%, 50%, 75% along Z
        colors = ['blue', 'green', 'red']
        alphas = [0.6, 0.7, 0.8]
        
        sections_created = 0
        for i, (z_idx, color, alpha) in enumerate(zip(z_indices, colors, alphas)):
            if z_idx < nz:
                # Extract 2D cross-section
                X_slice = X[:, :, z_idx]
                Y_slice = Y[:, :, z_idx]
                Z_slice = Z[:, :, z_idx]
                B_slice = B_magnitude[:, :, z_idx]
                
                # Create contour plot on this Z plane
                try:
                    # Convert coordinates to mm for better visualization
                    X_mm = X_slice * 1000
                    Y_mm = Y_slice * 1000
                    Z_mm = Z_slice * 1000
                    
                    # Create levels with better scaling for actual field values
                    max_slice_field = np.max(B_slice)
                    if max_slice_field > 1e-10:
                        levels = np.linspace(0, max_slice_field, 10)
                        contour = ax.contourf(X_mm, Y_mm, B_slice, 
                                            zdir='z', offset=np.mean(Z_mm), 
                                            levels=levels, cmap='plasma', alpha=alpha)
                        sections_created += 1
                        print(f"Created cross-section {i} at z={np.mean(Z_mm):.1f}mm, max field={max_slice_field:.6e}T")
                        
                except Exception as e:
                    print(f"Warning: Could not create cross-section {i}: {e}")
                    
                    # Fallback: simple surface plot
                    try:
                        # Scale the field for visualization
                        field_scale = 1000 if max_field < 0.01 else 100
                        ax.plot_surface(X_slice * 1000, Y_slice * 1000, 
                                      Z_slice * 1000 + B_slice * field_scale, 
                                      alpha=alpha/2, color=color)
                        sections_created += 1
                        print(f"Created fallback surface {i}")
                    except Exception as e2:
                        print(f"Warning: Fallback surface also failed: {e2}")
        
        if sections_created == 0:
            print("Warning: No cross-sections could be created")
        else:
            print(f"Successfully created {sections_created} field cross-sections")
    
    def _add_field_vectors_3d(self, ax, X, Y, Z, Bx, By, Bz, B_magnitude):
        """
        Add simplified magnetic field vectors to 3D plot.
        
        Args:
            ax: 3D axes object
            X, Y, Z: Coordinate arrays
            Bx, By, Bz: Field component arrays
            B_magnitude: Field magnitude array
        """
        # Heavily subsample for clarity (every 3rd point)
        step = 3
        X_sub = X[::step, ::step, ::step]
        Y_sub = Y[::step, ::step, ::step]
        Z_sub = Z[::step, ::step, ::step]
        Bx_sub = Bx[::step, ::step, ::step]
        By_sub = By[::step, ::step, ::step]
        Bz_sub = Bz[::step, ::step, ::step]
        B_mag_sub = B_magnitude[::step, ::step, ::step]
        
        # Only show vectors where field is significant
        threshold = np.max(B_magnitude) * 0.1
        mask = B_mag_sub > threshold
        
        if np.any(mask):
            # Normalize vector lengths for visualization
            scale_factor = 0.01 / np.max(B_mag_sub[mask]) if np.max(B_mag_sub[mask]) > 0 else 1
            
            try:
                ax.quiver(X_sub[mask] * 1000, Y_sub[mask] * 1000, Z_sub[mask] * 1000,
                         Bx_sub[mask] * scale_factor, By_sub[mask] * scale_factor, Bz_sub[mask] * scale_factor,
                         length=0.01, normalize=False, alpha=0.7, color='red')
            except Exception as e:
                print(f"Warning: Could not add field vectors: {e}")
    
    def _add_simple_coil_geometry(self, ax):
        """
        Add simplified 3D coil geometry that won't cause rendering errors.
        
        Args:
            ax: 3D axes object
        """
        if not self.physics:
            return
        
        try:
            # Coil parameters
            inner_radius = self.physics.coil_inner_radius * 1000  # Convert to mm
            outer_radius = self.physics.coil_outer_radius * 1000  # Convert to mm
            coil_length = self.physics.coil_length * 1000  # Convert to mm
            
            # Create simple cylindrical representation
            theta = np.linspace(0, 2*np.pi, 50)
            z_coil = np.array([0, coil_length])
            
            # Inner cylinder
            for z in z_coil:
                x_inner = inner_radius * np.cos(theta)
                y_inner = inner_radius * np.sin(theta)
                z_inner = np.full_like(x_inner, z)
                ax.plot(x_inner, y_inner, z_inner, 'b-', linewidth=2, alpha=0.8)
            
            # Outer cylinder
            for z in z_coil:
                x_outer = outer_radius * np.cos(theta)
                y_outer = outer_radius * np.sin(theta)
                z_outer = np.full_like(x_outer, z)
                ax.plot(x_outer, y_outer, z_outer, 'b-', linewidth=2, alpha=0.8)
            
            # Connect inner and outer at a few points
            for i in range(0, len(theta), len(theta)//8):
                x_connect = [inner_radius * np.cos(theta[i]), outer_radius * np.cos(theta[i])]
                y_connect = [inner_radius * np.sin(theta[i]), outer_radius * np.sin(theta[i])]
                for z in z_coil:
                    z_connect = [z, z]
                    ax.plot(x_connect, y_connect, z_connect, 'b-', linewidth=1, alpha=0.6)
                    
        except Exception as e:
            print(f"Warning: Could not add coil geometry: {e}")
    
    def _add_simple_projectile_geometry(self, ax, position):
        """
        Add simplified 3D projectile geometry.
        
        Args:
            ax: 3D axes object
            position: Projectile z-position (m)
        """
        if not self.physics:
            return
        
        try:
            # Projectile parameters
            projectile_radius = getattr(self.physics, 'projectile_radius', 0.002) * 1000  # Convert to mm
            projectile_length = 0.01 * 1000  # 10mm length in mm
            position_mm = position * 1000  # Convert to mm
            
            # Create simple cylindrical projectile
            theta = np.linspace(0, 2*np.pi, 20)
            z_proj = np.linspace(position_mm, position_mm + projectile_length, 5)
            
            # Projectile outline
            for z in [z_proj[0], z_proj[-1]]:
                x_proj = projectile_radius * np.cos(theta)
                y_proj = projectile_radius * np.sin(theta)
                z_proj_circle = np.full_like(x_proj, z)
                ax.plot(x_proj, y_proj, z_proj_circle, 'r-', linewidth=3, alpha=0.9)
            
            # Connect ends with lines
            for i in range(0, len(theta), len(theta)//4):
                x_line = [projectile_radius * np.cos(theta[i]), projectile_radius * np.cos(theta[i])]
                y_line = [projectile_radius * np.sin(theta[i]), projectile_radius * np.sin(theta[i])]
                z_line = [z_proj[0], z_proj[-1]]
                ax.plot(x_line, y_line, z_line, 'r-', linewidth=2, alpha=0.9)
                
        except Exception as e:
            print(f"Warning: Could not add projectile geometry: {e}")
    
    def plot_bfield_3d_surface(self, field_data, save_path=None):
        """
        Create 3D surface plot of magnetic field magnitude.
        
        Args:
            field_data: 3D field data dictionary
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y, Z = field_data['X'], field_data['Y'], field_data['Z']
        B_magnitude = field_data['B_magnitude']
        
        # Create surface plot at z=0 plane (midpoint)
        z_mid_idx = Z.shape[2] // 2
        X_slice = X[:, :, z_mid_idx]
        Y_slice = Y[:, :, z_mid_idx]
        B_slice = B_magnitude[:, :, z_mid_idx]
        
        surf = ax.plot_surface(X_slice, Y_slice, B_slice, cmap='plasma', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, label='|B| (T)', shrink=0.8)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('|B| (T)')
        ax.set_title('3D Magnetic Field Surface', fontsize=14)
        
        self.save_figure(fig, save_path)
        plt.show()
    
    def _plot_field_isosurface(self, ax, X, Y, Z, B_magnitude):
        """
        Plot magnetic field isosurfaces in 3D.
        
        Args:
            ax: 3D axes object
            X, Y, Z: Coordinate arrays
            B_magnitude: Field magnitude array
        """
        # Find appropriate isosurface level
        max_field = np.max(B_magnitude)
        if max_field < 1e-15:
            print("Warning: Field magnitude is too small for isosurface visualization")
            return
        
        # Use multiple isosurface levels
        levels = [max_field * 0.1, max_field * 0.3, max_field * 0.6]
        colors = ['blue', 'green', 'red']
        alphas = [0.2, 0.3, 0.5]
        
        for level, color, alpha in zip(levels, colors, alphas):
            if level > 1e-15:  # Only plot if level is significant
                try:
                    # Create isosurface contour
                    contour = ax.contour3D(X, Y, Z, B_magnitude, levels=[level], 
                                         colors=[color], alpha=alpha)
                    
                    # Set transparency for the contour (if it has the right attributes)
                    try:
                        if hasattr(contour, 'collections') and contour.collections:
                            for collection in contour.collections:
                                collection.set_alpha(alpha)
                    except (AttributeError, IndexError):
                        # Fallback: the contour object might handle transparency differently
                        pass
                        
                except Exception as e:
                    print(f"Warning: Could not create isosurface at level {level}: {e}")
    
    def _add_field_lines_3d(self, ax, field_data_3d):
        """
        Add magnetic field lines to 3D plot.
        
        Args:
            ax: 3D axes object
            field_data_3d: 3D field data dictionary
        """
        # Define starting points for field lines (around coil)
        if not self.physics:
            return
        
        coil_radius = (self.physics.coil_inner_radius + self.physics.coil_outer_radius) / 2
        start_points = []
        
        # Create field line starting points
        for theta in np.linspace(0, 2*np.pi, 8):
            for z_start in np.linspace(0, self.physics.coil_length, 3):
                x = coil_radius * np.cos(theta)
                y = coil_radius * np.sin(theta)
                start_points.append([x, y, z_start])
        
        # Trace field lines
        tracer = FieldLineTracer(field_data_3d)
        field_lines = tracer.trace_field_lines_3d(start_points)
        
        # Plot field lines
        for line in field_lines:
            if len(line) > 1:
                ax.plot(line[:, 0], line[:, 1], line[:, 2], 
                       'r-', linewidth=1, alpha=0.7)
    
    def _add_3d_coil_geometry(self, ax, num_turns_visual=20):
        """
        Add 3D coil geometry visualization.
        
        Args:
            ax: 3D axes object
            num_turns_visual: Number of turns to visualize
        """
        if not self.physics:
            return
        
        # Coil parameters
        inner_radius = self.physics.coil_inner_radius
        outer_radius = self.physics.coil_outer_radius
        coil_length = self.physics.coil_length
        
        # Create coil winding visualization
        theta = np.linspace(0, 2 * np.pi * num_turns_visual, 1000)
        z_coil = np.linspace(0, coil_length, len(theta))
        
        # Inner and outer coil boundaries
        x_inner = inner_radius * np.cos(theta)
        y_inner = inner_radius * np.sin(theta)
        x_outer = outer_radius * np.cos(theta)
        y_outer = outer_radius * np.sin(theta)
        
        # Plot coil windings
        ax.plot(x_inner, y_inner, z_coil, 'b-', linewidth=2, alpha=0.8, label='Inner Coil')
        ax.plot(x_outer, y_outer, z_coil, 'b-', linewidth=2, alpha=0.8, label='Outer Coil')
        
        # Add coil end faces
        theta_face = np.linspace(0, 2*np.pi, 50)
        r_face = np.linspace(inner_radius, outer_radius, 10)
        THETA_face, R_face = np.meshgrid(theta_face, r_face)
        X_face = R_face * np.cos(THETA_face)
        Y_face = R_face * np.sin(THETA_face)
        
        # Front face (z=0)
        Z_face_front = np.zeros_like(X_face)
        ax.plot_surface(X_face, Y_face, Z_face_front, alpha=0.3, color='lightblue')
        
        # Back face (z=coil_length)
        Z_face_back = np.full_like(X_face, coil_length)
        ax.plot_surface(X_face, Y_face, Z_face_back, alpha=0.3, color='lightblue')
    
    def _add_3d_projectile_geometry(self, ax, position):
        """
        Add 3D projectile geometry visualization.
        
        Args:
            ax: 3D axes object
            position: Projectile z-position (m)
        """
        if not self.physics:
            return
        
        # Projectile parameters
        projectile_radius = getattr(self.physics, 'projectile_radius', 0.002)
        projectile_length = getattr(self.physics, 'projectile_length', 0.01)
        
        # Create cylindrical projectile
        theta = np.linspace(0, 2*np.pi, 20)
        z_proj = np.linspace(position, position + projectile_length, 10)
        THETA_proj, Z_proj = np.meshgrid(theta, z_proj)
        
        X_proj = projectile_radius * np.cos(THETA_proj)
        Y_proj = projectile_radius * np.sin(THETA_proj)
        
        # Plot projectile surface
        ax.plot_surface(X_proj, Y_proj, Z_proj, alpha=0.8, color='red')
        
        # Add projectile end caps
        theta_cap = np.linspace(0, 2*np.pi, 20)
        r_cap = np.linspace(0, projectile_radius, 5)
        THETA_cap, R_cap = np.meshgrid(theta_cap, r_cap)
        X_cap = R_cap * np.cos(THETA_cap)
        Y_cap = R_cap * np.sin(THETA_cap)
        
        # Front cap
        Z_cap_front = np.full_like(X_cap, position)
        ax.plot_surface(X_cap, Y_cap, Z_cap_front, alpha=0.8, color='darkred')
        
        # Back cap
        Z_cap_back = np.full_like(X_cap, position + projectile_length)
        ax.plot_surface(X_cap, Y_cap, Z_cap_back, alpha=0.8, color='darkred')
    
    def _set_equal_aspect_3d(self, ax, field_data_3d):
        """
        Set equal aspect ratio for 3D plot.
        
        Args:
            ax: 3D axes object
            field_data_3d: Field data for determining limits
        """
        x_range = field_data_3d['x_range']
        y_range = field_data_3d['y_range']  
        z_range = field_data_3d['z_range']
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        
        # Try to set equal aspect ratio
        try:
            ax.set_box_aspect([1, 1, 2])  # Z-axis typically longer
        except AttributeError:
            # Fallback for older matplotlib versions
            pass

    def render_coil_geometry(self, save_path=None):
        """
        Render just the 3D coil geometry for visualization.
        
        Args:
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        print("Rendering 3D coil geometry...")
        self._add_simple_coil_geometry(ax)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Coil Geometry', fontsize=14)
        
        # Set reasonable limits based on coil dimensions
        if self.physics:
            max_radius = self.physics.coil_outer_radius * 1000 * 1.2  # 20% margin
            coil_length = self.physics.coil_length * 1000 * 1.2
            ax.set_xlim([-max_radius, max_radius])
            ax.set_ylim([-max_radius, max_radius])
            ax.set_zlim([0, coil_length])
        
        self.save_figure(fig, save_path)
        plt.show()

    # Comment out problematic methods to prevent errors
    def _plot_field_isosurface_OLD(self, ax, X, Y, Z, B_magnitude):
        """
        DISABLED: Old isosurface method that was causing errors.
        """
        print("Isosurface plotting disabled - using cross-sections instead")
        pass
    
    def _add_field_lines_3d_OLD(self, ax, field_data_3d):
        """
        DISABLED: Old field line tracing that was causing errors.
        """
        print("Field line tracing disabled - using vectors instead")
        pass
    
    def _add_3d_coil_geometry_OLD(self, ax, num_turns_visual=20):
        """
        DISABLED: Old coil geometry that was causing rendering errors.
        """
        print("Old coil geometry disabled - using simplified version")
        pass
    
    def _add_3d_projectile_geometry_OLD(self, ax, position):
        """
        DISABLED: Old projectile geometry that was causing rendering errors.
        """
        print("Old projectile geometry disabled - using simplified version")
        pass


class GeometryRenderer:
    """Class for rendering 3D coil and projectile geometries."""
    
    @staticmethod
    def create_3d_coil_geometry(physics_engine, num_turns_visual=20):
        """
        Create detailed 3D coil geometry data.
        
        Args:
            physics_engine: Physics engine with coil parameters
            num_turns_visual: Number of turns to visualize
            
        Returns:
            Dictionary with coil geometry data
        """
        if not physics_engine:
            return {}
        
        inner_radius = physics_engine.coil_inner_radius
        outer_radius = physics_engine.coil_outer_radius
        coil_length = physics_engine.coil_length
        
        # Create detailed coil winding
        theta = np.linspace(0, 2 * np.pi * num_turns_visual, 2000)
        z_winding = np.linspace(0, coil_length, len(theta))
        
        # Create multiple radial layers
        num_layers = 5
        coil_geometry = {
            'windings': [],
            'faces': []
        }
        
        for layer in range(num_layers):
            radius = inner_radius + (layer / (num_layers - 1)) * (outer_radius - inner_radius)
            x_winding = radius * np.cos(theta)
            y_winding = radius * np.sin(theta)
            
            coil_geometry['windings'].append({
                'x': x_winding,
                'y': y_winding,
                'z': z_winding,
                'radius': radius
            })
        
        # Create end faces
        theta_face = np.linspace(0, 2*np.pi, 50)
        r_face = np.linspace(inner_radius, outer_radius, 20)
        THETA_face, R_face = np.meshgrid(theta_face, r_face)
        X_face = R_face * np.cos(THETA_face)
        Y_face = R_face * np.sin(THETA_face)
        
        coil_geometry['faces'] = {
            'front': {'x': X_face, 'y': Y_face, 'z': np.zeros_like(X_face)},
            'back': {'x': X_face, 'y': Y_face, 'z': np.full_like(X_face, coil_length)}
        }
        
        return coil_geometry
    
    @staticmethod
    def create_3d_projectile_geometry(physics_engine, position):
        """
        Create detailed 3D projectile geometry data.
        
        Args:
            physics_engine: Physics engine with projectile parameters
            position: Projectile z-position
            
        Returns:
            Dictionary with projectile geometry data
        """
        if not physics_engine:
            return {}
        
        projectile_radius = getattr(physics_engine, 'projectile_radius', 0.002)
        projectile_length = getattr(physics_engine, 'projectile_length', 0.01)
        
        # Create cylindrical projectile surface
        theta = np.linspace(0, 2*np.pi, 30)
        z_proj = np.linspace(position, position + projectile_length, 20)
        THETA_proj, Z_proj = np.meshgrid(theta, z_proj)
        
        X_proj = projectile_radius * np.cos(THETA_proj)
        Y_proj = projectile_radius * np.sin(THETA_proj)
        
        # Create end caps
        theta_cap = np.linspace(0, 2*np.pi, 30)
        r_cap = np.linspace(0, projectile_radius, 10)
        THETA_cap, R_cap = np.meshgrid(theta_cap, r_cap)
        X_cap = R_cap * np.cos(THETA_cap)
        Y_cap = R_cap * np.sin(THETA_cap)
        
        projectile_geometry = {
            'surface': {'x': X_proj, 'y': Y_proj, 'z': Z_proj},
            'front_cap': {'x': X_cap, 'y': Y_cap, 'z': np.full_like(X_cap, position)},
            'back_cap': {'x': X_cap, 'y': Y_cap, 'z': np.full_like(X_cap, position + projectile_length)}
        }
        
        return projectile_geometry 