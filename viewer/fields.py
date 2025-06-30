"""
Magnetic field calculations and visualization for coilgun systems.

This module provides functions for calculating magnetic fields using Biot-Savart law,
creating field maps, and visualizing field distributions in 2D and 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Optional, Tuple, Dict, Any
import time

from .core import BaseVisualizer, VISUALIZATION_CONSTANTS


class MagneticFieldCalculator(BaseVisualizer):
    """Class for magnetic field calculations and visualization."""
    
    def __init__(self, physics_engine):
        """
        Initialize field calculator.
        
        Args:
            physics_engine: CoilgunPhysicsEngine instance
        """
        super().__init__(physics_engine)
    
    def calculate_bfield_map_2d(self, current, z_range=None, r_range=None, 
                               num_z=100, num_r=50, include_projectile=True, 
                               projectile_position=None):
        """
        Calculate 2D magnetic field map using Biot-Savart law.
        
        Args:
            current: Coil current (A)
            z_range: Z-axis range tuple (m)
            r_range: R-axis range tuple (m)
            num_z: Number of z points
            num_r: Number of r points
            include_projectile: Whether to include projectile effects
            projectile_position: Position of projectile (m)
            
        Returns:
            Dictionary with field data and coordinates
        """
        if z_range is None:
            z_range = VISUALIZATION_CONSTANTS['DEFAULT_Z_RANGE']
        if r_range is None:
            r_range = VISUALIZATION_CONSTANTS['DEFAULT_R_RANGE']
        
        # Create coordinate grids
        z_vals = np.linspace(z_range[0], z_range[1], num_z)
        r_vals = np.linspace(r_range[0], r_range[1], num_r)
        Z, R = np.meshgrid(z_vals, r_vals)
        
        # Calculate magnetic field at each point
        Bz = np.zeros_like(Z)
        Br = np.zeros_like(R)
        B_magnitude = np.zeros_like(Z)
        
        print(f"Calculating magnetic field map ({num_z}x{num_r} points)...")
        
        for i in range(num_r):
            for j in range(num_z):
                z = Z[i, j]
                r = R[i, j]
                
                # Calculate field using Biot-Savart law
                bz, br = self._biot_savart_total_field(z, r, current)
                
                Bz[i, j] = bz
                Br[i, j] = br
                B_magnitude[i, j] = np.sqrt(bz**2 + br**2)
        
        # Include projectile magnetic effects if specified
        if include_projectile and projectile_position is not None:
            Bz, Br, B_magnitude = self._add_projectile_field_effects(
                Z, R, Bz, Br, B_magnitude, projectile_position
            )
        
        return {
            'Z': Z,
            'R': R,
            'Bz': Bz,
            'Br': Br,
            'B_magnitude': B_magnitude,
            'z_range': z_range,
            'r_range': r_range
        }
    
    def calculate_bfield_3d(self, current, z_range=None, x_range=None, y_range=None,
                           num_z=50, num_x=30, num_y=30):
        """
        Calculate 3D magnetic field distribution.
        
        Args:
            current: Coil current (A)
            z_range: Z-axis range tuple (m)
            x_range: X-axis range tuple (m)
            y_range: Y-axis range tuple (m)
            num_z: Number of z points
            num_x: Number of x points
            num_y: Number of y points
            
        Returns:
            Dictionary with 3D field data
        """
        if z_range is None:
            z_range = VISUALIZATION_CONSTANTS['DEFAULT_Z_RANGE']
        if x_range is None:
            x_range = (-0.03, 0.03)
        if y_range is None:
            y_range = (-0.03, 0.03)
        
        # Create 3D coordinate grids
        z_vals = np.linspace(z_range[0], z_range[1], num_z)
        x_vals = np.linspace(x_range[0], x_range[1], num_x)
        y_vals = np.linspace(y_range[0], y_range[1], num_y)
        
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        
        # Calculate field components
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        Bz = np.zeros_like(Z)
        
        total_points = num_x * num_y * num_z
        print(f"Calculating 3D magnetic field ({num_x}x{num_y}x{num_z} = {total_points} points)...")
        
        # Progress tracking
        start_time = time.time()
        points_calculated = 0
        last_progress_time = start_time
        
        try:
            for i in range(num_x):
                # Progress reporting every 10% or every 2 seconds
                current_time = time.time()
                if (current_time - last_progress_time) > 2.0:  # Every 2 seconds
                    progress = (points_calculated / total_points) * 100
                    elapsed = current_time - start_time
                    estimated_total = elapsed / (points_calculated / total_points) if points_calculated > 0 else 0
                    remaining = estimated_total - elapsed
                    print(f"  Progress: {progress:.1f}% ({points_calculated}/{total_points} points), "
                          f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
                    last_progress_time = current_time
                
                for j in range(num_y):
                    for k in range(num_z):
                        x = X[i, j, k]
                        y = Y[i, j, k]
                        z = Z[i, j, k]
                        
                        # Convert to cylindrical coordinates
                        r = np.sqrt(x**2 + y**2)
                        phi = np.arctan2(y, x)
                        
                        # Calculate field in cylindrical coordinates with error handling
                        try:
                            bz, br = self._biot_savart_total_field(z, r, current)
                            
                            # Convert back to Cartesian coordinates
                            Bx[i, j, k] = br * np.cos(phi)
                            By[i, j, k] = br * np.sin(phi)
                            Bz[i, j, k] = bz
                            
                        except Exception as e:
                            # If individual point calculation fails, set to zero and continue
                            print(f"Warning: Field calculation failed at point ({i},{j},{k}): {e}")
                            Bx[i, j, k] = 0.0
                            By[i, j, k] = 0.0
                            Bz[i, j, k] = 0.0
                        
                        points_calculated += 1
                        
                        # Check for timeout (abort after reasonable time)
                        if (current_time - start_time) > 120:  # 2 minute timeout
                            print(f"Warning: 3D calculation timeout after {current_time - start_time:.1f}s")
                            print(f"Completed {points_calculated}/{total_points} points ({100*points_calculated/total_points:.1f}%)")
                            # Fill remaining points with zeros
                            remaining_slice = slice(i, None) if j == 0 and k == 0 else slice(None)
                            if i < num_x - 1:
                                Bx[remaining_slice, :, :] = 0.0
                                By[remaining_slice, :, :] = 0.0
                                Bz[remaining_slice, :, :] = 0.0
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
                
        except KeyboardInterrupt:
            print(f"\n3D calculation interrupted by user after {points_calculated}/{total_points} points")
            print("Using partial results...")
        except Exception as e:
            print(f"Error in 3D field calculation: {e}")
            print("Using partial results...")
        
        # Calculate field magnitude
        B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        calculation_time = time.time() - start_time
        print(f"3D field calculation completed in {calculation_time:.1f}s")
        print(f"Max field magnitude: {np.max(B_magnitude):.6e} T")
        
        return {
            'X': X, 'Y': Y, 'Z': Z,
            'Bx': Bx, 'By': By, 'Bz': Bz,
            'B_magnitude': B_magnitude,
            'x_range': x_range,
            'y_range': y_range,
            'z_range': z_range,
            'calculation_time': calculation_time,
            'points_calculated': points_calculated,
            'total_points': total_points
        }
    
    def _biot_savart_total_field(self, z, r, current):
        """
        Calculate total magnetic field using Biot-Savart law for multi-turn coil.
        
        Args:
            z: Axial position (m)
            r: Radial position (m)
            current: Coil current (A)
            
        Returns:
            Tuple of (Bz, Br) field components
        """
        if not self.physics:
            return 0.0, 0.0
        
        # Coil parameters
        coil_length = self.physics.coil_length
        inner_radius = self.physics.coil_inner_radius
        outer_radius = self.physics.coil_outer_radius
        num_turns = self.physics.num_turns
        
        # Calculate turn distribution
        num_radial_layers = max(1, int((outer_radius - inner_radius) / 0.0005))
        num_axial_layers = max(1, int(coil_length / 0.0005))
        turns_per_layer = num_turns / (num_radial_layers * num_axial_layers)
        
        total_bz = 0.0
        total_br = 0.0
        
        # Integrate over coil volume
        for i in range(num_radial_layers):
            for j in range(num_axial_layers):
                # Position of current loop
                loop_radius = inner_radius + (i + 0.5) * (outer_radius - inner_radius) / num_radial_layers
                loop_z = j * coil_length / num_axial_layers
                
                # Calculate field from this loop
                bz, br = self._biot_savart_circular_loop(z, r, loop_z, loop_radius, 
                                                       current * turns_per_layer)
                
                total_bz += bz
                total_br += br
        
        return total_bz, total_br
    
    def _biot_savart_circular_loop(self, z, r, loop_z, loop_radius, current):
        """
        Calculate magnetic field from a single circular current loop using robust formulas.
        
        Args:
            z: Field point z-coordinate (m)
            r: Field point r-coordinate (m)
            loop_z: Loop z-position (m)
            loop_radius: Loop radius (m)
            current: Loop current (A)
            
        Returns:
            Tuple of (Bz, Br) field components
        """
        mu0 = 4e-7 * np.pi  # Permeability of free space
        
        # Relative position
        z_rel = z - loop_z
        
        # Avoid singularities with more robust thresholds
        if r < 1e-6:  # Increased threshold for numerical stability
            r = 1e-6
        if np.abs(z_rel) < 1e-8:
            z_rel = 1e-8 if z_rel >= 0 else -1e-8
        
        # Special case: on-axis calculation (r ≈ 0)
        if r < 1e-5:
            # On-axis field is purely axial
            rho = np.sqrt(loop_radius**2 + z_rel**2)
            if rho < 1e-10:
                return 0.0, 0.0
            
            bz = (mu0 * current * loop_radius**2) / (2 * rho**3)
            br = 0.0  # No radial field on axis
            return bz, br
        
        # For off-axis points, use more stable formulation
        alpha_squared = loop_radius**2 + r**2 + z_rel**2
        beta_squared = loop_radius**2 + r**2 - z_rel**2
        
        # Calculate distance measures
        rho_plus = np.sqrt(alpha_squared + 2 * loop_radius * r)
        rho_minus = np.sqrt(alpha_squared - 2 * loop_radius * r)
        
        if rho_plus < 1e-10 or rho_minus < 1e-10:
            return 0.0, 0.0
        
        # More stable axial field calculation
        # Using the exact formula but with better numerical handling
        factor = (mu0 * current) / (4 * np.pi)
        
        # For axial field (Bz)
        bz_factor = (loop_radius**2 - r**2 - z_rel**2)
        denom_z = (alpha_squared + 2 * loop_radius * r)**(3/2)
        
        if denom_z > 1e-15:
            bz = factor * (2 * loop_radius**2 / denom_z)
        else:
            bz = 0.0
        
        # For radial field (Br) - use more stable formulation
        if np.abs(z_rel) < 1e-8:
            br = 0.0  # No radial field in the plane of the loop
        else:
            # Use a different, more stable formula for Br
            br_numerator = 2 * loop_radius * z_rel
            br_denominator = r * (alpha_squared + 2 * loop_radius * r)**(3/2)
            
            if br_denominator > 1e-15:
                br = factor * (br_numerator / br_denominator)
            else:
                br = 0.0
        
        # Sanity check: limit field values to reasonable ranges
        max_field = 100.0  # Tesla (unrealistically high but prevents numerical overflow)
        
        if np.abs(bz) > max_field:
            bz = np.sign(bz) * max_field
        if np.abs(br) > max_field:
            br = np.sign(br) * max_field
        
        return bz, br
    
    def _add_projectile_field_effects(self, Z, R, Bz, Br, B_magnitude, projectile_position):
        """
        Add magnetic field effects from ferromagnetic projectile.
        
        Args:
            Z, R: Coordinate grids
            Bz, Br: Field components
            B_magnitude: Field magnitude
            projectile_position: Projectile z-position
            
        Returns:
            Modified field arrays with projectile effects
        """
        if not self.physics:
            return Bz, Br, B_magnitude
        
        # Projectile parameters
        projectile_radius = getattr(self.physics, 'projectile_radius', 0.002)
        projectile_permeability = getattr(self.physics, 'projectile_permeability', 1000)
        
        # Calculate enhancement factor near projectile
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                z = Z[i, j]
                r = R[i, j]
                
                # Distance from projectile center
                distance = np.sqrt((z - projectile_position)**2 + r**2)
                
                # Apply magnetic enhancement within projectile vicinity
                if distance < projectile_radius * 2:
                    enhancement = 1 + (projectile_permeability - 1) * \
                                np.exp(-distance / (projectile_radius * 0.5))
                    Bz[i, j] *= enhancement
                    Br[i, j] *= enhancement
        
        # Recalculate magnitude
        B_magnitude = np.sqrt(Bz**2 + Br**2)
        
        return Bz, Br, B_magnitude
    
    def calculate_onaxis_field_profile(self, current_values=None, z_range=None, num_points=200):
        """
        Calculate magnetic field profile along the axis for different currents.
        
        Args:
            current_values: List of current values to plot
            z_range: Z-axis range tuple (m)
            num_points: Number of calculation points
            
        Returns:
            Dictionary with field profile data
        """
        if current_values is None:
            current_values = [50, 100, 150, 200, 250, 300]
        if z_range is None:
            z_range = VISUALIZATION_CONSTANTS['DEFAULT_Z_RANGE']
        
        z_vals = np.linspace(z_range[0], z_range[1], num_points)
        field_profiles = {}
        
        for current in current_values:
            bz_profile = []
            for z in z_vals:
                bz, _ = self._biot_savart_total_field(z, 0, current)  # On-axis (r=0)
                bz_profile.append(bz)
            
            field_profiles[current] = np.array(bz_profile)
        
        return {
            'z_vals': z_vals,
            'field_profiles': field_profiles,
            'current_values': current_values
        }


class FieldLineTracer:
    """Class for tracing magnetic field lines in 3D."""
    
    def __init__(self, field_data_3d):
        """
        Initialize field line tracer.
        
        Args:
            field_data_3d: 3D field data dictionary
        """
        self.field_data = field_data_3d
    
    def trace_field_lines_3d(self, start_points, max_length=0.2, step_size=0.001):
        """
        Trace magnetic field lines in 3D space.
        
        Args:
            start_points: List of starting points [(x, y, z), ...]
            max_length: Maximum line length
            step_size: Integration step size
            
        Returns:
            List of field line coordinates
        """
        field_lines = []
        
        # Extract field data
        X, Y, Z = self.field_data['X'], self.field_data['Y'], self.field_data['Z']
        Bx, By, Bz = self.field_data['Bx'], self.field_data['By'], self.field_data['Bz']
        
        def field_func(pos):
            """Interpolate field at given position."""
            x, y, z = pos
            
            # Simple nearest neighbor interpolation for now
            # In production, use scipy.interpolate.RegularGridInterpolator
            i = np.argmin(np.abs(X[:, 0, 0] - x))
            j = np.argmin(np.abs(Y[0, :, 0] - y))
            k = np.argmin(np.abs(Z[0, 0, :] - z))
            
            # Bounds checking
            if (i >= X.shape[0]-1 or j >= Y.shape[1]-1 or k >= Z.shape[2]-1 or
                i < 0 or j < 0 or k < 0):
                return np.array([0, 0, 0])
            
            return np.array([Bx[i, j, k], By[i, j, k], Bz[i, j, k]])
        
        # Trace each field line
        for start_point in start_points:
            line_points = [start_point]
            current_pos = np.array(start_point)
            total_length = 0
            
            while total_length < max_length:
                # Get field direction at current position
                field_vec = field_func(current_pos)
                field_magnitude = np.linalg.norm(field_vec)
                
                if field_magnitude < 1e-10:
                    break
                
                # Normalize and step
                direction = field_vec / field_magnitude
                next_pos = current_pos + direction * step_size
                
                line_points.append(next_pos.tolist())
                current_pos = next_pos
                total_length += step_size
            
            field_lines.append(np.array(line_points))
        
        return field_lines 