"""
Field Mapping and Visualization

This module provides utilities for creating magnetic field maps,
visualizations, and field analysis tools.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .biot_savart import BiotSavartCalculator


class FieldMapping:
    """
    Utility class for creating magnetic field maps and visualizations.
    
    Provides tools for:
    - 1D axial field profiles
    - 2D field maps in cylindrical coordinates
    - 3D field visualization data
    - Field gradient analysis
    - Field uniformity calculations
    """
    
    def __init__(self, field_calculator):
        """
        Initialize field mapping.
        
        Args:
            field_calculator: Field calculator instance (AdvancedMagneticFieldCalculator)
        """
        self.field_calc = field_calculator
        self.biot_savart = BiotSavartCalculator()
        
        print("🔬 Field mapping utilities initialized")
    
    def create_axial_field_profile(self, current: float, z_range: Tuple[float, float], 
                                 num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create magnetic field profile along the axis.
        
        Args:
            current: Current (A)
            z_range: (z_min, z_max) range (m)
            num_points: Number of points
            
        Returns:
            Tuple of (z_positions, B_field) arrays
        """
        z_positions = np.linspace(z_range[0], z_range[1], num_points)
        B_field = np.zeros(num_points)
        
        for i, z in enumerate(z_positions):
            B_field[i] = self.field_calc.magnetic_field_solenoid_on_axis(z, current)
        
        return z_positions, B_field
    
    def create_2d_field_map(self, current: float, z_range: Tuple[float, float], 
                          r_range: Tuple[float, float], num_z: int = 100, 
                          num_r: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create 2D magnetic field map in cylindrical coordinates.
        
        Args:
            current: Current (A)
            z_range: (z_min, z_max) range (m)
            r_range: (r_min, r_max) range (m)
            num_z: Number of axial points
            num_r: Number of radial points
            
        Returns:
            Tuple of (Z, R, B_z, B_r) meshgrids
        """
        z_array = np.linspace(z_range[0], z_range[1], num_z)
        r_array = np.linspace(r_range[0], r_range[1], num_r)
        
        Z, R = np.meshgrid(z_array, r_array, indexing='ij')
        B_z = np.zeros_like(Z)
        B_r = np.zeros_like(R)
        
        # Get coil parameters
        coil_params = {
            'inner_radius': self.field_calc.coil_inner_radius,
            'length': self.field_calc.coil_length,
            'total_turns': self.field_calc.total_turns
        }
        
        # Calculate field at each point
        for i in range(num_z):
            for j in range(num_r):
                B_z_val, B_r_val = self.biot_savart.biot_savart_total_field(
                    Z[i, j], R[i, j], current, coil_params
                )
                B_z[i, j] = B_z_val
                B_r[i, j] = B_r_val
        
        return Z, R, B_z, B_r
    
    def create_3d_field_map(self, current: float, x_range: Tuple[float, float],
                          y_range: Tuple[float, float], z_range: Tuple[float, float],
                          num_points: Tuple[int, int, int] = (50, 50, 50)) -> Dict:
        """
        Create 3D magnetic field map in Cartesian coordinates.
        
        Args:
            current: Current (A)
            x_range: (x_min, x_max) range (m)
            y_range: (y_min, y_max) range (m)
            z_range: (z_min, z_max) range (m)
            num_points: (nx, ny, nz) number of points in each direction
            
        Returns:
            Dictionary with field data and coordinate arrays
        """
        nx, ny, nz = num_points
        
        x_array = np.linspace(x_range[0], x_range[1], nx)
        y_array = np.linspace(y_range[0], y_range[1], ny)
        z_array = np.linspace(z_range[0], z_range[1], nz)
        
        X, Y, Z = np.meshgrid(x_array, y_array, z_array, indexing='ij')
        
        # Initialize field arrays
        B_x = np.zeros_like(X)
        B_y = np.zeros_like(Y)
        B_z = np.zeros_like(Z)
        B_magnitude = np.zeros_like(X)
        
        # Calculate field at each point
        total_points = nx * ny * nz
        point_count = 0
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    position = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    
                    # Calculate 3D field
                    B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                    
                    B_x[i, j, k] = B_field[0]
                    B_y[i, j, k] = B_field[1]
                    B_z[i, j, k] = B_field[2]
                    B_magnitude[i, j, k] = np.linalg.norm(B_field)
                    
                    point_count += 1
                    if point_count % (total_points // 10) == 0:
                        progress = 100 * point_count / total_points
                        print(f"   3D field calculation progress: {progress:.0f}%")
        
        return {
            'coordinates': {'X': X, 'Y': Y, 'Z': Z},
            'fields': {'B_x': B_x, 'B_y': B_y, 'B_z': B_z, 'B_magnitude': B_magnitude},
            'current': current,
            'ranges': {'x': x_range, 'y': y_range, 'z': z_range}
        }
    
    def analyze_field_uniformity(self, current: float, analysis_region: Dict) -> Dict:
        """
        Analyze magnetic field uniformity in a specified region.
        
        Args:
            current: Current (A)
            analysis_region: Dictionary with region parameters
            
        Returns:
            Dictionary with uniformity analysis results
        """
        region_type = analysis_region.get('type', 'cylinder')
        
        if region_type == 'cylinder':
            return self._analyze_cylindrical_uniformity(current, analysis_region)
        elif region_type == 'sphere':
            return self._analyze_spherical_uniformity(current, analysis_region)
        else:
            raise ValueError(f"Unknown region type: {region_type}")
    
    def _analyze_cylindrical_uniformity(self, current: float, region: Dict) -> Dict:
        """Analyze field uniformity in cylindrical region."""
        center_z = region.get('center_z', 0.0)
        radius = region.get('radius', 0.005)  # 5mm radius
        length = region.get('length', 0.01)   # 10mm length
        num_samples = region.get('num_samples', 100)
        
        # Generate sample points in cylindrical region
        sample_points = []
        field_values = []
        
        for i in range(num_samples):
            # Random points in cylinder
            r_sample = radius * np.sqrt(np.random.random())
            theta_sample = 2 * np.pi * np.random.random()
            z_sample = center_z + (np.random.random() - 0.5) * length
            
            x = r_sample * np.cos(theta_sample)
            y = r_sample * np.sin(theta_sample)
            position = np.array([x, y, z_sample])
            
            # Calculate field
            B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
            B_magnitude = np.linalg.norm(B_field)
            
            sample_points.append(position)
            field_values.append(B_magnitude)
        
        # Statistical analysis
        field_array = np.array(field_values)
        mean_field = np.mean(field_array)
        std_field = np.std(field_array)
        min_field = np.min(field_array)
        max_field = np.max(field_array)
        
        # Uniformity metrics
        uniformity_percent = (1.0 - std_field / mean_field) * 100 if mean_field > 0 else 0.0
        field_variation_percent = (max_field - min_field) / mean_field * 100 if mean_field > 0 else 0.0
        
        return {
            'region_type': 'cylinder',
            'region_parameters': region,
            'sample_count': num_samples,
            'field_statistics': {
                'mean': mean_field,
                'std': std_field,
                'min': min_field,
                'max': max_field,
                'range': max_field - min_field
            },
            'uniformity_metrics': {
                'uniformity_percent': uniformity_percent,
                'field_variation_percent': field_variation_percent,
                'coefficient_of_variation': std_field / mean_field if mean_field > 0 else 0.0
            },
            'sample_data': {
                'positions': sample_points,
                'field_values': field_values
            }
        }
    
    def _analyze_spherical_uniformity(self, current: float, region: Dict) -> Dict:
        """Analyze field uniformity in spherical region."""
        center_position = region.get('center', np.array([0.0, 0.0, 0.0]))
        radius = region.get('radius', 0.005)  # 5mm radius
        num_samples = region.get('num_samples', 100)
        
        # Generate random points in sphere
        sample_points = []
        field_values = []
        
        for i in range(num_samples):
            # Uniform random points in sphere
            u = np.random.random()
            v = np.random.random()
            w = np.random.random()
            
            # Convert to spherical coordinates
            r_sample = radius * (u ** (1.0/3.0))  # Uniform distribution in volume
            theta = 2 * np.pi * v
            phi = np.arccos(2 * w - 1)
            
            # Convert to Cartesian
            x = r_sample * np.sin(phi) * np.cos(theta)
            y = r_sample * np.sin(phi) * np.sin(theta)
            z = r_sample * np.cos(phi)
            
            position = center_position + np.array([x, y, z])
            
            # Calculate field
            B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
            B_magnitude = np.linalg.norm(B_field)
            
            sample_points.append(position)
            field_values.append(B_magnitude)
        
        # Statistical analysis (same as cylindrical)
        field_array = np.array(field_values)
        mean_field = np.mean(field_array)
        std_field = np.std(field_array)
        min_field = np.min(field_array)
        max_field = np.max(field_array)
        
        # Uniformity metrics
        uniformity_percent = (1.0 - std_field / mean_field) * 100 if mean_field > 0 else 0.0
        field_variation_percent = (max_field - min_field) / mean_field * 100 if mean_field > 0 else 0.0
        
        return {
            'region_type': 'sphere',
            'region_parameters': region,
            'sample_count': num_samples,
            'field_statistics': {
                'mean': mean_field,
                'std': std_field,
                'min': min_field,
                'max': max_field,
                'range': max_field - min_field
            },
            'uniformity_metrics': {
                'uniformity_percent': uniformity_percent,
                'field_variation_percent': field_variation_percent,
                'coefficient_of_variation': std_field / mean_field if mean_field > 0 else 0.0
            },
            'sample_data': {
                'positions': sample_points,
                'field_values': field_values
            }
        }
    
    def calculate_field_gradients(self, current: float, positions: np.ndarray, 
                                delta: float = 1e-6) -> Dict:
        """
        Calculate field gradients at specified positions.
        
        Args:
            current: Current (A)
            positions: Array of positions [N, 3] (m)
            delta: Step size for numerical differentiation (m)
            
        Returns:
            Dictionary with gradient data
        """
        num_positions = positions.shape[0]
        
        # Initialize gradient arrays
        grad_Bx = np.zeros((num_positions, 3))  # [dBx/dx, dBx/dy, dBx/dz]
        grad_By = np.zeros((num_positions, 3))  # [dBy/dx, dBy/dy, dBy/dz]
        grad_Bz = np.zeros((num_positions, 3))  # [dBz/dx, dBz/dy, dBz/dz]
        
        for i, pos in enumerate(positions):
            # Calculate gradient using central differences
            for j in range(3):  # x, y, z directions
                pos_plus = pos.copy()
                pos_minus = pos.copy()
                pos_plus[j] += delta
                pos_minus[j] -= delta
                
                # Calculate fields
                B_plus = self.field_calc.magnetic_field_3d_biot_savart(pos_plus, current)
                B_minus = self.field_calc.magnetic_field_3d_biot_savart(pos_minus, current)
                
                # Central difference
                gradient = (B_plus - B_minus) / (2 * delta)
                
                grad_Bx[i, j] = gradient[0]
                grad_By[i, j] = gradient[1]
                grad_Bz[i, j] = gradient[2]
        
        return {
            'positions': positions,
            'gradients': {
                'grad_Bx': grad_Bx,
                'grad_By': grad_By,
                'grad_Bz': grad_Bz
            },
            'current': current,
            'delta': delta
        }
    
    def export_field_data(self, field_data: Dict, filename: str, format: str = 'npz'):
        """
        Export field data to file.
        
        Args:
            field_data: Field data dictionary
            filename: Output filename
            format: Export format ('npz', 'csv', 'vtk')
        """
        if format == 'npz':
            np.savez_compressed(filename, **field_data)
            print(f"🔬 Field data exported to {filename}")
        elif format == 'csv':
            # Flatten data for CSV export
            self._export_csv(field_data, filename)
        elif format == 'vtk':
            # Export in VTK format for visualization
            self._export_vtk(field_data, filename)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_csv(self, field_data: Dict, filename: str):
        """Export field data to CSV format."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['x', 'y', 'z', 'Bx', 'By', 'Bz', 'B_magnitude'])
            
            # Extract coordinate and field arrays
            coords = field_data.get('coordinates', {})
            fields = field_data.get('fields', {})
            
            if 'X' in coords and 'B_x' in fields:
                X, Y, Z = coords['X'], coords['Y'], coords['Z']
                Bx, By, Bz = fields['B_x'], fields['B_y'], fields['B_z']
                B_mag = fields['B_magnitude']
                
                # Flatten and write data
                for i in range(X.size):
                    flat_idx = np.unravel_index(i, X.shape)
                    writer.writerow([
                        X[flat_idx], Y[flat_idx], Z[flat_idx],
                        Bx[flat_idx], By[flat_idx], Bz[flat_idx],
                        B_mag[flat_idx]
                    ])
        
        print(f"🔬 Field data exported to CSV: {filename}")
    
    def _export_vtk(self, field_data: Dict, filename: str):
        """Export field data to VTK format."""
        # This would require vtk or pyvista library
        # For now, just save as NPZ with VTK-compatible structure
        np.savez_compressed(filename.replace('.vtk', '.npz'), **field_data)
        print(f"🔬 Field data exported (VTK format not implemented): {filename}")


# Alias for backward compatibility
FieldVisualization = FieldMapping 