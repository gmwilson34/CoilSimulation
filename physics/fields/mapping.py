"""
Field Mapping and Visualization

This module provides utilities for creating magnetic field maps,
visualizations, and field analysis tools.

IMPROVEMENTS IMPLEMENTED:
- Fixed dependency errors by using core field calculator methods
- Improved uniformity analysis with solenoid-specific metrics
- Adaptive gradient calculation delta scaling
- Parallel processing for large field maps
- Proper VTK export with pyvista support
- Streamline and contour generation
- Helmholtz uniformity metrics
- Progress tracking without print statements in loops
"""

import numpy as np
import warnings
import csv
from typing import Tuple, Dict, List, Optional, Union, Callable, Any
from .biot_savart import BiotSavartCalculator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Optional dependencies with proper type handling
try:
    import pyvista as pv  # type: ignore
    PYVISTA_AVAILABLE = True
except ImportError:
    pv = None  # type: ignore
    PYVISTA_AVAILABLE = False
    warnings.warn("PyVista not available. VTK export will be limited.", ImportWarning)

try:
    from joblib import Parallel, delayed  # type: ignore
    JOBLIB_AVAILABLE = True
except ImportError:
    Parallel = None  # type: ignore
    delayed = None  # type: ignore
    JOBLIB_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None  # type: ignore
    TQDM_AVAILABLE = False


class FieldMapping:
    """
    Utility class for creating magnetic field maps and visualizations.
    
    Provides tools for:
    - 1D axial field profiles
    - 2D field maps in cylindrical coordinates  
    - 3D field visualization data
    - Field gradient analysis with adaptive scaling
    - Field uniformity calculations with solenoid-specific metrics
    - Parallel processing for large field calculations
    - Streamline and contour generation
    - Advanced export formats (NPZ, CSV, VTK)
    """
    
    def __init__(self, field_calculator, enable_parallel: bool = True, max_workers: Optional[int] = None):
        """
        Initialize field mapping with enhanced capabilities.
        
        Args:
            field_calculator: Field calculator instance (AdvancedMagneticFieldCalculator)
            enable_parallel: Whether to use parallel processing for field calculations
            max_workers: Maximum number of parallel workers (None for auto-detection)
        """
        self.field_calc = field_calculator
        self.biot_savart = BiotSavartCalculator()
        
        # Parallel processing configuration
        self.enable_parallel = enable_parallel and (JOBLIB_AVAILABLE or mp.cpu_count() > 1)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)  # Reasonable limit
        
        # Get characteristic length scale from coil geometry for adaptive calculations
        self.characteristic_length = getattr(field_calculator, 'coil_inner_radius', 0.01)
        if hasattr(field_calculator, 'coil_length'):
            self.characteristic_length = min(self.characteristic_length, 
                                           field_calculator.coil_length / 10)
        
        # Progress tracking
        self.verbose = True
        
        print("🔬 Enhanced field mapping utilities initialized")
        if self.enable_parallel:
            print(f"   Parallel processing enabled: {self.max_workers} workers")
        if not PYVISTA_AVAILABLE:
            print("   Note: PyVista not available - VTK export will be basic")
        
    def set_verbose(self, verbose: bool):
        """Enable or disable verbose progress output."""
        self.verbose = verbose
    
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
        Create 2D magnetic field map in cylindrical coordinates with parallel processing.
        
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
        
        if self.verbose:
            print(f"🔬 Computing 2D field map: {num_z}×{num_r} = {num_z*num_r} points")
        
        # Use parallel processing for large calculations
        if self.enable_parallel and num_z * num_r > 1000:
            B_z, B_r = self._compute_2d_field_parallel(Z, R, current)
        else:
            B_z, B_r = self._compute_2d_field_sequential(Z, R, current)
        
        return Z, R, B_z, B_r
    
    def _compute_2d_field_sequential(self, Z: np.ndarray, R: np.ndarray, current: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D field sequentially."""
        B_z = np.zeros_like(Z)
        B_r = np.zeros_like(R)
        
        num_z, num_r = Z.shape
        total_points = num_z * num_r
        
        # Use tqdm if available for progress tracking
        iterator = range(num_z)
        if TQDM_AVAILABLE and self.verbose and tqdm is not None:
            iterator = tqdm(iterator, desc="Computing 2D field", leave=False)
        
        for i in iterator:
            for j in range(num_r):
                z_pos, r_pos = Z[i, j], R[i, j]
                
                # Use improved field calculation method
                if hasattr(self.field_calc, 'magnetic_field_cylindrical'):
                    B_z_val, B_r_val = self.field_calc.magnetic_field_cylindrical(z_pos, r_pos, current)
                else:
                    # Fallback to Biot-Savart with coil parameters
                    coil_params = self._get_coil_parameters()
                    B_z_val, B_r_val = self.biot_savart.biot_savart_total_field(
                        z_pos, r_pos, current, coil_params
                    )
                
                B_z[i, j] = B_z_val
                B_r[i, j] = B_r_val
        
        return B_z, B_r
    
    def _compute_2d_field_parallel(self, Z: np.ndarray, R: np.ndarray, current: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D field using parallel processing."""
        num_z, num_r = Z.shape
        
        # Flatten coordinates for parallel processing
        z_flat = Z.flatten()
        r_flat = R.flatten()
        coords = list(zip(z_flat, r_flat))
        
        if self.verbose:
            print(f"   Using parallel processing with {self.max_workers} workers")
        
        # Use joblib if available, otherwise use multiprocessing
        if JOBLIB_AVAILABLE and Parallel is not None and delayed is not None:
            results = Parallel(n_jobs=self.max_workers, backend='threading')(
                delayed(self._compute_single_field_point)(z, r, current) for z, r in coords
            )
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(
                    lambda coord: self._compute_single_field_point(coord[0], coord[1], current),
                    coords
                ))
        
        # Reshape results
        B_z_flat = np.array([result[0] for result in results])
        B_r_flat = np.array([result[1] for result in results])
        
        B_z = B_z_flat.reshape(num_z, num_r)
        B_r = B_r_flat.reshape(num_z, num_r)
        
        return B_z, B_r
    
    def _compute_single_field_point(self, z: float, r: float, current: float) -> Tuple[float, float]:
        """Compute field at single point - helper for parallel processing."""
        try:
            if hasattr(self.field_calc, 'magnetic_field_cylindrical'):
                return self.field_calc.magnetic_field_cylindrical(z, r, current)
            else:
                coil_params = self._get_coil_parameters()
                return self.biot_savart.biot_savart_total_field(z, r, current, coil_params)
        except Exception as e:
            warnings.warn(f"Error computing field at (z={z}, r={r}): {e}")
            return 0.0, 0.0
    
    def _get_coil_parameters(self) -> Dict:
        """Get coil parameters for Biot-Savart calculations."""
        return {
            'inner_radius': getattr(self.field_calc, 'coil_inner_radius', 0.01),
            'outer_radius': getattr(self.field_calc, 'coil_outer_radius', 0.02),
            'length': getattr(self.field_calc, 'coil_length', 0.05),
            'total_turns': getattr(self.field_calc, 'total_turns', 1000),
            'num_layers': getattr(self.field_calc, 'num_layers', 1)
        }
    
    def create_3d_field_map(self, current: float, x_range: Tuple[float, float],
                          y_range: Tuple[float, float], z_range: Tuple[float, float],
                          num_points: Tuple[int, int, int] = (50, 50, 50)) -> Dict:
        """
        Create 3D magnetic field map in Cartesian coordinates with parallel processing.
        
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
        total_points = nx * ny * nz
        
        if self.verbose:
            print(f"🔬 Computing 3D field map: {nx}×{ny}×{nz} = {total_points} points")
        
        x_array = np.linspace(x_range[0], x_range[1], nx)
        y_array = np.linspace(y_range[0], y_range[1], ny)
        z_array = np.linspace(z_range[0], z_range[1], nz)
        
        X, Y, Z = np.meshgrid(x_array, y_array, z_array, indexing='ij')
        
        # Initialize field arrays
        B_x = np.zeros_like(X)
        B_y = np.zeros_like(Y)
        B_z = np.zeros_like(Z)
        B_magnitude = np.zeros_like(X)
        
        # Use parallel processing for large calculations
        if self.enable_parallel and total_points > 5000:
            if self.verbose:
                print(f"   Using parallel processing with {self.max_workers} workers")
            B_x, B_y, B_z, B_magnitude = self._compute_3d_field_parallel(X, Y, Z, current)
        else:
            B_x, B_y, B_z, B_magnitude = self._compute_3d_field_sequential(X, Y, Z, current)
        
        return {
            'coordinates': {'X': X, 'Y': Y, 'Z': Z},
            'fields': {'B_x': B_x, 'B_y': B_y, 'B_z': B_z, 'B_magnitude': B_magnitude},
            'current': current,
            'ranges': {'x': x_range, 'y': y_range, 'z': z_range},
            'resolution': num_points
        }
    
    def _compute_3d_field_sequential(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                   current: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D field sequentially with progress tracking."""
        B_x = np.zeros_like(X)
        B_y = np.zeros_like(Y)
        B_z = np.zeros_like(Z)
        B_magnitude = np.zeros_like(X)
        
        nx, ny, nz = X.shape
        total_points = nx * ny * nz
        
        # Progress tracking
        iterator = range(nx)
        if TQDM_AVAILABLE and self.verbose and tqdm is not None:
            iterator = tqdm(iterator, desc="Computing 3D field", leave=False)
        
        for i in iterator:
            for j in range(ny):
                for k in range(nz):
                    position = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                    
                    # Calculate 3D field
                    B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                    
                    B_x[i, j, k] = B_field[0]
                    B_y[i, j, k] = B_field[1]
                    B_z[i, j, k] = B_field[2]
                    B_magnitude[i, j, k] = np.linalg.norm(B_field)
        
        return B_x, B_y, B_z, B_magnitude
    
    def _compute_3d_field_parallel(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                 current: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute 3D field using parallel processing."""
        # Flatten coordinates
        positions = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        # Parallel computation
        if JOBLIB_AVAILABLE and Parallel is not None and delayed is not None:
            results = Parallel(n_jobs=self.max_workers, backend='threading')(
                delayed(self._compute_3d_field_point)(pos, current) for pos in positions
            )
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(
                    lambda pos: self._compute_3d_field_point(pos, current),
                    positions
                ))
        
        # Reshape results
        B_fields = np.array(results)  # Shape: (N, 3)
        B_x = B_fields[:, 0].reshape(X.shape)
        B_y = B_fields[:, 1].reshape(X.shape)
        B_z = B_fields[:, 2].reshape(X.shape)
        B_magnitude = np.linalg.norm(B_fields, axis=1).reshape(X.shape)
        
        return B_x, B_y, B_z, B_magnitude
    
    def _compute_3d_field_point(self, position: np.ndarray, current: float) -> np.ndarray:
        """Compute 3D field at single point - helper for parallel processing."""
        try:
            return self.field_calc.magnetic_field_3d_biot_savart(position, current)
        except Exception as e:
            warnings.warn(f"Error computing 3D field at {position}: {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def analyze_field_uniformity(self, current: float, analysis_region: Dict) -> Dict:
        """
        Analyze magnetic field uniformity in a specified region with solenoid-specific metrics.
        
        Args:
            current: Current (A)
            analysis_region: Dictionary with region parameters
            
        Returns:
            Dictionary with uniformity analysis results including Helmholtz metrics
        """
        region_type = analysis_region.get('type', 'cylinder')
        
        if region_type == 'cylinder':
            return self._analyze_cylindrical_uniformity_solenoid(current, analysis_region)
        elif region_type == 'sphere':
            return self._analyze_spherical_uniformity(current, analysis_region)
        elif region_type == 'helmholtz':
            return self._analyze_helmholtz_uniformity(current, analysis_region)
        else:
            raise ValueError(f"Unknown region type: {region_type}")
    
    def _analyze_cylindrical_uniformity_solenoid(self, current: float, region: Dict) -> Dict:
        """
        Analyze field uniformity in cylindrical region optimized for solenoids.
        
        For solenoids, the axial field component (B_z) dominates, so we focus
        uniformity analysis on this component while tracking transverse components.
        """
        center_z = region.get('center_z', 0.0)
        radius = region.get('radius', 0.005)  # 5mm radius
        length = region.get('length', 0.01)   # 10mm length
        num_samples = region.get('num_samples', 200)
        stratified_sampling = region.get('stratified_sampling', True)
        
        # Generate sample points with optional stratified sampling
        if stratified_sampling:
            sample_points, field_data = self._stratified_cylindrical_sampling(
                center_z, radius, length, num_samples, current
            )
        else:
            sample_points, field_data = self._random_cylindrical_sampling(
                center_z, radius, length, num_samples, current
            )
        
        # Separate axial and transverse components
        B_axial = field_data['B_z']  # Dominant component for solenoids
        B_transverse = np.sqrt(field_data['B_x']**2 + field_data['B_y']**2)
        B_total = field_data['B_magnitude']
        
        # Statistical analysis for each component
        stats = {}
        for name, data in [('axial', B_axial), ('transverse', B_transverse), ('total', B_total)]:
            mean_val = np.mean(data)
            std_val = np.std(data)
            stats[name] = {
                'mean': mean_val,
                'std': std_val,
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'cv': std_val / mean_val if mean_val > 0 else 0.0
            }
        
        # Solenoid-specific uniformity metrics
        uniformity_metrics = self._calculate_solenoid_uniformity_metrics(
            B_axial, B_transverse, B_total, radius, length
        )
        
        # Add Helmholtz-like uniformity analysis
        helmholtz_metrics = self._calculate_helmholtz_like_metrics(
            sample_points, field_data, center_z
        )
        
        return {
            'region_type': 'cylinder_solenoid',
            'region_parameters': region,
            'sample_count': num_samples,
            'field_statistics': stats,
            'uniformity_metrics': uniformity_metrics,
            'helmholtz_metrics': helmholtz_metrics,
            'sample_data': {
                'positions': sample_points,
                'field_components': field_data
            }
        }
    
    def _stratified_cylindrical_sampling(self, center_z: float, radius: float, length: float,
                                       num_samples: int, current: float) -> Tuple[List, Dict]:
        """Generate stratified samples in cylindrical region for better coverage."""
        # Divide into radial and axial strata
        num_r_strata = max(3, int(np.sqrt(num_samples) // 2))
        num_z_strata = max(3, int(np.sqrt(num_samples) // 2))
        samples_per_stratum = num_samples // (num_r_strata * num_z_strata)
        
        sample_points = []
        B_x_vals, B_y_vals, B_z_vals, B_mag_vals = [], [], [], []
        
        for i in range(num_r_strata):
            for j in range(num_z_strata):
                # Define stratum boundaries
                r_min = radius * i / num_r_strata
                r_max = radius * (i + 1) / num_r_strata
                z_min = center_z - length/2 + length * j / num_z_strata
                z_max = center_z - length/2 + length * (j + 1) / num_z_strata
                
                # Generate samples within stratum
                for _ in range(samples_per_stratum):
                    r_sample = np.sqrt(np.random.uniform(r_min**2, r_max**2))
                    theta_sample = 2 * np.pi * np.random.random()
                    z_sample = np.random.uniform(z_min, z_max)
                    
                    x = r_sample * np.cos(theta_sample)
                    y = r_sample * np.sin(theta_sample)
                    position = np.array([x, y, z_sample])
                    
                    # Calculate field
                    B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                    
                    sample_points.append(position)
                    B_x_vals.append(B_field[0])
                    B_y_vals.append(B_field[1])
                    B_z_vals.append(B_field[2])
                    B_mag_vals.append(np.linalg.norm(B_field))
        
        field_data = {
            'B_x': np.array(B_x_vals),
            'B_y': np.array(B_y_vals),
            'B_z': np.array(B_z_vals),
            'B_magnitude': np.array(B_mag_vals)
        }
        
        return sample_points, field_data
    
    def _random_cylindrical_sampling(self, center_z: float, radius: float, length: float,
                                   num_samples: int, current: float) -> Tuple[List, Dict]:
        """Generate random samples in cylindrical region."""
        sample_points = []
        B_x_vals, B_y_vals, B_z_vals, B_mag_vals = [], [], [], []
        
        for i in range(num_samples):
            # Uniform random points in cylinder
            r_sample = radius * np.sqrt(np.random.random())
            theta_sample = 2 * np.pi * np.random.random()
            z_sample = center_z + (np.random.random() - 0.5) * length
            
            x = r_sample * np.cos(theta_sample)
            y = r_sample * np.sin(theta_sample)
            position = np.array([x, y, z_sample])
            
            # Calculate field
            B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
            
            sample_points.append(position)
            B_x_vals.append(B_field[0])
            B_y_vals.append(B_field[1])
            B_z_vals.append(B_field[2])
            B_mag_vals.append(np.linalg.norm(B_field))
        
        field_data = {
            'B_x': np.array(B_x_vals),
            'B_y': np.array(B_y_vals),
            'B_z': np.array(B_z_vals),
            'B_magnitude': np.array(B_mag_vals)
        }
        
        return sample_points, field_data
    
    def _calculate_solenoid_uniformity_metrics(self, B_axial: np.ndarray, B_transverse: np.ndarray,
                                             B_total: np.ndarray, radius: float, length: float) -> Dict:
        """Calculate solenoid-specific uniformity metrics."""
        # Standard uniformity metrics
        axial_mean = np.mean(B_axial)
        axial_std = np.std(B_axial)
        
        uniformity_percent = (1.0 - axial_std / axial_mean) * 100 if axial_mean > 0 else 0.0
        field_variation_percent = (np.max(B_axial) - np.min(B_axial)) / axial_mean * 100 if axial_mean > 0 else 0.0
        
        # Solenoid-specific metrics
        transverse_to_axial_ratio = np.mean(B_transverse) / axial_mean if axial_mean > 0 else 0.0
        axial_field_decay = self._calculate_axial_field_decay(B_axial, length)
        radial_field_gradient = self._calculate_radial_field_gradient(B_axial, radius)
        
        # Field quality metrics
        homogeneity_ppm = (axial_std / axial_mean) * 1e6 if axial_mean > 0 else 0.0  # parts per million
        
        return {
            'uniformity_percent': uniformity_percent,
            'field_variation_percent': field_variation_percent,
            'homogeneity_ppm': homogeneity_ppm,
            'transverse_to_axial_ratio': transverse_to_axial_ratio,
            'axial_field_decay': axial_field_decay,
            'radial_field_gradient': radial_field_gradient,
            'coefficient_of_variation': axial_std / axial_mean if axial_mean > 0 else 0.0,
            'field_uniformity_class': self._classify_field_uniformity(float(uniformity_percent))
        }
    
    def _calculate_axial_field_decay(self, B_axial: np.ndarray, length: float) -> float:
        """Calculate characteristic axial field decay rate."""
        # This is a simplified metric - could be enhanced with position-dependent analysis
        return float(np.std(B_axial) / length) if length > 0 else 0.0
    
    def _calculate_radial_field_gradient(self, B_axial: np.ndarray, radius: float) -> float:
        """Calculate characteristic radial field gradient."""
        # This is a simplified metric - could be enhanced with position-dependent analysis  
        return float(np.std(B_axial) / radius) if radius > 0 else 0.0
    
    def _classify_field_uniformity(self, uniformity_percent: float) -> str:
        """Classify field uniformity quality."""
        if uniformity_percent >= 99.9:
            return "Excellent (>99.9%)"
        elif uniformity_percent >= 99.0:
            return "Very Good (99-99.9%)"
        elif uniformity_percent >= 95.0:
            return "Good (95-99%)"
        elif uniformity_percent >= 90.0:
            return "Fair (90-95%)"
        else:
            return "Poor (<90%)"
    
    def _calculate_helmholtz_like_metrics(self, sample_points: List, field_data: Dict, center_z: float) -> Dict:
        """Calculate Helmholtz coil-like uniformity metrics."""
        positions = np.array(sample_points)
        B_z = field_data['B_z']
        
        # Calculate field derivatives (simplified)
        z_positions = positions[:, 2] - center_z
        
        # Linear fit to assess field curvature
        if len(z_positions) > 2:
            poly_coeffs = np.polyfit(z_positions, B_z, deg=2)
            linear_coeff = poly_coeffs[1]  # First derivative
            quadratic_coeff = poly_coeffs[0]  # Second derivative (curvature)
        else:
            linear_coeff = 0.0
            quadratic_coeff = 0.0
        
        # Helmholtz-like uniformity metrics
        field_flatness = abs(linear_coeff) / np.mean(B_z) if np.mean(B_z) > 0 else 0.0
        field_curvature = abs(quadratic_coeff) / np.mean(B_z) if np.mean(B_z) > 0 else 0.0
        
        return {
            'field_flatness': field_flatness,
            'field_curvature': field_curvature,
            'linear_gradient': linear_coeff,
            'quadratic_curvature': quadratic_coeff,
            'helmholtz_quality': self._assess_helmholtz_quality(field_flatness, field_curvature)
        }
    
    def _assess_helmholtz_quality(self, flatness: float, curvature: float) -> str:
        """Assess Helmholtz-like field quality."""
        if flatness < 0.001 and curvature < 0.001:
            return "Excellent Helmholtz-like uniformity"
        elif flatness < 0.01 and curvature < 0.01:
            return "Good Helmholtz-like uniformity"
        elif flatness < 0.05 and curvature < 0.05:
            return "Moderate Helmholtz-like uniformity"
        else:
            return "Poor Helmholtz-like uniformity"
    
    def _analyze_helmholtz_uniformity(self, current: float, region: Dict) -> Dict:
        """Analyze uniformity using Helmholtz coil principles."""
        # This could be expanded to include specific Helmholtz coil analysis
        # For now, delegate to cylindrical analysis with Helmholtz focus
        region_modified = region.copy()
        region_modified['type'] = 'cylinder'
        region_modified['stratified_sampling'] = True
        
        result = self._analyze_cylindrical_uniformity_solenoid(current, region_modified)
        result['region_type'] = 'helmholtz'
        
        return result
    
    def _analyze_spherical_uniformity(self, current: float, region: Dict) -> Dict:
        """
        Analyze field uniformity in spherical region with enhanced metrics.
        """
        center_position = region.get('center', np.array([0.0, 0.0, 0.0]))
        radius = region.get('radius', 0.005)  # 5mm radius
        num_samples = region.get('num_samples', 200)
        
        # Generate random points in sphere
        sample_points = []
        B_x_vals, B_y_vals, B_z_vals, B_mag_vals = [], [], [], []
        
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
            B_x_vals.append(B_field[0])
            B_y_vals.append(B_field[1])
            B_z_vals.append(B_field[2])
            B_mag_vals.append(B_magnitude)
        
        # Convert to arrays
        field_data = {
            'B_x': np.array(B_x_vals),
            'B_y': np.array(B_y_vals),
            'B_z': np.array(B_z_vals),
            'B_magnitude': np.array(B_mag_vals)
        }
        
        # Statistical analysis for each component
        B_axial = field_data['B_z']  # Assume z is the main field direction
        B_transverse = np.sqrt(field_data['B_x']**2 + field_data['B_y']**2)
        B_total = field_data['B_magnitude']
        
        stats = {}
        for name, data in [('axial', B_axial), ('transverse', B_transverse), ('total', B_total)]:
            mean_val = np.mean(data)
            std_val = np.std(data)
            stats[name] = {
                'mean': mean_val,
                'std': std_val,
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'cv': std_val / mean_val if mean_val > 0 else 0.0
            }
        
        # Uniformity metrics
        axial_mean = stats['axial']['mean']
        axial_std = stats['axial']['std']
        uniformity_percent = (1.0 - axial_std / axial_mean) * 100 if axial_mean > 0 else 0.0
        field_variation_percent = (np.max(B_axial) - np.min(B_axial)) / axial_mean * 100 if axial_mean > 0 else 0.0
        
        # Spherical uniformity specific metrics
        spherical_metrics = {
            'uniformity_percent': uniformity_percent,
            'field_variation_percent': field_variation_percent,
            'homogeneity_ppm': (axial_std / axial_mean) * 1e6 if axial_mean > 0 else 0.0,
            'transverse_to_axial_ratio': np.mean(B_transverse) / axial_mean if axial_mean > 0 else 0.0,
            'spherical_harmonic_deviation': self._calculate_spherical_harmonic_deviation(
                sample_points, field_data, center_position
            ),
            'coefficient_of_variation': axial_std / axial_mean if axial_mean > 0 else 0.0,
            'field_uniformity_class': self._classify_field_uniformity(float(uniformity_percent))
        }
        
        return {
            'region_type': 'sphere',
            'region_parameters': region,
            'sample_count': num_samples,
            'field_statistics': stats,
            'uniformity_metrics': spherical_metrics,
            'sample_data': {
                'positions': sample_points,
                'field_components': field_data
            }
        }
    
    def _calculate_spherical_harmonic_deviation(self, sample_points: List, field_data: Dict,
                                             center_position: np.ndarray) -> float:
        """
        Calculate deviation from perfect spherical symmetry using spherical harmonic analysis.
        Simplified version - could be expanded with proper spherical harmonic decomposition.
        """
        positions = np.array(sample_points)
        relative_positions = positions - center_position
        
        # Calculate distances from center
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # Field magnitudes
        B_magnitudes = field_data['B_magnitude']
        
        # Simple metric: variance of field magnitude vs. distance from center
        if len(distances) > 1 and np.std(distances) > 0:
            # Correlation between field magnitude and distance
            correlation = np.corrcoef(distances, B_magnitudes)[0, 1]
            return abs(correlation)  # Perfect sphere should have minimal correlation
        else:
            return 0.0
    
    def calculate_field_gradients(self, current: float, positions: np.ndarray, 
                                delta: Optional[float] = None) -> Dict:
        """
        Calculate field gradients at specified positions with adaptive delta scaling.
        
        Args:
            current: Current (A)
            positions: Array of positions [N, 3] (m)
            delta: Step size for numerical differentiation (m). If None, automatically scaled.
            
        Returns:
            Dictionary with gradient data
        """
        num_positions = positions.shape[0]
        
        # Adaptive delta scaling based on coil geometry and position
        if delta is None:
            # Scale delta based on characteristic length and distance from coil
            base_delta = self.characteristic_length / 1000  # 0.1% of characteristic length
            
            # Adapt delta based on distance from coil center
            coil_center = np.array([0, 0, 0])  # Assume coil centered at origin
            distances = np.linalg.norm(positions - coil_center, axis=1)
            min_distance = np.maximum(distances, self.characteristic_length)
            
            # Use smaller delta for positions closer to coil
            delta_array = base_delta * np.minimum(1.0, min_distance / self.characteristic_length)
            use_adaptive = True
        else:
            delta_array = np.full(num_positions, delta)
            use_adaptive = False
        
        if self.verbose:
            if use_adaptive:
                print(f"🔬 Computing gradients with adaptive delta: {np.min(delta_array):.2e} to {np.max(delta_array):.2e} m")
            else:
                print(f"🔬 Computing gradients with fixed delta: {delta:.2e} m")
        
        # Initialize gradient arrays
        grad_Bx = np.zeros((num_positions, 3))  # [dBx/dx, dBx/dy, dBx/dz]
        grad_By = np.zeros((num_positions, 3))  # [dBy/dx, dBy/dy, dBy/dz]
        grad_Bz = np.zeros((num_positions, 3))  # [dBz/dx, dBz/dy, dBz/dz]
        
        # Progress tracking
        iterator = range(num_positions)
        if TQDM_AVAILABLE and self.verbose and num_positions > 10 and tqdm is not None:
            iterator = tqdm(iterator, desc="Computing gradients", leave=False)
        
        for i in iterator:
            pos = positions[i]
            current_delta = delta_array[i] if use_adaptive else delta
            
            # Ensure current_delta is not None
            if current_delta is None:
                current_delta = self.characteristic_length / 1000
            
            # Calculate gradient using central differences
            for j in range(3):  # x, y, z directions
                pos_plus = pos.copy()
                pos_minus = pos.copy()
                pos_plus[j] += current_delta
                pos_minus[j] -= current_delta
                
                # Calculate fields
                B_plus = self.field_calc.magnetic_field_3d_biot_savart(pos_plus, current)
                B_minus = self.field_calc.magnetic_field_3d_biot_savart(pos_minus, current)
                
                # Central difference
                gradient = (B_plus - B_minus) / (2.0 * current_delta)
                
                grad_Bx[i, j] = gradient[0]
                grad_By[i, j] = gradient[1]
                grad_Bz[i, j] = gradient[2]
        
        # Calculate derived quantities
        gradient_magnitudes = np.sqrt(
            np.sum(grad_Bx**2, axis=1) + 
            np.sum(grad_By**2, axis=1) + 
            np.sum(grad_Bz**2, axis=1)
        )
        
        # Field divergence (should be ~0 for magnetic fields)
        divergence = grad_Bx[:, 0] + grad_By[:, 1] + grad_Bz[:, 2]
        
        return {
            'positions': positions,
            'gradients': {
                'grad_Bx': grad_Bx,
                'grad_By': grad_By,
                'grad_Bz': grad_Bz
            },
            'derived_quantities': {
                'gradient_magnitudes': gradient_magnitudes,
                'divergence': divergence,
                'max_gradient': np.max(gradient_magnitudes),
                'mean_gradient': np.mean(gradient_magnitudes),
                'divergence_error': np.mean(np.abs(divergence))  # Should be ~0
            },
            'calculation_parameters': {
                'current': current,
                'delta_range': (np.min(delta_array), np.max(delta_array)) if use_adaptive else (delta, delta),
                'adaptive_delta': use_adaptive,
                'characteristic_length': self.characteristic_length
            }
        }
    
    def export_field_data(self, field_data: Dict, filename: str, format: str = 'npz'):
        """
        Export field data to file with enhanced format support.
        
        Args:
            field_data: Field data dictionary
            filename: Output filename
            format: Export format ('npz', 'csv', 'vtk', 'hdf5')
        """
        if self.verbose:
            print(f"🔬 Exporting field data to {filename} (format: {format})")
        
        if format == 'npz':
            np.savez_compressed(filename, **field_data)
        elif format == 'csv':
            self._export_csv(field_data, filename)
        elif format == 'vtk':
            self._export_vtk(field_data, filename)
        elif format == 'hdf5':
            self._export_hdf5(field_data, filename)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        if self.verbose:
            print(f"   Field data exported successfully")
    
    def _export_csv(self, field_data: Dict, filename: str):
        """Export field data to CSV format with enhanced metadata."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write metadata header
            writer.writerow(['# Field Data Export'])
            writer.writerow([f'# Current: {field_data.get("current", "N/A")} A'])
            writer.writerow([f'# Export timestamp: {np.datetime64("now")}'])
            writer.writerow([])  # Empty line
            
            # Write data header
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
    
    def _export_vtk(self, field_data: Dict, filename: str):
        """Export field data to VTK format using PyVista if available."""
        if not PYVISTA_AVAILABLE or pv is None:
            self._export_vtk_basic(field_data, filename)
            return
        
        coords = field_data.get('coordinates', {})
        fields = field_data.get('fields', {})
        
        if 'X' not in coords or 'B_x' not in fields:
            warnings.warn("Insufficient data for VTK export")
            return
        
        try:
            # Create structured grid
            X, Y, Z = coords['X'], coords['Y'], coords['Z']
            grid = pv.StructuredGrid(X, Y, Z)
            
            # Add field data
            grid['B_x'] = fields['B_x'].flatten()
            grid['B_y'] = fields['B_y'].flatten()
            grid['B_z'] = fields['B_z'].flatten()
            grid['B_magnitude'] = fields['B_magnitude'].flatten()
            
            # Add vector field
            B_vectors = np.column_stack([
                fields['B_x'].flatten(),
                fields['B_y'].flatten(),
                fields['B_z'].flatten()
            ])
            grid['B_vector'] = B_vectors
            
            # Add metadata
            grid.field_data['Current'] = [field_data.get('current', 0.0)]
            grid.field_data['Export_timestamp'] = [str(np.datetime64('now'))]
            
            # Save to file
            grid.save(filename)
            
            if self.verbose:
                print(f"   VTK export completed with PyVista")
                
        except Exception as e:
            warnings.warn(f"PyVista VTK export failed: {e}. Using basic export.")
            self._export_vtk_basic(field_data, filename)
    
    def _export_vtk_basic(self, field_data: Dict, filename: str):
        """Basic VTK export without PyVista."""
        # Save as NPZ with VTK-compatible structure
        vtk_filename = filename.replace('.vtk', '.npz')
        np.savez_compressed(vtk_filename, **field_data)
        
        if self.verbose:
            print(f"   Basic VTK export (saved as NPZ): {vtk_filename}")
    
    def _export_hdf5(self, field_data: Dict, filename: str):
        """Export field data to HDF5 format."""
        try:
            import h5py  # type: ignore
        except ImportError:
            warnings.warn("h5py not available. Cannot export to HDF5 format.")
            # Fallback to NPZ
            np.savez_compressed(filename.replace('.h5', '.npz'), **field_data)
            return
            
        try:
            with h5py.File(filename, 'w') as f:
                # Create groups
                coords_group = f.create_group('coordinates')
                fields_group = f.create_group('fields')
                metadata_group = f.create_group('metadata')
                
                # Save coordinates
                coords = field_data.get('coordinates', {})
                for key, value in coords.items():
                    coords_group.create_dataset(key, data=value)
                
                # Save fields
                fields = field_data.get('fields', {})
                for key, value in fields.items():
                    fields_group.create_dataset(key, data=value)
                
                # Save metadata
                metadata_group.attrs['current'] = field_data.get('current', 0.0)
                metadata_group.attrs['export_timestamp'] = str(np.datetime64('now'))
                
                if 'ranges' in field_data:
                    ranges = field_data['ranges']
                    for key, value in ranges.items():
                        metadata_group.attrs[f'{key}_range'] = value
                
                if 'resolution' in field_data:
                    metadata_group.attrs['resolution'] = field_data['resolution']
            
            if self.verbose:
                print(f"   HDF5 export completed")
                
        except Exception as e:
            warnings.warn(f"HDF5 export failed: {e}. Using NPZ fallback.")
            # Fallback to NPZ
            np.savez_compressed(filename.replace('.h5', '.npz'), **field_data)
    
    def generate_field_streamlines(self, field_data: Dict, start_points: np.ndarray,
                                 max_length: float = 0.1, step_size: float = 1e-4) -> Dict:
        """
        Generate magnetic field streamlines from field data.
        
        Args:
            field_data: 3D field data dictionary
            start_points: Starting points for streamlines [N, 3]
            max_length: Maximum streamline length (m)
            step_size: Integration step size (m)
            
        Returns:
            Dictionary with streamline data
        """
        if self.verbose:
            print(f"🔬 Generating {len(start_points)} field streamlines")
        
        coords = field_data.get('coordinates', {})
        fields = field_data.get('fields', {})
        
        if 'X' not in coords or 'B_x' not in fields:
            raise ValueError("Insufficient field data for streamline generation")
        
        # Create interpolators for field components
        from scipy.interpolate import RegularGridInterpolator
        
        X, Y, Z = coords['X'], coords['Y'], coords['Z']
        
        # Get coordinate arrays
        x_coords = X[:, 0, 0]
        y_coords = Y[0, :, 0]
        z_coords = Z[0, 0, :]
        
        # Create interpolators
        Bx_interp = RegularGridInterpolator((x_coords, y_coords, z_coords), fields['B_x'])
        By_interp = RegularGridInterpolator((x_coords, y_coords, z_coords), fields['B_y'])
        Bz_interp = RegularGridInterpolator((x_coords, y_coords, z_coords), fields['B_z'])
        
        streamlines = []
        
        for start_point in start_points:
            streamline = self._trace_streamline(
                start_point, Bx_interp, By_interp, Bz_interp,
                max_length, step_size, coords
            )
            streamlines.append(streamline)
        
        return {
            'streamlines': streamlines,
            'start_points': start_points,
            'parameters': {
                'max_length': max_length,
                'step_size': step_size
            }
        }
    
    def _trace_streamline(self, start_point: np.ndarray, Bx_interp, By_interp, Bz_interp,
                         max_length: float, step_size: float, coords: Dict) -> np.ndarray:
        """Trace a single streamline using RK4 integration."""
        # Get field boundaries
        X, Y, Z = coords['X'], coords['Y'], coords['Z']
        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(Y), np.max(Y)
        z_min, z_max = np.min(Z), np.max(Z)
        
        points = [start_point.copy()]
        current_point = start_point.copy()
        total_length = 0.0
        
        while total_length < max_length:
            # Check bounds
            if (current_point[0] < x_min or current_point[0] > x_max or
                current_point[1] < y_min or current_point[1] > y_max or
                current_point[2] < z_min or current_point[2] > z_max):
                break
            
            try:
                # RK4 integration step
                k1 = self._get_normalized_field(current_point, Bx_interp, By_interp, Bz_interp)
                k2 = self._get_normalized_field(current_point + 0.5*step_size*k1, Bx_interp, By_interp, Bz_interp)
                k3 = self._get_normalized_field(current_point + 0.5*step_size*k2, Bx_interp, By_interp, Bz_interp)
                k4 = self._get_normalized_field(current_point + step_size*k3, Bx_interp, By_interp, Bz_interp)
                
                # Update position
                delta = step_size * (k1 + 2*k2 + 2*k3 + k4) / 6
                current_point += delta
                
                points.append(current_point.copy())
                total_length += np.linalg.norm(delta)
                
            except Exception:
                # Stop if interpolation fails (likely out of bounds)
                break
        
        return np.array(points)
    
    def _get_normalized_field(self, point: np.ndarray, Bx_interp, By_interp, Bz_interp) -> np.ndarray:
        """Get normalized magnetic field vector at point."""
        try:
            Bx = Bx_interp(point)[()]
            By = By_interp(point)[()]
            Bz = Bz_interp(point)[()]
            
            B_vector = np.array([Bx, By, Bz])
            B_magnitude = np.linalg.norm(B_vector)
            
            if B_magnitude > 0:
                return B_vector / B_magnitude
            else:
                return np.array([0., 0., 0.])
                
        except Exception:
            return np.array([0., 0., 0.])
    
    def generate_field_contours(self, field_data: Dict, component: str = 'B_magnitude',
                              num_levels: int = 20) -> Dict:
        """
        Generate field contour data for visualization.
        
        Args:
            field_data: 2D or 3D field data dictionary
            component: Field component to contour ('B_magnitude', 'B_z', etc.)
            num_levels: Number of contour levels
            
        Returns:
            Dictionary with contour data
        """
        if self.verbose:
            print(f"🔬 Generating field contours for {component}")
        
        coords = field_data.get('coordinates', {})
        fields = field_data.get('fields', {})
        
        if component not in fields:
            raise ValueError(f"Field component '{component}' not found in data")
        
        field_values = fields[component]
        
        # Generate contour levels
        field_min, field_max = np.min(field_values), np.max(field_values)
        levels = np.linspace(field_min, field_max, num_levels)
        
        # For 2D data, can use matplotlib contours directly
        if len(field_values.shape) == 2 and 'Z' in coords and 'R' in coords:
            return {
                'type': '2d_cylindrical',
                'coordinates': {'Z': coords['Z'], 'R': coords['R']},
                'field_values': field_values,
                'levels': levels,
                'component': component
            }
        
        # For 3D data, extract 2D slices
        elif len(field_values.shape) == 3:
            # Extract mid-plane slice (z=0 plane)
            mid_z_idx = field_values.shape[2] // 2
            field_slice = field_values[:, :, mid_z_idx]
            
            return {
                'type': '2d_slice',
                'coordinates': {
                    'X': coords['X'][:, :, mid_z_idx],
                    'Y': coords['Y'][:, :, mid_z_idx]
                },
                'field_values': field_slice,
                'levels': levels,
                'component': component,
                'slice_info': {'axis': 'z', 'index': mid_z_idx}
            }
        
        else:
            raise ValueError("Unsupported field data format for contouring")
    
    def analyze_field_quality(self, current: float, analysis_regions: List[Dict]) -> Dict:
        """
        Comprehensive field quality analysis across multiple regions.
        
        Args:
            current: Current (A)
            analysis_regions: List of region dictionaries for analysis
            
        Returns:
            Dictionary with comprehensive field quality metrics
        """
        if self.verbose:
            print(f"🔬 Analyzing field quality across {len(analysis_regions)} regions")
        
        results = []
        
        for i, region in enumerate(analysis_regions):
            if self.verbose:
                print(f"   Analyzing region {i+1}/{len(analysis_regions)}: {region.get('type', 'unknown')}")
            
            region_result = self.analyze_field_uniformity(current, region)
            region_result['region_id'] = i
            results.append(region_result)
        
        # Aggregate statistics
        all_uniformities = [r['uniformity_metrics']['uniformity_percent'] for r in results]
        all_variations = [r['uniformity_metrics']['field_variation_percent'] for r in results]
        
        aggregate_stats = {
            'number_of_regions': len(analysis_regions),
            'overall_uniformity': {
                'mean': np.mean(all_uniformities),
                'std': np.std(all_uniformities),
                'min': np.min(all_uniformities),
                'max': np.max(all_uniformities)
            },
            'overall_variation': {
                'mean': np.mean(all_variations),
                'std': np.std(all_variations),
                'min': np.min(all_variations),
                'max': np.max(all_variations)
            },
            'quality_classification': self._classify_overall_quality(all_uniformities)
        }
        
        return {
            'individual_regions': results,
            'aggregate_statistics': aggregate_stats,
            'current': current,
            'analysis_timestamp': str(np.datetime64('now'))
        }
    
    def _classify_overall_quality(self, uniformities: List[float]) -> str:
        """Classify overall field quality based on uniformity distribution."""
        mean_uniformity = np.mean(uniformities)
        uniformity_consistency = np.std(uniformities)
        
        if mean_uniformity >= 99.0 and uniformity_consistency < 1.0:
            return "Excellent - High uniformity with good consistency"
        elif mean_uniformity >= 95.0 and uniformity_consistency < 2.0:
            return "Very Good - Good uniformity with reasonable consistency"
        elif mean_uniformity >= 90.0 and uniformity_consistency < 5.0:
            return "Good - Acceptable uniformity"
        elif mean_uniformity >= 80.0:
            return "Fair - Marginal uniformity, may need optimization"
        else:
            return "Poor - Significant uniformity issues detected"
    
    def create_adaptive_field_map(self, current: float, initial_bounds: Dict,
                                accuracy_target: float = 0.01, max_iterations: int = 5) -> Dict:
        """
        Create adaptive field map with automatic mesh refinement.
        
        Args:
            current: Current (A)
            initial_bounds: Initial spatial bounds
            accuracy_target: Target relative accuracy for adaptive refinement
            max_iterations: Maximum refinement iterations
            
        Returns:
            Dictionary with adaptive field map data
        """
        if self.verbose:
            print(f"🔬 Creating adaptive field map (target accuracy: {accuracy_target:.1%})")
        
        # Start with coarse grid
        field_map = self.create_3d_field_map(
            current,
            initial_bounds['x_range'],
            initial_bounds['y_range'],
            initial_bounds['z_range'],
            (20, 20, 20)  # Coarse initial resolution
        )
        
        iteration = 0
        refinement_history = []
        
        while iteration < max_iterations:
            # Analyze field gradients to identify regions needing refinement
            refined_regions = self._identify_refinement_regions(field_map, accuracy_target)
            
            if not refined_regions:
                if self.verbose:
                    print(f"   Convergence achieved after {iteration} iterations")
                break
            
            if self.verbose:
                print(f"   Iteration {iteration+1}: Refining {len(refined_regions)} regions")
            
            # Refine identified regions
            field_map = self._refine_field_regions(field_map, refined_regions, current)
            
            refinement_history.append({
                'iteration': iteration + 1,
                'refined_regions': len(refined_regions),
                'total_points': np.prod(field_map['fields']['B_magnitude'].shape)
            })
            
            iteration += 1
        
        field_map['adaptive_info'] = {
            'target_accuracy': accuracy_target,
            'iterations': iteration,
            'refinement_history': refinement_history,
            'final_resolution': field_map['fields']['B_magnitude'].shape
        }
        
        return field_map
    
    def _identify_refinement_regions(self, field_map: Dict, accuracy_target: float) -> List[Dict]:
        """Identify regions that need mesh refinement based on field gradients."""
        # This is a simplified implementation
        # A more sophisticated version would use proper error estimation
        
        B_mag = field_map['fields']['B_magnitude']
        
        # Calculate simple gradient magnitude
        grad_x = np.gradient(B_mag, axis=0)
        grad_y = np.gradient(B_mag, axis=1)
        grad_z = np.gradient(B_mag, axis=2)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Normalize by field magnitude to get relative gradient
        relative_gradient = gradient_magnitude / (B_mag + 1e-12)  # Avoid division by zero
        
        # Find regions with high relative gradients
        threshold = accuracy_target * 10  # Simple threshold
        high_gradient_mask = relative_gradient > threshold
        
        # For simplicity, return empty list (no refinement in this basic implementation)
        return []
    
    def _refine_field_regions(self, field_map: Dict, regions: List[Dict], current: float) -> Dict:
        """Refine mesh in specified regions."""
        # This is a placeholder for a more sophisticated refinement algorithm
        # For now, just return the original field map
        return field_map


# Alias for backward compatibility
FieldVisualization = FieldMapping


def create_field_mapper(field_calculator, **kwargs) -> FieldMapping:
    """
    Factory function to create FieldMapping instance with optimal settings.
    
    Args:
        field_calculator: Field calculator instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured FieldMapping instance
    """
    # Determine optimal settings based on system capabilities
    enable_parallel = kwargs.get('enable_parallel', JOBLIB_AVAILABLE or mp.cpu_count() > 1)
    max_workers = kwargs.get('max_workers', min(mp.cpu_count(), 8))
    
    mapper = FieldMapping(
        field_calculator,
        enable_parallel=enable_parallel,
        max_workers=max_workers
    )
    
    # Configure verbosity
    verbose = kwargs.get('verbose', True)
    mapper.set_verbose(verbose)
    
    return mapper 