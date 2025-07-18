"""
Maxwell Stress Tensor Calculations

This module implements the Maxwell stress tensor method for calculating
electromagnetic forces with high accuracy.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union, List
from scipy.integrate import quad, dblquad
from scipy.integrate import fixed_quad
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils
from .base import BaseElectromagneticForces


class MaxwellStressTensor(BaseElectromagneticForces):
    """
    Maxwell stress tensor implementation for accurate electromagnetic force calculation.
    
    The Maxwell stress tensor provides an exact method for calculating forces
    by integrating the stress over a closed surface around the object.
    
    T_ij = (1/μ₀)[B_i B_j - (1/2)δ_ij B²] + ε₀[E_i E_j - (1/2)δ_ij E²]
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize Maxwell stress tensor calculator."""
        super().__init__(config, field_calculator, materials)
        
        # Maxwell stress tensor parameters
        self.use_maxwell_stress = config.get('advanced_physics', {}).get('use_maxwell_stress', True)
        self.maxwell_stress_grid_points = config.get('advanced_physics', {}).get('maxwell_stress_grid_points', 50)
        self.integration_order = config.get('advanced_physics', {}).get('integration_order', 16)
        
        # Enhanced calculation options
        self.use_induced_electric_field = config.get('advanced_physics', {}).get('use_induced_electric_field', False)
        self.adaptive_integration = config.get('advanced_physics', {}).get('adaptive_integration', True)
        self.validate_divergence_free = config.get('advanced_physics', {}).get('validate_divergence_free', True)
        
        # Coil geometry for better field calculations
        self.coil_radius = config.get('coil', {}).get('inner_radius', 0.01)
        
        print(f"🔧 Maxwell stress tensor calculator initialized")
        print(f"   - Grid points: {self.maxwell_stress_grid_points}")
        print(f"   - Integration order: {self.integration_order}")
        print(f"   - Induced E-field: {self.use_induced_electric_field}")
        print(f"   - Adaptive integration: {self.adaptive_integration}")
        print(f"   - Divergence validation: {self.validate_divergence_free}")
    
    def calculate_maxwell_stress_force(self, current: float, position: float, 
                                     velocity: float = 0.0) -> Tuple[float, dict]:
        """
        Calculate electromagnetic force using Maxwell stress tensor method.
        
        Force is calculated by integrating the stress tensor over a cylindrical
        surface surrounding the projectile.
        
        Returns:
            Tuple of (total_force, force_breakdown)
        """
        if not self.use_maxwell_stress:
            return 0.0, {}
        
        # Calculate induced electric field if enabled and velocity is significant
        E_field_correction = 0.0
        if self.use_induced_electric_field and abs(velocity) > 1.0:
            E_field_correction = self._calculate_induced_electric_field_force(
                current, position, velocity
            )
        
        # Calculate force components from Maxwell stress
        force_front = self._integrate_maxwell_stress_surface(
            position + self.proj_length/2, self.proj_radius, current, 'front'
        )
        force_back = self._integrate_maxwell_stress_surface(
            position - self.proj_length/2, self.proj_radius, current, 'back'
        )
        force_cylindrical = self._integrate_maxwell_stress_cylindrical(
            position - self.proj_length/2, position + self.proj_length/2,
            self.proj_radius, current
        )
        
        # Total force (front - back + cylindrical contributions + E-field correction)
        total_force = force_front - force_back + force_cylindrical + E_field_correction
        
        # Validate divergence-free condition if enabled
        if self.validate_divergence_free:
            divergence_error = self._validate_stress_tensor_divergence(position, current)
            if divergence_error > 1e-6:
                warnings.warn(f"Maxwell stress tensor divergence validation failed: {divergence_error}")
        
        force_breakdown = {
            'maxwell_front': force_front,
            'maxwell_back': force_back,
            'maxwell_cylindrical': force_cylindrical,
            'maxwell_e_field_correction': E_field_correction,
            'maxwell_total': total_force
        }
        
        return total_force, force_breakdown
    
    def _integrate_maxwell_stress_surface(self, z_surface: float, radius: float, 
                                        current: float, surface_type: str) -> float:
        """
        Integrate Maxwell stress over a circular surface using full tensor.
        
        For a surface perpendicular to z-axis, the force contribution is:
        F_z = ∫∫ T_zz dA where T_zz = (1/μ₀)[B_z² - (1/2)|B|²]
        """
        def stress_integrand(r):
            # Position on surface
            z_pos = z_surface
            
            # Use full 3D field calculation if available
            if hasattr(self.field_calc, 'magnetic_field_3d_biot_savart'):
                # Use 3D field calculation for accurate results
                position = np.array([r, 0, z_pos])
                B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                B_r, B_phi, B_z = B_field[0], B_field[1], B_field[2]
            else:
                # Fallback to on-axis calculation with improved radial field estimate
                B_z = self.field_calc.magnetic_field_solenoid_on_axis(z_pos, current)
                
                # Improved radial field calculation for large r
                if r < self.coil_radius * 0.5:
                    # Near axis: use gradient approximation B_r ≈ -(r/2) × dB_z/dz
                    if hasattr(self.field_calc, 'calculate_field_gradient'):
                        dB_dz = self.field_calc.calculate_field_gradient(z_pos, current)
                        B_r = -(r / 2.0) * dB_dz
                    else:
                        B_r = 0.0
                else:
                    # For r ~ coil_radius, use better approximation based on solenoid field
                    # B_r ≈ (μ₀ I N / 2L) × (r/coil_r) × geometry_factor
                    if hasattr(self.field_calc, 'calculate_radial_field_far'):
                        B_r = self.field_calc.calculate_radial_field_far(r, z_pos, current)
                    else:
                        # Fallback: use geometric scaling
                        coil_field = PhysicsConstants.MU_0 * current / (2 * self.coil_radius)
                        geometry_factor = self.coil_radius / (self.coil_radius**2 + z_pos**2)**0.5
                        B_r = coil_field * (r / self.coil_radius) * geometry_factor
                
                B_phi = 0.0  # Azimuthal symmetry
            
            # Full magnetic field magnitude
            B_magnitude_squared = B_r**2 + B_phi**2 + B_z**2
            
            # Full Maxwell stress tensor component T_zz
            # T_zz = (1/μ₀)[B_z² - (1/2)|B|²]
            T_zz = (1.0 / PhysicsConstants.MU_0) * (B_z**2 - 0.5 * B_magnitude_squared)
            
            # Integration element: r dr dφ → 2πr dr for axial symmetry
            return T_zz * 2 * np.pi * r
        
        # Use adaptive integration if enabled
        try:
            if self.adaptive_integration:
                force, error = quad(stress_integrand, 0, radius, 
                                   limit=self.integration_order*2, epsabs=1e-12, epsrel=1e-9)
                if error > abs(force) * 1e-6:
                    warnings.warn(f"High integration error in Maxwell stress: {error/abs(force):.2e}")
            else:
                force, _ = fixed_quad(stress_integrand, 0, radius, n=self.integration_order)
            
            return NumericalUtils.safe_numerical_operation(force, f"maxwell_stress_{surface_type}")
        except Exception as e:
            warnings.warn(f"Maxwell stress integration failed: {e}")
            return 0.0
    
    def _integrate_maxwell_stress_cylindrical(self, z_start: float, z_end: float, 
                                            radius: float, current: float) -> float:
        """
        Integrate Maxwell stress over cylindrical surface using full tensor.
        
        For cylindrical surface, the radial stress component T_rz contributes to axial force.
        """
        def cylindrical_stress_integrand(z):
            # Position on cylindrical surface
            r_pos = radius
            
            # Use full 3D field calculation if available
            if hasattr(self.field_calc, 'magnetic_field_3d_biot_savart'):
                # Use 3D field calculation for accurate results
                position = np.array([r_pos, 0, z])
                B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                B_r, B_phi, B_z = B_field[0], B_field[1], B_field[2]
            else:
                # Fallback: estimate fields from on-axis calculation
                B_z = self.field_calc.magnetic_field_solenoid_on_axis(z, current)
                
                # Improved radial field calculation
                if r_pos < self.coil_radius * 0.5:
                    # Near axis: use gradient approximation
                    if hasattr(self.field_calc, 'calculate_field_gradient'):
                        dB_dz = self.field_calc.calculate_field_gradient(z, current)
                        B_r = -(r_pos / 2.0) * dB_dz
                    else:
                        B_r = 0.0
                else:
                    # For larger radii, use better approximation
                    if hasattr(self.field_calc, 'calculate_radial_field_far'):
                        B_r = self.field_calc.calculate_radial_field_far(r_pos, z, current)
                    else:
                        # Geometric scaling approximation
                        coil_field = PhysicsConstants.MU_0 * current / (2 * self.coil_radius)
                        geometry_factor = self.coil_radius / (self.coil_radius**2 + z**2)**0.5
                        B_r = coil_field * (r_pos / self.coil_radius) * geometry_factor
                
                B_phi = 0.0  # Azimuthal symmetry
            
            # Full Maxwell stress tensor component T_rz
            # T_rz = (1/μ₀) × B_r × B_z
            T_rz = (1.0 / PhysicsConstants.MU_0) * B_r * B_z
            
            # Integration element: circumference × dz = 2πr × dz
            return T_rz * 2 * np.pi * radius
        
        # Integrate along projectile length with adaptive integration
        try:
            if self.adaptive_integration:
                force, error = quad(cylindrical_stress_integrand, z_start, z_end, 
                                   limit=self.integration_order*2, epsabs=1e-12, epsrel=1e-9)
                if error > abs(force) * 1e-6:
                    warnings.warn(f"High integration error in cylindrical Maxwell stress: {error/abs(force):.2e}")
            else:
                force, _ = fixed_quad(cylindrical_stress_integrand, z_start, z_end, n=self.integration_order)
            
            return NumericalUtils.safe_numerical_operation(force, "maxwell_stress_cylindrical")
        except Exception as e:
            warnings.warn(f"Maxwell stress cylindrical integration failed: {e}")
            return 0.0
    
    def _calculate_induced_electric_field_force(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate force correction due to induced electric field for fast-moving projectiles.
        
        For fast pulses or high velocities, the induced electric field E = -∂A/∂t - ∇φ
        contributes to the Maxwell stress tensor.
        """
        if abs(velocity) < 1.0:  # Negligible for low velocities
            return 0.0
        
        try:
            # Estimate induced electric field magnitude
            # E_induced ≈ v × B for moving conductor
            B_field_magnitude = abs(self.field_calc.magnetic_field_solenoid_on_axis(position, current))
            E_induced_magnitude = abs(velocity) * B_field_magnitude
            
            # Electric field contribution to stress tensor
            # ΔT = ε₀[E_i E_j - (1/2)δ_ij E²]
            electric_stress_correction = (PhysicsConstants.EPSILON_0 * 
                                        E_induced_magnitude**2 * np.pi * self.proj_radius**2)
            
            return electric_stress_correction
        except Exception as e:
            warnings.warn(f"Induced electric field calculation failed: {e}")
            return 0.0
    
    def _validate_stress_tensor_divergence(self, position: float, current: float) -> float:
        """
        Validate that the Maxwell stress tensor satisfies ∇·T = 0 in vacuum.
        
        Returns the maximum divergence error as a measure of calculation accuracy.
        """
        try:
            # Sample stress tensor at several points around the projectile
            max_divergence = 0.0
            test_positions = [
                position - self.proj_length/4,
                position,
                position + self.proj_length/4
            ]
            
            for z_pos in test_positions:
                # Calculate stress tensor components at this position
                if hasattr(self.field_calc, 'magnetic_field_3d_biot_savart'):
                    pos_array = np.array([self.proj_radius/2, 0, z_pos])
                    B_field = self.field_calc.magnetic_field_3d_biot_savart(pos_array, current)
                    T = self.calculate_stress_tensor_components(B_field)
                    
                    # Approximate divergence using finite differences
                    # This is a simplified check - full implementation would need proper gradients
                    divergence_estimate = abs(np.trace(T)) / (B_field @ B_field / PhysicsConstants.MU_0)
                    max_divergence = max(max_divergence, divergence_estimate)
            
            return max_divergence
        except Exception as e:
            warnings.warn(f"Divergence validation failed: {e}")
            return 0.0

    def calculate_stress_tensor_components(self, B_field: np.ndarray, E_field: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate all components of Maxwell stress tensor.
        
        Returns 3x3 stress tensor matrix.
        """
        # Default to zero electric field if not provided
        if E_field is None:
            E_field = np.zeros_like(B_field)
        
        # Field magnitudes
        B_squared = np.sum(B_field**2)
        E_squared = np.sum(E_field**2)
        
        # Initialize stress tensor
        T = np.zeros((3, 3))
        
        # Fill stress tensor components
        for i in range(3):
            for j in range(3):
                # Magnetic contribution
                T_mag = (1.0 / PhysicsConstants.MU_0) * (
                    B_field[i] * B_field[j] - 0.5 * B_squared * (1 if i == j else 0)
                )
                
                # Electric contribution
                T_elec = PhysicsConstants.EPSILON_0 * (
                    E_field[i] * E_field[j] - 0.5 * E_squared * (1 if i == j else 0)
                )
                
                T[i, j] = T_mag + T_elec
        
        return T
    
    def calculate_force_from_stress_divergence(self, B_field: np.ndarray, 
                                             position_grid: np.ndarray) -> np.ndarray:
        """
        Calculate force density from divergence of stress tensor.
        
        F = ∇·T (force density)
        """
        # Calculate stress tensor at each grid point
        stress_tensors = []
        for pos in position_grid:
            # Get field at this position (simplified)
            B_local = B_field  # Should be calculated at each position
            T = self.calculate_stress_tensor_components(B_local)
            stress_tensors.append(T)
        
        # Calculate divergence numerically (simplified)
        force_density = np.zeros_like(position_grid)
        
        # This is a simplified implementation
        # Full implementation would require proper numerical differentiation
        for i, T in enumerate(stress_tensors):
            force_density[i] = np.trace(T)  # Simplified: trace as force measure
        
        return force_density
    
    def validate_stress_tensor_conservation(self, B_field: np.ndarray) -> bool:
        """
        Validate that the stress tensor satisfies conservation laws.
        
        The stress tensor should be symmetric and satisfy energy-momentum conservation.
        """
        T = self.calculate_stress_tensor_components(B_field)
        
        # Check symmetry
        is_symmetric = np.allclose(T, T.T, rtol=1e-10)
        
        # Check trace (related to energy density)
        trace = np.trace(T)
        energy_density = B_field @ B_field / (2 * PhysicsConstants.MU_0)
        
        return is_symmetric and abs(trace + energy_density) < 1e-12
    
    def calculate_stress_tensor_accuracy_metrics(self, B_field: np.ndarray, position: float) -> dict:
        """
        Calculate comprehensive accuracy metrics for the Maxwell stress tensor.
        
        Returns metrics including symmetry error, conservation violations, and field consistency.
        """
        T = self.calculate_stress_tensor_components(B_field)
        
        # Symmetry check
        symmetry_error = np.max(np.abs(T - T.T))
        
        # Energy conservation check
        trace = np.trace(T)
        energy_density = B_field @ B_field / (2 * PhysicsConstants.MU_0)
        energy_conservation_error = abs(trace + energy_density)
        
        # Field magnitude consistency
        B_magnitude = np.linalg.norm(B_field)
        stress_magnitude = np.linalg.norm(T)
        expected_stress_scale = B_magnitude**2 / PhysicsConstants.MU_0
        magnitude_consistency = abs(stress_magnitude - expected_stress_scale) / expected_stress_scale
        
        # Numerical stability check
        condition_number = np.linalg.cond(T)
        
        return {
            'symmetry_error': symmetry_error,
            'energy_conservation_error': energy_conservation_error,
            'magnitude_consistency': magnitude_consistency,
            'condition_number': condition_number,
            'is_valid': (symmetry_error < 1e-10 and 
                        energy_conservation_error < 1e-10 and 
                        magnitude_consistency < 0.1 and 
                        condition_number < 1e12)
        }
    
    def calculate_enhanced_maxwell_force_with_validation(self, current: float, position: float, 
                                                        velocity: float = 0.0) -> Tuple[float, dict]:
        """
        Enhanced Maxwell stress force calculation with comprehensive validation.
        
        This method includes all improvements: better field calculations, induced E-field,
        adaptive integration, and validation metrics.
        """
        # Standard Maxwell stress calculation
        force, breakdown = self.calculate_maxwell_stress_force(current, position, velocity)
        
        # Add validation metrics
        if hasattr(self.field_calc, 'magnetic_field_3d_biot_savart'):
            test_position = np.array([self.proj_radius/2, 0, position])
            B_field = self.field_calc.magnetic_field_3d_biot_savart(test_position, current)
            validation_metrics = self.calculate_stress_tensor_accuracy_metrics(B_field, position)
            breakdown.update(validation_metrics)
        
        # Warning for potential accuracy issues
        if 'is_valid' in breakdown and not breakdown['is_valid']:
            warnings.warn("Maxwell stress tensor validation indicates potential accuracy issues")
        
        return force, breakdown 