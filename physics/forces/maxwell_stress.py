"""
Maxwell Stress Tensor Calculations

This module implements the Maxwell stress tensor method for calculating
electromagnetic forces with high accuracy.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union, List
from scipy.integrate import quad, dblquad
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
        
        print(f"🔧 Maxwell stress tensor calculator initialized")
        print(f"   - Grid points: {self.maxwell_stress_grid_points}")
        print(f"   - Integration order: {self.integration_order}")
    
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
        
        # Total force (front - back + cylindrical contributions)
        total_force = force_front - force_back + force_cylindrical
        
        force_breakdown = {
            'maxwell_front': force_front,
            'maxwell_back': force_back,
            'maxwell_cylindrical': force_cylindrical,
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
            
            # CORRECTED: Get full magnetic field components at this position
            if hasattr(self.field_calc, 'magnetic_field_3d_biot_savart'):
                # Use 3D field calculation if available
                position = np.array([r, 0, z_pos])
                B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                B_r, B_phi, B_z = B_field[0], B_field[1], B_field[2]
            else:
                # Fallback to on-axis calculation with radial field estimate
                B_z = self.field_calc.magnetic_field_solenoid_on_axis(z_pos, current)
                
                # CORRECTED: Estimate radial field from field gradient
                # For solenoid: B_r ≈ -(r/2) × dB_z/dz
                if hasattr(self.field_calc, 'calculate_field_gradient'):
                    dB_dz = self.field_calc.calculate_field_gradient(z_pos, current)
                    B_r = -(r / 2.0) * dB_dz
                else:
                    B_r = 0.0
                B_phi = 0.0  # Azimuthal symmetry
            
            # Full magnetic field magnitude
            B_magnitude_squared = B_r**2 + B_phi**2 + B_z**2
            
            # CORRECTED: Full Maxwell stress tensor component T_zz
            # T_zz = (1/μ₀)[B_z² - (1/2)|B|²]
            T_zz = (1.0 / PhysicsConstants.MU_0) * (B_z**2 - 0.5 * B_magnitude_squared)
            
            # Integration element: r dr dφ → 2πr dr for axial symmetry
            return T_zz * 2 * np.pi * r
        
        # Integrate from 0 to projectile radius
        try:
            force, _ = quad(stress_integrand, 0, radius, 
                           limit=self.integration_order, epsabs=1e-12, epsrel=1e-9)
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
            
            # CORRECTED: Get full magnetic field components
            if hasattr(self.field_calc, 'magnetic_field_3d_biot_savart'):
                # Use 3D field calculation if available
                position = np.array([r_pos, 0, z])
                B_field = self.field_calc.magnetic_field_3d_biot_savart(position, current)
                B_r, B_phi, B_z = B_field[0], B_field[1], B_field[2]
            else:
                # Fallback: estimate fields from on-axis calculation
                B_z = self.field_calc.magnetic_field_solenoid_on_axis(z, current)
                
                # Estimate radial field using solenoid field gradient
                # B_r ≈ -(r/2) × dB_z/dz for cylindrical solenoid
                if hasattr(self.field_calc, 'calculate_field_gradient'):
                    dB_dz = self.field_calc.calculate_field_gradient(z, current)
                    B_r = -(r_pos / 2.0) * dB_dz
                else:
                    B_r = 0.0
                B_phi = 0.0  # Azimuthal symmetry
            
            # CORRECTED: Full Maxwell stress tensor component T_rz
            # T_rz = (1/μ₀) × B_r × B_z
            T_rz = (1.0 / PhysicsConstants.MU_0) * B_r * B_z
            
            # Integration element: circumference × dz = 2πr × dz
            return T_rz * 2 * np.pi * radius
        
        # Integrate along projectile length
        try:
            force, _ = quad(cylindrical_stress_integrand, z_start, z_end, 
                           limit=self.integration_order, epsabs=1e-12, epsrel=1e-9)
            return NumericalUtils.safe_numerical_operation(force, "maxwell_stress_cylindrical")
        except Exception as e:
            warnings.warn(f"Maxwell stress cylindrical integration failed: {e}")
            return 0.0
    
    def calculate_stress_tensor_components(self, B_field: np.ndarray, E_field: np.ndarray = None) -> np.ndarray:
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