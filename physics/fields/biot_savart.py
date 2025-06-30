"""
Biot-Savart Calculator

This module implements Biot-Savart law calculations for arbitrary current
configurations, including circular loops, solenoids, and complex geometries.
"""

import numpy as np
from scipy.special import ellipk, ellipe
from scipy.integrate import quad, simpson
from typing import Optional, Tuple, Union, List
import warnings
from ..core import PhysicsConstants, NumericalUtils


class BiotSavartCalculator:
    """
    Biot-Savart law calculations for arbitrary current configurations.
    
    Provides accurate magnetic field calculations for:
    - Circular current loops
    - Finite solenoids
    - Complex current distributions
    - Off-axis field calculations
    """
    
    def __init__(self):
        """Initialize Biot-Savart calculator."""
        self.integration_tolerance = 1e-12
        self.max_subdivisions = 100
        
        # Caching for performance
        self.field_cache = {}
        self.elliptic_integral_cache = {}
        
        print("🔬 Biot-Savart calculator initialized")
    
    def biot_savart_circular_loop(self, z: float, r: float, loop_z: float, 
                                loop_radius: float, current: float) -> Tuple[float, float]:
        """
        Calculate magnetic field from a circular current loop using Biot-Savart law.
        
        Args:
            z: Axial position (m)
            r: Radial position (m)
            loop_z: Loop axial position (m)
            loop_radius: Loop radius (m)
            current: Current in loop (A)
            
        Returns:
            Tuple of (B_z, B_r) field components (T)
        """
        # Relative coordinates
        z_rel = z - loop_z
        
        # Handle special cases
        if r < 1e-12:
            # On axis - exact analytical solution
            rho_squared = loop_radius**2 + z_rel**2
            if rho_squared < 1e-15:
                # At loop center
                B_z = PhysicsConstants.MU_0 * current / (2.0 * loop_radius) if loop_radius > 0 else 0.0
            else:
                # On axis but not at center
                B_z = PhysicsConstants.MU_0 * current * loop_radius**2 / (2.0 * rho_squared**(3.0/2.0))
            B_r = 0.0
            return B_z, B_r
        
        # For off-axis calculations, use exact elliptic integral solution
        return self._biot_savart_off_axis_exact(z_rel, r, loop_radius, current)
    
    def _biot_savart_off_axis_exact(self, z: float, r: float, a: float, current: float) -> Tuple[float, float]:
        """
        Exact off-axis Biot-Savart calculation using elliptic integrals.
        
        Uses the standard textbook formulation with improved numerical stability.
        Reference: Jackson "Classical Electrodynamics" and Ortner et al. (2023)
        
        Args:
            z: Axial distance from loop (m)
            r: Radial distance from axis (m)  
            a: Loop radius (m)
            current: Current in loop (A)
            
        Returns:
            Tuple of (B_z, B_r) field components (T)
        """
        # Check cache
        cache_key = (round(z, 10), round(r, 10), round(a, 10), round(current, 6))
        if cache_key in self.field_cache:
            return self.field_cache[cache_key]
        
        # Avoid division by zero
        if r < 1e-15 or a < 1e-15:
            return 0.0, 0.0
        
        # CORRECTED: Standard elliptic integral parameters
        # k² = 4ar/[(a+r)² + z²] (standard form from literature)
        denominator = (a + r)**2 + z**2
        k_squared = 4 * a * r / denominator
        
        # Ensure k² is in valid range [0, 1)
        if k_squared >= 1.0:
            k_squared = 0.99999999
        elif k_squared < 0:
            k_squared = 0.0
        
        try:
            # Complete elliptic integrals
            K, E = self._get_elliptic_integrals(k_squared)
            
            # CORRECTED: Standard formulation for better numerical stability
            # Common factor: μ₀I/(4π√[(a+r)² + z²])
            sqrt_denominator = np.sqrt(denominator)
            factor = PhysicsConstants.MU_0 * current / (4 * np.pi * sqrt_denominator)
            
            # CORRECTED: Standard axial component B_z
            # B_z = factor × [(2-k²)K(k²) - 2E(k²)]
            B_z = factor * ((2 - k_squared) * K - 2 * E)
            
            # CORRECTED: Standard radial component B_r  
            # B_r = factor × (z/r) × [K(k²) - ((a²+r²+z²)/((a-r)²+z²))E(k²)]
            if abs(r) > 1e-12:
                # Handle the coefficient of E(k²) carefully
                numerator = a**2 + r**2 + z**2
                denominator_E = (a - r)**2 + z**2
                
                if abs(denominator_E) > 1e-12:
                    coefficient_E = numerator / denominator_E
                    B_r = factor * z * (K - coefficient_E * E) / r
                else:
                    # Handle singularity when a ≈ r and z ≈ 0
                    # Use limiting form: B_r ≈ factor × z × (K - E) / r
                    B_r = factor * z * (K - E) / r
            else:
                B_r = 0.0
            
            # Cache result
            result = (NumericalUtils.safe_numerical_operation(B_z, "biot_savart_Bz"),
                     NumericalUtils.safe_numerical_operation(B_r, "biot_savart_Br"))
            
            self.field_cache[cache_key] = result
            return result
            
        except Exception as e:
            warnings.warn(f"Elliptic integral calculation failed: {e}, using approximation")
            return self._biot_savart_off_axis_approximation(z, r, a, current)
    
    def _get_elliptic_integrals(self, k_squared: float) -> Tuple[float, float]:
        """Get complete elliptic integrals with caching."""
        k_key = round(k_squared, 12)
        
        if k_key in self.elliptic_integral_cache:
            return self.elliptic_integral_cache[k_key]
        
        K = ellipk(k_squared)
        E = ellipe(k_squared)
        
        self.elliptic_integral_cache[k_key] = (K, E)
        return K, E
    
    def _biot_savart_off_axis_approximation(self, z: float, r: float, a: float, current: float) -> Tuple[float, float]:
        """
        Approximate off-axis calculation for fallback.
        """
        # Corrected distance calculation
        rho_total = np.sqrt(r**2 + z**2)
        
        if rho_total < 1e-15:
            return 0.0, 0.0
        
        # Magnetic dipole moment
        m = np.pi * a**2 * current
        
        # Dipole field components (CORRECTED)
        factor = PhysicsConstants.MU_0 * m / (4 * np.pi * rho_total**3)
        
        # Far-field dipole approximation
        cos_theta = z / rho_total
        sin_theta = r / rho_total
        
        B_z = factor * (3 * cos_theta**2 - 1)
        B_r = factor * (3 * cos_theta * sin_theta) if r > 1e-12 else 0.0
        
        return B_z, B_r
    
    def biot_savart_total_field(self, z: float, r: float, current: float, 
                              coil_params: dict, num_loops: int = 50) -> Tuple[float, float]:
        """
        Calculate total magnetic field from a solenoid using Biot-Savart law.
        
        Args:
            z: Axial position (m)
            r: Radial position (m)
            current: Total current (A)
            coil_params: Dictionary with coil parameters
            num_loops: Number of loops to discretize coil into
            
        Returns:
            Tuple of (B_z, B_r) field components (T)
        """
        coil_radius = coil_params.get('inner_radius', 0.01)
        coil_length = coil_params.get('length', 0.05)
        total_turns = coil_params.get('total_turns', 1000)
        
        # Current per discretized loop
        turns_per_loop = total_turns / num_loops
        current_per_loop = current * turns_per_loop
        
        B_z_total = 0.0
        B_r_total = 0.0
        
        # Discretize coil into loops
        dz = coil_length / num_loops
        
        for i in range(num_loops):
            loop_z = -coil_length/2.0 + (i + 0.5) * dz
            
            B_z_loop, B_r_loop = self.biot_savart_circular_loop(
                z, r, loop_z, coil_radius, current_per_loop
            )
            
            B_z_total += B_z_loop
            B_r_total += B_r_loop
        
        return B_z_total, B_r_total
    
    def magnetic_field_on_axis_circular_loop(self, z: float, loop_radius: float, 
                                           current: float, loop_position: float = 0.0) -> float:
        """
        Calculate magnetic field on axis from a circular loop.
        
        Args:
            z: Axial position (m)
            loop_radius: Loop radius (m)
            current: Current in loop (A)
            loop_position: Axial position of loop (m)
            
        Returns:
            Magnetic field strength on axis (T)
        """
        z_rel = z - loop_position
        
        # Handle special case at loop center
        if abs(z_rel) < 1e-15:
            return PhysicsConstants.MU_0 * current / (2.0 * loop_radius) if loop_radius > 0 else 0.0
        
        # General case
        denominator = (loop_radius**2 + z_rel**2)**(3.0/2.0)
        
        if denominator < 1e-15:
            return 0.0
        
        B_z = PhysicsConstants.MU_0 * current * loop_radius**2 / (2.0 * denominator)
        
        return NumericalUtils.safe_numerical_operation(B_z, "circular_loop_field")
    
    def magnetic_field_finite_solenoid_on_axis(self, z: float, a: float, l: float, 
                                             N: int, current: float) -> float:
        """
        Calculate magnetic field on axis of finite solenoid using integration.
        
        Args:
            z: Axial position (m)
            a: Solenoid radius (m)
            l: Solenoid length (m)
            N: Number of turns
            current: Current (A)
            
        Returns:
            Magnetic field strength on axis (T)
        """
        # Current density (turns per unit length)
        n = N / l
        
        def field_contribution(z_dist):
            """Field contribution from element at z_dist."""
            z_rel = z - z_dist
            denominator = (a**2 + z_rel**2)**(3.0/2.0)
            
            if denominator < 1e-15:
                return 0.0
            
            return PhysicsConstants.MU_0 * current * n * a**2 / (2.0 * denominator)
        
        try:
            # Integrate over solenoid length
            field_total, _ = quad(
                field_contribution,
                -l/2.0,
                l/2.0,
                epsabs=self.integration_tolerance,
                epsrel=self.integration_tolerance,
                limit=self.max_subdivisions
            )
            
            return NumericalUtils.safe_numerical_operation(field_total, "finite_solenoid_field")
            
        except Exception as e:
            warnings.warn(f"Integration failed: {e}, using simple approximation")
            
            # Fallback to simple approximation
            return self._finite_solenoid_approximation(z, a, l, N, current)
    
    def _finite_solenoid_approximation(self, z: float, a: float, l: float, N: int, current: float) -> float:
        """Simple approximation for finite solenoid field."""
        # Use end effects formula
        z1 = z + l/2.0  # Distance to one end
        z2 = z - l/2.0  # Distance to other end
        
        term1 = z1 / np.sqrt(a**2 + z1**2) if (a**2 + z1**2) > 0 else 0.0
        term2 = z2 / np.sqrt(a**2 + z2**2) if (a**2 + z2**2) > 0 else 0.0
        
        n = N / l  # turns per unit length
        field = PhysicsConstants.MU_0 * n * current * (term1 - term2) / 2.0
        
        return field
    
    def calculate_field_gradient_analytical(self, position: float, current: float, 
                                          coil_params: dict) -> float:
        """
        Calculate analytical field gradient for circular loop or solenoid.
        
        Args:
            position: Axial position (m)
            current: Current (A)
            coil_params: Coil parameters dictionary
            
        Returns:
            Field gradient dB/dz (T/m)
        """
        coil_radius = coil_params.get('inner_radius', 0.01)
        coil_length = coil_params.get('length', 0.05)
        total_turns = coil_params.get('total_turns', 1000)
        
        def gradient_contribution(z_dist):
            """Gradient contribution from element at z_dist."""
            z_rel = position - z_dist
            a = coil_radius
            
            # Analytical gradient of circular loop field
            if abs(z_rel) < 1e-15:
                return 0.0  # Zero gradient at loop center
            
            denominator = (a**2 + z_rel**2)**(5.0/2.0)
            
            if denominator < 1e-15:
                return 0.0
            
            # dB/dz for circular loop
            gradient = PhysicsConstants.MU_0 * current * total_turns * a**2 * z_rel * (-3.0) / (2.0 * coil_length * denominator)
            
            return gradient
        
        try:
            # Integrate gradient contributions
            gradient_total, _ = quad(
                gradient_contribution,
                -coil_length/2.0,
                coil_length/2.0,
                epsabs=self.integration_tolerance,
                epsrel=self.integration_tolerance,
                limit=self.max_subdivisions
            )
            
            return NumericalUtils.safe_numerical_operation(gradient_total, "field_gradient")
            
        except Exception as e:
            warnings.warn(f"Gradient integration failed: {e}")
            return 0.0
    
    def calculate_multipole_expansion(self, position: np.ndarray, current: float, 
                                    coil_params: dict, max_order: int = 6) -> np.ndarray:
        """
        Calculate magnetic field using multipole expansion.
        
        Args:
            position: Position vector [x, y, z] (m)
            current: Current (A)
            coil_params: Coil parameters
            max_order: Maximum multipole order
            
        Returns:
            Magnetic field vector [Bx, By, Bz] (T)
        """
        # Calculate multipole moments
        moments = self._calculate_multipole_moments(current, coil_params, max_order)
        
        # Position in spherical coordinates
        r = np.linalg.norm(position)
        if r < 1e-15:
            return np.zeros(3)
        
        theta = np.arccos(position[2] / r) if r > 0 else 0.0
        phi = np.arctan2(position[1], position[0])
        
        # Calculate field from multipole expansion
        B_field = np.zeros(3)
        
        for l in range(max_order + 1):
            for m in range(-l, l + 1):
                moment_key = (l, m)
                if moment_key in moments:
                    B_contribution = self._multipole_field_contribution(
                        position, moments[moment_key], l, m
                    )
                    B_field += B_contribution
        
        return B_field
    
    def _calculate_multipole_moments(self, current: float, coil_params: dict, max_order: int) -> dict:
        """Calculate multipole moments for coil."""
        moments = {}
        
        coil_radius = coil_params.get('inner_radius', 0.01)
        coil_length = coil_params.get('length', 0.05)
        total_turns = coil_params.get('total_turns', 1000)
        
        # For a solenoid, only axial moments (m=0) are non-zero due to symmetry
        for l in range(0, max_order + 1, 2):  # Only even l for solenoid
            # Simplified moment calculation for solenoid
            if l == 0:
                # Magnetic dipole moment
                moments[(l, 0)] = np.pi * coil_radius**2 * current * total_turns
            else:
                # Higher-order moments (simplified)
                moments[(l, 0)] = moments[(0, 0)] * (coil_radius / 1.0)**(l-2) / (l + 1)
        
        return moments
    
    def _multipole_field_contribution(self, position: np.ndarray, moment: float, 
                                    l: int, m: int) -> np.ndarray:
        """Calculate field contribution from multipole moment."""
        r = np.linalg.norm(position)
        
        if r < 1e-15 or l < 1:
            return np.zeros(3)
        
        # Simplified calculation for axial symmetry (m=0)
        if m == 0:
            z = position[2]
            rho = np.sqrt(position[0]**2 + position[1]**2)
            
            # Field components for axisymmetric multipole
            factor = PhysicsConstants.MU_0 * moment / (4 * np.pi * r**(l+2))
            
            # Axial component
            B_z = factor * (l + 1) * z**(l) / r**l
            
            # Radial component
            if rho > 1e-12:
                B_rho = -factor * l * z**(l-1) * rho / r**(l-1)
                B_x = B_rho * position[0] / rho
                B_y = B_rho * position[1] / rho
            else:
                B_x = B_y = 0.0
            
            return np.array([B_x, B_y, B_z])
        
        return np.zeros(3)  # Non-axisymmetric terms are zero for solenoid
    
    def clear_cache(self):
        """Clear calculation caches."""
        self.field_cache.clear()
        self.elliptic_integral_cache.clear()
        print("🔬 Biot-Savart calculator caches cleared") 