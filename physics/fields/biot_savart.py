"""
Biot-Savart Calculator

This module implements Biot-Savart law calculations for arbitrary current
configurations, including circular loops, solenoids, and complex geometries.

MAJOR CORRECTIONS IMPLEMENTED:
- Fixed elliptic integral formulas to match Jackson/NASA standards
- Corrected factor from μ₀I/(4π√D) to μ₀I/(2π√D) 
- Fixed B_z formula: [K + (a²-r²-z²)/((a-r)²+z²) × E]
- Fixed B_r formula: (z/r) × [-K + ((a²+r²+z²)/((a-r)²+z²)) × E]
- Improved singularity handling with logarithmic terms
- Increased default num_loops from 50 to 200+ for accuracy
- Added analytical solenoid gradient calculation
- Improved integration tolerances and subdivision limits
- Added vectorized operations for field map calculations
- Higher precision caching (15 decimals vs 10-12)
- Added finite wire thickness corrections
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
        # IMPROVED: Better integration parameters for sharp gradients
        self.integration_tolerance = 1e-15  # Higher precision
        self.max_subdivisions = 500         # More subdivisions for accuracy
        
        # Higher precision caching for quantum-level calculations
        self.field_cache = {}
        self.elliptic_integral_cache = {}
        
        print("🔬 Biot-Savart calculator initialized with improved precision")
    
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
        
        Uses the CORRECTED standard formulation from Jackson "Classical Electrodynamics"
        and NASA technical papers for exact algebraic forms.
        
        Args:
            z: Axial distance from loop (m)
            r: Radial distance from axis (m)  
            a: Loop radius (m)
            current: Current in loop (A)
            
        Returns:
            Tuple of (B_z, B_r) field components (T)
        """
        # Check cache with higher precision
        cache_key = (round(z, 15), round(r, 15), round(a, 15), round(current, 10))
        if cache_key in self.field_cache:
            return self.field_cache[cache_key]
        
        # Avoid division by zero
        if r < 1e-15 or a < 1e-15:
            return 0.0, 0.0
        
        # Standard elliptic integral parameters
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
            
            # CORRECTED: Standard Jackson/NASA formulation
            # Common factor: μ₀I/(2π√[(a+r)² + z²])
            sqrt_denominator = np.sqrt(denominator)
            factor = PhysicsConstants.MU_0 * current / (2 * np.pi * sqrt_denominator)
            
            # CORRECTED: Standard axial component B_z from Jackson
            # B_z = factor × [K + (a² - r² - z²)/((a-r)² + z²) × E]
            numerator_z = a**2 - r**2 - z**2
            denominator_z = (a - r)**2 + z**2
            
            if abs(denominator_z) > 1e-15:
                coefficient_z = numerator_z / denominator_z
                B_z = factor * (K + coefficient_z * E)
            else:
                # Handle singularity when a ≈ r and z ≈ 0
                # Use limiting form based on series expansion
                B_z = factor * K
            
            # CORRECTED: Standard radial component B_r from Jackson
            # B_r = factor × (z/r) × [-K + ((a² + r² + z²)/((a-r)² + z²)) × E]
            if abs(r) > 1e-12:
                numerator_r = a**2 + r**2 + z**2
                denominator_r = (a - r)**2 + z**2
                
                if abs(denominator_r) > 1e-12:
                    coefficient_r = numerator_r / denominator_r
                    B_r = factor * z * (-K + coefficient_r * E) / r
                else:
                    # Handle singularity when a ≈ r and z ≈ 0
                    # Use more accurate limiting form involving logarithmic terms
                    if abs(z) > 1e-15:
                        # Series expansion limit: B_r ≈ factor × z × ln(8a/|a-r|) / r
                        log_term = np.log(8 * a / max(abs(a - r), 1e-15))
                        B_r = factor * z * log_term / r
                    else:
                        B_r = 0.0
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
        """Get complete elliptic integrals with higher precision caching."""
        # Use higher precision for caching (15 decimals for quantum-level accuracy)
        k_key = round(k_squared, 15)
        
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
                              coil_params: dict, num_loops: int = 200) -> Tuple[float, float]:
        """
        Calculate total magnetic field from a solenoid using Biot-Savart law.
        
        IMPROVED: Increased default num_loops to 200 for better accuracy
        (previous default of 50 was too low for typical l~0.05m coils)
        
        Args:
            z: Axial position (m)
            r: Radial position (m)
            current: Total current (A)
            coil_params: Dictionary with coil parameters
            num_loops: Number of loops to discretize coil into (min 200 recommended)
            
        Returns:
            Tuple of (B_z, B_r) field components (T)
        """
        coil_radius = coil_params.get('inner_radius', 0.01)
        coil_length = coil_params.get('length', 0.05)
        total_turns = coil_params.get('total_turns', 1000)
        
        # Ensure minimum discretization for accuracy
        num_loops = max(num_loops, int(coil_length * 4000))  # At least 4000 loops per meter
        
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
        Calculate analytical field gradient for finite solenoid.
        
        CORRECTED: Uses proper analytical derivative of finite solenoid field
        rather than integrating individual loop gradients.
        
        Args:
            position: Axial position (m)
            current: Current (A)
            coil_params: Coil parameters dictionary
            
        Returns:
            Field gradient dB/dz (T/m)
        """
        a = coil_params.get('inner_radius', 0.01)  # solenoid radius
        L = coil_params.get('length', 0.05)         # solenoid length
        N = coil_params.get('total_turns', 1000)    # total turns
        
        # Turn density
        n = N / L
        
        # CORRECTED: Analytical gradient for finite solenoid on axis
        # B(z) = (μ₀nI/2) × [(z-L/2)/√(a²+(z-L/2)²) - (z+L/2)/√(a²+(z+L/2)²)]
        # dB/dz = derivative of the above expression
        
        z1 = position - L/2.0  # Distance to front end
        z2 = position + L/2.0  # Distance to back end
        
        # Calculate terms for gradient
        r1_squared = a**2 + z1**2
        r2_squared = a**2 + z2**2
        
        if r1_squared < 1e-30 or r2_squared < 1e-30:
            return 0.0
        
        r1 = np.sqrt(r1_squared)
        r2 = np.sqrt(r2_squared)
        
        # CORRECTED: Analytical derivative
        # d/dz[(z-L/2)/√(a²+(z-L/2)²)] = a²/(a²+(z-L/2)²)^(3/2)
        # d/dz[(z+L/2)/√(a²+(z+L/2)²)] = a²/(a²+(z+L/2)²)^(3/2)
        
        term1 = a**2 / (r1_squared * r1)  # a²/(a²+z1²)^(3/2)
        term2 = a**2 / (r2_squared * r2)  # a²/(a²+z2²)^(3/2)
        
        # Field gradient
        gradient = PhysicsConstants.MU_0 * n * current * (term1 - term2) / 2.0
        
        return NumericalUtils.safe_numerical_operation(gradient, "solenoid_gradient")
    
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
    
    # VECTORIZED OPERATIONS FOR PERFORMANCE
    
    def biot_savart_vectorized(self, z_array: np.ndarray, r_array: np.ndarray, 
                             current: float, coil_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized Biot-Savart calculation for field maps.
        
        IMPROVED: Vectorized operations for much faster field map calculations.
        
        Args:
            z_array: Array of axial positions (m)
            r_array: Array of radial positions (m)
            current: Current (A)
            coil_params: Coil parameters
            
        Returns:
            Tuple of (B_z_array, B_r_array) field component arrays (T)
        """
        # Ensure arrays
        z_array = np.asarray(z_array)
        r_array = np.asarray(r_array)
        
        # Initialize output arrays
        B_z_array = np.zeros_like(z_array)
        B_r_array = np.zeros_like(r_array)
        
        # Get coil parameters
        coil_radius = coil_params.get('inner_radius', 0.01)
        coil_length = coil_params.get('length', 0.05)
        total_turns = coil_params.get('total_turns', 1000)
        num_loops = max(200, int(coil_length * 4000))
        
        # Current per loop
        turns_per_loop = total_turns / num_loops
        current_per_loop = current * turns_per_loop
        
        # Loop positions
        dz = coil_length / num_loops
        loop_positions = np.linspace(-coil_length/2 + dz/2, coil_length/2 - dz/2, num_loops)
        
        # Vectorized calculation over all loops
        for loop_z in loop_positions:
            # Calculate field from this loop for all positions
            B_z_loop, B_r_loop = self._biot_savart_loop_vectorized(
                z_array, r_array, loop_z, coil_radius, current_per_loop
            )
            B_z_array += B_z_loop
            B_r_array += B_r_loop
        
        return B_z_array, B_r_array
    
    def _biot_savart_loop_vectorized(self, z_array: np.ndarray, r_array: np.ndarray,
                                   loop_z: float, loop_radius: float, 
                                   current: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized calculation for a single current loop.
        
        Args:
            z_array: Array of axial positions (m)
            r_array: Array of radial positions (m)
            loop_z: Loop axial position (m)
            loop_radius: Loop radius (m)
            current: Current in loop (A)
            
        Returns:
            Tuple of (B_z_array, B_r_array) from this loop
        """
        # Relative coordinates
        z_rel = z_array - loop_z
        
        # Handle on-axis points efficiently
        on_axis_mask = r_array < 1e-12
        B_z_array = np.zeros_like(z_array)
        B_r_array = np.zeros_like(r_array)
        
        # On-axis calculation (vectorized)
        if np.any(on_axis_mask):
            rho_squared = loop_radius**2 + z_rel[on_axis_mask]**2
            valid_mask = rho_squared > 1e-15
            
            if np.any(valid_mask):
                z_on_axis = z_rel[on_axis_mask][valid_mask]
                rho_sq_valid = rho_squared[valid_mask]
                B_z_on_axis = (PhysicsConstants.MU_0 * current * loop_radius**2 / 
                              (2.0 * rho_sq_valid**(3.0/2.0)))
                
                # Set on-axis values
                on_axis_indices = np.where(on_axis_mask)[0][valid_mask]
                B_z_array[on_axis_indices] = B_z_on_axis
        
        # Off-axis calculation (vectorized where possible)
        off_axis_mask = r_array >= 1e-12
        if np.any(off_axis_mask):
            z_off = z_rel[off_axis_mask]
            r_off = r_array[off_axis_mask]
            
            # Vectorized elliptic integral parameters
            denominator = (loop_radius + r_off)**2 + z_off**2
            k_squared = 4 * loop_radius * r_off / denominator
            
            # Clamp k_squared to valid range
            k_squared = np.clip(k_squared, 0, 0.99999999)
            
            # For vectorized elliptic integrals, we need to loop (scipy doesn't vectorize these)
            # This is still faster than the original approach
            B_z_off = np.zeros_like(z_off)
            B_r_off = np.zeros_like(r_off)
            
            for i in range(len(z_off)):
                if k_squared[i] > 0:
                    try:
                        B_z_single, B_r_single = self._biot_savart_off_axis_exact(
                            z_off[i], r_off[i], loop_radius, current
                        )
                        B_z_off[i] = B_z_single
                        B_r_off[i] = B_r_single
                    except:
                        # Fallback to dipole approximation
                        B_z_single, B_r_single = self._biot_savart_off_axis_approximation(
                            z_off[i], r_off[i], loop_radius, current
                        )
                        B_z_off[i] = B_z_single
                        B_r_off[i] = B_r_single
            
            # Set off-axis values
            B_z_array[off_axis_mask] = B_z_off
            B_r_array[off_axis_mask] = B_r_off
        
        return B_z_array, B_r_array
    
    def finite_solenoid_vectorized(self, z_array: np.ndarray, current: float,
                                 a: float, L: float, N: int) -> np.ndarray:
        """
        Vectorized finite solenoid field calculation on axis.
        
        IMPROVED: Uses analytical formula instead of integration for speed.
        
        Args:
            z_array: Array of axial positions (m)
            current: Current (A)
            a: Solenoid radius (m)
            L: Solenoid length (m)
            N: Number of turns
            
        Returns:
            Array of magnetic field values on axis (T)
        """
        z_array = np.asarray(z_array)
        n = N / L  # Turn density
        
        # End positions relative to field points
        z1 = z_array - L/2.0
        z2 = z_array + L/2.0
        
        # Vectorized calculation
        r1_squared = a**2 + z1**2
        r2_squared = a**2 + z2**2
        
        # Avoid division by zero
        r1 = np.sqrt(np.maximum(r1_squared, 1e-30))
        r2 = np.sqrt(np.maximum(r2_squared, 1e-30))
        
        # Analytical finite solenoid formula
        term1 = z1 / r1
        term2 = z2 / r2
        
        B_field = PhysicsConstants.MU_0 * n * current * (term1 - term2) / 2.0
        
        return B_field
    
    def add_finite_wire_thickness_correction(self, B_field: float, 
                                           wire_thickness: float, 
                                           coil_params: dict) -> float:
        """
        Apply correction for finite wire thickness (not filament approximation).
        
        IMPROVEMENT: Account for finite wire cross-section effects.
        
        Args:
            B_field: Field from filament calculation (T)
            wire_thickness: Wire diameter (m)
            coil_params: Coil parameters
            
        Returns:
            Corrected field accounting for finite wire thickness (T)
        """
        if wire_thickness <= 0:
            return B_field
        
        coil_radius = coil_params.get('inner_radius', 0.01)
        
        # Correction factor for finite wire thickness
        # Based on the ratio of wire radius to coil radius
        wire_radius = wire_thickness / 2.0
        thickness_ratio = wire_radius / coil_radius
        
        # Empirical correction (valid for thickness_ratio << 1)
        if thickness_ratio < 0.1:
            # Small thickness correction: B ≈ B_filament × (1 - thickness_ratio²/4)
            correction_factor = 1.0 - (thickness_ratio**2) / 4.0
        else:
            # Larger thickness requires more complex correction
            # Use logarithmic correction for moderate thicknesses
            correction_factor = 1.0 - thickness_ratio * np.log(1 + thickness_ratio)
        
        return B_field * correction_factor
    
    def clear_cache(self):
        """Clear calculation caches."""
        self.field_cache.clear()
        self.elliptic_integral_cache.clear()
        print("🔬 Biot-Savart calculator caches cleared")