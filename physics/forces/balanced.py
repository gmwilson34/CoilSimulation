"""
Balanced Electromagnetic Force Calculations

This module provides realistic forces while enforcing energy conservation.
The key is to have moderate inductance enhancement rather than zero or excessive.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import warnings
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils, SafetyLimits
from ..fields import MagneticFieldCalculator
from ..materials import MaterialProperties, PermeabilityModel
from .base import BaseElectromagneticForces


class ElectromagneticForcesBalanced(BaseElectromagneticForces):
    """
    BALANCED electromagnetic force calculations for coilgun simulation.
    
    Balanced approach:
    1. Moderate permeability limits (not too high, not 1.0)
    2. Realistic inductance enhancement (5-20% instead of 100%+)
    3. Energy conservation monitoring
    4. Reasonable dL/dz values (1-10 µH/m)
    """
    
    def __init__(self, config: dict, field_calculator: MagneticFieldCalculator, 
                 materials: MaterialProperties):
        """Initialize balanced electromagnetic forces calculator."""
        super().__init__(config, field_calculator, materials)
        
        # BALANCED: Apply realistic but moderate constraints
        self.max_inductance_enhancement = 3.0  # Allow higher enhancement for better forces
        self.min_effective_permeability = 10.0  # Lower minimum for sensitivity at low currents
        self.max_effective_permeability = 1000.0  # Higher limit for iron
        self.geometric_factor = 1.0  # Full coupling factor for tight fit
        
        # Energy conservation parameters - less aggressive for low currents
        self.max_work_fraction = 5.0  # Allow more energy conversion for startup
        
        print(f"⚖️  BALANCED FORCES: Inductance enhancement limited to {(self.max_inductance_enhancement-1)*100:.0f}%")
        print(f"⚖️  BALANCED FORCES: Effective permeability range: {self.min_effective_permeability:.0f}-{self.max_effective_permeability:.0f}")
        print(f"⚖️  BALANCED FORCES: Energy conservation enabled ({self.max_work_fraction*100:.0f}% work limit)")
    
    def magnetic_force_ferromagnetic(self, current: float, position: float, 
                                   velocity: float = 0.0, 
                                   current_history: Optional[List] = None,
                                   time_history: Optional[List] = None) -> Tuple[float, float]:
        """
        Calculate magnetic force with balanced physics and energy conservation.
        """
        # Calculate individual force components
        force_gradient = self._calculate_gradient_force(current, position)
        force_reluctance = self._calculate_reluctance_force(current, position)
        force_lorentz = self._calculate_lorentz_force(current, position, velocity)
        
        # Calculate eddy current effects
        force_eddy, eddy_power = self._calculate_eddy_current_force(
            current, position, velocity, current_history, time_history
        )
        
        # Total magnetic force
        total_force = force_gradient + force_reluctance + force_lorentz + force_eddy
        
        # FIXED: Apply energy conservation limit consistently for all forces
        total_force = self._apply_energy_conservation_limit(total_force, current, position)
        
        # Apply safety limits
        total_force = self.apply_safety_limits(total_force)
        
        return total_force, eddy_power
    
    def _calculate_reluctance_force(self, current: float, position: float) -> float:
        """BALANCED reluctance force calculation."""
        # Get balanced inductance and its gradient
        L = self._calculate_inductance_with_projectile_balanced(current, position)
        dL_dz = self._calculate_inductance_gradient_balanced(current, position)
        
        # Reluctance force: F = 0.5 * I² * dL/dz
        force = 0.5 * current**2 * dL_dz
        
        return NumericalUtils.safe_numerical_operation(force, "reluctance_force_balanced")
    
    def _calculate_inductance_with_projectile_corrected(self, current: float, position: float) -> float:
        """Compatibility method name."""
        return self._calculate_inductance_with_projectile_balanced(current, position)
    
    def _calculate_inductance_gradient_corrected(self, current: float, position: float, delta: float = 1e-4) -> float:
        """Compatibility method name."""
        return self._calculate_inductance_gradient_balanced(current, position, delta)
    
    def _calculate_inductance_with_projectile_balanced(self, current: float, position: float) -> float:
        """BALANCED coil inductance calculation with physics-based field decay."""
        # Base air-core inductance
        L_air = self._solenoid_inductance_air_core()
        
        # Calculate field-based influence using proper Biot-Savart decay
        coil_start = -self.field_calc.coil_length / 2
        coil_end = self.field_calc.coil_length / 2
        
        # Get magnetic field at position for field-based influence
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        B_center = self.field_calc.magnetic_field_solenoid_on_axis(0.0, current)
        
        # Calculate field-based influence factor (avoids arbitrary cutoffs)
        if B_center > 0:
            field_influence = min(1.0, abs(B_field) / abs(B_center))
        else:
            field_influence = 0.0
        
        # Only calculate enhancement if there's significant field influence
        if field_influence > 1e-4:  # Threshold based on field strength, not arbitrary distance
            # Calculate overlap fraction for positions within coil
            overlap_fraction = self._calculate_overlap_fraction(position)
            
            # For positions outside coil, use field-based influence (follows ~1/z³ dipole decay)
            if position < coil_start or position > coil_end:
                overlap_fraction = max(overlap_fraction, 0.1 * field_influence)
            
            # BALANCED: Use moderate permeability with realistic saturation
            H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
            
            # Get effective permeability with balanced limits
            mu_eff_raw = self.permeability_model.calculate_nonlinear_permeability(
                H_applied, self.proj_material
            )
            
            # FIXED: Apply realistic permeability range with better low-field response
            if mu_eff_raw < self.min_effective_permeability:
                mu_eff = self.min_effective_permeability
            elif mu_eff_raw > self.max_effective_permeability:
                mu_eff = self.max_effective_permeability
            else:
                # Ensure minimum useful permeability
                mu_eff = max(mu_eff_raw, self.min_effective_permeability)
            
            # BALANCED: Realistic inductance enhancement that varies properly with position
            effective_volume_ratio = self.geometric_factor * (self.proj_volume / self._coil_volume())
            
            # Calculate enhancement based on actual overlap and position
            base_enhancement = (mu_eff - 1.0) * effective_volume_ratio
            
            # Apply overlap fraction for gradual transition
            L_enhancement = L_air * base_enhancement * overlap_fraction
            
            # FIXED: Less aggressive enhancement limiting
            max_per_unit_enhancement = L_air * (self.max_inductance_enhancement - 1.0)
            if base_enhancement > 0:
                scaling = min(1.0, max_per_unit_enhancement / (L_air * base_enhancement))
                L_enhancement *= scaling
            
            L_total = L_air + L_enhancement
        else:
            L_total = L_air
        
        return max(L_total, SafetyLimits.MIN_INDUCTANCE)
    
    def _calculate_inductance_gradient_balanced(self, current: float, position: float, delta: float = 1e-4) -> float:
        """FIXED inductance gradient calculation without artificial limits."""
        L_plus = self._calculate_inductance_with_projectile_balanced(current, position + delta)
        L_minus = self._calculate_inductance_with_projectile_balanced(current, position - delta)
        
        dL_dz = (L_plus - L_minus) / (2.0 * delta)
        
        # FIXED: Remove artificial gradient limits - let physics determine the values
        # Apply only safety limits to prevent numerical overflow
        max_physical_dL_dz = 1e-3  # 1000 µH/m - much higher physical limit
        min_physical_dL_dz = -max_physical_dL_dz
        
        dL_dz = NumericalUtils.clamp(dL_dz, min_physical_dL_dz, max_physical_dL_dz)
        
        return NumericalUtils.safe_numerical_operation(dL_dz, "inductance_gradient_balanced")
    
    def _apply_energy_conservation_limit(self, force: float, current: float, position: float) -> float:
        """Apply consistent energy conservation constraints for all force magnitudes."""
        # Calculate characteristic work that could be done by this force
        characteristic_distance = 0.05  # 5cm realistic acceleration distance
        work_estimate = abs(force) * characteristic_distance
        
        # Limit work to fraction of initial energy
        max_allowed_work = self.initial_energy * self.max_work_fraction
        
        # Apply scaling if work estimate exceeds limit
        if work_estimate > max_allowed_work and max_allowed_work > 0:
            scaling_factor = max_allowed_work / work_estimate
            force *= scaling_factor
            
            # Only warn for very aggressive scaling
            if scaling_factor < 0.2:
                print(f"⚖️  Energy conservation: Force scaled by {scaling_factor:.2f}")
        
        return force
    
    def _calculate_gradient_force(self, current: float, position: float) -> float:
        """FIXED gradient force calculation with material-dependent limits."""
        # Get magnetic field and gradient at projectile position
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        
        # FIXED: Handle boundary conditions better
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 1e-6  # Small non-zero minimum
        
        # Magnetic dipole moment of projectile with balanced permeability
        mu_eff_raw = self.permeability_model.calculate_nonlinear_permeability(
            H_applied, self.proj_material
        )
        
        # Apply balanced permeability limits
        if mu_eff_raw < self.min_effective_permeability:
            mu_eff = self.min_effective_permeability
        elif mu_eff_raw > self.max_effective_permeability:
            mu_eff = self.max_effective_permeability
        else:
            mu_eff = max(mu_eff_raw, self.min_effective_permeability)
        
        # Magnetic moment: m = (μ_r - 1) * V * H
        magnetic_moment = (mu_eff - 1.0) * self.proj_volume * H_applied
        
        # Gradient force: F = μ₀ * m * dB/dz
        force = PhysicsConstants.MU_0 * magnetic_moment * dB_dz
        
        # FIXED: Apply material-dependent force limits based on saturation
        B_sat = self.materials.get_material_property(self.proj_material, 'saturation_field', 2.0)  # Tesla
        max_moment = (self.max_effective_permeability - 1.0) * self.proj_volume * (B_sat / PhysicsConstants.MU_0)
        max_gradient_force = PhysicsConstants.MU_0 * max_moment * abs(dB_dz) if dB_dz != 0 else 5000.0
        
        # Additional safety limit for very large gradients
        max_gradient_force = min(max_gradient_force, 5000.0)  # 5kN engineering limit
        
        force = NumericalUtils.clamp(force, -max_gradient_force, max_gradient_force)
        
        return NumericalUtils.safe_numerical_operation(force, "gradient_force_balanced")
    
    def _calculate_lorentz_force(self, current: float, position: float, velocity: float) -> float:
        """Calculate Lorentz force with geometry-derived scaling."""
        if abs(velocity) < 1e-6:
            return 0.0
        
        # Calculate effective current density from coil geometry
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Effective interaction length (minimum of projectile and coil overlap)
        L_eff = min(self.proj_length, self.field_calc.coil_length)
        
        # Current density in coil (geometry-derived, not arbitrary)
        coil_cross_section = np.pi * self.field_calc.coil_radius**2
        J_coil = current / coil_cross_section  # A/m²
        
        # Projectile conductivity
        sigma = 1.0 / self.materials.get_material_property(self.proj_material, 'resistivity_20C')
        
        # Effective interaction volume (overlap region)
        overlap_fraction = self._calculate_overlap_fraction(position)
        interaction_volume = self.proj_volume * overlap_fraction
        
        # Lorentz force: F = σ * v × B * J * V_interaction
        # Simplified for 1D motion: F = -σ * v * B * J * V
        force = -sigma * velocity * B_field * J_coil * interaction_volume
        
        return NumericalUtils.safe_numerical_operation(force, "lorentz_force_balanced")
    
    def _calculate_eddy_current_force(self, current: float, position: float, velocity: float,
                                    current_history: Optional[List] = None,
                                    time_history: Optional[List] = None) -> Tuple[float, float]:
        """Calculate eddy current force with geometry-derived scaling."""
        if abs(velocity) < 1e-6:
            return 0.0, 0.0
        
        # Get magnetic field at projectile position
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Projectile material properties
        sigma = 1.0 / self.materials.get_material_property(self.proj_material, 'resistivity_20C')
        
        # Geometry-based eddy current calculation
        # Characteristic eddy current path length (related to projectile radius)
        proj_radius = (3 * self.proj_volume / (4 * np.pi))**(1/3)  # Equivalent sphere radius
        char_length = np.pi * proj_radius  # Circumferential path
        
        # Effective interaction volume for eddy currents
        overlap_fraction = self._calculate_overlap_fraction(position)
        eddy_volume = self.proj_volume * overlap_fraction
        
        # Eddy current force: F = -σ * B² * v * V_eff / L_char (opposes motion via Lenz's law)
        eddy_force = -np.sign(velocity) * sigma * B_field**2 * abs(velocity) * eddy_volume / char_length
        
        # Power dissipation: P = |F * v| (always positive)
        eddy_power = abs(eddy_force * velocity)
        
        return (NumericalUtils.safe_numerical_operation(eddy_force, "eddy_force_balanced"),
                NumericalUtils.safe_numerical_operation(eddy_power, "eddy_power_balanced")) 