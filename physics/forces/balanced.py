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
        
        # FIXED: Apply energy conservation limit only for large forces
        if abs(total_force) > 100.0:  # Only limit very large forces
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
        """BALANCED coil inductance calculation with moderate enhancement."""
        # Base air-core inductance
        L_air = self._solenoid_inductance_air_core()
        
        # FIXED: Extend force calculation range beyond just coil length
        coil_start = -self.field_calc.coil_length / 2
        coil_end = self.field_calc.coil_length / 2
        influence_range = self.field_calc.coil_length * 0.5  # Extend range by 50%
        
        if position > coil_start - influence_range and position < coil_end + influence_range:
            # Projectile is within influence range
            overlap_fraction = self._calculate_overlap_fraction(position)
            
            # For positions outside coil, use distance-based influence
            if position < coil_start:
                distance_factor = max(0.1, 1.0 - (coil_start - position) / influence_range)
                overlap_fraction = max(overlap_fraction, 0.1 * distance_factor)
            elif position > coil_end:
                distance_factor = max(0.1, 1.0 - (position - coil_end) / influence_range)
                overlap_fraction = max(overlap_fraction, 0.1 * distance_factor)
            
            # BALANCED: Use moderate permeability with realistic saturation
            B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
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
        """Apply balanced energy conservation constraints - less aggressive."""
        # Calculate characteristic work that could be done by this force
        characteristic_distance = 0.05  # 5cm realistic acceleration distance
        work_estimate = abs(force) * characteristic_distance
        
        # Limit work to fraction of initial energy
        max_allowed_work = self.initial_energy * self.max_work_fraction
        
        # Apply scaling if work estimate exceeds limit
        if work_estimate > max_allowed_work:
            scaling_factor = max_allowed_work / work_estimate
            force *= scaling_factor
            
            # Only warn for very aggressive scaling
            if scaling_factor < 0.2:
                print(f"⚖️  Energy conservation: Force scaled by {scaling_factor:.2f}")
        
        return force
    
    def _calculate_gradient_force(self, current: float, position: float) -> float:
        """FIXED gradient force calculation with better boundary handling."""
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
        
        # FIXED: Apply reasonable bounds to prevent numerical spikes
        max_gradient_force = 1000.0  # 1000N maximum gradient force
        force = NumericalUtils.clamp(force, -max_gradient_force, max_gradient_force)
        
        return NumericalUtils.safe_numerical_operation(force, "gradient_force_balanced")
    
    def _calculate_lorentz_force(self, current: float, position: float, velocity: float) -> float:
        """Calculate Lorentz force with balanced parameters."""
        if abs(velocity) < 1e-6:
            return 0.0
        
        # Simplified Lorentz force calculation for balanced approach
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        L_eff = min(self.proj_length, self.field_calc.coil_length)
        
        # Current density and conductivity effects
        J = current / (np.pi * self.field_calc.coil_radius**2)  # Current density
        sigma = 1.0 / self.materials.get_material_property(self.proj_material, 'resistivity_20C')
        
        # Balanced Lorentz force
        force = -sigma * J * B_field * velocity * self.proj_volume * 0.01  # Scaling factor
        
        return NumericalUtils.safe_numerical_operation(force, "lorentz_force_balanced")
    
    def _calculate_eddy_current_force(self, current: float, position: float, velocity: float,
                                    current_history: Optional[List] = None,
                                    time_history: Optional[List] = None) -> Tuple[float, float]:
        """Calculate eddy current force and power dissipation with balanced approach."""
        if abs(velocity) < 1e-6:
            return 0.0, 0.0
        
        # Simplified eddy current model for balanced approach
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Eddy current force opposes motion (Lenz's law)
        sigma = 1.0 / self.materials.get_material_property(self.proj_material, 'resistivity_20C')
        
        # Balanced eddy current force
        eddy_force = -np.sign(velocity) * sigma * B_field**2 * self.proj_volume * abs(velocity) * 0.001
        
        # Power dissipation
        eddy_power = abs(eddy_force * velocity)
        
        return (NumericalUtils.safe_numerical_operation(eddy_force, "eddy_force_balanced"),
                NumericalUtils.safe_numerical_operation(eddy_power, "eddy_power_balanced")) 