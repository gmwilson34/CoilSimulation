"""
Base Electromagnetic Force Calculations

This module provides the foundational electromagnetic force calculation methods
shared across all force calculators.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import warnings
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils, SafetyLimits
from ..fields import MagneticFieldCalculator, AdvancedMagneticFieldCalculator
from ..materials import MaterialProperties, PermeabilityModel


class BaseElectromagneticForces(BasePhysicsModel):
    """
    Base class for electromagnetic force calculations.
    
    Provides common functionality for:
    - Basic force calculations (gradient, reluctance, motional EMF)
    - Projectile parameter handling
    - Safety limits and numerical validation
    - Energy conservation checks
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize base electromagnetic forces calculator."""
        super().__init__(config)
        self.field_calc = field_calculator
        self.materials = materials
        self.permeability_model = PermeabilityModel(materials)
        
        # Extract projectile parameters
        proj_cfg = config.get('projectile', {})
        self.proj_length = proj_cfg.get('length', 0.01)
        self.proj_diameter = proj_cfg.get('diameter', 0.008)
        self.proj_mass = proj_cfg.get('mass', 0.01)
        self.proj_material = proj_cfg.get('material', 'Low_Carbon_Steel')
        
        # Calculate projectile properties
        self.proj_radius = self.proj_diameter / 2.0
        self.proj_volume = np.pi * self.proj_radius**2 * self.proj_length
        self.proj_mu_r = self.materials.get_material_property(self.proj_material, 'mu_r')
        
        # Calculate demagnetization factors for cylindrical projectile
        self.N_z, self.N_r = self._calculate_demagnetization_factors()
        
        # Energy conservation parameters
        cap_cfg = config.get('capacitor', {})
        self.initial_energy = 0.5 * cap_cfg.get('capacitance', 0.02) * cap_cfg.get('initial_voltage', 600)**2
        
        # Basic force calculation parameters
        self.integration_order = config.get('physics', {}).get('integration_order', 8)
        self.adaptive_tolerance = config.get('physics', {}).get('adaptive_tolerance', 1e-12)
        
    def _calculate_demagnetization_factors(self) -> Tuple[float, float]:
        """
        Calculate demagnetization factors for cylindrical projectile.
        
        Returns:
            Tuple of (N_z, N_r) demagnetization factors
        """
        # Aspect ratio of cylinder
        aspect_ratio = self.proj_length / self.proj_diameter
        
        if aspect_ratio > 10:
            # Long cylinder (rod-like)
            N_z = 0.0  # Negligible demagnetization along axis
            N_r = 0.5  # Strong demagnetization radially
        elif aspect_ratio < 0.1:
            # Flat disc
            N_z = 1.0  # Strong demagnetization along axis
            N_r = 0.0  # Negligible demagnetization radially
        else:
            # General cylinder - use approximation formulas
            # For ellipsoid approximation
            a = self.proj_length / 2.0  # Semi-major axis
            b = self.proj_radius       # Semi-minor axis
            
            if a > b:  # Prolate ellipsoid
                e = np.sqrt(1 - (b/a)**2)  # Eccentricity
                N_z = (1.0 / e**2) * (1.0 / (2 * e) * np.log((1 + e) / (1 - e)) - 1.0)
                N_r = (1.0 - N_z) / 2.0
            else:  # Oblate ellipsoid
                e = np.sqrt(1 - (a/b)**2)
                N_z = (1.0 / e**2) * (np.arcsin(e) / e - np.sqrt(1 - e**2))
                N_r = (1.0 - N_z) / 2.0
        
        return N_z, N_r
    
    def calculate_gradient_force(self, current: float, position: float) -> float:
        """
        Calculate magnetic gradient force F = μ₀ * ∇(m · B).
        
        Properly accounts for demagnetization effects in finite geometry.
        """
        # Get magnetic field and gradient at projectile position
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        
        # Applied field intensity
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
        
        # Get material permeability
        mu_r = self.permeability_model.calculate_nonlinear_permeability(H_applied, self.proj_material)
        
        # CORRECTED: Effective permeability with demagnetization factor
        # For axial magnetization (field along cylinder axis)
        mu_eff = 1.0 + (mu_r - 1.0) / (1.0 + self.N_z * (mu_r - 1.0))
        
        # Magnetic moment: m = (μ_eff - 1) * V * H_applied
        magnetic_moment = (mu_eff - 1.0) * self.proj_volume * H_applied
        
        # CORRECTED: Gradient force F = μ₀ * m * dH/dz = m * dB/dz
        force = magnetic_moment * dB_dz
        
        return NumericalUtils.safe_numerical_operation(force, "gradient_force")
    
    def calculate_reluctance_force(self, current: float, position: float) -> float:
        """Calculate reluctance force F = 0.5 * I² * dL/dz."""
        # Get inductance and its gradient
        L = self._calculate_inductance_with_projectile(current, position)
        dL_dz = self._calculate_inductance_gradient(current, position)
        
        # Reluctance force: F = 0.5 * I² * dL/dz
        force = 0.5 * current**2 * dL_dz
        
        return NumericalUtils.safe_numerical_operation(force, "reluctance_force")
    
    def calculate_motional_emf_force(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate force due to motional EMF in moving conductor.
        
        This replaces the incorrect "Lorentz force" implementation.
        """
        if abs(velocity) < 1e-6:
            return 0.0
        
        # Get magnetic field at projectile position
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Motional EMF: ε = v × B · L for conductor moving through field
        # Simplified for axial motion through axial field
        motional_emf = velocity * B_field * self.proj_length
        
        # Induced current in projectile (simplified resistive model)
        # Resistivity and geometry determine eddy current magnitude
        proj_resistivity = self.materials.get_material_property(self.proj_material, 'resistivity_20C')
        characteristic_resistance = proj_resistivity * self.proj_length / (np.pi * self.proj_radius**2)
        
        induced_current = motional_emf / characteristic_resistance if characteristic_resistance > 0 else 0
        
        # Force on induced current: F = I_induced * L_eff * B
        # Opposes motion (Lenz's law)
        effective_length = self.proj_length
        force = -induced_current * effective_length * B_field
        
        return NumericalUtils.safe_numerical_operation(force, "motional_emf_force")
    
    def calculate_lorentz_force(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate Lorentz force (alias for motional EMF force).
        
        This method provides backward compatibility for code expecting a calculate_lorentz_force method.
        """
        return self.calculate_motional_emf_force(current, position, velocity)
    
    def _calculate_inductance_with_projectile(self, current: float, position: float) -> float:
        """Calculate coil inductance with ferromagnetic projectile."""
        # Base air-core inductance
        L_air = self._solenoid_inductance_air_core()
        
        # Enhancement due to ferromagnetic core
        if abs(position) < self.field_calc.coil_length:
            # Projectile is inside or partially inside coil
            overlap_fraction = self._calculate_overlap_fraction(position)
            
            # Get effective permeability with demagnetization
            B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
            H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
            mu_r = self.permeability_model.calculate_nonlinear_permeability(H_applied, self.proj_material)
            
            # Effective permeability with demagnetization
            mu_eff = 1.0 + (mu_r - 1.0) / (1.0 + self.N_z * (mu_r - 1.0))
            
            # Inductance enhancement
            fill_factor = self.proj_volume / self._coil_volume()
            L_enhancement = L_air * (mu_eff - 1.0) * overlap_fraction * fill_factor
            
            L_total = L_air + L_enhancement
        else:
            L_total = L_air
        
        return max(L_total, SafetyLimits.MIN_INDUCTANCE)
    
    def _calculate_inductance_gradient(self, current: float, position: float, delta: float = 1e-4) -> float:
        """Calculate inductance gradient dL/dz using finite differences."""
        delta = 1e-6 * self.field_calc.coil_length
        L_plus = self._calculate_inductance_with_projectile(current, position + delta)
        L_minus = self._calculate_inductance_with_projectile(current, position - delta)
        
        dL_dz = (L_plus - L_minus) / (2.0 * delta)
        
        return NumericalUtils.safe_numerical_operation(dL_dz, "inductance_gradient")
    
    def _solenoid_inductance_air_core(self) -> float:
        """
        Calculate air-core solenoid inductance using Nagaoka correction.
        L = μ₀ N² π r² / l * k, with k approximation.
        """
        N = self.field_calc.num_turns
        l = self.field_calc.coil_length
        r = self.field_calc.coil_radius
        if l <= 0:
            return 0.0
        alpha = (2 * r) / l  # d/l
        k = 1 / (1 + 0.45 * alpha + 0.005 * alpha**2)
        L_infinite = PhysicsConstants.MU_0 * N**2 * np.pi * r**2 / l
        return k * L_infinite
    
    def _calculate_overlap_fraction(self, position: float) -> float:
        """Calculate fraction of projectile overlapping with coil."""
        coil_start = -self.field_calc.coil_length / 2
        coil_end = self.field_calc.coil_length / 2
        proj_start = position - self.proj_length / 2
        proj_end = position + self.proj_length / 2
        
        # Calculate overlap
        overlap_start = max(coil_start, proj_start)
        overlap_end = min(coil_end, proj_end)
        overlap_length = max(0, overlap_end - overlap_start)
        
        return overlap_length / self.proj_length
    
    def _coil_volume(self) -> float:
        """Calculate effective coil volume."""
        return np.pi * self.field_calc.coil_radius**2 * self.field_calc.coil_length
    
    def apply_safety_limits(self, force: float) -> float:
        """Apply safety limits to calculated force."""
        return NumericalUtils.clamp(force, -SafetyLimits.MAX_FORCE, SafetyLimits.MAX_FORCE)
    
    def validate_energy_conservation(self, force: float, current: float, position: float, 
                                   tolerance: float = 0.1) -> bool:
        """Basic energy conservation validation."""
        # Calculate work that could be done by this force
        characteristic_distance = self.field_calc.coil_length  # Dynamic based on coil size
        work_estimate = abs(force) * characteristic_distance
        
        # Check against initial energy
        return work_estimate <= self.initial_energy * (1.0 + tolerance)
    
    def magnetic_force_ferromagnetic(self, current: float, position: float, 
                                   velocity: float = 0.0, 
                                   current_history: Optional[List] = None,
                                   time_history: Optional[List] = None) -> Tuple[float, float]:
        """
        Calculate total magnetic force on ferromagnetic projectile.
        
        Args:
            current: Coil current (A)
            position: Projectile position (m)
            velocity: Projectile velocity (m/s)
            current_history: Historical current values (optional)
            time_history: Historical time values (optional)
            
        Returns:
            Tuple of (total_force, eddy_power_loss)
        """
        # Calculate individual force components
        # force_gradient = self.calculate_gradient_force(current, position)  # Commented to avoid double counting with reluctance
        force_gradient = 0.0
        force_reluctance = self.calculate_reluctance_force(current, position)
        force_motional = self.calculate_motional_emf_force(current, position, velocity)
        
        # Total magnetic force
        total_force = force_gradient + force_reluctance + force_motional
        
        # Apply safety limits
        total_force = self.apply_safety_limits(total_force)
        
        # Estimate eddy current power loss (simplified model)
        eddy_power = 0.0
        if abs(velocity) > 1e-6:
            B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
            # Simplified eddy current power: P ∝ B² * v² * σ * V
            proj_resistivity = self.materials.get_material_property(self.proj_material, 'resistivity_20C')
            conductivity = 1.0 / proj_resistivity if proj_resistivity > 0 else 0
            eddy_power = abs(0.5 * B_field**2 * velocity**2 * conductivity * self.proj_volume)
        
        return total_force, eddy_power 