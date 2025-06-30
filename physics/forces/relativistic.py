"""
Relativistic Force Corrections

This module implements relativistic corrections for high-velocity electromagnetic forces.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils
from .base import BaseElectromagneticForces


class RelativisticForces(BaseElectromagneticForces):
    """
    Relativistic corrections for electromagnetic forces at high velocities.
    
    Includes:
    - Lorentz transformation effects
    - Relativistic mass increase
    - Time dilation effects on induction
    - Special relativistic field transformations
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize relativistic forces calculator."""
        super().__init__(config, field_calculator, materials)
        
        # Relativistic parameters
        self.include_relativistic = config.get('advanced_physics', {}).get('include_relativistic', True)
        self.relativistic_threshold = config.get('simulation', {}).get('relativistic_threshold', 0.0001 * PhysicsConstants.C)
        self.max_velocity = config.get('simulation', {}).get('max_velocity', 15000)
        
        print(f"🚀 Relativistic forces initialized")
        print(f"   - Threshold velocity: {self.relativistic_threshold:.0f} m/s ({self.relativistic_threshold/PhysicsConstants.C*100:.4f}% c)")
        print(f"   - Maximum velocity: {self.max_velocity:.0f} m/s ({self.max_velocity/PhysicsConstants.C*100:.4f}% c)")
    
    def calculate_relativistic_corrections(self, current: float, position: float, 
                                         velocity: float, acceleration: float) -> Tuple[float, dict]:
        """
        Calculate relativistic force corrections.
        
        Returns:
            Tuple of (correction_force, correction_breakdown)
        """
        if not self.include_relativistic or abs(velocity) < self.relativistic_threshold:
            return 0.0, {}
        
        correction_breakdown = {}
        total_correction = 0.0
        
        # Lorentz factor
        gamma = self._calculate_lorentz_factor(velocity)
        correction_breakdown['lorentz_factor'] = gamma
        
        # Mass increase correction
        mass_correction = self._calculate_relativistic_mass_correction(velocity, acceleration)
        correction_breakdown['mass_correction'] = mass_correction
        total_correction += mass_correction
        
        # Field transformation correction
        field_correction = self._calculate_field_transformation_correction(current, position, velocity)
        correction_breakdown['field_transformation'] = field_correction
        total_correction += field_correction
        
        # Time dilation effect on inductance
        time_dilation_correction = self._calculate_time_dilation_correction(current, position, velocity)
        correction_breakdown['time_dilation'] = time_dilation_correction
        total_correction += time_dilation_correction
        
        # Relativistic energy-momentum correction
        energy_momentum_correction = self._calculate_energy_momentum_correction(velocity, acceleration)
        correction_breakdown['energy_momentum'] = energy_momentum_correction
        total_correction += energy_momentum_correction
        
        return total_correction, correction_breakdown
    
    def _calculate_lorentz_factor(self, velocity: float) -> float:
        """Calculate Lorentz factor γ = 1/√(1 - v²/c²)."""
        beta_squared = (velocity / PhysicsConstants.C)**2
        
        # Avoid division by zero and ensure physical values
        if beta_squared >= 1.0:
            beta_squared = 0.9999999  # Cap at just below light speed
        
        gamma = 1.0 / np.sqrt(1.0 - beta_squared)
        return gamma
    
    def _calculate_relativistic_mass_correction(self, velocity: float, acceleration: float) -> float:
        """
        Calculate force correction due to relativistic mass increase.
        
        F_rel = γ³ma (longitudinal) for acceleration parallel to velocity
        """
        if abs(acceleration) < 1e-6:
            return 0.0
        
        gamma = self._calculate_lorentz_factor(velocity)
        
        # Relativistic mass correction factor
        # For motion along the axis (longitudinal), γ³ factor applies
        relativistic_factor = gamma**3
        
        # Classical force would be F = ma
        classical_force = self.proj_mass * acceleration
        
        # Relativistic correction
        relativistic_correction = classical_force * (relativistic_factor - 1.0)
        
        return NumericalUtils.safe_numerical_operation(relativistic_correction, "relativistic_mass_correction")
    
    def _calculate_field_transformation_correction(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate correction due to relativistic field transformations.
        
        Moving charges see transformed E and B fields.
        """
        # Get electromagnetic fields in lab frame
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        E_field = 0.0  # Assuming no significant electric field
        
        gamma = self._calculate_lorentz_factor(velocity)
        beta = velocity / PhysicsConstants.C
        
        # Field transformations (simplified for axial motion)
        # B'_perp = γ(B_perp - βE_perp/c)
        # E'_perp = γ(E_perp + βB_perp*c)
        
        # For our geometry, mainly B_z component
        B_transformed = B_field  # B_parallel unchanged
        
        # Induced electric field due to motion
        E_induced = beta * B_field * PhysicsConstants.C
        
        # Force correction from transformed fields
        # This affects the magnetic dipole interaction
        magnetic_moment = self._calculate_magnetic_moment(current, position)
        
        # Field gradient (needed for force calculation)
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        
        # Correction to magnetic force
        field_correction = PhysicsConstants.MU_0 * magnetic_moment * dB_dz * (gamma - 1.0)
        
        return NumericalUtils.safe_numerical_operation(field_correction, "field_transformation_correction")
    
    def _calculate_magnetic_moment(self, current: float, position: float) -> float:
        """Calculate magnetic dipole moment of projectile."""
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
        
        # Effective permeability
        mu_eff = self.permeability_model.calculate_nonlinear_permeability(H_applied, self.proj_material)
        
        # Magnetic moment: m = (μ_r - 1) * V * H
        magnetic_moment = (mu_eff - 1.0) * self.proj_volume * H_applied
        
        return magnetic_moment
    
    def _calculate_time_dilation_correction(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate correction due to time dilation effects on induction.
        
        Moving clocks run slow, affecting the rate of change of magnetic flux.
        """
        gamma = self._calculate_lorentz_factor(velocity)
        
        # Time dilation factor affects dΦ/dt calculations
        # Proper time τ = t/γ, so dΦ/dτ = γ * dΦ/dt
        
        # Get inductance and its time derivative (through current change)
        L = self._calculate_inductance_with_projectile(current, position)
        
        # Simplified: assume current changes contribute to time-dependent flux
        # This would require actual current derivative information
        # For now, use position-dependent inductance change rate
        dL_dt = self._calculate_inductance_time_derivative(current, position, velocity)
        
        # Time dilation correction to induced EMF
        emf_correction = L * (gamma - 1.0) * dL_dt
        
        # Force from corrected EMF (simplified)
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        force_correction = emf_correction * B_field / (velocity if abs(velocity) > 1e-6 else 1e-6)
        
        return NumericalUtils.safe_numerical_operation(force_correction, "time_dilation_correction")
    
    def _calculate_inductance_time_derivative(self, current: float, position: float, velocity: float) -> float:
        """Calculate time derivative of inductance due to motion."""
        # dL/dt = (dL/dz) * (dz/dt) = (dL/dz) * v
        dL_dz = self._calculate_inductance_gradient(current, position)
        dL_dt = dL_dz * velocity
        
        return dL_dt
    
    def _calculate_energy_momentum_correction(self, velocity: float, acceleration: float) -> float:
        """
        Calculate correction from relativistic energy-momentum relation.
        
        E² = (pc)² + (mc²)²
        """
        if abs(acceleration) < 1e-6:
            return 0.0
        
        gamma = self._calculate_lorentz_factor(velocity)
        
        # Relativistic momentum
        p_rel = gamma * self.proj_mass * velocity
        
        # Energy-momentum relation correction
        # dp/dt = γ³ma for longitudinal acceleration
        dp_dt_rel = gamma**3 * self.proj_mass * acceleration
        dp_dt_classical = self.proj_mass * acceleration
        
        # Force correction
        force_correction = dp_dt_rel - dp_dt_classical
        
        return NumericalUtils.safe_numerical_operation(force_correction, "energy_momentum_correction")
    
    def calculate_relativistic_kinetic_energy(self, velocity: float) -> float:
        """Calculate relativistic kinetic energy."""
        gamma = self._calculate_lorentz_factor(velocity)
        
        # Relativistic kinetic energy: KE = (γ - 1)mc²
        rest_energy = self.proj_mass * PhysicsConstants.C**2
        kinetic_energy = (gamma - 1.0) * rest_energy
        
        return kinetic_energy
    
    def calculate_relativistic_momentum(self, velocity: float) -> float:
        """Calculate relativistic momentum."""
        gamma = self._calculate_lorentz_factor(velocity)
        return gamma * self.proj_mass * velocity
    
    def is_relativistic_regime(self, velocity: float) -> bool:
        """Check if velocity requires relativistic treatment."""
        return abs(velocity) >= self.relativistic_threshold
    
    def get_relativistic_factors(self, velocity: float) -> dict:
        """Get all relativistic correction factors."""
        gamma = self._calculate_lorentz_factor(velocity)
        beta = velocity / PhysicsConstants.C
        
        return {
            'gamma': gamma,
            'beta': beta,
            'velocity_fraction_c': beta,
            'mass_factor': gamma,
            'length_contraction': 1.0 / gamma,
            'time_dilation': gamma
        } 