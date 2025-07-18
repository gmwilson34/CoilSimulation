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
    
    Physically negligible for typical coilgun velocities (v << c), but implemented
    for completeness and high-velocity edge cases.
    
    Key Improvements:
    - Avoids circular dependency in mass corrections by providing effective mass factors
    - Proper frame transformations between lab and projectile reference frames
    - First-order relativistic corrections for field transformations
    - Threshold-based activation to avoid unnecessary computation
    
    Includes:
    - Lorentz transformation effects on effective mass (γ³ factor for longitudinal motion)
    - Reference frame considerations (lab vs. projectile frame)
    - Field transformation corrections (E and B field mixing due to motion)
    - Time dilation effects on electromagnetic induction (when significant)
    
    Usage:
    - Use get_effective_mass_factor() to get mass correction factors
    - Use apply_relativistic_corrections_to_force() for proper force corrections
    - Corrections only activate above relativistic_threshold velocity
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
                                         velocity: float, acceleration: Optional[float] = None) -> Tuple[float, dict]:
        """
        Calculate relativistic force corrections.
        
        Returns:
            Tuple of (correction_force, correction_breakdown)
            
        Note: For mass correction, returns effective mass factor instead of force correction
        to avoid circular dependency with acceleration.
        """
        if not self.include_relativistic or abs(velocity) < self.relativistic_threshold:
            return 0.0, {}
        
        correction_breakdown = {}
        total_correction = 0.0
        
        # Lorentz factor
        gamma = self._calculate_lorentz_factor(velocity)
        correction_breakdown['lorentz_factor'] = gamma
        correction_breakdown['velocity_regime'] = f"v = {abs(velocity):.1f} m/s ({abs(velocity)/PhysicsConstants.C*100:.6f}% c)"
        
        # Effective mass factor (not a force correction)
        mass_factor = self._calculate_relativistic_mass_correction(velocity, 0.0)  # Pass dummy acceleration
        correction_breakdown['mass_factor_correction'] = mass_factor
        correction_breakdown['note_mass'] = "Mass factor should multiply forces, not add to them"
        
        # Field transformation correction (frame effects)
        field_correction = self.calculate_frame_dependent_force_correction(current, position, velocity)
        correction_breakdown['field_transformation'] = field_correction
        total_correction += field_correction
        
        # Time dilation effect on inductance (if significant)
        if gamma > 1.001:  # Only for non-negligible time dilation
            time_dilation_correction = self._calculate_time_dilation_correction(current, position, velocity)
            correction_breakdown['time_dilation'] = time_dilation_correction
            total_correction += time_dilation_correction
        else:
            correction_breakdown['time_dilation'] = 0.0
        
        # Energy-momentum consistency check (if acceleration provided)
        if acceleration is not None:
            energy_momentum_correction = self._calculate_energy_momentum_correction(velocity, acceleration)
            correction_breakdown['energy_momentum'] = energy_momentum_correction
            total_correction += energy_momentum_correction
        else:
            correction_breakdown['energy_momentum'] = 0.0
            correction_breakdown['note_momentum'] = "No acceleration provided for momentum correction"
        
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
        Calculate effective mass factor for relativistic motion.
        
        Instead of force correction, returns the factor by which the effective mass
        changes, avoiding circular dependency with acceleration.
        
        For longitudinal motion: m_eff = γ³ m₀
        Returns: (γ³ - 1) correction factor to multiply with any force calculation
        """
        if abs(velocity) < self.relativistic_threshold:
            return 0.0
        
        gamma = self._calculate_lorentz_factor(velocity)
        
        # For longitudinal acceleration parallel to velocity
        # Effective mass factor is γ³
        # Return correction factor (γ³ - 1) to be applied to forces
        mass_factor_correction = gamma**3 - 1.0
        
        return NumericalUtils.safe_numerical_operation(mass_factor_correction, "relativistic_mass_factor")
    
    def _calculate_field_transformation_correction(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate correction due to relativistic field transformations between lab and projectile frames.
        
        For v << c, this is a small correction to the magnetic force due to 
        frame-dependent field mixing.
        """
        if abs(velocity) < self.relativistic_threshold:
            return 0.0
        
        # Get electromagnetic fields in lab frame
        B_lab = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        E_lab = 0.0  # Negligible electric field in lab frame
        
        gamma = self._calculate_lorentz_factor(velocity)
        beta = velocity / PhysicsConstants.C
        
        # For low velocities (v << c), field transformations give:
        # In projectile frame: E'_⊥ ≈ v × B (to first order in β)
        # This creates an effective electric field experienced by the projectile
        
        # Motional electric field in projectile frame
        E_motional = velocity * B_lab
        
        # The projectile experiences this as an additional force contribution
        # For a magnetic dipole: F = ∇(μ·B), with small correction from E field
        
        # Magnetic moment of projectile
        magnetic_moment = self._calculate_magnetic_moment(current, position)
        
        # First-order relativistic correction to force
        # This accounts for the fact that field transformations slightly modify
        # the effective field gradient seen by the projectile
        
        # Field gradient for force calculation
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        
        # Frame transformation correction (first order in β)
        # The correction scales with β² for small velocities
        beta_squared = beta**2
        frame_correction = 0.5 * beta_squared * magnetic_moment * dB_dz
        
        return NumericalUtils.safe_numerical_operation(frame_correction, "field_transformation_correction")
    
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
        Calculate correction due to time dilation effects on electromagnetic induction.
        
        Time dilation affects the rate of change of magnetic flux as seen in different frames.
        For proper time τ in projectile frame: τ = t/γ
        This affects EMF calculations: ε = -dΦ/dτ = -γ(dΦ/dt)
        
        However, this effect is typically negligible for v << c.
        """
        gamma = self._calculate_lorentz_factor(velocity)
        
        # Only calculate if time dilation is significant
        if gamma < 1.001:  # Less than 0.1% time dilation
            return 0.0
        
        # Get magnetic flux and its rate of change
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Flux through projectile cross-section
        flux = B_field * self.proj_volume / self.proj_length  # Effective area
        
        # Time derivative of flux due to motion through field gradient
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        dflux_dt = dB_dz * velocity * (self.proj_volume / self.proj_length)
        
        # Time dilation correction to EMF
        # EMF in projectile frame differs by factor of γ
        emf_correction = dflux_dt * (gamma - 1.0)
        
        # Convert EMF correction to force (simplified)
        # F ≈ I * dB/dz, where I relates to EMF
        if abs(velocity) > 1e-6 and abs(B_field) > 1e-9:
            equivalent_current = emf_correction / (velocity * B_field)
            force_correction = equivalent_current * dB_dz
        else:
            force_correction = 0.0
        
        return NumericalUtils.safe_numerical_operation(force_correction, "time_dilation_correction")
    
    def _calculate_inductance_time_derivative(self, current: float, position: float, velocity: float) -> float:
        """Calculate time derivative of inductance due to motion."""
        # dL/dt = (dL/dz) * (dz/dt) = (dL/dz) * v
        dL_dz = self._calculate_inductance_gradient(current, position)
        dL_dt = dL_dz * velocity
        
        return dL_dt
    
    def _calculate_energy_momentum_correction(self, velocity: float, acceleration: Optional[float]) -> float:
        """
        Calculate correction from relativistic energy-momentum relation.
        
        Ensures consistency with E² = (pc)² + (mc²)² and relativistic force law.
        This is mainly a consistency check rather than a primary correction.
        """
        if acceleration is None or abs(acceleration) < 1e-6:
            return 0.0
        
        gamma = self._calculate_lorentz_factor(velocity)
        
        # For longitudinal motion, the force is F = dp/dt = γ³ma
        # This should be consistent with energy-momentum relation
        
        # Classical momentum and force
        p_classical = self.proj_mass * velocity
        F_classical = self.proj_mass * acceleration
        
        # Relativistic momentum and force
        p_relativistic = gamma * self.proj_mass * velocity
        F_relativistic = gamma**3 * self.proj_mass * acceleration
        
        # The correction represents the difference
        force_correction = F_relativistic - F_classical
        
        # However, this creates circular dependency, so return as factor
        # This should match the mass correction already calculated
        return 0.0  # Already handled by mass factor correction
    
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
    
    def get_effective_mass_factor(self, velocity: float) -> float:
        """
        Get the relativistic mass factor for use in force calculations.
        
        This should be used to multiply classical forces by (1 + factor)
        where factor is the return value.
        
        Returns:
            Mass factor correction: (γ³ - 1) for longitudinal motion
        """
        if not self.include_relativistic or abs(velocity) < self.relativistic_threshold:
            return 0.0
            
        gamma = self._calculate_lorentz_factor(velocity)
        return gamma**3 - 1.0
    
    def transform_fields_to_projectile_frame(self, B_lab: float, E_lab: float, velocity: float) -> Tuple[float, float]:
        """
        Transform electromagnetic fields from lab frame to projectile frame.
        
        For low velocities (v << c), use first-order approximations:
        B'_∥ = B_∥ (parallel component unchanged)
        E'_⊥ ≈ E_⊥ + v × B (motional electric field)
        
        Args:
            B_lab: Magnetic field in lab frame (Tesla)
            E_lab: Electric field in lab frame (V/m)
            velocity: Projectile velocity (m/s)
            
        Returns:
            Tuple of (B_projectile, E_projectile)
        """
        if abs(velocity) < self.relativistic_threshold:
            return B_lab, E_lab
        
        gamma = self._calculate_lorentz_factor(velocity)
        beta = velocity / PhysicsConstants.C
        
        # For axial motion, B_parallel unchanged to first order
        B_projectile = B_lab
        
        # Motional electric field in projectile frame
        E_motional = velocity * B_lab
        E_projectile = E_lab + E_motional
        
        # For higher precision (if needed), include γ corrections
        if gamma > 1.01:  # For significant relativistic effects
            B_projectile *= gamma  # More precise transformation
            E_projectile *= gamma
        
        return B_projectile, E_projectile
    
    def calculate_frame_dependent_force_correction(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate force corrections due to reference frame differences.
        
        This accounts for the fact that forces transform differently between
        reference frames in special relativity.
        """
        if abs(velocity) < self.relativistic_threshold:
            return 0.0
        
        # Get fields in lab frame
        B_lab = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        E_lab = 0.0
        
        # Transform to projectile frame
        B_proj, E_proj = self.transform_fields_to_projectile_frame(B_lab, E_lab, velocity)
        
        # Calculate magnetic moment and gradient
        magnetic_moment = self._calculate_magnetic_moment(current, position)
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        
        # Frame-dependent force correction
        beta = velocity / PhysicsConstants.C
        
        # First-order correction scales with β²
        frame_correction = 0.5 * beta**2 * magnetic_moment * dB_dz
        
        return frame_correction
    
    def apply_relativistic_corrections_to_force(self, classical_force: float, velocity: float) -> float:
        """
        Apply relativistic corrections to a classical force calculation.
        
        This is the proper way to use relativistic corrections in the solver,
        avoiding circular dependency issues.
        
        Args:
            classical_force: The force calculated using classical physics (N)
            velocity: Current velocity of the projectile (m/s)
            
        Returns:
            Corrected force including relativistic effects (N)
        """
        if not self.include_relativistic or abs(velocity) < self.relativistic_threshold:
            return classical_force
        
        # Apply effective mass factor to the force
        mass_factor = self.get_effective_mass_factor(velocity)
        corrected_force = classical_force * (1.0 + mass_factor)
        
        return corrected_force
    
    def get_relativistic_diagnostics(self, velocity: float) -> dict:
        """
        Get diagnostic information about relativistic effects.
        
        Useful for understanding when relativistic corrections become significant.
        """
        gamma = self._calculate_lorentz_factor(velocity)
        beta = velocity / PhysicsConstants.C
        
        return {
            'velocity_ms': velocity,
            'velocity_fraction_c': beta,
            'velocity_percent_c': beta * 100,
            'lorentz_factor': gamma,
            'relativistic_regime': abs(velocity) >= self.relativistic_threshold,
            'mass_increase_factor': gamma,
            'mass_increase_percent': (gamma - 1) * 100,
            'time_dilation_factor': gamma,
            'length_contraction_factor': 1.0 / gamma,
            'effective_mass_correction': gamma**3 - 1.0,
            'threshold_velocity': self.relativistic_threshold,
            'threshold_percent_c': self.relativistic_threshold / PhysicsConstants.C * 100
        }
    
    @staticmethod
    def estimate_relativistic_significance(velocity: float) -> dict:
        """
        Estimate the significance of relativistic effects for a given velocity.
        
        Provides guidance on when relativistic corrections matter.
        
        Args:
            velocity: Velocity to analyze (m/s)
            
        Returns:
            Dictionary with significance analysis
        """
        beta = abs(velocity) / PhysicsConstants.C
        gamma = 1.0 / np.sqrt(1.0 - min(beta**2, 0.9999999))
        
        # Significance levels
        if beta < 0.0001:  # < 0.01% c
            significance = "negligible"
            description = "Classical mechanics fully adequate"
        elif beta < 0.001:  # < 0.1% c
            significance = "very_small"
            description = "Relativistic effects < 0.05%"
        elif beta < 0.01:   # < 1% c
            significance = "small"
            description = "Relativistic effects < 5%"
        elif beta < 0.1:    # < 10% c
            significance = "moderate"
            description = "Relativistic effects become noticeable"
        elif beta < 0.5:    # < 50% c
            significance = "large"
            description = "Significant relativistic effects"
        else:              # > 50% c
            significance = "extreme"
            description = "Highly relativistic regime"
        
        return {
            'velocity_ms': velocity,
            'velocity_fraction_c': beta,
            'velocity_percent_c': beta * 100,
            'lorentz_factor': gamma,
            'mass_increase_percent': (gamma - 1) * 100,
            'significance_level': significance,
            'description': description,
            'mass_correction_factor': gamma**3 - 1,
            'recommendation': "Use relativistic corrections" if beta > 0.001 else "Classical mechanics sufficient"
        }