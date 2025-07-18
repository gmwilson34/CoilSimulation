"""
Quantum Force Calculations

This module implements quantum mechanical force corrections including
Casimir forces, zero-point energy effects, and quantum tunneling.

WARNING: These effects are negligible for macroscopic coilgun simulations
and are disabled by default. Enable only for nanoscale simulations where
gap sizes and projectile dimensions are < 100 nm.

For typical coilgun parameters:
- Casimir forces: ~pN for mm gaps (negligible)
- Tunneling probability: ~0 for macroscopic objects  
- Formulas are approximate and not experimentally validated for these applications
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils
from .base import BaseElectromagneticForces


class QuantumForceCalculator(BaseElectromagneticForces):
    """
    Quantum mechanical force corrections for electromagnetic forces.
    
    Includes:
    - Casimir effect for conductor-conductor interactions
    - Zero-point energy contributions to magnetic forces
    - Quantum tunneling probabilities for magnetic barriers
    - Quantum vacuum fluctuation force corrections
    - de Broglie wavelength considerations
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize quantum force calculator."""
        super().__init__(config, field_calculator, materials)
        
        # QUANTUM PHYSICS PARAMETERS - DISABLED BY DEFAULT
        # Quantum effects are negligible for macroscopic coilgun simulations
        # Only enable for nanoscale simulations where gap sizes < 100 nm
        self.include_quantum_effects = config.get('quantum_physics', {}).get('enable_quantum_forces', False)
        self.casimir_force_enabled = config.get('quantum_physics', {}).get('casimir_force', False)
        self.quantum_tunneling_effects = config.get('quantum_physics', {}).get('quantum_tunneling', False)
        self.zero_point_energy_effects = config.get('quantum_physics', {}).get('zero_point_energy', False)
        
        # Scale checking for quantum effects applicability
        self.nanoscale_threshold = 100e-9  # 100 nm - below this, quantum effects may be relevant
        self.is_nanoscale_simulation = self._check_nanoscale_applicability()
        
        if self.include_quantum_effects:
            if not self.is_nanoscale_simulation:
                print("⚠️  WARNING: Quantum effects enabled for macroscopic simulation!")
                print("   Quantum forces are negligible for mm-scale gaps and macro projectiles.")
                print("   Consider disabling quantum effects for better performance.")
            self._initialize_quantum_force_models()
        
        print(f"🔬 Quantum force calculator initialized")
        print(f"   - Quantum effects: {'✓' if self.include_quantum_effects else '✗ (disabled by default)'}")
        print(f"   - Nanoscale regime: {'✓' if self.is_nanoscale_simulation else '✗'}")
        if self.include_quantum_effects:
            print(f"   - Casimir forces: {'✓' if self.casimir_force_enabled else '✗'}")
            print(f"   - Zero-point energy: {'✓' if self.zero_point_energy_effects else '✗'}")
            print(f"   - Quantum tunneling: {'✓' if self.quantum_tunneling_effects else '✗'}")
    
    def _check_nanoscale_applicability(self) -> bool:
        """
        Check if the simulation is in the nanoscale regime where quantum effects matter.
        
        Returns:
            True if gaps and dimensions are small enough for quantum effects
        """
        # Check typical gap size (coil-projectile separation)
        typical_gap = abs(self.field_calc.coil_radius - self.proj_radius)
        
        # Check projectile dimensions
        projectile_size = min(self.proj_radius, self.proj_length)
        
        # Quantum effects only relevant for nanoscale dimensions
        is_nanoscale = (typical_gap < self.nanoscale_threshold or 
                       projectile_size < self.nanoscale_threshold)
        
        return is_nanoscale
    
    def _initialize_quantum_force_models(self):
        """Initialize quantum mechanical force corrections."""
        # Casimir effect parameters for conductor-conductor interactions
        self.casimir_coefficient = (np.pi**2 / 240) * (PhysicsConstants.C * 1.054571817e-34)  # ħc factor
        
        # Zero-point energy contributions to magnetic forces
        self.zero_point_cutoff_frequency = 1e15  # Hz (typical cutoff)
        
        # Quantum tunneling probabilities for magnetic barriers
        self.tunneling_barrier_height = 1e-19  # J (typical magnetic barrier)
        
        # Quantum vacuum fluctuation force corrections
        self.vacuum_fluctuation_amplitude = 1e-20  # N (theoretical estimate)
        
        # de Broglie wavelength considerations for massive projectiles
        self.quantum_mechanical_threshold = 1e-15  # m (de Broglie wavelength scale)
        
        print(f"🔬 Quantum force models initialized:")
        print(f"   - Casimir coefficient: {self.casimir_coefficient:.2e}")
        print(f"   - Zero-point cutoff: {self.zero_point_cutoff_frequency:.0e} Hz")
        print(f"   - Quantum threshold: {self.quantum_mechanical_threshold:.0e} m")
    
    def calculate_quantum_force_corrections(self, current: float, position: float, 
                                         velocity: float = 0.0) -> Tuple[float, dict]:
        """
        Calculate total quantum force corrections.
        
        WARNING: These calculations use approximate formulas and are only 
        meaningful for nanoscale simulations. For typical coilgun parameters:
        - Casimir forces: ~pN for mm gaps (negligible)
        - Tunneling: ~0 for macroscopic objects
        - Zero-point energy: Highly approximate estimates
        
        Returns:
            Tuple of (total_quantum_force, force_breakdown)
        """
        if not self.include_quantum_effects:
            return 0.0, {}
        
        # Early return for macroscopic simulations
        if not self.is_nanoscale_simulation:
            return 0.0, {'warning': 'quantum_effects_disabled_for_macroscopic_scale'}
        
        force_breakdown = {}
        total_quantum_force = 0.0
        
        # Casimir force
        if self.casimir_force_enabled:
            casimir_force = self._calculate_casimir_force(position, velocity)
            force_breakdown['casimir'] = casimir_force
            total_quantum_force += casimir_force
        
        # Zero-point energy force
        if self.zero_point_energy_effects:
            zpe_force = self._calculate_zero_point_energy_force(current, position)
            force_breakdown['zero_point_energy'] = zpe_force
            total_quantum_force += zpe_force
        
        # Quantum tunneling force
        if self.quantum_tunneling_effects:
            tunneling_force = self._calculate_quantum_tunneling_force(current, position, velocity)
            force_breakdown['quantum_tunneling'] = tunneling_force
            total_quantum_force += tunneling_force
        
        # Quantum vacuum stress
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        position_array = np.array([0, 0, position])
        B_field_array = np.array([0, 0, B_field])
        vacuum_stress_force = self._calculate_quantum_vacuum_stress(B_field_array, position_array)
        force_breakdown['vacuum_stress'] = vacuum_stress_force
        total_quantum_force += vacuum_stress_force
        
        # Add magnitude check - warn if forces are negligible
        if abs(total_quantum_force) < 1e-12:  # 1 pN threshold
            force_breakdown['magnitude_warning'] = 'quantum_forces_negligible'
        
        return total_quantum_force, force_breakdown
    
    def _calculate_casimir_force(self, position: float, velocity: float) -> float:
        """
        Calculate Casimir force between conductor surfaces.
        
        F_casimir = -π²ħc/(240 * d⁴) * A
        where d is the separation distance and A is the area.
        
        WARNING: This formula is approximate and only valid for:
        - Perfect conductor surfaces
        - Gap sizes much smaller than characteristic system size
        - Non-relativistic velocities
        
        For mm-scale gaps typical in coilguns, forces are ~pN (negligible).
        """
        # Distance from projectile to coil inner surface
        distance = abs(self.field_calc.coil_radius - self.proj_radius)
        
        # Minimum distance for validity (avoid singularities)
        min_distance = 1e-9  # 1 nm
        distance = max(distance, min_distance)
        
        # Check if gap is too large for meaningful Casimir force
        if distance > 1e-6:  # 1 μm
            # For gaps > 1 μm, Casimir force is negligible
            return 0.0
        
        # Effective interaction area (simplified cylindrical approximation)
        interaction_area = 2 * np.pi * self.proj_radius * self.proj_length
        
        # Casimir force (attractive, hence negative)
        # Note: This is the idealized formula for perfect conductors
        casimir_force = -self.casimir_coefficient * interaction_area / (distance**4)
        
        # Remove ad-hoc velocity corrections - not physically justified
        # for the simple parallel plate/cylinder approximation used here
        
        return NumericalUtils.safe_numerical_operation(casimir_force, "casimir_force")
    
    def _calculate_zero_point_energy_force(self, current: float, position: float) -> float:
        """
        Calculate force corrections due to zero-point energy fluctuations.
        
        WARNING: This is a highly approximate calculation. Zero-point energy 
        contributions to macroscopic forces are not well established and 
        depend heavily on cutoff frequency assumptions.
        
        Zero-point energy affects the electromagnetic field energy density,
        leading to additional force contributions.
        """
        # For macroscopic systems, zero-point corrections are negligible
        if not self.is_nanoscale_simulation:
            return 0.0
        
        # Magnetic field energy density
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        magnetic_energy_density = B_field**2 / (2 * PhysicsConstants.MU_0)
        
        # Zero-point energy density (cutoff at characteristic frequency)
        # NOTE: Cutoff frequency is arbitrary and affects results significantly
        hbar = 1.054571817e-34  # J⋅s
        zero_point_density = (hbar * self.zero_point_cutoff_frequency) / (2 * PhysicsConstants.C**3)
        
        # Force correction proportional to gradient of zero-point energy
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        zero_point_force = zero_point_density * B_field * dB_dz * self.proj_volume / PhysicsConstants.MU_0
        
        return NumericalUtils.safe_numerical_operation(zero_point_force, "zero_point_energy_force")
    
    def _calculate_quantum_tunneling_force(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate force corrections due to quantum tunneling through magnetic barriers.
        
        WARNING: For macroscopic objects (mass > 1e-15 kg), tunneling probability 
        is effectively zero due to exponential suppression. This calculation is 
        only meaningful for nanoscale particles.
        
        Quantum tunneling affects the effective permeability and thus the forces.
        """
        # For macroscopic projectiles, tunneling is negligible
        if self.proj_mass > 1e-15:  # kg - atomic scale threshold
            return 0.0
        
        # Magnetic field strength determines barrier height
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Effective barrier height in terms of magnetic energy
        barrier_height = self.tunneling_barrier_height * (B_field / 1.0)**2  # Tesla normalization
        
        # de Broglie wavelength of projectile
        momentum = self.proj_mass * abs(velocity) if abs(velocity) > 1e-6 else self.proj_mass * 1e-6
        de_broglie_wavelength = 6.62607015e-34 / momentum  # h/p
        
        # Tunneling probability (simplified WKB approximation)
        if de_broglie_wavelength > self.quantum_mechanical_threshold:
            kappa = np.sqrt(2 * self.proj_mass * barrier_height) / 1.054571817e-34
            barrier_width = self.proj_length  # Approximate barrier width
            
            # WKB tunneling probability - exponentially suppressed for macro objects
            tunneling_prob = np.exp(-2 * kappa * barrier_width)
            
            # For typical coilgun parameters, this probability is ~0 for macro objects
            if tunneling_prob < 1e-100:  # Numerical cutoff
                return 0.0
            
            # Force correction due to tunneling (highly approximate)
            tunneling_force = tunneling_prob * self.vacuum_fluctuation_amplitude * np.sign(velocity)
        else:
            tunneling_force = 0.0
        
        return NumericalUtils.safe_numerical_operation(tunneling_force, "quantum_tunneling_force")
    
    def _calculate_quantum_vacuum_stress(self, B_field: np.ndarray, position: np.ndarray) -> float:
        """
        Calculate quantum vacuum stress tensor contributions to force.
        
        WARNING: This calculation is highly speculative and approximate.
        Quantum vacuum stress effects on macroscopic forces are not 
        experimentally established and involve significant theoretical 
        uncertainties.
        
        Quantum vacuum fluctuations contribute to the stress tensor,
        modifying the Maxwell stress calculation.
        """
        # For macroscopic systems, vacuum stress corrections are negligible
        if not self.is_nanoscale_simulation:
            return 0.0
        
        # Quantum vacuum energy density
        hbar_c = 1.054571817e-34 * PhysicsConstants.C
        vacuum_energy_density = hbar_c * self.zero_point_cutoff_frequency / (8 * np.pi**3)
        
        # Quantum stress tensor correction (simplified)
        B_magnitude = np.linalg.norm(B_field)
        if B_magnitude > 1e-12:
            # Directional quantum stress along field direction
            field_direction = B_field / B_magnitude
            quantum_stress = vacuum_energy_density * (B_magnitude / 1.0)**2  # Tesla normalization
            
            # Force from stress gradient (simplified 1D approximation)
            force_z = quantum_stress * field_direction[2] * self.proj_volume
        else:
            force_z = 0.0
        
        return NumericalUtils.safe_numerical_operation(force_z, "quantum_vacuum_stress")
    
    def calculate_de_broglie_wavelength(self, velocity: float) -> float:
        """Calculate de Broglie wavelength of projectile."""
        momentum = self.proj_mass * abs(velocity) if abs(velocity) > 1e-6 else self.proj_mass * 1e-6
        return 6.62607015e-34 / momentum  # h/p
    
    def is_quantum_regime(self, velocity: float) -> bool:
        """Check if projectile is in quantum mechanical regime."""
        de_broglie = self.calculate_de_broglie_wavelength(velocity)
        return de_broglie > self.quantum_mechanical_threshold 