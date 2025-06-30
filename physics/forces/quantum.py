"""
Quantum Force Calculations

This module implements quantum mechanical force corrections including
Casimir forces, zero-point energy effects, and quantum tunneling.
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
        
        # QUANTUM PHYSICS PARAMETERS
        self.include_quantum_effects = config.get('quantum_physics', {}).get('enable_quantum_forces', True)
        self.casimir_force_enabled = config.get('quantum_physics', {}).get('casimir_force', True)
        self.quantum_tunneling_effects = config.get('quantum_physics', {}).get('quantum_tunneling', True)
        self.zero_point_energy_effects = config.get('quantum_physics', {}).get('zero_point_energy', True)
        
        if self.include_quantum_effects:
            self._initialize_quantum_force_models()
        
        print(f"🔬 Quantum force calculator initialized")
        print(f"   - Quantum effects: {'✓' if self.include_quantum_effects else '✗'}")
        print(f"   - Casimir forces: {'✓' if self.casimir_force_enabled else '✗'}")
        print(f"   - Zero-point energy: {'✓' if self.zero_point_energy_effects else '✗'}")
        print(f"   - Quantum tunneling: {'✓' if self.quantum_tunneling_effects else '✗'}")
    
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
        
        Returns:
            Tuple of (total_quantum_force, force_breakdown)
        """
        if not self.include_quantum_effects:
            return 0.0, {}
        
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
        
        return total_quantum_force, force_breakdown
    
    def _calculate_casimir_force(self, position: float, velocity: float) -> float:
        """
        Calculate Casimir force between conductor surfaces.
        
        F_casimir = -π²ħc/(240 * d⁴) * A
        where d is the separation distance and A is the area.
        """
        # Distance from projectile to coil inner surface
        distance = abs(self.field_calc.coil_radius - self.proj_radius)
        
        # Minimum distance for validity (avoid singularities)
        min_distance = 1e-9  # 1 nm
        distance = max(distance, min_distance)
        
        # Effective interaction area
        interaction_area = 2 * np.pi * self.proj_radius * self.proj_length
        
        # Casimir force (attractive, hence negative)
        casimir_force = -self.casimir_coefficient * interaction_area / (distance**4)
        
        # Apply velocity-dependent corrections (relativistic quantum field theory)
        if abs(velocity) > 1e-6:
            velocity_factor = 1.0 - (velocity / PhysicsConstants.C)**2
            casimir_force *= velocity_factor
        
        return NumericalUtils.safe_numerical_operation(casimir_force, "casimir_force")
    
    def _calculate_zero_point_energy_force(self, current: float, position: float) -> float:
        """
        Calculate force corrections due to zero-point energy fluctuations.
        
        Zero-point energy affects the electromagnetic field energy density,
        leading to additional force contributions.
        """
        # Magnetic field energy density
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        magnetic_energy_density = B_field**2 / (2 * PhysicsConstants.MU_0)
        
        # Zero-point energy density (cutoff at characteristic frequency)
        hbar = 1.054571817e-34  # J⋅s
        zero_point_density = (hbar * self.zero_point_cutoff_frequency) / (2 * PhysicsConstants.C**3)
        
        # Force correction proportional to gradient of zero-point energy
        dB_dz = self.field_calc.calculate_field_gradient(position, current)
        zero_point_force = zero_point_density * B_field * dB_dz * self.proj_volume / PhysicsConstants.MU_0
        
        return NumericalUtils.safe_numerical_operation(zero_point_force, "zero_point_energy_force")
    
    def _calculate_quantum_tunneling_force(self, current: float, position: float, velocity: float) -> float:
        """
        Calculate force corrections due to quantum tunneling through magnetic barriers.
        
        Quantum tunneling affects the effective permeability and thus the forces.
        """
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
            tunneling_prob = np.exp(-2 * kappa * self.proj_length)
            
            # Force correction due to tunneling
            tunneling_force = tunneling_prob * self.vacuum_fluctuation_amplitude * np.sign(velocity)
        else:
            tunneling_force = 0.0
        
        return NumericalUtils.safe_numerical_operation(tunneling_force, "quantum_tunneling_force")
    
    def _calculate_quantum_vacuum_stress(self, B_field: np.ndarray, position: np.ndarray) -> float:
        """
        Calculate quantum vacuum stress tensor contributions to force.
        
        Quantum vacuum fluctuations contribute to the stress tensor,
        modifying the Maxwell stress calculation.
        """
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