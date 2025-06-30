"""
Quantum Field Effects

This module handles quantum field theory corrections, vacuum birefringence,
and other extreme physics effects for ultra-high-field coilgun simulations.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Union
import warnings
from ..core import PhysicsConstants


class QuantumFieldEffects:
    """
    Quantum field theory corrections for extreme magnetic fields.
    
    Handles:
    - Schwinger pair production effects
    - Vacuum magnetic birefringence
    - Quantum vacuum polarization
    - Heisenberg-Euler Lagrangian effects
    - Plasma physics corrections
    - Synchrotron radiation losses
    """
    
    def __init__(self, config: dict):
        """Initialize quantum field effects calculator."""
        quantum_cfg = config.get('quantum_physics', {})
        
        # Quantum corrections
        self.include_quantum_corrections = quantum_cfg.get('enable_quantum', True)
        self.include_vacuum_birefringence = quantum_cfg.get('vacuum_birefringence', True)
        
        # Schwinger limit parameters
        self.schwinger_limit = 4.4e13  # Tesla - quantum field theory limit
        self.include_plasma_effects = quantum_cfg.get('plasma_effects', True)
        self.include_synchrotron_radiation = quantum_cfg.get('synchrotron_radiation', True)
        
        if self.include_quantum_corrections:
            self._initialize_quantum_field_corrections()
        
        if self.include_vacuum_birefringence:
            self._initialize_vacuum_birefringence()
        
        print(f"🔬 Quantum field effects initialized")
        print(f"   - Quantum corrections: {'✓' if self.include_quantum_corrections else '✗'}")
        print(f"   - Vacuum birefringence: {'✓' if self.include_vacuum_birefringence else '✗'}")
        print(f"   - Schwinger limit: {self.schwinger_limit:.1e} T")
    
    def _initialize_quantum_field_corrections(self):
        """Initialize quantum field theory corrections for extreme magnetic fields."""
        # Schwinger pair production threshold
        self.schwinger_field = 4.4e13  # Tesla
        
        # Quantum vacuum polarization effects
        # At extreme fields, virtual electron-positron pairs affect field propagation
        self.vacuum_polarization_coefficient = 2.3e-19  # Theoretical value
        
        # Heisenberg-Euler Lagrangian parameters for nonlinear QED
        self.alpha_fine_structure = 1/137.036  # Fine structure constant
        self.critical_field_ratio_cache = {}
        
        # Quantum correction lookup tables for efficiency
        self.quantum_correction_interpolator = None
        
        # Plasma physics corrections for ultra-high-speed projectiles
        self.plasma_frequency_threshold = 1e12  # Hz - when projectile ionizes air
        self.debye_length_vacuum = 7.4e2  # m - vacuum Debye length
        self.landau_damping_coefficient = 0.15  # Typical value for electromagnetic waves
        
        # Magnetic reconnection parameters for extreme current densities
        self.reconnection_threshold_current = 1e7  # A - when magnetic reconnection occurs
        self.hall_parameter_metals = 0.01  # Hall parameter for metallic projectiles
        self.magnetic_resistivity_anomalous = 1e-4  # Anomalous resistivity from turbulence
        
        # Synchrotron radiation losses for ultra-relativistic motion
        self.synchrotron_power_coefficient = 6.22e-12  # W⋅s²⋅kg⁻²⋅m⁻²
        self.radiation_reaction_threshold = 0.01 * PhysicsConstants.C  # 1% c
        
        # Thermal-magnetic coupling
        self.magnetic_thermal_coupling = True
        self.thermal_diffusivity_metals = 2e-5  # m²/s typical for metals
        self.magnetic_thermal_exchange_rate = 1e6  # W/(m³⋅K) heat exchange coefficient
        
        # Non-equilibrium magnetodynamics
        self.magnetic_relaxation_time = 1e-9  # s - magnetic moment relaxation
        self.domain_switching_time = 1e-12  # s - magnetic domain switching
        self.demagnetization_time_constant = 1e-6  # s - thermal demagnetization
        
        # Multi-physics coupling coefficients
        self.magnetostriction_coefficient = 20e-6  # Typical magnetostriction strain
        self.piezomagnetic_coefficient = 1e-8  # Pa⁻¹ - stress-induced magnetization change
        self.thermoelectric_seebeck = 10e-6  # V/K - Seebeck coefficient for thermal EMF
        
        print(f"   - Plasma physics: ν_p = {self.plasma_frequency_threshold:.0e} Hz")
        print(f"   - Magnetic reconnection: I_crit = {self.reconnection_threshold_current:.0e} A")
        print(f"   - Synchrotron radiation: v_thresh = {self.radiation_reaction_threshold/PhysicsConstants.C:.1%} c")
        print(f"   - Thermal-magnetic coupling: ✓ Enabled")
        print(f"   - Non-equilibrium dynamics: τ_mag = {self.magnetic_relaxation_time:.0e} s")
    
    def _initialize_vacuum_birefringence(self):
        """Initialize vacuum magnetic birefringence calculations."""
        # In ultra-strong magnetic fields, vacuum becomes birefringent
        # This affects field propagation and creates nonlinear effects
        
        # Cotton-Mouton constant for vacuum
        self.vacuum_cotton_mouton = 4e-24  # T^-2 (theoretical)
        
        # Vacuum permeability becomes field-dependent
        self.nonlinear_vacuum_effects = True
        
        # Birefringence tensor components
        self.vacuum_birefringence_cache = {}
        
        print(f"   - Vacuum Cotton-Mouton constant: {self.vacuum_cotton_mouton:.0e} T⁻²")
    
    def apply_quantum_corrections(self, B_field: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        Apply quantum field theory corrections to magnetic field.
        
        Args:
            B_field: Classical magnetic field vector (T)
            position: Position vector (m)
            
        Returns:
            Quantum-corrected magnetic field vector (T)
        """
        if not self.include_quantum_corrections:
            return B_field
        
        B_magnitude = np.linalg.norm(B_field)
        
        # Only apply corrections for very high fields
        if B_magnitude < 1e6:  # 1 MegaTesla threshold
            return B_field
        
        # Calculate field strength ratio to Schwinger limit
        field_ratio = B_magnitude / self.schwinger_field
        
        if field_ratio > 0.01:  # 1% of Schwinger limit
            # Vacuum polarization correction
            polarization_correction = self._calculate_vacuum_polarization_correction(field_ratio)
            
            # Pair production suppression
            pair_production_factor = self._calculate_pair_production_factor(field_ratio)
            
            # Apply corrections
            correction_factor = (1 + polarization_correction) * pair_production_factor
            
            # Cache result for performance
            position_key = tuple(position.tolist())
            self.critical_field_ratio_cache[position_key] = field_ratio
            
            return B_field * correction_factor
        
        return B_field
    
    def _calculate_vacuum_polarization_correction(self, field_ratio: float) -> float:
        """Calculate vacuum polarization correction to field."""
        # Based on Heisenberg-Euler Lagrangian
        # ΔB/B ≈ (α/π) * (B/B_c)² for B << B_c
        
        if field_ratio < 1e-6:
            return 0.0
        
        # First-order correction
        correction = (self.alpha_fine_structure / np.pi) * field_ratio**2
        
        # Higher-order corrections for stronger fields
        if field_ratio > 0.1:
            correction += 0.1 * (self.alpha_fine_structure / np.pi) * field_ratio**3
        
        return correction
    
    def _calculate_pair_production_factor(self, field_ratio: float) -> float:
        """Calculate field reduction due to pair production."""
        if field_ratio < 0.01:
            return 1.0  # No significant pair production
        
        # Exponential suppression near Schwinger limit
        suppression = np.exp(-1.0 / field_ratio) if field_ratio > 0 else 1.0
        
        return 1.0 - 0.1 * suppression  # 10% maximum suppression
    
    def apply_vacuum_birefringence_corrections(self, B_field: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        Apply vacuum magnetic birefringence corrections.
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            
        Returns:
            Birefringence-corrected magnetic field vector (T)
        """
        if not self.include_vacuum_birefringence:
            return B_field
        
        B_magnitude = np.linalg.norm(B_field)
        
        # Birefringence becomes significant at very high fields
        if B_magnitude < 1e8:  # 100 MegaTesla threshold
            return B_field
        
        # Calculate birefringence tensor
        birefringence_tensor = self._calculate_birefringence_tensor(B_field)
        
        # Apply tensor transformation
        B_corrected = np.dot(birefringence_tensor, B_field)
        
        return B_corrected
    
    def _calculate_birefringence_tensor(self, B_field: np.ndarray) -> np.ndarray:
        """Calculate vacuum birefringence tensor."""
        B_magnitude = np.linalg.norm(B_field)
        
        if B_magnitude < 1e-15:
            return np.eye(3)
        
        # Unit vector in field direction
        b_hat = B_field / B_magnitude
        
        # Birefringence strength
        birefringence_strength = self.vacuum_cotton_mouton * B_magnitude**2
        
        # Birefringence tensor (simplified isotropic model)
        # Full treatment requires complex tensor analysis
        tensor = np.eye(3) * (1 + birefringence_strength)
        
        # Anisotropic correction along field direction
        tensor += birefringence_strength * 0.1 * np.outer(b_hat, b_hat)
        
        return tensor
    
    def apply_plasma_physics_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                       velocity: float, current: float) -> np.ndarray:
        """
        Apply plasma physics corrections for ultra-high-speed projectiles.
        """
        if not self.include_plasma_effects:
            return B_field
        
        # Estimate plasma density from projectile velocity and current
        plasma_density = self._estimate_plasma_density(velocity, current)
        
        if plasma_density < 1e15:  # m^-3 threshold
            return B_field
        
        # Plasma frequency
        plasma_freq = np.sqrt(plasma_density * PhysicsConstants.ELECTRON_CHARGE**2 / 
                             (PhysicsConstants.EPSILON_0 * PhysicsConstants.ELECTRON_MASS))
        
        # Plasma effects on field propagation
        if plasma_freq > self.plasma_frequency_threshold:
            # Field shielding by plasma
            debye_length = np.sqrt(PhysicsConstants.EPSILON_0 * PhysicsConstants.K_BOLTZMANN * 1e4 / 
                                  (plasma_density * PhysicsConstants.ELECTRON_CHARGE**2))  # Assume 10^4 K plasma
            
            # Distance-dependent shielding
            r = np.linalg.norm(position)
            shielding_factor = np.exp(-r / debye_length) if debye_length > 0 else 1.0
            
            return B_field * shielding_factor
        
        return B_field
    
    def _estimate_plasma_density(self, velocity: float, current: float) -> float:
        """Estimate plasma density from projectile parameters."""
        # Empirical model for air ionization by high-speed metallic projectiles
        
        # Critical velocity for air ionization (~5 km/s)
        critical_velocity = 5000.0  # m/s
        
        if velocity < critical_velocity:
            return 0.0
        
        # Velocity-dependent ionization
        velocity_factor = (velocity / critical_velocity)**2
        
        # Current-dependent ionization (Joule heating)
        current_factor = (current / 1000.0)**1.5  # Normalized to 1 kA
        
        # Baseline atmospheric density
        atmospheric_density = 2.5e25  # molecules/m³ at STP
        
        # Ionization fraction (rough estimate)
        ionization_fraction = min(0.1, velocity_factor * current_factor * 1e-6)
        
        return atmospheric_density * ionization_fraction
    
    def apply_synchrotron_radiation_corrections(self, B_field: np.ndarray, velocity: float, 
                                              mass: float = 0.01) -> np.ndarray:
        """
        Apply synchrotron radiation energy loss corrections.
        """
        if not self.include_synchrotron_radiation:
            return B_field
        
        if abs(velocity) < self.radiation_reaction_threshold:
            return B_field
        
        # Relativistic gamma factor
        beta = velocity / PhysicsConstants.C
        gamma = 1.0 / np.sqrt(1 - beta**2) if abs(beta) < 0.99 else 100.0
        
        # Synchrotron power radiated
        B_magnitude = np.linalg.norm(B_field)
        synchrotron_power = (self.synchrotron_power_coefficient * 
                           PhysicsConstants.ELECTRON_CHARGE**4 * B_magnitude**2 * 
                           gamma**4 / (6 * np.pi * PhysicsConstants.EPSILON_0 * 
                           PhysicsConstants.ELECTRON_MASS**2 * PhysicsConstants.C**3))
        
        # Energy loss rate affects field interaction
        if synchrotron_power > 1e6:  # Significant power loss (W)
            energy_loss_factor = 1.0 - min(0.1, synchrotron_power / 1e8)  # Max 10% reduction
            return B_field * energy_loss_factor
        
        return B_field
    
    def calculate_field_stability_analysis(self, position: np.ndarray, current: float, 
                                         B_field: np.ndarray, perturbation_amplitude: float = 1e-6) -> dict:
        """
        Analyze magnetic field stability under quantum corrections.
        """
        B_magnitude = np.linalg.norm(B_field)
        
        # Test stability against small perturbations
        perturbations = []
        sensitivity_max = 0.0
        
        for i in range(10):
            # Random perturbation
            perturbation = perturbation_amplitude * (2 * np.random.random(3) - 1)
            perturbed_position = position + perturbation
            
            # Calculate field at perturbed position
            B_perturbed = self.apply_quantum_corrections(B_field, perturbed_position)
            
            # Sensitivity measure
            delta_B = np.linalg.norm(B_perturbed - B_field)
            sensitivity = delta_B / (B_magnitude * perturbation_amplitude)
            
            perturbations.append(sensitivity)
            sensitivity_max = max(sensitivity_max, sensitivity)
        
        # Statistical analysis
        mean_sensitivity = np.mean(perturbations)
        std_sensitivity = np.std(perturbations)
        
        # Stability factor (lower is more stable)
        stability_factor = 1.0 / (1.0 + mean_sensitivity)
        
        return {
            'mean_sensitivity': mean_sensitivity,
            'std_sensitivity': std_sensitivity,
            'max_sensitivity': sensitivity_max,
            'stability_factor': stability_factor,
            'is_stable': sensitivity_max < 10.0,  # Threshold for acceptable stability
            'stability_classification': self._classify_stability(stability_factor)
        }
    
    def _classify_stability(self, stability_factor: float) -> str:
        """Classify magnetic field stability."""
        if stability_factor > 0.9:
            return "Highly Stable"
        elif stability_factor > 0.7:
            return "Stable"
        elif stability_factor > 0.5:
            return "Marginally Stable"
        elif stability_factor > 0.3:
            return "Unstable"
        else:
            return "Highly Unstable" 