"""
Quantum Field Effects

This module handles quantum field theory corrections, vacuum birefringence,
and other extreme physics effects for ultra-high-field coilgun simulations.

NOTE: These effects are only relevant for fields approaching the Schwinger limit
(~10^13 T) and are disabled by default as they are irrelevant for realistic
coilgun fields (B << B_Schwinger).

Key Physics References:
- Heisenberg & Euler, Z. Phys. 98, 714 (1936) - Heisenberg-Euler Lagrangian
- Schwinger, Phys. Rev. 82, 664 (1951) - Pair production in strong fields  
- Adler, Ann. Phys. 67, 599 (1971) - Vacuum birefringence
- Heyl & Hernquist, Phys. Rev. D 58, 043005 (1998) - Birefringence calculations

Quantum Effects Included:
- Vacuum polarization corrections (Heisenberg-Euler Lagrangian)
- Schwinger pair production enhancement
- Vacuum magnetic birefringence with proper anisotropy
- Minimal plasma physics corrections (corrected magnetic screening)
- Synchrotron radiation (negligible for macroscopic projectiles)

Field Threshold: 10^9 T (still far below Schwinger limit but numerically significant)
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
        
        # Quantum corrections - disabled by default (irrelevant for B << Schwinger limit)
        self.include_quantum_corrections = quantum_cfg.get('enable_quantum', False)
        self.include_vacuum_birefringence = quantum_cfg.get('vacuum_birefringence', False)
        
        # Schwinger limit parameters (Heisenberg-Euler Lagrangian critical field)
        self.schwinger_limit = 4.4e13  # Tesla - quantum field theory limit
        self.field_threshold = 1e9  # Tesla - minimum field for quantum effects consideration
        self.include_plasma_effects = quantum_cfg.get('plasma_effects', False)
        self.include_synchrotron_radiation = quantum_cfg.get('synchrotron_radiation', False)
        
        if self.include_quantum_corrections:
            self._initialize_quantum_field_corrections()
        
        if self.include_vacuum_birefringence:
            self._initialize_vacuum_birefringence()
        
        print(f"🔬 Quantum field effects initialized (mostly disabled - irrelevant for B << B_Schwinger)")
        print(f"   - Quantum corrections: {'✓' if self.include_quantum_corrections else '✗ (disabled)'}")
        print(f"   - Vacuum birefringence: {'✓' if self.include_vacuum_birefringence else '✗ (disabled)'}")
        print(f"   - Plasma effects: {'✓' if self.include_plasma_effects else '✗ (disabled)'}")
        print(f"   - Synchrotron radiation: {'✓' if self.include_synchrotron_radiation else '✗ (disabled)'}")
        print(f"   - Schwinger limit: {self.schwinger_limit:.1e} T")
        print(f"   - Field threshold: {self.field_threshold:.1e} T")
    
    def _initialize_quantum_field_corrections(self):
        """Initialize quantum field theory corrections for extreme magnetic fields."""
        # Schwinger pair production threshold
        self.schwinger_field = 4.4e13  # Tesla (Heisenberg-Euler Lagrangian critical field)
        
        # Fundamental constants for QED calculations
        self.electron_charge = 1.602176634e-19  # C
        self.electron_mass = 9.1093837015e-31   # kg
        self.fine_structure_constant = 1/137.035999084  # α (2018 CODATA)
        
        # Quantum vacuum polarization effects
        # Based on Heisenberg-Euler Lagrangian for nonlinear QED
        # Reference: Heisenberg & Euler, Z. Phys. 98, 714 (1936)
        self.vacuum_polarization_coefficient = 2.3e-19  # Theoretical value
        
        # Critical field ratio cache for performance
        self.critical_field_ratio_cache = {}
        
        # Quantum correction lookup tables for efficiency
        self.quantum_correction_interpolator = None
        
        print(f"   - Schwinger critical field: {self.schwinger_field:.1e} T")
        print(f"   - Fine structure constant: α = {self.fine_structure_constant:.6f}")
        print(f"   - Field threshold: {self.field_threshold:.1e} T")
    
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
        
        # Only apply corrections for extremely high fields (10^9 T+)
        if B_magnitude < self.field_threshold:
            return B_field
        
        # Calculate field strength ratio to Schwinger limit
        field_ratio = float(B_magnitude / self.schwinger_field)
        
        if field_ratio > 1e-4:  # 0.01% of Schwinger limit
            # Vacuum polarization correction (Heisenberg-Euler Lagrangian)
            polarization_correction = self._calculate_vacuum_polarization_correction(field_ratio)
            
            # Pair production enhancement (not suppression)
            pair_production_factor = self._calculate_pair_production_factor(field_ratio)
            
            # Apply corrections
            correction_factor = (1 + polarization_correction) * pair_production_factor
            
            # Cache result for performance
            position_key = tuple(position.tolist())
            self.critical_field_ratio_cache[position_key] = field_ratio
            
            return B_field * correction_factor
        
        return B_field
    
    def _calculate_vacuum_polarization_correction(self, field_ratio: float) -> float:
        """
        Calculate vacuum polarization correction based on Heisenberg-Euler Lagrangian.
        
        Reference: Heisenberg & Euler, Z. Phys. 98, 714 (1936)
                  Schwinger, Phys. Rev. 82, 664 (1951)
        """
        if field_ratio < 1e-8:
            return 0.0
        
        # First-order correction: ΔB/B ≈ (α/π) * (B/B_c)²
        # This is the correct lowest-order term from QED
        correction = (self.fine_structure_constant / np.pi) * field_ratio**2
        
        # Higher-order corrections become important for stronger fields
        # Based on full Heisenberg-Euler effective Lagrangian
        if field_ratio > 0.01:
            # Fourth-order term (approximate)
            correction += (self.fine_structure_constant / np.pi) * 0.02 * field_ratio**4
        
        return correction
    
    def _calculate_pair_production_factor(self, field_ratio: float) -> float:
        """
        Calculate field enhancement due to pair production effects.
        
        Note: Pair production creates particles, enhancing rather than suppressing fields.
        Schwinger rate ~ exp(-B_c/B) describes the creation rate, not field suppression.
        
        Reference: Schwinger, Phys. Rev. 82, 664 (1951)
        """
        if field_ratio < 1e-6:
            return 1.0  # No significant pair production
        
        # Pair production rate increases exponentially as field approaches Schwinger limit
        # This enhances field effects rather than suppressing them
        if field_ratio > 1e-3:  # Significant pair production regime
            # Enhancement factor due to created charge carriers
            enhancement = 1.0 + 0.01 * field_ratio  # Small enhancement for realistic fields
            return enhancement
        
        return 1.0
    
    def apply_vacuum_birefringence_corrections(self, B_field: np.ndarray, position: np.ndarray) -> np.ndarray:
        """
        Apply vacuum magnetic birefringence corrections with proper anisotropy.
        
        Vacuum becomes birefringent in strong magnetic fields with different indices
        for parallel and perpendicular polarizations.
        
        Reference: Adler, Ann. Phys. 67, 599 (1971)
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            
        Returns:
            Birefringence-corrected magnetic field vector (T)
        """
        if not self.include_vacuum_birefringence:
            return B_field
        
        B_magnitude = np.linalg.norm(B_field)
        
        # Birefringence becomes significant at extremely high fields
        if B_magnitude < self.field_threshold:
            return B_field
        
        # Calculate birefringence tensor with proper anisotropy
        birefringence_tensor = self._calculate_birefringence_tensor(B_field)
        
        # Apply tensor transformation
        B_corrected = np.dot(birefringence_tensor, B_field)
        
        return B_corrected
    
    def _calculate_birefringence_tensor(self, B_field: np.ndarray) -> np.ndarray:
        """
        Calculate vacuum birefringence tensor with proper anisotropy.
        
        The vacuum refractive index becomes anisotropic in strong magnetic fields:
        - n_parallel = 1 + Δn_∥ (polarization parallel to B)
        - n_perpendicular = 1 + Δn_⊥ (polarization perpendicular to B)
        
        Reference: Heyl & Hernquist, Phys. Rev. D 58, 043005 (1998)
        """
        B_magnitude = np.linalg.norm(B_field)
        
        if B_magnitude < 1e-15:
            return np.eye(3)
        
        # Unit vector in field direction
        b_hat = B_field / B_magnitude
        
        # Cotton-Mouton constant and field strength
        field_ratio = B_magnitude / self.schwinger_field
        birefringence_strength = self.vacuum_cotton_mouton * B_magnitude**2
        
        # Different refractive indices for parallel and perpendicular polarizations
        # Δn_∥ ≈ (α/π) * (B/B_c)² * (7/45)  (parallel to B)
        # Δn_⊥ ≈ (α/π) * (B/B_c)² * (2/45)  (perpendicular to B)
        delta_n_parallel = (self.fine_structure_constant / np.pi) * field_ratio**2 * (7.0/45.0)
        delta_n_perpendicular = (self.fine_structure_constant / np.pi) * field_ratio**2 * (2.0/45.0)
        
        # Construct anisotropic tensor
        # T = (1 + Δn_⊥) * I + (Δn_∥ - Δn_⊥) * b⊗b
        tensor = (1 + delta_n_perpendicular) * np.eye(3)
        tensor += (delta_n_parallel - delta_n_perpendicular) * np.outer(b_hat, b_hat)
        
        return tensor
    
    def apply_plasma_physics_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                       velocity: float, current: float) -> np.ndarray:
        """
        Apply plasma physics corrections for ultra-high-speed projectiles.
        
        Note: Magnetic fields are not shielded by Debye screening like electric fields.
        Instead, plasma affects field propagation through current-driven effects.
        """
        if not self.include_plasma_effects:
            return B_field
        
        # Estimate plasma density from projectile velocity and current
        plasma_density = self._estimate_plasma_density(velocity, current)
        
        if plasma_density < 1e15:  # m^-3 threshold
            return B_field
        
        # Plasma frequency (correct formula)
        plasma_freq = np.sqrt(plasma_density * self.electron_charge**2 / 
                             (PhysicsConstants.EPSILON_0 * self.electron_mass))
        
        # For magnetic fields, plasma effects occur through induced currents, not Debye shielding
        # High conductivity plasma can support currents that oppose field changes (Lenz's law)
        if plasma_freq > 1e12:  # Significant plasma effects
            # Plasma conductivity affects field diffusion, not static shielding
            # This is relevant for time-varying fields, not static B-field corrections
            # For static fields, plasma has minimal direct effect
            conductivity_factor = 1.0 - min(0.05, plasma_freq / 1e15)  # Small reduction for AC components
            return B_field * conductivity_factor
        
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
        
        Note: Classical synchrotron formula applies to point charges (electrons).
        For macroscopic projectiles, synchrotron radiation is negligible due to:
        1. m^4 dependence in denominator makes it tiny for large masses
        2. Projectiles are not point charges but extended objects
        """
        if not self.include_synchrotron_radiation:
            return B_field
        
        # For macroscopic objects (mass >> electron mass), synchrotron radiation is negligible
        if mass > 1e-6:  # More than microgram - synchrotron radiation irrelevant
            return B_field
        
        # Only consider for extreme relativistic velocities
        if abs(velocity) < 0.1 * PhysicsConstants.C:  # Less than 10% speed of light
            return B_field
        
        # Relativistic gamma factor
        beta = velocity / PhysicsConstants.C
        gamma = 1.0 / np.sqrt(1 - beta**2) if abs(beta) < 0.99 else 100.0
        
        # Synchrotron power radiated (Larmor formula for relativistic motion)
        # P = (μ₀ e² a²)/(6π c) γ⁴  where a is acceleration
        B_magnitude = np.linalg.norm(B_field)
        
        # For charged particle in magnetic field: a = qvB/m
        charge_to_mass = self.electron_charge / mass  # Assume singly charged
        acceleration = charge_to_mass * velocity * B_magnitude
        
        # Classical radiation power (much smaller for heavy particles)
        radiation_power = (PhysicsConstants.MU_0 * self.electron_charge**2 * acceleration**2 * gamma**4) / \
                         (6 * np.pi * PhysicsConstants.C)
        
        # Energy loss affects field interaction (minimal for realistic scenarios)
        if radiation_power > 1e3:  # Significant power loss (W) - very unlikely for projectiles
            energy_loss_factor = 1.0 - min(0.01, float(radiation_power) / 1e6)  # Max 1% reduction
            return B_field * energy_loss_factor
        
        return B_field
    
    def calculate_field_stability_analysis(self, position: np.ndarray, current: float, 
                                         B_field: np.ndarray, perturbation_amplitude: float = 1e-6) -> dict:
        """
        Analyze magnetic field stability under quantum corrections.
        
        Uses more realistic stability threshold based on typical engineering tolerances.
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
            sensitivity = delta_B / (B_magnitude * perturbation_amplitude) if B_magnitude > 0 else 0.0
            
            perturbations.append(sensitivity)
            sensitivity_max = max(sensitivity_max, float(sensitivity))
        
        # Statistical analysis
        mean_sensitivity = float(np.mean(perturbations))
        std_sensitivity = float(np.std(perturbations))
        
        # Stability factor (lower is more stable)
        stability_factor = 1.0 / (1.0 + mean_sensitivity)
        
        # More realistic stability threshold (1% field variation tolerance)
        stability_threshold = 100.0  # 1% relative field change per unit perturbation
        
        return {
            'mean_sensitivity': mean_sensitivity,
            'std_sensitivity': std_sensitivity,
            'max_sensitivity': sensitivity_max,
            'stability_factor': stability_factor,
            'is_stable': sensitivity_max < stability_threshold,
            'stability_classification': self._classify_stability(float(stability_factor)),
            'stability_threshold': stability_threshold
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
    
    def are_quantum_effects_relevant(self, B_field: np.ndarray) -> dict:
        """
        Check if quantum effects are relevant for the given magnetic field.
        
        Args:
            B_field: Magnetic field vector (T)
            
        Returns:
            Dictionary with relevance assessment
        """
        B_magnitude = float(np.linalg.norm(B_field))
        field_ratio = float(B_magnitude / self.schwinger_limit)
        
        # Field strength classifications
        if B_magnitude < 1e6:  # < 1 MT
            regime = "Classical (quantum effects negligible)"
        elif B_magnitude < self.field_threshold:  # < 1 GT
            regime = "High field (quantum effects still negligible)"
        elif B_magnitude < 1e12:  # < 1000 GT
            regime = "Extreme field (weak quantum corrections)"
        elif field_ratio < 0.01:  # < 1% Schwinger
            regime = "Ultra-extreme field (moderate quantum effects)"
        elif field_ratio < 0.1:  # < 10% Schwinger
            regime = "Near-critical field (strong quantum effects)"
        else:
            regime = "Beyond Schwinger limit (breakdown of perturbative QED)"
        
        return {
            'field_magnitude': B_magnitude,
            'field_ratio_to_schwinger': field_ratio,
            'regime': regime,
            'quantum_relevant': B_magnitude >= self.field_threshold,
            'vacuum_polarization_relevant': field_ratio > 1e-6,
            'pair_production_relevant': field_ratio > 1e-4,
            'birefringence_relevant': B_magnitude >= self.field_threshold,
            'recommendations': self._get_field_recommendations(B_magnitude, field_ratio)
        }
    
    def _get_field_recommendations(self, B_magnitude: float, field_ratio: float) -> list:
        """Get recommendations based on field strength."""
        recommendations = []
        
        if B_magnitude < 1e6:
            recommendations.append("Use classical electromagnetic theory only")
        elif B_magnitude < self.field_threshold:
            recommendations.append("Classical theory sufficient; quantum effects < 0.01%")
        elif field_ratio < 1e-4:
            recommendations.append("Consider vacuum polarization corrections")
        elif field_ratio < 1e-2:
            recommendations.append("Include vacuum birefringence and pair production effects")
        else:
            recommendations.append("Full QED treatment required")
            recommendations.append("Perturbative expansion may break down")
            
        if B_magnitude > 1e15:
            recommendations.append("⚠️  Field exceeds all known material limits")
            
        return recommendations