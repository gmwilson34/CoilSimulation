"""
Field Corrections

This module handles various field corrections including relativistic effects,
magnetic diffusion, thermal-magnetic coupling, and other advanced corrections.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Union
import warnings
from ..core import PhysicsConstants, NumericalUtils


class FieldCorrections:
    """
    Field corrections for advanced physics effects.
    
    Handles:
    - Relativistic field transformations
    - Magnetic diffusion effects
    - Thermal-magnetic coupling
    - Piezomagnetic effects
    - Non-equilibrium magnetodynamics
    """
    
    def __init__(self, config: dict):
        """Initialize field corrections calculator."""
        self.config = config
        
        # Relativistic corrections
        self.include_relativistic = config.get('advanced_physics', {}).get('include_relativistic', True)
        self.relativistic_threshold = 0.001 * PhysicsConstants.C  # 0.1% of light speed
        
        # Magnetic diffusion
        self.include_magnetic_diffusion = config.get('advanced_physics', {}).get('include_magnetic_diffusion', True)
        self.diffusion_time_constant = 1e-6
        self.field_memory_depth = 1000
        
        # Thermal effects
        self.include_thermal_effects = config.get('advanced_physics', {}).get('thermal_magnetic_coupling', True)
        
        if self.include_relativistic:
            self._initialize_relativistic_corrections()
        
        if self.include_magnetic_diffusion:
            self._initialize_magnetic_diffusion()
        
        if self.include_thermal_effects:
            self._initialize_thermal_coupling()
        
        print(f"🔬 Field corrections initialized")
        print(f"   - Relativistic: {'✓' if self.include_relativistic else '✗'}")
        print(f"   - Magnetic diffusion: {'✓' if self.include_magnetic_diffusion else '✗'}")
        print(f"   - Thermal coupling: {'✓' if self.include_thermal_effects else '✗'}")
    
    def _initialize_relativistic_corrections(self):
        """Initialize relativistic field correction parameters."""
        self.lorentz_factor_cache = {}
        self.relativistic_field_cache = {}
        
        # Relativistic correction parameters
        self.length_contraction_enabled = True
        self.time_dilation_enabled = True
        self.electromagnetic_field_transformation = True
        
        print(f"   - Relativistic threshold: {self.relativistic_threshold/PhysicsConstants.C:.1%} c")
    
    def _initialize_magnetic_diffusion(self):
        """Initialize magnetic diffusion model."""
        self.field_history = []
        self.diffusion_kernel_cache = {}
        
        # Diffusion parameters
        self.magnetic_diffusivity = 1e-6  # m²/s - typical for metals
        self.eddy_current_time_constant = 1e-3  # s
        self.skin_depth_frequency_dependence = True
        
        print(f"   - Diffusion time constant: {self.diffusion_time_constant:.0e} s")
    
    def _initialize_thermal_coupling(self):
        """Initialize thermal-magnetic coupling parameters."""
        # Thermal properties
        self.thermal_diffusivity = 2e-5  # m²/s for metals
        self.curie_temperature = 1043  # K for iron
        self.thermal_expansion_coefficient = 12e-6  # K⁻¹
        
        # Magnetothermal effects
        self.magnetocaloric_effect = True
        self.thermomagnetic_seebeck = 10e-6  # V/K
        
        print(f"   - Curie temperature: {self.curie_temperature} K")
    
    def apply_relativistic_field_transform(self, B_field: np.ndarray, velocity: float) -> np.ndarray:
        """
        Apply relativistic field transformations for high-speed motion.
        
        Args:
            B_field: Magnetic field vector (T)
            velocity: Velocity along z-axis (m/s)
            
        Returns:
            Relativistically corrected magnetic field vector (T)
        """
        if not self.include_relativistic or abs(velocity) < self.relativistic_threshold:
            return B_field
        
        # Relativistic parameters
        beta = velocity / PhysicsConstants.C
        gamma = self._calculate_lorentz_factor(beta)
        
        # Field transformation (simplified for motion along z-axis)
        # B'_parallel = B_parallel (unchanged)
        # B'_perpendicular = γ * B_perpendicular
        
        if len(B_field) == 3:
            # 3D case
            B_parallel = B_field[2]  # z-component
            B_perp = np.array([B_field[0], B_field[1]])  # x,y components
            
            # Apply transformation
            B_perp_transformed = gamma * B_perp
            
            return np.array([B_perp_transformed[0], B_perp_transformed[1], B_parallel])
        else:
            # 1D case (assume field along z-axis)
            return B_field  # Parallel field unchanged
    
    def _calculate_lorentz_factor(self, beta: float) -> float:
        """Calculate Lorentz factor with caching."""
        if abs(beta) < 1e-10:
            return 1.0
        
        # Check cache
        beta_key = round(beta, 10)
        if beta_key in self.lorentz_factor_cache:
            return self.lorentz_factor_cache[beta_key]
        
        # Calculate gamma
        if abs(beta) >= 0.99:
            # Use series expansion for extreme relativistic case
            gamma = 100.0  # Approximate for very high speeds
        else:
            gamma = 1.0 / np.sqrt(1 - beta**2)
        
        # Cache result
        self.lorentz_factor_cache[beta_key] = gamma
        
        return gamma
    
    def apply_magnetic_diffusion_quantum(self, B_field: np.ndarray, position: np.ndarray, 
                                       time: float) -> np.ndarray:
        """
        Apply magnetic diffusion corrections with quantum memory effects.
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            time: Current time (s)
            
        Returns:
            Diffusion-corrected magnetic field vector (T)
        """
        if not self.include_magnetic_diffusion:
            return B_field
        
        # Store field history for diffusion calculations
        self._update_field_history(B_field, position, time)
        
        # Calculate diffusion correction
        diffusion_correction = self._calculate_diffusion_correction(position, time)
        
        # Apply correction
        return B_field * (1 + diffusion_correction)
    
    def _update_field_history(self, B_field: np.ndarray, position: np.ndarray, time: float):
        """Update field history for diffusion calculations."""
        history_entry = {
            'field': B_field.copy(),
            'position': position.copy(),
            'time': time
        }
        
        self.field_history.append(history_entry)
        
        # Limit history size
        if len(self.field_history) > self.field_memory_depth:
            self.field_history.pop(0)
    
    def _calculate_diffusion_correction(self, position: np.ndarray, time: float) -> float:
        """Calculate magnetic diffusion correction factor."""
        if len(self.field_history) < 2:
            return 0.0
        
        # Simple diffusion model based on field change rate
        recent_history = self.field_history[-10:]  # Last 10 entries
        
        if len(recent_history) < 2:
            return 0.0
        
        # Calculate field change rate
        dt = recent_history[-1]['time'] - recent_history[0]['time']
        if dt <= 0:
            return 0.0
        
        dB_dt = (np.linalg.norm(recent_history[-1]['field']) - 
                np.linalg.norm(recent_history[0]['field'])) / dt
        
        # Diffusion correction based on change rate and material properties
        diffusion_factor = self.magnetic_diffusivity * dB_dt * self.diffusion_time_constant
        
        # Limit correction magnitude
        return np.clip(diffusion_factor, -0.1, 0.1)  # Max ±10% correction
    
    def apply_thermal_magnetic_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                         temperature: float) -> np.ndarray:
        """
        Apply thermal-magnetic coupling corrections.
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            temperature: Temperature (K)
            
        Returns:
            Thermally corrected magnetic field vector (T)
        """
        if not self.include_thermal_effects:
            return B_field
        
        # Temperature-dependent permeability
        temp_factor = self._calculate_temperature_permeability_factor(temperature)
        
        # Magnetocaloric effect
        magnetocaloric_correction = self._calculate_magnetocaloric_correction(B_field, temperature)
        
        # Apply corrections
        correction_factor = temp_factor * (1 + magnetocaloric_correction)
        
        return B_field * correction_factor
    
    def _calculate_temperature_permeability_factor(self, temperature: float) -> float:
        """Calculate temperature-dependent permeability factor."""
        if temperature <= 0:
            return 1.0
        
        # Simple model for ferromagnetic materials
        if temperature < self.curie_temperature:
            # Below Curie temperature - ferromagnetic
            temp_ratio = temperature / self.curie_temperature
            
            # Empirical model: μ decreases with temperature
            factor = 1.0 - 0.3 * temp_ratio**2  # 30% reduction at Curie point
        else:
            # Above Curie temperature - paramagnetic
            factor = 0.001  # Very low permeability
        
        return max(factor, 0.001)  # Minimum permeability
    
    def _calculate_magnetocaloric_correction(self, B_field: np.ndarray, temperature: float) -> float:
        """Calculate magnetocaloric effect correction."""
        B_magnitude = np.linalg.norm(B_field)
        
        if B_magnitude < 0.1 or temperature <= 0:  # Weak field or invalid temperature
            return 0.0
        
        # Magnetocaloric temperature change
        # ΔT ≈ -T * (∂M/∂T)_H * ΔB / (ρ * c_p)
        # Simplified model
        
        magnetocaloric_coefficient = 1e-3  # Empirical coefficient
        delta_T = magnetocaloric_coefficient * B_magnitude**2 / temperature
        
        # Correction to field due to temperature change
        temp_correction = self._calculate_temperature_permeability_factor(temperature + delta_T) - \
                         self._calculate_temperature_permeability_factor(temperature)
        
        return temp_correction
    
    def apply_piezomagnetic_corrections(self, B_field: np.ndarray, stress_tensor: np.ndarray, 
                                      position: np.ndarray) -> np.ndarray:
        """
        Apply piezomagnetic corrections due to mechanical stress.
        
        Args:
            B_field: Magnetic field vector (T)
            stress_tensor: Stress tensor (Pa)
            position: Position vector (m)
            
        Returns:
            Stress-corrected magnetic field vector (T)
        """
        if stress_tensor is None:
            return B_field
        
        # Piezomagnetic coefficient
        piezomagnetic_d = 1e-8  # Pa⁻¹ - typical value
        
        # Calculate hydrostatic stress
        hydrostatic_stress = np.trace(stress_tensor) / 3.0
        
        # Piezomagnetic correction
        correction_factor = 1.0 + piezomagnetic_d * hydrostatic_stress
        
        return B_field * correction_factor
    
    def apply_nonequilibrium_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                       velocity: float, temperature: float) -> np.ndarray:
        """
        Apply non-equilibrium magnetodynamics corrections.
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            velocity: Velocity (m/s)
            temperature: Temperature (K)
            
        Returns:
            Non-equilibrium corrected magnetic field vector (T)
        """
        B_magnitude = np.linalg.norm(B_field)
        
        if B_magnitude < 0.01:  # Weak field
            return B_field
        
        # Magnetic relaxation effects
        relaxation_correction = self._calculate_magnetic_relaxation_correction(
            B_magnitude, velocity, temperature
        )
        
        # Domain switching effects
        domain_correction = self._calculate_domain_switching_correction(
            B_magnitude, velocity
        )
        
        # Apply corrections
        total_correction = 1.0 + relaxation_correction + domain_correction
        
        return B_field * total_correction
    
    def _calculate_magnetic_relaxation_correction(self, B_magnitude: float, 
                                                velocity: float, temperature: float) -> float:
        """Calculate magnetic relaxation correction."""
        # Relaxation time depends on temperature and field strength
        thermal_energy = PhysicsConstants.K_BOLTZMANN * temperature
        magnetic_energy = B_magnitude**2 / (2 * PhysicsConstants.MU_0)
        
        # Energy ratio affects relaxation
        energy_ratio = thermal_energy / magnetic_energy if magnetic_energy > 0 else 0
        
        # Velocity-dependent relaxation
        velocity_factor = min(1.0, abs(velocity) / 1000.0)  # Normalize to 1 km/s
        
        # Relaxation correction
        relaxation_factor = 0.01 * energy_ratio * velocity_factor  # Max 1% correction
        
        return np.clip(relaxation_factor, -0.05, 0.05)  # Limit to ±5%
    
    def _calculate_domain_switching_correction(self, B_magnitude: float, velocity: float) -> float:
        """Calculate magnetic domain switching correction."""
        # Domain switching threshold
        switching_threshold = 0.1  # Tesla
        
        if B_magnitude < switching_threshold:
            return 0.0
        
        # Velocity-dependent domain switching
        switching_rate = abs(velocity) / 100.0  # Normalize to 100 m/s
        field_factor = (B_magnitude / switching_threshold - 1.0)
        
        # Domain switching correction
        domain_factor = 0.005 * switching_rate * field_factor  # Small correction
        
        return np.clip(domain_factor, 0, 0.02)  # Max 2% increase
    
    def calculate_field_with_thermal_magnetic_coupling(self, position: np.ndarray, current: float, 
                                                     velocity: float, temperature: float, 
                                                     stress_tensor: np.ndarray = None) -> np.ndarray:
        """
        Calculate field with full thermal-magnetic coupling.
        
        This is a high-level method that applies multiple corrections together.
        """
        # Start with base field (this would need to be calculated elsewhere)
        # For this correction module, we assume the base field is provided
        
        # Create a dummy base field for demonstration
        # In practice, this would be calculated by the core field calculator
        base_field = np.array([0.0, 0.0, 0.1])  # Placeholder
        
        # Apply thermal corrections
        field_with_thermal = self.apply_thermal_magnetic_corrections(
            base_field, position, temperature
        )
        
        # Apply piezomagnetic corrections
        if stress_tensor is not None:
            field_with_stress = self.apply_piezomagnetic_corrections(
                field_with_thermal, stress_tensor, position
            )
        else:
            field_with_stress = field_with_thermal
        
        # Apply non-equilibrium corrections
        field_final = self.apply_nonequilibrium_corrections(
            field_with_stress, position, velocity, temperature
        )
        
        return field_final 