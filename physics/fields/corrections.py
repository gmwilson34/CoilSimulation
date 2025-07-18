"""
Field Corrections

This module handles various field corrections including relativistic effects,
magnetic diffusion, thermal-magnetic coupling, and other advanced corrections.

Note: These corrections are typically negligible for coilgun applications and are
disabled by default. They are provided for completeness and advanced research.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Union, Dict, Any
import warnings
from ..core import PhysicsConstants, NumericalUtils, BasePhysicsModel
from ..materials import AdvancedMaterialProperties


class FieldCorrections(BasePhysicsModel):
    """
    Field corrections for advanced physics effects.
    
    Note: These corrections are typically negligible for coilgun applications
    and are disabled by default to avoid unnecessary computational overhead.
    
    Handles:
    - Relativistic field transformations (for high-velocity projectiles)
    - Magnetic diffusion effects (eddy currents, skin effect)
    - Thermal-magnetic coupling (Curie-Weiss law, magnetocaloric effect)
    - Piezomagnetic effects (stress-induced magnetic changes)
    - Non-equilibrium magnetodynamics (Landau-Lifshitz-Gilbert equation)
    """
    
    def __init__(self, config: dict, material_properties: Optional[AdvancedMaterialProperties] = None):
        """Initialize field corrections calculator."""
        super().__init__(config)
        
        # Material properties integration
        self.materials = material_properties
        
        # Enable/disable specific corrections (all disabled by default for coilguns)
        advanced_physics = config.get('advanced_physics', {})
        self.include_relativistic = advanced_physics.get('include_relativistic', False)
        self.include_magnetic_diffusion = advanced_physics.get('include_magnetic_diffusion', False)
        self.include_thermal_effects = advanced_physics.get('thermal_magnetic_coupling', False)
        self.include_piezomagnetic = advanced_physics.get('include_piezomagnetic', False)
        self.include_nonequilibrium = advanced_physics.get('include_nonequilibrium', False)
        
        # Relativistic corrections
        self.relativistic_threshold = 0.01 * PhysicsConstants.C  # 1% of light speed
        
        # Magnetic diffusion parameters
        self.diffusion_time_constant = 1e-6  # s
        self.field_memory_depth = 100  # Reduced from 1000 for memory efficiency
        self.field_history = []
        
        # Initialize subsystems if enabled
        if self.include_relativistic:
            self._initialize_relativistic_corrections()
        
        if self.include_magnetic_diffusion:
            self._initialize_magnetic_diffusion()
        
        if self.include_thermal_effects:
            self._initialize_thermal_coupling()
        
        # Print status (only if any corrections are enabled)
        if any([self.include_relativistic, self.include_magnetic_diffusion, 
               self.include_thermal_effects, self.include_piezomagnetic, 
               self.include_nonequilibrium]):
            print(f"🔬 Field corrections initialized")
            print(f"   - Relativistic: {'✓' if self.include_relativistic else '✗'}")
            print(f"   - Magnetic diffusion: {'✓' if self.include_magnetic_diffusion else '✗'}")
            print(f"   - Thermal coupling: {'✓' if self.include_thermal_effects else '✗'}")
            print(f"   - Piezomagnetic: {'✓' if self.include_piezomagnetic else '✗'}")
            print(f"   - Non-equilibrium: {'✓' if self.include_nonequilibrium else '✗'}")
        else:
            print(f"🔬 Field corrections: All disabled (recommended for coilguns)")
    
    def _initialize_relativistic_corrections(self):
        """Initialize relativistic field correction parameters."""
        self.lorentz_factor_cache = {}
        print(f"   - Relativistic threshold: {self.relativistic_threshold/PhysicsConstants.C:.1%} c")
    
    def _initialize_magnetic_diffusion(self):
        """Initialize magnetic diffusion model."""
        self.diffusion_kernel_cache = {}
        
        # Default diffusion parameters (can be overridden by material properties)
        self.magnetic_diffusivity = 1e-6  # m²/s - typical for metals
        self.eddy_current_time_constant = 1e-3  # s
        
        print(f"   - Diffusion time constant: {self.diffusion_time_constant:.0e} s")
    
    def _initialize_thermal_coupling(self):
        """Initialize thermal-magnetic coupling parameters."""
        # Default thermal properties (will be overridden by material data)
        self.thermal_diffusivity = 2e-5  # m²/s for metals
        
        print(f"   - Thermal coupling initialized")
    
    def apply_relativistic_field_transform(self, B_field: np.ndarray, velocity: float, 
                                         reference_frame: str = 'lab') -> np.ndarray:
        """
        Apply relativistic field transformations for high-speed motion.
        
        Proper relativistic electromagnetic field transformations between
        lab frame and projectile frame.
        
        Args:
            B_field: Magnetic field vector (T)
            velocity: Velocity along z-axis (m/s)
            reference_frame: 'lab' or 'projectile' frame
            
        Returns:
            Relativistically corrected magnetic field vector (T)
        """
        if not self.include_relativistic or abs(velocity) < self.relativistic_threshold:
            return B_field
        
        # Relativistic parameters
        beta = velocity / PhysicsConstants.C
        gamma = self._calculate_lorentz_factor(beta)
        
        # Proper field transformation (Lorentz transformation)
        # For motion along z-axis:
        # B'_parallel = B_parallel (unchanged)
        # B'_perpendicular = γ(B_perpendicular - β × E_perpendicular/c)
        # For magnetic fields only (quasistatic approximation E ≈ 0):
        
        if len(B_field) == 3:
            # 3D case
            B_parallel = B_field[2]  # z-component (unchanged)
            B_perp = np.array([B_field[0], B_field[1]])  # x,y components
            
            if reference_frame == 'lab':
                # Transform from lab to moving frame
                B_perp_transformed = gamma * B_perp
            else:
                # Transform from moving frame to lab
                B_perp_transformed = B_perp / gamma
            
            return np.array([B_perp_transformed[0], B_perp_transformed[1], B_parallel])
        else:
            # 1D case - assume field along z-axis (parallel component unchanged)
            return B_field
    
    def _calculate_lorentz_factor(self, beta: float) -> float:
        """Calculate Lorentz factor with caching and proper handling of extreme cases."""
        if abs(beta) < 1e-10:
            return 1.0
        
        # Check cache
        beta_key = round(beta, 10)
        if beta_key in self.lorentz_factor_cache:
            return self.lorentz_factor_cache[beta_key]
        
        # Calculate gamma with proper relativistic formula
        if abs(beta) >= 0.999:
            # Use series expansion for extreme relativistic case to avoid numerical issues
            # γ ≈ 1/(2(1-β)) for β → 1
            gamma = 1.0 / (2 * (1 - abs(beta)))
        else:
            gamma = 1.0 / np.sqrt(1 - beta**2)
        
        # Cache result
        self.lorentz_factor_cache[beta_key] = gamma
        
        return gamma
    
    def apply_magnetic_diffusion_correction(self, B_field: np.ndarray, position: np.ndarray, 
                                          time: float, material_name: str = 'Low_Carbon_Steel') -> np.ndarray:
        """
        Apply magnetic diffusion corrections based on eddy currents and skin effect.
        
        Uses proper diffusion equation: ∂B/∂t = η∇²B where η = 1/(μσ)
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            time: Current time (s)
            material_name: Material for diffusion properties
            
        Returns:
            Diffusion-corrected magnetic field vector (T)
        """
        if not self.include_magnetic_diffusion:
            return B_field
        
        # Get material-specific diffusion properties
        diffusivity = self._get_magnetic_diffusivity(material_name)
        
        # Store field history for diffusion calculations
        self._update_field_history(B_field, position, time)
        
        # Calculate diffusion correction based on field gradient
        diffusion_correction = self._calculate_diffusion_correction(position, time, diffusivity)
        
        # Apply correction with proper physics
        return B_field * (1 + diffusion_correction)
    
    def _get_magnetic_diffusivity(self, material_name: str) -> float:
        """Get magnetic diffusivity η = 1/(μσ) from material properties."""
        if self.materials is None:
            return self.magnetic_diffusivity  # Default value
        
        try:
            # Get material properties
            mu_r = self.materials.get_material_property(material_name, 'mu_r', 1000.0)
            resistivity = self.materials.get_material_property(material_name, 'resistivity_20C', 1e-7)
            
            # Calculate diffusivity: η = ρ/(μ₀μᵣ)
            mu_abs = PhysicsConstants.MU_0 * mu_r
            conductivity = 1.0 / resistivity
            diffusivity = 1.0 / (mu_abs * conductivity)
            
            return diffusivity
        except:
            return self.magnetic_diffusivity  # Fallback
    
    def _update_field_history(self, B_field: np.ndarray, position: np.ndarray, time: float):
        """Update field history for diffusion calculations with memory management."""
        history_entry = {
            'field': B_field.copy(),
            'position': position.copy(),
            'time': time
        }
        
        self.field_history.append(history_entry)
        
        # Limit history size for memory efficiency
        if len(self.field_history) > self.field_memory_depth:
            self.field_history.pop(0)
    
    def _calculate_diffusion_correction(self, position: np.ndarray, time: float, 
                                      diffusivity: float) -> float:
        """Calculate magnetic diffusion correction based on field evolution."""
        if len(self.field_history) < 3:
            return 0.0
        
        # Use recent history for gradient calculation
        recent_history = self.field_history[-3:]
        
        # Calculate temporal derivative ∂B/∂t
        dt = recent_history[-1]['time'] - recent_history[0]['time']
        if dt <= 0:
            return 0.0
        
        dB_dt = (np.linalg.norm(recent_history[-1]['field']) - 
                np.linalg.norm(recent_history[0]['field'])) / dt
        
        # Estimate spatial scale for ∇²B
        spatial_scale = 0.01  # 1 cm characteristic length scale
        laplacian_B = dB_dt / diffusivity  # From diffusion equation ∂B/∂t = η∇²B
        
        # Diffusion correction based on characteristic time
        characteristic_time = spatial_scale**2 / diffusivity
        diffusion_factor = self.diffusion_time_constant / characteristic_time * laplacian_B
        
        # Limit correction magnitude (max ±5% correction)
        return NumericalUtils.clamp(diffusion_factor, -0.05, 0.05)
    
    def apply_thermal_magnetic_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                         temperature: float, material_name: str = 'Low_Carbon_Steel') -> np.ndarray:
        """
        Apply thermal-magnetic coupling corrections using proper Curie-Weiss law.
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            temperature: Temperature (K)
            material_name: Material for thermal properties
            
        Returns:
            Thermally corrected magnetic field vector (T)
        """
        if not self.include_thermal_effects or temperature <= 0:
            return B_field
        
        # Get material-specific Curie temperature
        curie_temp = self._get_curie_temperature(material_name)
        
        # Temperature-dependent permeability using proper Curie-Weiss law
        temp_factor = self._calculate_curie_weiss_factor(temperature, curie_temp)
        
        # Magnetocaloric effect
        magnetocaloric_correction = self._calculate_magnetocaloric_correction(
            B_field, temperature, material_name)
        
        # Apply corrections
        correction_factor = temp_factor * (1 + magnetocaloric_correction)
        
        return B_field * correction_factor
    
    def _get_curie_temperature(self, material_name: str) -> float:
        """Get Curie temperature from material properties."""
        if self.materials is None:
            return 1043.0  # Iron default
        
        return self.materials.get_material_property(material_name, 'curie_temperature', 1043.0)
    
    def _calculate_curie_weiss_factor(self, temperature: float, curie_temp: float) -> float:
        """Calculate temperature-dependent permeability using Curie-Weiss law."""
        if temperature < curie_temp:
            # Below Curie temperature - ferromagnetic
            # Proper Curie-Weiss law: χ = C/(T - θ) where θ is Weiss temperature
            # For simplicity, use θ ≈ Tc and χ ∝ μ_r - 1
            
            # Avoid singularity at Curie temperature
            temp_diff = max(temperature - curie_temp, -curie_temp * 0.9)
            
            # Curie-Weiss behavior
            factor = curie_temp / (curie_temp - temp_diff)
            
            # Limit factor to reasonable range
            return NumericalUtils.clamp(factor, 0.1, 10.0)
        else:
            # Above Curie temperature - paramagnetic
            # μ_r ≈ 1 + C/T where C is Curie constant
            curie_constant = 1.0  # Typical value
            factor = 1.0 + curie_constant / temperature
            
            return max(factor, 1.0)  # Ensure μ_r ≥ 1
    
    def _calculate_magnetocaloric_correction(self, B_field: np.ndarray, temperature: float, 
                                           material_name: str) -> float:
        """Calculate magnetocaloric effect using proper thermodynamic relations."""
        B_magnitude = np.linalg.norm(B_field)
        
        if B_magnitude < 0.1 or temperature <= 0:
            return 0.0
        
        # Get material properties for proper calculation
        if self.materials is not None:
            try:
                density = self.materials.get_material_property(material_name, 'density', 7850.0)
                specific_heat = self.materials.get_material_property(material_name, 'specific_heat', 450.0)
            except:
                density, specific_heat = 7850.0, 450.0  # Steel defaults
        else:
            density, specific_heat = 7850.0, 450.0
        
        # Magnetocaloric temperature change using proper thermodynamics
        # ΔT = -T * (∂M/∂T)_H * ΔB / (ρ * c_p)
        # For ferromagnetic materials: (∂M/∂T)_H ≈ -M/T near Curie point
        
        # Simplified but physically motivated model
        magnetization_temp_coeff = -1e3  # A/(m·K) - typical for iron
        delta_T = temperature * magnetization_temp_coeff * B_magnitude / (density * specific_heat)
        
        # Temperature change affects permeability
        curie_temp = self._get_curie_temperature(material_name)
        temp_correction = (self._calculate_curie_weiss_factor(float(temperature + delta_T), curie_temp) - 
                          self._calculate_curie_weiss_factor(temperature, curie_temp))
        
        # Limit correction
        return NumericalUtils.clamp(temp_correction, -0.1, 0.1)
    
    def apply_piezomagnetic_corrections(self, B_field: np.ndarray, stress_tensor: Optional[np.ndarray], 
                                      position: np.ndarray) -> np.ndarray:
        """
        Apply piezomagnetic corrections due to mechanical stress.
        
        Uses proper piezomagnetic tensor relations: ΔB = d·σ
        
        Args:
            B_field: Magnetic field vector (T)
            stress_tensor: Stress tensor (Pa) - can be None
            position: Position vector (m)
            
        Returns:
            Stress-corrected magnetic field vector (T)
        """
        if not self.include_piezomagnetic or stress_tensor is None:
            return B_field
        
        # Piezomagnetic coefficient (material-dependent)
        piezomagnetic_d = 1e-9  # T/Pa - realistic value for iron
        
        # Calculate relevant stress invariants
        hydrostatic_stress = np.trace(stress_tensor) / 3.0
        deviatoric_stress = stress_tensor - hydrostatic_stress * np.eye(3)
        von_mises_stress = np.sqrt(1.5 * np.sum(deviatoric_stress**2))
        
        # Piezomagnetic correction (simplified isotropic model)
        # In reality, piezomagnetic tensor is anisotropic and material-dependent
        correction_factor = 1.0 + piezomagnetic_d * von_mises_stress
        
        return B_field * correction_factor
    
    def apply_nonequilibrium_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                       velocity: float, temperature: float, 
                                       material_name: str = 'Low_Carbon_Steel') -> np.ndarray:
        """
        Apply non-equilibrium magnetodynamics corrections using simplified LLG dynamics.
        
        Based on Landau-Lifshitz-Gilbert equation for magnetization dynamics.
        
        Args:
            B_field: Magnetic field vector (T)
            position: Position vector (m)
            velocity: Velocity (m/s)
            temperature: Temperature (K)
            material_name: Material for magnetic properties
            
        Returns:
            Non-equilibrium corrected magnetic field vector (T)
        """
        if not self.include_nonequilibrium:
            return B_field
        
        B_magnitude = np.linalg.norm(B_field)
        
        if B_magnitude < 0.01:  # Weak field
            return B_field
        
        # Magnetic relaxation time from material properties
        relaxation_time = self._calculate_magnetic_relaxation_time(material_name, temperature)
        
        # Characteristic velocity for magnetic domain switching
        switching_velocity = 1000.0  # m/s - typical for domain wall motion
        
        # Non-equilibrium correction based on velocity and relaxation time
        velocity_factor = abs(velocity) / switching_velocity
        relaxation_factor = self.diffusion_time_constant / relaxation_time
        
        # Simplified non-equilibrium correction
        correction = 0.01 * velocity_factor * relaxation_factor  # Max 1% correction
        
        # Apply correction with proper bounds
        correction_factor = 1.0 + NumericalUtils.clamp(correction, -0.02, 0.02)
        
        return B_field * correction_factor
    
    def _calculate_magnetic_relaxation_time(self, material_name: str, temperature: float) -> float:
        """Calculate magnetic relaxation time using thermal activation model."""
        # Get magnetic anisotropy energy
        if self.materials is not None:
            try:
                anisotropy_K = self.materials.get_material_property(material_name, 'anisotropy_constant_K1', 48000.0)
            except:
                anisotropy_K = 48000.0  # J/m³ for iron
        else:
            anisotropy_K = 48000.0
        
        # Thermal activation model: τ = τ₀ * exp(KV/kT)
        # where V is magnetic volume, typically ~ 1e-21 m³ for single domain
        
        tau_0 = 1e-9  # s - attempt time
        magnetic_volume = 1e-21  # m³
        
        if temperature > 0:
            thermal_energy = PhysicsConstants.K_BOLTZMANN * temperature
            activation_energy = anisotropy_K * magnetic_volume
            
            # Limit exponential to avoid overflow
            exponent = min(activation_energy / thermal_energy, 50.0)
            relaxation_time = tau_0 * np.exp(exponent)
        else:
            relaxation_time = 1e-6  # Default value
        
        return max(relaxation_time, 1e-12)  # Minimum relaxation time
    
    def calculate_comprehensive_field_corrections(self, B_field: np.ndarray, position: np.ndarray, 
                                                current: float, velocity: float, temperature: float,
                                                material_name: str = 'Low_Carbon_Steel',
                                                stress_tensor: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate field with all enabled corrections applied in proper order.
        
        This is the main method for applying comprehensive field corrections.
        Corrections are applied in order of physical significance.
        """
        corrected_field = B_field.copy()
        
        # 1. Thermal corrections (most significant for coilguns)
        if self.include_thermal_effects:
            corrected_field = self.apply_thermal_magnetic_corrections(
                corrected_field, position, temperature, material_name)
        
        # 2. Magnetic diffusion (eddy currents)
        if self.include_magnetic_diffusion:
            corrected_field = self.apply_magnetic_diffusion_correction(
                corrected_field, position, 0.0, material_name)  # Time would need to be tracked externally
        
        # 3. Piezomagnetic effects (stress-induced)
        if self.include_piezomagnetic and stress_tensor is not None:
            corrected_field = self.apply_piezomagnetic_corrections(
                corrected_field, stress_tensor, position)
        
        # 4. Non-equilibrium magnetodynamics
        if self.include_nonequilibrium:
            corrected_field = self.apply_nonequilibrium_corrections(
                corrected_field, position, velocity, temperature, material_name)
        
        # 5. Relativistic corrections (only for extreme velocities)
        if self.include_relativistic:
            corrected_field = self.apply_relativistic_field_transform(
                corrected_field, velocity, 'lab')
        
        return corrected_field 