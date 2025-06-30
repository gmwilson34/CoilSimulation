"""
Eddy Current Force Calculations

This module implements eddy current modeling with skin depth variations
and frequency-dependent effects.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils
from .base import BaseElectromagneticForces


class EddyCurrentForces(BaseElectromagneticForces):
    """
    Advanced eddy current force calculations for moving conductors.
    
    Includes:
    - Skin depth calculations with frequency dependence
    - Eddy current resistance modeling
    - Power dissipation calculations
    - Time-dependent current effects
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize eddy current forces calculator."""
        super().__init__(config, field_calculator, materials)
        
        # Eddy current parameters
        self.include_eddy_currents = config.get('advanced_physics', {}).get('include_eddy_currents', True)
        self.proj_conductivity = 1.0 / self.materials.get_material_property(self.proj_material, 'resistivity_20C')
        
        print(f"⚡ Eddy current forces initialized")
        print(f"   - Conductivity: {self.proj_conductivity:.2e} S/m")
    
    def calculate_eddy_current_force(self, current: float, position: float, velocity: float,
                                   current_history: Optional[List] = None,
                                   time_history: Optional[List] = None) -> Tuple[float, float]:
        """
        Calculate eddy current force and power dissipation.
        
        Returns:
            Tuple of (eddy_force, power_dissipation)
        """
        if not self.include_eddy_currents or abs(velocity) < 1e-6:
            return 0.0, 0.0
        
        # Get magnetic field
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Calculate frequency from current variation
        frequency = self._estimate_frequency(current_history, time_history)
        
        # Skin depth calculation
        skin_depth = self._calculate_skin_depth(frequency, B_field)
        
        # Effective resistance for eddy currents
        R_eddy = self._calculate_eddy_current_resistance(skin_depth, B_field)
        
        # Induced EMF due to motion
        induced_emf = velocity * B_field * self.proj_length
        
        # Eddy current magnitude
        I_eddy = induced_emf / R_eddy if R_eddy > 0 else 0.0
        
        # Force opposing motion (Lenz's law)
        eddy_force = -np.sign(velocity) * I_eddy * B_field * self.proj_length
        
        # Power dissipation
        power_dissipation = I_eddy**2 * R_eddy
        
        return (NumericalUtils.safe_numerical_operation(eddy_force, "eddy_force"),
                NumericalUtils.safe_numerical_operation(power_dissipation, "eddy_power"))
    
    def _estimate_frequency(self, current_history: Optional[List], 
                          time_history: Optional[List]) -> float:
        """
        CORRECTED: Sophisticated frequency estimation from current variation.
        
        Uses multiple methods to estimate the dominant frequency:
        1. Zero-crossing analysis
        2. Peak-finding in frequency domain
        3. RMS-based characteristic frequency
        """
        if not current_history or not time_history or len(current_history) < 5:
            return 1000.0  # Default frequency
        
        current_array = np.array(current_history)
        time_array = np.array(time_history)
        
        # Ensure uniform time spacing
        dt_mean = np.mean(np.diff(time_array))
        if dt_mean <= 0:
            return 1000.0
        
        # Method 1: Zero-crossing frequency estimation
        zero_crossings = 0
        for i in range(1, len(current_array)):
            if current_array[i] * current_array[i-1] < 0:  # Sign change
                zero_crossings += 1
        
        if zero_crossings > 1:
            total_time = time_array[-1] - time_array[0]
            zero_crossing_freq = zero_crossings / (2 * total_time)  # Half period per crossing
        else:
            zero_crossing_freq = 0.0
        
        # Method 2: Frequency domain analysis (simplified FFT approach)
        if len(current_array) >= 8:
            try:
                # Remove DC component
                current_ac = current_array - np.mean(current_array)
                
                # Simple power spectral analysis using autocorrelation
                # Find the lag with maximum correlation (excluding zero lag)
                correlations = np.correlate(current_ac, current_ac, mode='full')
                correlations = correlations[len(correlations)//2:]  # Take positive lags only
                
                if len(correlations) > 2:
                    # Find first significant peak after zero lag
                    max_correlation = np.max(correlations[1:])
                    if max_correlation > 0.1 * correlations[0]:  # 10% threshold
                        peak_lag = np.argmax(correlations[1:]) + 1
                        autocorr_freq = 1.0 / (peak_lag * dt_mean)
                    else:
                        autocorr_freq = 0.0
                else:
                    autocorr_freq = 0.0
                    
            except:
                autocorr_freq = 0.0
        else:
            autocorr_freq = 0.0
        
        # Method 3: RMS-based characteristic frequency
        # Estimate frequency from RMS current variation
        current_rms = np.sqrt(np.mean(current_ac**2)) if 'current_ac' in locals() else np.std(current_array)
        current_peak = np.max(np.abs(current_array))
        
        if current_peak > 0:
            # Characteristic frequency based on current variation rate
            dI_dt_rms = np.sqrt(np.mean(np.diff(current_array)**2)) / dt_mean
            rms_freq = dI_dt_rms / (2 * np.pi * current_rms) if current_rms > 0 else 0.0
        else:
            rms_freq = 0.0
        
        # Combine frequency estimates with weighted average
        frequencies = [zero_crossing_freq, autocorr_freq, rms_freq]
        weights = [0.4, 0.4, 0.2]  # Emphasize zero-crossing and autocorrelation
        
        # Filter out invalid frequencies
        valid_frequencies = []
        valid_weights = []
        for freq, weight in zip(frequencies, weights):
            if freq > 10.0 and freq < 1e6:  # Reasonable frequency range
                valid_frequencies.append(freq)
                valid_weights.append(weight)
        
        if valid_frequencies:
            # Weighted average of valid frequencies
            total_weight = sum(valid_weights)
            estimated_frequency = sum(f * w for f, w in zip(valid_frequencies, valid_weights)) / total_weight
        else:
            # Fallback: simple derivative-based estimate
            if len(current_array) >= 2:
                dI_dt = abs(current_array[-1] - current_array[-2]) / dt_mean
                estimated_frequency = dI_dt / (2 * np.pi * max(abs(current_array[-1]), 1.0))
            else:
                estimated_frequency = 1000.0
        
        # Clamp to reasonable range
        return max(min(estimated_frequency, 1e6), 100.0)  # 100 Hz to 1 MHz
    
    def _calculate_skin_depth(self, frequency: float, B_field: float) -> float:
        """
        Calculate skin depth with magnetic field dependence.
        
        δ = √(2/(ωμσ))
        """
        omega = 2 * np.pi * frequency
        
        # Effective permeability (field-dependent)
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
        mu_eff = self.permeability_model.calculate_nonlinear_permeability(H_applied, self.proj_material)
        mu_absolute = mu_eff * PhysicsConstants.MU_0
        
        # Skin depth
        skin_depth = np.sqrt(2.0 / (omega * mu_absolute * self.proj_conductivity))
        
        # Clamp to reasonable values
        min_skin_depth = self.proj_radius / 100  # At least 1% of radius
        max_skin_depth = self.proj_radius  # At most the full radius
        
        return np.clip(skin_depth, min_skin_depth, max_skin_depth)
    
    def _calculate_eddy_current_resistance(self, skin_depth: float, B_field: float) -> float:
        """
        Calculate effective resistance for eddy currents.
        
        Accounts for skin effect and geometry.
        """
        # Cross-sectional area affected by eddy currents
        if skin_depth >= self.proj_radius:
            # Uniform current distribution
            effective_area = np.pi * self.proj_radius**2
        else:
            # Current concentrated in skin depth
            effective_area = 2 * np.pi * self.proj_radius * skin_depth
        
        # Effective length for current path
        current_path_length = np.pi * self.proj_radius  # Circumferential path
        
        # Resistance calculation
        resistance = current_path_length / (self.proj_conductivity * effective_area)
        
        return max(resistance, 1e-12)  # Minimum resistance
    
    def calculate_mutual_inductance(self, position: float) -> float:
        """Calculate mutual inductance between coil and projectile."""
        # Simplified mutual inductance calculation
        # Based on overlap and geometry
        
        overlap_fraction = self._calculate_overlap_fraction(position)
        
        # Base mutual inductance (geometric factor)
        k_coupling = 0.1  # Typical coupling coefficient
        L_coil = self._solenoid_inductance_air_core()
        L_proj = self._calculate_projectile_self_inductance()
        
        M = k_coupling * np.sqrt(L_coil * L_proj) * overlap_fraction
        
        return M
    
    def _calculate_projectile_self_inductance(self) -> float:
        """Calculate self-inductance of projectile (treated as short solenoid)."""
        # Simplified self-inductance for cylindrical conductor
        # L = μ₀ * length / (8π) * ln(length/radius)
        
        if self.proj_length > 2 * self.proj_radius:
            # Long cylinder approximation
            L_self = (PhysicsConstants.MU_0 * self.proj_length / (8 * np.pi)) * \
                     np.log(self.proj_length / self.proj_radius)
        else:
            # Short cylinder - use geometric mean
            L_self = PhysicsConstants.MU_0 * self.proj_radius
        
        return max(L_self, 1e-12)
    
    def calculate_frequency_dependent_force(self, current: float, position: float, 
                                          velocity: float, frequency: float) -> Tuple[float, float]:
        """
        Calculate eddy current force with explicit frequency dependence.
        
        Useful for multi-frequency analysis or harmonic current drives.
        """
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Frequency-dependent skin depth
        skin_depth = self._calculate_skin_depth(frequency, B_field)
        
        # Frequency-dependent resistance
        R_eddy = self._calculate_eddy_current_resistance(skin_depth, B_field)
        
        # Frequency-dependent reactance
        omega = 2 * np.pi * frequency
        L_eddy = self._calculate_projectile_self_inductance()
        X_eddy = omega * L_eddy
        
        # Complex impedance
        Z_magnitude = np.sqrt(R_eddy**2 + X_eddy**2)
        
        # Motional EMF
        induced_emf = velocity * B_field * self.proj_length
        
        # Current with frequency dependence
        I_eddy = induced_emf / Z_magnitude if Z_magnitude > 0 else 0.0
        
        # Force (reduced by reactive component)
        phase_factor = R_eddy / Z_magnitude if Z_magnitude > 0 else 1.0
        eddy_force = -np.sign(velocity) * I_eddy * B_field * self.proj_length * phase_factor
        
        # Power dissipation (only resistive component)
        power_dissipation = I_eddy**2 * R_eddy * phase_factor**2
        
        return (NumericalUtils.safe_numerical_operation(eddy_force, "freq_eddy_force"),
                NumericalUtils.safe_numerical_operation(power_dissipation, "freq_eddy_power")) 