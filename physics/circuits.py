"""
Circuit Modeling and Dynamics

This module handles electrical circuit modeling including inductance calculations,
circuit dynamics, and energy analysis for coilgun systems.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import warnings
from scipy.special import ellipk, ellipe
from scipy.integrate import quad
from .core import BasePhysicsModel, PhysicsConstants, NumericalUtils, SafetyLimits
from .materials import AdvancedMaterialProperties


class CircuitModel(BasePhysicsModel):
    """
    Enhanced circuit modeling for coilgun electrical systems.
    
    Includes:
    - High-accuracy inductance calculations using elliptic integrals
    - Frequency-dependent resistance and inductance
    - Parasitic effects modeling
    - Energy analysis with loss accounting
    """
    
    def __init__(self, config: dict, materials: AdvancedMaterialProperties):
        """Initialize enhanced circuit model."""
        super().__init__(config)
        self.materials = materials
        
        # Extract circuit parameters
        cap_cfg = config.get('capacitor', {})
        self.capacitance = cap_cfg.get('capacitance', 0.001)  # F
        self.initial_voltage = cap_cfg.get('initial_voltage', 400)  # V
        self.esr = cap_cfg.get('esr', 0.01)  # Equivalent series resistance
        self.esl = cap_cfg.get('esl', 50e-9)  # Equivalent series inductance (nH)
        
        # Extract coil parameters
        coil_cfg = config.get('coil', {})
        self.total_turns = coil_cfg.get('total_turns', 1000)
        self.wire_awg = coil_cfg.get('wire_gauge_awg', 16)
        self.coil_length = coil_cfg.get('length', 0.05)
        self.coil_inner_radius = coil_cfg.get('inner_diameter', 0.02) / 2.0
        self.coil_outer_radius = coil_cfg.get('outer_diameter', 0.04) / 2.0
        self.num_layers = coil_cfg.get('num_layers', 1)
        self.turn_spacing = coil_cfg.get('turn_spacing', 'auto')
        
        # Enhanced parameters
        self.operating_frequency = config.get('circuit', {}).get('frequency', 1000)  # Hz
        self.temperature = config.get('environment', {}).get('temperature', 293.15)  # K
        
        # NEW: Ultra-high-current circuit effects
        self.max_design_current = config.get('circuit', {}).get('max_current', 1e6)  # A (1 MA)
        self.current_density_limit = 1e9  # A/m² (extreme current density limit)
        self.skin_effect_threshold = 1000  # Hz (when skin effect becomes significant)
        self.proximity_effect_coefficient = 0.15  # Inter-turn proximity factor
        
        # NEW: Distributed parameter modeling for extreme frequencies
        self.distributed_model_enabled = config.get('circuit', {}).get('distributed_model', True)
        self.transmission_line_impedance = 377  # Ohm (free space impedance)
        self.dielectric_constant = 1.0006  # Air dielectric (slight correction)
        
        # NEW: Electromagnetic transient effects
        self.transient_analysis_enabled = config.get('circuit', {}).get('transient_analysis', True)
        self.maxwell_displacement_current = config.get('circuit', {}).get('displacement_current', True)
        self.electromagnetic_wave_propagation = config.get('circuit', {}).get('wave_propagation', True)
        
        # NEW: Extreme condition parameters
        self.magnetic_energy_density_limit = 1e8  # J/m³ (extreme energy density)
        self.dielectric_breakdown_threshold = 3e6  # V/m (air breakdown)
        self.corona_discharge_threshold = 30e3  # V/m (corona inception)
        
        # NEW: Quantum circuit effects for extreme conditions
        self.quantum_flux_enabled = config.get('quantum_physics', {}).get('flux_quantization', False)
        self.josephson_junction_effects = config.get('quantum_physics', {}).get('josephson_effects', False)
        self.macroscopic_quantum_coherence = config.get('quantum_physics', {}).get('macroscopic_coherence', False)
        
        # CRITICAL ENHANCEMENT: Plasma physics in ultra-high-current conductors
        self.plasma_formation_threshold = 1e8  # A/m² (current density for plasma formation)
        self.pinch_effect_threshold = 1e6  # A (current for magnetic pinch effects)
        self.ohmic_heating_runaway_threshold = 1000  # K (temperature for thermal runaway)
        
        # CRITICAL ENHANCEMENT: Non-linear circuit effects
        self.parasitic_inductance_variation = config.get('circuit', {}).get('parasitic_variation', 0.1)  # 10% variation
        self.parasitic_capacitance_coil = 100e-12  # F (inter-turn capacitance)
        self.parasitic_resistance_variation = 0.05  # 5% variation with current
        
        # CRITICAL ENHANCEMENT: Multi-physics coupling in circuits
        self.thermal_circuit_coupling = True
        self.mechanical_circuit_coupling = True  # Conductor expansion affects inductance
        self.magnetic_circuit_coupling = True   # Field affects conductor properties
        
        # Calculate enhanced coil parameters
        self.coil_resistance_dc = self._calculate_coil_resistance_enhanced()
        self.coil_inductance_air = self._calculate_enhanced_air_core_inductance()
        
        # Frequency-dependent parameters
        self._calculate_frequency_dependent_parameters()
        
        # Total circuit parameters
        self.total_resistance_dc = self.coil_resistance_dc + self.esr
        
        # Validate parameters
        self._validate_circuit_parameters()
        
        print(f"🔬 Enhanced circuit model initialized")
        print(f"   - Coil: L = {self.coil_inductance_air*1e6:.1f} μH, R = {self.coil_resistance_dc*1e3:.1f} mΩ")
        print(f"   - Capacitor: C = {self.capacitance*1e3:.1f} mF, ESR = {self.esr*1e3:.1f} mΩ")
    
    def _calculate_coil_resistance_enhanced(self) -> float:
        """Calculate enhanced coil resistance with temperature and frequency effects."""
        # Get wire properties
        wire_area = self.materials.get_wire_area(self.wire_awg)
        resistivity = self.materials.get_temperature_dependent_property(
            'Copper', 'resistivity_20C', self.temperature
        )
        
        # CORRECTED: Enhanced wire path geometry calculation
        if self.num_layers > 1:
            # Multi-layer coil with realistic helical winding path
            total_wire_length = 0.0
            layer_thickness = (self.coil_outer_radius - self.coil_inner_radius) / self.num_layers
            turns_per_layer = self.total_turns / self.num_layers
            
            # Account for wire diameter in layer spacing
            wire_diameter = self.materials.get_wire_diameter(self.wire_awg)
            effective_layer_thickness = max(layer_thickness, wire_diameter * 1.1)  # 10% spacing
            
            for layer in range(self.num_layers):
                # CORRECTED: More accurate layer radius calculation
                # Account for actual wire position in the layer
                layer_radius = self.coil_inner_radius + (layer * effective_layer_thickness) + wire_diameter/2
                
                # CORRECTED: Improved helical pitch calculation
                # Account for layer-to-layer transitions and realistic winding
                if layer == 0:
                    # First layer: uniform pitch
                    axial_pitch = self.coil_length / turns_per_layer
                else:
                    # Subsequent layers: account for layer transition wire length
                    effective_turns = turns_per_layer + 0.5  # Add half turn for layer transition
                    axial_pitch = self.coil_length / effective_turns
                
                # Helical path length per turn in this layer
                circumference = 2 * np.pi * layer_radius
                helix_turn_length = np.sqrt(circumference**2 + axial_pitch**2)
                
                # Add length for this layer (including layer transitions)
                layer_wire_length = helix_turn_length * turns_per_layer
                
                # Add transition length between layers (except for last layer)
                if layer < self.num_layers - 1:
                    # Transition between layers - typically one pitch worth of extra wire
                    transition_length = np.sqrt((2 * np.pi * layer_radius)**2 + effective_layer_thickness**2)
                    layer_wire_length += transition_length
                
                total_wire_length += layer_wire_length
                
        else:
            # Single layer coil - enhanced calculation
            # Account for lead-in and lead-out wire
            avg_turn_radius = (self.coil_inner_radius + self.coil_outer_radius) / 2
            axial_pitch = self.coil_length / self.total_turns
            
            # Main helical winding
            circumference = 2 * np.pi * avg_turn_radius
            helix_turn_length = np.sqrt(circumference**2 + axial_pitch**2)
            main_wire_length = helix_turn_length * self.total_turns
            
            # Add lead wire estimates (typically 2-3 turn circumferences)
            lead_wire_length = 2.5 * circumference
            
            total_wire_length = main_wire_length + lead_wire_length
        
        # DC resistance calculation: R = ρL/A
        resistance_dc = resistivity * total_wire_length / wire_area
        
        return max(resistance_dc, SafetyLimits.MIN_RESISTANCE)
    
    def _calculate_enhanced_air_core_inductance(self) -> float:
        """
        Calculate air-core inductance using exact methods for maximum accuracy.
        
        Uses elliptic integral formulations for multilayer solenoids.
        """
        if self.num_layers == 1:
            # Single layer solenoid - use exact formula
            return self._single_layer_inductance_exact()
        else:
            # Multi-layer solenoid - integration over layers
            return self._multilayer_inductance_exact()
    
    def _single_layer_inductance_exact(self) -> float:
        """Exact single-layer inductance using elliptic integrals."""
        mu_0 = PhysicsConstants.MU_0
        N = self.total_turns
        a = self.coil_inner_radius
        l = self.coil_length
        
        # Exact formula using elliptic integrals for finite solenoid
        # L = μ₀N²a ∫ K(k) dk over the solenoid geometry
        
        # For a finite solenoid, the exact solution involves:
        aspect_ratio = l / (2 * a)
        
        if aspect_ratio > 5:  # CORRECTED: More conservative threshold
            # Long solenoid approximation (accurate for l >> 2a)
            # CORRECTED formula: L = μ₀N²πa²/l
            L = mu_0 * N**2 * np.pi * a**2 / l
        else:
            # Exact calculation using Nagaoka's coefficient
            K_nagaoka = self._calculate_nagaoka_coefficient(aspect_ratio)
            L = mu_0 * N**2 * a * K_nagaoka
        
        return max(L, SafetyLimits.MIN_INDUCTANCE)
    
    def _calculate_nagaoka_coefficient(self, beta: float) -> float:
        """
        Calculate Nagaoka's coefficient for exact inductance of finite solenoid.
        
        Nagaoka's coefficient K relates the inductance of a finite solenoid to that
        of an infinite solenoid: L = μ₀n²V × K, where V is coil volume.
        
        Args:
            beta: Aspect ratio l/(2a) where l is length, a is radius
            
        Returns:
            Nagaoka coefficient K
        """
        # CORRECTED: Complete Nagaoka's formula using elliptic integrals
        # Full formula: K = (8β/3) × [(2-k²)K(k²) - 2E(k²)] / k²
        # where k² = 4β²/(1+4β²) and β = l/(2a)
        
        if beta < 1e-12:
            # Limiting case for very short coils
            return 0.0
        
        k_squared = 4 * beta**2 / (1 + 4 * beta**2)
        
        if k_squared < 1e-8:
            # CORRECTED: Series expansion for small k (short coils)
            # K ≈ π²β/4 × [1 - k²/8 + 9k⁴/128 - 225k⁶/8192 + ...]
            k2 = k_squared
            K_nagaoka = (np.pi**2 * beta / 4) * (1 - k2/8 + 9*k2**2/128 - 225*k2**3/8192)
        elif beta > 50:
            # CORRECTED: Long solenoid approximation (β >> 1)
            # K ≈ 1 - 1/(2β) + 1/(8β²) - 1/(16β³) + ...
            inv_beta = 1.0 / beta
            K_nagaoka = 1.0 - 0.5*inv_beta + 0.125*inv_beta**2 - 0.0625*inv_beta**3
        else:
            try:
                # CORRECTED: Full elliptic integral calculation
                K_elliptic = ellipk(k_squared)
                E_elliptic = ellipe(k_squared)
                
                # Nagaoka's exact coefficient
                if k_squared > 1e-12:
                    # K = (8β/3) × [(2-k²)K(k²) - 2E(k²)] / k²
                    elliptic_factor = (2 - k_squared) * K_elliptic - 2 * E_elliptic
                    K_nagaoka = (8 * beta / 3) * elliptic_factor / k_squared
                else:
                    # Limiting case as k² → 0
                    K_nagaoka = np.pi**2 * beta / 4
                    
            except Exception as e:
                warnings.warn(f"Elliptic integral calculation failed: {e}, using approximation")
                # Fallback to empirical approximation (Rosa's formula)
                # K ≈ 1 / (1 + 0.9β + 0.1β²)  (approximate but stable)
                K_nagaoka = 1.0 / (1 + 0.9/beta + 0.1/beta**2) if beta > 0 else 0.0
        
        return max(K_nagaoka, 0.0)
    
    def _multilayer_inductance_exact(self) -> float:
        """
        CORRECTED: Calculate exact multi-layer inductance using layer integration.
        """
        mu_0 = PhysicsConstants.MU_0
        layer_thickness = (self.coil_outer_radius - self.coil_inner_radius) / self.num_layers
        turns_per_layer = self.total_turns / self.num_layers
        
        # Self-inductance of each layer
        total_self_inductance = 0.0
        
        # Mutual inductance between layers  
        total_mutual_inductance = 0.0
        
        for i in range(self.num_layers):
            # Radius of layer i
            r_i = self.coil_inner_radius + (i + 0.5) * layer_thickness
            
            # Self-inductance of layer i (using single-layer formula)
            aspect_ratio_i = self.coil_length / (2 * r_i)
            if aspect_ratio_i > 5:
                L_self_i = mu_0 * turns_per_layer**2 * np.pi * r_i**2 / self.coil_length
            else:
                K_nagaoka_i = self._calculate_nagaoka_coefficient(aspect_ratio_i)
                L_self_i = mu_0 * turns_per_layer**2 * r_i * K_nagaoka_i
            
            total_self_inductance += L_self_i
            
            # Mutual inductance with other layers
            for j in range(i + 1, self.num_layers):
                r_j = self.coil_inner_radius + (j + 0.5) * layer_thickness
                M_ij = self._calculate_mutual_inductance_layers(r_i, r_j, turns_per_layer)
                total_mutual_inductance += 2 * M_ij  # Factor of 2 for symmetry
        
        total_inductance = total_self_inductance + total_mutual_inductance
        return max(total_inductance, SafetyLimits.MIN_INDUCTANCE)
    
    def _calculate_mutual_inductance_layers(self, r1: float, r2: float, turns_per_layer: float) -> float:
        """
        Calculate mutual inductance between two coaxial layers.
        
        Uses Neumann's formula with elliptic integrals.
        """
        # CORRECTED: Simplified mutual inductance for coaxial circular loops
        # M = μ₀√(r₁r₂) * [(2-k²)K(k) - 2E(k)]
        # where k² = 4r₁r₂/[(r₁+r₂)² + z²], z = 0 for coaxial
        
        k_squared = 4 * r1 * r2 / (r1 + r2)**2
        
        if k_squared >= 1.0:
            k_squared = 0.99999999
        
        try:
            K = ellipk(k_squared)
            E = ellipe(k_squared)
            
            # CORRECTED: Mutual inductance between single turns (Neumann formula)
            # M = μ₀√(r₁r₂) * [(2-k²)K(k) - 2E(k)] for k² = 4r₁r₂/(r₁+r₂)²
            M_single = PhysicsConstants.MU_0 * np.sqrt(r1 * r2) * ((2 - k_squared) * K - 2 * E)
            
            # CORRECTED: Scale by number of turns - each layer interacts with each turn in other layer
            # For two layers each with N turns: M_total = N² * M_single_turn
            M_total = M_single * turns_per_layer**2
            
            return M_total
            
        except:
            # Fallback approximation
            return 0.1 * PhysicsConstants.MU_0 * np.sqrt(r1 * r2) * turns_per_layer**2
    
    def _calculate_frequency_dependent_parameters(self):
        """Calculate frequency-dependent circuit parameters."""
        # Skin depth for AC resistance
        omega = 2 * np.pi * self.operating_frequency
        resistivity = self.materials.get_temperature_dependent_property('Copper', 'resistivity_20C')
        mu = PhysicsConstants.MU_0  # Copper is non-magnetic
        
        self.skin_depth = np.sqrt(2 * resistivity / (omega * mu))
        
        # Wire radius
        wire_diameter = self.materials.get_wire_diameter(self.wire_awg)
        wire_radius = wire_diameter / 2
        
        # AC resistance factor
        if self.skin_depth < wire_radius:
            # Skin effect significant
            self.ac_resistance_factor = wire_radius / self.skin_depth
        else:
            # DC resistance dominates
            self.ac_resistance_factor = 1.0
        
        print(f"🔬 Skin depth: {self.skin_depth*1e6:.1f} μm, AC factor: {self.ac_resistance_factor:.2f}")
    
    def _validate_circuit_parameters(self):
        """Validate circuit parameters."""
        if self.capacitance <= 0:
            raise ValueError("Capacitance must be positive")
        if self.initial_voltage <= 0:
            raise ValueError("Initial voltage must be positive")
        if self.total_resistance_dc <= 0:
            raise ValueError("Total resistance must be positive")
    
    def calculate_inductance_with_core(self, position: float, mu_eff: float, 
                                     overlap_fraction: float) -> float:
        """
        Calculate inductance with ferromagnetic core present.
        
        Args:
            position: Projectile position (m)
            mu_eff: Effective permeability
            overlap_fraction: Fraction of projectile overlapping with coil
            
        Returns:
            Total inductance (H)
        """
        L_air = self.coil_inductance_air
        
        if overlap_fraction > 0:
            # Calculate inductance enhancement due to ferromagnetic core
            # Simplified model: L_total = L_air * (1 + (μ_eff - 1) * fill_factor)
            proj_volume_fraction = overlap_fraction * 0.1  # Simplified geometric factor
            L_enhancement = L_air * (mu_eff - 1.0) * proj_volume_fraction
            L_total = L_air + L_enhancement
        else:
            L_total = L_air
        
        return max(L_total, SafetyLimits.MIN_INDUCTANCE)
    
    def calculate_circuit_energy(self, current: float, voltage: float) -> Tuple[float, float, float]:
        """
        Calculate circuit energy components.
        
        Args:
            current: Current (A)
            voltage: Capacitor voltage (V)
            
        Returns:
            Tuple of (capacitor_energy, magnetic_energy, total_energy) in Joules
        """
        # Capacitor energy
        E_cap = 0.5 * self.capacitance * voltage**2
        
        # Magnetic energy (requires inductance)
        L = self.coil_inductance_air  # Simplified - could be position-dependent
        E_mag = 0.5 * L * current**2
        
        # Total energy
        E_total = E_cap + E_mag
        
        return E_cap, E_mag, E_total
    
    def calculate_power_dissipation(self, current: float) -> float:
        """
        Calculate resistive power dissipation.
        
        Args:
            current: Current (A)
            
        Returns:
            Power dissipation (W)
        """
        power = current**2 * self.total_resistance_dc
        return NumericalUtils.safe_numerical_operation(power, "power_dissipation", SafetyLimits.MAX_POWER)
    
    def get_initial_conditions(self) -> Tuple[float, float]:
        """
        Get initial conditions for circuit simulation.
        
        Returns:
            Tuple of (initial_charge, initial_current)
        """
        Q0 = self.capacitance * self.initial_voltage
        I0 = 0.0  # Start with zero current
        
        return Q0, I0
    
    def get_effective_resistance(self) -> float:
        """
        Get effective circuit resistance.
        
        Returns:
            Total effective resistance (Ohms)
        """
        return self.total_resistance_dc
    
    def calculate_time_constant(self, inductance: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate circuit time constants.
        
        Args:
            inductance: Inductance value (uses air-core if None)
            
        Returns:
            Tuple of (RC_time_constant, LC_resonant_period)
        """
        if inductance is None:
            inductance = self.coil_inductance_air
        
        # RC time constant
        tau_RC = self.total_resistance_dc * self.capacitance
        
        # LC resonant period
        T_LC = 2 * np.pi * np.sqrt(inductance * self.capacitance)
        
        return tau_RC, T_LC
    
    def _calculate_ultra_high_current_effects(self, current: float) -> Tuple[float, float]:
        """Calculate resistance and inductance modifications for ultra-high currents."""
        # Current density in conductor
        wire_area = self.materials.get_wire_area(self.wire_awg)
        current_density = current / wire_area
        
        resistance_factor = 1.0
        inductance_factor = 1.0
        
        if current_density > self.current_density_limit:
            print(f"⚠️  Extreme current density: {current_density/1e9:.1f} GA/m²")
            
            # Magnetic pressure effects on conductor geometry
            magnetic_pressure = PhysicsConstants.MU_0 * current_density**2 / 2
            
            # Conductor expansion (magnetostriction + thermal + magnetic pressure)
            expansion_factor = 1.0 + magnetic_pressure / (self.materials.get_material_property('Copper', 'young_modulus'))
            
            # Modified conductor geometry affects resistance and inductance
            resistance_factor = 1.0 / expansion_factor  # Lower resistance due to larger cross-section
            inductance_factor = expansion_factor**0.5  # Inductance scales with geometry
        
        # Skin effect at high frequencies (from rapid current changes)
        frequency_effective = abs(current) / (1e-6 * self.max_design_current)  # Effective frequency from di/dt
        
        if frequency_effective > self.skin_effect_threshold:
            skin_depth = self._calculate_skin_depth_frequency(frequency_effective)
            wire_radius = np.sqrt(wire_area / np.pi)
            
            if skin_depth < wire_radius:
                # Significant skin effect - current flows only in outer layer
                effective_area = 2 * np.pi * wire_radius * skin_depth
                skin_factor = wire_area / effective_area
                resistance_factor *= skin_factor
                
                # Inductance is less affected by skin effect
                inductance_factor *= (1.0 + 0.1 * (1.0 - 1.0/skin_factor))
        
        return resistance_factor, inductance_factor
    
    def _calculate_skin_depth_frequency(self, frequency: float) -> float:
        """Calculate skin depth for given frequency."""
        conductivity = 1.0 / self.materials.get_material_property('Copper', 'resistivity_20C')
        mu_r = self.materials.get_material_property('Copper', 'mu_r')
        
        skin_depth = np.sqrt(2 / (2 * np.pi * frequency * PhysicsConstants.MU_0 * mu_r * conductivity))
        return skin_depth
    
    def _calculate_distributed_parameter_effects(self, frequency: float) -> Tuple[float, float]:
        """Calculate distributed parameter effects (transmission line behavior)."""
        if not self.distributed_model_enabled or frequency < 1e6:  # Below 1 MHz
            return 1.0, 1.0  # No significant distributed effects
        
        # Coil as transmission line - calculate characteristic impedance
        # For a solenoid: Z0 = sqrt(L/C) where L and C are per-unit-length values
        
        # Inductance per unit length
        L_per_length = self.coil_inductance_air / self.coil_length
        
        # Capacitance per unit length (inter-turn capacitance)
        wire_diameter = np.sqrt(4 * self.materials.get_wire_area(self.wire_awg) / np.pi)
        turn_spacing = self.coil_length / self.total_turns
        
        # Parallel plate capacitor approximation for adjacent turns
        epsilon_0 = PhysicsConstants.EPSILON_0
        C_per_length = epsilon_0 * self.dielectric_constant * wire_diameter / turn_spacing
        
        # Characteristic impedance
        Z_characteristic = np.sqrt(L_per_length / C_per_length)
        
        # Electrical length (coil length in wavelengths)
        wave_velocity = PhysicsConstants.C / np.sqrt(self.dielectric_constant)
        wavelength = wave_velocity / frequency
        electrical_length = self.coil_length / wavelength
        
        # Transmission line effects become significant when electrical_length > 0.1
        if electrical_length > 0.1:
            # Standing wave effects modify effective L and R
            beta = 2 * np.pi * electrical_length
            
            # Input impedance of open-circuited transmission line
            Z_input = Z_characteristic / (1j * np.tan(beta))
            
            # Extract real and imaginary parts for R and L modifications
            resistance_factor = np.real(Z_input) / self.total_resistance_dc
            inductance_factor = np.imag(Z_input) / (2 * np.pi * frequency * self.coil_inductance_air)
            
            print(f"📡 Distributed effects: λ/L = {1/electrical_length:.1f}, Z₀ = {Z_characteristic:.1f} Ω")
            
            return max(0.1, resistance_factor), max(0.1, inductance_factor)
        
        return 1.0, 1.0
    
    def _calculate_electromagnetic_transient_effects(self, current: float, di_dt: float) -> dict:
        """Calculate electromagnetic transient effects for rapid current changes."""
        transient_effects = {
            'displacement_current': 0.0,
            'wave_propagation_delay': 0.0,
            'electromagnetic_radiation': 0.0
        }
        
        if not self.transient_analysis_enabled:
            return transient_effects
        
        # Maxwell displacement current: I_d = ε₀ * dE/dt
        if self.maxwell_displacement_current and di_dt != 0:
            # Estimate electric field change from current change
            dV_dt = di_dt * self.total_resistance_dc  # Simple ohmic approximation
            electric_field_change = dV_dt / self.coil_length
            
            # Displacement current density
            displacement_current_density = PhysicsConstants.EPSILON_0 * electric_field_change
            
            # Total displacement current (over coil volume)
            coil_volume = np.pi * self.coil_inner_radius**2 * self.coil_length
            displacement_current = displacement_current_density * coil_volume
            
            transient_effects['displacement_current'] = displacement_current
        
        # Electromagnetic wave propagation delay
        if self.electromagnetic_wave_propagation:
            # Time for electromagnetic signal to propagate across coil
            propagation_distance = self.coil_length
            wave_speed = PhysicsConstants.C / np.sqrt(self.dielectric_constant)
            propagation_delay = propagation_distance / wave_speed
            
            transient_effects['wave_propagation_delay'] = propagation_delay
        
        # Electromagnetic radiation power (Larmor formula adaptation)
        if abs(di_dt) > 1e9:  # Extreme current change rate (GA/s)
            # Radiated power from accelerating charges in conductors
            # P = (μ₀q²a²)/(6πc) adapted for current loops
            
            acceleration_equivalent = di_dt / (self.total_turns * 2 * np.pi * self.coil_inner_radius)
            radiated_power = (PhysicsConstants.MU_0 * (1.602e-19)**2 * acceleration_equivalent**2) / \
                           (6 * np.pi * PhysicsConstants.C)
            
            # Scale by number of charge carriers
            carrier_density = 8.5e28  # electrons/m³ in copper
            total_volume = self.total_turns * 2 * np.pi * self.coil_inner_radius * \
                          self.materials.get_wire_area(self.wire_awg)
            total_carriers = carrier_density * total_volume
            
            total_radiated_power = radiated_power * total_carriers
            transient_effects['electromagnetic_radiation'] = total_radiated_power
        
        return transient_effects
    
    def _check_extreme_condition_limits(self, current: float, voltage: float) -> dict:
        """Check if circuit operates within extreme condition limits."""
        warnings = []
        
        # Magnetic energy density check
        magnetic_field = PhysicsConstants.MU_0 * current * self.total_turns / self.coil_length
        energy_density = magnetic_field**2 / (2 * PhysicsConstants.MU_0)
        
        if energy_density > self.magnetic_energy_density_limit:
            warnings.append(f"Extreme magnetic energy density: {energy_density/1e8:.1f} × 10⁸ J/m³")
        
        # Electric field breakdown check
        electric_field = voltage / self.coil_length
        
        if electric_field > self.corona_discharge_threshold:
            warnings.append(f"Corona discharge risk: E = {electric_field/1e3:.1f} kV/m")
        
        if electric_field > self.dielectric_breakdown_threshold:
            warnings.append(f"Dielectric breakdown risk: E = {electric_field/1e6:.1f} MV/m")
        
        # Current density check
        wire_area = self.materials.get_wire_area(self.wire_awg)
        current_density = current / wire_area
        
        if current_density > self.current_density_limit:
            warnings.append(f"Extreme current density: {current_density/1e9:.1f} GA/m²")
        
        return {
            'energy_density': energy_density,
            'electric_field': electric_field,
            'current_density': current_density,
            'warnings': warnings
        }

    def analyze_ultra_high_current_circuit_effects(self, current: float, di_dt: float,
                                                  temperature: float = None,
                                                  frequency: float = None) -> dict:
        """
        CRITICAL NEW METHOD: Analyze ultra-high-current circuit effects.
        
        Essential for extreme coilgun applications operating at mega-ampere levels.
        
        Analyzes:
        1. Magnetic pinch effects on conductor geometry
        2. Plasma formation in conductors  
        3. Ohmic heating and thermal runaway
        4. Skin effect and proximity effect variations
        5. Electromagnetic forces on circuit structure
        6. Distributed parameter effects
        7. Non-linear parasitic element variations
        """
        if temperature is None:
            temperature = self.temperature
        if frequency is None:
            frequency = self.operating_frequency
        
        analysis = {
            'current_level': current,
            'current_density': current / self.wire_area,
            'effects_summary': {},
            'safety_assessment': {},
            'performance_impact': {},
            'mitigation_required': []
        }
        
        # 1. MAGNETIC PINCH EFFECTS
        if current > self.pinch_effect_threshold:
            pinch_effects = self._analyze_magnetic_pinch_effects(current, temperature)
            analysis['effects_summary']['magnetic_pinch'] = pinch_effects
            
            if pinch_effects['conductor_deformation'] > 0.01:  # 1% deformation
                analysis['safety_assessment']['pinch_deformation_risk'] = 'HIGH'
                analysis['mitigation_required'].append('Structural reinforcement required')
        
        # 2. PLASMA FORMATION IN CONDUCTORS
        current_density = analysis['current_density']
        if current_density > self.plasma_formation_threshold:
            plasma_effects = self._analyze_conductor_plasma_formation(current, current_density, temperature)
            analysis['effects_summary']['plasma_formation'] = plasma_effects
            
            if plasma_effects['plasma_probability'] > 0.1:  # 10% chance
                analysis['safety_assessment']['plasma_risk'] = 'CRITICAL'
                analysis['mitigation_required'].append('Plasma suppression required')
        
        # 3. OHMIC HEATING AND THERMAL RUNAWAY
        thermal_effects = self._analyze_ohmic_heating_runaway(current, temperature, di_dt)
        analysis['effects_summary']['thermal_runaway'] = thermal_effects
        
        if thermal_effects['runaway_risk']:
            analysis['safety_assessment']['thermal_runaway_risk'] = 'HIGH'
            analysis['mitigation_required'].append('Active cooling required')
        
        # 4. ENHANCED SKIN AND PROXIMITY EFFECTS
        skin_proximity_effects = self._analyze_enhanced_skin_proximity_effects(current, frequency, temperature)
        analysis['effects_summary']['skin_proximity'] = skin_proximity_effects
        analysis['performance_impact']['resistance_increase'] = skin_proximity_effects['resistance_multiplier']
        
        # 5. ELECTROMAGNETIC STRUCTURAL FORCES
        structural_forces = self._analyze_electromagnetic_structural_forces(current, di_dt)
        analysis['effects_summary']['structural_forces'] = structural_forces
        
        if structural_forces['max_stress'] > 100e6:  # 100 MPa
            analysis['safety_assessment']['structural_stress_risk'] = 'HIGH'
            analysis['mitigation_required'].append('Structural reinforcement required')
        
        # 6. DISTRIBUTED PARAMETER EFFECTS
        if self.distributed_model_enabled:
            distributed_effects = self._analyze_distributed_parameter_effects(current, frequency, di_dt)
            analysis['effects_summary']['distributed_parameters'] = distributed_effects
            analysis['performance_impact']['wave_propagation_delay'] = distributed_effects['propagation_delay']
        
        # 7. NON-LINEAR PARASITIC VARIATIONS
        parasitic_variations = self._analyze_nonlinear_parasitic_variations(current, temperature)
        analysis['effects_summary']['parasitic_variations'] = parasitic_variations
        analysis['performance_impact']['inductance_variation'] = parasitic_variations['inductance_change']
        analysis['performance_impact']['resistance_variation'] = parasitic_variations['resistance_change']
        
        # Overall risk assessment
        high_risks = sum(1 for assessment in analysis['safety_assessment'].values() 
                        if assessment in ['HIGH', 'CRITICAL'])
        
        if high_risks >= 3:
            analysis['overall_risk_level'] = 'CRITICAL'
        elif high_risks >= 1:
            analysis['overall_risk_level'] = 'HIGH'
        else:
            analysis['overall_risk_level'] = 'MODERATE'
        
        return analysis
    
    def _analyze_magnetic_pinch_effects(self, current: float, temperature: float) -> dict:
        """Analyze magnetic pinch effects on conductor geometry."""
        # Magnetic pressure: P_mag = B²/(2μ₀)
        B_field_surface = PhysicsConstants.MU_0 * current / (2 * np.pi * np.sqrt(self.wire_area / np.pi))
        magnetic_pressure = B_field_surface**2 / (2 * PhysicsConstants.MU_0)
        
        # Material yield strength at temperature
        yield_strength = 200e6  # Pa (typical for copper at room temperature)
        if temperature > PhysicsConstants.ROOM_TEMPERATURE:
            thermal_factor = max(0.1, 1.0 - (temperature - PhysicsConstants.ROOM_TEMPERATURE) / 1000)
            yield_strength *= thermal_factor
        
        # Conductor deformation estimate
        conductor_deformation = magnetic_pressure / yield_strength if yield_strength > 0 else 1.0
        
        # Pinch force per unit length: F = μ₀I²/(2πr) for parallel conductors
        conductor_radius = np.sqrt(self.wire_area / np.pi)
        pinch_force_per_length = PhysicsConstants.MU_0 * current**2 / (2 * np.pi * conductor_radius)
        
        return {
            'magnetic_pressure': magnetic_pressure,
            'conductor_deformation': conductor_deformation,
            'pinch_force_per_length': pinch_force_per_length,
            'surface_magnetic_field': B_field_surface,
            'deformation_risk': conductor_deformation > 0.005  # 0.5% threshold
        }
    
    def _analyze_conductor_plasma_formation(self, current: float, current_density: float, 
                                          temperature: float) -> dict:
        """Analyze plasma formation probability in conductors."""
        # Joule heating rate: P = J²ρ where J is current density
        resistivity = self.materials.get_temperature_dependent_property('Copper', 'resistivity_20C', temperature)
        joule_heating_rate = current_density**2 * resistivity  # W/m³
        
        # Estimate temperature rise rate
        copper_density = 8960  # kg/m³
        copper_specific_heat = 385  # J/(kg⋅K)
        thermal_mass_density = copper_density * copper_specific_heat
        
        temperature_rise_rate = joule_heating_rate / thermal_mass_density  # K/s
        
        # Plasma formation criteria (simplified)
        plasma_temperature_threshold = 5000  # K (copper plasma)
        time_to_plasma = (plasma_temperature_threshold - temperature) / temperature_rise_rate if temperature_rise_rate > 0 else np.inf
        
        # Plasma formation probability (empirical model)
        if current_density > self.plasma_formation_threshold:
            plasma_probability = min(1.0, (current_density / self.plasma_formation_threshold - 1.0) * 0.5)
        else:
            plasma_probability = 0.0
        
        return {
            'joule_heating_rate': joule_heating_rate,
            'temperature_rise_rate': temperature_rise_rate,
            'time_to_plasma': time_to_plasma,
            'plasma_probability': plasma_probability,
            'plasma_formation_risk': plasma_probability > 0.01
        }
    
    def _analyze_ohmic_heating_runaway(self, current: float, temperature: float, di_dt: float) -> dict:
        """Analyze thermal runaway risk from ohmic heating."""
        # Current thermal power generation
        resistance_current = self.coil_resistance_dc * (1 + 0.004 * (temperature - PhysicsConstants.ROOM_TEMPERATURE))
        power_generation = current**2 * resistance_current
        
        # Thermal capacity of coil
        coil_mass = self.total_turns * self.wire_area * 2 * np.pi * self.coil_inner_radius * 8960  # kg (copper)
        thermal_capacity = coil_mass * 385  # J/K (copper specific heat)
        
        # Heat dissipation (simplified convection + radiation)
        surface_area = 2 * np.pi * self.coil_inner_radius * self.coil_length * self.num_layers
        convection_coefficient = 25  # W/(m²⋅K) (natural convection)
        stefan_boltzmann = 5.67e-8  # W/(m²⋅K⁴)
        emissivity = 0.8  # Typical for oxidized copper
        
        heat_dissipation_convection = convection_coefficient * surface_area * (temperature - PhysicsConstants.ROOM_TEMPERATURE)
        heat_dissipation_radiation = stefan_boltzmann * emissivity * surface_area * (temperature**4 - PhysicsConstants.ROOM_TEMPERATURE**4)
        total_heat_dissipation = heat_dissipation_convection + heat_dissipation_radiation
        
        # Net heat accumulation
        net_heat_rate = power_generation - total_heat_dissipation
        temperature_rise_rate = net_heat_rate / thermal_capacity
        
        # Runaway condition: when generation exceeds dissipation capability
        runaway_risk = (net_heat_rate > 0) and (temperature > self.ohmic_heating_runaway_threshold)
        
        # Time to critical temperature (copper melting point: 1358 K)
        if temperature_rise_rate > 0:
            time_to_critical = (1358 - temperature) / temperature_rise_rate
        else:
            time_to_critical = np.inf
        
        return {
            'power_generation': power_generation,
            'total_heat_dissipation': total_heat_dissipation,
            'net_heat_rate': net_heat_rate,
            'temperature_rise_rate': temperature_rise_rate,
            'runaway_risk': runaway_risk,
            'time_to_critical_temperature': time_to_critical,
            'thermal_safety_margin': total_heat_dissipation / power_generation if power_generation > 0 else np.inf
        }
    
    def _analyze_enhanced_skin_proximity_effects(self, current: float, frequency: float, 
                                               temperature: float) -> dict:
        """Analyze enhanced skin and proximity effects at extreme currents."""
        # Basic skin depth calculation
        base_skin_depth = self._calculate_skin_depth_frequency(frequency)
        
        # Enhanced effects due to extreme current density
        current_density = current / self.wire_area
        
        # Non-linear skin depth reduction due to magnetic field saturation
        B_field = PhysicsConstants.MU_0 * current / (2 * np.pi * np.sqrt(self.wire_area / np.pi))
        if B_field > 1.0:  # Tesla - significant field
            skin_depth_reduction_factor = 1.0 / (1.0 + 0.1 * B_field)  # Empirical
        else:
            skin_depth_reduction_factor = 1.0
        
        effective_skin_depth = base_skin_depth * skin_depth_reduction_factor
        
        # Proximity effect enhancement
        wire_diameter = 2 * np.sqrt(self.wire_area / np.pi)
        proximity_factor = 1.0 + self.proximity_effect_coefficient * (current / 1e3)**0.5  # Empirical
        
        # Total resistance multiplier
        if effective_skin_depth < wire_diameter / 4:
            # Significant skin effect
            skin_effect_multiplier = (wire_diameter / 2) / effective_skin_depth
        else:
            skin_effect_multiplier = 1.0
        
        total_resistance_multiplier = skin_effect_multiplier * proximity_factor
        
        return {
            'base_skin_depth': base_skin_depth,
            'effective_skin_depth': effective_skin_depth,
            'skin_depth_reduction_factor': skin_depth_reduction_factor,
            'proximity_factor': proximity_factor,
            'skin_effect_multiplier': skin_effect_multiplier,
            'resistance_multiplier': total_resistance_multiplier,
            'significant_skin_effect': effective_skin_depth < wire_diameter / 4
        }
    
    def _analyze_electromagnetic_structural_forces(self, current: float, di_dt: float) -> dict:
        """Analyze electromagnetic forces on coil structure."""
        # Radial force on coil windings: F_r = B_z * I * l
        B_axial = PhysicsConstants.MU_0 * self.total_turns * current / self.coil_length  # Simplified
        radial_force_per_turn = B_axial * current * (2 * np.pi * self.coil_inner_radius)
        total_radial_force = radial_force_per_turn * self.total_turns
        
        # Axial force between turns: F_a = μ₀I²/(2πr) * ln(r_o/r_i)
        axial_force_per_turn = (PhysicsConstants.MU_0 * current**2) / (2 * np.pi) * \
                              np.log(self.coil_outer_radius / self.coil_inner_radius) if self.coil_outer_radius > self.coil_inner_radius else 0
        
        # Stress in coil structure
        coil_cross_sectional_area = (self.coil_outer_radius - self.coil_inner_radius) * self.coil_length
        radial_stress = total_radial_force / coil_cross_sectional_area if coil_cross_sectional_area > 0 else 0
        
        # Dynamic forces from di/dt
        if di_dt != 0:
            dynamic_force_factor = 1.0 + abs(di_dt) / 1e6  # Enhanced forces during current changes
            total_radial_force *= dynamic_force_factor
            radial_stress *= dynamic_force_factor
        
        return {
            'radial_force_per_turn': radial_force_per_turn,
            'total_radial_force': total_radial_force,
            'axial_force_per_turn': axial_force_per_turn,
            'radial_stress': radial_stress,
            'max_stress': radial_stress,  # Simplified - use radial as maximum
            'structural_safety_factor': 200e6 / radial_stress if radial_stress > 0 else np.inf  # Assume 200 MPa yield
        }
    
    def _analyze_distributed_parameter_effects(self, current: float, frequency: float, di_dt: float) -> dict:
        """Analyze distributed parameter effects in high-frequency operation."""
        # Transmission line parameters
        characteristic_impedance = np.sqrt(self.coil_inductance_air / (self.parasitic_capacitance_coil * self.coil_length))
        
        # Wave propagation velocity
        propagation_velocity = 1.0 / np.sqrt(self.coil_inductance_air * self.parasitic_capacitance_coil / self.coil_length)
        
        # Electrical length
        wavelength = propagation_velocity / frequency if frequency > 0 else np.inf
        electrical_length = self.coil_length / wavelength if wavelength != np.inf else 0
        
        # Propagation delay
        propagation_delay = self.coil_length / propagation_velocity
        
        # Standing wave effects
        if electrical_length > 0.1:  # 10% of wavelength
            standing_wave_ratio = 1.0 + 0.5 * electrical_length  # Simplified
            distributed_effects_significant = True
        else:
            standing_wave_ratio = 1.0
            distributed_effects_significant = False
        
        return {
            'characteristic_impedance': characteristic_impedance,
            'propagation_velocity': propagation_velocity,
            'electrical_length': electrical_length,
            'propagation_delay': propagation_delay,
            'standing_wave_ratio': standing_wave_ratio,
            'distributed_effects_significant': distributed_effects_significant
        }
    
    def _analyze_nonlinear_parasitic_variations(self, current: float, temperature: float) -> dict:
        """Analyze non-linear variations in parasitic circuit elements."""
        # Inductance variation due to core saturation and geometry changes
        # Thermal expansion effects
        thermal_expansion_factor = 1.0 + 12e-6 * (temperature - PhysicsConstants.ROOM_TEMPERATURE)  # Copper expansion
        inductance_thermal_change = (thermal_expansion_factor**2 - 1.0)  # Area scales as L²
        
        # Current-dependent inductance variation (simplified)
        current_normalized = current / 1e6  # Normalize to mega-ampere
        inductance_current_change = -self.parasitic_inductance_variation * current_normalized**2  # Saturation effects
        
        total_inductance_change = inductance_thermal_change + inductance_current_change
        
        # Resistance variation
        # Temperature coefficient
        resistance_thermal_change = 0.004 * (temperature - PhysicsConstants.ROOM_TEMPERATURE)
        
        # Current-dependent resistance (skin effect, proximity effect)
        resistance_current_change = self.parasitic_resistance_variation * current_normalized
        
        total_resistance_change = resistance_thermal_change + resistance_current_change
        
        # Capacitance variation (minimal for air-core)
        capacitance_change = 0.0001 * (temperature - PhysicsConstants.ROOM_TEMPERATURE) / PhysicsConstants.ROOM_TEMPERATURE
        
        return {
            'inductance_change': total_inductance_change,
            'inductance_thermal_component': inductance_thermal_change,
            'inductance_current_component': inductance_current_change,
            'resistance_change': total_resistance_change,
            'resistance_thermal_component': resistance_thermal_change,
            'resistance_current_component': resistance_current_change,
            'capacitance_change': capacitance_change
        }


class InductanceCalculator:
    """Advanced inductance calculations for coilgun systems."""
    
    def __init__(self, circuit_model: CircuitModel):
        """Initialize inductance calculator."""
        self.circuit = circuit_model
    
    def calculate_mutual_inductance(self, position: float) -> float:
        """
        Calculate mutual inductance between coil and projectile.
        
        Args:
            position: Projectile position (m)
            
        Returns:
            Mutual inductance (H)
        """
        # Simplified mutual inductance calculation
        # For a ferromagnetic projectile in a solenoid
        
        # Calculate overlap factor
        overlap = self._calculate_overlap_factor(position)
        
        if overlap > 0:
            # Base mutual inductance
            M_base = np.sqrt(self.circuit.coil_inductance_air * 1e-6)  # Simplified
            M = M_base * overlap
        else:
            M = 0.0
        
        return M
    
    def calculate_inductance_gradient(self, position: float, mu_eff: float, 
                                   delta: float = 1e-6) -> float:
        """
        Calculate inductance gradient dL/dz.
        
        Args:
            position: Position (m)
            mu_eff: Effective permeability
            delta: Step size for numerical differentiation
            
        Returns:
            Inductance gradient (H/m)
        """
        # Calculate overlap fractions
        overlap_plus = self._calculate_overlap_factor(position + delta)
        overlap_minus = self._calculate_overlap_factor(position - delta)
        
        # Calculate inductances
        L_plus = self.circuit.calculate_inductance_with_core(position + delta, mu_eff, overlap_plus)
        L_minus = self.circuit.calculate_inductance_with_core(position - delta, mu_eff, overlap_minus)
        
        # Numerical gradient
        dL_dz = (L_plus - L_minus) / (2.0 * delta)
        
        return dL_dz
    
    def _calculate_overlap_factor(self, position: float) -> float:
        """Calculate overlap factor between projectile and coil."""
        # Simplified overlap calculation
        coil_start = -self.circuit.coil_length / 2.0
        coil_end = self.circuit.coil_length / 2.0
        
        # Assume projectile length (would come from config in full implementation)
        proj_length = 0.01  # 1 cm default
        proj_start = position - proj_length / 2.0
        proj_end = position + proj_length / 2.0
        
        # Calculate overlap
        overlap_start = max(coil_start, proj_start)
        overlap_end = min(coil_end, proj_end)
        
        if overlap_end > overlap_start:
            overlap_length = overlap_end - overlap_start
            return overlap_length / proj_length
        else:
            return 0.0


class EnergyAnalyzer:
    """Energy analysis and conservation tracking for coilgun circuits."""
    
    def __init__(self, circuit_model: CircuitModel):
        """Initialize energy analyzer."""
        self.circuit = circuit_model
        self.initial_energy = None
        
    def initialize_energy_tracking(self):
        """Initialize energy tracking with initial conditions."""
        Q0, I0 = self.circuit.get_initial_conditions()
        V0 = Q0 / self.circuit.capacitance
        
        E_cap_initial, E_mag_initial, E_total_initial = self.circuit.calculate_circuit_energy(I0, V0)
        self.initial_energy = E_total_initial
        
        return self.initial_energy
    
    def calculate_energy_balance(self, current: float, voltage: float, 
                               kinetic_energy: float, 
                               cumulative_losses: float) -> dict:
        """
        Calculate energy balance and conservation.
        
        Args:
            current: Current (A)
            voltage: Capacitor voltage (V)
            kinetic_energy: Projectile kinetic energy (J)
            cumulative_losses: Cumulative energy losses (J)
            
        Returns:
            Dictionary with energy analysis
        """
        # Calculate current energy components
        E_cap, E_mag, E_circuit = self.circuit.calculate_circuit_energy(current, voltage)
        
        # Total energy accounting
        E_total_current = E_circuit + kinetic_energy + cumulative_losses
        
        # Energy conservation check
        if self.initial_energy is not None and self.initial_energy > 0:
            energy_error = (E_total_current - self.initial_energy) / self.initial_energy
        else:
            energy_error = 0.0
        
        return {
            'capacitor_energy': E_cap,
            'magnetic_energy': E_mag,
            'kinetic_energy': kinetic_energy,
            'cumulative_losses': cumulative_losses,
            'total_current': E_total_current,
            'initial_energy': self.initial_energy,
            'energy_error': energy_error,
            'energy_conservation_quality': 'good' if abs(energy_error) < 0.05 else 'poor'
        }
    
    def calculate_efficiency(self, final_kinetic_energy: float) -> float:
        """
        Calculate overall energy conversion efficiency.
        
        Args:
            final_kinetic_energy: Final kinetic energy of projectile (J)
            
        Returns:
            Efficiency (0 to 1)
        """
        if self.initial_energy is not None and self.initial_energy > 0:
            efficiency = final_kinetic_energy / self.initial_energy
        else:
            efficiency = 0.0
        
        return min(1.0, max(0.0, efficiency))  # Clamp between 0 and 1 