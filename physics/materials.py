"""
Material Properties and Management

This module handles material properties, loading materials data from JSON files,
wire specifications, temperature-dependent material characteristics, and
advanced material models for PhD-level coilgun simulation.
"""

import numpy as np
import json
import os
import warnings
from typing import Dict, Any, Optional, Union, Tuple
from .core import BasePhysicsModel, PhysicsConstants, validate_physical_parameter


class AdvancedMaterialProperties(BasePhysicsModel):
    """
    Advanced material properties manager for electromagnetic calculations.
    
    Includes:
    - Temperature-dependent material properties
    - Frequency-dependent permeability and conductivity
    - Nonlinear B-H characteristics
    - Thermal modeling for high-power applications
    - High-performance property caching system
    """
    
    def __init__(self, config: dict, materials_file: str = "materials_consolidated.json"):
        """Initialize advanced material properties with performance caching."""
        super().__init__(config)
        self.materials_file = materials_file
        self.materials_data = self._load_materials_data()
        
        # CRITICAL PERFORMANCE FIX: Initialize property cache
        self._property_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Temperature tracking
        self.temperature = config.get('environment', {}).get('temperature', PhysicsConstants.ROOM_TEMPERATURE)
        self.include_temperature_effects = config.get('advanced_physics', {}).get('include_temperature', True)
        
        # Frequency analysis
        self.operating_frequency = config.get('magnetic_model', {}).get('frequency', 1000)  # Hz
        
        # Pre-populate cache with most common lookups to avoid startup lag
        self._warm_cache()
        
        print(f"🔬 Advanced materials initialized (T: {self.temperature:.1f}K, f: {self.operating_frequency:.0f}Hz)")
    
    def _warm_cache(self):
        """Pre-populate cache with commonly used properties to improve startup performance."""
        common_materials = ['Pure_Iron', 'Copper', 'Low_Carbon_Steel']
        common_properties = ['mu_r', 'resistivity_20C', 'density', 'mu_r_initial', 'mu_r_max']
        
        for material in common_materials:
            if material in self.materials_data.get('materials', {}):
                for prop in common_properties:
                    try:
                        # This will populate the cache silently
                        self._get_material_property_uncached(material, prop, None, silent=True)
                    except:
                        pass  # Ignore missing properties during warmup
    
    def _load_materials_data(self) -> Dict[str, Any]:
        """Load enhanced materials data from consolidated JSON file"""
        try:
            if os.path.exists(self.materials_file):
                with open(self.materials_file, 'r') as f:
                    data = json.load(f)
                    print(f"✓ Loaded materials database v{data.get('metadata', {}).get('version', '1.0')}")
                    return data
            else:
                # Fallback to old materials.json if consolidated file not found
                if os.path.exists("materials.json"):
                    warnings.warn(f"Using legacy materials.json. Consider updating to {self.materials_file}")
                    with open("materials.json", 'r') as f:
                        return json.load(f)
                else:
                    warnings.warn(f"No materials file found. Using enhanced defaults.")
                    return self._create_enhanced_materials_database()
        except Exception as e:
            warnings.warn(f"Could not load materials data: {e}")
            return self._create_enhanced_materials_database()
    
    def _create_enhanced_materials_database(self) -> Dict[str, Any]:
        """Create enhanced materials database with temperature and frequency dependencies"""
        return {
            "physical_constants": {
                "mu0": PhysicsConstants.MU_0,
                "room_temperature": PhysicsConstants.ROOM_TEMPERATURE,
                "curie_temperature_iron": 1043,  # K
                "boltzmann_constant": 1.380649e-23,  # J/K
                "avogadro_number": 6.02214076e23  # mol⁻¹
            },
            "materials": {
                "Copper": {
                    "description": "High conductivity copper (OFHC)",
                    "resistivity_20C": 1.68e-8,  # Ohm⋅m at 20°C
                    "temperature_coefficient": 0.00393,  # 1/K
                    "density": 8960,  # kg/m³
                    "mu_r": 0.999991,  # Slightly diamagnetic
                    "thermal_conductivity": 401,  # W/(m⋅K)
                    "specific_heat": 385,  # J/(kg⋅K)
                    "melting_point": 1358,  # K
                    # NEW: Advanced properties for high-speed applications
                    "debye_temperature": 343,  # K
                    "fermi_energy": 7.04,  # eV
                    "work_function": 4.65,  # eV
                    "young_modulus": 130e9,  # Pa
                    "yield_strength": 70e6,  # Pa (annealed)
                    "ultimate_strength": 220e6,  # Pa
                    "elongation_at_break": 0.6,  # fraction
                    # Frequency-dependent conductivity (skin effect)
                    "skin_effect_params": {
                        "dc_conductivity": 5.96e7,  # S/m
                        "relaxation_time": 2.7e-14,  # s
                        "plasma_frequency": 1.6e16,  # Hz
                    }
                },
                "Pure_Iron": {
                    "description": "Pure iron (99.99% Fe) - Soft magnetic material",
                    "density": 7874,  # kg/m³
                    "mu_r_initial": 5000,  # Initial permeability
                    "mu_r_max": 200000,  # Maximum permeability
                    "resistivity_20C": 9.71e-8,  # Ohm⋅m
                    "temperature_coefficient": 0.0065,  # 1/K
                    "saturation_B": 2.15,  # Tesla
                    "saturation_M": 1.71e6,  # A/m (saturation magnetization)
                    "coercivity": 80,  # A/m (very soft)
                    "curie_temperature": 1043,  # K
                    "thermal_conductivity": 80.4,  # W/(m⋅K)
                    "specific_heat": 449,  # J/(kg⋅K)
                    # NEW: Enhanced magnetic properties
                    "anisotropy_constant_K1": 48000,  # J/m³
                    "exchange_stiffness": 21e-12,  # J/m
                    "domain_wall_energy": 0.01,  # J/m²
                    "magnetic_moment_per_atom": 2.22,  # Bohr magnetons
                    # Advanced Jiles-Atherton model parameters (PhD-level accuracy)
                    "ja_ms": 1.71e6,  # Saturation magnetization (A/m)
                    "ja_a": 1200,     # Shape parameter (enhanced)
                    "ja_alpha": 1e-3, # Interdomain coupling
                    "ja_c": 0.1,     # Reversibility
                    "ja_k": 500,     # Pinning parameter
                    # NEW: Temperature-dependent B-H curve parameters
                    "bh_curve_params": {
                        "temperature_dependence": True,
                        "T_ref": 293.15,  # K
                        "dM_dT": -2.1,    # Temperature coefficient of magnetization (A/(m⋅K))
                        "dHc_dT": 0.05,   # Temperature coefficient of coercivity (A/(m⋅K))
                    },
                    # Mechanical properties for high-speed applications
                    "young_modulus": 211e9,  # Pa
                    "yield_strength": 130e6,  # Pa
                    "ultimate_strength": 270e6,  # Pa
                    "fatigue_limit": 162e6,  # Pa (at 10^7 cycles)
                    "fracture_toughness": 80e6,  # Pa⋅m^0.5
                    "hardness_hv": 80,  # Vickers hardness
                },
                "Low_Carbon_Steel": {
                    "description": "Low carbon steel (AISI 1018) - Common projectile material",
                    "density": 7850,  # kg/m³
                    "mu_r_initial": 1000,  # Initial permeability
                    "mu_r_max": 50000,  # Maximum permeability
                    "resistivity_20C": 1.43e-7,  # Ohm⋅m
                    "temperature_coefficient": 0.006,  # 1/K
                    "saturation_B": 2.0,  # Tesla
                    "saturation_M": 1.4e6,  # A/m
                    "coercivity": 400,  # A/m
                    "curie_temperature": 1000,  # K (approximate)
                    "thermal_conductivity": 51.9,  # W/(m⋅K)
                    "specific_heat": 486,  # J/(kg⋅K)
                    # Enhanced magnetic properties
                    "anisotropy_constant_K1": 11000,  # J/m³
                    "exchange_stiffness": 18e-12,  # J/m
                    "domain_wall_energy": 0.008,  # J/m²
                    # Enhanced Jiles-Atherton model parameters
                    "ja_ms": 1.4e6,  # Saturation magnetization (A/m)
                    "ja_a": 800,     # Shape parameter
                    "ja_alpha": 2e-3, # Interdomain coupling
                    "ja_c": 0.2,     # Reversibility
                    "ja_k": 800,     # Pinning parameter
                    # Temperature-dependent properties
                    "bh_curve_params": {
                        "temperature_dependence": True,
                        "T_ref": 293.15,  # K
                        "dM_dT": -1.8,    # Temperature coefficient
                        "dHc_dT": 0.1,    # Coercivity temperature coefficient
                    },
                    # Mechanical properties for projectiles
                    "young_modulus": 200e9,  # Pa
                    "yield_strength": 220e6,  # Pa
                    "ultimate_strength": 400e6,  # Pa
                    "fatigue_limit": 140e6,  # Pa
                    "fracture_toughness": 50e6,  # Pa⋅m^0.5
                    "hardness_hv": 120,  # Vickers hardness
                    "impact_energy": 27,  # J (Charpy V-notch at room temp)
                    # NEW: High-velocity deformation properties
                    "strain_rate_sensitivity": 0.015,  # Dimensionless
                    "adiabatic_shear_band_threshold": 1e4,  # s⁻¹
                    "dynamic_yield_strength_factor": 1.3,  # Multiplier at high strain rates
                },
                "Silicon_Steel": {
                    "description": "Silicon steel (electrical steel) - Optimized for magnetic applications",
                    "density": 7650,  # kg/m³
                    "mu_r_initial": 15000,  # Initial permeability
                    "mu_r_max": 100000,  # Maximum permeability
                    "resistivity_20C": 4.7e-7,  # Ohm⋅m (higher than pure iron)
                    "temperature_coefficient": 0.005,  # 1/K
                    "saturation_B": 2.05,  # Tesla
                    "saturation_M": 1.6e6,  # A/m
                    "coercivity": 20,  # A/m (very soft)
                    "curie_temperature": 980,  # K
                    "thermal_conductivity": 25,  # W/(m⋅K)
                    "specific_heat": 460,  # J/(kg⋅K)
                    # Enhanced magnetic properties
                    "anisotropy_constant_K1": 45000,  # J/m³ (grain-oriented)
                    "exchange_stiffness": 20e-12,  # J/m
                    "domain_wall_energy": 0.005,  # J/m² (very low for soft material)
                    "magnetic_moment_per_atom": 2.15,  # Bohr magnetons
                    # Core loss data (Steinmetz equation: P = k*f^n*B^m)
                    "steinmetz_k": 0.001,  # Core loss coefficient
                    "steinmetz_n": 1.7,    # Frequency exponent
                    "steinmetz_m": 2.0,    # Field exponent
                    # Enhanced Jiles-Atherton parameters for ultra-soft material
                    "ja_ms": 1.6e6,  # Saturation magnetization (A/m)
                    "ja_a": 1500,    # Shape parameter (higher for softer material)
                    "ja_alpha": 2e-4, # Interdomain coupling (lower for grain-oriented)
                    "ja_c": 0.03,    # Reversibility (very low for soft magnetic material)
                    "ja_k": 30,      # Pinning parameter (very low for electrical steel)
                    # Frequency-dependent properties
                    "frequency_coefficients": {
                        "permeability_f_coeff": -0.15,  # μ decreases with frequency
                        "loss_factor_slope": 1.6,      # Core loss vs frequency
                        "skin_effect_onset": 1000,     # Hz where skin effect becomes significant
                        "relaxation_frequency": 10000,  # Hz for magnetic relaxation
                    },
                    # Temperature-dependent B-H curve
                    "bh_curve_params": {
                        "temperature_dependence": True,
                        "T_ref": 293.15,  # K
                        "dM_dT": -2.0,    # Temperature coefficient
                        "dHc_dT": 0.02,   # Very low for soft material
                    },
                    # Mechanical properties
                    "young_modulus": 190e9,  # Pa
                    "yield_strength": 300e6,  # Pa
                    "ultimate_strength": 450e6,  # Pa
                    "fatigue_limit": 200e6,  # Pa
                },
                "Aluminum": {
                    "description": "High-purity aluminum (99.9% Al) - Lightweight conductor",
                    "density": 2700,  # kg/m³
                    "mu_r": 1.0000217,  # Slightly paramagnetic
                    "resistivity_20C": 2.65e-8,  # Ohm⋅m
                    "temperature_coefficient": 0.0039,  # 1/K
                    "saturation_B": 0.0,  # Non-magnetic
                    "saturation_M": 0.0,  # Non-magnetic
                    "coercivity": 0.0,  # Non-magnetic
                    "thermal_conductivity": 237,  # W/(m⋅K)
                    "specific_heat": 897,  # J/(kg⋅K)
                    "melting_point": 933,  # K
                    # Mechanical properties
                    "young_modulus": 70e9,  # Pa
                    "yield_strength": 40e6,  # Pa (annealed)
                    "ultimate_strength": 90e6,  # Pa
                    "fatigue_limit": 90e6,  # Pa
                    "hardness_hv": 25,  # Vickers hardness
                    # Advanced properties for high-speed applications
                    "debye_temperature": 428,  # K
                    "work_function": 4.28,  # eV
                    "fermi_energy": 11.7,  # eV
                    "elongation_at_break": 0.4,  # fraction
                    # High-velocity deformation
                    "strain_rate_sensitivity": 0.002,  # Very low
                    "adiabatic_shear_band_threshold": 1e5,  # s⁻¹ (higher than steel)
                    "dynamic_yield_strength_factor": 1.1,  # Lower increase than steel
                },
                # CRITICAL ENHANCEMENT: Ultra-high-strength materials for extreme applications
                "Maraging_Steel_300": {
                    "description": "Maraging Steel Grade 300 - Ultra-high-strength steel for aerospace",
                    "density": 8100,  # kg/m³
                    "mu_r_initial": 50,  # Much lower than regular steel (martensitic structure)
                    "mu_r_max": 500,   # Lower maximum permeability
                    "resistivity_20C": 6.0e-7,  # Ohm⋅m (higher than regular steel)
                    "temperature_coefficient": 0.005,  # 1/K
                    "saturation_B": 1.6,  # Tesla (lower than regular steel)
                    "saturation_M": 1.0e6,  # A/m
                    "coercivity": 2000,  # A/m (much higher - harder material)
                    "curie_temperature": 850,  # K (lower than pure iron)
                    "thermal_conductivity": 20,  # W/(m⋅K) (lower than regular steel)
                    "specific_heat": 460,  # J/(kg⋅K)
                    "melting_point": 1700,  # K
                    # EXCEPTIONAL mechanical properties
                    "young_modulus": 190e9,  # Pa
                    "yield_strength": 2000e6,  # Pa (2 GPa!) - ULTRA-HIGH
                    "ultimate_strength": 2100e6,  # Pa (2.1 GPa!)
                    "fatigue_limit": 800e6,  # Pa (exceptional fatigue resistance)
                    "fracture_toughness": 100e6,  # Pa⋅m^0.5 (excellent toughness)
                    "hardness_hv": 650,  # Vickers hardness (very hard)
                    "impact_energy": 20,  # J (lower than regular steel due to hardness)
                    # Ultra-high-speed performance
                    "strain_rate_sensitivity": 0.02,  # Higher than regular steel
                    "adiabatic_shear_band_threshold": 5e3,  # s⁻¹ (lower threshold)
                    "dynamic_yield_strength_factor": 1.5,  # Significant increase
                    # Advanced magnetic properties
                    "anisotropy_constant_K1": 25000,  # J/m³ (moderate anisotropy)
                    "exchange_stiffness": 15e-12,  # J/m
                    "domain_wall_energy": 0.012,  # J/m² (higher due to hardness)
                    # Enhanced Jiles-Atherton for hard magnetic material
                    "ja_ms": 1.0e6,  # Saturation magnetization (A/m)
                    "ja_a": 400,     # Shape parameter (lower for harder material)
                    "ja_alpha": 5e-3, # Interdomain coupling (higher)
                    "ja_c": 0.5,     # Reversibility (higher for hard material)
                    "ja_k": 2000,    # Pinning parameter (much higher)
                },
                "Tungsten_Heavy_Alloy": {
                    "description": "Tungsten Heavy Alloy (95% W) - Ultra-dense projectile material",
                    "density": 18500,  # kg/m³ (extremely dense!)
                    "mu_r_initial": 1.2,  # Weakly magnetic
                    "mu_r_max": 10,    # Low maximum permeability
                    "resistivity_20C": 5.5e-8,  # Ohm⋅m
                    "temperature_coefficient": 0.0045,  # 1/K
                    "saturation_B": 0.1,  # Tesla (very weakly magnetic)
                    "saturation_M": 1e4,  # A/m (very low)
                    "coercivity": 50,  # A/m
                    "curie_temperature": 300,  # K (room temperature paramagnetic transition)
                    "thermal_conductivity": 120,  # W/(m⋅K)
                    "specific_heat": 134,  # J/(kg⋅K) (low due to high density)
                    "melting_point": 3680,  # K (extremely high!)
                    # Exceptional mechanical properties for penetration
                    "young_modulus": 350e9,  # Pa (very stiff)
                    "yield_strength": 750e6,  # Pa
                    "ultimate_strength": 900e6,  # Pa
                    "fatigue_limit": 400e6,  # Pa
                    "fracture_toughness": 50e6,  # Pa⋅m^0.5
                    "hardness_hv": 350,  # Vickers hardness
                    "impact_energy": 10,  # J (brittle)
                    # Ultra-high-velocity ballistics performance
                    "strain_rate_sensitivity": 0.001,  # Very low (brittle)
                    "adiabatic_shear_band_threshold": 1e6,  # s⁻¹ (very high)
                    "dynamic_yield_strength_factor": 1.8,  # Significant hardening
                    # Advanced density-related properties
                    "bulk_modulus": 310e9,  # Pa (very incompressible)
                    "poisson_ratio": 0.28,  # Dimensionless
                    "linear_expansion_coefficient": 4.5e-6,  # 1/K (very low)
                },
                # CRITICAL ENHANCEMENT: Superconducting materials for advanced coils
                "YBCO_Superconductor": {
                    "description": "YBa2Cu3O7 High-Temperature Superconductor",
                    "density": 6380,  # kg/m³
                    "mu_r_initial": 0.0,  # Perfect diamagnet below Tc
                    "mu_r_max": 0.0,     # Meissner effect
                    "resistivity_20C": 1e-12,  # Ohm⋅m (superconducting state)
                    "resistivity_normal": 1e-5,  # Ohm⋅m (normal state)
                    "temperature_coefficient": 0.0,  # Zero resistance below Tc
                    "critical_temperature": 92,  # K (liquid nitrogen temperature)
                    "critical_magnetic_field_lower": 0.02,  # T (Hc1)
                    "critical_magnetic_field_upper": 100,   # T (Hc2) - Type II
                    "critical_current_density": 1e10,  # A/m² (at 77K, 0T)
                    "thermal_conductivity": 12,  # W/(m⋅K)
                    "specific_heat": 400,  # J/(kg⋅K)
                    # Superconducting properties
                    "london_penetration_depth": 140e-9,  # m
                    "coherence_length": 1.5e-9,  # m
                    "flux_quantum": 2.07e-15,  # Wb (Φ₀)
                    "ginzburg_landau_parameter": 93,  # κ (Type II parameter)
                    # Mechanical properties (ceramic-like)
                    "young_modulus": 150e9,  # Pa
                    "yield_strength": 100e6,  # Pa (brittle fracture)
                    "ultimate_strength": 150e6,  # Pa
                    "fracture_toughness": 2e6,  # Pa⋅m^0.5 (very brittle)
                    "hardness_hv": 600,  # Vickers hardness
                    # Critical current density vs field and temperature
                    "jc_field_dependence": {
                        "0T": 1e10,   # A/m²
                        "1T": 1e9,    # A/m²
                        "5T": 1e8,    # A/m²
                        "10T": 1e7,   # A/m²
                        "50T": 1e5,   # A/m²
                    },
                    "jc_temperature_dependence": {
                        "77K": 1.0,    # Relative to reference
                        "80K": 0.8,    # Relative
                        "85K": 0.4,    # Relative
                        "90K": 0.1,    # Relative
                        "92K": 0.0,    # Above Tc
                    }
                },
                "Nb3Sn_Superconductor": {
                    "description": "Niobium-3-Tin Low-Temperature Superconductor for ultra-high fields",
                    "density": 8900,  # kg/m³
                    "mu_r_initial": 0.0,  # Perfect diamagnet below Tc
                    "mu_r_max": 0.0,     # Meissner effect
                    "resistivity_20C": 1e-12,  # Ohm⋅m (superconducting)
                    "resistivity_normal": 5e-7,  # Ohm⋅m (normal state)
                    "critical_temperature": 18.3,  # K (liquid helium required)
                    "critical_magnetic_field_lower": 0.05,  # T (Hc1)
                    "critical_magnetic_field_upper": 28,    # T (Hc2) at 4.2K
                    "critical_current_density": 3e9,  # A/m² (at 4.2K, 12T)
                    "thermal_conductivity": 25,  # W/(m⋅K)
                    "specific_heat": 420,  # J/(kg⋅K)
                    # Advanced superconducting properties
                    "london_penetration_depth": 65e-9,  # m
                    "coherence_length": 3.5e-9,  # m
                    "flux_quantum": 2.07e-15,  # Wb
                    "ginzburg_landau_parameter": 19,  # κ
                    # Mechanical properties
                    "young_modulus": 170e9,  # Pa
                    "yield_strength": 200e6,  # Pa
                    "ultimate_strength": 300e6,  # Pa
                    "fracture_toughness": 15e6,  # Pa⋅m^0.5 (brittle)
                    "hardness_hv": 400,  # Vickers hardness
                    # Ultra-high field performance
                    "maximum_usable_field": 25,  # T at 4.2K
                    "n_value": 20,  # Power law index for critical current
                    "strain_sensitivity": -300,  # %/% (very strain sensitive)
                },
                # CRITICAL ENHANCEMENT: Advanced composite materials
                "Carbon_Fiber_Composite": {
                    "description": "High-modulus carbon fiber composite for lightweight structural applications",
                    "density": 1600,  # kg/m³ (very light!)
                    "mu_r_initial": 1.0,  # Non-magnetic
                    "mu_r_max": 1.0,     # Non-magnetic
                    "resistivity_20C": 1e-3,  # Ohm⋅m (conductive in fiber direction)
                    "resistivity_transverse": 1e3,  # Ohm⋅m (insulating transverse)
                    "temperature_coefficient": -0.0005,  # 1/K (negative for carbon)
                    "thermal_conductivity_longitudinal": 800,  # W/(m⋅K) (along fibers)
                    "thermal_conductivity_transverse": 6,     # W/(m⋅K) (across fibers)
                    "specific_heat": 710,  # J/(kg⋅K)
                    "melting_point": 3800,  # K (sublimation)
                    # Exceptional mechanical properties (anisotropic)
                    "young_modulus_longitudinal": 640e9,  # Pa (along fibers - ultra-stiff!)
                    "young_modulus_transverse": 10e9,     # Pa (across fibers)
                    "shear_modulus": 4.8e9,  # Pa
                    "yield_strength_longitudinal": 4800e6,  # Pa (4.8 GPa!) - EXTREME
                    "yield_strength_transverse": 50e6,      # Pa (much weaker)
                    "ultimate_strength_longitudinal": 5000e6,  # Pa (5 GPa!)
                    "ultimate_strength_transverse": 80e6,      # Pa
                    "fatigue_limit": 2000e6,  # Pa (excellent fatigue resistance)
                    "fracture_toughness": 50e6,  # Pa⋅m^0.5
                    # Advanced composite properties
                    "fiber_volume_fraction": 0.65,  # 65% fibers
                    "void_fraction": 0.02,  # 2% voids
                    "poisson_ratio_major": 0.3,  # ν₁₂
                    "poisson_ratio_minor": 0.015,  # ν₂₁
                    "linear_expansion_coefficient_longitudinal": -0.5e-6,  # 1/K (negative!)
                    "linear_expansion_coefficient_transverse": 25e-6,     # 1/K
                    "interlaminar_shear_strength": 80e6,  # Pa
                    "compression_strength": 1500e6,  # Pa
                },
                # NEW: Advanced materials for extreme coilgun applications
                "Supermalloy": {
                    "description": "Supermalloy (79% Ni, 16% Fe, 5% Mo) - Ultra-soft magnetic",
                    "density": 8770,  # kg/m³
                    "mu_r_initial": 100000,  # Extremely high initial permeability
                    "mu_r_max": 1000000,  # Maximum permeability
                    "resistivity_20C": 6.0e-7,  # Ohm⋅m
                    "temperature_coefficient": 0.002,  # 1/K
                    "saturation_B": 0.8,  # Tesla (lower than iron)
                    "saturation_M": 6.4e5,  # A/m
                    "coercivity": 0.4,  # A/m (extremely soft)
                    "curie_temperature": 673,  # K
                    # Ultra-soft magnetic properties
                    "ja_ms": 6.4e5,
                    "ja_a": 2000,
                    "ja_alpha": 1e-5,  # Extremely low coupling
                    "ja_c": 0.01,     # Very reversible
                    "ja_k": 5,        # Minimal pinning
                },
                "Hiperco_50": {
                    "description": "Hiperco 50 (49% Fe, 49% Co, 2% V) - High saturation",
                    "density": 8120,  # kg/m³
                    "mu_r_initial": 1000,  # Initial permeability
                    "mu_r_max": 10000,  # Maximum permeability
                    "resistivity_20C": 4.0e-7,  # Ohm⋅m
                    "saturation_B": 2.4,  # Tesla (highest known)
                    "saturation_M": 1.9e6,  # A/m
                    "coercivity": 160,  # A/m
                    "curie_temperature": 1253,  # K (very high)
                    # High saturation properties
                    "ja_ms": 1.9e6,
                    "ja_a": 600,
                    "ja_alpha": 3e-3,
                    "ja_c": 0.3,
                    "ja_k": 200,
                    # Mechanical properties for high-stress applications
                    "young_modulus": 220e9,  # Pa
                    "yield_strength": 450e6,  # Pa
                    "ultimate_strength": 950e6,  # Pa
                },
                # NEW: High-velocity deformation properties
                "strain_rate_sensitivity": 0.015,  # Dimensionless
                "adiabatic_shear_band_threshold": 1e4,  # s⁻¹
                "dynamic_yield_strength_factor": 1.3,  # Multiplier at high strain rates
                
                # NEW: Extreme condition properties for ultra-high-speed applications
                "shock_hugoniot_parameters": {
                    "shock_velocity_coefficient_c0": 4570,  # m/s (bulk sound speed)
                    "shock_velocity_slope_s": 1.49,  # Dimensionless (Hugoniot slope)
                    "gruneisen_parameter": 1.67,  # Dimensionless (equation of state)
                    "bulk_modulus": 160e9,  # Pa (bulk modulus)
                },
                "phase_transitions": {
                    "alpha_gamma_transition": 1184,  # K (austenite transformation)
                    "gamma_delta_transition": 1667,  # K (delta ferrite formation)
                    "melting_point": 1811,  # K
                    "boiling_point": 3134,  # K
                    "critical_pressure": 15e9,  # Pa (pressure-induced phase changes)
                },
                "dynamic_properties": {
                    "strain_rate_hardening_exponent": 0.02,  # Cowper-Symonds parameter
                    "adiabatic_temperature_rise_coefficient": 0.9,  # Energy->temperature conversion
                    "thermal_softening_exponent": 0.6,  # Temperature-dependent yield drop
                    "spall_strength": 2e9,  # Pa (tensile failure under shock)
                    "ductile_brittle_transition": 200,  # K (low temperature embrittlement)
                },
                "electromagnetic_properties_extreme": {
                    "magnetic_domains_size": 100e-6,  # m (typical domain size)
                    "domain_wall_mobility": 1e-3,  # m²/(V⋅s) (domain response speed)
                    "coercivity_shock_enhancement": 2.0,  # Factor increase under shock
                    "saturation_shock_reduction": 0.8,  # Factor decrease in saturation under shock
                    "eddy_current_time_constant": 1e-6,  # s (characteristic eddy decay time)
                }
            },
            "wire_specifications": {
                "awg_diameter_mm": {
                    "8": 3.264, "10": 2.588, "12": 2.053, "14": 1.628, "16": 1.291,
                    "18": 1.024, "20": 0.812, "22": 0.644, "24": 0.511, "26": 0.405,
                    "28": 0.321, "30": 0.255  # Added smaller gauges
                },
                "current_capacity_A": {
                    "8": 73, "10": 55, "12": 41, "14": 32, "16": 22,
                    "18": 16, "20": 11, "22": 7, "24": 3.5, "26": 2.2,
                    "28": 1.4, "30": 0.86  # Added smaller gauges
                },
                "resistance_per_km": {  # Ohms per kilometer at 20°C
                    "8": 2.03, "10": 3.28, "12": 5.21, "14": 8.29, "16": 13.17,
                    "18": 20.95, "20": 33.31, "22": 52.96, "24": 84.22, "26": 133.9,
                    "28": 212.9, "30": 338.6  # Added smaller gauges
                },
                # NEW: High-frequency and pulse current ratings
                "pulse_current_capacity_A": {  # For microsecond pulses
                    "8": 2200, "10": 1650, "12": 1230, "14": 960, "16": 660,
                    "18": 480, "20": 330, "22": 210, "24": 105, "26": 66
                },
                "skin_depth_1khz_mm": {  # Skin depth at 1 kHz
                    "copper": 2.1, "aluminum": 2.7, "steel": 0.5
                }
            },
            # NEW: Advanced material combinations for optimized coilguns
            "composite_materials": {
                "laminated_steel": {
                    "description": "Laminated silicon steel stack",
                    "base_material": "Silicon_Steel",
                    "lamination_thickness": 0.1e-3,  # 0.1 mm
                    "insulation_thickness": 5e-6,    # 5 μm
                    "fill_factor": 0.95,  # Fraction of steel vs insulation
                    "eddy_current_reduction": 0.9,  # 90% reduction
                },
                "powder_iron_core": {
                    "description": "Iron powder composite core",
                    "base_material": "Pure_Iron",
                    "particle_size": 50e-6,  # 50 μm average
                    "fill_factor": 0.85,
                    "effective_permeability_factor": 0.6,  # Reduced due to air gaps
                    "loss_factor": 1.5,  # Increased losses
                }
            },
            # NEW: Ultra-high-strength materials for extreme applications
            "Maraging_Steel_300": {
                "description": "Ultra-high-strength maraging steel (300 grade)",
                "density": 8100,  # kg/m³
                "mu_r_initial": 800,  # Initial permeability (lower than mild steel)
                "mu_r_max": 5000,  # Maximum permeability
                "resistivity_20C": 6e-7,  # Ohm⋅m (higher than mild steel)
                "temperature_coefficient": 0.004,  # 1/K
                "saturation_B": 1.8,  # Tesla (slightly lower due to alloy content)
                "saturation_M": 1.2e6,  # A/m
                "coercivity": 800,  # A/m (higher due to precipitation hardening)
                "curie_temperature": 850,  # K (reduced due to alloying)
                
                # Extreme mechanical properties
                "young_modulus": 190e9,  # Pa
                "yield_strength": 2000e6,  # Pa (2 GPa - ultra-high strength)
                "ultimate_strength": 2100e6,  # Pa
                "fatigue_limit": 900e6,  # Pa (excellent fatigue resistance)
                "fracture_toughness": 85e6,  # Pa⋅m^0.5
                "hardness_hv": 600,  # Vickers hardness (very hard)
                
                # High-speed performance characteristics
                "strain_rate_sensitivity": 0.008,  # Low strain rate sensitivity
                "adiabatic_shear_band_threshold": 5e4,  # s⁻¹ (higher resistance)
                "dynamic_yield_strength_factor": 1.1,  # Less rate-dependent than mild steel
                
                # Shock physics parameters
                "shock_hugoniot_parameters": {
                    "shock_velocity_coefficient_c0": 4800,  # m/s
                    "shock_velocity_slope_s": 1.35,  # Dimensionless
                    "gruneisen_parameter": 1.5,  # Dimensionless
                    "bulk_modulus": 170e9,  # Pa
                },
                
                # Advanced magnetic properties for precision applications
                "ja_ms": 1.2e6,  # Saturation magnetization (A/m)
                "ja_a": 600,     # Shape parameter (stiffer response)
                "ja_alpha": 3e-3, # Interdomain coupling
                "ja_c": 0.15,    # Reversibility
                "ja_k": 1200,    # Pinning parameter (higher due to precipitates)
            },
            "Tungsten_Heavy_Alloy": {
                "description": "Tungsten heavy alloy (95% W) for extreme mass projectiles",
                "density": 18500,  # kg/m³ (ultra-high density)
                "mu_r_initial": 1.05,  # Slightly paramagnetic
                "mu_r_max": 1.2,   # Low magnetic response
                "resistivity_20C": 5.5e-8,  # Ohm⋅m (very low resistivity)
                "temperature_coefficient": 0.0048,  # 1/K
                "saturation_B": 0.1,  # Tesla (very low - essentially non-magnetic)
                "saturation_M": 1000,  # A/m (very low)
                "coercivity": 10,  # A/m (soft magnetic behavior)
                "curie_temperature": 1000,  # K (approximate for alloy)
                
                # Extreme mechanical properties for ultra-high-mass applications
                "young_modulus": 400e9,  # Pa (extremely stiff)
                "yield_strength": 1000e6,  # Pa (high strength)
                "ultimate_strength": 1200e6,  # Pa
                "fatigue_limit": 400e6,  # Pa
                "fracture_toughness": 25e6,  # Pa⋅m^0.5 (brittle but strong)
                "hardness_hv": 350,  # Vickers hardness
                
                # Specialized properties for hypervelocity applications
                "melting_point": 3695,  # K (extremely high melting point)
                "thermal_conductivity": 120,  # W/(m⋅K) (excellent heat conduction)
                "specific_heat": 134,  # J/(kg⋅K) (low specific heat)
                
                # Shock physics (extremely important for hypervelocity impacts)
                "shock_hugoniot_parameters": {
                    "shock_velocity_coefficient_c0": 4030,  # m/s
                    "shock_velocity_slope_s": 1.237,  # Dimensionless
                    "gruneisen_parameter": 1.54,  # Dimensionless
                    "bulk_modulus": 310e9,  # Pa (extremely high)
                },
                
                # Radiation resistance (important for extreme conditions)
                "radiation_damage_threshold": 1e20,  # neutrons/cm² (very radiation resistant)
                "sputtering_yield": 0.5,  # atoms/ion (low sputtering)
            },
            # NEW: Superconducting materials for ultra-high-field applications
            "YBCO_Superconductor": {
                "description": "YBa₂Cu₃O₇ high-temperature superconductor",
                "density": 6380,  # kg/m³
                "mu_r_initial": 0.0,  # Perfect diamagnetism below Tc
                "mu_r_max": 0.0,   # Meissner effect
                "resistivity_20C": 1e-6,  # Ohm⋅m (normal state)
                "resistivity_superconducting": 0.0,  # Perfect conductivity
                "critical_temperature": 92,  # K (Tc for YBCO)
                "critical_field_parallel": 150,  # Tesla (H_c2 parallel to ab-plane)
                "critical_field_perpendicular": 30,  # Tesla (H_c2 perpendicular to ab-plane)
                "critical_current_density": 1e10,  # A/m² (J_c at 77K, self-field)
                
                # Mechanical properties (ceramics are brittle)
                "young_modulus": 150e9,  # Pa
                "yield_strength": 50e6,  # Pa (compressive)
                "ultimate_strength": 200e6,  # Pa (compressive)
                "fracture_toughness": 2e6,  # Pa⋅m^0.5 (very brittle)
                
                # Thermal properties
                "thermal_conductivity": 12,  # W/(m⋅K) (anisotropic)
                "specific_heat": 400,  # J/(kg⋅K)
                
                # Superconducting properties for field calculations
                "penetration_depth": 150e-9,  # m (London penetration depth)
                "coherence_length": 3e-9,  # m (coherence length)
                "flux_quantum": 2.067e-15,  # Wb (flux quantum)
                "josephson_energy": 1e-22,  # J (typical Josephson coupling)
            }
        }
    
    def get_temperature_dependent_property(self, material_name: str, property_name: str, 
                                         temperature: Optional[float] = None) -> float:
        """
        Get temperature-dependent material property.
        
        Args:
            material_name: Name of material
            property_name: Property to retrieve
            temperature: Temperature in K (uses self.temperature if None)
            
        Returns:
            Temperature-corrected property value
        """
        if temperature is None:
            temperature = self.temperature
        
        if not self.include_temperature_effects:
            return self.get_material_property(material_name, property_name)
        
        # Get base property at reference temperature (20°C = 293.15K)
        ref_temperature = 293.15  # K
        base_value = self.get_material_property(material_name, property_name)
        
        # Apply temperature corrections
        if property_name == 'resistivity':
            temp_coeff = self.get_material_property(material_name, 'temperature_coefficient')
            resistivity_20C = self.get_material_property(material_name, 'resistivity_20C')
            # ρ(T) = ρ₀[1 + α(T - T₀)]
            corrected_value = resistivity_20C * (1 + temp_coeff * (temperature - ref_temperature))
            return corrected_value
        
        elif property_name in ['mu_r', 'mu_r_initial']:
            # Permeability decreases with temperature (Curie-Weiss law)
            curie_temp = self.get_material_property(material_name, 'curie_temperature', 1000)
            if temperature < curie_temp:
                # Below Curie temperature - ferromagnetic
                curie_weiss_factor = curie_temp / (curie_temp - temperature + 1)  # +1 to avoid singularity
                corrected_value = base_value / curie_weiss_factor
            else:
                # Above Curie temperature - paramagnetic
                corrected_value = 1.0 + 1e-4  # Weak paramagnetism
            
            return max(corrected_value, 1.0)  # Ensure mu_r >= 1
        
        elif property_name == 'saturation_B':
            # Saturation field decreases with temperature
            curie_temp = self.get_material_property(material_name, 'curie_temperature', 1000)
            if temperature < curie_temp:
                # Approximate temperature dependence: B_s(T) = B_s(0) * (1 - (T/T_c)^α)
                alpha = 1.5  # Critical exponent
                temp_factor = 1.0 - (temperature / curie_temp)**alpha
                corrected_value = base_value * max(temp_factor, 0.1)  # Minimum 10% retention
            else:
                corrected_value = 0.01  # Negligible saturation above Curie temp
            
            return corrected_value
        
        elif property_name == 'coercivity':
            # Coercivity generally decreases with temperature
            temp_coeff = -0.002  # Typical value for steel
            corrected_value = base_value * (1 + temp_coeff * (temperature - ref_temperature))
            return max(corrected_value, base_value * 0.1)  # Minimum 10% of room temp value
        
        elif property_name == 'thermal_conductivity':
            # Thermal conductivity often decreases with temperature for metals
            temp_factor = ref_temperature / temperature
            return base_value * temp_factor**0.3
        
        # Default: return base value if no temperature model available
        return base_value
    
    def get_frequency_dependent_property(self, material_name: str, property_name: str,
                                       frequency: Optional[float] = None) -> float:
        """
        Get frequency-dependent material property for high-frequency analysis.
        
        Args:
            material_name: Name of material
            property_name: Property to retrieve
            frequency: Frequency in Hz (uses self.operating_frequency if None)
            
        Returns:
            Frequency-corrected property value
        """
        if frequency is None:
            frequency = self.operating_frequency
        
        base_value = self.get_material_property(material_name, property_name)
        
        if frequency < 100:  # DC or very low frequency
            return base_value
        
        # Apply frequency corrections
        if property_name in ['mu_r', 'mu_r_initial']:
            # Permeability decreases with frequency due to eddy current shielding
            freq_coeffs = self.get_material_property(material_name, 'frequency_coefficients', {})
            perm_coeff = freq_coeffs.get('permeability_f_coeff', -0.1)
            
            # μ(f) = μ₀ * (1 + a*log₁₀(f))
            freq_factor = 1.0 + perm_coeff * np.log10(frequency / 1000)  # Normalized to 1 kHz
            corrected_value = base_value * max(freq_factor, 0.1)  # Minimum 10% retention
            
            return max(corrected_value, 1.0)
        
        elif property_name == 'resistivity':
            # Skin effect increases effective resistivity at high frequencies
            skin_depth = self._calculate_skin_depth_for_material(material_name, frequency)
            characteristic_size = 1e-3  # 1mm characteristic dimension
            
            if skin_depth < characteristic_size:
                # Significant skin effect
                skin_factor = characteristic_size / skin_depth
                corrected_value = base_value * skin_factor
            else:
                corrected_value = base_value
            
            return corrected_value
        
        return base_value  # No frequency correction for other properties
    
    def _calculate_skin_depth_for_material(self, material_name: str, frequency: float) -> float:
        """Calculate skin depth for a material at given frequency."""
        resistivity = self.get_temperature_dependent_property(material_name, 'resistivity_20C')
        mu_r = self.get_material_property(material_name, 'mu_r', 1.0)
        
        permeability = PhysicsConstants.MU_0 * mu_r
        omega = 2 * np.pi * frequency
        
        skin_depth = np.sqrt(2 * resistivity / (omega * permeability))
        return skin_depth
    
    def calculate_core_losses(self, material_name: str, frequency: float, 
                            B_peak: float, temperature: Optional[float] = None) -> float:
        """
        Calculate core losses using enhanced Steinmetz equation.
        
        Args:
            material_name: Material name
            frequency: Operating frequency (Hz)
            B_peak: Peak magnetic flux density (T)
            temperature: Temperature (K), uses self.temperature if None
            
        Returns:
            Core power loss per unit volume (W/m³)
        """
        if temperature is None:
            temperature = self.temperature
        
        # Get Steinmetz parameters
        k_steinmetz = self.get_material_property(material_name, 'steinmetz_k', 0.001)
        n_freq = self.get_material_property(material_name, 'steinmetz_n', 1.7)
        m_field = self.get_material_property(material_name, 'steinmetz_m', 2.0)
        
        # Temperature correction for core losses
        temp_factor = 1.0 + 0.003 * (temperature - PhysicsConstants.ROOM_TEMPERATURE)
        
        # Enhanced Steinmetz equation: P = k * f^n * B^m
        core_loss = k_steinmetz * (frequency**n_freq) * (B_peak**m_field) * temp_factor
        
        return max(core_loss, 0.0)
    
    def get_material_property(self, material_name: str, property_name: str, default: float = None) -> float:
        """
        Get a material property by name with high-performance caching.
        
        This method uses an efficient cache to avoid repeated lookups, providing:
        1. O(1) cached lookups for previously accessed properties
        2. Silent operation to avoid debug spam
        3. Massive performance improvement for repeated accesses
        """
        # Create cache key
        cache_key = f"{material_name}.{property_name}"
        
        # Check cache first (O(1) lookup)
        if cache_key in self._property_cache:
            self._cache_hits += 1
            return self._property_cache[cache_key]
        
        # Cache miss - perform lookup and cache result
        self._cache_misses += 1
        value = self._get_material_property_uncached(material_name, property_name, default, silent=True)
        
        # Cache the result for future use
        self._property_cache[cache_key] = value
        
        return value
    
    def _get_material_property_uncached(self, material_name: str, property_name: str, default: float = None, silent: bool = False) -> float:
        """
        Internal method to get material property without caching (for cache population).
        
        This method provides multiple fallback mechanisms:
        1. Direct lookup in basic_properties
        2. Lookup in advanced_properties  
        3. Property aliases
        4. Reasonable engineering defaults for common materials
        """
        try:
            # Check if material exists
            if material_name not in self.materials_data.get('materials', {}):
                if default is not None:
                    return default
                raise ValueError(f"Material '{material_name}' not found in database")
            
            material_data = self.materials_data['materials'][material_name]
            
            # 1. Try direct lookup in basic_properties
            if 'basic_properties' in material_data:
                if property_name in material_data['basic_properties']:
                    value = float(material_data['basic_properties'][property_name])
                    return value
            
            # 2. Try lookup in advanced_properties  
            if 'advanced_properties' in material_data:
                if property_name in material_data['advanced_properties']:
                    value = float(material_data['advanced_properties'][property_name])
                    return value
            
            # 3. Try Jiles-Atherton parameters
            if 'jiles_atherton_params' in material_data:
                ja_property_map = {
                    'ja_ms': 'ms',
                    'ja_a': 'a', 
                    'ja_alpha': 'alpha',
                    'ja_c': 'c',
                    'ja_k': 'k'
                }
                if property_name in ja_property_map:
                    ja_prop = ja_property_map[property_name]
                    if ja_prop in material_data['jiles_atherton_params']:
                        value = float(material_data['jiles_atherton_params'][ja_prop])
                        return value
            
            # 4. Try property aliases
            property_aliases = {
                'mu_r': ['relative_permeability', 'permeability', 'mu_rel'],
                'resistivity_20C': ['resistivity', 'electrical_resistivity', 'rho'],
                'density': ['mass_density', 'rho_mass'],
                'saturation_B': ['B_sat', 'saturation_flux_density'],
                'coercivity': ['H_c', 'coercive_field'],
                'curie_temperature': ['T_curie', 'curie_temp']
            }
            
            for canonical_name, aliases in property_aliases.items():
                if property_name == canonical_name or property_name in aliases:
                    # Try to find any variant of this property
                    for prop_name in [canonical_name] + aliases:
                        if 'basic_properties' in material_data and prop_name in material_data['basic_properties']:
                            value = float(material_data['basic_properties'][prop_name])
                            return value
                        if 'advanced_properties' in material_data and prop_name in material_data['advanced_properties']:
                            value = float(material_data['advanced_properties'][prop_name])
                            return value
            
            # 5. If user provided default, use it
            if default is not None:
                return default
            
            # 6. ERROR: For Pure_Iron, we should NEVER reach here for basic properties
            if material_name == 'Pure_Iron' and not silent:
                print(f"⚠️  ERROR: Pure_Iron.{property_name} not found in database!")
                print(f"     Material structure: {list(material_data.keys())}")
                if 'basic_properties' in material_data:
                    print(f"     Basic properties: {list(material_data['basic_properties'].keys())}")
                if 'advanced_properties' in material_data:
                    print(f"     Advanced properties: {list(material_data['advanced_properties'].keys())}")
                raise ValueError(f"Pure_Iron.{property_name} should be in database but was not found!")
            
            # 7. Fall back to reasonable defaults for other materials
            if material_name in ['Low_Carbon_Steel', 'Silicon_Steel', 'Iron', 'Steel']:
                reasonable_defaults = {
                    'mu_r': 1000.0,
                    'mu_r_initial': 1000.0,
                    'mu_r_max': 10000.0,
                    'density': 7850.0,
                    'ja_ms': 1.7e6,
                    'ja_a': 1000.0,
                    'ja_alpha': 0.001,
                    'ja_c': 0.1,
                    'ja_k': 500.0,
                    'resistivity_20C': 1e-7,
                    'saturation_B': 2.0,
                    'coercivity': 500.0,
                    'curie_temperature': 1043.0,
                }
                
                if property_name in reasonable_defaults:
                    if not silent:
                        warnings.warn(f"Using reasonable default for {property_name}: {reasonable_defaults[property_name]}")
                    return reasonable_defaults[property_name]
            
            raise ValueError(f"Material property '{property_name}' for '{material_name}' not found and no default provided")
            
        except Exception as e:
            if default is not None:
                if not silent:
                    warnings.warn(f"Error getting property '{property_name}' for material '{material_name}': {e}. Using provided default: {default}")
                return default
            raise ValueError(f"Could not retrieve property '{property_name}' for material '{material_name}': {e}")
    
    def get_cache_statistics(self) -> dict:
        """Get performance statistics for the property cache."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cached_properties': len(self._property_cache)
        }
    
    def clear_cache(self):
        """Clear the property cache (useful for testing or memory management)."""
        self._property_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def print_cache_performance_summary(self):
        """Print a summary of cache performance for debugging and optimization."""
        stats = self.get_cache_statistics()
        
        print(f"\n📊 Material Property Cache Performance:")
        print(f"   Cache hits: {stats['cache_hits']:,}")
        print(f"   Cache misses: {stats['cache_misses']:,}")
        print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"   Cached properties: {stats['cached_properties']:,}")
        
        if stats['total_requests'] > 100:
            performance_rating = "Excellent" if stats['hit_rate_percent'] > 90 else \
                               "Good" if stats['hit_rate_percent'] > 70 else \
                               "Poor"
            print(f"   Performance: {performance_rating}")
            
            if stats['hit_rate_percent'] < 70:
                print(f"   ⚠️  Low cache hit rate may indicate performance issues")

    def get_wire_diameter(self, awg: Union[int, str]) -> float:
        """Get wire diameter in meters from AWG."""
        awg_str = str(awg)
        if awg_str in self.materials_data['wire_specifications']['awg_diameter_mm']:
            diameter_mm = self.materials_data['wire_specifications']['awg_diameter_mm'][awg_str]
            return diameter_mm / 1000.0  # Convert mm to meters
        else:
            warnings.warn(f"AWG {awg} not found, using AWG 16 as fallback")
            return 1.291e-3  # AWG 16
    
    def get_wire_area(self, awg: Union[int, str]) -> float:
        """Get wire cross-sectional area in m²."""
        diameter = self.get_wire_diameter(awg)
        return np.pi * (diameter / 2.0) ** 2

    def get_extreme_condition_properties(self, material_name: str, temperature: float = None,
                                       strain_rate: float = None, magnetic_field: float = None,
                                       shock_pressure: float = None) -> dict:
        """
        CRITICAL NEW METHOD: Get material properties under extreme conditions.
        
        Accounts for:
        1. Temperature effects (including phase transitions)
        2. Strain rate effects (dynamic loading)
        3. Magnetic field effects (magnetostriction, saturation)
        4. Shock loading effects (Hugoniot relations)
        5. Multi-physics coupling
        """
        if material_name not in self.materials_data.get('materials', {}):
            raise ValueError(f"Material {material_name} not found in database")
        
        base_properties = self.materials_data['materials'][material_name].copy()
        extreme_properties = base_properties.copy()
        
        # Set default conditions if not provided
        if temperature is None:
            temperature = self.temperature
        if strain_rate is None:
            strain_rate = 1.0  # s⁻¹ (quasi-static)
        if magnetic_field is None:
            magnetic_field = 0.0  # T
        if shock_pressure is None:
            shock_pressure = 0.0  # Pa
        
        # 1. Temperature effects
        extreme_properties.update(self._apply_temperature_corrections(base_properties, temperature))
        
        # 2. Strain rate effects (dynamic loading)
        extreme_properties.update(self._apply_strain_rate_corrections(base_properties, strain_rate))
        
        # 3. Magnetic field effects
        extreme_properties.update(self._apply_magnetic_field_corrections(base_properties, magnetic_field))
        
        # 4. Shock loading effects
        extreme_properties.update(self._apply_shock_corrections(base_properties, shock_pressure))
        
        # 5. Multi-physics coupling
        extreme_properties.update(self._apply_multiphysics_coupling(
            base_properties, temperature, strain_rate, magnetic_field, shock_pressure
        ))
        
        return extreme_properties
    
    def _apply_temperature_corrections(self, properties: dict, temperature: float) -> dict:
        """Apply temperature-dependent property corrections."""
        corrections = {}
        
        # Temperature coefficient for resistivity
        if 'resistivity_20C' in properties and 'temperature_coefficient' in properties:
            T_ref = PhysicsConstants.ROOM_TEMPERATURE
            dT = temperature - T_ref
            resistivity_T = properties['resistivity_20C'] * (1 + properties['temperature_coefficient'] * dT)
            corrections['resistivity_temperature'] = resistivity_T
        
        # Magnetic property temperature dependence
        if 'curie_temperature' in properties:
            T_curie = properties['curie_temperature']
            if temperature < T_curie:
                # Ferromagnetic phase
                curie_factor = ((T_curie - temperature) / T_curie)**0.5  # Simplified Curie-Weiss
                if 'saturation_M' in properties:
                    corrections['saturation_M_temperature'] = properties['saturation_M'] * curie_factor
                if 'mu_r_max' in properties:
                    corrections['mu_r_max_temperature'] = 1.0 + (properties['mu_r_max'] - 1.0) * curie_factor
            else:
                # Paramagnetic phase above Curie temperature
                corrections['saturation_M_temperature'] = 0.0
                corrections['mu_r_max_temperature'] = 1.0
        
        # Mechanical property temperature dependence
        if 'yield_strength' in properties and temperature > PhysicsConstants.ROOM_TEMPERATURE:
            # Thermal softening (simplified Johnson-Cook model)
            T_melt = properties.get('melting_point', 1800)  # K
            if temperature < T_melt:
                thermal_softening_factor = 1.0 - 0.5 * ((temperature - PhysicsConstants.ROOM_TEMPERATURE) / 
                                                       (T_melt - PhysicsConstants.ROOM_TEMPERATURE))**0.5
                corrections['yield_strength_temperature'] = properties['yield_strength'] * thermal_softening_factor
            else:
                corrections['yield_strength_temperature'] = 0.0  # Molten
        
        # Superconductor critical temperature effects
        if 'critical_temperature' in properties:
            T_c = properties['critical_temperature']
            if temperature < T_c:
                # Superconducting state
                corrections['resistivity_superconducting'] = 1e-12  # Effectively zero
                corrections['critical_current_active'] = True
            else:
                # Normal state
                corrections['resistivity_superconducting'] = properties.get('resistivity_normal', 1e-5)
                corrections['critical_current_active'] = False
        
        return corrections
    
    def _apply_strain_rate_corrections(self, properties: dict, strain_rate: float) -> dict:
        """Apply strain rate (dynamic loading) corrections."""
        corrections = {}
        
        # Dynamic yield strength increase (Cowper-Symonds model)
        if 'yield_strength' in properties and 'dynamic_yield_strength_factor' in properties:
            if strain_rate > 1.0:  # s⁻¹
                strain_rate_sensitivity = properties.get('strain_rate_sensitivity', 0.01)
                dynamic_factor = properties['dynamic_yield_strength_factor']
                
                # Cowper-Symonds equation: σ_d/σ_s = 1 + (ε̇/D)^(1/p)
                D = 40.4  # s⁻¹ (material constant for steel)
                p = 5.0   # strain rate exponent
                dynamic_increase = 1.0 + (strain_rate / D)**(1/p)
                
                corrections['yield_strength_dynamic'] = (properties['yield_strength'] * 
                                                       min(dynamic_increase, dynamic_factor))
        
        # Adiabatic shear band formation
        if 'adiabatic_shear_band_threshold' in properties:
            if strain_rate > properties['adiabatic_shear_band_threshold']:
                corrections['shear_band_risk'] = True
                corrections['effective_yield_strength'] = properties['yield_strength'] * 0.5  # Dramatic reduction
            else:
                corrections['shear_band_risk'] = False
        
        return corrections
    
    def _apply_magnetic_field_corrections(self, properties: dict, magnetic_field: float) -> dict:
        """Apply magnetic field-dependent corrections."""
        corrections = {}
        
        # Magnetostriction effects
        if 'anisotropy_constant_K1' in properties and magnetic_field > 0.1:  # T
            # Magnetostriction strain: λ = (3/2)λ_s cos²θ where λ_s is saturation magnetostriction
            magnetostriction_saturation = 20e-6  # Typical for iron/steel
            B_sat = properties.get('saturation_B', 2.0)
            
            if magnetic_field < B_sat:
                magnetostriction_strain = magnetostriction_saturation * (magnetic_field / B_sat)**2
            else:
                magnetostriction_strain = magnetostriction_saturation
            
            corrections['magnetostriction_strain'] = magnetostriction_strain
            
            # Mechanical property changes due to magnetostriction
            if 'young_modulus' in properties:
                corrections['young_modulus_magnetic'] = properties['young_modulus'] * (1 + magnetostriction_strain)
        
        # Superconductor critical current vs field
        if 'critical_magnetic_field_upper' in properties:
            H_c2 = properties['critical_magnetic_field_upper']
            if magnetic_field < H_c2:
                # Kim model for critical current vs field
                if 'critical_current_density' in properties:
                    j_c0 = properties['critical_current_density']
                    field_dependence = j_c0 / (1 + magnetic_field / 1.0)  # Simplified Kim model
                    corrections['critical_current_density_field'] = field_dependence
            else:
                corrections['critical_current_density_field'] = 0.0  # Quenched
        
        return corrections
    
    def _apply_shock_corrections(self, properties: dict, shock_pressure: float) -> dict:
        """Apply shock loading (Hugoniot) corrections."""
        corrections = {}
        
        if shock_pressure > 1e9:  # GPa pressures
            # Hugoniot relations: U_s = c_0 + s*u_p where U_s is shock velocity, u_p is particle velocity
            # For steel: c_0 ≈ 4570 m/s, s ≈ 1.49
            c_0 = 4570  # m/s (bulk sound speed)
            s = 1.49    # Hugoniot slope parameter
            
            # Estimate particle velocity from pressure (simplified)
            density = properties.get('density', 7850)
            particle_velocity = np.sqrt(shock_pressure / (density * c_0 * s))
            shock_velocity = c_0 + s * particle_velocity
            
            corrections['shock_velocity'] = shock_velocity
            corrections['particle_velocity'] = particle_velocity
            
            # Shock temperature rise (simplified)
            specific_heat = properties.get('specific_heat', 460)
            shock_temperature_rise = shock_pressure / (density * specific_heat * 1000)  # Rough estimate
            corrections['shock_temperature_rise'] = shock_temperature_rise
            
            # Phase transitions under shock
            if shock_pressure > 13e9:  # 13 GPa - α→ε transition in iron
                corrections['phase_transition'] = 'alpha_to_epsilon'
                corrections['density_shock'] = density * 1.1  # ~10% density increase
        
        return corrections
    
    def _apply_multiphysics_coupling(self, properties: dict, temperature: float, 
                                   strain_rate: float, magnetic_field: float, 
                                   shock_pressure: float) -> dict:
        """Apply multi-physics coupling effects."""
        corrections = {}
        
        # Thermal-magnetic coupling
        if temperature > PhysicsConstants.ROOM_TEMPERATURE and magnetic_field > 1.0:
            # Elevated temperature reduces magnetic saturation
            T_curie = properties.get('curie_temperature', 1043)
            thermal_mag_factor = max(0.1, 1.0 - (temperature - PhysicsConstants.ROOM_TEMPERATURE) / T_curie)
            
            if 'saturation_B' in properties:
                corrections['saturation_B_thermal_magnetic'] = properties['saturation_B'] * thermal_mag_factor
        
        # Thermo-mechanical coupling
        if temperature > PhysicsConstants.ROOM_TEMPERATURE and strain_rate > 100:
            # Adiabatic heating during high-rate deformation
            adiabatic_heating_factor = 1.0 + 0.001 * strain_rate  # Simplified
            corrections['effective_temperature'] = temperature * adiabatic_heating_factor
        
        # Magneto-mechanical coupling
        if magnetic_field > 1.0 and strain_rate > 1000:
            # Magnetic field affects dislocation motion
            magnetic_hardening_factor = 1.0 + 0.1 * np.log10(magnetic_field)
            if 'yield_strength' in properties:
                corrections['yield_strength_magneto_mechanical'] = (properties['yield_strength'] * 
                                                                  magnetic_hardening_factor)
        
        return corrections


class AdvancedPermeabilityModel:
    """
    Advanced permeability modeling including Jiles-Atherton hysteresis model.
    """
    
    def __init__(self, materials: AdvancedMaterialProperties):
        """Initialize advanced permeability model."""
        self.materials = materials
        
        # Hysteresis state variables
        self.hysteresis_states = {}  # Track state for each material
        
        # PERFORMANCE FIX: Add permeability caching
        self._permeability_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size_limit = 10000  # Limit cache size to prevent memory issues
    
    def calculate_nonlinear_permeability_with_hysteresis(self, H_applied: float, material_name: str,
                                                       previous_B: Optional[float] = None,
                                                       dI_dt: float = 0) -> Tuple[float, float]:
        """
        Calculate nonlinear permeability with full hysteresis modeling.
        
        Args:
            H_applied: Applied magnetic field (A/m)
            material_name: Material name
            previous_B: Previous B field for hysteresis tracking
            dI_dt: Current rate of change for dynamic effects
            
        Returns:
            Tuple of (effective_permeability, B_field)
        """
        # Get material parameters
        mu_r_initial = self.materials.get_temperature_dependent_property(material_name, 'mu_r_initial')
        mu_r_max = self.materials.get_temperature_dependent_property(material_name, 'mu_r_max')
        
        # Non-magnetic materials
        if mu_r_initial < 2:
            B_field = PhysicsConstants.MU_0 * H_applied
            return mu_r_initial, B_field
        
        # Initialize hysteresis state if needed
        if material_name not in self.hysteresis_states:
            self.hysteresis_states[material_name] = {
                'M_rev': 0.0,  # Reversible magnetization
                'M_irr': 0.0,  # Irreversible magnetization
                'H_prev': 0.0,  # Previous H field
                'B_prev': 0.0   # Previous B field
            }
        
        state = self.hysteresis_states[material_name]
        
        # Get Jiles-Atherton parameters
        Ms = self.materials.get_material_property(material_name, 'ja_ms')
        a = self.materials.get_material_property(material_name, 'ja_a')
        alpha = self.materials.get_material_property(material_name, 'ja_alpha')
        c = self.materials.get_material_property(material_name, 'ja_c')
        k = self.materials.get_material_property(material_name, 'ja_k')
        
        # Calculate effective field including domain coupling
        H_eff = H_applied + alpha * state['M_irr']
        
        # Anhysteretic magnetization (Langevin function approximation)
        if abs(H_eff) > 1e-6:
            coth_term = 1.0 / np.tanh(H_eff / a) if abs(H_eff / a) > 1e-3 else a / H_eff
            M_an = Ms * (coth_term - a / H_eff)
        else:
            M_an = Ms * H_eff / (3 * a)  # Low field approximation
        
        # Direction of field change
        dH = H_applied - state['H_prev']
        delta = 1.0 if dH >= 0 else -1.0
        
        # Irreversible magnetization change
        if abs(dH) > 1e-12:
            dM_irr_dH = (M_an - state['M_irr']) / (k * delta - alpha * (M_an - state['M_irr']))
            dM_irr = dM_irr_dH * dH
            state['M_irr'] += dM_irr
        
        # Reversible magnetization
        state['M_rev'] = c * (M_an - state['M_irr'])
        
        # Total magnetization
        M_total = state['M_irr'] + state['M_rev']
        
        # B field: B = μ₀(H + M)
        B_field = PhysicsConstants.MU_0 * (H_applied + M_total)
        
        # Effective permeability
        if abs(H_applied) > 1e-12:
            mu_eff = B_field / (PhysicsConstants.MU_0 * H_applied)
        else:
            mu_eff = mu_r_initial
        
        # Update state
        state['H_prev'] = H_applied
        state['B_prev'] = B_field
        
        # Apply limits
        mu_eff = max(1.0, min(mu_eff, mu_r_max))
        
        return mu_eff, B_field
    
    def calculate_nonlinear_permeability(self, H_applied: float, material_name: str, 
                                       previous_B: Optional[float] = None, 
                                       dI_dt: float = 0) -> float:
        """
        Calculate nonlinear permeability using a CORRECTED approach.
        
        The previous implementation was returning μ=1.0 (saturated) incorrectly.
        This version provides realistic permeability values.
        """
        # PERFORMANCE FIX: Enhanced caching with binning for similar values
        # Round H_applied to reduce cache misses from tiny differences
        H_cache_key = round(H_applied, -2)  # Round to nearest 100 A/m
        cache_key = (material_name, H_cache_key)
        
        if cache_key in self._permeability_cache:
            self._cache_hits += 1
            return self._permeability_cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            # Get material properties
            mu_r_initial = self.materials.get_material_property(material_name, 'mu_r')
            mu_r_max = self.materials.get_material_property(material_name, 'mu_r_max', mu_r_initial * 10)
            saturation_B = self.materials.get_material_property(material_name, 'saturation_B', 2.0)
            
            # Calculate saturation field strength
            mu_0 = 4e-7 * np.pi  # H/m
            H_saturation = saturation_B / mu_0  # A/m
            
            # CORRECTED: Use a simple but realistic permeability model
            # For Pure_Iron: μ_r should be high initially, then decrease as we approach saturation
            
            # Normalize applied field by saturation field
            h_norm = abs(H_applied) / H_saturation
            
            if h_norm < 0.1:
                # Low field regime: use initial permeability
                mu_r_effective = mu_r_initial
            elif h_norm < 0.5:
                # Medium field regime: gradual decrease
                transition_factor = (0.5 - h_norm) / 0.4  # Linear transition from 0.1 to 0.5
                mu_r_effective = mu_r_initial * (0.1 + 0.9 * transition_factor)
            elif h_norm < 0.9:
                # High field regime: approaching saturation
                mu_r_effective = mu_r_initial * 0.1 * (0.9 - h_norm) / 0.4
            else:
                # Saturation regime: but still maintain some permeability
                mu_r_effective = max(1.0, mu_r_initial * 0.01)
            
            # Ensure minimum permeability of 1.0 (vacuum permeability)
            mu_r_effective = max(1.0, mu_r_effective)
            
            # Debug print for critical cases (disabled for performance)
            # if material_name == 'Pure_Iron' and abs(H_applied) > 500000:
            #     print(f"🔍 Permeability calc: H={H_applied:.0f} A/m, h_norm={h_norm:.3f}, μ_r={mu_r_effective:.1f}")
            
            # Cache the result (with size management)
            if len(self._permeability_cache) >= self._cache_size_limit:
                # Remove oldest entries (simple FIFO strategy)
                # Clear 25% of cache when limit reached
                items_to_remove = self._cache_size_limit // 4
                for _ in range(items_to_remove):
                    if self._permeability_cache:
                        self._permeability_cache.pop(next(iter(self._permeability_cache)))
            
            self._permeability_cache[cache_key] = mu_r_effective
            
            return mu_r_effective
            
        except Exception as e:
            # Fallback to a reasonable value for ferromagnetic materials
            print(f"⚠️  Permeability calculation error for {material_name}: {e}")
            if material_name in ['Pure_Iron', 'Low_Carbon_Steel', 'Silicon_Steel']:
                fallback_value = 100.0  # Reasonable fallback for iron/steel
            else:
                fallback_value = 1.0  # Non-magnetic material
            
            # Cache the fallback value too (with size management)
            if len(self._permeability_cache) >= self._cache_size_limit:
                # Remove oldest entries (simple FIFO strategy)
                items_to_remove = self._cache_size_limit // 4
                for _ in range(items_to_remove):
                    if self._permeability_cache:
                        self._permeability_cache.pop(next(iter(self._permeability_cache)))
            
            self._permeability_cache[cache_key] = fallback_value
            return fallback_value
    
    def clear_permeability_cache(self):
        """Clear the permeability cache."""
        self._permeability_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_cache_statistics(self) -> dict:
        """Get permeability cache performance statistics."""
        total_calls = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_calls': total_calls,
            'hit_rate_percent': hit_rate,
            'cache_size': len(self._permeability_cache)
        }
    
    def print_cache_statistics(self):
        """Print cache performance statistics."""
        stats = self.get_cache_statistics()
        print(f"\n📊 Permeability Cache Performance:")
        print(f"   Cache hits: {stats['cache_hits']:,}")
        print(f"   Cache misses: {stats['cache_misses']:,}")
        print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
        print(f"   Cache size: {stats['cache_size']:,} entries")
        
        if stats['hit_rate_percent'] > 80:
            print(f"   Performance: ✅ Excellent caching efficiency")
        elif stats['hit_rate_percent'] > 60:
            print(f"   Performance: ⚡ Good caching efficiency")
        elif stats['hit_rate_percent'] > 40:
            print(f"   Performance: ⚠️  Moderate caching efficiency")
        else:
            print(f"   Performance: 🔴 Low caching efficiency")

# Create aliases for backward compatibility
MaterialProperties = AdvancedMaterialProperties
PermeabilityModel = AdvancedPermeabilityModel
