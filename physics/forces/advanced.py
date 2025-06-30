"""
Advanced Electromagnetic Forces

This module combines all specialized force calculators into a comprehensive
advanced electromagnetic forces system.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict
import warnings
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils, SafetyLimits
from ..fields import AdvancedMagneticFieldCalculator
from ..materials import AdvancedMaterialProperties, AdvancedPermeabilityModel
from .base import BaseElectromagneticForces
from .quantum import QuantumForceCalculator
from .maxwell_stress import MaxwellStressTensor
from .eddy_currents import EddyCurrentForces
from .hysteresis import HysteresisForces
from .relativistic import RelativisticForces
from .multiscale import MultiscaleForces


class AdvancedElectromagneticForces(BaseElectromagneticForces):
    """
    Advanced electromagnetic force calculations combining all specialized methods.
    
    Integrates:
    - Quantum mechanical corrections
    - Maxwell stress tensor
    - Eddy current effects
    - Hysteresis modeling
    - Relativistic corrections
    - Multi-scale coupling
    """
    
    def __init__(self, config: dict, field_calculator: AdvancedMagneticFieldCalculator, 
                 materials: AdvancedMaterialProperties):
        """Initialize advanced electromagnetic forces calculator."""
        super().__init__(config, field_calculator, materials)
        
        # Initialize specialized calculators
        self._initialize_specialized_calculators(config, field_calculator, materials)
        
        # Advanced parameters
        self.max_acceleration = config.get('simulation', {}).get('max_acceleration', 1e8)
        self.max_velocity = config.get('simulation', {}).get('max_velocity', 15000)
        self.max_force = config.get('simulation', {}).get('max_force', 1e9)
        
        print("🔬 ADVANCED electromagnetic forces initialized")
        print(f"   - Max acceleration: {self.max_acceleration/1e6:.0f} Mg")
        print(f"   - Max velocity: {self.max_velocity/1000:.1f} km/s")
        print(f"   - Max force: {self.max_force/1e9:.1f} GN")
    
    def _initialize_specialized_calculators(self, config, field_calculator, materials):
        """Initialize all specialized force calculators."""
        
        # Quantum forces
        if config.get('quantum_physics', {}).get('enable_quantum_forces', False):
            self.quantum_calc = QuantumForceCalculator(config, field_calculator, materials)
        else:
            self.quantum_calc = None
        
        # Maxwell stress tensor
        if config.get('advanced_physics', {}).get('use_maxwell_stress', False):
            self.maxwell_calc = MaxwellStressTensor(config, field_calculator, materials)
        else:
            self.maxwell_calc = None
        
        # Eddy currents
        if config.get('advanced_physics', {}).get('include_eddy_currents', True):
            self.eddy_calc = EddyCurrentForces(config, field_calculator, materials)
        else:
            self.eddy_calc = None
        
        # Hysteresis
        if config.get('advanced_physics', {}).get('include_hysteresis', False):
            self.hysteresis_calc = HysteresisForces(config, field_calculator, materials)
        else:
            self.hysteresis_calc = None
        
        # Relativistic corrections
        if config.get('advanced_physics', {}).get('include_relativistic', False):
            self.relativistic_calc = RelativisticForces(config, field_calculator, materials)
        else:
            self.relativistic_calc = None
        
        # Multi-scale coupling
        self.multiscale_calc = MultiscaleForces(config, field_calculator, materials)
    
    def calculate_total_electromagnetic_force_quantum_enhanced(self, current: float, position: float, 
                                                             velocity: float = 0.0, 
                                                             acceleration: float = 0.0,
                                                             current_history: Optional[List] = None,
                                                             time_history: Optional[List] = None,
                                                             time: float = 0.0) -> Tuple[float, dict]:
        """
        Calculate total electromagnetic force with all advanced corrections.
        
        Returns:
            Tuple of (total_force, detailed_breakdown)
        """
        force_breakdown = {}
        total_force = 0.0
        
        # Basic classical forces
        f_gradient = self.calculate_gradient_force(current, position)
        f_reluctance = self.calculate_reluctance_force(current, position)
        f_lorentz = self.calculate_lorentz_force(current, position, velocity)
        
        force_breakdown['gradient'] = f_gradient
        force_breakdown['reluctance'] = f_reluctance
        force_breakdown['lorentz'] = f_lorentz
        
        total_force = f_gradient + f_reluctance + f_lorentz
        
        # Eddy current forces
        if self.eddy_calc:
            f_eddy, eddy_power = self.eddy_calc.calculate_eddy_current_force(
                current, position, velocity, current_history, time_history
            )
            force_breakdown['eddy_current'] = f_eddy
            force_breakdown['eddy_power'] = eddy_power
            total_force += f_eddy
        
        # Hysteresis forces
        if self.hysteresis_calc:
            f_hysteresis, hysteresis_loss = self.hysteresis_calc.calculate_hysteresis_force(
                current, position, time
            )
            force_breakdown['hysteresis'] = f_hysteresis
            force_breakdown['hysteresis_loss'] = hysteresis_loss
            total_force += f_hysteresis
        
        # Maxwell stress tensor (alternative calculation)
        if self.maxwell_calc:
            f_maxwell, maxwell_breakdown = self.maxwell_calc.calculate_maxwell_stress_force(
                current, position, velocity
            )
            force_breakdown['maxwell_stress'] = maxwell_breakdown
            # Note: Maxwell stress is alternative to classical calculation, not additive
        
        # Relativistic corrections
        if self.relativistic_calc:
            f_relativistic, rel_breakdown = self.relativistic_calc.calculate_relativistic_corrections(
                current, position, velocity, acceleration
            )
            force_breakdown['relativistic'] = rel_breakdown
            total_force += f_relativistic
        
        # Quantum corrections
        if self.quantum_calc:
            f_quantum, quantum_breakdown = self.quantum_calc.calculate_quantum_force_corrections(
                current, position, velocity
            )
            force_breakdown['quantum'] = quantum_breakdown
            total_force += f_quantum
        
        # Multi-scale coupling
        f_multiscale = self.multiscale_calc.calculate_multiscale_coupling_force(
            current, position, velocity, force_breakdown
        )
        force_breakdown['multiscale_coupling'] = f_multiscale
        total_force += f_multiscale
        
        # Apply safety limits
        total_force = self.apply_safety_limits(total_force)
        force_breakdown['total_force'] = total_force
        
        # Energy conservation validation
        if self.validate_energy_conservation(total_force, current, position):
            force_breakdown['energy_conservation'] = 'PASS'
        else:
            force_breakdown['energy_conservation'] = 'FAIL'
            warnings.warn("Energy conservation check failed")
        
        return total_force, force_breakdown
    
    def magnetic_force_ferromagnetic(self, current: float, position: float, 
                                   velocity: float = 0.0, 
                                   current_history: Optional[List] = None,
                                   time_history: Optional[List] = None) -> Tuple[float, float]:
        """
        Calculate magnetic force for ferromagnetic projectile (simplified interface).
        
        Returns:
            Tuple of (total_force, total_power_dissipation)
        """
        total_force, breakdown = self.calculate_total_electromagnetic_force_quantum_enhanced(
            current, position, velocity, 0.0, current_history, time_history, 0.0
        )
        
        # Calculate total power dissipation
        total_power = breakdown.get('eddy_power', 0.0) + breakdown.get('hysteresis_loss', 0.0)
        
        return total_force, total_power
    
    def get_force_component_summary(self, force_breakdown: dict) -> dict:
        """Generate a summary of force components for analysis."""
        summary = {}
        
        # Extract main components
        main_components = ['gradient', 'reluctance', 'lorentz', 'eddy_current', 'hysteresis']
        for component in main_components:
            summary[component] = force_breakdown.get(component, 0.0)
        
        # Calculate total of main components
        total_main = sum(summary.values())
        summary['total_main_components'] = total_main
        
        # Add correction terms
        corrections = {}
        if 'relativistic' in force_breakdown:
            rel_data = force_breakdown['relativistic']
            if isinstance(rel_data, dict):
                corrections['relativistic_total'] = sum(rel_data.values()) if rel_data else 0.0
            else:
                corrections['relativistic_total'] = rel_data
        
        if 'quantum' in force_breakdown:
            quantum_data = force_breakdown['quantum']
            if isinstance(quantum_data, dict):
                corrections['quantum_total'] = sum(quantum_data.values()) if quantum_data else 0.0
            else:
                corrections['quantum_total'] = quantum_data
        
        corrections['multiscale_coupling'] = force_breakdown.get('multiscale_coupling', 0.0)
        
        summary['corrections'] = corrections
        summary['total_corrections'] = sum(corrections.values())
        summary['grand_total'] = total_main + summary['total_corrections']
        
        return summary 