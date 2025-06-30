"""
Multiscale Force Modeling

This module implements multi-scale force modeling from quantum to classical scales.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils
from .base import BaseElectromagneticForces


class MultiscaleForces(BaseElectromagneticForces):
    """
    Multi-scale force modeling from quantum to classical regimes.
    
    Scale hierarchy: quantum → atomic → mesoscopic → macroscopic
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize multiscale forces calculator."""
        super().__init__(config, field_calculator, materials)
        
        self._initialize_multiscale_parameters()
        
        print(f"🔬 Multi-scale force modeling initialized (4 scales: quantum → macro)")
    
    def _initialize_multiscale_parameters(self):
        """Initialize multi-scale force modeling parameters."""
        # Scale hierarchy: quantum → atomic → mesoscopic → macroscopic
        
        # Quantum scale (10^-12 m): Quantum mechanical forces
        self.quantum_scale = 1e-12  # m
        
        # Atomic scale (10^-10 m): Atomic magnetic dipole interactions
        self.atomic_scale = 1e-10  # m
        
        # Mesoscopic scale (10^-6 m): Domain wall dynamics, microstructure
        self.mesoscopic_scale = 1e-6  # m
        
        # Macroscopic scale (10^-3 m): Continuum electromagnetics
        self.macroscopic_scale = 1e-3  # m
        
        # Scale coupling parameters
        self.scale_coupling_strengths = {
            'quantum_atomic': 1e-3,
            'atomic_mesoscopic': 1e-2,
            'mesoscopic_macroscopic': 1e-1
        }
    
    def calculate_multiscale_coupling_force(self, current: float, position: float, 
                                          velocity: float, force_breakdown: dict) -> float:
        """
        Calculate force corrections from multi-scale coupling.
        
        Couples quantum → atomic → mesoscopic → macroscopic scales.
        """
        total_coupling_force = 0.0
        
        # Quantum to atomic scale coupling
        if 'quantum' in force_breakdown:
            quantum_atomic_coupling = self._calculate_quantum_atomic_coupling(
                force_breakdown.get('quantum', 0.0), current, position
            )
            total_coupling_force += quantum_atomic_coupling
        
        # Atomic to mesoscopic scale coupling
        atomic_force = force_breakdown.get('atomic', 0.0)
        atomic_mesoscopic_coupling = self._calculate_atomic_mesoscopic_coupling(
            atomic_force, current, position, velocity
        )
        total_coupling_force += atomic_mesoscopic_coupling
        
        # Mesoscopic to macroscopic scale coupling
        mesoscopic_force = force_breakdown.get('mesoscopic', 0.0)
        mesoscopic_macroscopic_coupling = self._calculate_mesoscopic_macroscopic_coupling(
            mesoscopic_force, current, position
        )
        total_coupling_force += mesoscopic_macroscopic_coupling
        
        return NumericalUtils.safe_numerical_operation(total_coupling_force, "multiscale_coupling")
    
    def _calculate_quantum_atomic_coupling(self, quantum_force: float, current: float, position: float) -> float:
        """Calculate quantum to atomic scale coupling."""
        coupling_strength = self.scale_coupling_strengths['quantum_atomic']
        
        # Characteristic length ratio
        scale_ratio = self.quantum_scale / self.atomic_scale
        
        # Coupling depends on local field conditions
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        field_factor = np.tanh(B_field / 1.0)  # Saturation at 1 Tesla
        
        coupling_force = quantum_force * coupling_strength * scale_ratio * field_factor
        
        return coupling_force
    
    def _calculate_atomic_mesoscopic_coupling(self, atomic_force: float, current: float, 
                                           position: float, velocity: float) -> float:
        """Calculate atomic to mesoscopic scale coupling."""
        coupling_strength = self.scale_coupling_strengths['atomic_mesoscopic']
        
        # Domain wall dynamics effects
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Velocity-dependent domain wall motion
        velocity_factor = np.tanh(abs(velocity) / 1000.0)  # Saturation at 1 km/s
        
        coupling_force = atomic_force * coupling_strength * velocity_factor
        
        return coupling_force
    
    def _calculate_mesoscopic_macroscopic_coupling(self, mesoscopic_force: float, 
                                                 current: float, position: float) -> float:
        """Calculate mesoscopic to macroscopic scale coupling."""
        coupling_strength = self.scale_coupling_strengths['mesoscopic_macroscopic']
        
        # Microstructure to continuum transition
        # Depends on material properties and field strength
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        
        # Permeability effects
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
        mu_eff = self.permeability_model.calculate_nonlinear_permeability(H_applied, self.proj_material)
        
        permeability_factor = (mu_eff - 1.0) / max(mu_eff, 1.0)
        
        coupling_force = mesoscopic_force * coupling_strength * permeability_factor
        
        return coupling_force 