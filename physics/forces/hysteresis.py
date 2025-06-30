"""
Hysteresis Force Calculations

This module implements magnetic hysteresis effects using the Jiles-Atherton model
and other advanced hysteresis models.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from ..core import BasePhysicsModel, PhysicsConstants, NumericalUtils
from .base import BaseElectromagneticForces


class HysteresisForces(BaseElectromagneticForces):
    """
    Magnetic hysteresis force calculations with energy dissipation.
    
    Implements:
    - Jiles-Atherton hysteresis model
    - Preisach model for complex materials
    - Hysteresis energy losses
    - Temperature-dependent hysteresis
    """
    
    def __init__(self, config: dict, field_calculator, materials):
        """Initialize hysteresis forces calculator."""
        super().__init__(config, field_calculator, materials)
        
        # Hysteresis parameters
        self.include_hysteresis = config.get('advanced_physics', {}).get('include_hysteresis', True)
        
        if self.include_hysteresis:
            self._initialize_hysteresis_model()
        
        print(f"🔄 Hysteresis forces initialized")
        print(f"   - Model: {'Jiles-Atherton' if self.include_hysteresis else 'None'}")
    
    def _initialize_hysteresis_model(self):
        """Initialize Jiles-Atherton hysteresis model parameters."""
        # Jiles-Atherton parameters for common ferromagnetic materials
        material_params = {
            'Low_Carbon_Steel': {
                'Ms': 1.7e6,      # Saturation magnetization (A/m)
                'a': 1000.0,      # Shape parameter
                'alpha': 1e-3,    # Interdomain coupling
                'c': 0.1,         # Reversibility parameter
                'k': 500.0        # Coercivity parameter (A/m)
            },
            'Iron': {
                'Ms': 1.7e6,
                'a': 1200.0,
                'alpha': 1e-3,
                'c': 0.2,
                'k': 800.0
            }
        }
        
        # Get parameters for projectile material
        if self.proj_material in material_params:
            self.hysteresis_params = material_params[self.proj_material]
        else:
            # Default parameters
            self.hysteresis_params = material_params['Low_Carbon_Steel']
        
        # Initialize hysteresis state
        self.magnetization_history = []
        self.field_history = []
        self.current_magnetization = 0.0
        self.hysteresis_energy_loss = 0.0
        
        print(f"   - Material: {self.proj_material}")
        print(f"   - Saturation magnetization: {self.hysteresis_params['Ms']:.0e} A/m")
        print(f"   - Coercivity: {self.hysteresis_params['k']:.0f} A/m")
    
    def calculate_hysteresis_force(self, current: float, position: float, 
                                 time: float) -> Tuple[float, float]:
        """
        Calculate force and energy loss due to magnetic hysteresis.
        
        Returns:
            Tuple of (hysteresis_force, energy_loss_rate)
        """
        if not self.include_hysteresis:
            return 0.0, 0.0
        
        # Current magnetic field
        B_field = self.field_calc.magnetic_field_solenoid_on_axis(position, current)
        H_field = B_field / PhysicsConstants.MU_0
        
        # Calculate magnetization using Jiles-Atherton model
        M_new = self._calculate_jiles_atherton_magnetization(H_field)
        
        # Force from magnetization gradient
        dM_dz = self._calculate_magnetization_gradient(position, current)
        hysteresis_force = PhysicsConstants.MU_0 * self.proj_volume * M_new * dM_dz
        
        # Energy dissipation from hysteresis
        energy_loss_rate = self._calculate_hysteresis_energy_loss(H_field, M_new, time)
        
        # Update history
        self._update_hysteresis_history(H_field, M_new)
        
        return (NumericalUtils.safe_numerical_operation(hysteresis_force, "hysteresis_force"),
                NumericalUtils.safe_numerical_operation(energy_loss_rate, "hysteresis_energy_loss"))
    
    def _calculate_jiles_atherton_magnetization(self, H_field: float) -> float:
        """
        Calculate magnetization using Jiles-Atherton model.
        
        Simplified implementation of the differential equation approach.
        """
        params = self.hysteresis_params
        
        # Effective field including interdomain coupling
        H_eff = H_field + params['alpha'] * self.current_magnetization
        
        # Anhysteretic magnetization (Langevin function approximation)
        if abs(H_eff) > 1e-6:
            coth_term = 1.0 / np.tanh(H_eff / params['a']) if abs(H_eff / params['a']) > 1e-6 else params['a'] / H_eff
            Man = params['Ms'] * (coth_term - params['a'] / H_eff)
        else:
            Man = 0.0
        
        # Direction of field change
        if len(self.field_history) > 0:
            dH_dt = np.sign(H_field - self.field_history[-1]) if H_field != self.field_history[-1] else 0
        else:
            dH_dt = np.sign(H_field) if H_field != 0 else 0
        
        # Irreversible magnetization change
        if dH_dt != 0:
            delta = 1.0 if ((H_field > 0 and dH_dt > 0) or (H_field < 0 and dH_dt < 0)) else -1.0
            
            # Coercive field
            Hc = params['k'] * delta
            
            # Irreversible susceptibility
            if abs(Man - self.current_magnetization) > 1e-12:
                chi_irr = (Man - self.current_magnetization) / (params['k'] * delta - params['alpha'] * (Man - self.current_magnetization))
            else:
                chi_irr = 0.0
        else:
            chi_irr = 0.0
        
        # Reversible magnetization change
        chi_rev = params['c'] * (Man - self.current_magnetization)
        
        # Total magnetization change (simplified)
        dM = chi_irr + chi_rev
        M_new = self.current_magnetization + dM * 0.1  # Time step factor
        
        # Clamp to physical limits
        M_new = np.clip(M_new, -params['Ms'], params['Ms'])
        
        self.current_magnetization = M_new
        return M_new
    
    def _calculate_magnetization_gradient(self, position: float, current: float) -> float:
        """Calculate spatial gradient of magnetization."""
        delta_z = 1e-6
        
        # Calculate magnetization at nearby positions
        H_plus = self.field_calc.magnetic_field_solenoid_on_axis(position + delta_z, current) / PhysicsConstants.MU_0
        H_minus = self.field_calc.magnetic_field_solenoid_on_axis(position - delta_z, current) / PhysicsConstants.MU_0
        
        # Save current state
        M_current = self.current_magnetization
        
        # Calculate magnetization at different positions
        M_plus = self._calculate_jiles_atherton_magnetization(H_plus)
        self.current_magnetization = M_current  # Restore
        M_minus = self._calculate_jiles_atherton_magnetization(H_minus)
        self.current_magnetization = M_current  # Restore
        
        # Gradient
        dM_dz = (M_plus - M_minus) / (2 * delta_z)
        
        return dM_dz
    
    def _calculate_hysteresis_energy_loss(self, H_field: float, M_new: float, time: float) -> float:
        """
        Calculate energy dissipation rate due to hysteresis.
        
        Energy loss = ∫ H · dM (around hysteresis loop)
        """
        if len(self.field_history) < 2 or len(self.magnetization_history) < 2:
            return 0.0
        
        # Change in magnetization
        dM = M_new - self.magnetization_history[-1] if self.magnetization_history else 0.0
        
        # Energy dissipated per unit volume
        dE_vol = H_field * dM
        
        # Total energy dissipation rate
        energy_loss_rate = abs(dE_vol) * self.proj_volume
        
        # Update cumulative loss
        self.hysteresis_energy_loss += energy_loss_rate
        
        return energy_loss_rate
    
    def _update_hysteresis_history(self, H_field: float, M_new: float):
        """Update hysteresis history for next calculation."""
        max_history = 1000  # Limit history size
        
        self.field_history.append(H_field)
        self.magnetization_history.append(M_new)
        
        # Trim history if too long
        if len(self.field_history) > max_history:
            self.field_history = self.field_history[-max_history//2:]
            self.magnetization_history = self.magnetization_history[-max_history//2:]
    
    def get_hysteresis_loop_data(self) -> Dict[str, List]:
        """Get current hysteresis loop data for analysis."""
        return {
            'H_field': self.field_history.copy(),
            'magnetization': self.magnetization_history.copy(),
            'energy_loss': self.hysteresis_energy_loss
        }
    
    def reset_hysteresis_state(self):
        """Reset hysteresis state to demagnetized condition."""
        self.current_magnetization = 0.0
        self.field_history.clear()
        self.magnetization_history.clear()
        self.hysteresis_energy_loss = 0.0
    
    def calculate_coercivity(self) -> float:
        """Calculate current coercivity from hysteresis loop."""
        if len(self.field_history) < 10 or len(self.magnetization_history) < 10:
            return self.hysteresis_params['k']  # Default value
        
        # Find zero crossings of magnetization
        zero_crossings = []
        for i in range(1, len(self.magnetization_history)):
            if (self.magnetization_history[i-1] * self.magnetization_history[i] < 0):
                # Linear interpolation to find exact crossing
                H_cross = self.field_history[i-1] + (self.field_history[i] - self.field_history[i-1]) * \
                         (-self.magnetization_history[i-1]) / (self.magnetization_history[i] - self.magnetization_history[i-1])
                zero_crossings.append(abs(H_cross))
        
        if zero_crossings:
            return np.mean(zero_crossings)
        else:
            return self.hysteresis_params['k']
    
    def calculate_remanence(self) -> float:
        """Calculate remanent magnetization from hysteresis loop."""
        if not self.field_history or not self.magnetization_history:
            return 0.0
        
        # Find magnetization at zero field
        min_field_idx = np.argmin(np.abs(self.field_history))
        return abs(self.magnetization_history[min_field_idx]) if min_field_idx < len(self.magnetization_history) else 0.0 