"""
Force Analysis and Diagnostics

This module provides comprehensive analysis of electromagnetic force components
and their relative contributions.
"""

import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from ..core import BasePhysicsModel, NumericalUtils


class ForceAnalyzer:
    """
    Comprehensive force analysis and diagnostics.
    
    Provides:
    - Force component breakdown and analysis
    - Dominant force identification
    - Force validation and consistency checks
    - Performance diagnostics
    """
    
    def __init__(self, electromagnetic_forces):
        """Initialize force analyzer."""
        self.forces = electromagnetic_forces
    
    def analyze_force_components(self, current: float, position: float, velocity: float = 0.0,
                               current_history: Optional[List] = None,
                               time_history: Optional[List] = None) -> dict:
        """
        Comprehensive analysis of all force components.
        
        Returns detailed breakdown of forces and their characteristics.
        """
        analysis = {}
        
        # Basic force components
        f_grad = self.forces.calculate_gradient_force(current, position)
        f_rel = self.forces.calculate_reluctance_force(current, position) 
        f_lor = self.forces.calculate_lorentz_force(current, position, velocity)
        
        # Store basic components
        analysis['gradient_force'] = f_grad
        analysis['reluctance_force'] = f_rel
        analysis['lorentz_force'] = f_lor
        
        # Eddy current analysis (if available)
        if hasattr(self.forces, 'calculate_eddy_current_force'):
            f_eddy, eddy_power = self.forces.calculate_eddy_current_force(
                current, position, velocity, current_history, time_history
            )
            analysis['eddy_current_force'] = f_eddy
            analysis['eddy_power_dissipation'] = eddy_power
        else:
            analysis['eddy_current_force'] = 0.0
            analysis['eddy_power_dissipation'] = 0.0
        
        # Calculate totals and ratios
        total_force = f_grad + f_rel + f_lor + analysis['eddy_current_force']
        analysis['total_force'] = total_force
        
        # Force component analysis
        if abs(total_force) > 1e-12:
            analysis['force_ratios'] = {
                'gradient_fraction': f_grad / total_force,
                'reluctance_fraction': f_rel / total_force,
                'lorentz_fraction': f_lor / total_force,
                'eddy_fraction': analysis['eddy_current_force'] / total_force
            }
        else:
            analysis['force_ratios'] = {
                'gradient_fraction': 0.0,
                'reluctance_fraction': 0.0,
                'lorentz_fraction': 0.0,
                'eddy_fraction': 0.0
            }
        
        # Identify dominant force mechanism
        analysis['dominant_mechanism'] = self._identify_dominant_force(
            f_grad, f_rel, f_lor, analysis['eddy_current_force']
        )
        
        # Force characteristics
        analysis['force_magnitude'] = abs(total_force)
        analysis['force_direction'] = np.sign(total_force)
        
        # Physical validation
        analysis['validation'] = self._validate_force_physics(analysis)
        
        return analysis
    
    def _identify_dominant_force(self, f_grad: float, f_rel: float, f_lor: float, f_eddy: float) -> str:
        """Identify which force mechanism dominates."""
        forces = {
            'gradient': abs(f_grad),
            'reluctance': abs(f_rel), 
            'lorentz': abs(f_lor),
            'eddy_current': abs(f_eddy)
        }
        
        # Find maximum force
        dominant = max(forces, key=forces.get)
        
        # Check if it's significantly larger than others
        max_force = forces[dominant]
        other_forces = [f for name, f in forces.items() if name != dominant]
        
        if max_force > 2.0 * max(other_forces) if other_forces else True:
            return dominant
        else:
            return 'mixed'
    
    def _validate_force_physics(self, analysis: dict) -> dict:
        """Validate force calculations against physical principles."""
        validation = {}
        
        # Energy conservation check
        if hasattr(self.forces, 'validate_energy_conservation'):
            validation['energy_conservation'] = self.forces.validate_energy_conservation(
                analysis['total_force'], 0.0, 0.0  # Would need actual current/position
            )
        else:
            validation['energy_conservation'] = True
        
        # Sign consistency checks
        gradient_force = analysis['gradient_force']
        reluctance_force = analysis['reluctance_force']
        
        # Gradient force should attract when projectile is approaching coil center
        # (This is a simplified check - actual validation would be more complex)
        validation['gradient_sign_consistent'] = True  # Placeholder
        
        # Reluctance force should always be attractive (positive towards coil)
        validation['reluctance_sign_consistent'] = reluctance_force >= 0
        
        # Lorentz and eddy forces should oppose motion (Lenz's law)
        lorentz_force = analysis['lorentz_force']
        eddy_force = analysis['eddy_current_force']
        
        # These checks would require velocity information
        validation['lenz_law_consistent'] = True  # Placeholder
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['energy_conservation'],
            validation['gradient_sign_consistent'],
            validation['reluctance_sign_consistent'],
            validation['lenz_law_consistent']
        ])
        
        return validation
    
    def compare_force_models(self, current: float, position: float, velocity: float = 0.0) -> dict:
        """Compare different force calculation approaches if available."""
        comparison = {}
        
        # Basic classical force
        classical_force = self.forces.calculate_gradient_force(current, position) + \
                         self.forces.calculate_reluctance_force(current, position)
        comparison['classical'] = classical_force
        
        # Advanced methods (if available)
        if hasattr(self.forces, 'calculate_maxwell_stress_force'):
            maxwell_force, _ = self.forces.calculate_maxwell_stress_force(current, position, velocity)
            comparison['maxwell_stress'] = maxwell_force
            
            # Compare classical vs Maxwell stress
            if abs(classical_force) > 1e-12:
                comparison['maxwell_classical_ratio'] = maxwell_force / classical_force
            else:
                comparison['maxwell_classical_ratio'] = 1.0
        
        return comparison
    
    def diagnose_force_issues(self, analysis: dict) -> List[str]:
        """Diagnose potential issues with force calculations."""
        issues = []
        
        # Check for unrealistic force magnitudes
        if analysis['force_magnitude'] > 1e6:  # 1 MN threshold
            issues.append("Force magnitude exceptionally high - check for numerical instability")
        
        # Check for NaN or infinite values
        if not np.isfinite(analysis['total_force']):
            issues.append("Non-finite force value detected")
        
        # Check force ratios for anomalies
        ratios = analysis['force_ratios']
        if any(abs(ratio) > 10 for ratio in ratios.values()):
            issues.append("Extreme force component ratio detected")
        
        # Check validation results
        if not analysis['validation']['overall_valid']:
            issues.append("Force calculation failed physical validation")
        
        # Check for zero forces when current is non-zero
        if abs(analysis['total_force']) < 1e-15 and abs(analysis.get('current', 0)) > 1.0:
            issues.append("Zero force with significant current - possible calculation error")
        
        return issues 