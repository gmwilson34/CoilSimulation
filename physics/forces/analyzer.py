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
        if hasattr(self.forces, '_calculate_gradient_force'):
            f_grad = self.forces._calculate_gradient_force(current, position)
        else:
            f_grad = 0.0
            
        if hasattr(self.forces, '_calculate_reluctance_force'):
            f_rel = self.forces._calculate_reluctance_force(current, position)
        else:
            f_rel = 0.0
            
        if hasattr(self.forces, '_calculate_lorentz_force'):
            f_lor = self.forces._calculate_lorentz_force(current, position, velocity)
        else:
            f_lor = 0.0
        
        # Store basic components
        analysis['gradient_force'] = f_grad
        analysis['reluctance_force'] = f_rel
        analysis['lorentz_force'] = f_lor
        
        # Eddy current analysis (if available)
        if hasattr(self.forces, '_calculate_eddy_current_force'):
            f_eddy, eddy_power = self.forces._calculate_eddy_current_force(
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
        analysis['validation'] = self._validate_force_physics(
            analysis, current, position, velocity
        )
        
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
        dominant = max(forces.keys(), key=lambda k: forces[k])
        
        # Check if it's significantly larger than others
        max_force = forces[dominant]
        other_forces = [f for name, f in forces.items() if name != dominant]
        
        if max_force > 2.0 * max(other_forces) if other_forces else True:
            return dominant
        else:
            return 'mixed'
    
    def _validate_force_physics(self, analysis: dict, current: float = 0.0, position: float = 0.0, velocity: float = 0.0) -> dict:
        """Validate force calculations against physical principles with position awareness."""
        validation = {}
        
        # Energy conservation check
        if hasattr(self.forces, 'validate_energy_conservation'):
            validation['energy_conservation'] = self.forces.validate_energy_conservation(
                analysis['total_force'], current, position
            )
        else:
            validation['energy_conservation'] = True
        
        # Get force components
        gradient_force = analysis['gradient_force']
        reluctance_force = analysis['reluctance_force']
        lorentz_force = analysis['lorentz_force']
        eddy_force = analysis['eddy_current_force']
        
        # Position-aware gradient force validation
        # Gradient force should attract toward field maximum (usually coil center)
        coil_center = 0.0  # Assuming coil centered at z=0
        distance_to_center = position - coil_center
        
        # For positions far from coil, gradient force should point toward coil
        if abs(distance_to_center) > 0.01:  # 1cm threshold for near-center region
            expected_gradient_sign = -np.sign(distance_to_center)  # Attractive toward center
            actual_gradient_sign = np.sign(gradient_force) if abs(gradient_force) > 1e-12 else 0
            validation['gradient_sign_consistent'] = (
                actual_gradient_sign == expected_gradient_sign or abs(gradient_force) < 1e-12
            )
        else:
            # Near center, gradient force can vary - less strict validation
            validation['gradient_sign_consistent'] = True
        
        # Reluctance force validation - FIXED: can change sign past coil center
        # Reluctance force depends on dL/dz, which changes sign across coil
        # For symmetric coil: F > 0 when approaching center, F < 0 when leaving
        if hasattr(self.forces, 'field_calc') and hasattr(self.forces.field_calc, 'coil_length'):
            coil_half_length = self.forces.field_calc.coil_length / 2
            if position < -coil_half_length:
                # Before coil: should be attractive (positive)
                validation['reluctance_sign_consistent'] = reluctance_force >= -1e-12  # Allow small negative due to numerics
            elif position > coil_half_length:
                # After coil: can be repulsive (negative) as dL/dz < 0
                validation['reluctance_sign_consistent'] = True  # Both signs valid
            else:
                # Within coil: complex behavior, both signs possible
                validation['reluctance_sign_consistent'] = True
        else:
            # Fallback: allow both signs as dL/dz can change
            validation['reluctance_sign_consistent'] = True
        
        # Lenz's law validation - FIXED: forces should oppose motion
        if abs(velocity) > 1e-6:
            # Lorentz force should oppose velocity: F·v < 0
            lorentz_opposes_motion = (lorentz_force * velocity) <= 1e-12  # Allow small positive due to numerics
            eddy_opposes_motion = (eddy_force * velocity) <= 1e-12
            
            validation['lenz_law_consistent'] = lorentz_opposes_motion and eddy_opposes_motion
            validation['lorentz_opposes_motion'] = lorentz_opposes_motion
            validation['eddy_opposes_motion'] = eddy_opposes_motion
        else:
            # No motion, Lenz's law not applicable
            validation['lenz_law_consistent'] = True
            validation['lorentz_opposes_motion'] = True
            validation['eddy_opposes_motion'] = True
        
        # Momentum conservation for symmetric cases
        validation['momentum_conservation'] = self._check_momentum_conservation(
            analysis, position, velocity
        )
        
        # Overall validation
        validation['overall_valid'] = all([
            validation['energy_conservation'],
            validation['gradient_sign_consistent'],
            validation['reluctance_sign_consistent'],
            validation['lenz_law_consistent'],
            validation['momentum_conservation']
        ])
        
        return validation
    
    def compare_force_models(self, current: float, position: float, velocity: float = 0.0) -> dict:
        """Compare different force calculation approaches if available."""
        comparison = {}
        
        # Basic classical force
        classical_force = 0.0
        if hasattr(self.forces, '_calculate_gradient_force'):
            classical_force += self.forces._calculate_gradient_force(current, position)
        if hasattr(self.forces, '_calculate_reluctance_force'):
            classical_force += self.forces._calculate_reluctance_force(current, position)
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
    
    def diagnose_force_issues(self, analysis: dict, current: float = 0.0, position: float = 0.0) -> List[str]:
        """Diagnose potential issues with force calculations using realistic thresholds."""
        issues = []
        
        # Get force magnitude and determine realistic thresholds based on system scale
        force_magnitude = analysis['force_magnitude']
        
        # Realistic force thresholds based on coilgun scale
        if hasattr(self.forces, 'field_calc'):
            # Estimate system scale from coil parameters
            coil_volume = getattr(self.forces.field_calc, 'coil_length', 0.1) * \
                         np.pi * getattr(self.forces.field_calc, 'coil_radius', 0.01)**2
            
            # Scale thresholds with system size
            # Small coilgun (~1cm): max ~100N, Large coilgun (~10cm): max ~10kN
            max_reasonable_force = min(10000.0, max(100.0, coil_volume * 1e6))  # N
            min_detectable_force = max(0.001, coil_volume * 0.1)  # mN to N range
        else:
            # Default thresholds for unknown system
            max_reasonable_force = 1000.0  # 1kN default
            min_detectable_force = 0.01   # 10mN default
        
        # Check for unrealistic force magnitudes
        if force_magnitude > max_reasonable_force:
            issues.append(f"Force magnitude ({force_magnitude:.1f}N) exceeds reasonable limit "
                         f"({max_reasonable_force:.1f}N) - check for numerical instability")
        
        # Check for NaN or infinite values
        if not np.isfinite(analysis['total_force']):
            issues.append("Non-finite force value detected")
        
        # Check individual components for NaN/inf
        for component in ['gradient_force', 'reluctance_force', 'lorentz_force', 'eddy_current_force']:
            if component in analysis and not np.isfinite(analysis[component]):
                issues.append(f"Non-finite {component} detected")
        
        # Check force ratios for anomalies - more reasonable thresholds
        ratios = analysis['force_ratios']
        max_reasonable_ratio = 100.0  # One component shouldn't dominate by more than 100x
        
        for component, ratio in ratios.items():
            if abs(ratio) > max_reasonable_ratio:
                issues.append(f"Extreme {component} ratio ({ratio:.1f}) - one force component dominates excessively")
        
        # Check validation results
        if 'validation' in analysis and not analysis['validation']['overall_valid']:
            issues.append("Force calculation failed physical validation")
            
            # Provide specific validation failure details
            validation = analysis['validation']
            if not validation.get('energy_conservation', True):
                issues.append("Energy conservation violation detected")
            if not validation.get('gradient_sign_consistent', True):
                issues.append("Gradient force sign inconsistent with position")
            if not validation.get('lenz_law_consistent', True):
                issues.append("Lenz's law violation - forces don't oppose motion")
            if not validation.get('momentum_conservation', True):
                issues.append("Momentum conservation check failed")
        
        # Check for zero forces when current is significant
        if force_magnitude < min_detectable_force and abs(current) > 1.0:
            issues.append(f"Very small force ({force_magnitude:.2e}N) with significant current "
                         f"({current:.1f}A) - possible calculation error")
        
        # Check for oscillatory behavior (if we have force history)
        if hasattr(analysis, 'force_history') and len(analysis['force_history']) > 3:
            recent_forces = analysis['force_history'][-4:]
            if self._detect_oscillation(recent_forces):
                issues.append("Oscillatory force behavior detected - possible numerical instability")
        
        # Check component balance for very small total forces
        if force_magnitude < min_detectable_force:
            component_magnitudes = [
                abs(analysis.get('gradient_force', 0)),
                abs(analysis.get('reluctance_force', 0)),
                abs(analysis.get('lorentz_force', 0)),
                abs(analysis.get('eddy_current_force', 0))
            ]
            max_component = max(component_magnitudes)
            
            if max_component > 10 * force_magnitude:
                issues.append("Large force components nearly cancel - possible numerical precision issue")
        
        return issues
    
    def _check_momentum_conservation(self, analysis: dict, position: float, velocity: float) -> bool:
        """Check momentum conservation for symmetric cases."""
        # For symmetric coil geometry, check if forces behave symmetrically
        # This is a simplified check - full momentum conservation requires time integration
        
        # If we have access to the force calculator's geometry
        if hasattr(self.forces, 'field_calc') and hasattr(self.forces.field_calc, 'coil_length'):
            coil_center = 0.0  # Assuming centered coil
            coil_half_length = self.forces.field_calc.coil_length / 2
            
            # For positions symmetric about center, check if gradient forces have expected symmetry
            distance_from_center = abs(position - coil_center)
            
            # Within coil bounds, gradient force should have reasonable magnitude
            if distance_from_center <= coil_half_length:
                # Inside coil: force magnitude should be reasonable
                gradient_magnitude = abs(analysis['gradient_force'])
                # Rough check: gradient force shouldn't exceed characteristic magnetic force
                max_reasonable_gradient = 1000.0  # 1kN for typical coilgun
                return gradient_magnitude <= max_reasonable_gradient
            else:
                # Outside coil: force should decay with distance
                # This is a simplified check - could be more sophisticated
                return True
        
        # Default to valid if we can't perform geometry-based checks
        return True
    
    def _detect_oscillation(self, force_history: List[float]) -> bool:
        """Detect oscillatory behavior in force history."""
        if len(force_history) < 4:
            return False
        
        # Simple oscillation detection: check for sign changes
        signs = [np.sign(f) for f in force_history if abs(f) > 1e-12]
        
        if len(signs) < 3:
            return False
        
        # Count sign changes
        sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
        
        # If more than half the intervals have sign changes, consider it oscillatory
        return sign_changes > len(signs) // 2