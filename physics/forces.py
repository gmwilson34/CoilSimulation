"""
Electromagnetic Force Calculations - Main Interface

This module provides the main interface to the refactored electromagnetic force
calculation system. The actual implementations have been split into specialized
modules for better organization and maintainability.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
import warnings
from .core import BasePhysicsModel, PhysicsConstants, NumericalUtils, SafetyLimits
from .fields import AdvancedMagneticFieldCalculator, MagneticFieldCalculator
from .materials import AdvancedMaterialProperties, AdvancedPermeabilityModel, MaterialProperties

# Import the modular force calculators
from .forces import (
    AdvancedElectromagneticForces,
    ElectromagneticForcesBalanced,
    ForceAnalyzer
)


def create_electromagnetic_forces(config: dict, field_calculator, materials, force_type: str = 'advanced'):
    """
    Factory function to create electromagnetic force calculators.
    
    Args:
        config: Configuration dictionary
        field_calculator: Magnetic field calculator instance
        materials: Materials properties instance
        force_type: Type of force calculator ('advanced', 'balanced', 'basic')
    
    Returns:
        Appropriate electromagnetic force calculator instance
    """
    if force_type == 'advanced':
        return AdvancedElectromagneticForces(config, field_calculator, materials)
    elif force_type == 'balanced':
        return ElectromagneticForcesBalanced(config, field_calculator, materials)
    else:
        # Default to balanced for compatibility
        return ElectromagneticForcesBalanced(config, field_calculator, materials)


def create_force_analyzer(electromagnetic_forces):
    """
    Create a force analyzer for the given electromagnetic forces calculator.
    
    Args:
        electromagnetic_forces: An electromagnetic forces calculator instance
    
    Returns:
        ForceAnalyzer instance
    """
    return ForceAnalyzer(electromagnetic_forces)


# Backward compatibility exports
__all__ = [
    'create_electromagnetic_forces',
    'create_force_analyzer',
    'AdvancedElectromagneticForces',
    'ElectromagneticForcesBalanced',
    'ForceAnalyzer'
] 