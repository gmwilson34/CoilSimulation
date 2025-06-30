"""
Magnetic Field Calculations Module

This module provides comprehensive magnetic field calculation capabilities
organized into specialized submodules for better maintainability.

Submodules:
- core: Main field calculator classes and interfaces
- quantum: Quantum field theory corrections and advanced physics
- biot_savart: Biot-Savart law calculations
- mapping: Field visualization and mapping utilities
- corrections: Relativistic, thermal, and other field corrections
"""

from .core import AdvancedMagneticFieldCalculator, MagneticFieldCalculator
from .biot_savart import BiotSavartCalculator
from .mapping import FieldMapping
from .quantum import QuantumFieldEffects
from .corrections import FieldCorrections

__all__ = [
    'AdvancedMagneticFieldCalculator',
    'MagneticFieldCalculator',
    'BiotSavartCalculator', 
    'FieldMapping',
    'QuantumFieldEffects',
    'FieldCorrections'
] 