"""
Magnetic Field Calculations - Legacy Interface

This module provides backward compatibility for the refactored magnetic field
calculation system. The functionality has been reorganized into specialized
submodules for better maintainability.

NEW MODULAR STRUCTURE:
- fields.core: Main field calculator classes
- fields.quantum: Quantum field theory corrections
- fields.corrections: Relativistic and thermal corrections
- fields.biot_savart: Biot-Savart law calculations
- fields.mapping: Field visualization and analysis

For new code, import from the specific submodules:
    from physics.fields import AdvancedMagneticFieldCalculator
    from physics.fields.quantum import QuantumFieldEffects
    from physics.fields.mapping import FieldMapping
"""

# Import everything from the new modular structure for backward compatibility
from .fields.core import (
    AdvancedMagneticFieldCalculator,
    MagneticFieldCalculator
)
from .fields.biot_savart import BiotSavartCalculator
from .fields.mapping import FieldMapping
from .fields.quantum import QuantumFieldEffects
from .fields.corrections import FieldCorrections

# Legacy aliases for backward compatibility
FieldVisualization = FieldMapping

__all__ = [
    'AdvancedMagneticFieldCalculator',
    'MagneticFieldCalculator', 
    'BiotSavartCalculator',
    'FieldMapping',
    'FieldVisualization',  # Legacy alias
    'QuantumFieldEffects',
    'FieldCorrections'
] 