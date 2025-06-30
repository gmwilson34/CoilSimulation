"""
Advanced Electromagnetic Physics Engine for Coilgun Simulation

This package provides a modular physics engine implementing Maxwell's equations,
electromagnetic field calculations, and coilgun-specific physics.

Usage:
    from physics import CoilgunPhysicsEngine
    from physics.fields import MagneticFieldCalculator
    from physics.forces import ElectromagneticForces
    from physics.materials import MaterialProperties
    from physics.circuits import CircuitModel
    from physics.core import PhysicsConstants
"""

# Import main physics engine class
from .engine import CoilgunPhysicsEngine

# Import individual modules for fine-grained access
from . import core
from . import fields
from . import forces
from . import materials
from . import circuits
from . import utils

# Version information
__version__ = "2.0.0"
__author__ = "Graham Wilson"

# Define what gets imported with "from physics import *"
__all__ = [
    'CoilgunPhysicsEngine',
    'core',
    'fields', 
    'forces',
    'materials',
    'circuits',
    'utils'
] 