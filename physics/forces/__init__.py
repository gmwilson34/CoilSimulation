"""
Forces Package - Electromagnetic Force Calculations

This package contains modular electromagnetic force calculation methods for coilgun simulation.
"""

from .base import BaseElectromagneticForces
from .advanced import AdvancedElectromagneticForces
from .balanced import ElectromagneticForcesBalanced
from .quantum import QuantumForceCalculator
from .maxwell_stress import MaxwellStressTensor
from .eddy_currents import EddyCurrentForces
from .hysteresis import HysteresisForces
from .relativistic import RelativisticForces
from .multiscale import MultiscaleForces
from .analyzer import ForceAnalyzer

__all__ = [
    'BaseElectromagneticForces',
    'AdvancedElectromagneticForces', 
    'ElectromagneticForcesBalanced',
    'QuantumForceCalculator',
    'MaxwellStressTensor',
    'EddyCurrentForces',
    'HysteresisForces',
    'RelativisticForces',
    'MultiscaleForces',
    'ForceAnalyzer'
] 