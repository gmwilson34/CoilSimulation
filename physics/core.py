"""
Core Physics Constants and Base Classes

This module contains fundamental physics constants, safety limits,
and base classes used throughout the physics engine.
"""

import numpy as np
import warnings
from typing import Optional, Union, Tuple, Any


class PhysicsConstants:
    """Fundamental physics constants for electromagnetic calculations."""
    
    # Electromagnetic constants
    MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    C = 299792458  # Speed of light (m/s)
    
    # Thermodynamic constants
    K_BOLTZMANN = 1.380649e-23  # Boltzmann constant (J/K)
    
    # Standard conditions
    ROOM_TEMPERATURE = 293.15  # K (20°C)
    STANDARD_PRESSURE = 101325  # Pa


class SafetyLimits:
    """Safety constants for numerical stability and realistic bounds."""
    
    # Maximum values - set high enough for realistic high-power coilguns
    MAX_CURRENT = 1e7      # Maximum current in Amperes (10 MA)
    MAX_FORCE = 1e8        # Maximum force in Newtons (100 MN)
    MAX_VOLTAGE = 1e7      # Maximum voltage in Volts (10 MV)
    MAX_FIELD = 50.0       # Maximum magnetic field in Tesla (50 T)
    MAX_ENERGY = 1e15      # Maximum energy in Joules (1 PJ)
    MAX_POWER = 1e15       # Maximum power in Watts (1 PW)
    
    # Minimum values to prevent division by zero
    MIN_INDUCTANCE = 1e-12  # Minimum inductance in H
    MIN_RESISTANCE = 1e-9   # Minimum resistance in Ohms
    MIN_CAPACITANCE = 1e-12 # Minimum capacitance in F
    MIN_MASS = 1e-6        # Minimum mass in kg
    
    # Numerical precision limits
    NUMERICAL_EPSILON = 1e-15
    FORCE_EPSILON = 1e-9
    CURRENT_EPSILON = 1e-12


class NumericalUtils:
    """Numerical utility functions for safe mathematical operations."""
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is too small."""
        if abs(denominator) < SafetyLimits.NUMERICAL_EPSILON:
            return default
        return numerator / denominator
    
    @staticmethod
    def safe_sqrt(value: float) -> float:
        """Safely compute square root, ensuring non-negative input."""
        return np.sqrt(max(0.0, value))
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between minimum and maximum bounds."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def safe_numerical_operation(value: float, operation_name: str, max_value: Optional[float] = None) -> float:
        """Perform safe numerical operations with bounds checking."""
        if not np.isfinite(value):
            warnings.warn(f"Non-finite value in {operation_name}: {value}")
            return 0.0
        
        if max_value is not None and abs(value) > max_value:
            warnings.warn(f"Value {value} exceeds maximum {max_value} in {operation_name}")
            return np.sign(value) * max_value
        
        return value


class BasePhysicsModel:
    """Base class for physics models with common functionality."""
    
    def __init__(self, config: dict):
        """Initialize base physics model."""
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation path."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PhysicsCalculationError(Exception):
    """Custom exception for physics calculation errors."""
    pass


def validate_physical_parameter(value: float, name: str, min_val: Optional[float] = None, 
                              max_val: Optional[float] = None, allow_zero: bool = True) -> float:
    """Validate a physical parameter against bounds and finite checks."""
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if not allow_zero and value == 0:
        raise ValidationError(f"{name} must be non-zero")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    
    return value 