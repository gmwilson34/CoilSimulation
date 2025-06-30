"""
Core Solver Components

This module contains base classes, constants, and common functionality
for the coilgun simulation solver engine.
"""

import numpy as np
import json
import time
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod

from physics import CoilgunPhysicsEngine
from physics.core import ValidationError


class SolverConstants:
    """Constants for solver configuration and safety limits."""
    
    # Default solver settings
    DEFAULT_METHOD = 'RK45'
    DEFAULT_RTOL = 1e-8
    DEFAULT_ATOL = 1e-10
    DEFAULT_MAX_STEP = 1e-4
    
    # Progress tracking settings
    DEFAULT_UPDATE_INTERVAL = 0.1
    DEFAULT_BAR_WIDTH = 50
    
    # Termination thresholds
    DEFAULT_EXIT_VELOCITY_THRESHOLD = 1e-3
    DEFAULT_CURRENT_THRESHOLD = 1e-3
    DEFAULT_MIN_FORCE_THRESHOLD = 1e-6


class SolverConfig:
    """Configuration class for solver settings."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize solver configuration."""
        self.config = config_dict or {}
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default values for solver parameters."""
        solver_cfg = self.config.setdefault('solver', {})
        
        # Integration settings
        solver_cfg.setdefault('method', SolverConstants.DEFAULT_METHOD)
        solver_cfg.setdefault('rtol', SolverConstants.DEFAULT_RTOL)
        solver_cfg.setdefault('atol', SolverConstants.DEFAULT_ATOL)
        solver_cfg.setdefault('max_step', SolverConstants.DEFAULT_MAX_STEP)
        
        # Progress settings
        progress_cfg = solver_cfg.setdefault('progress', {})
        progress_cfg.setdefault('update_interval', SolverConstants.DEFAULT_UPDATE_INTERVAL)
        progress_cfg.setdefault('bar_width', SolverConstants.DEFAULT_BAR_WIDTH)
        
        # Termination settings
        term_cfg = solver_cfg.setdefault('termination', {})
        term_cfg.setdefault('exit_velocity_threshold', SolverConstants.DEFAULT_EXIT_VELOCITY_THRESHOLD)
        term_cfg.setdefault('current_threshold', SolverConstants.DEFAULT_CURRENT_THRESHOLD)
        term_cfg.setdefault('min_force_threshold', SolverConstants.DEFAULT_MIN_FORCE_THRESHOLD)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


class BaseSolver(ABC):
    """Base class for all solver implementations."""
    
    def __init__(self, config_file: str, solver_config: Optional[SolverConfig] = None):
        """Initialize base solver."""
        self.config_file = config_file
        self.solver_config = solver_config or SolverConfig()
        
        # Load physics configuration
        with open(config_file, 'r') as f:
            self.physics_config = json.load(f)
        
        # Initialize physics engine
        self.physics = CoilgunPhysicsEngine(config_file)
        
        # Initialize solver state
        self.solution = None
        self.results = {}
        self.simulation_time = 0.0
        self.step_count = 0
        
        # Initialize callbacks
        self.event_callbacks: List[Callable] = []
        self.step_callbacks: List[Callable] = []
    
    @abstractmethod
    def run_simulation(self, **kwargs) -> Dict[str, Any]:
        """Run the simulation. Must be implemented by subclasses."""
        pass
    
    def add_event_callback(self, callback: Callable):
        """Add an event detection callback."""
        self.event_callbacks.append(callback)
    
    def add_step_callback(self, callback: Callable):
        """Add a step callback for monitoring."""
        self.step_callbacks.append(callback)
    
    def get_initial_conditions(self) -> Tuple[float, float, float, float]:
        """Get initial conditions for the simulation."""
        return self.physics.get_initial_conditions()
    
    def create_ode_function(self) -> Callable:
        """Create the ODE function for integration."""
        return self.physics.circuit_derivatives
    
    def _create_events(self) -> List[Callable]:
        """Create event functions for simulation termination."""
        events = []
        
        # Projectile at center event
        def projectile_at_center(t, y):
            return y[2]  # position = 0
        projectile_at_center.terminal = False
        projectile_at_center.direction = 1
        events.append(projectile_at_center)
        
        # Projectile exits coil event
        def projectile_exits_coil(t, y):
            coil_end = self.physics.coil_length / 2.0
            exit_threshold = coil_end + self.physics.proj_length / 2.0
            return y[2] - exit_threshold
        projectile_exits_coil.terminal = True
        projectile_exits_coil.direction = 1
        events.append(projectile_exits_coil)
        
        # Current reverses event
        def current_reverses(t, y):
            return y[1]  # current
        current_reverses.terminal = False
        current_reverses.direction = -1
        events.append(current_reverses)
        
        # Low current threshold event
        def low_current_threshold(t, y):
            threshold = self.solver_config.get('termination.current_threshold', 
                                             SolverConstants.DEFAULT_CURRENT_THRESHOLD)
            return abs(y[1]) - threshold
        low_current_threshold.terminal = False
        low_current_threshold.direction = -1
        events.append(low_current_threshold)
        
        return events
    
    def _enhanced_ode_wrapper(self, original_func: Callable) -> Callable:
        """Create enhanced ODE wrapper with step callbacks."""
        def wrapped_func(t, y):
            try:
                # Call original function
                dydt = original_func(t, y)
                
                # Execute step callbacks
                for callback in self.step_callbacks:
                    try:
                        callback(t, y, dydt)
                    except Exception as e:
                        print(f"Warning: Step callback failed: {e}")
                
                self.step_count += 1
                return dydt
                
            except Exception as e:
                print(f"Error in ODE function at t={t}: {e}")
                raise
        
        return wrapped_func


class SolverUtils:
    """Utility functions for solver operations."""
    
    @staticmethod
    def find_config_files(directory: str = ".") -> List[Path]:
        """Find all coilgun configuration files in directory."""
        config_files = set()  # Use set to avoid duplicates
        search_path = Path(directory)
        
        # Look for JSON files with coilgun-related names first, then general ones
        patterns = [
            "*coilgun*.json",
            "*config*.json", 
        ]
        
        for pattern in patterns:
            config_files.update(search_path.glob(pattern))
        
        # Add any other JSON files that don't match the above patterns
        all_json_files = search_path.glob("*.json")
        config_files.update(all_json_files)
        
        # Filter out non-config files and convert back to list
        valid_configs = []
        for file in config_files:
            try:
                with open(file, 'r') as f:
                    config = json.load(f)
                    # Check if it looks like a coilgun config
                    if any(key in config for key in ['coil', 'projectile', 'capacitor']):
                        valid_configs.append(file)
            except (json.JSONDecodeError, IOError):
                continue
        
        return sorted(list(set(valid_configs)))
    
    @staticmethod
    def select_config_file() -> Optional[str]:
        """Interactive config file selection."""
        config_files = SolverUtils.find_config_files()
        
        if not config_files:
            print("No coilgun configuration files found.")
            return None
        
        if len(config_files) == 1:
            print(f"Using configuration file: {config_files[0]}")
            return str(config_files[0])
        
        print("\nAvailable configuration files:")
        for i, file in enumerate(config_files, 1):
            print(f"{i}. {file.name}")
        
        while True:
            try:
                choice = input(f"\nSelect configuration file (1-{len(config_files)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(config_files):
                    return str(config_files[idx])
                else:
                    print("Invalid selection. Please try again.")
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Please enter a number.")
    
    @staticmethod
    def validate_solver_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate solver configuration."""
        errors = []
        
        # Check required sections
        required_sections = ['coil', 'projectile', 'capacitor']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate solver-specific settings
        if 'solver' in config:
            solver_cfg = config['solver']
            
            # Validate integration method
            valid_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
            method = solver_cfg.get('method', SolverConstants.DEFAULT_METHOD)
            if method not in valid_methods:
                errors.append(f"Invalid integration method: {method}")
            
            # Validate tolerances
            rtol = solver_cfg.get('rtol', SolverConstants.DEFAULT_RTOL)
            atol = solver_cfg.get('atol', SolverConstants.DEFAULT_ATOL)
            
            if not (1e-12 <= rtol <= 1e-3):
                errors.append(f"Relative tolerance out of range: {rtol}")
            
            if not (1e-15 <= atol <= 1e-6):
                errors.append(f"Absolute tolerance out of range: {atol}")
        
        return len(errors) == 0, errors


class SolverError(Exception):
    """Custom exception for solver errors."""
    pass


class SimulationError(Exception):
    """Custom exception for simulation runtime errors."""
    pass 