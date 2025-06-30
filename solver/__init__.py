"""
Advanced Coilgun Simulation Solver

This package provides a modular solver engine for coilgun simulations,
implementing advanced ODE integration with progress monitoring and analysis.
Enhanced with advanced physics engine integration.

Usage:
    from solver import CoilgunSolver  # Enhanced by default
    from solver import EnhancedCoilgunSolver
    from solver.single_stage import EnhancedSingleStageSimulation
    from solver.multi_stage import EnhancedMultiStageSimulation
    from solver.progress import ProgressTracker
    from solver.analysis import ResultsAnalyzer
"""

# Import enhanced solver classes (default)
from .engine import EnhancedCoilgunSolver, CoilgunSolver
from .single_stage import EnhancedSingleStageSimulation, SingleStageSimulation
from .multi_stage import EnhancedMultiStageSimulation

# Import optimization classes
from .optimization import ParametricStudy, CoilgunOptimizer

# Import core configuration
from .core import SolverConfig

# Import utility functions
from .utils import cleanup_temp_files

# Import individual modules for fine-grained access
from . import core
from . import progress
from . import analysis
from . import optimization
from . import utils

# Version information
__version__ = "2.1.0"  # Updated version for enhanced capabilities
__author__ = "Graham Wilson"

# Define what gets imported with "from solver import *"
__all__ = [
    # Enhanced classes (default)
    'CoilgunSolver',  # Points to EnhancedCoilgunSolver
    'EnhancedCoilgunSolver',
    'EnhancedSingleStageSimulation',
    'EnhancedMultiStageSimulation',
    
    # Backward compatibility
    'SingleStageSimulation',
    
    # Optimization and utilities
    'ParametricStudy',
    'CoilgunOptimizer',
    'SolverConfig',
    'cleanup_temp_files',
    
    # Modules
    'core',
    'progress',
    'analysis', 
    'optimization',
    'utils'
]

# Backward compatibility aliases
MultiStageSimulation = EnhancedMultiStageSimulation
MultiStageCoilgunSimulation = EnhancedMultiStageSimulation 