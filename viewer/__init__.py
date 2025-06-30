"""
Advanced Coilgun Visualization Package

This package provides comprehensive visualization capabilities for coilgun simulations,
including magnetic field visualization, force analysis, 3D plotting, animations,
and multi-stage analysis.

Usage:
    from view import CoilgunVisualizationEngine
    from view.fields import MagneticFieldCalculator
    from view.plots import ContourPlotter, ForcePlotter
    from view.plots3d import Plot3DVisualizer
    from view.animations import AnimationEngine
    from view.analysis import PhysicsAnalyzer
    from view.multistage import MultistageVisualizer
    
Quick usage:
    from view import quick_visualization, launch_interactive_visualization
    
    # Quick analysis
    quick_visualization(simulation_results, physics_engine)
    
    # Interactive menu
    launch_interactive_visualization(physics_engine)
"""

# Import main visualization engine class
from .engine import (CoilgunVisualizationEngine, 
                    create_field_visualization_suite,
                    quick_visualization,
                    launch_interactive_visualization)

# Import individual modules for fine-grained access
from . import core
from . import fields
from . import plots
from . import plots3d
from . import animations
from . import analysis
from . import multistage
from . import utils

# Import commonly used classes for convenience
from .fields import MagneticFieldCalculator, FieldLineTracer
from .plots import ContourPlotter, ForcePlotter, ProfilePlotter
from .plots3d import Plot3DVisualizer, GeometryRenderer
from .animations import AnimationEngine
from .analysis import PhysicsAnalyzer, ElectromagneticAnalyzer
from .multistage import MultistageVisualizer

# Version information
__version__ = "2.0.0"
__author__ = "Graham Wilson"

# Define what gets imported with "from view import *"
__all__ = [
    # Main engine
    'CoilgunVisualizationEngine',
    
    # Convenience functions
    'create_field_visualization_suite',
    'quick_visualization', 
    'launch_interactive_visualization',
    
    # Field calculation and visualization
    'MagneticFieldCalculator',
    'FieldLineTracer',
    
    # 2D plotting classes
    'ContourPlotter',
    'ForcePlotter', 
    'ProfilePlotter',
    
    # 3D visualization classes
    'Plot3DVisualizer',
    'GeometryRenderer',
    
    # Animation engine
    'AnimationEngine',
    
    # Analysis classes
    'PhysicsAnalyzer',
    'ElectromagneticAnalyzer',
    
    # Multi-stage visualization
    'MultistageVisualizer',
    
    # Modules for advanced usage
    'core',
    'fields',
    'plots',
    'plots3d', 
    'animations',
    'analysis',
    'multistage',
    'utils'
]

# Package-level documentation
__doc__ += """

Module Overview:
- core: Base classes, configuration, and shared utilities
- fields: Magnetic field calculations and field-specific visualization
- plots: 2D plotting functions for contours, profiles, and force maps
- plots3d: 3D visualization including field surfaces and geometry rendering
- animations: Animation functions for projectile motion and field evolution
- analysis: Advanced physics analysis and diagnostics visualization
- multistage: Multi-stage coilgun specific visualizations
- utils: Utility functions for data processing and file management
- engine: Main visualization engine combining all components

Quick Start Example:
    ```python
    from physics import CoilgunPhysicsEngine
    from view import CoilgunVisualizationEngine
    
    # Initialize physics and visualization engines
    physics = CoilgunPhysicsEngine(config_file="my_config.json")
    viz = CoilgunVisualizationEngine(physics)
    
    # Create comprehensive visualization suite
    viz.create_comprehensive_visualization_suite(
        simulation_results=my_simulation_results
    )
    
    # Or use interactive menu
    viz.interactive_menu()
    ```

For interactive usage:
    ```python
    from view import launch_interactive_visualization
    from physics import CoilgunPhysicsEngine
    
    physics = CoilgunPhysicsEngine("config.json")
    launch_interactive_visualization(physics)
    ```
""" 