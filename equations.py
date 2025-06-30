# equations.py
"""
Advanced Electromagnetic Physics Engine for Coilgun Simulation

This module provides backward compatibility wrapper for the new modular physics engine.
The actual implementation has been moved to the physics package for better maintainability.

For new code, please use:
    from physics import CoilgunPhysicsEngine
    
For compatibility with existing code, this module re-exports the main class.
"""

# Import the modular physics engine
from physics import CoilgunPhysicsEngine

# Re-export for backward compatibility
__all__ = ['CoilgunPhysicsEngine']