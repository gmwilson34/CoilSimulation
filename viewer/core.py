"""
Core visualization constants, base classes, and shared functionality.

This module provides the foundational elements for the coilgun visualization system,
including plotting setup, style configuration, and base visualization classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import json
import sys
import signal
from typing import Optional, Tuple, List, Any, Union

# Set up plotting style
plt.style.use('default')

# Optional seaborn import for enhanced styling
try:
    import seaborn as sns
    sns.set_palette("viridis")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class VisualizationConfig:
    """Configuration class for visualization settings."""
    
    # Default figure sizes
    DEFAULT_FIG_SIZE = (15, 10)
    LARGE_FIG_SIZE = (20, 16)
    ANIMATION_FIG_SIZE = (12, 8)
    
    # Color schemes
    FORCE_COLORS = {
        'gradient': 'b-',
        'reluctance': 'g--', 
        'eddy': 'r:',
        'lorentz': 'm-.',
        'maxwell': 'c--',
        'total': 'k-'
    }
    
    ENERGY_COLORS = {
        'capacitor': 'b-',
        'kinetic': 'r-',
        'total': 'k--',
        'magnetic': 'g-'
    }
    
    # Line styles and widths
    DEFAULT_LINE_WIDTH = 2
    THICK_LINE_WIDTH = 3
    THIN_LINE_WIDTH = 1.5
    
    # Grid and alpha settings
    GRID_ALPHA = 0.3
    PLOT_ALPHA = 0.8
    HIGHLIGHT_ALPHA = 0.9

class BaseVisualizer:
    """Base class for all visualization components."""
    
    def __init__(self, physics_engine=None):
        """
        Initialize base visualizer.
        
        Args:
            physics_engine: CoilgunPhysicsEngine instance
        """
        self.physics = physics_engine
        self.config = VisualizationConfig()
        self.fig_size = self.config.DEFAULT_FIG_SIZE
    
    def setup_figure(self, figsize=None, **kwargs):
        """
        Set up a matplotlib figure with consistent styling.
        
        Args:
            figsize: Figure size tuple
            **kwargs: Additional figure parameters
            
        Returns:
            Figure object
        """
        if figsize is None:
            figsize = self.fig_size
            
        fig = plt.figure(figsize=figsize, **kwargs)
        return fig
    
    def setup_subplot_grid(self, rows, cols, figsize=None, **kwargs):
        """
        Set up a subplot grid with consistent styling.
        
        Args:
            rows: Number of rows
            cols: Number of columns
            figsize: Figure size tuple
            **kwargs: Additional gridspec parameters
            
        Returns:
            Figure and GridSpec objects
        """
        if figsize is None:
            figsize = self.config.LARGE_FIG_SIZE
            
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(rows, cols, **kwargs)
        return fig, gs
    
    def apply_common_styling(self, ax, title=None, xlabel=None, ylabel=None, grid=True):
        """
        Apply common styling to a subplot.
        
        Args:
            ax: Matplotlib axes object
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
        """
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=self.config.GRID_ALPHA)
    
    def save_figure(self, fig, save_path, dpi=300, bbox_inches='tight'):
        """
        Save figure with consistent settings.
        
        Args:
            fig: Figure object
            save_path: Path to save the figure
            dpi: Resolution for saving
            bbox_inches: Bounding box setting
        """
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Figure saved to: {save_path}")

class CoilGeometry:
    """Helper class for coil geometry visualization."""
    
    @staticmethod
    def add_coil_boundaries(ax, physics_engine):
        """
        Add coil boundary visualization to a plot.
        
        Args:
            ax: Matplotlib axes object
            physics_engine: Physics engine with coil parameters
        """
        if not physics_engine:
            return
            
        # Add coil boundaries as rectangles
        coil_length = physics_engine.coil_length
        coil_inner_radius = physics_engine.coil_inner_radius
        coil_outer_radius = physics_engine.coil_outer_radius
        
        # Inner coil boundary
        inner_rect = patches.Rectangle((0, coil_inner_radius), coil_length, 
                                     coil_outer_radius - coil_inner_radius,
                                     linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax.add_patch(inner_rect)
        
        # Mirror for full coil visualization
        mirror_rect = patches.Rectangle((0, -coil_outer_radius), coil_length,
                                      coil_outer_radius - coil_inner_radius,
                                      linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
        ax.add_patch(mirror_rect)
    
    @staticmethod
    def add_projectile_marker(ax, position, radius=0.001):
        """
        Add projectile position marker to a plot.
        
        Args:
            ax: Matplotlib axes object
            position: Projectile z-position
            radius: Projectile radius for visualization
        """
        if position is not None:
            # Add projectile as a circle
            projectile = patches.Circle((position, 0), radius, color='red', alpha=0.8)
            ax.add_patch(projectile)

def setup_signal_handling():
    """Set up signal handling for graceful exit."""
    def signal_handler(signum, frame):
        print("\nVisualization interrupted. Cleaning up...")
        plt.close('all')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

# Global constants for visualization
VISUALIZATION_CONSTANTS = {
    'DEFAULT_Z_RANGE': (-0.05, 0.15),
    'DEFAULT_R_RANGE': (0, 0.05),
    'DEFAULT_RESOLUTION': {'z': 100, 'r': 50, 'x': 30, 'y': 30},
    'FIELD_LINE_PARAMS': {'max_length': 0.2, 'step_size': 0.001},
    'ANIMATION_PARAMS': {'interval': 50, 'frames': 100}
} 