"""
Progress Tracking Module

This module provides enhanced progress tracking functionality with physics
diagnostics and real-time monitoring for coilgun simulations.
"""

import sys
import time
import threading
import numpy as np
from typing import Optional, List, Dict, Any, Callable

from physics import CoilgunPhysicsEngine
from .core import SolverConstants


class ProgressTracker:
    """
    Enhanced progress tracking class with physics diagnostics and monitoring.
    """
    
    def __init__(self, t_span: tuple, physics_engine: Optional[CoilgunPhysicsEngine] = None,
                 update_interval: Optional[float] = None, bar_width: Optional[int] = None):
        """
        Initialize enhanced progress tracker.
        
        Args:
            t_span: Time span tuple (t_start, t_end)
            physics_engine: Physics engine for diagnostics
            update_interval: Update interval in seconds
            bar_width: Width of progress bar
        """
        self.t_start, self.t_end = t_span
        self.t_duration = self.t_end - self.t_start
        self.physics = physics_engine
        
        # Configuration
        self.update_interval = update_interval or SolverConstants.DEFAULT_UPDATE_INTERVAL
        self.bar_width = bar_width or SolverConstants.DEFAULT_BAR_WIDTH
        
        # Progress tracking state
        self.current_time = self.t_start
        self.current_state = None
        self.step_count = 0
        self.start_real_time = time.time()
        self.last_update_time = self.start_real_time
        self.last_step_count = 0
        
        # Performance metrics
        self.current_integration_rate = 0.0
        self.average_integration_rate = 0.0
        
        # Physics diagnostics
        self.max_current = 0
        self.max_force = 0
        self.max_velocity = 0
        self.current_position = 0
        self.physics_warnings: List[str] = []
        self.displayed_warnings = set()
        
        # Display state
        self.running = True
        self.stopped = False
        self.progress_active = False
        self.last_progress_length = 0
        self.integration_started = False
        self.last_displayed_warning = None
        
        # Start progress display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def start_integration_display(self):
        """Start displaying the progress bar when integration begins."""
        self.integration_started = True
    
    def update(self, t: float, y: np.ndarray, dydt: Optional[np.ndarray] = None):
        """
        Update progress with current simulation state.
        
        Args:
            t: Current time
            y: Current state vector [Q, I, x, v, ...]
            dydt: State derivatives (optional)
        """
        self.current_time = t
        self.current_state = y
        self.step_count += 1
        
        # Start displaying on first update
        if not self.integration_started:
            self.integration_started = True
        
        # Update physics diagnostics
        if len(y) >= 4:
            self._update_physics_diagnostics(t, y)
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_physics_diagnostics(self, t: float, y: np.ndarray):
        """Update physics diagnostics from current state."""
        Q, I, x, v = y[:4]  # Handle thermal case with extra state variables
        
        # Update maximums
        self.max_current = max(self.max_current, abs(I))
        self.max_velocity = max(self.max_velocity, abs(v))
        self.current_position = x
        
        # Calculate force for diagnostics
        if self.physics and abs(I) > 1e-6:
            try:
                force_result = self._calculate_current_force(I, x, t, v)
                
                # Handle tuple return (force, eddy_power_loss)
                if isinstance(force_result, tuple):
                    force = force_result[0]
                else:
                    force = force_result
                
                self.max_force = max(self.max_force, abs(force))
                
            except Exception as e:
                # Store warning for later display
                warning_msg = f"Force calculation warning at t={t:.2e}s: {str(e)[:50]}"
                if warning_msg not in self.displayed_warnings:
                    self.physics_warnings.append(warning_msg)
    
    def _calculate_current_force(self, I: float, x: float, t: float, v: float = 0.0):
        """Calculate current force using appropriate physics method."""
        if self.physics is None:
            return 0.0
            
        if hasattr(self.physics, 'magnetic_force_with_circuit_logic'):
            return self.physics.magnetic_force_with_circuit_logic(I, x, t, v)
        else:
            return self.physics.magnetic_force_ferromagnetic(I, x, v)
    
    def _update_performance_metrics(self):
        """Update integration performance metrics."""
        current_real_time = time.time()
        elapsed_real_time = current_real_time - self.last_update_time
        
        if elapsed_real_time > 0:
            steps_since_last = self.step_count - self.last_step_count
            self.current_integration_rate = steps_since_last / elapsed_real_time
            
            # Update averages periodically
            if elapsed_real_time >= self.update_interval:
                total_elapsed = current_real_time - self.start_real_time
                if total_elapsed > 0:
                    self.average_integration_rate = self.step_count / total_elapsed
                
                self.last_update_time = current_real_time
                self.last_step_count = self.step_count
    
    def _display_loop(self):
        """Main display loop running in separate thread."""
        while self.running and not self.stopped:
            if self.integration_started:
                self._check_for_new_warnings()
                self._draw_progress_bar()
            time.sleep(self.update_interval)
    
    def _check_for_new_warnings(self):
        """Check for and display new warnings."""
        new_warning = None
        
        # Check for physics warnings
        for warning in self.physics_warnings:
            if warning not in self.displayed_warnings:
                new_warning = f"⚠  {warning}"
                self.displayed_warnings.add(warning)
                break
        
        # Display new warning
        if new_warning:
            self._clear_progress_line()
            print(new_warning)
            sys.stdout.flush()
    
    def _draw_progress_bar(self, force_draw: bool = False):
        """Draw enhanced progress bar with physics diagnostics."""
        if self.stopped and not force_draw:
            return
        
        # Calculate progress
        if self.t_duration > 0:
            progress = min(1.0, max(0.0, (self.current_time - self.t_start) / self.t_duration))
        else:
            progress = 0.0
        
        # Create progress bar
        filled_width = int(self.bar_width * progress)
        bar = '█' * filled_width + '░' * (self.bar_width - filled_width)
        
        # Format time information
        elapsed_real_time = time.time() - self.start_real_time
        if progress > 0.01:
            estimated_total_time = elapsed_real_time / progress
            remaining_time = estimated_total_time - elapsed_real_time
        else:
            remaining_time = 0.0
        
        # Format physics diagnostics
        diagnostics = self._format_physics_diagnostics()
        
        # Create progress line
        progress_line = (
            f"\r{bar} {progress*100:5.1f}% | "
            f"t={self.current_time:.4f}s | "
            f"Steps: {self.step_count:,} | "
            f"Rate: {self.current_integration_rate:.0f}/s | "
            f"ETA: {remaining_time:.1f}s | "
            f"{diagnostics}"
        )
        
        # Display progress
        self.last_progress_length = len(progress_line)
        sys.stdout.write(progress_line)
        sys.stdout.flush()
        self.progress_active = True
    
    def _format_physics_diagnostics(self) -> str:
        """Format physics diagnostics for display."""
        if self.current_state is None or len(self.current_state) < 4:
            return "No data"
        
        Q, I, x, v = self.current_state[:4]
        
        diagnostics_parts = [
            f"I={I:.1f}A",
            f"x={x*1000:.3f}mm",  # Higher precision for position
            f"v={v:.6f}m/s"       # Higher precision for velocity
        ]
        
        return " | ".join(diagnostics_parts)
    
    def _clear_progress_line(self):
        """Clear the current progress line from terminal."""
        if self.progress_active and self.last_progress_length > 0:
            sys.stdout.write('\r' + ' ' * self.last_progress_length + '\r')
            sys.stdout.flush()
        self.progress_active = False
        self.last_progress_length = 0
    
    def stop(self):
        """Stop the progress tracker."""
        self.stopped = True
        self.running = False
        
        # Clear progress line
        self._clear_progress_line()
        
        # Wait for display thread to finish
        if self.display_thread.is_alive():
            self.display_thread.join(timeout=0.5)
    
    def show_final_progress(self):
        """Show final progress state."""
        if not self.integration_started:
            return
        
        # Clear current progress
        self._clear_progress_line()
        
        # Show final progress bar
        self._draw_progress_bar(force_draw=True)
        print()  # New line after progress bar
        
        # Show final statistics
        elapsed_time = time.time() - self.start_real_time
        print(f"Integration completed in {elapsed_time:.2f}s with {self.step_count:,} steps")
        
        if self.current_state is not None and len(self.current_state) >= 4:
            Q, I, x, v = self.current_state[:4]
            print(f"Final state: I={I:.3f}A, x={x*1000:.2f}mm, v={v:.3f}m/s")
        
        print(f"Max values: I={self.max_current:.1f}A, F={self.max_force:.1f}N, v={self.max_velocity:.3f}m/s")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get current diagnostics data."""
        elapsed_time = time.time() - self.start_real_time
        
        diagnostics = {
            'elapsed_time': elapsed_time,
            'step_count': self.step_count,
            'current_time': self.current_time,
            'progress': (self.current_time - self.t_start) / self.t_duration if self.t_duration > 0 else 0,
            'integration_rate': self.current_integration_rate,
            'average_rate': self.average_integration_rate,
            'max_current': self.max_current,
            'max_force': self.max_force, 
            'max_velocity': self.max_velocity,
            'current_position': self.current_position,
            'warning_count': len(self.physics_warnings)
        }
        
        if self.current_state is not None:
            diagnostics['current_state'] = self.current_state.copy()
        
        return diagnostics


class SimpleProgressTracker:
    """
    Simplified progress tracker for basic monitoring without threading.
    """
    
    def __init__(self, t_span: tuple, update_frequency: int = 100):
        """
        Initialize simple progress tracker.
        
        Args:
            t_span: Time span tuple (t_start, t_end)
            update_frequency: Update every N steps
        """
        self.t_start, self.t_end = t_span
        self.t_duration = self.t_end - self.t_start
        self.update_frequency = update_frequency
        
        self.step_count = 0
        self.start_time = time.time()
        self.last_update_step = 0
    
    def update(self, t: float, y: np.ndarray, dydt: Optional[np.ndarray] = None):
        """Update progress."""
        self.step_count += 1
        
        # Update display periodically
        if self.step_count % self.update_frequency == 0:
            progress = (t - self.t_start) / self.t_duration if self.t_duration > 0 else 0
            elapsed = time.time() - self.start_time
            rate = self.step_count / elapsed if elapsed > 0 else 0
            
            print(f"\rProgress: {progress*100:.1f}% | t={t:.4f}s | Steps: {self.step_count:,} | Rate: {rate:.0f}/s", 
                  end='', flush=True)
    
    def stop(self):
        """Stop progress tracking."""
        print()  # New line
    
    def show_final_progress(self):
        """Show final progress."""
        elapsed = time.time() - self.start_time
        print(f"\nCompleted in {elapsed:.2f}s with {self.step_count:,} steps")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get current diagnostics data (simplified for SimpleProgressTracker)."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'elapsed_time': elapsed_time,
            'step_count': self.step_count,
            'current_time': self.t_end,  # Simple tracker doesn't track current time
            'progress': 1.0,  # Assume complete
            'integration_rate': self.step_count / elapsed_time if elapsed_time > 0 else 0,
            'average_rate': self.step_count / elapsed_time if elapsed_time > 0 else 0,
            'max_current': 0,  # Not tracked by simple tracker
            'max_force': 0,
            'max_velocity': 0, 
            'current_position': 0,
            'warning_count': 0
        } 