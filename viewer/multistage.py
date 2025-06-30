"""
Multi-stage coilgun visualization functions.

This module provides specialized visualization for multi-stage coilgun systems,
including stage comparisons, performance progression, and timing analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

from .core import BaseVisualizer
from .plots import ProfilePlotter


class MultistageVisualizer(BaseVisualizer):
    """Class for multi-stage coilgun visualization."""
    
    def create_multistage_visualizations(self, config_file, time_series_data, 
                                       summary_data, output_dir):
        """
        Create comprehensive multi-stage visualization suite.
        
        Args:
            config_file: Configuration file path
            time_series_data: Time series data dictionary
            summary_data: Summary data from multi-stage simulation
            output_dir: Output directory for saving plots
        """
        output_path = str(output_dir)
        
        print("Creating multi-stage visualization suite...")
        
        # 1. Time series plots for each stage
        if time_series_data:
            self.create_multistage_time_series_plots(
                time_series_data, summary_data, output_path
            )
        
        # 2. Velocity progression analysis
        if summary_data:
            profile_plotter = ProfilePlotter(self.physics)
            profile_plotter.plot_velocity_progression(
                summary_data, f"{output_path}/velocity_progression.png"
            )
        
        # 3. Efficiency comparison
        if summary_data:
            profile_plotter = ProfilePlotter(self.physics)
            profile_plotter.plot_efficiency_comparison(
                summary_data, f"{output_path}/efficiency_comparison.png"
            )
        
        # 4. Stage performance summary
        if summary_data:
            self.create_stage_performance_summary(
                summary_data, f"{output_path}/stage_performance_summary.png"
            )
        
        # 5. Field comparison across stages
        self.create_stage_comparison_field_plot(
            config_file, output_path
        )
        
        print(f"Multi-stage visualizations saved to: {output_dir}")
    
    def create_multistage_time_series_plots(self, time_series_data, summary_data, output_path):
        """
        Create time series plots for multi-stage analysis.
        
        Args:
            time_series_data: Time series data dictionary
            summary_data: Summary data dictionary
            output_path: Output path for saving
        """
        # Determine stage transitions
        stage_transitions = []
        if summary_data:
            cumulative_time = 0
            for stage, data in summary_data.items():
                stage_time = data.get('simulation_time', 0.001)
                stage_transitions.append(cumulative_time + stage_time)
                cumulative_time += stage_time
        
        # Create comprehensive time series plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Multi-Stage Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Extract time data
        if 'time' in time_series_data:
            t_ms = np.array(time_series_data['time']) * 1000
        else:
            print("No time data available for multi-stage analysis")
            return
        
        # Plot 1: Current vs Time
        ax1 = axes[0, 0]
        if 'current' in time_series_data:
            ax1.plot(t_ms, time_series_data['current'], 'b-', linewidth=2, alpha=0.8)
            self._add_stage_boundaries(ax1, stage_transitions)
            self.apply_common_styling(ax1, 'Current vs Time', 'Time (ms)', 'Current (A)')
        
        # Plot 2: Voltage vs Time
        ax2 = axes[0, 1]
        if 'voltage' in time_series_data:
            ax2.plot(t_ms, time_series_data['voltage'], 'r-', linewidth=2, alpha=0.8)
            self._add_stage_boundaries(ax2, stage_transitions)
            self.apply_common_styling(ax2, 'Voltage vs Time', 'Time (ms)', 'Voltage (V)')
        
        # Plot 3: Position vs Time
        ax3 = axes[1, 0]
        if 'position' in time_series_data:
            position_mm = np.array(time_series_data['position']) * 1000
            ax3.plot(t_ms, position_mm, 'g-', linewidth=2, alpha=0.8)
            self._add_stage_boundaries(ax3, stage_transitions)
            self.apply_common_styling(ax3, 'Position vs Time', 'Time (ms)', 'Position (mm)')
        
        # Plot 4: Velocity vs Time
        ax4 = axes[1, 1]
        if 'velocity' in time_series_data:
            ax4.plot(t_ms, time_series_data['velocity'], 'm-', linewidth=2, alpha=0.8)
            self._add_stage_boundaries(ax4, stage_transitions)
            self.apply_common_styling(ax4, 'Velocity vs Time', 'Time (ms)', 'Velocity (m/s)')
        
        # Plot 5: Force vs Time
        ax5 = axes[2, 0]
        if 'force' in time_series_data:
            ax5.plot(t_ms, time_series_data['force'], 'orange', linewidth=2, alpha=0.8)
            self._add_stage_boundaries(ax5, stage_transitions)
            self.apply_common_styling(ax5, 'Force vs Time', 'Time (ms)', 'Force (N)')
        
        # Plot 6: Energy vs Time
        ax6 = axes[2, 1]
        if 'energy_kinetic' in time_series_data and 'energy_capacitor' in time_series_data:
            ax6.plot(t_ms, time_series_data['energy_kinetic'], 'b-', 
                    linewidth=2, label='Kinetic', alpha=0.8)
            ax6.plot(t_ms, time_series_data['energy_capacitor'], 'r-', 
                    linewidth=2, label='Capacitor', alpha=0.8)
            self._add_stage_boundaries(ax6, stage_transitions)
            ax6.legend()
            self.apply_common_styling(ax6, 'Energy vs Time', 'Time (ms)', 'Energy (J)')
        
        plt.tight_layout()
        save_path = f"{output_path}/multistage_timeseries.png"
        self.save_figure(fig, save_path)
        plt.show()
    
    def create_stage_performance_summary(self, summary_data, save_path):
        """
        Create stage performance summary visualization.
        
        Args:
            summary_data: Summary data dictionary
            save_path: Path to save the plot
        """
        if not summary_data:
            print("No summary data available for stage performance")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Stage Performance Summary', fontsize=16, fontweight='bold')
        
        stages = list(summary_data.keys())
        
        # Extract performance metrics
        initial_velocities = [summary_data[stage].get('initial_velocity', 0) for stage in stages]
        final_velocities = [summary_data[stage].get('final_velocity', 0) for stage in stages]
        efficiencies = [summary_data[stage].get('efficiency', 0) * 100 for stage in stages]
        energy_consumed = [summary_data[stage].get('energy_consumed', 0) for stage in stages]
        peak_currents = [summary_data[stage].get('peak_current', 0) for stage in stages]
        max_forces = [summary_data[stage].get('max_force', 0) for stage in stages]
        
        # Plot 1: Velocity progression
        x_pos = np.arange(len(stages))
        width = 0.35
        
        ax1.bar(x_pos - width/2, initial_velocities, width, 
               label='Initial Velocity', alpha=0.7, color='blue')
        ax1.bar(x_pos + width/2, final_velocities, width, 
               label='Final Velocity', alpha=0.7, color='red')
        
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Velocity by Stage')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stages)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency by stage
        ax2.bar(stages, efficiencies, alpha=0.7, color='green')
        self.apply_common_styling(ax2, 'Efficiency by Stage', 'Stage', 'Efficiency (%)')
        
        # Plot 3: Energy consumption
        ax3.bar(stages, energy_consumed, alpha=0.7, color='orange')
        self.apply_common_styling(ax3, 'Energy Consumption', 'Stage', 'Energy (J)')
        
        # Plot 4: Peak performance metrics
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(stages, peak_currents, 'bo-', linewidth=2, 
                        label='Peak Current (A)', alpha=0.8)
        line2 = ax4_twin.plot(stages, max_forces, 'ro-', linewidth=2, 
                             label='Max Force (N)', alpha=0.8)
        
        ax4.set_xlabel('Stage')
        ax4.set_ylabel('Peak Current (A)', color='b')
        ax4_twin.set_ylabel('Max Force (N)', color='r')
        ax4.set_title('Peak Performance Metrics')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        self.save_figure(fig, save_path)
        plt.show()
    
    def create_stage_comparison_field_plot(self, config_file, output_path):
        """
        Create magnetic field comparison across stages.
        
        Args:
            config_file: Configuration file path
            output_path: Output directory path
        """
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Could not load config file: {e}")
            return
        
        if 'stages' not in config:
            print("No stage configuration found")
            return
        
        from .fields import MagneticFieldCalculator
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Magnetic Field Comparison Across Stages', fontsize=16, fontweight='bold')
        
        stage_currents = []
        stage_labels = []
        
        # Extract stage information
        for stage_name, stage_config in config['stages'].items():
            if 'peak_current' in stage_config:
                stage_currents.append(stage_config['peak_current'])
                stage_labels.append(stage_name)
        
        if not stage_currents:
            print("No current data found in stage configuration")
            return
        
        # Calculate field profiles for each stage
        calculator = MagneticFieldCalculator(self.physics)
        
        # On-axis field profiles
        ax1 = axes[0, 0]
        z_range = (-0.05, 0.15)
        z_vals = np.linspace(z_range[0], z_range[1], 200)
        
        for i, (current, label) in enumerate(zip(stage_currents, stage_labels)):
            bz_profile = []
            for z in z_vals:
                bz, _ = calculator._biot_savart_total_field(z, 0, current)
                bz_profile.append(bz)
            
            ax1.plot(z_vals * 1000, bz_profile, linewidth=2, 
                    label=f'{label} ({current}A)', alpha=0.8)
        
        self.apply_common_styling(ax1, 'On-Axis Field Profiles', 
                                'Z position (mm)', 'Bz (T)')
        ax1.legend()
        
        # Field gradient comparison
        ax2 = axes[0, 1]
        for i, (current, label) in enumerate(zip(stage_currents, stage_labels)):
            bz_profile = []
            for z in z_vals:
                bz, _ = calculator._biot_savart_total_field(z, 0, current)
                bz_profile.append(bz)
            
            gradient = np.gradient(bz_profile, z_vals)
            ax2.plot(z_vals[1:-1] * 1000, gradient[1:-1], linewidth=2, 
                    label=f'{label}', alpha=0.8)
        
        self.apply_common_styling(ax2, 'Field Gradient Comparison', 
                                'Z position (mm)', 'dBz/dz (T/m)')
        ax2.legend()
        
        # Peak field strength comparison
        ax3 = axes[1, 0]
        peak_fields = []
        for current in stage_currents:
            bz, _ = calculator._biot_savart_total_field(0.01, 0, current)  # Near coil center
            peak_fields.append(abs(bz))
        
        ax3.bar(stage_labels, peak_fields, alpha=0.7, color='purple')
        self.apply_common_styling(ax3, 'Peak Field Strength', 'Stage', 'Peak |B| (T)')
        
        # Force comparison (approximate)
        ax4 = axes[1, 1]
        forces = []
        for current in stage_currents:
            # Approximate force calculation
            bz1, _ = calculator._biot_savart_total_field(0.01, 0, current)
            bz2, _ = calculator._biot_savart_total_field(0.015, 0, current)
            gradient = (bz2 - bz1) / 0.005
            # F ≈ μ * ∇B for magnetic dipole
            force = abs(gradient) * 1e-6  # Approximate force in N
            forces.append(force)
        
        ax4.bar(stage_labels, forces, alpha=0.7, color='red')
        self.apply_common_styling(ax4, 'Estimated Force', 'Stage', 'Force (N)')
        
        plt.tight_layout()
        save_path = f"{output_path}/stage_field_comparison.png"
        self.save_figure(fig, save_path)
        plt.show()
    
    def _add_stage_boundaries(self, ax, stage_transitions):
        """
        Add vertical lines to indicate stage boundaries.
        
        Args:
            ax: Matplotlib axes object
            stage_transitions: List of stage transition times
        """
        if not stage_transitions:
            return
        
        for i, transition_time in enumerate(stage_transitions):
            transition_ms = transition_time * 1000
            ax.axvline(transition_ms, color='gray', linestyle='--', 
                      alpha=0.7, linewidth=1)
            
            # Add stage labels
            if i == 0:
                ax.text(transition_ms/2, ax.get_ylim()[1]*0.9, f'Stage {i+1}', 
                       ha='center', va='top', fontsize=10, alpha=0.7)
            else:
                prev_transition = stage_transitions[i-1] * 1000
                mid_point = (prev_transition + transition_ms) / 2
                ax.text(mid_point, ax.get_ylim()[1]*0.9, f'Stage {i+1}', 
                       ha='center', va='top', fontsize=10, alpha=0.7)


class StageComparisonAnalyzer:
    """Class for detailed stage comparison analysis."""
    
    @staticmethod
    def analyze_stage_transitions(time_series_data, stage_config):
        """
        Analyze the transitions between stages.
        
        Args:
            time_series_data: Time series data dictionary
            stage_config: Stage configuration dictionary
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            'transition_times': [],
            'velocity_jumps': [],
            'current_overlaps': [],
            'energy_transfers': []
        }
        
        if 'time' not in time_series_data:
            return analysis
        
        # Implementation would analyze stage transitions
        # This is a placeholder for the actual analysis logic
        
        return analysis
    
    @staticmethod
    def calculate_stage_efficiency(stage_data):
        """
        Calculate efficiency metrics for each stage.
        
        Args:
            stage_data: Individual stage data dictionary
            
        Returns:
            Efficiency metrics dictionary
        """
        efficiency_metrics = {
            'energy_efficiency': 0,
            'force_efficiency': 0,
            'time_efficiency': 0
        }
        
        # Implementation would calculate various efficiency metrics
        # This is a placeholder for the actual calculation logic
        
        return efficiency_metrics 