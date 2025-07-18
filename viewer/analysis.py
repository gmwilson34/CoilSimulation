"""
Advanced physics analysis and visualization for coilgun systems.

This module provides comprehensive analysis visualization including force decomposition,
energy conservation tracking, electromagnetic field analysis, and physics diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List

from .core import BaseVisualizer


class PhysicsAnalyzer(BaseVisualizer):
    """Class for advanced physics analysis and visualization."""
    
    def plot_enhanced_physics_analysis(self, results, save_path=None):
        """
        Create comprehensive visualization of enhanced physics analysis results.
        
        Args:
            results: Results dictionary from enhanced simulation
            save_path: Path to save the plot
        """
        # Create figure with multiple subplots for comprehensive analysis
        fig, gs = self.setup_subplot_grid(4, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Enhanced Electromagnetic Physics Analysis', fontsize=16, fontweight='bold')
        
        # Check if we have time data
        if 'time' not in results or len(results['time']) == 0:
            print("No time data available for enhanced physics analysis")
            plt.close()
            return
        
        t_ms = np.array(results['time']) * 1000  # Convert to ms
        
        # 1. Force decomposition plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_force_components(ax1, t_ms, results)
        
        # 2. Eddy current analysis
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_eddy_current_analysis(ax2, t_ms, results)
        
        # 3. Power loss analysis
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_power_analysis(ax3, t_ms, results)
        
        # 4. Energy conservation tracking
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_energy_conservation(ax4, t_ms, results)
        
        # 5. Magnetic field and saturation analysis
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_magnetic_field_analysis(ax5, t_ms, results)
        
        # 6. Frequency analysis
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_frequency_analysis(ax6, t_ms, results)
        
        # 7. Inductance variation
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_inductance_analysis(ax7, t_ms, results)
        
        # 8. Efficiency metrics
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_efficiency_metrics(ax8, t_ms, results)
        
        # 9. Temperature effects
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_temperature_effects(ax9, t_ms, results)
        
        # 10. Field gradient analysis
        ax10 = fig.add_subplot(gs[3, 0])
        self._plot_field_gradient_analysis(ax10, t_ms, results)
        
        # 11. Circuit parameter evolution
        ax11 = fig.add_subplot(gs[3, 1])
        self._plot_circuit_parameters(ax11, t_ms, results)
        
        # 12. Performance metrics summary
        ax12 = fig.add_subplot(gs[3, 2])
        self._plot_performance_summary(ax12, results)
        
        # Removed tight_layout to avoid warning with twinx axes; spacing handled by GridSpec
        self.save_figure(fig, save_path)
        plt.show()
    
    def _plot_force_components(self, ax, t_ms, results):
        """Plot force component decomposition."""
        force_components = {
            'force_gradient': ('Gradient Force', 'b-'),
            'force_reluctance': ('Reluctance Force', 'g--'),
            'force_eddy': ('Eddy Current Force', 'r:'),
            'force_lorentz': ('Lorentz Force', 'm-.'),
            'force_maxwell': ('Maxwell Stress', 'c--'),
            'force_total': ('Total Force', 'k-')
        }
        
        for key, (label, style) in force_components.items():
            if key in results:
                linewidth = 3 if 'total' in key else 2
                alpha = 0.9 if 'total' in key else 0.8
                ax.plot(t_ms, results[key], style, linewidth=linewidth, 
                       label=label, alpha=alpha)
        
        self.apply_common_styling(ax, 'Force Component Decomposition', 
                                'Time (ms)', 'Force (N)')
        if ax.lines:
            ax.legend(fontsize=8)
    
    def _plot_eddy_current_analysis(self, ax, t_ms, results):
        """Plot eddy current effects and skin depth."""
        if 'eddy_current_magnitude' in results and np.any(results['eddy_current_magnitude'] > 0):
            ax.plot(t_ms, results['eddy_current_magnitude'], 'r-', 
                   linewidth=2, label='Eddy Current')
            
            if 'skin_depth' in results:
                ax_twin = ax.twinx()
                finite_skin = np.where(results['skin_depth'] < 1e6, 
                                     results['skin_depth'] * 1000, np.nan)
                ax_twin.plot(t_ms, finite_skin, 'b--', linewidth=1.5, 
                           label='Skin Depth', alpha=0.7)
                ax_twin.set_ylabel('Skin Depth (mm)', color='b')
        
        self.apply_common_styling(ax, 'Eddy Current Effects', 
                                'Time (ms)', 'Eddy Current (A)')
    
    def _plot_power_analysis(self, ax, t_ms, results):
        """Plot power loss analysis."""
        power_components = {
            'power_loss_resistive': ('Resistive Loss', 'g-'),
            'power_loss_eddy': ('Eddy Loss', 'r-'),
            'power_mechanical': ('Mechanical Power', 'b-')
        }
        
        for key, (label, style) in power_components.items():
            if key in results:
                ax.plot(t_ms, results[key], style, linewidth=2, 
                       label=label, alpha=0.8)
        
        self.apply_common_styling(ax, 'Power Analysis', 'Time (ms)', 'Power (W)')
        if ax.lines:
            ax.legend(fontsize=9)
    
    def _plot_energy_conservation(self, ax, t_ms, results):
        """Plot energy conservation tracking."""
        if 'energy_capacitor' in results and 'energy_kinetic' in results:
            E_cap = results['energy_capacitor']
            E_kin = results['energy_kinetic']
            E_total = E_cap + E_kin
            
            ax.plot(t_ms, E_cap, 'b-', linewidth=2, label='Capacitor Energy', alpha=0.8)
            ax.plot(t_ms, E_kin, 'r-', linewidth=2, label='Kinetic Energy', alpha=0.8)
            ax.plot(t_ms, E_total, 'k--', linewidth=2, label='Total Energy', alpha=0.8)
            
            # Add energy conservation error if available
            if 'energy_conservation' in results:
                ax_twin = ax.twinx()
                ax_twin.plot(t_ms, results['energy_conservation'] * 100, 
                           'orange', linewidth=1, alpha=0.6)
                ax_twin.set_ylabel('Energy Error (%)', color='orange')
        
        self.apply_common_styling(ax, 'Energy Conservation', 'Time (ms)', 'Energy (J)')
        if ax.lines:
            ax.legend(fontsize=9)
    
    def _plot_magnetic_field_analysis(self, ax, t_ms, results):
        """Plot magnetic field and saturation analysis."""
        if 'magnetic_field' in results:
            ax.plot(t_ms, results['magnetic_field'], 'purple', 
                   linewidth=2, label='B-field', alpha=0.8)
            
            # Add saturation analysis if available
            if 'saturation_factor' in results:
                ax_twin = ax.twinx()
                ax_twin.plot(t_ms, results['saturation_factor'], 'orange', 
                           linewidth=1.5, label='Saturation Factor', alpha=0.7)
                ax_twin.set_ylabel('Saturation Factor', color='orange')
                ax_twin.set_ylim(0, 1.1)
        
        self.apply_common_styling(ax, 'Magnetic Field & Saturation', 
                                'Time (ms)', 'Magnetic Field (T)')
    
    def _plot_frequency_analysis(self, ax, t_ms, results):
        """Plot frequency content analysis."""
        if 'frequency_content' in results and np.any(results['frequency_content'] > 0):
            ax.plot(t_ms, results['frequency_content'], 'cyan', 
                   linewidth=2, label='Peak Frequency', alpha=0.8)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Frequency Content Analysis')
        else:
            # Current frequency spectrum
            if 'current' in results and len(results['current']) > 100:
                current_data = results['current']
                dt = results['time'][1] - results['time'][0]
                freqs = np.fft.fftfreq(len(current_data), dt)
                fft_current = np.abs(np.fft.fft(current_data))
                
                # Plot positive frequencies only
                pos_freqs = freqs[freqs > 0][:len(freqs)//4]  # Show first quarter
                pos_fft = fft_current[freqs > 0][:len(freqs)//4]
                
                ax.semilogy(pos_freqs, pos_fft, 'cyan', linewidth=1.5, alpha=0.8)
                ax.set_ylabel('Current FFT Magnitude')
                ax.set_title('Current Frequency Spectrum')
        
        ax.set_xlabel('Time (ms)' if 'frequency_content' in results else 'Frequency (Hz)')
        ax.grid(True, alpha=self.config.GRID_ALPHA)
    
    def _plot_inductance_analysis(self, ax, t_ms, results):
        """Plot inductance variation analysis."""
        if 'inductance' in results:
            ax.plot(t_ms, results['inductance'] * 1000, 'g-', 
                   linewidth=2, alpha=0.8)  # Convert to mH
            self.apply_common_styling(ax, 'Inductance Variation', 
                                    'Time (ms)', 'Inductance (mH)')
        elif 'position' in results and self.physics:
            # Calculate approximate inductance variation
            positions = np.array(results['position'])
            L_base = getattr(self.physics, 'inductance', 1e-3)
            # Simple model: inductance varies with projectile position
            inductance_variation = L_base * (1 + 0.1 * positions / 0.05)
            ax.plot(t_ms, inductance_variation * 1000, 'g-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'Est. Inductance Variation', 
                                    'Time (ms)', 'Inductance (mH)')
    
    def _plot_efficiency_metrics(self, ax, t_ms, results):
        """Plot efficiency metrics over time."""
        if 'efficiency' in results:
            ax.plot(t_ms, np.array(results['efficiency']) * 100, 'b-', 
                   linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'System Efficiency', 
                                    'Time (ms)', 'Efficiency (%)')
        elif 'energy_kinetic' in results and 'energy_capacitor' in results:
            # Calculate instantaneous efficiency
            E_kin = np.array(results['energy_kinetic'])
            E_cap_initial = results['energy_capacitor'][0]
            efficiency = (E_kin / E_cap_initial) * 100
            ax.plot(t_ms, efficiency, 'b-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'Energy Transfer Efficiency', 
                                    'Time (ms)', 'Efficiency (%)')
    
    def _plot_temperature_effects(self, ax, t_ms, results):
        """Plot temperature effects if available."""
        if 'temperature' in results:
            ax.plot(t_ms, results['temperature'], 'r-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'Temperature Rise', 
                                    'Time (ms)', 'Temperature (°C)')
        elif 'current' in results:
            # Estimate temperature rise from I²R losses
            current = np.array(results['current'])
            resistance = getattr(self.physics, 'resistance', 0.1) if self.physics else 0.1
            power_loss = current**2 * resistance
            # Simple thermal model
            temp_rise = np.cumsum(power_loss) * 0.001  # Simplified
            ax.plot(t_ms, temp_rise, 'r-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'Est. Temperature Rise', 
                                    'Time (ms)', 'ΔT (°C)')
    
    def _plot_field_gradient_analysis(self, ax, t_ms, results):
        """Plot magnetic field gradient analysis."""
        if 'field_gradient' in results:
            ax.plot(t_ms, results['field_gradient'], 'purple', 
                   linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'Magnetic Field Gradient', 
                                    'Time (ms)', 'dB/dz (T/m)')
        elif 'magnetic_field' in results:
            # Calculate approximate gradient with safety checks for numpy warnings
            B_field = np.array(results['magnetic_field'])
            
            # Check for sufficient data points and avoid duplicate values
            if len(B_field) > 1:
                # Remove duplicate adjacent values to prevent divide by zero
                unique_indices = np.concatenate([[0], np.where(np.diff(B_field) != 0)[0] + 1])
                if len(unique_indices) > 1:
                    B_field_unique = B_field[unique_indices]
                    t_ms_unique = t_ms[unique_indices]
                    
                    # Use numerical gradient with spacing information
                    if len(B_field_unique) > 1:
                        dt_unique = np.diff(t_ms_unique)
                        if np.all(dt_unique > 1e-10):  # Ensure no zero time differences
                            gradient = np.gradient(B_field_unique, t_ms_unique)
                            ax.plot(t_ms_unique, gradient, 'purple', linewidth=2, alpha=0.8)
                        else:
                            # Fallback: simple difference calculation
                            gradient = np.diff(B_field_unique) / np.maximum(dt_unique, 1e-10)
                            ax.plot(t_ms_unique[1:], gradient, 'purple', linewidth=2, alpha=0.8)
                    else:
                        # Not enough unique data points
                        ax.text(0.5, 0.5, 'Insufficient unique data\nfor gradient calculation', 
                               transform=ax.transAxes, ha='center', va='center')
                else:
                    # All values are the same
                    ax.axhline(y=0, color='purple', linewidth=2, alpha=0.8)
            else:
                # Not enough data points
                ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                       transform=ax.transAxes, ha='center', va='center')
            
            self.apply_common_styling(ax, 'Est. Field Gradient', 
                                    'Time (ms)', 'dB/dt (T/s)')
    
    def _plot_circuit_parameters(self, ax, t_ms, results):
        """Plot circuit parameter evolution."""
        if 'resistance' in results and 'inductance' in results:
            ax_R = ax
            ax_L = ax.twinx()
            
            ax_R.plot(t_ms, results['resistance'], 'r-', linewidth=2, alpha=0.8)
            ax_L.plot(t_ms, np.array(results['inductance']) * 1000, 'b-', 
                     linewidth=2, alpha=0.8)
            
            ax_R.set_xlabel('Time (ms)')
            ax_R.set_ylabel('Resistance (Ω)', color='r')
            ax_L.set_ylabel('Inductance (mH)', color='b')
            ax_R.set_title('Circuit Parameters')
            ax_R.grid(True, alpha=0.3)
        elif 'current' in results and 'voltage' in results:
            # Calculate apparent resistance
            current = np.array(results['current'])
            voltage = np.array(results['voltage'])
            resistance = np.where(current > 1, voltage / current, 0)
            ax.plot(t_ms, resistance, 'r-', linewidth=2, alpha=0.8)
            self.apply_common_styling(ax, 'Apparent Resistance', 
                                    'Time (ms)', 'Resistance (Ω)')
    
    def _plot_performance_summary(self, ax, results):
        """Plot performance metrics summary."""
        # Calculate key performance metrics
        metrics = {}
        
        if 'velocity' in results and len(results['velocity']) > 0:
            metrics['Max Velocity (m/s)'] = max(results['velocity'])
        
        if 'force' in results and len(results['force']) > 0:
            metrics['Max Force (N)'] = max(results['force'])
        
        if 'energy_kinetic' in results and 'energy_capacitor' in results:
            E_kin_final = results['energy_kinetic'][-1]
            E_cap_initial = results['energy_capacitor'][0]
            metrics['Efficiency (%)'] = (E_kin_final / E_cap_initial) * 100
        
        if 'current' in results and len(results['current']) > 0:
            metrics['Peak Current (A)'] = max(results['current'])
        
        if metrics:
            names = list(metrics.keys())
            values = list(metrics.values())
            
            bars = ax.bar(range(len(names)), values, alpha=0.7, 
                         color=['blue', 'red', 'green', 'orange'][:len(names)])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_title('Performance Summary')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                          xytext=(0, 3), textcoords="offset points", 
                          ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No performance\nmetrics available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Performance Summary')


class ElectromagneticAnalyzer(BaseVisualizer):
    """Class for electromagnetic field analysis."""
    
    def plot_electromagnetic_field_analysis(self, results, save_path=None):
        """
        Create detailed electromagnetic field analysis visualization.
        
        Args:
            results: Results dictionary with field data
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Electromagnetic Field Analysis', fontsize=16, fontweight='bold')
        
        if 'time' not in results:
            print("No time data available for electromagnetic analysis")
            return
        
        t_ms = np.array(results['time']) * 1000
        
        # 1. Electric and magnetic field components
        if 'electric_field' in results and 'magnetic_field' in results:
            ax1.plot(t_ms, results['electric_field'], 'b-', linewidth=2, 
                    label='Electric Field', alpha=0.8)
            ax1_twin = ax1.twinx()
            ax1_twin.plot(t_ms, results['magnetic_field'], 'r-', linewidth=2, 
                         label='Magnetic Field', alpha=0.8)
            
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Electric Field (V/m)', color='b')
            ax1_twin.set_ylabel('Magnetic Field (T)', color='r')
            ax1.set_title('E&M Field Components')
            ax1.grid(True, alpha=0.3)
        
        # 2. Field energy density
        if 'field_energy_density' in results:
            ax2.plot(t_ms, results['field_energy_density'], 'purple', 
                    linewidth=2, alpha=0.8)
            self.apply_common_styling(ax2, 'Field Energy Density', 
                                    'Time (ms)', 'Energy Density (J/m³)')
        
        # 3. Poynting vector (energy flux)
        if 'poynting_vector' in results:
            ax3.plot(t_ms, results['poynting_vector'], 'green', 
                    linewidth=2, alpha=0.8)
            self.apply_common_styling(ax3, 'Poynting Vector (Energy Flux)', 
                                    'Time (ms)', 'Power Flux (W/m²)')
        
        # 4. Field coupling effects
        if 'coupling_coefficient' in results:
            ax4.plot(t_ms, results['coupling_coefficient'], 'orange', 
                    linewidth=2, alpha=0.8)
            self.apply_common_styling(ax4, 'Field Coupling Coefficient', 
                                    'Time (ms)', 'Coupling Factor')
        
        plt.tight_layout()
        self.save_figure(fig, save_path)
        plt.show() 