#!/usr/bin/env python3
"""
Advanced Coilgun Configuration Setup

This refactored setup system leverages the new physics engine and creates
configurations that fully utilize the advanced simulation capabilities.

Features:
- Integrated material system using AdvancedMaterialProperties
- Physics-engine aware configuration generation
- Solver-optimized parameter setup
- Intelligent defaults based on physics models
- Comprehensive validation
"""

import os
import sys
import signal
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

# Import physics engine for advanced material properties and validation
try:
    from physics.materials import AdvancedMaterialProperties
    from physics.utils import validate_coilgun_config, calculate_coil_metrics
    from physics.core import PhysicsConstants, SafetyLimits
    from solver.core import SolverConfig, SolverConstants
    from solver.utils import load_and_validate_config
    PHYSICS_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Physics engine not available: {e}")
    PHYSICS_ENGINE_AVAILABLE = False


class ConfigurationBuilder:
    """Advanced configuration builder using physics engine integration."""
    
    def __init__(self):
        """Initialize the configuration builder."""
        self.config = {}
        self.materials = None
        self._initialize_materials()
    
    def _initialize_materials(self):
        """Initialize the advanced materials system."""
        if PHYSICS_ENGINE_AVAILABLE:
            try:
                # Create a minimal config for materials initialization
                temp_config = {
                    'environment': {'temperature': PhysicsConstants.ROOM_TEMPERATURE},
                    'advanced_physics': {'include_temperature': True},
                    'magnetic_model': {'frequency': 1000}
                }
                self.materials = AdvancedMaterialProperties(temp_config)
                print("✓ Advanced materials system initialized")
            except Exception as e:
                print(f"Warning: Could not initialize advanced materials: {e}")
                self.materials = None
        else:
            self.materials = None
    
    def get_available_materials(self) -> List[str]:
        """Get list of available materials."""
        if self.materials:
            return list(self.materials.materials_data.get('materials', {}).keys())
        else:
            # Fallback to basic materials
            return ['Copper', 'Pure_Iron', 'Low_Carbon_Steel', 'Silicon_Steel', 
                   'Aluminum', 'Brass', 'Neodymium_Magnet']
    
    def get_material_properties(self, material_name: str) -> Dict[str, Any]:
        """Get material properties with temperature/frequency effects."""
        if self.materials:
            try:
                # Get base properties
                base_props = self.materials.materials_data['materials'].get(material_name, {})
                
                # Add temperature-dependent properties if available
                enhanced_props = base_props.copy()
                
                # Add calculated properties
                if 'resistivity_20C' in base_props:
                    enhanced_props['conductivity'] = 1.0 / base_props['resistivity_20C']
                
                return enhanced_props
            except Exception as e:
                print(f"Warning: Could not get enhanced properties for {material_name}: {e}")
                return {}
        else:
            # Fallback to basic properties
            basic_materials = {
                'Copper': {'resistivity': 1.68e-8, 'mu_r': 0.999991, 'density': 8960},
                'Pure_Iron': {'resistivity': 9.71e-8, 'mu_r': 5000, 'density': 7874},
                'Low_Carbon_Steel': {'resistivity': 1.43e-7, 'mu_r': 1000, 'density': 7850}
            }
            return basic_materials.get(material_name, {})
    
    def get_wire_specifications(self) -> Dict[str, Dict[str, float]]:
        """Get wire specifications (AWG diameters and current ratings)."""
        if self.materials and hasattr(self.materials, 'materials_data'):
            return self.materials.materials_data.get('wire_specifications', {})
        else:
            # Fallback wire data
            return {
                'awg_diameter_mm': {
                    '10': 2.588, '12': 2.053, '14': 1.628, '16': 1.291,
                    '18': 1.024, '20': 0.812, '22': 0.644, '24': 0.511
                },
                'current_capacity_A': {
                    '10': 55, '12': 41, '14': 32, '16': 22,
                    '18': 16, '20': 11, '22': 7, '24': 3.5
                }
            }


class AdvancedSetupInterface:
    """Advanced setup interface with physics-aware configuration and backwards navigation."""
    
    def __init__(self):
        """Initialize the setup interface."""
        self.builder = ConfigurationBuilder()
        self.setup_modes = ['Basic', 'Advanced', 'Expert', 'Custom']
        self.current_mode = 'Advanced'
        
        # History tracking for backwards navigation
        self.input_history = []
        self.current_step = 0
        self.setup_steps = []
        
        # Setup flow state machine
        self.setup_flow = [
            ('setup_mode', 'Select setup mode'),
            ('coil_wire_material', 'Select coil wire material'),
            ('coil_inner_diameter', 'Coil inner diameter'),
            ('coil_length', 'Coil length'),
            ('coil_wire_gauge', 'Wire gauge selection'),
            ('coil_packing_factor', 'Wire packing factor'),
            ('coil_auto_calculate', 'Auto-calculate turns'),
            ('coil_num_layers', 'Number of layers'),
            ('coil_total_turns', 'Total turns'),
            ('projectile_material', 'Projectile material'),
            ('projectile_diameter', 'Projectile diameter'),
            ('projectile_length', 'Projectile length'),
            ('projectile_use_calculated_mass', 'Use calculated mass'),
            ('projectile_mass', 'Projectile mass'),
            ('projectile_initial_position', 'Initial position'),
            ('capacitor_capacitance', 'Capacitance'),
            ('capacitor_initial_voltage', 'Initial voltage'),
            ('capacitor_esr', 'Capacitor ESR'),
            ('capacitor_esl', 'Capacitor ESL'),
            ('simulation_time_span', 'Simulation time'),
            ('magnetic_calculation_method', 'Magnetic field calculation method'),
            ('magnetic_axial_discretization', 'Axial discretization points'),
            ('magnetic_radial_discretization', 'Radial discretization points'),
            ('magnetic_include_saturation', 'Include magnetic saturation'),
            ('magnetic_include_hysteresis', 'Include hysteresis effects'),
            ('magnetic_include_eddy_currents', 'Include eddy current effects'),
            ('magnetic_include_skin_effect', 'Include skin effect'),
            ('force_reluctance', 'Include reluctance force'),
            ('force_lorentz', 'Include Lorentz force'),
            ('force_maxwell_stress', 'Include Maxwell stress tensor'),
            ('force_image', 'Include magnetic image force'),
            ('force_eddy', 'Include eddy current force'),
            ('ja_enabled', 'Enable Jiles-Atherton hysteresis'),
            ('eddy_enabled', 'Enable 3D eddy current modeling'),
            ('thermal_enabled', 'Enable thermal modeling'),
            ('energy_conservation_tol', 'Energy conservation tolerance'),
            ('solver_method', 'Solver method'),
            ('solver_rtol', 'Relative tolerance'),
            ('solver_atol', 'Absolute tolerance'),
            ('solver_max_step', 'Maximum step size'),
            ('config_filename', 'Configuration filename')
        ]
        self.current_flow_index = 0
        
    def add_to_history(self, step_name: str, value: Any, prompt: str = ""):
        """Add an input to the history for backwards navigation."""
        # Remove any existing entry for this step
        self.input_history = [entry for entry in self.input_history if entry['step_name'] != step_name]
        
        self.input_history.append({
            'step_name': step_name,
            'value': value,
            'prompt': prompt,
            'timestamp': len(self.input_history)
        })
    
    def go_back(self) -> bool:
        """Go back one step in the setup process."""
        if self.current_flow_index > 0:
            self.current_flow_index -= 1
            return True
        return False
    
    def go_forward(self) -> bool:
        """Go forward one step in the setup process."""
        if self.current_flow_index < len(self.setup_flow) - 1:
            self.current_flow_index += 1
            return True
        return False
    
    def get_previous_value(self, step_name: str) -> Any:
        """Get the previous value for a specific step."""
        for entry in self.input_history:
            if entry['step_name'] == step_name:
                return entry['value']
        return None
    
    def show_navigation_help(self):
        """Show navigation help information."""
        print("\n📋 Navigation Commands:")
        print("  'back' or 'b' - Go back to previous input")
        print("  'forward' or 'f' - Skip to next input")
        print("  'help' or 'h' - Show this help")
        print("  'restart' or 'r' - Restart from beginning")
        print("  'quit' or 'q' - Exit setup")
        print("  'history' - Show input history")
        print("  'goto <step>' - Jump to specific step (e.g., 'goto coil_diameter')")
    
    def handle_special_command(self, user_input: str) -> Optional[str]:
        """Handle special navigation commands."""
        user_input = user_input.strip().lower()
        
        if user_input in ['back', 'b']:
            if self.go_back():
                print("↩ Going back to previous input...")
                return "BACK"
            else:
                print("❌ Already at the beginning")
                return None
        elif user_input in ['forward', 'f']:
            if self.go_forward():
                print("→ Skipping to next input...")
                return "FORWARD"
            else:
                print("❌ Already at the end")
                return None
        elif user_input in ['help', 'h']:
            self.show_navigation_help()
            return "HELP"
        elif user_input in ['restart', 'r']:
            if input("Are you sure you want to restart? [y/N]: ").strip().lower() in ['y', 'yes']:
                self.input_history = []
                self.current_flow_index = 0
                print("🔄 Restarting setup...")
                return "RESTART"
            return None
        elif user_input in ['quit', 'q']:
            if input("Are you sure you want to quit? [y/N]: ").strip().lower() in ['y', 'yes']:
                print("\nSetup cancelled.")
                sys.exit(0)
            return None
        elif user_input == 'history':
            self.show_history()
            return "HISTORY"
        elif user_input.startswith('goto '):
            step_name = user_input[5:].strip()
            return self.goto_step(step_name)
        
        return None
    
    def goto_step(self, step_name: str) -> Optional[str]:
        """Jump to a specific step by name."""
        for i, (name, description) in enumerate(self.setup_flow):
            if name == step_name:
                self.current_flow_index = i
                print(f"→ Jumping to: {description}")
                return "GOTO"
        
        print(f"❌ Step '{step_name}' not found")
        print("Available steps:")
        for name, description in self.setup_flow:
            print(f"  {name}: {description}")
        return None
    
    def show_history(self):
        """Show the input history."""
        if not self.input_history:
            print("No input history yet.")
            return
        
        print("\n📜 Input History:")
        for i, entry in enumerate(self.input_history):
            current_marker = " ←" if self.setup_flow[self.current_flow_index][0] == entry['step_name'] else ""
            print(f"  {entry['step_name']}: {entry['value']}{current_marker}")
        
        print(f"\nCurrent step: {self.setup_flow[self.current_flow_index][1]}")
    
    def run_setup_flow(self) -> Dict[str, Any]:
        """Run the complete setup flow with backwards navigation."""
        print("\n🚀 Starting interactive setup with backwards navigation...")
        print("Type 'help' at any time to see navigation commands.")
        
        config = {}
        
        while self.current_flow_index < len(self.setup_flow):
            step_name, step_description = self.setup_flow[self.current_flow_index]
            
            print(f"\n📍 Step {self.current_flow_index + 1}/{len(self.setup_flow)}: {step_description}")
            
            # Get the value for this step
            value = self.get_step_value(step_name, step_description)
            
            if value is not None:
                config[step_name] = value
                self.add_to_history(step_name, value, step_description)
                self.current_flow_index += 1
            else:
                # User wants to go back or quit
                continue
        
        return config
    
    def get_step_value(self, step_name: str, step_description: str) -> Any:
        """Get the value for a specific setup step."""
        
        # Check if we have a previous value for this step
        previous_value = self.get_previous_value(step_name)
        if previous_value is not None:
            print(f"↩ Previous value: {previous_value}")
        
        if step_name == 'setup_mode':
            return self.select_setup_mode()
        elif step_name == 'coil_wire_material':
            return self.get_material_choice("coil wire", default="Copper", step_name=step_name)
        elif step_name == 'coil_inner_diameter':
            return self.get_float_input("Coil inner diameter", default=0.053, min_val=0.001, max_val=0.5, unit="m", step_name=step_name)
        elif step_name == 'coil_length':
            return self.get_float_input("Coil length", default=0.063, min_val=0.001, max_val=1.0, unit="m", step_name=step_name)
        elif step_name == 'coil_wire_gauge':
            return self.get_wire_gauge_input(step_name)
        elif step_name == 'coil_packing_factor':
            return self.get_float_input("Wire packing factor", default=0.9, min_val=0.5, max_val=0.95, step_name=step_name)
        elif step_name == 'coil_auto_calculate':
            return self.get_boolean_input("Auto-calculate optimal turns?", default=True, step_name=step_name)
        elif step_name == 'coil_num_layers':
            # Check if auto-calculate is enabled
            auto_calc = self.get_previous_value('coil_auto_calculate')
            if auto_calc:
                # Auto-calculate based on geometry and wire specs
                inner_diameter = self.get_previous_value('coil_inner_diameter') or 0.053
                length = self.get_previous_value('coil_length') or 0.063
                wire_gauge = self.get_previous_value('coil_wire_gauge') or '12'
                packing_factor = self.get_previous_value('coil_packing_factor') or 0.9
                
                # Calculate optimal geometry
                wire_specs = self.builder.get_wire_specifications()
                wire_diameter = wire_specs['awg_diameter_mm'].get(wire_gauge, 1.291) / 1000  # Convert to meters
                wire_with_insulation = wire_diameter * 1.1  # Assume 10% insulation
                turns_per_layer = int(length / (wire_with_insulation / packing_factor))
                
                # Suggest reasonable number of layers
                suggested_layers = max(1, min(10, int(0.01 / wire_diameter)))  # Aim for ~10mm total thickness
                
                print(f"\n🧮 Auto-calculated coil geometry:")
                print(f"  Wire diameter: {wire_diameter*1000:.3f}mm")
                print(f"  Turns per layer: {turns_per_layer}")
                print(f"  Suggested layers: {suggested_layers}")
                
                return suggested_layers
            else:
                return int(self.get_float_input("Number of layers", default=4, min_val=1, max_val=50, step_name=step_name))
        elif step_name == 'coil_total_turns':
            # Check if auto-calculate is enabled
            auto_calc = self.get_previous_value('coil_auto_calculate')
            if auto_calc:
                # Calculate total turns from layers and turns per layer
                num_layers = self.get_previous_value('coil_num_layers') or 4
                inner_diameter = self.get_previous_value('coil_inner_diameter') or 0.053
                length = self.get_previous_value('coil_length') or 0.063
                wire_gauge = self.get_previous_value('coil_wire_gauge') or '12'
                packing_factor = self.get_previous_value('coil_packing_factor') or 0.9
                
                wire_specs = self.builder.get_wire_specifications()
                wire_diameter = wire_specs['awg_diameter_mm'].get(wire_gauge, 1.291) / 1000
                wire_with_insulation = wire_diameter * 1.1
                turns_per_layer = int(length / (wire_with_insulation / packing_factor))
                total_turns = turns_per_layer * num_layers
                
                print(f"✓ Auto-calculated: {turns_per_layer} turns/layer × {num_layers} layers = {total_turns} total turns")
                return total_turns
            else:
                return int(self.get_float_input("Total turns", default=100, min_val=10, max_val=100000, step_name=step_name))
        elif step_name == 'projectile_material':
            return self.get_material_choice("projectile", default="Pure_Iron", step_name=step_name)
        elif step_name == 'projectile_diameter':
            return self.get_float_input("Projectile diameter", default=0.0508, min_val=0.001, max_val=0.1, unit="m", step_name=step_name)
        elif step_name == 'projectile_length':
            return self.get_float_input("Projectile length", default=0.063, min_val=0.001, max_val=0.5, unit="m", step_name=step_name)
        elif step_name == 'projectile_use_calculated_mass':
            return self.get_boolean_input("Use calculated mass?", default=True, step_name=step_name)
        elif step_name == 'projectile_mass':
            # Calculate mass from previous inputs
            diameter = self.get_previous_value('projectile_diameter') or 0.0508
            length = self.get_previous_value('projectile_length') or 0.063
            material = self.get_previous_value('projectile_material') or "Pure_Iron"
            material_props = self.builder.get_material_properties(material)
            material_density = material_props.get('density', 7850)
            volume = np.pi * (diameter/2)**2 * length
            calculated_mass = material_density * volume
            return self.get_float_input("Custom projectile mass", default=1.00, min_val=1e-6, max_val=5.0, unit="kg", step_name=step_name)
        elif step_name == 'projectile_initial_position':
            length = self.get_previous_value('projectile_length') or 0.063
            return self.get_float_input("Initial position (negative = behind coil center)", default=-0.025, min_val=-1.0, max_val=1.0, unit="m", step_name=step_name)
        elif step_name == 'capacitor_capacitance':
            return self.get_float_input("Capacitance", default=0.032, min_val=1e-6, max_val=1.0, unit="F", step_name=step_name)
        elif step_name == 'capacitor_initial_voltage':
            return self.get_float_input("Initial voltage", default=425.0, min_val=1, max_val=10000, unit="V", step_name=step_name)
        elif step_name == 'capacitor_esr':
            return self.get_float_input("Capacitor ESR (Equivalent Series Resistance)", default=0.01, min_val=1e-6, max_val=10.0, unit="Ω", step_name=step_name)
        elif step_name == 'capacitor_esl':
            return self.get_float_input("Capacitor ESL (Equivalent Series Inductance)", default=1e-7, min_val=1e-12, max_val=1e-3, unit="H", step_name=step_name)
        elif step_name == 'simulation_time_span':
            return self.get_float_input("Simulation time", default=0.1, min_val=1e-6, max_val=1.0, unit="s", step_name=step_name)
        elif step_name == 'magnetic_calculation_method':
            methods = ['finite_solenoid', 'biot_savart', 'finite_element', 'analytical']
            method_descriptions = {
                'finite_solenoid': 'Fast solenoid approximation (good for most cases)',
                'biot_savart': 'Accurate Biot-Savart integration (slower, very accurate)',
                'finite_element': 'FEM solver (research-grade, very slow)',
                'analytical': 'Analytical expressions where available'
            }
            print("\nMagnetic field calculation methods:")
            for i, method in enumerate(methods, 1):
                print(f"  {i}. {method} - {method_descriptions[method]}")
            choice = self.get_string_input(f"Select method (1-{len(methods)})", default="1", step_name=step_name)
            return methods[int(choice) - 1]
        elif step_name == 'magnetic_axial_discretization':
            return int(self.get_float_input("Axial discretization points", default=1000, min_val=100, max_val=20000, step_name=step_name))
        elif step_name == 'magnetic_radial_discretization':
            return int(self.get_float_input("Radial discretization points", default=100, min_val=10, max_val=2000, step_name=step_name))
        elif step_name == 'magnetic_include_saturation':
            return self.get_boolean_input("Include magnetic saturation?", default=True, step_name=step_name)
        elif step_name == 'magnetic_include_hysteresis':
            return self.get_boolean_input("Include hysteresis effects?", default=True, step_name=step_name)
        elif step_name == 'magnetic_include_eddy_currents':
            return self.get_boolean_input("Include eddy current effects?", default=True, step_name=step_name)
        elif step_name == 'magnetic_include_skin_effect':
            return self.get_boolean_input("Include skin effect?", default=True, step_name=step_name)
        elif step_name == 'force_reluctance':
            return self.get_boolean_input("Include reluctance force (∇L method)?", default=True, step_name=step_name)
        elif step_name == 'force_lorentz':
            return self.get_boolean_input("Include Lorentz force (J×B)?", default=True, step_name=step_name)
        elif step_name == 'force_maxwell_stress':
            return self.get_boolean_input("Include Maxwell stress tensor?", default=True, step_name=step_name)
        elif step_name == 'force_image':
            return self.get_boolean_input("Include magnetic image force?", default=True, step_name=step_name)
        elif step_name == 'force_eddy':
            return self.get_boolean_input("Include eddy current force?", default=True, step_name=step_name)
        elif step_name == 'ja_enabled':
            return self.get_boolean_input("Enable Jiles-Atherton hysteresis model?", default=True, step_name=step_name)
        elif step_name == 'eddy_enabled':
            return self.get_boolean_input("Enable 3D eddy current modeling?", default=True, step_name=step_name)
        elif step_name == 'thermal_enabled':
            return self.get_boolean_input("Enable thermal modeling?", default=True, step_name=step_name)
        elif step_name == 'energy_conservation_tol':
            return self.get_float_input("Energy conservation tolerance", default=1e-6, min_val=1e-12, max_val=1e-3, step_name=step_name)
        elif step_name == 'solver_method':
            methods = ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA']
            method_descriptions = {
                'RK45': 'Runge-Kutta 4(5) - Good general purpose',
                'RK23': 'Runge-Kutta 2(3) - Faster, less accurate',
                'Radau': 'Implicit Radau - Stiff problems',
                'BDF': 'Backward differentiation - Very stiff',
                'LSODA': 'Adaptive stiff/non-stiff'
            }
            print("\nSolver methods:")
            for i, method in enumerate(methods, 1):
                print(f"  {i}. {method} - {method_descriptions[method]}")
            choice = self.get_string_input(f"Select solver method (1-{len(methods)})", default="3", step_name=step_name)
            return methods[int(choice) - 1]
        elif step_name == 'solver_rtol':
            return self.get_float_input("Relative tolerance", default=1e-8, min_val=1e-12, max_val=1e-3, step_name=step_name)
        elif step_name == 'solver_atol':
            return self.get_float_input("Absolute tolerance", default=1e-10, min_val=1e-15, max_val=1e-6, step_name=step_name)
        elif step_name == 'solver_max_step':
            return self.get_float_input("Maximum step size", default=1e-4, min_val=1e-8, max_val=1e-2, unit="s", step_name=step_name)
        elif step_name == 'config_filename':
            suggested_name = f"coilgun_config_{self.current_mode.lower().replace('-', '_')}.json"
            return self.get_string_input("Configuration filename", default=suggested_name, step_name=step_name)
        else:
            print(f"❌ Unknown step: {step_name}")
            return None
    
    def get_wire_gauge_input(self, step_name: str) -> str:
        """Get wire gauge selection with backwards navigation."""
        wire_specs = self.builder.get_wire_specifications()
        available_awg = list(wire_specs.get('awg_diameter_mm', {}).keys())
        
        print("\nAvailable wire gauges (AWG):")
        for awg in available_awg:
            diameter = wire_specs['awg_diameter_mm'].get(awg, 0)
            current = wire_specs['current_capacity_A'].get(awg, 0)
            print(f"  AWG {awg}: {diameter:.3f}mm diameter, {current}A capacity")
        
        return self.get_string_input("Select wire gauge", default="12", step_name=step_name)
    
    def build_configuration_from_flow(self, flow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete configuration from flow results."""
        mode = flow_config.get('setup_mode', 'Advanced')
        
        # Build coil configuration
        coil_config = {
            'wire_material': flow_config.get('coil_wire_material', 'Copper'),
            'inner_diameter': flow_config.get('coil_inner_diameter', 0.053),
            'length': flow_config.get('coil_length', 0.063),
            'wire_gauge_awg': int(flow_config.get('coil_wire_gauge', '12')),
            'packing_factor': flow_config.get('coil_packing_factor', 0.9),
            'num_layers': flow_config.get('coil_num_layers', 4),
            'total_turns': flow_config.get('coil_total_turns', 100),
            'insulation_thickness': 3e-5
        }
        
        # Build projectile configuration
        projectile_config = {
            'material': flow_config.get('projectile_material', 'Pure_Iron'),
            'diameter': flow_config.get('projectile_diameter', 0.0508),
            'length': flow_config.get('projectile_length', 0.063),
            'mass': flow_config.get('projectile_mass', 1.0023687351506698),
            'initial_position': flow_config.get('projectile_initial_position', -0.025),
            'initial_velocity': 0.0
        }
        
        # Build capacitor configuration
        capacitor_config = {
            'capacitance': flow_config.get('capacitor_capacitance', 0.032),
            'initial_voltage': flow_config.get('capacitor_initial_voltage', 425.0),
            'esr': flow_config.get('capacitor_esr', 0.01),
            'esl': flow_config.get('capacitor_esl', 1e-7)
        }
        
        # Build simulation configuration
        simulation_time = flow_config.get('simulation_time_span', 0.1)
        simulation_config = {
            'time_span': [0, simulation_time],
            'max_step': 1e-6,
            'tolerance': 1e-8,
            'method': 'Radau'
        }
        
        # Build complete configuration
        complete_config = {
            'coil': coil_config,
            'projectile': projectile_config,
            'capacitor': capacitor_config,
            'simulation': simulation_config,
            'circuit_model': {
                'switch_resistance': 0.001,
                'switch_inductance': 1e-8,
                'parasitic_capacitance': 1e-11,
                'include_skin_effect': mode in ['Advanced', 'Expert', 'Custom'],
                'include_proximity_effect': mode in ['Expert', 'Custom']
            },
            'magnetic_model': {
                'calculation_method': flow_config.get('magnetic_calculation_method', 'finite_solenoid'),
                'axial_discretization': flow_config.get('magnetic_axial_discretization', 1000),
                'radial_discretization': flow_config.get('magnetic_radial_discretization', 100),
                'include_saturation': flow_config.get('magnetic_include_saturation', True),
                'include_hysteresis': flow_config.get('magnetic_include_hysteresis', True),
                'include_eddy_currents': flow_config.get('magnetic_include_eddy_currents', True),
                'include_skin_effect': flow_config.get('magnetic_include_skin_effect', True),
                'force_components': {
                    'reluctance_force': flow_config.get('force_reluctance', True),
                    'lorentz_force': flow_config.get('force_lorentz', True),
                    'maxwell_stress': flow_config.get('force_maxwell_stress', True),
                    'image_force': flow_config.get('force_image', True),
                    'eddy_force': flow_config.get('force_eddy', True)
                }
            },
            'advanced_physics': {
                'jiles_atherton': {'enabled': flow_config.get('ja_enabled', True)},
                'eddy_currents': {'enabled': flow_config.get('eddy_enabled', True)},
                'thermal': {'enabled': flow_config.get('thermal_enabled', True)},
                'energy_conservation': {'enabled': True, 'tolerance': flow_config.get('energy_conservation_tol', 1e-6)}
            },
            'solver': {
                'method': flow_config.get('solver_method', 'RK45'),
                'rtol': flow_config.get('solver_rtol', 1e-8),
                'atol': flow_config.get('solver_atol', 1e-10),
                'max_step': flow_config.get('solver_max_step', 1e-4)
            },
            'output': {
                'save_trajectory': True,
                'save_current_profile': True,
                'save_field_data': mode in ['Advanced', 'Expert', 'Custom'],
                'print_progress': True,
                'save_interval': 100 if mode != 'Expert' else 50
            },
            'environment': {
                'temperature': 293.15,
                'pressure': 101325,
                'humidity': 0.5
            }
        }
        
        return complete_config
    
    def get_float_input(self, prompt: str, default: Optional[float] = None, 
                       min_val: Optional[float] = None, max_val: Optional[float] = None,
                       unit: str = "", step_name: str = "") -> float:
        """Enhanced float input with physics validation and backwards navigation."""
        unit_str = f" ({unit})" if unit else ""
        
        while True:
            try:
                if default is not None:
                    full_prompt = f"{prompt}{unit_str} [default: {default}]: "
                    user_input = input(full_prompt).strip()
                    if not user_input:
                        value = default
                    else:
                        # Check for special commands
                        special_result = self.handle_special_command(user_input)
                        if special_result == "BACK":
                            return None  # Signal to go back
                        elif special_result == "FORWARD":
                            return default  # Use default and continue
                        elif special_result in ["HELP", "HISTORY", "GOTO"]:
                            continue
                        elif special_result == "RESTART":
                            return self.get_float_input(prompt, default, min_val, max_val, unit, step_name)
                        else:
                            value = float(user_input)
                else:
                    user_input = input(f"{prompt}{unit_str}: ").strip()
                    
                    # Check for special commands
                    special_result = self.handle_special_command(user_input)
                    if special_result == "BACK":
                        return None  # Signal to go back
                    elif special_result == "FORWARD":
                        return None  # Can't forward without default
                    elif special_result in ["HELP", "HISTORY", "GOTO"]:
                        continue
                    elif special_result == "RESTART":
                        return self.get_float_input(prompt, default, min_val, max_val, unit, step_name)
                    else:
                        value = float(user_input)
                
                # Physics-based validation
                if min_val is not None and value < min_val:
                    print(f"❌ Value must be >= {min_val} (physics constraint)")
                    continue
                if max_val is not None and value > max_val:
                    print(f"❌ Value must be <= {max_val} (safety limit)")
                    continue
                
                return value
                
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                sys.exit(0)
    
    def get_material_choice(self, purpose: str, default: str = None, step_name: str = "") -> str:
        """Get material choice with physics properties display and backwards navigation."""
        available_materials = self.builder.get_available_materials()
        
        print(f"\nSelect material for {purpose}:")
        for i, material in enumerate(available_materials, 1):
            props = self.builder.get_material_properties(material)
            desc = props.get('description', 'No description available')
            print(f"  {i}. {material} - {desc}")
        
        while True:
            try:
                if default:
                    choice = input(f"\nEnter choice (1-{len(available_materials)}) [default: {default}]: ").strip()
                    if not choice:
                        selected = default
                    else:
                        # Check for special commands
                        special_result = self.handle_special_command(choice)
                        if special_result == "BACK":
                            return None  # Signal to go back
                        elif special_result == "FORWARD":
                            return default  # Use default and continue
                        elif special_result in ["HELP", "HISTORY", "GOTO"]:
                            continue
                        elif special_result == "RESTART":
                            return self.get_material_choice(purpose, default, step_name)
                        else:
                            choice_idx = int(choice) - 1
                            if 0 <= choice_idx < len(available_materials):
                                selected = available_materials[choice_idx]
                            else:
                                print(f"Please enter a number between 1 and {len(available_materials)}")
                                continue
                else:
                    choice = input(f"\nEnter choice (1-{len(available_materials)}): ").strip()
                    
                    # Check for special commands
                    special_result = self.handle_special_command(choice)
                    if special_result == "BACK":
                        return None  # Signal to go back
                    elif special_result == "FORWARD":
                        return None  # Can't forward without default
                    elif special_result in ["HELP", "HISTORY", "GOTO"]:
                        continue
                    elif special_result == "RESTART":
                        return self.get_material_choice(purpose, default, step_name)
                    else:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(available_materials):
                            selected = available_materials[choice_idx]
                        else:
                            print(f"Please enter a number between 1 and {len(available_materials)}")
                            continue
                
                # Show material properties
                props = self.builder.get_material_properties(selected)
                print(f"\n✓ Selected: {selected}")
                if props:
                    print("  Properties:")
                    for key, value in props.items():
                        if isinstance(value, (int, float)) and key != 'description':
                            print(f"    {key}: {value}")
                
                return selected
                
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                sys.exit(0)
    
    def setup_coil_parameters(self, stage_info: str = "") -> Dict[str, Any]:
        """Setup coil parameters with physics-aware calculations."""
        print(f"\n{'='*60}")
        print(f"COIL CONFIGURATION{' - ' + stage_info if stage_info else ''}")
        print("="*60)
        
        # Select wire material first for resistance calculations
        wire_material = self.get_material_choice("coil wire", default="Copper", step_name="coil_wire_material")
        
        # Basic geometry
        inner_diameter = self.get_float_input(
            "Coil inner diameter", default=0.053, min_val=0.001, max_val=0.5, unit="m", step_name="coil_inner_diameter"
        )
        length = self.get_float_input(
            "Coil length", default=0.063, min_val=0.001, max_val=1.0, unit="m", step_name="coil_length"
        )
        
        # Wire selection with specifications
        wire_specs = self.builder.get_wire_specifications()
        available_awg = list(wire_specs.get('awg_diameter_mm', {}).keys())
        
        print("\nAvailable wire gauges (AWG):")
        for awg in available_awg:
            diameter = wire_specs['awg_diameter_mm'].get(awg, 0)
            current = wire_specs['current_capacity_A'].get(awg, 0)
            print(f"  AWG {awg}: {diameter:.3f}mm diameter, {current}A capacity")
        
        wire_gauge_awg = self.get_string_input("Select wire gauge", default="12", step_name="coil_wire_gauge")
        
        # Calculate optimal number of layers and turns
        if self.current_mode in ['Advanced', 'Expert']:
            print("\n🧮 Calculating optimal coil geometry...")
            
            wire_diameter = wire_specs['awg_diameter_mm'].get(wire_gauge_awg, 1.291) / 1000  # Convert to meters
            packing_factor = self.get_float_input(
                "Wire packing factor", default=0.9, min_val=0.5, max_val=0.95, step_name="coil_packing_factor"
            )
            
            # Auto-calculate or manual input
            auto_calc = self.get_boolean_input("Auto-calculate optimal turns?", default=True, step_name="coil_auto_calculate")
            if auto_calc:
                # Calculate based on fill factor and geometry
                wire_with_insulation = wire_diameter * 1.1  # Assume 10% insulation
                turns_per_layer = int(length / (wire_with_insulation / packing_factor))
                
                # Suggest reasonable number of layers
                suggested_layers = max(1, min(10, int(0.01 / wire_diameter)))  # Aim for ~10mm total thickness
                
                num_layers = int(self.get_float_input(
                    f"Number of layers (suggested: {suggested_layers})", 
                    default=suggested_layers, min_val=1, max_val=50, step_name="coil_num_layers"
                ))
                
                total_turns = turns_per_layer * num_layers
                print(f"✓ Calculated: {turns_per_layer} turns/layer × {num_layers} layers = {total_turns} total turns")
            else:
                num_layers = int(self.get_float_input("Number of layers", default=4, min_val=1, max_val=50, step_name="coil_num_layers"))
                total_turns = int(self.get_float_input("Total turns", default=100, min_val=10, max_val=100000, step_name="coil_total_turns"))
        else:
            # Basic mode - simple inputs
            num_layers = int(self.get_float_input("Number of layers", default=4, min_val=1, max_val=20, step_name="coil_num_layers"))
            total_turns = int(self.get_float_input("Total turns", default=100, min_val=10, max_val=50000, step_name="coil_total_turns"))
        
        config = {
            'inner_diameter': inner_diameter,
            'length': length,
            'wire_gauge_awg': int(wire_gauge_awg),
            'num_layers': num_layers,
            'total_turns': total_turns,
            'wire_material': wire_material,
            'packing_factor': packing_factor if self.current_mode in ['Advanced', 'Expert'] else 0.9,
            'insulation_thickness': 3e-5  # Default 30 micrometers
        }
        
        # Add advanced parameters for higher complexity modes
        if self.current_mode in ['Expert', 'Custom']:
            config['thermal_class'] = self.get_string_input("Wire thermal class", default="H", step_name="coil_thermal_class")
            config['insulation_type'] = self.get_string_input("Insulation type", default="polyimide", step_name="coil_insulation_type")
            
            # Calculate and display coil metrics
            temp_config = {'coil': config}
            if PHYSICS_ENGINE_AVAILABLE:
                try:
                    metrics = calculate_coil_metrics(temp_config)
                    print(f"\n📊 Calculated coil metrics:")
                    print(f"  Wire length: {metrics['total_wire_length']:.1f} m")
                    print(f"  Coil aspect ratio: {metrics['aspect_ratio']:.2f}")
                    print(f"  Turn density: {metrics['turn_density']:.0f} turns/m")
                except Exception as e:
                    print(f"Warning: Could not calculate metrics: {e}")
        
        return config
    
    def setup_projectile_parameters(self) -> Dict[str, Any]:
        """Setup projectile parameters with material-aware mass calculation."""
        print("\n" + "="*60)
        print("PROJECTILE CONFIGURATION")
        print("="*60)
        
        # Select material first for density
        material = self.get_material_choice("projectile", default="Pure_Iron", step_name="projectile_material")
        material_props = self.builder.get_material_properties(material)
        material_density = material_props.get('density', 7850)  # kg/m³
        
        # Geometry
        diameter = self.get_float_input(
            "Projectile diameter", default=0.0508, min_val=0.001, max_val=0.1, unit="m", step_name="projectile_diameter"
        )
        length = self.get_float_input(
            "Projectile length", default=0.063, min_val=0.001, max_val=0.5, unit="m", step_name="projectile_length"
        )
        
        # Calculate mass from material density
        volume = np.pi * (diameter/2)**2 * length
        calculated_mass = material_density * volume
        
        print(f"\n🧮 Calculated mass from material density:")
        print(f"  Volume: {volume*1e6:.2f} mm³")
        print(f"  Density ({material}): {material_density} kg/m³")
        print(f"  Calculated mass: {calculated_mass*1000:.2f} g")
        
        use_calculated = self.get_boolean_input("Use calculated mass?", default=True, step_name="projectile_use_calculated_mass")
        if use_calculated:
            mass = calculated_mass
        else:
            mass = self.get_float_input(
                "Custom projectile mass", default=1.0023687351506698, 
                min_val=1e-6, max_val=5.0, unit="kg", step_name="projectile_mass"
            )
        
        # Initial conditions
        initial_position = self.get_float_input(
            "Initial position (negative = behind coil center)", 
            default=-0.025, min_val=-1.0, max_val=1.0, unit="m", step_name="projectile_initial_position"
        )
        
        config = {
            'diameter': diameter,
            'length': length,
            'mass': mass,
            'material': material,
            'initial_position': initial_position,
            'initial_velocity': 0.0
        }
        
        # Add advanced parameters for complex modes
        if self.current_mode in ['Expert', 'Custom']:
            config['surface_roughness'] = self.get_float_input(
                "Surface roughness (RMS)", default=1e-6, min_val=1e-9, max_val=1e-3, unit="m", step_name="projectile_surface_roughness"
            )
            config['hardness'] = material_props.get('hardness_hv', 100)
        
        return config
    
    def setup_capacitor_parameters(self, stage_info: str = "") -> Dict[str, Any]:
        """Setup capacitor parameters with energy calculations."""
        print(f"\n{'='*60}")
        print(f"CAPACITOR CONFIGURATION{' - ' + stage_info if stage_info else ''}")
        print("="*60)
        
        # Basic parameters
        capacitance = self.get_float_input(
            "Capacitance", default=0.032, min_val=1e-6, max_val=1.0, unit="F", step_name="capacitor_capacitance"
        )
        initial_voltage = self.get_float_input(
            "Initial voltage", default=425.0, min_val=1, max_val=10000, unit="V", step_name="capacitor_initial_voltage"
        )
        
        # Calculate and display energy
        energy = 0.5 * capacitance * initial_voltage**2
        print(f"\n⚡ Stored energy: {energy:.2f} J ({energy*1000:.0f} mJ)")
        
        config = {
            'capacitance': capacitance,
            'initial_voltage': initial_voltage
        }
        
        # Add parasitic elements for advanced modes
        if self.current_mode in ['Advanced', 'Expert', 'Custom']:
            config['esr'] = self.get_float_input(
                "Equivalent series resistance (ESR)", 
                default=0.01, min_val=1e-6, max_val=10.0, unit="Ω", step_name="capacitor_esr"
            )
            config['esl'] = self.get_float_input(
                "Equivalent series inductance (ESL)", 
                default=1e-7, min_val=1e-12, max_val=1e-3, unit="H", step_name="capacitor_esl"
            )
        else:
            config['esr'] = 0.01
            config['esl'] = 1e-7
        
        return config


    def setup_advanced_physics_parameters(self) -> Dict[str, Any]:
        """Setup advanced physics model parameters based on complexity level."""
        print("\n" + "="*60)
        print("ADVANCED PHYSICS CONFIGURATION")
        print("="*60)
        
        config = {}
        
        if self.current_mode == 'Basic':
            # Basic mode - minimal physics
            config = {
                'jiles_atherton': {'enabled': False},
                'eddy_currents': {'enabled': False},
                'thermal': {'enabled': False},
                'energy_conservation': {'enabled': True, 'tolerance': 1e-4}
            }
        
        elif self.current_mode == 'Advanced':
            # Advanced mode - standard accuracy physics
            config = {
                'jiles_atherton': {
                    'enabled': True,
                    'use_default_params': True
                },
                'eddy_currents': {
                    'enabled': True,
                    'discretization_method': 'adaptive'
                },
                'thermal': {
                    'enabled': True,
                    'coupling_strength': 'weak'
                },
                'energy_conservation': {
                    'enabled': True,
                    'tolerance': 1e-6
                }
            }
        
        elif self.current_mode in ['Expert', 'Custom']:
            # Expert - full physics with custom parameters
            print("Configuring research-grade physics models...")
            
            # Jiles-Atherton hysteresis
            if self.get_boolean_input("\nEnable Jiles-Atherton hysteresis model?", default=True, step_name="ja_enabled"):
                ja_config = {'enabled': True}
                
                if self.get_boolean_input("Use material-specific J-A parameters?", default=True, step_name="ja_use_material_defaults"):
                    ja_config['use_material_defaults'] = True
                else:
                    # Custom J-A parameters
                    ja_config.update({
                        'Ms': self.get_float_input("Saturation magnetization Ms", default=1.7e6, unit="A/m", step_name="ja_Ms"),
                        'a': self.get_float_input("Shape parameter 'a'", default=1000, unit="A/m", step_name="ja_a"),
                        'alpha': self.get_float_input("Interdomain coupling 'alpha'", default=1e-3, step_name="ja_alpha"),
                        'c': self.get_float_input("Reversible fraction 'c'", default=0.1, min_val=0, max_val=1, step_name="ja_c"),
                        'k': self.get_float_input("Pinning parameter 'k'", default=500, unit="A/m", step_name="ja_k")
                    })
                
                config['jiles_atherton'] = ja_config
            else:
                config['jiles_atherton'] = {'enabled': False}
            
            # Eddy current modeling
            if self.get_boolean_input("\nEnable 3D eddy current modeling?", default=True, step_name="eddy_enabled"):
                eddy_config = {
                    'enabled': True,
                    'discretization_method': 'adaptive',
                    'skin_depth_calculation': True,
                    'frequency_dependence': True
                }
                
                if self.current_mode == 'Custom':
                    eddy_config['mesh_refinement_levels'] = int(self.get_float_input(
                        "Mesh refinement levels", default=3, min_val=1, max_val=6, step_name="eddy_mesh_levels"
                    ))
                    eddy_config['convergence_tolerance'] = self.get_float_input(
                        "Convergence tolerance", default=1e-6, min_val=1e-12, max_val=1e-3, step_name="eddy_convergence_tol"
                    )
                
                config['eddy_currents'] = eddy_config
            else:
                config['eddy_currents'] = {'enabled': False}
            
            # Thermal modeling
            if self.get_boolean_input("\nEnable thermal modeling?", default=True, step_name="thermal_enabled"):
                thermal_config = {
                    'enabled': True,
                    'coupling_strength': 'strong',
                    'ambient_temperature': self.get_float_input(
                        "Ambient temperature", default=293.15, min_val=200, max_val=400, unit="K", step_name="thermal_ambient_temp"
                    ),
                    'convection_coefficient': self.get_float_input(
                        "Convection coefficient", default=10, min_val=1, max_val=100, unit="W/(m²⋅K)", step_name="thermal_convection_coeff"
                    )
                }
                config['thermal'] = thermal_config
            else:
                config['thermal'] = {'enabled': False}
            
            # Energy conservation
            config['energy_conservation'] = {
                'enabled': True,
                'tolerance': self.get_float_input(
                    "Energy conservation tolerance", default=1e-8, min_val=1e-12, max_val=1e-3, step_name="energy_conservation_tol"
                ),
                'check_interval': int(self.get_float_input(
                    "Energy check interval (steps)", default=100, min_val=1, max_val=1000, step_name="energy_conservation_check_interval"
                ))
            }
        
        return config
    
    def setup_magnetic_model_parameters(self) -> Dict[str, Any]:
        """Setup magnetic field calculation parameters."""
        print("\n" + "="*60)
        print("MAGNETIC MODEL CONFIGURATION")
        print("="*60)
        
        if self.current_mode == 'Basic':
            # Basic mode - simple finite solenoid
            return {
                'calculation_method': 'finite_solenoid',
                'axial_discretization': 1000,
                'radial_discretization': 100,
                'include_saturation': False,
                'include_hysteresis': False,
                'include_eddy_currents': False,
                'force_components': {
                    'reluctance_force': True,
                    'lorentz_force': False,
                    'maxwell_stress': False,
                    'image_force': False,
                    'eddy_force': False
                }
            }
        
        # Advanced modes get user choice
        methods = ['finite_solenoid', 'biot_savart', 'finite_element', 'analytical']
        method_descriptions = {
            'finite_solenoid': 'Fast solenoid approximation (good for most cases)',
            'biot_savart': 'Accurate Biot-Savart integration (slower, very accurate)',
            'finite_element': 'FEM solver (research-grade, very slow)',
            'analytical': 'Analytical expressions where available'
        }
        
        print("\nMagnetic field calculation methods:")
        for i, method in enumerate(methods, 1):
            print(f"  {i}. {method} - {method_descriptions[method]}")
        
        choice = self.get_string_input(f"Select method (1-{len(methods)})", default="1", step_name="magnetic_calculation_method")
        calculation_method = methods[int(choice) - 1]
        
        config = {
            'calculation_method': calculation_method,
            'axial_discretization': int(self.get_float_input(
                "Axial discretization points", 
                default=1000 if self.current_mode != 'Expert' else 5000,
                min_val=100, max_val=20000, step_name="magnetic_axial_discretization"
            )),
            'radial_discretization': int(self.get_float_input(
                "Radial discretization points",
                default=100 if self.current_mode != 'Expert' else 500,
                min_val=10, max_val=2000, step_name="magnetic_radial_discretization"
            ))
        }
        
        # Physics effects
        config.update({
            'include_saturation': self.get_boolean_input("Include magnetic saturation?", default=True, step_name="magnetic_include_saturation"),
            'include_hysteresis': self.get_boolean_input("Include hysteresis effects?", default=False, step_name="magnetic_include_hysteresis"),
            'include_eddy_currents': self.get_boolean_input("Include eddy current effects?", default=True, step_name="magnetic_include_eddy_currents"),
            'include_skin_effect': self.get_boolean_input("Include skin effect?", default=True, step_name="magnetic_include_skin_effect"),
        })
        
        # Force components
        print("\nForce calculation components:")
        force_components = {
            'reluctance_force': self.get_boolean_input("Include reluctance force (∇L method)?", default=True, step_name="force_reluctance"),
            'lorentz_force': self.get_boolean_input("Include Lorentz force (J×B)?", default=True, step_name="force_lorentz"),
            'maxwell_stress': self.get_boolean_input("Include Maxwell stress tensor?", default=True, step_name="force_maxwell_stress"),
            'image_force': self.get_boolean_input("Include magnetic image force?", default=True, step_name="force_image"),
            'eddy_force': config['include_eddy_currents'] and self.get_boolean_input("Include eddy current force?", default=True, step_name="force_eddy")
        }
        config['force_components'] = force_components
        
        return config
    
    def setup_solver_parameters(self) -> Dict[str, Any]:
        """Setup solver-specific parameters."""
        print("\n" + "="*60)
        print("SOLVER CONFIGURATION")
        print("="*60)
        
        # Basic solver parameters
        config = {
            'method': 'RK45',  # Default scipy method
            'rtol': 1e-8,
            'atol': 1e-10,
            'max_step': 1e-4
        }
        
        if self.current_mode in ['Advanced', 'Expert', 'Custom']:
            # Solver method selection
            methods = ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA']
            method_descriptions = {
                'RK45': 'Runge-Kutta 4(5) - Good general purpose',
                'RK23': 'Runge-Kutta 2(3) - Faster, less accurate',
                'Radau': 'Implicit Radau - Stiff problems',
                'BDF': 'Backward differentiation - Very stiff',
                'LSODA': 'Adaptive stiff/non-stiff'
            }
            
            print("\nSolver methods:")
            for i, method in enumerate(methods, 1):
                print(f"  {i}. {method} - {method_descriptions[method]}")
            
            choice = self.get_string_input(f"Select solver method (1-{len(methods)})", default="3", step_name="solver_method")
            config['method'] = methods[int(choice) - 1]
            
            # Tolerances
            config['rtol'] = self.get_float_input(
                "Relative tolerance", default=1e-8, min_val=1e-12, max_val=1e-3, step_name="solver_rtol"
            )
            config['atol'] = self.get_float_input(
                "Absolute tolerance", default=1e-10, min_val=1e-15, max_val=1e-6, step_name="solver_atol"
            )
            config['max_step'] = self.get_float_input(
                "Maximum step size", default=1e-4, min_val=1e-8, max_val=1e-2, unit="s", step_name="solver_max_step"
            )
        
        return config
    
    def create_complete_configuration(self) -> Dict[str, Any]:
        """Create a complete configuration based on the selected mode."""
        print(f"\n🔧 Creating {self.current_mode} configuration...")
        
        config = {}
        
        # Basic components (always required)
        config['coil'] = self.setup_coil_parameters()
        config['projectile'] = self.setup_projectile_parameters()
        config['capacitor'] = self.setup_capacitor_parameters()
        
        # Simulation parameters
        config['simulation'] = {
            'time_span': [0, self.get_float_input("Simulation time", default=0.1, min_val=1e-6, max_val=1.0, unit="s", step_name="simulation_time_span")],
            'max_step': 1e-6,
            'tolerance': 1e-8,
            'method': 'Radau'
        }
        
        # Circuit model (basic for all modes)
        config['circuit_model'] = {
            'switch_resistance': 0.001,
            'switch_inductance': 1e-8,
            'parasitic_capacitance': 1e-11,
            'include_skin_effect': self.current_mode in ['Advanced', 'Expert', 'Custom'],
            'include_proximity_effect': self.current_mode in ['Expert', 'Custom']
        }
        
        # Advanced configurations
        config['magnetic_model'] = self.setup_magnetic_model_parameters()
        config['advanced_physics'] = self.setup_advanced_physics_parameters()
        config['solver'] = self.setup_solver_parameters()
        
        # Output configuration
        config['output'] = {
            'save_trajectory': True,
            'save_current_profile': True,
            'save_field_data': self.current_mode in ['Advanced', 'Expert', 'Custom'],
            'print_progress': True,
            'save_interval': 100 if self.current_mode != 'Expert' else 50
        }
        
        # Environment settings
        config['environment'] = {
            'temperature': 293.15,  # Room temperature fallback
            'pressure': 101325,  # Pa
            'humidity': 0.5
        }
        
        return config

    def select_setup_mode(self) -> str:
        """Allow user to select setup complexity level."""
        print("\n" + "="*60)
        print("CONFIGURATION SETUP MODE SELECTION")
        print("="*60)
        print("Choose your setup complexity level:")
        print("  1. Basic     - Simple single-stage coilgun (educational)")
        print("  2. Advanced  - Multi-physics with standard accuracy")
        print("  3. Expert    - Full physics with research-grade accuracy")
        print("  4. Custom    - Expert mode with all options")
        
        while True:
            try:
                choice = input("\nSelect mode (1-4) [2]: ").strip()
                if not choice:
                    choice = '2'
                
                # Check for special commands
                special_result = self.handle_special_command(choice)
                if special_result == "BACK":
                    return None  # Signal to go back
                elif special_result == "FORWARD":
                    return "Advanced"  # Use default and continue
                elif special_result in ["HELP", "HISTORY", "GOTO"]:
                    continue
                elif special_result == "RESTART":
                    return self.select_setup_mode()
                
                mode_map = {'1': 'Basic', '2': 'Advanced', '3': 'Expert', '4': 'Custom'}
                if choice in mode_map:
                    self.current_mode = mode_map[choice]
                    print(f"Selected: {self.current_mode} mode")
                    return self.current_mode
                else:
                    print("Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                sys.exit(0)
    
    def get_string_input(self, prompt: str, default: Optional[str] = None, 
                        step_name: str = "", allow_empty: bool = False) -> str:
        """Get string input with backwards navigation support."""
        while True:
            try:
                if default is not None:
                    full_prompt = f"{prompt} [default: {default}]: "
                    user_input = input(full_prompt).strip()
                    if not user_input:
                        value = default
                    else:
                        # Check for special commands
                        special_result = self.handle_special_command(user_input)
                        if special_result == "BACK":
                            return None  # Signal to go back
                        elif special_result == "FORWARD":
                            return default  # Use default and continue
                        elif special_result in ["HELP", "HISTORY", "GOTO"]:
                            continue
                        elif special_result == "RESTART":
                            return self.get_string_input(prompt, default, step_name, allow_empty)
                        else:
                            value = user_input
                else:
                    user_input = input(f"{prompt}: ").strip()
                    
                    # Check for special commands
                    special_result = self.handle_special_command(user_input)
                    if special_result == "BACK":
                        return None  # Signal to go back
                    elif special_result == "FORWARD":
                        return None  # Can't forward without default
                    elif special_result in ["HELP", "HISTORY", "GOTO"]:
                        continue
                    elif special_result == "RESTART":
                        return self.get_string_input(prompt, default, step_name, allow_empty)
                    else:
                        value = user_input
                
                # Validation
                if not allow_empty and not value:
                    print("❌ This field cannot be empty")
                    continue
                
                return value
                
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                sys.exit(0)
    
    def get_boolean_input(self, prompt: str, default: Optional[bool] = None, 
                         step_name: str = "") -> bool:
        """Get boolean input with backwards navigation support."""
        while True:
            try:
                if default is not None:
                    default_str = "Y" if default else "N"
                    full_prompt = f"{prompt} [Y/n] [default: {default_str}]: "
                    user_input = input(full_prompt).strip().lower()
                    if not user_input:
                        value = default
                    else:
                        # Check for special commands
                        special_result = self.handle_special_command(user_input)
                        if special_result == "BACK":
                            return None  # Signal to go back
                        elif special_result == "FORWARD":
                            return default  # Use default and continue
                        elif special_result in ["HELP", "HISTORY", "GOTO"]:
                            continue
                        elif special_result == "RESTART":
                            return self.get_boolean_input(prompt, default, step_name)
                        elif user_input in ['y', 'yes', 'true', '1']:
                            value = True
                        elif user_input in ['n', 'no', 'false', '0']:
                            value = False
                        else:
                            print("Please enter Y/n, yes/no, true/false, or 1/0")
                            continue
                else:
                    user_input = input(f"{prompt} [Y/n]: ").strip().lower()
                    
                    # Check for special commands
                    special_result = self.handle_special_command(user_input)
                    if special_result == "BACK":
                        return None  # Signal to go back
                    elif special_result == "FORWARD":
                        return None  # Can't forward without default
                    elif special_result in ["HELP", "HISTORY", "GOTO"]:
                        continue
                    elif special_result == "RESTART":
                        return self.get_boolean_input(prompt, default, step_name)
                    elif user_input in ['y', 'yes', 'true', '1']:
                        value = True
                    elif user_input in ['n', 'no', 'false', '0']:
                        value = False
                    else:
                        print("Please enter Y/n, yes/no, true/false, or 1/0")
                        continue
                
                return value
                
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                sys.exit(0)


def validate_and_save_configuration(config: Dict[str, Any], filename: str) -> bool:
    """Validate configuration against physics engine and save."""
    print(f"\n🔍 Validating configuration...")
    
    # Physics engine validation
    if PHYSICS_ENGINE_AVAILABLE:
        try:
            from physics.utils import validate_coilgun_config
            is_valid, errors = validate_coilgun_config(config)
            
            if not is_valid:
                print("❌ Physics validation errors:")
                for error in errors:
                    print(f"  • {error}")
                
                # Note: This is outside the setup interface, so we can't use the enhanced input methods
                if input("Save anyway? [y/N]: ").strip().lower() not in ['y', 'yes']:
                    return False
            else:
                print("✓ Physics validation passed")
        except Exception as e:
            print(f"⚠ Could not validate with physics engine: {e}")
    
    # Save configuration
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"✓ Configuration saved to: {filename}")
        
        # Test load
        with open(filename, 'r') as f:
            test_config = json.load(f)
        print("✓ Configuration file verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False


def print_configuration_summary(config: Dict[str, Any], mode: str) -> None:
    """Print a comprehensive configuration summary."""
    print("\n" + "="*70)
    print(f"CONFIGURATION SUMMARY - {mode.upper()} MODE")
    print("="*70)
    
    # Coil summary
    coil = config['coil']
    print(f"🔧 COIL:")
    print(f"  Geometry: {coil['inner_diameter']*1000:.1f}mm ID × {coil['length']*1000:.1f}mm length")
    print(f"  Wire: AWG {coil['wire_gauge_awg']} {coil['wire_material']}")
    print(f"  Turns: {coil['total_turns']} ({coil['num_layers']} layers)")
    
    # Projectile summary
    proj = config['projectile']
    print(f"\n🎯 PROJECTILE:")
    print(f"  Material: {proj['material']}")
    print(f"  Dimensions: {proj['diameter']*1000:.1f}mm × {proj['length']*1000:.1f}mm")
    print(f"  Mass: {proj['mass']*1000:.2f}g")
    
    # Capacitor summary
    cap = config['capacitor']
    energy = 0.5 * cap['capacitance'] * cap['initial_voltage']**2
    print(f"\n⚡ CAPACITOR:")
    print(f"  Capacitance: {cap['capacitance']*1000:.1f}mF")
    print(f"  Voltage: {cap['initial_voltage']:.0f}V")
    print(f"  Energy: {energy:.2f}J")
    
    # Physics summary
    mag_model = config.get('magnetic_model', {})
    adv_physics = config.get('advanced_physics', {})
    print(f"\n🔬 PHYSICS:")
    print(f"  Field method: {mag_model.get('calculation_method', 'finite_solenoid')}")
    print(f"  Discretization: {mag_model.get('axial_discretization', 1000)} axial points")
    print(f"  Saturation: {'✓' if mag_model.get('include_saturation') else '✗'}")
    print(f"  Hysteresis: {'✓' if adv_physics.get('jiles_atherton', {}).get('enabled') else '✗'}")
    print(f"  Eddy currents: {'✓' if adv_physics.get('eddy_currents', {}).get('enabled') else '✗'}")
    
    print(f"\n{'='*70}")


def main():
    """Main setup function with enhanced physics integration."""
    print("\n" + "="*70)
    print("ADVANCED COILGUN CONFIGURATION SETUP")
    print("="*70)
    print("Physics-Engine Integrated Configuration Builder")
    if PHYSICS_ENGINE_AVAILABLE:
        print("✓ Advanced physics engine available")
    else:
        print("⚠ Physics engine not available - using basic mode")
    print("="*70)
    
    try:
        setup_interface = AdvancedSetupInterface()
        
        # Show navigation help at the beginning
        setup_interface.show_navigation_help()
        
        # Run the interactive setup flow
        config = setup_interface.run_setup_flow()
        
        if not config:
            print("\n❌ Setup cancelled or failed")
            return
        
        # Extract key values for configuration building
        mode = config.get('setup_mode', 'Advanced')
        filename = config.get('config_filename', f"coilgun_config_{mode.lower().replace('-', '_')}.json")
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Build complete configuration from flow results
        complete_config = setup_interface.build_configuration_from_flow(config)
        
        # Validate and save
        if validate_and_save_configuration(complete_config, filename):
            print_configuration_summary(complete_config, mode)
            
            print(f"\n🚀 Configuration created successfully!")
            print(f"\nTo run simulation:")
            print(f"  python solve.py {filename}")
            print(f"\nTo visualize results:")
            print(f"  python view.py {filename}")
        else:
            print("\n❌ Configuration creation failed")
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    
    main() 