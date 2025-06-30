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
    """Advanced setup interface with physics-aware configuration."""
    
    def __init__(self):
        """Initialize the setup interface."""
        self.builder = ConfigurationBuilder()
        self.setup_modes = ['Basic', 'Advanced', 'Expert', 'Custom']
        self.current_mode = 'Advanced'
    
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
    
    def get_float_input(self, prompt: str, default: Optional[float] = None, 
                       min_val: Optional[float] = None, max_val: Optional[float] = None,
                       unit: str = "") -> float:
        """Enhanced float input with physics validation."""
        unit_str = f" ({unit})" if unit else ""
        while True:
            try:
                if default is not None:
                    full_prompt = f"{prompt}{unit_str} [default: {default}]: "
                    user_input = input(full_prompt).strip()
                    if not user_input:
                        return default
                else:
                    user_input = input(f"{prompt}{unit_str}: ").strip()
                
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
    
    def get_material_choice(self, purpose: str, default: str = None) -> str:
        """Get material choice with physics properties display."""
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
                        return default
                else:
                    choice = input(f"\nEnter choice (1-{len(available_materials)}): ").strip()
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_materials):
                    selected = available_materials[choice_idx]
                    
                    # Show material properties
                    props = self.builder.get_material_properties(selected)
                    print(f"\n✓ Selected: {selected}")
                    if props:
                        print("  Properties:")
                        for key, value in props.items():
                            if isinstance(value, (int, float)) and key != 'description':
                                print(f"    {key}: {value}")
                    
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(available_materials)}")
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
        wire_material = self.get_material_choice("coil wire", default="Copper")
        
        # Basic geometry
        inner_diameter = self.get_float_input(
            "Coil inner diameter", default=0.02, min_val=0.001, max_val=0.5, unit="m"
        )
        length = self.get_float_input(
            "Coil length", default=0.05, min_val=0.001, max_val=1.0, unit="m"
        )
        
        # Wire selection with specifications
        wire_specs = self.builder.get_wire_specifications()
        available_awg = list(wire_specs.get('awg_diameter_mm', {}).keys())
        
        print("\nAvailable wire gauges (AWG):")
        for awg in available_awg:
            diameter = wire_specs['awg_diameter_mm'].get(awg, 0)
            current = wire_specs['current_capacity_A'].get(awg, 0)
            print(f"  AWG {awg}: {diameter:.3f}mm diameter, {current}A capacity")
        
        wire_gauge_awg = input(f"\nSelect wire gauge [16]: ").strip() or "16"
        
        # Calculate optimal number of layers and turns
        if self.current_mode in ['Advanced', 'Expert']:
            print("\n🧮 Calculating optimal coil geometry...")
            
            wire_diameter = wire_specs['awg_diameter_mm'].get(wire_gauge_awg, 1.291) / 1000  # Convert to meters
            packing_factor = self.get_float_input(
                "Wire packing factor", default=0.85, min_val=0.5, max_val=0.95
            )
            
            # Auto-calculate or manual input
            auto_calc = input("Auto-calculate optimal turns? [Y/n]: ").strip().lower()
            if auto_calc in ['', 'y', 'yes']:
                # Calculate based on fill factor and geometry
                wire_with_insulation = wire_diameter * 1.1  # Assume 10% insulation
                turns_per_layer = int(length / (wire_with_insulation / packing_factor))
                
                # Suggest reasonable number of layers
                suggested_layers = max(1, min(10, int(0.01 / wire_diameter)))  # Aim for ~10mm total thickness
                
                num_layers = int(self.get_float_input(
                    f"Number of layers (suggested: {suggested_layers})", 
                    default=suggested_layers, min_val=1, max_val=50
                ))
                
                total_turns = turns_per_layer * num_layers
                print(f"✓ Calculated: {turns_per_layer} turns/layer × {num_layers} layers = {total_turns} total turns")
            else:
                num_layers = int(self.get_float_input("Number of layers", default=3, min_val=1, max_val=50))
                total_turns = int(self.get_float_input("Total turns", default=1000, min_val=10, max_val=100000))
        else:
            # Basic mode - simple inputs
            num_layers = int(self.get_float_input("Number of layers", default=3, min_val=1, max_val=20))
            total_turns = int(self.get_float_input("Total turns", default=1000, min_val=10, max_val=50000))
        
        config = {
            'inner_diameter': inner_diameter,
            'length': length,
            'wire_gauge_awg': int(wire_gauge_awg),
            'num_layers': num_layers,
            'total_turns': total_turns,
            'wire_material': wire_material,
            'packing_factor': packing_factor if self.current_mode in ['Advanced', 'Expert'] else 0.85,
            'insulation_thickness': 3e-5  # Default 30 micrometers
        }
        
        # Add advanced parameters for higher complexity modes
        if self.current_mode in ['Expert', 'Custom']:
            config['thermal_class'] = input("Wire thermal class [H]: ").strip() or "H"
            config['insulation_type'] = input("Insulation type [polyimide]: ").strip() or "polyimide"
            
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
        material = self.get_material_choice("projectile", default="Pure_Iron")
        material_props = self.builder.get_material_properties(material)
        material_density = material_props.get('density', 7850)  # kg/m³
        
        # Geometry
        diameter = self.get_float_input(
            "Projectile diameter", default=0.008, min_val=0.001, max_val=0.1, unit="m"
        )
        length = self.get_float_input(
            "Projectile length", default=0.015, min_val=0.001, max_val=0.5, unit="m"
        )
        
        # Calculate mass from material density
        volume = np.pi * (diameter/2)**2 * length
        calculated_mass = material_density * volume
        
        print(f"\n🧮 Calculated mass from material density:")
        print(f"  Volume: {volume*1e6:.2f} mm³")
        print(f"  Density ({material}): {material_density} kg/m³")
        print(f"  Calculated mass: {calculated_mass*1000:.2f} g")
        
        use_calculated = input("Use calculated mass? [Y/n]: ").strip().lower()
        if use_calculated in ['', 'y', 'yes']:
            mass = calculated_mass
        else:
            mass = self.get_float_input(
                "Custom projectile mass", default=calculated_mass, 
                min_val=1e-6, max_val=5.0, unit="kg"
            )
        
        # Initial conditions
        initial_position = self.get_float_input(
            "Initial position (negative = behind coil center)", 
            default=-length, min_val=-1.0, max_val=1.0, unit="m"
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
                "Surface roughness (RMS)", default=1e-6, min_val=1e-9, max_val=1e-3, unit="m"
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
            "Capacitance", default=0.001, min_val=1e-6, max_val=1.0, unit="F"
        )
        initial_voltage = self.get_float_input(
            "Initial voltage", default=400, min_val=1, max_val=10000, unit="V"  # SafetyLimits.MAX_VOLTAGE fallback
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
                default=0.01, min_val=1e-6, max_val=10.0, unit="Ω"
            )
            config['esl'] = self.get_float_input(
                "Equivalent series inductance (ESL)", 
                default=1e-7, min_val=1e-12, max_val=1e-3, unit="H"
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
            if input("\nEnable Jiles-Atherton hysteresis model? [Y/n]: ").strip().lower() not in ['n', 'no']:
                ja_config = {'enabled': True}
                
                if input("Use material-specific J-A parameters? [Y/n]: ").strip().lower() not in ['n', 'no']:
                    ja_config['use_material_defaults'] = True
                else:
                    # Custom J-A parameters
                    ja_config.update({
                        'Ms': self.get_float_input("Saturation magnetization Ms", default=1.7e6, unit="A/m"),
                        'a': self.get_float_input("Shape parameter 'a'", default=1000, unit="A/m"),
                        'alpha': self.get_float_input("Interdomain coupling 'alpha'", default=1e-3),
                        'c': self.get_float_input("Reversible fraction 'c'", default=0.1, min_val=0, max_val=1),
                        'k': self.get_float_input("Pinning parameter 'k'", default=500, unit="A/m")
                    })
                
                config['jiles_atherton'] = ja_config
            else:
                config['jiles_atherton'] = {'enabled': False}
            
            # Eddy current modeling
            if input("\nEnable 3D eddy current modeling? [Y/n]: ").strip().lower() not in ['n', 'no']:
                eddy_config = {
                    'enabled': True,
                    'discretization_method': 'adaptive',
                    'skin_depth_calculation': True,
                    'frequency_dependence': True
                }
                
                if self.current_mode == 'Custom':
                    eddy_config['mesh_refinement_levels'] = int(self.get_float_input(
                        "Mesh refinement levels", default=3, min_val=1, max_val=6
                    ))
                    eddy_config['convergence_tolerance'] = self.get_float_input(
                        "Convergence tolerance", default=1e-6, min_val=1e-12, max_val=1e-3
                    )
                
                config['eddy_currents'] = eddy_config
            else:
                config['eddy_currents'] = {'enabled': False}
            
            # Thermal modeling
            if input("\nEnable thermal modeling? [Y/n]: ").strip().lower() not in ['n', 'no']:
                thermal_config = {
                    'enabled': True,
                    'coupling_strength': 'strong',
                    'ambient_temperature': self.get_float_input(
                        "Ambient temperature", default=293.15, min_val=200, max_val=400, unit="K"
                    ),
                    'convection_coefficient': self.get_float_input(
                        "Convection coefficient", default=10, min_val=1, max_val=100, unit="W/(m²⋅K)"
                    )
                }
                config['thermal'] = thermal_config
            else:
                config['thermal'] = {'enabled': False}
            
            # Energy conservation
            config['energy_conservation'] = {
                'enabled': True,
                'tolerance': self.get_float_input(
                    "Energy conservation tolerance", default=1e-8, min_val=1e-12, max_val=1e-3
                ),
                'check_interval': int(self.get_float_input(
                    "Energy check interval (steps)", default=100, min_val=1, max_val=1000
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
        
        choice = input(f"\nSelect method (1-{len(methods)}) [1]: ").strip() or "1"
        calculation_method = methods[int(choice) - 1]
        
        config = {
            'calculation_method': calculation_method,
            'axial_discretization': int(self.get_float_input(
                "Axial discretization points", 
                default=1000 if self.current_mode != 'Expert' else 5000,
                min_val=100, max_val=20000
            )),
            'radial_discretization': int(self.get_float_input(
                "Radial discretization points",
                default=100 if self.current_mode != 'Expert' else 500,
                min_val=10, max_val=2000
            ))
        }
        
        # Physics effects
        config.update({
            'include_saturation': input("Include magnetic saturation? [Y/n]: ").strip().lower() not in ['n', 'no'],
            'include_hysteresis': input("Include hysteresis effects? [y/N]: ").strip().lower() in ['y', 'yes'],
            'include_eddy_currents': input("Include eddy current effects? [Y/n]: ").strip().lower() not in ['n', 'no'],
            'include_skin_effect': input("Include skin effect? [Y/n]: ").strip().lower() not in ['n', 'no'],
        })
        
        # Force components
        print("\nForce calculation components:")
        force_components = {
            'reluctance_force': input("Include reluctance force (∇L method)? [Y/n]: ").strip().lower() not in ['n', 'no'],
            'lorentz_force': input("Include Lorentz force (J×B)? [Y/n]: ").strip().lower() not in ['n', 'no'],
            'maxwell_stress': input("Include Maxwell stress tensor? [Y/n]: ").strip().lower() not in ['n', 'no'],
            'image_force': input("Include magnetic image force? [Y/n]: ").strip().lower() not in ['n', 'no'],
            'eddy_force': config['include_eddy_currents'] and input("Include eddy current force? [Y/n]: ").strip().lower() not in ['n', 'no']
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
            
            choice = input(f"\nSelect solver method (1-{len(methods)}) [3]: ").strip() or "3"
            config['method'] = methods[int(choice) - 1]
            
            # Tolerances
            config['rtol'] = self.get_float_input(
                "Relative tolerance", default=1e-8, min_val=1e-12, max_val=1e-3
            )
            config['atol'] = self.get_float_input(
                "Absolute tolerance", default=1e-10, min_val=1e-15, max_val=1e-6
            )
            config['max_step'] = self.get_float_input(
                "Maximum step size", default=1e-4, min_val=1e-8, max_val=1e-2, unit="s"
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
            'time_span': [0, self.get_float_input("Simulation time", default=0.01, min_val=1e-6, max_val=1.0, unit="s")],
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
        
        # Select setup complexity level
        mode = setup_interface.select_setup_mode()
        
        # Create complete configuration
        config = setup_interface.create_complete_configuration()
        
        # Get filename
        suggested_name = f"coilgun_config_{mode.lower().replace('-', '_')}.json"
        filename = input(f"\nConfiguration filename [{suggested_name}]: ").strip() or suggested_name
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Validate and save
        if validate_and_save_configuration(config, filename):
            print_configuration_summary(config, mode)
            
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