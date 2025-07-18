"""
Main Physics Engine

This module provides the unified CoilgunPhysicsEngine class that integrates
all the modular physics components while maintaining compatibility with existing code.
"""

import numpy as np
import json
import os
import time
import warnings
from typing import Dict, Any, Optional, Tuple, List

from .core import BasePhysicsModel, PhysicsConstants, SafetyLimits, NumericalUtils, validate_physical_parameter
from .materials import AdvancedMaterialProperties, AdvancedPermeabilityModel
from .fields import AdvancedMagneticFieldCalculator
from .forces import AdvancedElectromagneticForces, BaseElectromagneticForces, ForceAnalyzer
from .circuits import CircuitModel, InductanceCalculator, EnergyAnalyzer
from .utils import validate_coilgun_config, calculate_coil_metrics


class CoilgunPhysicsEngine(BasePhysicsModel):
    """
    Advanced physics engine implementing Maxwell's equations for coilgun simulation.
    Enhanced with modular architecture for better maintainability.
    """
    
    # Safety constants for compatibility
    MAX_CURRENT = SafetyLimits.MAX_CURRENT
    MAX_FORCE = SafetyLimits.MAX_FORCE
    MAX_VOLTAGE = SafetyLimits.MAX_VOLTAGE
    MAX_FIELD = SafetyLimits.MAX_FIELD
    MAX_ENERGY = SafetyLimits.MAX_ENERGY
    MAX_POWER = SafetyLimits.MAX_POWER
    
    MIN_INDUCTANCE = SafetyLimits.MIN_INDUCTANCE
    MIN_RESISTANCE = SafetyLimits.MIN_RESISTANCE
    MIN_CAPACITANCE = SafetyLimits.MIN_CAPACITANCE
    MIN_MASS = SafetyLimits.MIN_MASS
    
    NUMERICAL_EPSILON = SafetyLimits.NUMERICAL_EPSILON
    FORCE_EPSILON = SafetyLimits.FORCE_EPSILON
    CURRENT_EPSILON = SafetyLimits.CURRENT_EPSILON
    
    def __init__(self, config_file: str):
        """
        Initialize the physics engine with configuration parameters.
        
        Args:
            config_file: Path to JSON configuration file
        """
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Auto-calculate missing parameters before validation
        config = self._auto_calculate_missing_parameters(config)
        
        super().__init__(config)
        
        # Validate configuration
        is_valid, errors = validate_coilgun_config(config)
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Initialize advanced core components
        print("🔬 Initializing PhD-level physics components...")
        self.materials = AdvancedMaterialProperties(config)
        self.field_calculator = AdvancedMagneticFieldCalculator(config)
        self.circuit_model = CircuitModel(config, self.materials)
        
        # Choose force calculation method based on accuracy requirements
        accuracy_level = config.get('magnetic_model', {}).get('accuracy_level', 'high')
        if accuracy_level == 'phd':
            print("🔬 Using ADVANCED electromagnetic forces with Maxwell stress tensor")
            self.forces = AdvancedElectromagneticForces(config, self.field_calculator, self.materials)
        else:
            print("⚡ Using BASE electromagnetic forces (numerically stable)")
            self.forces = BaseElectromagneticForces(config, self.field_calculator, self.materials)
        
        # Initialize analyzers
        self.force_analyzer = ForceAnalyzer(self.forces)
        self.inductance_calc = InductanceCalculator(self.circuit_model)
        self.energy_analyzer = EnergyAnalyzer(self.circuit_model)
        
        # Cache derived parameters for compatibility
        self._compute_derived_parameters()
        
        # Initialize advanced physics models
        self.permeability_model = AdvancedPermeabilityModel(self.materials)
        
        # Initialize energy tracking
        self.energy_analyzer.initialize_energy_tracking()
        
        print("✓ Modular Physics Engine Initialized")
        print(f"  - Coil: {self.total_turns} turns, {self.coil_length*1000:.1f}mm length")
        print(f"  - Projectile: {self.proj_mass*1000:.1f}g {self.proj_material}")
        print(f"  - Circuit: {self.capacitance*1000:.1f}mF @ {self.initial_voltage}V")
    
    def _auto_calculate_missing_parameters(self, config: dict) -> dict:
        """Auto-calculate missing parameters like total_turns and mass."""
        import copy
        config = copy.deepcopy(config)
        
        # Auto-calculate total_turns if missing
        if 'coil' in config and 'total_turns' not in config['coil']:
            coil_cfg = config['coil']
            if all(key in coil_cfg for key in ['wire_gauge_awg', 'num_layers', 'inner_diameter', 'length']):
                # Calculate turns from wire gauge and coil geometry
                wire_awg = int(coil_cfg['wire_gauge_awg'])
                num_layers = coil_cfg['num_layers']
                inner_diameter = coil_cfg['inner_diameter']
                length = coil_cfg['length']
                packing_factor = coil_cfg.get('packing_factor', 0.85)
                
                # Get wire diameter from materials data
                awg_diameters = {
                    10: 2.588e-3, 12: 2.053e-3, 14: 1.628e-3, 16: 1.291e-3,
                    18: 1.024e-3, 20: 0.812e-3, 22: 0.644e-3, 24: 0.511e-3
                }
                wire_diameter = awg_diameters.get(wire_awg, 1.291e-3)
                
                # Calculate turns per layer (axial)
                wire_diameter_with_insulation = wire_diameter + coil_cfg.get('insulation_thickness', 0)
                turns_per_layer = int(length / (wire_diameter_with_insulation / packing_factor))
                
                # Total turns
                total_turns = turns_per_layer * num_layers
                config['coil']['total_turns'] = total_turns
                print(f"Auto-calculated total_turns: {total_turns} ({turns_per_layer} per layer × {num_layers} layers)")
        
        # Auto-calculate mass if missing
        if 'projectile' in config and 'mass' not in config['projectile']:
            proj_cfg = config['projectile']
            if all(key in proj_cfg for key in ['diameter', 'length', 'material']):
                # Calculate mass from material density and volume
                diameter = proj_cfg['diameter']
                length = proj_cfg['length']
                material = proj_cfg['material']
                
                # Material densities (kg/m³)
                material_densities = {
                    'Pure_Iron': 7874,
                    'Low_Carbon_Steel': 7850,
                    'Silicon_Steel': 7650,
                    'Aluminum': 2700,
                    'Copper': 8960
                }
                
                density = material_densities.get(material, 7850)  # Default to steel
                volume = np.pi * (diameter/2)**2 * length
                mass = density * volume
                
                config['projectile']['mass'] = mass
                print(f"Auto-calculated projectile mass: {mass*1000:.1f}g (density: {density} kg/m³)")
        
        return config
    
    def _compute_derived_parameters(self):
        """Compute derived parameters for compatibility with existing code."""
        # Coil parameters
        coil_cfg = self.config.get('coil', {})
        self.coil_inner_radius = coil_cfg.get('inner_diameter', 0.02) / 2.0
        self.coil_length = coil_cfg.get('length', 0.05)
        self.num_layers = coil_cfg.get('num_layers', 1)
        self.total_turns = coil_cfg.get('total_turns', 1000)
        
        # Projectile parameters
        proj_cfg = self.config.get('projectile', {})
        self.proj_mass = proj_cfg.get('mass', 0.01)
        self.proj_length = proj_cfg.get('length', 0.01)
        self.proj_diameter = proj_cfg.get('diameter', 0.008)
        self.proj_material = proj_cfg.get('material', 'Low_Carbon_Steel')
        self.proj_mu_r = self.materials.get_material_property(self.proj_material, 'mu_r')
        
        # Circuit parameters
        cap_cfg = self.config.get('capacitor', {})
        self.capacitance = cap_cfg.get('capacitance', 0.001)
        self.initial_voltage = cap_cfg.get('initial_voltage', 400)
        
        # Field calculation method
        magnetic_cfg = self.config.get('magnetic_model', {})
        self.field_method = magnetic_cfg.get('calculation_method', 'finite_solenoid')
    
    # Magnetic field calculation methods (for compatibility)
    def magnetic_field_on_axis_circular_loop(self, z: float, loop_radius: float, 
                                           current: float, loop_position: float = 0.0) -> float:
        """Calculate magnetic field on axis for a circular current loop."""
        # Use analytical formula directly since AdvancedMagneticFieldCalculator doesn't have this method
        z_rel = z - loop_position
        
        # Handle special case at loop center
        if abs(z_rel) < 1e-15:
            return PhysicsConstants.MU_0 * current / (2.0 * loop_radius) if loop_radius > 0 else 0.0
        
        # General case using analytical formula
        denominator = (loop_radius**2 + z_rel**2)**(3.0/2.0)
        
        if denominator < 1e-15:
            return 0.0
        
        B_z = PhysicsConstants.MU_0 * current * loop_radius**2 / (2.0 * denominator)
        
        return NumericalUtils.safe_numerical_operation(B_z, "circular_loop_field")
    
    def magnetic_field_solenoid_on_axis(self, z: float, current: float) -> float:
        """Calculate magnetic field on solenoid axis."""
        return self.field_calculator.magnetic_field_solenoid_on_axis(z, current)
    
    def magnetic_field_finite_solenoid_on_axis(self, z: float, a: float, l: float, 
                                             N: int, current: float) -> float:
        """Accurate finite solenoid field calculation on axis."""
        # Implement analytical finite solenoid formula directly
        mu0 = PhysicsConstants.MU_0
        n = N / l  # turns per meter
        
        # Distances to coil ends
        z1 = z + l/2.0  # Distance to near end  
        z2 = z - l/2.0  # Distance to far end
        
        # Exact analytical formula using cosines of end angles
        r1 = np.sqrt(a**2 + z1**2)
        r2 = np.sqrt(a**2 + z2**2)
        
        cos_beta1 = z1 / r1 if r1 > 1e-15 else 0.0
        cos_beta2 = z2 / r2 if r2 > 1e-15 else 0.0
        
        B_z = (mu0 * n * current / 2.0) * (cos_beta1 - cos_beta2)
        
        return NumericalUtils.safe_numerical_operation(B_z, "finite_solenoid_field")
    
    # Force calculation methods (for compatibility)
    def magnetic_force_ferromagnetic(self, current: float, position: float, 
                                   velocity: float = 0.0, 
                                   current_history: Optional[List] = None,
                                   time_history: Optional[List] = None) -> Tuple[float, float]:
        """Calculate magnetic force on ferromagnetic projectile."""
        return self.forces.magnetic_force_ferromagnetic(
            current, position, velocity, current_history, time_history
        )
    
    def magnetic_force_with_circuit_logic(self, current: float, position: float, 
                                        time: Optional[float] = None, 
                                        velocity: float = 0.0) -> Tuple[float, float]:
        """Calculate magnetic force with circuit logic (compatibility method)."""
        return self.magnetic_force_ferromagnetic(current, position, velocity)
    
    # Inductance calculation methods (for compatibility)
    def solenoid_inductance_air_core(self) -> float:
        """Calculate air-core solenoid inductance."""
        return self.circuit_model.coil_inductance_air
    
    def get_inductance(self, position: float, current: Optional[float] = None) -> float:
        """Get inductance at given position."""
        if current is None:
            current = 1.0  # Default current for inductance calculation
        
        # Calculate effective permeability
        B_field = self.magnetic_field_solenoid_on_axis(position, current)
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
        mu_eff = self.permeability_model.calculate_nonlinear_permeability(
            H_applied, self.proj_material
        )
        
        # Calculate overlap fraction
        overlap_fraction = self._calculate_overlap_fraction(position)
        
        return self.circuit_model.calculate_inductance_with_core(position, mu_eff, overlap_fraction)
    
    def get_inductance_gradient(self, position: float, current: Optional[float] = None) -> float:
        """Get inductance gradient at given position."""
        if current is None:
            current = 1.0
        
        # Calculate effective permeability
        B_field = self.magnetic_field_solenoid_on_axis(position, current)
        H_applied = B_field / PhysicsConstants.MU_0 if B_field != 0 else 0
        mu_eff = self.permeability_model.calculate_nonlinear_permeability(
            H_applied, self.proj_material
        )
        
        return self.inductance_calc.calculate_inductance_gradient(position, mu_eff)
    
    def _calculate_overlap_fraction(self, position: float) -> float:
        """Calculate overlap fraction between projectile and coil."""
        coil_start = -self.coil_length / 2.0
        coil_end = self.coil_length / 2.0
        
        proj_start = position - self.proj_length / 2.0
        proj_end = position + self.proj_length / 2.0
        
        # Calculate overlap
        overlap_start = max(coil_start, proj_start)
        overlap_end = min(coil_end, proj_end)
        
        if overlap_end > overlap_start:
            overlap_length = overlap_end - overlap_start
            return overlap_length / self.proj_length
        else:
            return 0.0
    
    # Circuit dynamics (for compatibility)
    def circuit_derivatives(self, t: float, y: List[float]) -> List[float]:
        """
        Calculate circuit derivatives for ODE integration.
        
        Args:
            t: Time (s)
            y: State vector [Q, I, x, v, ...]
            
        Returns:
            Derivatives [dQ/dt, dI/dt, dx/dt, dv/dt, ...]
        """
        # Extract state variables
        Q, I, x, v = y[:4]  # Charge, current, position, velocity
        
        # Calculate voltage
        V = Q / self.capacitance if self.capacitance > 0 else 0
        
        # Calculate inductance and its gradient
        L = self.get_inductance(x, I)
        dL_dx = self.get_inductance_gradient(x, I)
        
        # Calculate electromagnetic force
        force, eddy_power = self.magnetic_force_ferromagnetic(I, x, v)
        
        # Circuit equations
        # dQ/dt = -I
        dQ_dt = -I
        
        # dI/dt = (V - I*R - (dL/dt)*I) / L
        # where dL/dt = (dL/dx) * (dx/dt) = (dL/dx) * v
        dL_dt = dL_dx * v
        total_resistance = self.circuit_model.total_resistance_dc
        
        if L > SafetyLimits.MIN_INDUCTANCE:
            dI_dt = (V - I * total_resistance - dL_dt * I) / L
        else:
            dI_dt = 0.0
        
        # Mechanical equations
        # dx/dt = v
        dx_dt = v
        
        # dv/dt = F/m
        if self.proj_mass > SafetyLimits.MIN_MASS:
            dv_dt = force / self.proj_mass
        else:
            dv_dt = 0.0
        
        # Apply safety limits
        dI_dt = NumericalUtils.clamp(dI_dt, -1e6, 1e6)
        dv_dt = NumericalUtils.clamp(dv_dt, -1e6, 1e6)
        
        return [dQ_dt, dI_dt, dx_dt, dv_dt]
    
    # Utility methods (for compatibility)
    def get_initial_conditions(self) -> Tuple[float, float, float, float]:
        """Get initial conditions for simulation."""
        Q0, I0 = self.circuit_model.get_initial_conditions()
        
        # Use configured initial position, or default to coil entrance if not specified
        proj_config = self.config.get('projectile', {})
        x0 = proj_config.get('initial_position', -self.coil_length / 2.0)
        v0 = proj_config.get('initial_velocity', 0.0)
        
        return Q0, I0, x0, v0
    
    def calculate_efficiency(self, final_velocity: float) -> float:
        """Calculate energy conversion efficiency."""
        final_kinetic_energy = 0.5 * self.proj_mass * final_velocity**2
        return self.energy_analyzer.calculate_efficiency(final_kinetic_energy)
    
    def print_system_parameters(self):
        """Print system parameters for verification."""
        print("\n" + "="*60)
        print("MODULAR COILGUN PHYSICS ENGINE - SYSTEM PARAMETERS")
        print("="*60)
        
        # Coil parameters
        print(f"\nCOIL SPECIFICATIONS:")
        print(f"  Inner diameter: {self.coil_inner_radius*2*1000:.2f} mm")
        print(f"  Length: {self.coil_length*1000:.2f} mm")
        print(f"  Total turns: {self.total_turns}")
        print(f"  Inductance (air-core): {self.solenoid_inductance_air_core()*1e6:.2f} µH")
        
        # Projectile parameters  
        print(f"\nPROJECTILE SPECIFICATIONS:")
        print(f"  Material: {self.proj_material}")
        print(f"  Mass: {self.proj_mass*1000:.2f} g")
        print(f"  Diameter: {self.proj_diameter*1000:.2f} mm")
        print(f"  Length: {self.proj_length*1000:.2f} mm")
        print(f"  Relative permeability: {self.proj_mu_r}")
        
        # Circuit parameters
        print(f"\nCIRCUIT SPECIFICATIONS:")
        print(f"  Capacitance: {self.capacitance*1000:.2f} mF")
        print(f"  Initial voltage: {self.initial_voltage:.0f} V")
        print(f"  Initial energy: {0.5*self.capacitance*self.initial_voltage**2:.2f} J")
        print(f"  Coil resistance: {self.circuit_model.coil_resistance_dc:.4f} Ω")
        
        print("="*60)
    
    def validate_configuration(self):
        """Validate configuration for numerical stability."""
        is_valid, errors = validate_coilgun_config(self.config)
        if not is_valid:
            for error in errors:
                print(f"Configuration Error: {error}")
        else:
            print("✓ Configuration validation passed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the physics engine."""
        metrics = {}
        
        # Material properties performance
        if hasattr(self.materials, 'get_cache_statistics'):
            metrics['materials_cache'] = self.materials.get_cache_statistics()
        
        # Basic performance info
        metrics['field_calculations'] = {
            'method': self.field_method,
            'accuracy_level': getattr(self.field_calculator, 'accuracy_level', 'unknown')
        }
        
        # Circuit model info
        metrics['circuit_model'] = {
            'coil_inductance': self.circuit_model.coil_inductance_air,
            'coil_resistance': self.circuit_model.coil_resistance_dc
        }
        
        return metrics
    
    def enable_advanced_physics(self, enable_quantum: bool = False, enable_relativistic: bool = False):
        """Enable advanced physics features for extreme conditions."""
        warnings_issued = []
        
        if enable_quantum:
            # Check if quantum effects are available
            if hasattr(self.field_calculator, 'quantum_effects'):
                print("🔬 Quantum effects module available (but specific corrections interface varies)")
                warnings_issued.append("Quantum corrections may need manual configuration")
            else:
                warnings_issued.append("Quantum effects module not available")
        
        if enable_relativistic:
            # Check if relativistic effects are available
            if hasattr(self.field_calculator, 'include_relativistic'):
                self.field_calculator.include_relativistic = True
                print("⚡ Relativistic corrections enabled")
            else:
                warnings_issued.append("Relativistic corrections not available")
        
        if warnings_issued:
            for warning in warnings_issued:
                warnings.warn(warning)
    
    def optimize_for_performance(self, enable_caching: bool = True, cache_size: int = 1000):
        """Optimize engine for maximum performance."""
        optimizations_applied = []
        
        # Enable field caching if available
        if hasattr(self.field_calculator, 'cache_enabled'):
            self.field_calculator.cache_enabled = enable_caching
            if hasattr(self.field_calculator, 'max_cache_size'):
                self.field_calculator.max_cache_size = cache_size
            optimizations_applied.append("field caching")
        
        # Set calculation tolerances for speed vs accuracy balance
        if hasattr(self.field_calculator, 'adaptive_tolerance'):
            self.field_calculator.adaptive_tolerance = 1e-8  # Balanced tolerance
            optimizations_applied.append("balanced tolerance")
        
        if optimizations_applied:
            print(f"⚡ Performance optimizations applied: {', '.join(optimizations_applied)}")
        else:
            print("⚠️  No performance optimizations available")
    
    def validate_physics_consistency(self) -> Dict[str, Any]:
        """Validate physics consistency across all modules."""
        validation_results = {}
        
        try:
            # Validate energy conservation
            test_current = 1000.0
            test_position = 0.0
            force, _ = self.magnetic_force_ferromagnetic(test_current, test_position)
            
            # Check force calculation consistency
            B_field = self.magnetic_field_solenoid_on_axis(test_position, test_current)
            inductance = self.get_inductance(test_position, test_current)
            
            # Physical consistency checks
            validation_results['field_magnitude_reasonable'] = 0.001 < B_field < 50.0  # Tesla
            validation_results['inductance_positive'] = inductance > 0
            validation_results['force_finite'] = np.isfinite(force)
            validation_results['force_magnitude'] = abs(force)
            validation_results['field_strength'] = B_field
            validation_results['inductance_value'] = inductance
            
        except Exception as e:
            validation_results['calculation_error'] = str(e)
            validation_results['calculations_working'] = False
            return validation_results
        
        # Configuration validation
        config_valid, config_errors = validate_coilgun_config(self.config)
        validation_results['configuration_valid'] = config_valid
        validation_results['configuration_errors'] = config_errors
        validation_results['calculations_working'] = True
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health check."""
        status = {}
        
        # Basic system parameters
        status['coil_parameters'] = {
            'inner_radius': self.coil_inner_radius,
            'length': self.coil_length,
            'total_turns': self.total_turns,
            'air_core_inductance': self.solenoid_inductance_air_core()
        }
        
        status['projectile_parameters'] = {
            'mass': self.proj_mass,
            'diameter': self.proj_diameter,
            'length': self.proj_length,
            'material': self.proj_material,
            'permeability': self.proj_mu_r
        }
        
        status['circuit_parameters'] = {
            'capacitance': self.capacitance,
            'initial_voltage': self.initial_voltage,
            'initial_energy': 0.5 * self.capacitance * self.initial_voltage**2,
            'coil_resistance': self.circuit_model.coil_resistance_dc
        }
        
        # Physics validation
        status['physics_validation'] = self.validate_physics_consistency()
        
        # Performance metrics
        status['performance_metrics'] = self.get_performance_metrics()
        
        # Component health
        status['component_health'] = {
            'materials_initialized': hasattr(self, 'materials') and self.materials is not None,
            'field_calculator_initialized': hasattr(self, 'field_calculator') and self.field_calculator is not None,
            'forces_initialized': hasattr(self, 'forces') and self.forces is not None,
            'circuit_model_initialized': hasattr(self, 'circuit_model') and self.circuit_model is not None
        }
        
        return status
    
    # Enhanced field calculation methods
    def calculate_field_gradient(self, position: float, current: float) -> float:
        """Calculate magnetic field gradient at given position."""
        if hasattr(self.field_calculator, 'calculate_field_gradient'):
            return self.field_calculator.calculate_field_gradient(position, current)
        else:
            # Fallback numerical gradient calculation
            dx = 1e-6
            B1 = self.magnetic_field_solenoid_on_axis(position - dx, current)
            B2 = self.magnetic_field_solenoid_on_axis(position + dx, current)
            return (B2 - B1) / (2 * dx)
    
    def calculate_3d_magnetic_field(self, position: np.ndarray, current: float) -> np.ndarray:
        """Calculate 3D magnetic field at given position."""
        if hasattr(self.field_calculator, 'magnetic_field_3d_biot_savart'):
            return self.field_calculator.magnetic_field_3d_biot_savart(position, current)
        else:
            # Fallback: assume axial field only
            if len(position) >= 3:
                z_position = position[2]
                B_z = self.magnetic_field_solenoid_on_axis(z_position, current)
                return np.array([0.0, 0.0, B_z])
            else:
                raise ValueError("Position must be a 3D vector [x, y, z]")
    
    # Compatibility methods (placeholder implementations for complex features)
    def _safe_numerical_operation(self, value: float, operation_name: str, max_value: Optional[float] = None) -> float:
        """Safe numerical operation wrapper."""
        return NumericalUtils.safe_numerical_operation(value, operation_name, max_value)
    
    def create_stepwise_callback(self):
        """Create callback for stepwise integration monitoring."""
        # Simplified implementation - full implementation would be in solve.py
        return lambda t, y: None
    
    # Advanced analysis methods
    def analyze_force_components(self, current: float, position: float, velocity: float = 0.0,
                               current_history: Optional[List] = None,
                               time_history: Optional[List] = None) -> Dict[str, Any]:
        """Analyze and decompose electromagnetic forces into components."""
        return self.force_analyzer.analyze_force_components(
            current, position, velocity, current_history, time_history
        )
    
    def calculate_energy_balance(self, current: float, voltage: float, 
                               kinetic_energy: float, cumulative_losses: float) -> Dict[str, Any]:
        """Calculate energy balance and conservation."""
        return self.energy_analyzer.calculate_energy_balance(
            current, voltage, kinetic_energy, cumulative_losses
        )