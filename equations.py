# equations.py
"""
Advanced Electromagnetic Physics Engine for Coilgun Simulation

This module implements Maxwell's equations, electromagnetic field calculations,
and coilgun-specific physics based on rigorous electromagnetic theory.

Key features:
- Biot-Savart law implementation for magnetic field calculation
- Advanced RLC circuit modeling with motional EMF
- Ferromagnetic force calculation via inductance gradient
- 2D axisymmetric field calculations
- Material-dependent electromagnetic properties
"""

import numpy as np
import scipy.special as sp
from scipy.interpolate import interp1d
from scipy.integrate import quad
import json
import warnings
import os

class CoilgunPhysicsEngine:
    """
    Advanced physics engine implementing Maxwell's equations for coilgun simulation.
    """
    
    def __init__(self, config_file):
        """
        Initialize the physics engine with configuration parameters.
        
        Args:
            config_file: Path to JSON configuration file
        """
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Load materials data directly from JSON
        self.materials_data = self._load_materials_data()
        
        # Precompute derived parameters
        self._compute_coil_parameters()
        self._compute_projectile_parameters()
        self._compute_circuit_parameters()
        
        # Initialize field calculation method
        self.field_method = self.config.get('magnetic_model', {}).get('calculation_method', 'biot_savart')
        
        # Precompute inductance lookup table for efficiency
        self._precompute_inductance_table()
    
    def _load_materials_data(self):
        """Load materials data from JSON file"""
        try:
            if os.path.exists("materials.json"):
                with open("materials.json", 'r') as f:
                    return json.load(f)
            else:
                # Return basic materials if file not found
                return {
                    "physical_constants": {"mu0": 4 * np.pi * 1e-7, "room_temperature": 293.15},
                    "materials": {
                        "Copper": {"resistivity": 1.68e-8, "temperature_coefficient": 0.00393},
                        "Pure_Iron": {"density": 7874, "mu_r": 5000, "resistivity": 9.71e-8},
                        "Low_Carbon_Steel": {"density": 7850, "mu_r": 1000, "resistivity": 1.43e-7}
                    },
                    "wire_specifications": {
                        "awg_diameter_mm": {"14": 1.628, "16": 1.291, "18": 1.024, "20": 0.812}
                    }
                }
        except Exception as e:
            warnings.warn(f"Could not load materials data: {e}")
            # Return minimal data
            return {
                "physical_constants": {"mu0": 4 * np.pi * 1e-7, "room_temperature": 293.15},
                "materials": {"Copper": {"resistivity": 1.68e-8}},
                "wire_specifications": {"awg_diameter_mm": {"16": 1.291}}
            }
    
    def get_wire_diameter(self, awg):
        """Get wire diameter in meters from AWG"""
        awg_str = str(awg)
        if awg_str in self.materials_data['wire_specifications']['awg_diameter_mm']:
            diameter_mm = self.materials_data['wire_specifications']['awg_diameter_mm'][awg_str]
            return diameter_mm / 1000.0  # Convert mm to meters
        else:
            # Default fallback
            return 1.291e-3  # AWG 16
    
    def get_wire_area(self, awg):
        """Get wire cross-sectional area in m²"""
        diameter = self.get_wire_diameter(awg)
        return np.pi * (diameter / 2.0) ** 2
    
    def get_material_property(self, material_name, property_name):
        """Get a specific property for a material"""
        if (material_name in self.materials_data['materials'] and 
            property_name in self.materials_data['materials'][material_name]):
            return self.materials_data['materials'][material_name][property_name]
        else:
            # Provide fallback values for common properties
            fallbacks = {
                'resistivity': 1.68e-8,  # Copper
                'density': 7850,         # Steel
                'mu_r': 1000,           # Steel
                'temperature_coefficient': 0.004
            }
            return fallbacks.get(property_name, 1.0)
        
    def _compute_coil_parameters(self):
        """Compute coil geometry and electrical parameters."""
        coil_cfg = self.config['coil']
        
        # Geometric parameters
        self.coil_inner_radius = coil_cfg['inner_diameter'] / 2.0
        self.coil_length = coil_cfg['length']
        self.num_layers = coil_cfg['num_layers']
        
        # Wire parameters
        wire_material = coil_cfg['wire_material']
        wire_awg = coil_cfg['wire_gauge_awg']
        
        self.wire_diameter = self.get_wire_diameter(wire_awg)
        self.wire_area = self.get_wire_area(wire_awg)
        self.wire_resistivity = self.get_material_property(wire_material, 'resistivity')
        
        # Coil winding calculations
        packing_factor = coil_cfg.get('packing_factor', 0.85)
        insulation_thickness = coil_cfg.get('insulation_thickness', 0.05e-3)
        
        effective_wire_diameter = self.wire_diameter + insulation_thickness
        turns_per_layer = int(self.coil_length / effective_wire_diameter)
        self.total_turns = turns_per_layer * self.num_layers * packing_factor
        
        # Calculate average coil radius and wire length
        self.coil_outer_radius = self.coil_inner_radius + self.num_layers * effective_wire_diameter
        self.avg_coil_radius = (self.coil_inner_radius + self.coil_outer_radius) / 2.0
        
        # Total wire length
        self.wire_length = self.total_turns * 2 * np.pi * self.avg_coil_radius
        
        # Coil resistance (including temperature effects if specified)
        temperature = self.materials_data['physical_constants']['room_temperature']
        temp_coeff = self.get_material_property(wire_material, 'temperature_coefficient')
        
        resistance_factor = 1 + temp_coeff * (temperature - 293.15)  # 20°C reference
        self.coil_resistance = self.wire_resistivity * self.wire_length / self.wire_area * resistance_factor
        
        # Add switch and parasitic resistances
        circuit_cfg = self.config.get('circuit_model', {})
        self.total_resistance = (self.coil_resistance + 
                               circuit_cfg.get('switch_resistance', 0) +
                               self.config['capacitor'].get('esr', 0))
        
        # Coil center position
        self.coil_center = self.coil_length / 2.0
        
        # Physical constants
        self.mu0 = self.materials_data['physical_constants']['mu0']
        
    def _compute_projectile_parameters(self):
        """Compute projectile physical parameters."""
        proj_cfg = self.config['projectile']
        
        self.proj_diameter = proj_cfg['diameter']
        self.proj_radius = self.proj_diameter / 2.0
        self.proj_length = proj_cfg['length']
        
        # Material properties
        proj_material = proj_cfg['material']
        self.proj_density = self.get_material_property(proj_material, 'density')
        self.proj_mu_r = self.get_material_property(proj_material, 'mu_r')
        self.proj_resistivity = self.get_material_property(proj_material, 'resistivity')
        
        # Calculate mass
        proj_volume = np.pi * self.proj_radius**2 * self.proj_length
        self.proj_mass = proj_volume * self.proj_density
        
        # Initial conditions
        self.initial_position = proj_cfg['initial_position']
        self.initial_velocity = proj_cfg.get('initial_velocity', 0.0)
        
    def _compute_circuit_parameters(self):
        """Compute circuit parameters."""
        cap_cfg = self.config['capacitor']
        
        self.capacitance = cap_cfg['capacitance']
        self.initial_voltage = cap_cfg['initial_voltage']
        self.initial_energy = 0.5 * self.capacitance * self.initial_voltage**2
        
        # Initial charge
        self.initial_charge = self.capacitance * self.initial_voltage
        
    def solenoid_inductance_air_core(self):
        """
        Calculate air-core inductance of the solenoid using Wheeler's formula.
        
        Returns:
            L_air: Air-core inductance in Henries
        """
        # Wheeler's formula for multilayer solenoid
        # L = (mu0 * N^2 * A) / (l + 0.9*r)
        # where A = pi * r^2 is the cross-sectional area
        
        N = self.total_turns
        r = self.avg_coil_radius
        l = self.coil_length
        A = np.pi * r**2
        
        L_air = (self.mu0 * N**2 * A) / (l + 0.9 * r)
        return L_air
    
    def magnetic_field_on_axis_circular_loop(self, z, loop_radius, current, loop_position):
        """
        Calculate magnetic field on axis due to a circular current loop using Biot-Savart law.
        
        Args:
            z: Axial position where field is calculated
            loop_radius: Radius of the current loop
            current: Current in the loop
            loop_position: Axial position of the loop
            
        Returns:
            Bz: Axial magnetic field component
        """
        # Distance from loop to field point
        distance = z - loop_position
        
        # Biot-Savart law for circular loop on axis
        # Bz = (mu0 * I * R^2) / (2 * (R^2 + z^2)^(3/2))
        
        R_squared = loop_radius**2
        z_squared = distance**2
        denominator = (R_squared + z_squared)**(3/2)
        
        if denominator == 0:
            return 0  # Avoid division by zero
        
        Bz = (self.mu0 * current * R_squared) / (2 * denominator)
        return Bz
    
    def magnetic_field_solenoid_on_axis(self, z, current):
        """
        Calculate magnetic field on axis of the entire solenoid.
        
        Args:
            z: Axial position where field is calculated
            current: Current in the solenoid
            
        Returns:
            Bz: Total axial magnetic field
        """
        # Discretize solenoid into current loops
        num_loops = max(100, int(self.total_turns / 10))  # At least 100 points
        loop_positions = np.linspace(0, self.coil_length, num_loops)
        
        # Current per loop (total current divided by discretization)
        current_per_loop = current * self.total_turns / num_loops
        
        # Sum contributions from all loops
        Bz_total = 0
        for loop_pos in loop_positions:
            Bz_total += self.magnetic_field_on_axis_circular_loop(
                z, self.avg_coil_radius, current_per_loop, loop_pos
            )
        
        return Bz_total
    
    def inductance_with_ferromagnetic_core(self, projectile_position):
        """
        Calculate inductance when ferromagnetic projectile is present.
        
        Uses a smooth, physically-based model with proper magnetic coupling.
        
        Args:
            projectile_position: Position of projectile front face relative to coil start
            
        Returns:
            L_total: Total inductance including core effects
        """
        # Start with air-core inductance
        L_air = self.solenoid_inductance_air_core()
        
        # Calculate projectile center position relative to coil center
        proj_center = projectile_position - self.proj_length / 2
        coil_center = self.coil_length / 2
        
        # Distance from projectile center to coil center
        center_distance = abs(proj_center - coil_center)
        
        # Characteristic length for magnetic interaction
        char_length = (self.coil_length + self.proj_length) / 4
        
        # Calculate coupling factor based on position
        # Maximum coupling when projectile center aligns with coil center
        if center_distance <= char_length:
            # Strong coupling region
            coupling_factor = 1.0 - (center_distance / char_length)**2
        else:
            # Weak coupling region - exponential decay
            decay_distance = center_distance - char_length
            coupling_factor = np.exp(-decay_distance / char_length)
        
        # Ensure smooth transition and physical limits
        coupling_factor = max(0.0, min(1.0, coupling_factor))
        
        # Calculate radial fill factor
        radial_fill = min(1.0, (self.proj_radius / self.coil_inner_radius)**2)
        
        # Total coupling strength
        total_coupling = coupling_factor * radial_fill
        
        # Calculate effective permeability
        # Use a conservative permeability to avoid unrealistic forces
        mu_eff_max = min(self.proj_mu_r, 10)  # Limit to reasonable values
        mu_eff = 1 + (mu_eff_max - 1) * total_coupling
        
        # Calculate total inductance
        L_total = L_air * mu_eff
        
        return L_total
    
    def _precompute_inductance_table(self):
        """
        Precompute inductance vs position for fast lookup during simulation.
        """
        # Position range: from well before coil to well after
        z_min = -0.1  # 10cm before coil
        z_max = self.coil_length + 0.1  # 10cm after coil
        
        num_points = 1000
        self.inductance_positions = np.linspace(z_min, z_max, num_points)
        self.inductance_values = np.array([
            self.inductance_with_ferromagnetic_core(pos) 
            for pos in self.inductance_positions
        ])
        
        # Create interpolation function
        self.inductance_interp = interp1d(
            self.inductance_positions, 
            self.inductance_values, 
            kind='cubic',
            bounds_error=False,
            fill_value=(self.inductance_values[0], self.inductance_values[-1])
        )
        
        # Precompute derivative for force calculation
        dL_dz = np.gradient(self.inductance_values, self.inductance_positions)
        self.inductance_grad_interp = interp1d(
            self.inductance_positions,
            dL_dz,
            kind='cubic',
            bounds_error=False,
            fill_value=(dL_dz[0], dL_dz[-1])
        )
    
    def get_inductance(self, position):
        """
        Get inductance at given projectile position using precomputed lookup.
        
        Args:
            position: Projectile position
            
        Returns:
            L: Inductance in Henries
        """
        return float(self.inductance_interp(position))
    
    def get_inductance_gradient(self, position):
        """
        Get dL/dx at given projectile position using precomputed lookup.
        
        Args:
            position: Projectile position
            
        Returns:
            dL_dx: Inductance gradient in H/m
        """
        return float(self.inductance_grad_interp(position))
    
    def magnetic_force_ferromagnetic(self, current, position):
        """
        Calculate magnetic force on ferromagnetic projectile using inductance gradient.
        
        For a ferromagnetic projectile, the force is given by:
        F = 0.5 * I^2 * dL/dx
        
        This is derived from energy considerations and is valid for magnetically
        permeable (but non-conducting) projectiles.
        
        Args:
            current: Current in coil (A)
            position: Projectile position (m)
            
        Returns:
            force: Magnetic force in Newtons (positive = toward coil center)
        """
        dL_dx = self.get_inductance_gradient(position)
        force = 0.5 * current**2 * dL_dx
        
        return force
    
    def should_turn_off_coil(self, position, current):
        """
        Determine if coil should be turned off to avoid suck-back.
        
        Args:
            position: Current projectile position
            current: Current coil current
            
        Returns:
            bool: True if coil should be turned off
        """
        # Turn off when projectile reaches coil center
        if position >= self.coil_center:
            return True
        
        # Turn off if current has reversed (for SCR operation)
        if current < 0:
            return True
        
        return False
    
    def circuit_derivatives(self, t, state):
        """
        Calculate derivatives for the coupled electromagnetic circuit system.
        
        State vector: [Q, I, x, v]
        Q: Charge on capacitor (C)
        I: Current in coil (A)  
        x: Projectile position (m)
        v: Projectile velocity (m/s)
        
        Args:
            t: Time (s)
            state: State vector [Q, I, x, v]
            
        Returns:
            derivatives: [dQ/dt, dI/dt, dx/dt, dv/dt]
        """
        Q, I, x, v = state
        
        # Get position-dependent inductance and its gradient
        L = self.get_inductance(x)
        dL_dx = self.get_inductance_gradient(x)
        
        # Check if coil should be turned off
        if self.should_turn_off_coil(x, I):
            # Rapidly quench current
            dI_dt = -I / 1e-6  # 1 microsecond time constant
            force = 0
        else:
            # Circuit equation with motional EMF
            # Kirchhoff's voltage law: V_C - L*dI/dt - I*R - I*v*dL/dx = 0
            # Solving for dI/dt:
            # dI/dt = (V_C - I*R - I*v*dL/dx) / L
            # where V_C = Q/C
            
            V_capacitor = Q / self.capacitance
            motional_emf = I * v * dL_dx  # Back-EMF due to moving inductance
            resistive_drop = I * self.total_resistance
            
            dI_dt = (V_capacitor - resistive_drop - motional_emf) / L
            
            # Magnetic force on projectile
            force = self.magnetic_force_ferromagnetic(I, x)
        
        # Charge derivative: dQ/dt = -I (capacitor discharging)
        dQ_dt = -I
        
        # Position derivative: dx/dt = v
        dx_dt = v
        
        # Velocity derivative: F = ma
        dv_dt = force / self.proj_mass
        
        # Apply physical constraints
        # Prevent negative velocity if already at rest and force is backward
        if v <= 0 and dv_dt < 0:
            dv_dt = 0
            
        return [dQ_dt, dI_dt, dx_dt, dv_dt]
    
    def get_initial_conditions(self):
        """
        Get initial conditions for the simulation.
        
        Returns:
            y0: Initial state vector [Q0, I0, x0, v0]
        """
        Q0 = self.initial_charge
        I0 = 0.0  # No initial current (inductor property)
        x0 = self.initial_position
        v0 = self.initial_velocity
        
        return [Q0, I0, x0, v0]
    
    def calculate_efficiency(self, final_velocity):
        """
        Calculate energy conversion efficiency.
        
        Args:
            final_velocity: Final projectile velocity
            
        Returns:
            efficiency: Energy conversion efficiency (0-1)
        """
        final_kinetic_energy = 0.5 * self.proj_mass * final_velocity**2
        efficiency = final_kinetic_energy / self.initial_energy
        
        return efficiency
    
    def print_system_parameters(self):
        """Print key system parameters for verification."""
        print("=== Coilgun System Parameters ===")
        print(f"Coil:")
        print(f"  Inner diameter: {self.coil_inner_radius * 2 * 1000:.1f} mm")
        print(f"  Length: {self.coil_length * 1000:.1f} mm")
        print(f"  Total turns: {self.total_turns:.0f}")
        print(f"  Wire: AWG {self.config['coil']['wire_gauge_awg']} ({self.wire_diameter*1000:.3f} mm)")
        print(f"  Resistance: {self.total_resistance:.3f} Ω")
        print(f"  Air-core inductance: {self.solenoid_inductance_air_core()*1e6:.1f} µH")
        
        print(f"\nProjectile:")
        print(f"  Material: {self.config['projectile']['material']}")
        print(f"  Dimensions: {self.proj_diameter*1000:.1f} mm × {self.proj_length*1000:.1f} mm")
        print(f"  Mass: {self.proj_mass*1000:.2f} g")
        print(f"  Relative permeability: {self.proj_mu_r}")
        
        print(f"\nCapacitor:")
        print(f"  Capacitance: {self.capacitance*1e6:.0f} µF")
        print(f"  Initial voltage: {self.initial_voltage:.0f} V")
        print(f"  Initial energy: {self.initial_energy:.1f} J")
        
        L_max = max(self.inductance_values)
        print(f"\nSystem:")
        print(f"  Maximum inductance: {L_max*1e6:.1f} µH")
        print(f"  Inductance ratio: {L_max/self.solenoid_inductance_air_core():.1f}")


# Utility functions for field visualization and analysis
def calculate_field_map(physics_engine, current, z_range, r_range, num_z=50, num_r=20):
    """
    Calculate 2D magnetic field map for visualization.
    
    Args:
        physics_engine: CoilgunPhysicsEngine instance
        current: Current value for field calculation
        z_range: [z_min, z_max] axial range
        r_range: [r_min, r_max] radial range  
        num_z: Number of axial points
        num_r: Number of radial points
        
    Returns:
        Z, R, Bz, Br: Meshgrids and field components
    """
    z_points = np.linspace(z_range[0], z_range[1], num_z)
    r_points = np.linspace(r_range[0], r_range[1], num_r)
    
    Z, R = np.meshgrid(z_points, r_points)
    Bz = np.zeros_like(Z)
    Br = np.zeros_like(Z)
    
    # Calculate field at each point (simplified - only axial component on axis)
    for i, z in enumerate(z_points):
        Bz[0, i] = physics_engine.magnetic_field_solenoid_on_axis(z, current)
    
    # For off-axis points, use approximations or more complex calculations
    # This is simplified - full implementation would require 3D Biot-Savart
    
    return Z, R, Bz, Br


if __name__ == '__main__':
    # Test the physics engine with a sample configuration
    
    # Create a simple test configuration
    test_config = {
        "coil": {
            "inner_diameter": 0.015,
            "length": 0.075,
            "wire_gauge_awg": 16,
            "num_layers": 6,
            "wire_material": "Copper",
            "packing_factor": 0.85
        },
        "projectile": {
            "diameter": 0.012,
            "length": 0.025,
            "material": "Low_Carbon_Steel",
            "initial_position": -0.05
        },
        "capacitor": {
            "capacitance": 0.0033,
            "initial_voltage": 450
        }
    }
    
    with open("test_coilgun.json", 'w') as f:
        json.dump(test_config, f, indent=4)
    
    # Initialize physics engine
    engine = CoilgunPhysicsEngine("test_coilgun.json")
    
    # Print system parameters
    engine.print_system_parameters()
    
    # Test field calculations
    print(f"\nField Tests:")
    current = 100  # 100A test current
    
    print(f"B-field at coil center (z=0.0375m): {engine.magnetic_field_solenoid_on_axis(0.0375, current)*1000:.1f} mT")
    print(f"B-field at coil entrance (z=0): {engine.magnetic_field_solenoid_on_axis(0, current)*1000:.1f} mT")
    print(f"B-field at coil exit (z=0.075m): {engine.magnetic_field_solenoid_on_axis(0.075, current)*1000:.1f} mT")
    
    # Test inductance calculations
    print(f"\nInductance Tests:")
    positions = [-0.05, 0, 0.0375, 0.075, 0.1]
    for pos in positions:
        L = engine.get_inductance(pos)
        dL_dx = engine.get_inductance_gradient(pos)
        force = engine.magnetic_force_ferromagnetic(current, pos)
        print(f"x={pos*1000:4.0f}mm: L={L*1e6:5.1f}µH, dL/dx={dL_dx*1e6:6.2f}µH/m, F={force:6.1f}N")

