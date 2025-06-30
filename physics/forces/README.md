# Forces Package - Electromagnetic Force Calculations

This package contains the refactored electromagnetic force calculation system for the coilgun simulation. The monolithic force scripts have been broken down into specialized, focused modules for better maintainability and readability.

## Package Structure

### Core Modules

- **`base.py`** - Base electromagnetic forces class with common functionality
- **`balanced.py`** - Balanced forces implementation with energy conservation
- **`advanced.py`** - Advanced forces combining all specialized calculators

### Specialized Force Calculators

- **`quantum.py`** - Quantum mechanical force corrections (Casimir, zero-point energy, tunneling)
- **`maxwell_stress.py`** - Maxwell stress tensor calculations for accurate force computation
- **`eddy_currents.py`** - Eddy current forces with skin depth and frequency effects
- **`hysteresis.py`** - Magnetic hysteresis modeling using Jiles-Atherton model
- **`relativistic.py`** - Relativistic corrections for high-velocity projectiles
- **`multiscale.py`** - Multi-scale coupling from quantum to classical regimes

### Analysis and Diagnostics

- **`analyzer.py`** - Force analysis, validation, and diagnostics

## Usage

### Basic Usage

```python
from physics.forces import create_electromagnetic_forces

# Create a balanced force calculator (recommended for most applications)
forces = create_electromagnetic_forces(config, field_calc, materials, 'balanced')

# Calculate forces
force, power = forces.magnetic_force_ferromagnetic(current, position, velocity)
```

### Advanced Usage

```python
from physics.forces import create_electromagnetic_forces, create_force_analyzer

# Create advanced force calculator with all corrections
forces = create_electromagnetic_forces(config, field_calc, materials, 'advanced')

# Calculate total force with detailed breakdown
total_force, breakdown = forces.calculate_total_electromagnetic_force_quantum_enhanced(
    current, position, velocity, acceleration, current_history, time_history, time
)

# Analyze force components
analyzer = create_force_analyzer(forces)
analysis = analyzer.analyze_force_components(current, position, velocity)
```

### Individual Force Components

```python
from physics.forces.quantum import QuantumForceCalculator
from physics.forces.eddy_currents import EddyCurrentForces

# Use individual specialized calculators
quantum_calc = QuantumForceCalculator(config, field_calc, materials)
quantum_force, breakdown = quantum_calc.calculate_quantum_force_corrections(
    current, position, velocity
)

eddy_calc = EddyCurrentForces(config, field_calc, materials)
eddy_force, power = eddy_calc.calculate_eddy_current_force(
    current, position, velocity, current_history, time_history
)
```

## Force Calculator Types

### Balanced Forces (`ElectromagneticForcesBalanced`)
- Realistic force calculations with energy conservation
- Moderate permeability limits and inductance enhancement
- Recommended for practical coilgun simulations
- Based on the previous `forces_final_fix.py`

### Advanced Forces (`AdvancedElectromagneticForces`)
- Comprehensive implementation with all available corrections
- Quantum mechanical effects
- Relativistic corrections
- Multi-scale coupling
- High-precision numerical methods
- Suitable for research and extreme-performance applications

## Key Improvements

1. **Modularity**: Each force mechanism is now in its own focused module
2. **Maintainability**: Easier to understand, modify, and extend individual components
3. **Reusability**: Individual calculators can be used independently
4. **Testing**: Smaller modules are easier to unit test
5. **Documentation**: Each module has clear, focused documentation
6. **Performance**: Selective loading of only needed components

## Backward Compatibility

The main `physics/forces.py` module provides factory functions and imports to maintain backward compatibility with existing code:

```python
# Legacy usage still works
from physics.forces import AdvancedElectromagneticForces, ElectromagneticForcesBalanced
```

## Configuration

Force calculators are configured through the main configuration dictionary:

```python
config = {
    'advanced_physics': {
        'include_eddy_currents': True,
        'include_hysteresis': False,
        'include_relativistic': False,
        'use_maxwell_stress': False,
        'integration_order': 8
    },
    'quantum_physics': {
        'enable_quantum_forces': False,
        'casimir_force': False,
        'quantum_tunneling': False,
        'zero_point_energy': False
    }
}
```

## Migration from Old System

If you were using the old monolithic force classes:

1. Replace direct class instantiation with factory functions
2. Use `create_electromagnetic_forces()` with appropriate type
3. Existing method calls remain the same
4. Add force analysis with `create_force_analyzer()` if desired

The refactored system provides the same functionality while being much more organized and maintainable. 