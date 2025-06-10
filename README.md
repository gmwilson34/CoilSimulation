# Electromagnetic Coilgun Simulation

A comprehensive, physics-based simulation framework for electromagnetic coilgun design and analysis. This tool implements Maxwell's equations, advanced circuit modeling, and precise magnetic field calculations to help you design and analyze coilgun systems.

## Table of Contents
- [Overview](#overview)
- [Installation (Mac)](#installation-mac)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Examples](#examples)
- [Physics Background](#physics-background)
- [Troubleshooting](#troubleshooting)

---

## Overview

This simulation engine provides a complete framework for modeling electromagnetic coilgun systems with engineering precision. The simulation implements:

- **Maxwell's Equations** for magnetic field calculation using Biot-Savart Law
- **RLC Circuit Analysis** with position-dependent inductance 
- **Magnetic Force Calculation** via inductance gradient method
- **Energy Conservation** and efficiency analysis
- **Comprehensive Visualization** of magnetic fields and simulation results

### Key Features

- **Three-Step Workflow**: Setup → Solve → Visualize
- **Interactive Configuration**: Guided parameter selection with validation
- **CSV Data Export**: Detailed time-series data for analysis
- **Comprehensive Visualizations**: Magnetic field plots, force maps, and animations
- **Performance Analysis**: Efficiency calculations and energy balance
- **Material Database**: Real electromagnetic material properties

### Applications

- **Engineering Design**: Coilgun optimization and performance prediction
- **Research**: Electromagnetic system analysis and parameter studies  
- **Education**: Understanding electromagnetic principles and system behavior
- **Prototyping**: Virtual testing before physical implementation

---

## Installation (Mac)

### Prerequisites

- **macOS 10.14+** (Mojave or later)
- **Python 3.8+** (Python 3.9+ recommended)
- **Homebrew** (recommended for Python installation)

### Step 1: Install Python (if needed)

Using Homebrew (recommended):
```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python
```

Or download from [python.org](https://www.python.org/downloads/mac-osx/)

### Step 2: Clone/Download the Project

```bash
# If using git:
git clone <your-repository-url>
cd CoilSimulation

# Or download and extract the ZIP file, then:
cd CoilSimulation
```

### Step 3: Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Your terminal prompt should now show (.venv)
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install numpy scipy matplotlib

# Optional: Enhanced plotting aesthetics
pip install seaborn pandas
```

### Step 5: Verify Installation

```bash
# Test imports
python -c "import numpy, scipy, matplotlib; print('✅ All dependencies installed successfully!')"

# Test the physics engine
python -c "import equations; print('✅ Physics engine loaded successfully!')"
```

---

## Quick Start

The coilgun simulation follows a simple three-step workflow:

### 1. Configure Your System
```bash
python setup.py
```
This interactive script guides you through setting up your coilgun parameters.

### 2. Run the Simulation  
```bash
python solve.py your_config.json
```
This runs the electromagnetic simulation and saves results to CSV and JSON files.

### 3. View Results and Visualizations
```bash
python view.py your_config.json
```
This creates detailed magnetic field visualizations and animations.

### Complete Example
```bash
# Step 1: Create configuration
python setup.py
# Follow the prompts, save as "my_coilgun.json"

# Step 2: Run simulation
python solve.py my_coilgun.json

# Step 3: Create visualizations
python view.py my_coilgun.json
```

---

## Workflow

### Step 1: Configuration (`setup.py`)

Run the interactive setup script:
```bash
python setup.py
```

The script will guide you through configuring:

#### **Coil Parameters**
- Inner diameter (typically 10-50mm)
- Length (usually 2-5× diameter)  
- Wire gauge (AWG 14-20 for medium power)
- Number of layers
- Wire material (Copper recommended)
- Packing factor and insulation thickness

#### **Projectile Parameters**
- Diameter (80-95% of coil inner diameter)
- Length (2-4× diameter for good coupling)
- Material (Pure Iron, Low Carbon Steel, or Silicon Steel)
- Initial position (typically -50mm to start before coil)

#### **Capacitor Bank**
- Capacitance (1-10 mF typical)
- Initial voltage (200-500V typical)
- Equivalent series resistance (ESR)

#### **Simulation Settings**
- Time span (10-50ms typical)
- Integration method and precision
- Output options

The script validates your inputs and saves everything to a JSON configuration file.

### Step 2: Simulation (`solve.py`)

Run the electromagnetic simulation:
```bash
python solve.py config_file.json
```

**What it does:**
- Loads your configuration
- Calculates coil inductance and resistance
- Computes position-dependent magnetic coupling
- Solves the coupled electromagnetic-mechanical differential equations
- Analyzes performance and efficiency
- Saves detailed results

**Output:**
```
============================================================
COILGUN SIMULATION SOLVER
============================================================
Configuration file: my_coilgun.json

=== Coilgun System Parameters ===
Coil:
  Inner diameter: 15.0 mm
  Length: 75.0 mm
  Total turns: 280
  Wire: AWG 16 (1.291 mm)
  Resistance: 0.272 Ω
  Air-core inductance: 483.1 µH

Projectile:
  Material: Low_Carbon_Steel
  Dimensions: 12.0 mm × 25.0 mm
  Mass: 22.20 g
  Relative permeability: 1000

Capacitor:
  Capacitance: 3000 µF
  Initial voltage: 400 V
  Initial energy: 240.0 J

[Simulation runs...]

==================================================
SIMULATION SUMMARY
==================================================
Final velocity: 45.2 m/s
Efficiency: 12.3%
Max current: 832.9 A
Max force: 1327.7 N
Simulation time: 0.045 s
Exit reason: Projectile reached coil center

ENERGY ANALYSIS:
Initial capacitor energy: 240.0 J
Final kinetic energy: 22.7 J
Energy transferred to projectile: 22.7 J

Results saved to directory: results_my_coilgun/
- time_series_data.csv (detailed time series)
- simulation_summary.json (summary results)

To view detailed visualizations, run:
python view.py my_coilgun.json
```

### Step 3: Visualization (`view.py`)

Create comprehensive visualizations:
```bash
python view.py config_file.json
```

**What it creates:**
- **Magnetic field contour plots** at different current levels
- **3D surface plots** of magnetic field strength
- **On-axis field profiles** showing field uniformity
- **Animated field evolution** during projectile motion

**Output:**
```
============================================================
COILGUN VISUALIZATION SUITE
============================================================
Configuration file: my_coilgun.json

Creating visualizations in: visualizations_my_coilgun/

1. Creating static field analysis...
   Calculating field for 100A...
   Calculating field for 300A...
   Calculating field for 500A...

2. Creating on-axis field profiles...

3. Running simulation for field animation...

4. Creating field evolution animation...

==================================================
VISUALIZATION COMPLETE
==================================================
Files saved to: visualizations_my_coilgun/
- Magnetic field contour plots
- 3D field surface plots
- On-axis field profiles  
- Field evolution animation

Check the output directory for all visualization files!
```

---

## Configuration

### Example Configuration File

```json
{
    "coil": {
        "inner_diameter": 0.015,
        "length": 0.075,
        "wire_gauge_awg": 16,
        "num_layers": 6,
        "wire_material": "Copper",
        "insulation_thickness": 5e-05,
        "packing_factor": 0.85
    },
    "projectile": {
        "diameter": 0.012,
        "length": 0.025,
        "material": "Low_Carbon_Steel",
        "initial_position": -0.05,
        "initial_velocity": 0.0
    },
    "capacitor": {
        "capacitance": 0.003,
        "initial_voltage": 400,
        "esr": 0.01,
        "esl": 5e-08
    },
    "simulation": {
        "time_span": [0, 0.02],
        "max_step": 1e-06,
        "tolerance": 1e-09,
        "method": "RK45"
    }
}
```

### Available Materials

#### **Conductor Materials**
- **Copper**: Standard for coil winding (high conductivity)
- **Aluminum**: Lighter weight alternative

#### **Ferromagnetic Materials**  
- **Pure Iron**: μᵣ ≈ 5,000 (highest coupling)
- **Low Carbon Steel**: μᵣ ≈ 1,000 (common, good performance)  
- **Silicon Steel**: μᵣ ≈ 15,000 (electrical steel, very high coupling)

#### **Wire Gauges**
- AWG 10-36 supported with current capacity ratings
- Recommended: AWG 14-18 for medium power systems

---

## Output Files

### Simulation Results (`results_<config>/`)

#### **time_series_data.csv**
Detailed time-series data with columns:
- `time_s`: Time in seconds
- `charge_C`: Capacitor charge
- `current_A`: Coil current  
- `position_m`: Projectile position
- `velocity_ms`: Projectile velocity
- `force_N`: Magnetic force
- `inductance_H`: Instantaneous inductance
- `power_W`: Instantaneous power
- `energy_capacitor_J`: Capacitor energy
- `energy_kinetic_J`: Projectile kinetic energy

#### **simulation_summary.json**
Summary results including:
- Final velocity and efficiency
- Maximum current and force
- Energy analysis
- System parameters
- Performance metrics

### Visualization Files (`visualizations_<config>/`)

- **bfield_contours_<current>A.png**: Magnetic field contour plots
- **bfield_3d_<current>A.png**: 3D surface plots of magnetic field
- **onaxis_field_profiles.png**: On-axis field variation
- **field_evolution.gif**: Animated field evolution during simulation

---

## Examples

### Basic Design Study

```bash
# Create a medium-power coilgun configuration
python setup.py
# Configure: 15mm ID, 75mm length, AWG 16 wire, 6 layers
# Projectile: 12mm steel projectile  
# Capacitor: 3mF @ 400V
# Save as: medium_coilgun.json

# Run simulation
python solve.py medium_coilgun.json

# View results  
python view.py medium_coilgun.json
```

### Parameter Optimization

To study the effect of different voltages:

1. Create base configuration with `setup.py`
2. Make copies with different voltages:
```bash
cp my_config.json config_300V.json
cp my_config.json config_400V.json  
cp my_config.json config_500V.json
```
3. Edit each file to change `"initial_voltage"`
4. Run simulations:
```bash
python solve.py config_300V.json
python solve.py config_400V.json
python solve.py config_500V.json
```
5. Compare the results from each `simulation_summary.json`

### Analyzing CSV Data

You can analyze the detailed time-series data using any tool that reads CSV:

#### **Python/Pandas**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('results_my_coilgun/time_series_data.csv')

# Plot current vs time
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(data['time_s'] * 1000, data['current_A'])
plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.title('Coil Current')

# Plot velocity vs time  
plt.subplot(2, 2, 2)
plt.plot(data['time_s'] * 1000, data['velocity_ms'])
plt.xlabel('Time (ms)')
plt.ylabel('Velocity (m/s)')
plt.title('Projectile Velocity')

# Plot force vs position
plt.subplot(2, 2, 3)
plt.plot(data['position_m'] * 1000, data['force_N'])
plt.xlabel('Position (mm)')
plt.ylabel('Force (N)')
plt.title('Force vs Position')

# Plot energy vs time
plt.subplot(2, 2, 4)
plt.plot(data['time_s'] * 1000, data['energy_capacitor_J'], label='Capacitor')
plt.plot(data['time_s'] * 1000, data['energy_kinetic_J'], label='Kinetic')
plt.xlabel('Time (ms)')
plt.ylabel('Energy (J)')
plt.title('Energy Transfer')
plt.legend()

plt.tight_layout()
plt.show()
```

#### **Excel/Google Sheets**
1. Open the CSV file in Excel or Google Sheets
2. Create charts using the time-series columns
3. Calculate derived quantities like power = current × voltage

---

## Physics Background

### Overview of the Electromagnetic System

A coilgun operates by using electromagnetic forces to accelerate a ferromagnetic projectile. The system consists of:

1. **Capacitor Bank**: Stores electrical energy
2. **Coil**: Converts electrical energy to magnetic fields
3. **Ferromagnetic Projectile**: Experiences magnetic force
4. **Control Circuit**: Manages current flow and timing

The simulation solves the coupled electromagnetic-mechanical differential equations that govern this system.

---

### Core Differential Equations

The coilgun system is described by four coupled first-order differential equations:

#### **State Vector**
The system state is described by:
```
State = [Q, I, x, v]
```
Where:
- `Q` = Capacitor charge (C, Coulombs)
- `I` = Coil current (A, Amperes) 
- `x` = Projectile position (m, meters)
- `v` = Projectile velocity (m/s, meters per second)

#### **Differential Equations**
```
dQ/dt = -I
dI/dt = (V_C - I·R - I·v·(dL/dx)) / L(x)
dx/dt = v  
dv/dt = F / m
```

---

### 1. Capacitor Discharge Equation

#### **Charge Conservation**
```
dQ/dt = -I
```

**Physical Meaning**: The rate of change of capacitor charge equals the negative current (capacitor is discharging).

**Variables**:
- `Q` = Instantaneous charge on capacitor (C)
- `I` = Current flowing from capacitor (A)
- `t` = Time (s)

**Units**: [C/s] = [A]

#### **Capacitor Voltage**
```
V_C = Q/C
```

**Variables**:
- `V_C` = Capacitor voltage (V, Volts)
- `Q` = Capacitor charge (C)
- `C` = Capacitance (F, Farads)

**Physical Meaning**: Voltage across capacitor is proportional to stored charge.

---

### 2. Circuit Equation (Kirchhoff's Voltage Law)

#### **Complete Circuit Equation**
```
V_C - L(dI/dt) - I·R - I·v·(dL/dx) = 0
```

**Rearranged for simulation**:
```
dI/dt = (V_C - I·R - I·v·(dL/dx)) / L(x)
```

**Physical Meaning**: The sum of voltage drops around the circuit equals zero.

**Voltage Terms**:

1. **Capacitor Voltage**: `V_C = Q/C` (V)
   - Driving voltage that pushes current through circuit

2. **Inductive Voltage Drop**: `L(dI/dt)` (V)
   - Voltage across inductor opposes current changes
   - `L(x)` = Position-dependent inductance (H, Henries)
   - `dI/dt` = Rate of current change (A/s)

3. **Resistive Voltage Drop**: `I·R` (V)
   - Ohmic losses in wire resistance
   - `R` = Total circuit resistance (Ω, Ohms)

4. **Motional EMF (Back-EMF)**: `I·v·(dL/dx)` (V)
   - Voltage induced by moving magnetic system
   - `v` = Projectile velocity (m/s)
   - `dL/dx` = Inductance gradient (H/m)

**Units**: [V] - [V] - [V] - [V] = 0

---

### 3. Magnetic Force Calculation

#### **Force from Magnetic Energy**
```
F = (1/2) × I² × (dL/dx)
```

**Derivation**: Force is the negative gradient of magnetic energy:
```
U_magnetic = (1/2) × L(x) × I²
F = -dU/dx = -(1/2) × I² × (dL/dx)
```

Since `dL/dx > 0` when projectile enters coil, force is positive (accelerating).

**Variables**:
- `F` = Magnetic force on projectile (N, Newtons)
- `I` = Instantaneous current (A)
- `dL/dx` = Inductance gradient (H/m, Henries per meter)

**Physical Meaning**: Ferromagnetic projectile is pulled toward region of higher magnetic field (higher inductance).

**Units**: [A²] × [H/m] = [N]

---

### 4. Projectile Motion (Newton's Laws)

#### **Position Update**
```
dx/dt = v
```

**Variables**:
- `x` = Position along coil axis (m)
- `v` = Velocity (m/s)

#### **Velocity Update (Newton's Second Law)**
```
dv/dt = F/m
```

**Variables**:
- `v` = Projectile velocity (m/s)
- `F` = Net force on projectile (N)
- `m` = Projectile mass (kg)

**Units**: [N]/[kg] = [m/s²]

---

### Magnetic Field Calculations

#### **Biot-Savart Law for Circular Current Loop**

For a circular loop of radius `R` carrying current `I`, the magnetic field on-axis at distance `z` is:

```
B_z(z) = (μ₀ × I × R²) / (2 × (R² + z²)^(3/2))
```

**Variables**:
- `B_z` = Axial magnetic field component (T, Tesla)
- `μ₀` = Permeability of free space = 4π × 10⁻⁷ (H/m)
- `I` = Current in loop (A)
- `R` = Loop radius (m)
- `z` = Axial distance from loop center (m)

**Physical Meaning**: Field strength decreases as cube of distance from small loops.

#### **Solenoid Field (Superposition)**

For a solenoid with `N` total turns over length `L`:

```
B_total(z) = Σ B_loop_i(z)
```

Where each loop contributes according to Biot-Savart law.

**Current per loop**:
```
I_loop = I_total × (N_total / N_loops_calculated)
```

---

### Inductance Calculations

#### **Air-Core Solenoid Inductance (Wheeler's Formula)**

```
L₀ = (μ₀ × N² × A) / (l + 0.9 × r)
```

**Variables**:
- `L₀` = Air-core inductance (H)
- `N` = Total number of turns (dimensionless)
- `A` = Cross-sectional area = π × r² (m²)
- `l` = Coil length (m)
- `r` = Average coil radius (m)

**Physical Meaning**: Inductance depends on coil geometry and turns squared.

#### **Position-Dependent Inductance with Ferromagnetic Core**

```
L(x) = L₀ × μ_effective(x)
```

**Effective Permeability**:
```
μ_effective(x) = 1 + (μᵣ - 1) × f_coupling(x) × f_fill
```

**Coupling Factor** (position-dependent):
```
f_coupling(x) = exp(-|x_center - x_coil_center| / L_char)
```

**Fill Factor** (geometric):
```
f_fill = (r_projectile / r_coil_inner)²
```

**Variables**:
- `μᵣ` = Relative permeability of projectile material (dimensionless)
- `f_coupling(x)` = Position-dependent coupling (0 to 1)
- `f_fill` = Radial fill factor (0 to 1)
- `L_char` = Characteristic length scale (m)

---

### Energy Analysis

#### **Initial Energy Storage**
```
E_initial = (1/2) × C × V₀²
```

**Variables**:
- `E_initial` = Initial capacitor energy (J, Joules)
- `C` = Capacitance (F)
- `V₀` = Initial voltage (V)

#### **Instantaneous Energy Distribution**

**Capacitor Energy**:
```
E_capacitor(t) = (1/2) × C × V_C(t)² = Q(t)² / (2×C)
```

**Magnetic Energy**:
```
E_magnetic(t) = (1/2) × L(x(t)) × I(t)²
```

**Kinetic Energy**:
```
E_kinetic(t) = (1/2) × m × v(t)²
```

**Energy Dissipated** (cumulative):
```
E_dissipated(t) = ∫₀ᵗ I(τ)² × R dτ
```

#### **Energy Conservation Check**:
```
E_initial = E_capacitor(t) + E_magnetic(t) + E_kinetic(t) + E_dissipated(t)
```

#### **Efficiency Calculation**
```
η = E_kinetic_final / E_initial × 100%
```

Where:
- `η` = Efficiency (%)
- `E_kinetic_final` = Final projectile kinetic energy (J)

---

### Power Analysis

#### **Instantaneous Power**

**Power delivered by capacitor**:
```
P_capacitor = V_C × I = (Q/C) × I
```

**Power dissipated in resistance**:
```
P_resistive = I² × R
```

**Power transferred to magnetic field**:
```
P_magnetic = I × (dI/dt) × L + I² × v × (dL/dx)
```

**Power transferred to projectile**:
```
P_mechanical = F × v = (1/2) × I² × (dL/dx) × v
```

**Units**: All power terms in Watts [W] = [J/s]

---

### Material Properties and Physical Constants

#### **Fundamental Constants**
- `μ₀` = 4π × 10⁻⁷ H/m (Permeability of free space)
- `ε₀` = 8.854 × 10⁻¹² F/m (Permittivity of free space)

#### **Material Properties**

**Copper (Wire)**:
- Resistivity: `ρ = 1.68 × 10⁻⁸ Ω·m`
- Temperature coefficient: `α = 0.00393 K⁻¹`
- Density: `8960 kg/m³`

**Low Carbon Steel (Projectile)**:
- Relative permeability: `μᵣ ≈ 1000`
- Resistivity: `ρ = 1.43 × 10⁻⁷ Ω·m`
- Density: `7850 kg/m³`

#### **Resistance Calculation**
```
R = ρ × L_wire / A_wire × (1 + α × ΔT)
```

**Variables**:
- `R` = Total resistance (Ω)
- `ρ` = Material resistivity (Ω·m)
- `L_wire` = Total wire length (m)
- `A_wire` = Wire cross-sectional area (m²)
- `α` = Temperature coefficient (K⁻¹)
- `ΔT` = Temperature rise above 20°C (K)

---

### Coordinate System and Conventions

#### **Position Reference**
- `x = 0`: Front face of coil (entrance)
- `x < 0`: Projectile before coil
- `x = L_coil`: Back face of coil (exit)
- `x = L_coil/2`: Center of coil

#### **Current Direction**
- Positive current creates magnetic field that attracts ferromagnetic projectile
- Current flows in coil windings to create axial magnetic field

#### **Force Direction**
- Positive force accelerates projectile in +x direction (into/through coil)
- Force magnitude depends on current squared and inductance gradient

---

### Numerical Integration Details

#### **State Space Form**
The system is solved as:
```
d/dt [Q, I, x, v] = f(t, [Q, I, x, v])
```

Where:
```
f(t, [Q, I, x, v]) = [
    -I,
    (Q/C - I×R - I×v×(dL/dx)) / L(x),
    v,
    (1/2)×I²×(dL/dx) / m
]
```

#### **Integration Method**
- **Default**: Runge-Kutta 4th/5th order (RK45)
- **Adaptive step size**: Maintains specified tolerance
- **Event detection**: Stops at coil center or current reversal

#### **Typical Integration Parameters**
- Time span: 0 to 20-50 ms
- Relative tolerance: 1×10⁻⁹
- Maximum step size: 1×10⁻⁶ s

---

### Performance Metrics and Analysis

#### **Key Performance Indicators**

**Velocity and Energy**:
- Final muzzle velocity (m/s)
- Final kinetic energy (J)
- Energy conversion efficiency (%)

**Current and Force**:
- Peak current (A)
- Peak magnetic force (N)
- Average acceleration (m/s²)

**Timing**:
- Time to peak current (ms)
- Time to reach coil center (ms)
- Acceleration duration (ms)

#### **Optimization Parameters**

**Design Variables**:
- Coil dimensions (inner diameter, length)
- Wire gauge and number of layers
- Capacitor bank (capacitance, voltage)
- Projectile material and geometry

**Performance Trade-offs**:
- More turns → Higher inductance but higher resistance
- Higher voltage → More energy but harder to switch
- Larger diameter → Better coupling but more material

This comprehensive physics foundation enables accurate modeling and optimization of electromagnetic coilgun systems.

---

## Troubleshooting

### Common Issues

#### **"Integration failed" Error**
- **Cause**: Stiff differential equation (usually high capacitance + low voltage)
- **Solution**: Use more reasonable parameters (capacitance < 10mF, voltage > 200V)

#### **"Module not found" Error**  
- **Cause**: Missing Python dependencies
- **Solution**: 
```bash
pip install numpy scipy matplotlib
```

#### **Very Low Efficiency (<5%)**
- **Cause**: High resistance or poor magnetic coupling
- **Solution**: 
  - Use thicker wire (lower AWG number)
  - Reduce projectile-to-coil gap
  - Use higher permeability projectile material

#### **Projectile Doesn't Move**
- **Cause**: Insufficient energy or poor configuration
- **Solution**:
  - Increase capacitor voltage
  - Check projectile is ferromagnetic material
  - Verify projectile starts before coil (negative position)

#### **Visualization Fails**
- **Cause**: Missing matplotlib or display issues
- **Solution**:
```bash
pip install matplotlib
# For macOS display issues:
export MPLBACKEND=TkAgg
```

### Getting Help

1. **Check Configuration**: Verify all parameters are physically reasonable
2. **Review Output**: Look at the simulation summary for clues
3. **Start Simple**: Use default parameters first, then modify gradually
4. **Check Units**: All inputs should be in SI units (meters, seconds, etc.)

### Typical Parameter Ranges

| Parameter | Typical Range | Units |
|-----------|---------------|--------|
| Coil inner diameter | 10-50 | mm |
| Coil length | 30-200 | mm |  
| Wire gauge | 14-20 | AWG |
| Projectile diameter | 8-45 | mm |
| Capacitance | 1-10 | mF |
| Voltage | 200-500 | V |
| Expected velocity | 20-150 | m/s |
| Efficiency | 5-25 | % |

---

## License

This simulation framework is provided for educational and research purposes.

---

*Electromagnetic Coilgun Simulation - Physics-based coilgun design and analysis tool* 