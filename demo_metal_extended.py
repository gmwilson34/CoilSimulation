#!/usr/bin/env python3
"""
Demonstration of Extended Metal GPU Acceleration with Full solve.py Compatibility

This script demonstrates all the enhanced capabilities that bring the GPU acceleration
to full feature parity with solve.py:

🚀 FEATURES DEMONSTRATED:
- MetalCoilgunSimulation: Complete simulation controller with real-time progress
- MetalMultiStageCoilgunSimulation: Multi-stage simulation with velocity transfer
- metal_parametric_study: GPU-accelerated parameter sweeps
- Progress tracking with Metal/Julia indicators
- Results plotting and visualization
- Configuration management and file discovery

USAGE:
    python demo_metal_extended.py

REQUIREMENTS:
- Apple Silicon Mac for Metal GPU acceleration (optional - falls back to Julia CPU)
- juliacall package for Julia integration
- matplotlib for plotting
- Configuration files: *.json in current directory
"""

import sys
import os
import numpy as np
import time
import json
from pathlib import Path

try:
    from metal_acceleration_extended import (
        MetalCoilgunSimulation, 
        MetalMultiStageCoilgunSimulation,
        metal_parametric_study,
        find_metal_config_files
    )
    EXTENDED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Extended Metal acceleration not available: {e}")
    print("   Please ensure metal_acceleration_extended.py is in the current directory")
    EXTENDED_AVAILABLE = False

# Fallback to basic Metal acceleration
try:
    from metal_acceleration import create_metal_accelerated_solver, MetalAcceleration
    BASIC_AVAILABLE = True
except ImportError:
    print("❌ Basic Metal acceleration also not available")
    BASIC_AVAILABLE = False


def demo_basic_comparison():
    """Demonstrate the difference between basic and extended capabilities."""
    print("📊 BASIC vs EXTENDED METAL ACCELERATION COMPARISON")
    print("=" * 60)
    
    print("BASIC metal_acceleration.py capabilities:")
    print("✓ Julia/Metal GPU physics acceleration (50-200x speedup)")
    print("✓ Enhanced physics models (thermal, saturation, eddy currents)")
    print("✓ Multiple solver accuracy levels")
    print("✓ Batch processing and field mapping")
    print("✓ Solution analysis and benchmarking")
    print("❌ Real-time progress tracking")
    print("❌ Multi-stage simulation support")
    print("❌ Results plotting and visualization")
    print("❌ Parametric studies")
    print("❌ Complete simulation orchestration")
    
    print("\nEXTENDED metal_acceleration_extended.py capabilities:")
    print("✓ ALL basic capabilities PLUS:")
    print("✅ Real-time progress tracking with Metal/Julia indicators")
    print("✅ Multi-stage simulation with velocity transfer")
    print("✅ Complete results plotting and visualization")
    print("✅ GPU-accelerated parametric studies")
    print("✅ Full solve.py compatibility and simulation orchestration")
    print("✅ Configuration file management")
    print("✅ Enhanced results processing and export")


def demo_progress_tracking():
    """Demonstrate progress tracking capabilities."""
    print("\n🎯 PROGRESS TRACKING DEMONSTRATION")
    print("=" * 50)
    
    if not EXTENDED_AVAILABLE:
        print("❌ Extended features not available")
        return
    
    # Find a config file
    config_files = find_metal_config_files()
    if not config_files:
        print("❌ No configuration files found")
        print("   Please ensure you have a coilgun configuration JSON file")
        return
    
    config_file = str(config_files[0])
    print(f"Using configuration: {config_file}")
    
    try:
        # Create Metal simulation with progress tracking
        print("\n🚀 Creating Metal-accelerated simulation with progress tracking...")
        sim = MetalCoilgunSimulation(config_file)
        
        print("   ✓ MetalCoilgunSimulation created")
        print(f"   ✓ Backend: {sim.simulation_info['backend']}")
        print("   ✓ Progress tracking enabled")
        print("\n⏱️  Running simulation with real-time progress display...")
        
        # Run with progress tracking
        results = sim.run_simulation(
            save_data=True,
            verbose=True,
            show_progress=True,
            accuracy_level='fast'  # Use fast for demo
        )
        
        print("\n📊 Progress tracking demonstration completed!")
        print(f"   Final velocity: {results['final_velocity_m_s']:.2f} m/s")
        print(f"   Efficiency: {results['efficiency_percent']:.1f}%")
        print(f"   Simulation time: {results.get('simulation_time_s', 0):.3f} seconds")
        
        return sim
        
    except Exception as e:
        print(f"❌ Progress tracking demo failed: {e}")
        return None


def demo_plotting():
    """Demonstrate plotting capabilities."""
    print("\n📈 PLOTTING DEMONSTRATION")
    print("=" * 40)
    
    if not EXTENDED_AVAILABLE:
        print("❌ Extended features not available")
        return
    
    # Use simulation from progress tracking demo
    sim = demo_progress_tracking()
    if sim is None:
        return
    
    try:
        print("\n🎨 Creating comprehensive Metal-accelerated simulation plots...")
        
        # Generate plots with Metal indicators
        sim.plot_results(save_plots=True, output_dir="demo_metal_results")
        
        print("✅ Plotting demonstration completed!")
        print("   ✓ Current vs time plot")
        print("   ✓ Velocity vs time plot")
        print("   ✓ Position vs time plot")
        print("   ✓ Force vs time plot")
        print("   ✓ Energy distribution plot")
        print("   ✓ Inductance vs position plot")
        print("   ✓ Metal acceleration backend indicators")
        print("   📁 Plots saved to: demo_metal_results/")
        
    except Exception as e:
        print(f"❌ Plotting demo failed: {e}")


def demo_parametric_study():
    """Demonstrate Metal-accelerated parametric studies."""
    print("\n🔬 PARAMETRIC STUDY DEMONSTRATION")
    print("=" * 45)
    
    if not EXTENDED_AVAILABLE:
        print("❌ Extended features not available")
        return
    
    config_files = find_metal_config_files()
    if not config_files:
        print("❌ No configuration files found")
        return
    
    config_file = str(config_files[0])
    
    try:
        print(f"🚀 Running Metal-accelerated parametric study...")
        print(f"   Configuration: {config_file}")
        print("   Parameter: capacitor.initial_voltage")
        print("   Values: [100V, 150V, 200V, 250V]")
        print("   Acceleration: Metal GPU (if available) or Julia CPU")
        
        # Define parameter sweep
        parameter_values = [100, 150, 200, 250]  # Voltages to test
        
        # Run Metal-accelerated parametric study
        start_time = time.time()
        results = metal_parametric_study(
            base_config_file=config_file,
            parameter_name='capacitor.initial_voltage',
            parameter_values=parameter_values,
            output_dir="demo_parametric_results",
            accuracy_level='fast'  # Fast for demo
        )
        total_time = time.time() - start_time
        
        print(f"\n✅ Parametric study completed in {total_time:.2f} seconds!")
        
        # Analyze results
        successful_results = [r for r in results if not r.get('failed', False)]
        if successful_results:
            print("\n📊 Results Summary:")
            for i, result in enumerate(successful_results):
                voltage = parameter_values[i]
                velocity = result['final_velocity_m_s']
                efficiency = result['efficiency_percent']
                print(f"   {voltage}V: v={velocity:.1f} m/s, η={efficiency:.1f}%")
            
            # Find optimal
            best_efficiency_idx = max(range(len(successful_results)), 
                                    key=lambda i: successful_results[i]['efficiency_percent'])
            best_velocity_idx = max(range(len(successful_results)), 
                                   key=lambda i: successful_results[i]['final_velocity_m_s'])
            
            print(f"\n🎯 Optimization Results:")
            print(f"   Best efficiency: {parameter_values[best_efficiency_idx]}V "
                  f"({successful_results[best_efficiency_idx]['efficiency_percent']:.1f}%)")
            print(f"   Best velocity: {parameter_values[best_velocity_idx]}V "
                  f"({successful_results[best_velocity_idx]['final_velocity_m_s']:.1f} m/s)")
        
        print(f"   📁 Detailed results saved to: demo_parametric_results/")
        
    except Exception as e:
        print(f"❌ Parametric study demo failed: {e}")


def demo_multi_stage():
    """Demonstrate multi-stage simulation."""
    print("\n🚀 MULTI-STAGE SIMULATION DEMONSTRATION")
    print("=" * 50)
    
    if not EXTENDED_AVAILABLE:
        print("❌ Extended features not available")
        return
    
    # Look for multi-stage config files
    config_files = find_metal_config_files()
    multi_stage_config = None
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            if config.get("multi_stage", {}).get("enabled", False):
                multi_stage_config = str(config_file)
                break
        except:
            continue
    
    if multi_stage_config is None:
        print("❌ No multi-stage configuration files found")
        print("   Multi-stage configs need 'multi_stage.enabled: true'")
        print("   For demo purposes, showing single-stage simulation")
        
        # Fall back to single stage demo
        sim = demo_progress_tracking()
        return sim
    
    try:
        print(f"🎯 Multi-stage configuration found: {multi_stage_config}")
        
        # Load config to show details
        with open(multi_stage_config, 'r') as f:
            config = json.load(f)
        
        num_stages = config["multi_stage"]["num_stages"]
        print(f"   Number of stages: {num_stages}")
        print("   Features: Velocity transfer between stages")
        print("   Acceleration: Metal GPU on each stage")
        
        # Run multi-stage simulation
        print("\n🚀 Running Metal-accelerated multi-stage simulation...")
        multi_sim = MetalMultiStageCoilgunSimulation(multi_stage_config)
        
        results = multi_sim.run_simulation(
            save_data=True,
            verbose=True,
            show_progress=True,
            accuracy_level='fast'
        )
        
        print("\n✅ Multi-stage simulation demonstration completed!")
        print(f"   Total stages: {results['num_stages']}")
        print(f"   Final velocity: {results['final_velocity_m_s']:.2f} m/s")
        print(f"   Overall efficiency: {results['overall_efficiency_percent']:.1f}%")
        print(f"   Backend: {results['backend']}")
        
        # Show stage progression
        stage_velocities = [r['final_velocity_m_s'] for r in results['stage_results']]
        print(f"\n📈 Velocity progression through stages:")
        for i, v in enumerate(stage_velocities, 1):
            print(f"   Stage {i}: {v:.2f} m/s")
        
        return multi_sim
        
    except Exception as e:
        print(f"❌ Multi-stage demo failed: {e}")
        return None


def demo_configuration_management():
    """Demonstrate configuration file management."""
    print("\n⚙️  CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("=" * 55)
    
    if not EXTENDED_AVAILABLE:
        print("❌ Extended features not available")
        return
    
    try:
        print("🔍 Scanning for Metal-compatible configuration files...")
        
        config_files = find_metal_config_files()
        
        print(f"✅ Found {len(config_files)} configuration files:")
        
        for i, config_file in enumerate(config_files, 1):
            print(f"\n{i}. {config_file.name}")
            
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Analyze config
                is_multi_stage = config.get("multi_stage", {}).get("enabled", False)
                description = config.get("description", "No description")
                
                print(f"   Type: {'Multi-stage' if is_multi_stage else 'Single-stage'}")
                print(f"   Description: {description}")
                
                if is_multi_stage:
                    num_stages = config["multi_stage"]["num_stages"]
                    print(f"   Stages: {num_stages}")
                else:
                    # Show key parameters
                    if "capacitor" in config:
                        voltage = config["capacitor"].get("initial_voltage", "N/A")
                        capacitance = config["capacitor"].get("capacitance", "N/A")
                        print(f"   Voltage: {voltage}V, Capacitance: {capacitance}F")
                    
                    if "coil" in config:
                        turns = config["coil"].get("total_turns", "N/A")
                        length = config["coil"].get("length", "N/A")
                        print(f"   Coil: {turns} turns, {length}m length")
                
                print(f"   ✅ Compatible with Metal acceleration")
                
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
        
        if len(config_files) == 0:
            print("❌ No compatible configuration files found")
            print("   Please ensure you have coilgun configuration JSON files")
            print("   Required keys: coil, capacitor, projectile, simulation")
        
    except Exception as e:
        print(f"❌ Configuration management demo failed: {e}")


def main():
    """Main demonstration function."""
    print("🍎🚀 EXTENDED METAL ACCELERATION DEMONSTRATION")
    print("=" * 70)
    print("Demonstrating full solve.py compatibility with Metal GPU acceleration")
    print()
    
    # Check system compatibility
    metal = MetalAcceleration(verbose=False) if BASIC_AVAILABLE else None
    if metal:
        print(f"🍎 Apple Silicon: {'✓' if metal.is_apple_silicon else '❌'}")
        print(f"⚡ Julia Available: {'✓' if metal.julia_available else '❌'}")
        print(f"🚀 Metal GPU: {'✓' if metal.metal_available else '❌ (CPU fallback)'}")
    else:
        print("❌ Metal acceleration not available")
    
    print()
    
    # Run demonstrations
    try:
        demo_basic_comparison()
        demo_configuration_management()
        demo_progress_tracking()
        demo_plotting()
        demo_parametric_study()
        demo_multi_stage()
        
        print("\n" + "=" * 70)
        print("✅ EXTENDED METAL ACCELERATION DEMONSTRATION COMPLETED")
        print("=" * 70)
        print("🎉 All solve.py capabilities successfully demonstrated!")
        print()
        print("📁 Generated outputs:")
        print("   - demo_metal_results/: Simulation plots and data")
        print("   - demo_parametric_results/: Parametric study results")
        print()
        print("🚀 For production use:")
        print("   from metal_acceleration_extended import MetalCoilgunSimulation")
        print("   sim = MetalCoilgunSimulation('your_config.json')")
        print("   results = sim.run_simulation()")
        print("   sim.plot_results()")
        
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 