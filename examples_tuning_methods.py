#!/usr/bin/env python3
"""
Simple Example: Using PID Auto-Tuning Methods
Demonstrates how to use the three tuning methods in practice
"""

import sys
import numpy as np
sys.path.append('scripts')

from room_thermal_model_pid import SimplePID, PIDTuner

def example_1_ziegler_nichols():
    """Example 1: Using Ziegler-Nichols tuning"""
    print("📖 Example 1: Ziegler-Nichols Tuning")
    print("-" * 40)
    
    # Step 1: You've determined ultimate gain and period experimentally
    print("You performed an ultimate gain test and found:")
    Ku = 4000  # Ultimate gain at which system oscillates
    Tu = 5.0   # Period of oscillation in minutes
    print(f"  Ultimate gain (Ku): {Ku}")
    print(f"  Ultimate period (Tu): {Tu} minutes")
    
    # Step 2: Calculate PID parameters
    print("\nCalculating PID parameters...")
    kp, ki, kd = PIDTuner.ziegler_nichols_closed_loop(Ku, Tu, "PID")
    
    print(f"Ziegler-Nichols PID tuning:")
    print(f"  Kp = {kp}")
    print(f"  Ki = {ki:.3f}")
    print(f"  Kd = {kd}")
    
    # Step 3: Create and configure PID controller
    pid = SimplePID(kp, ki, kd, "ZN-Tuned")
    print(f"\nPID controller created: {pid.name}")
    
    return pid

def example_2_cohen_coon():
    """Example 2: Using Cohen-Coon tuning from step test"""
    print("\n📖 Example 2: Cohen-Coon Tuning")
    print("-" * 40)
    
    # Step 1: You performed a step test and analyzed the response
    print("You performed a step test and identified:")
    K = 0.0025  # Process gain: 0.0025°C/W
    L = 1.5     # Dead time: 1.5 minutes
    T = 6.0     # Time constant: 6.0 minutes
    
    print(f"  Process gain (K): {K} °C/W")
    print(f"  Dead time (L): {L} minutes")
    print(f"  Time constant (T): {T} minutes")
    print(f"  Controllability ratio (L/T): {L/T:.3f}")
    
    # Step 2: Check if suitable for Cohen-Coon
    if L/T > 1.0:
        print("⚠️  Warning: L/T > 1.0, Cohen-Coon may not be suitable")
    else:
        print("✅ Good controllability ratio for Cohen-Coon method")
    
    # Step 3: Calculate PID parameters
    print("\nCalculating PID parameters...")
    kp, ki, kd = PIDTuner.cohen_coon(K, L, T)
    
    print(f"Cohen-Coon PID tuning:")
    print(f"  Kp = {kp:.1f}")
    print(f"  Ki = {ki:.4f}")
    print(f"  Kd = {kd:.1f}")
    
    # Step 4: Create PID controller
    pid = SimplePID(kp, ki, kd, "Cohen-Coon-Tuned")
    print(f"\nPID controller created: {pid.name}")
    
    return pid

def example_3_relay_auto_tuning():
    """Example 3: Relay auto-tuning procedure"""
    print("\n📖 Example 3: Relay Auto-Tuning")
    print("-" * 40)
    
    print("Relay auto-tuning procedure:")
    print("1. Set controller to manual mode")
    print("2. Apply relay test with appropriate amplitude")
    print("3. Wait for sustained oscillation")
    print("4. Measure period and amplitude")
    print("5. Calculate ultimate parameters")
    
    # Simulated relay test results
    print("\nSimulated relay test results:")
    relay_amplitude = 3000      # 3kW relay amplitude
    oscillation_period = 4.2    # 4.2 minute period
    temp_amplitude = 0.8        # 0.8°C temperature oscillation
    
    print(f"  Relay amplitude: {relay_amplitude} W")
    print(f"  Oscillation period: {oscillation_period} minutes")
    print(f"  Temperature amplitude: {temp_amplitude} °C")
    
    # Calculate ultimate parameters
    Ku, Tu = PIDTuner.relay_auto_tune(relay_amplitude, oscillation_period, temp_amplitude)
    
    print(f"\nCalculated ultimate parameters:")
    print(f"  Ultimate gain (Ku): {Ku:.0f}")
    print(f"  Ultimate period (Tu): {Tu} minutes")
    
    # Apply Ziegler-Nichols with these parameters
    kp, ki, kd = PIDTuner.ziegler_nichols_closed_loop(Ku, Tu, "PID")
    
    print(f"\nRelay-based PID tuning:")
    print(f"  Kp = {kp:.0f}")
    print(f"  Ki = {ki:.4f}")
    print(f"  Kd = {kd:.0f}")
    
    # Create PID controller
    pid = SimplePID(kp, ki, kd, "Relay-Tuned")
    print(f"\nPID controller created: {pid.name}")
    
    return pid

def example_4_process_identification():
    """Example 4: Process identification from step response data"""
    print("\n📖 Example 4: Process Identification")
    print("-" * 40)
    
    print("Analyzing step response data...")
    
    # Simulated step response data
    # In practice, this would come from your actual system
    time_data = np.linspace(0, 20, 200)  # 20 minutes, 0.1 min resolution
    
    # Parameters for simulation
    true_K = 0.002   # True process gain
    true_L = 1.0     # True dead time  
    true_T = 4.0     # True time constant
    step_input = 5000  # 5kW step input
    
    # Generate ideal response with noise
    response_data = np.zeros_like(time_data)
    for i, t in enumerate(time_data):
        if t > true_L:
            response_data[i] = true_K * step_input * (1 - np.exp(-(t - true_L) / true_T))
    
    # Add realistic measurement noise
    noise = np.random.normal(0, 0.02, len(response_data))
    response_data += noise
    
    print(f"Step input: {step_input} W")
    print(f"Data points: {len(time_data)}")
    print(f"Duration: {time_data[-1]} minutes")
    
    # Identify process parameters
    K_est, L_est, T_est = PIDTuner.process_reaction_curve(step_input, time_data, response_data)
    
    print(f"\nProcess identification results:")
    print(f"  Estimated gain (K): {K_est:.4f} °C/W")
    print(f"  Estimated dead time (L): {L_est:.2f} minutes")
    print(f"  Estimated time constant (T): {T_est:.2f} minutes")
    
    print(f"\nActual parameters (for comparison):")
    print(f"  True gain (K): {true_K:.4f} °C/W")
    print(f"  True dead time (L): {true_L:.2f} minutes")
    print(f"  True time constant (T): {true_T:.2f} minutes")
    
    # Use identified parameters for Cohen-Coon tuning
    kp, ki, kd = PIDTuner.cohen_coon(K_est, L_est, T_est)
    
    print(f"\nCohen-Coon tuning from identified model:")
    print(f"  Kp = {kp:.1f}")
    print(f"  Ki = {ki:.4f}")
    print(f"  Kd = {kd:.1f}")
    
    pid = SimplePID(kp, ki, kd, "Identified-CC-Tuned")
    print(f"\nPID controller created: {pid.name}")
    
    return pid

def quick_tuning_guide():
    """Quick reference guide for choosing tuning methods"""
    print("\n🎯 Quick Tuning Method Selection Guide")
    print("=" * 50)
    
    print("\n🔄 Ziegler-Nichols (Closed-Loop)")
    print("   Use when: Need quick tuning, can tolerate temporary oscillation")
    print("   Pros: Fast, no model required, widely applicable")
    print("   Cons: Aggressive tuning, may cause large overshoots")
    print("   Best for: Systems that can handle oscillation during tuning")
    
    print("\n📊 Cohen-Coon")
    print("   Use when: Have good step response data, L/T < 1.0")
    print("   Pros: Good for first-order systems, considers dead time")
    print("   Cons: Requires process identification, limited to FOPDT")
    print("   Best for: Well-behaved thermal systems with known dynamics")
    
    print("\n🔁 Relay Auto-Tuning")
    print("   Use when: Want automated tuning, prefer safe oscillation")
    print("   Pros: Automated, controlled oscillation, good compromise")
    print("   Cons: Takes time, may not work with very slow systems")
    print("   Best for: Production systems needing automated tuning")
    
    print("\n💡 Practical Tips:")
    print("   • Start with conservative gains (0.6x calculated values)")
    print("   • Test in manual mode first")
    print("   • Monitor for oscillation and adjust if needed")
    print("   • Consider your system's safety constraints")

def main():
    """Run all examples"""
    print("🔧 PID Auto-Tuning Methods - Practical Examples")
    print("=" * 55)
    
    # Run examples
    pid1 = example_1_ziegler_nichols()
    pid2 = example_2_cohen_coon()
    pid3 = example_3_relay_auto_tuning()
    pid4 = example_4_process_identification()
    
    # Show comparison
    print("\n📋 Summary of Tuned Controllers")
    print("-" * 45)
    controllers = [pid1, pid2, pid3, pid4]
    
    for pid in controllers:
        print(f"{pid.name:20s}: Kp={pid.kp:6.0f}, Ki={pid.ki:7.4f}, Kd={pid.kd:6.0f}")
    
    # Show selection guide
    quick_tuning_guide()
    
    print("\n✅ All examples completed!")
    print("\n🚀 Next steps:")
    print("   1. Choose the most appropriate method for your system")
    print("   2. Implement the tuning procedure")
    print("   3. Test and fine-tune as needed")
    print("   4. Monitor performance in operation")

if __name__ == "__main__":
    main()
