#!/usr/bin/env python3
"""
PID Auto-Tuning Methods Test Script
Tests all three tuning methods: Ziegler-Nichols, Cohen-Coon, and Relay Auto-Tuning
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('scripts')

try:
    from room_thermal_model_pid import SimplePID, PIDTuner, ThermalSimulation
    print("✅ Successfully imported PID components")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_ziegler_nichols():
    """Test Ziegler-Nichols tuning method"""
    print("\n🧪 Testing Ziegler-Nichols Tuning")
    print("-" * 50)
    
    # Example ultimate values (would be determined experimentally)
    Ku = 4000  # Ultimate gain
    Tu = 5.0   # Ultimate period (minutes)
    
    print(f"Given: Ku = {Ku}, Tu = {Tu} minutes")
    print("\nZiegler-Nichols Tuning Results:")
    print("-" * 35)
    
    # Test different controller types
    for controller_type in ["P", "PI", "PD", "PID"]:
        kp, ki, kd = PIDTuner.ziegler_nichols_closed_loop(Ku, Tu, controller_type)
        print(f"{controller_type:3s}: Kp={kp:7.1f}, Ki={ki:7.3f}, Kd={kd:7.1f}")
    
    # Test conservative tuning
    kp_c, ki_c, kd_c = PIDTuner.ziegler_nichols_conservative(Ku, Tu)
    print(f"Conservative: Kp={kp_c:7.1f}, Ki={ki_c:7.3f}, Kd={kd_c:7.1f}")
    
    return kp, ki, kd  # Return PID values for comparison

def test_cohen_coon():
    """Test Cohen-Coon tuning method"""
    print("\n🧪 Testing Cohen-Coon Tuning")
    print("-" * 50)
    
    # Example process parameters (from step response)
    K = 2.5    # Process gain (°C/kW)
    L = 2.0    # Dead time (minutes) 
    T = 8.0    # Time constant (minutes)
    
    print(f"Process Parameters:")
    print(f"  Gain (K): {K} °C/kW")
    print(f"  Dead time (L): {L} minutes")
    print(f"  Time constant (T): {T} minutes")
    print(f"  L/T ratio: {L/T:.3f}")
    
    kp, ki, kd = PIDTuner.cohen_coon(K, L, T)
    
    print(f"\nCohen-Coon PID Tuning:")
    print(f"  Kp = {kp:.1f}")
    print(f"  Ki = {ki:.3f}")  
    print(f"  Kd = {kd:.1f}")
    
    return kp, ki, kd

def test_process_reaction_curve():
    """Test process reaction curve analysis"""
    print("\n🧪 Testing Process Reaction Curve Analysis")
    print("-" * 50)
    
    # Simulate a step response
    time_data = np.linspace(0, 30, 300)  # 30 minutes, 0.1 min steps
    
    # First-order plus dead time response
    K = 2.0      # Process gain
    L = 2.0      # Dead time  
    T = 5.0      # Time constant
    step_input = 10.0  # Step input magnitude (kW)
    
    # Generate ideal response
    response_data = np.zeros_like(time_data)
    for i, t in enumerate(time_data):
        if t > L:  # After dead time
            response_data[i] = K * step_input * (1 - np.exp(-(t - L) / T))
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(response_data))
    response_data += noise
    
    # Analyze the curve
    K_est, L_est, T_est = PIDTuner.process_reaction_curve(step_input, time_data, response_data)
    
    print(f"True parameters:      K={K:.2f}, L={L:.2f}, T={T:.2f}")
    print(f"Estimated parameters: K={K_est:.2f}, L={L_est:.2f}, T={T_est:.2f}")
    
    # Use Cohen-Coon with estimated parameters
    kp, ki, kd = PIDTuner.cohen_coon(K_est, L_est, T_est)
    
    print(f"Cohen-Coon tuning from curve: Kp={kp:.1f}, Ki={ki:.3f}, Kd={kd:.1f}")
    
    return kp, ki, kd, time_data, response_data

def simulate_tuning_comparison():
    """Compare different tuning methods in simulation"""
    print("\n🧪 Comparing Tuning Methods in Simulation")
    print("-" * 50)
    
    # Get tuning parameters from different methods
    print("Calculating tuning parameters...")
    
    # Ziegler-Nichols (using typical values for thermal system)
    Ku = 3000
    Tu = 4.0
    zn_kp, zn_ki, zn_kd = PIDTuner.ziegler_nichols_closed_loop(Ku, Tu, "PID")
    
    # Cohen-Coon (using typical thermal process parameters)
    K = 0.002  # °C/W
    L = 1.0    # 1 minute dead time
    T = 5.0    # 5 minute time constant
    cc_kp, cc_ki, cc_kd = PIDTuner.cohen_coon(K, L, T)
    
    # Conservative Ziegler-Nichols
    cons_kp, cons_ki, cons_kd = PIDTuner.ziegler_nichols_conservative(Ku, Tu)
    
    print(f"Ziegler-Nichols:     Kp={zn_kp:.0f}, Ki={zn_ki:.3f}, Kd={zn_kd:.0f}")
    print(f"Cohen-Coon:          Kp={cc_kp:.0f}, Ki={cc_ki:.3f}, Kd={cc_kd:.0f}")
    print(f"Conservative ZN:     Kp={cons_kp:.0f}, Ki={cons_ki:.3f}, Kd={cons_kd:.0f}")
    
    # Create PID controllers with different tunings
    controllers = [
        (SimplePID(zn_kp, zn_ki, zn_kd, "Ziegler-Nichols"), "blue"),
        (SimplePID(cc_kp, cc_ki, cc_kd, "Cohen-Coon"), "red"),
        (SimplePID(cons_kp, cons_ki, cons_kd, "Conservative-ZN"), "green")
    ]
    
    # Simple thermal simulation
    dt = 0.1  # 0.1 minute time step
    duration = 20  # 20 minutes
    time_steps = int(duration / dt)
    time_array = np.arange(0, duration, dt)
    
    # Setpoint profile with step changes
    setpoint = np.ones(time_steps) * 20.0  # Start at 20°C
    setpoint[50:] = 23.0   # Step to 23°C at t=5min
    setpoint[100:] = 25.0  # Step to 25°C at t=10min
    setpoint[150:] = 22.0  # Step to 22°C at t=15min
    
    print(f"\nRunning {duration}-minute simulation with step changes...")
    
    # Plot setup
    plt.figure(figsize=(15, 10))
    
    results = {}
    
    for pid, color in controllers:
        # Reset controller
        pid.reset()
        
        # Simulation arrays
        temperature = np.zeros(time_steps)
        control_output = np.zeros(time_steps)
        temperature[0] = 20.0  # Initial temperature
        
        # Simple thermal model parameters
        thermal_mass = 100000  # J/K (thermal capacity)
        heat_loss_coeff = 500  # W/K (heat loss to ambient)
        ambient_temp = 18.0    # °C
        
        for i in range(1, time_steps):
            current_temp = temperature[i-1]
            current_setpoint = setpoint[i]
            current_time = time_array[i]
            
            # PID control
            control = pid.compute(current_setpoint, current_temp, current_time)
            control_output[i] = control
            
            # Simple thermal dynamics
            # dT/dt = (Q_heater - Q_loss) / thermal_mass
            Q_heater = control  # W
            Q_loss = heat_loss_coeff * (current_temp - ambient_temp)  # W
            
            dT_dt = (Q_heater - Q_loss) / thermal_mass  # K/s
            temperature[i] = current_temp + dT_dt * (dt * 60)  # Convert dt to seconds
        
        # Store results
        results[pid.name] = {
            'temperature': temperature,
            'control': control_output,
            'color': color
        }
        
        # Calculate performance metrics
        error = temperature - setpoint
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        max_control = np.max(control_output)
        
        print(f"{pid.name:15s}: MAE={mae:.3f}°C, RMSE={rmse:.3f}°C, Max Control={max_control:.0f}W")
    
    # Create plots
    plt.subplot(2, 1, 1)
    plt.plot(time_array, setpoint, 'k--', linewidth=3, label='Setpoint', alpha=0.8)
    for name, data in results.items():
        plt.plot(time_array, data['temperature'], color=data['color'], 
                linewidth=2, label=name, alpha=0.8)
    
    plt.ylabel('Temperature (°C)')
    plt.title('PID Tuning Method Comparison - Temperature Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for name, data in results.items():
        plt.plot(time_array, data['control']/1000, color=data['color'], 
                linewidth=2, label=name, alpha=0.8)
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Control Output (kW)')
    plt.title('Control Effort Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_relay_tuning_concept():
    """Demonstrate the concept of relay auto-tuning"""
    print("\n🧪 Relay Auto-Tuning Concept Demonstration")
    print("-" * 50)
    
    print("Relay Auto-Tuning Process:")
    print("1. Switch PID to manual mode")
    print("2. Apply relay control: +amplitude when temp < setpoint")
    print("                        -amplitude when temp > setpoint")
    print("3. Measure oscillation period (Tu) and amplitude")
    print("4. Calculate ultimate gain: Ku = 4*relay_amplitude/(π*oscillation_amplitude)")
    print("5. Apply Ziegler-Nichols formulas with Ku and Tu")
    
    # Simulate relay response
    dt = 0.05  # Small time step for smooth oscillation
    duration = 20
    time_steps = int(duration / dt)
    time_array = np.arange(0, duration, dt)
    
    # Simple thermal model
    temperature = np.zeros(time_steps)
    control_output = np.zeros(time_steps)
    temperature[0] = 20.0
    setpoint = 22.0
    relay_amplitude = 5000  # 5kW relay
    
    # Thermal parameters
    thermal_mass = 50000
    heat_loss_coeff = 300
    ambient_temp = 18.0
    
    relay_state = 1  # Start positive
    switch_times = []
    
    print(f"\nSimulating relay control...")
    print(f"Setpoint: {setpoint}°C, Relay amplitude: ±{relay_amplitude/1000}kW")
    
    for i in range(1, time_steps):
        current_temp = temperature[i-1]
        current_time = time_array[i]
        
        # Relay logic
        if current_temp > setpoint and relay_state == 1:
            relay_state = -1
            switch_times.append(current_time)
        elif current_temp < setpoint and relay_state == -1:
            relay_state = 1
            switch_times.append(current_time)
        
        control = relay_state * relay_amplitude
        control_output[i] = control
        
        # Thermal dynamics
        Q_heater = max(0, control)  # Only positive heating
        Q_loss = heat_loss_coeff * (current_temp - ambient_temp)
        
        dT_dt = (Q_heater - Q_loss) / thermal_mass
        temperature[i] = current_temp + dT_dt * (dt * 60)
    
    # Analyze oscillation
    if len(switch_times) >= 4:
        # Calculate period from switch times
        periods = []
        for i in range(2, len(switch_times)-1, 2):
            period = switch_times[i+2] - switch_times[i]
            periods.append(period)
        
        Tu = np.mean(periods) if periods else 0
        
        # Calculate amplitude
        temp_max = np.max(temperature)
        temp_min = np.min(temperature)
        oscillation_amplitude = (temp_max - temp_min) / 2
        
        # Calculate ultimate gain
        Ku = (4 * relay_amplitude) / (np.pi * oscillation_amplitude)
        
        print(f"Oscillation analysis:")
        print(f"  Period (Tu): {Tu:.2f} minutes")
        print(f"  Amplitude: {oscillation_amplitude:.3f}°C")
        print(f"  Ultimate gain (Ku): {Ku:.0f}")
        
        # Calculate PID gains
        kp, ki, kd = PIDTuner.ziegler_nichols_closed_loop(Ku, Tu, "PID")
        print(f"  Calculated PID: Kp={kp:.0f}, Ki={ki:.3f}, Kd={kd:.0f}")
    
    # Plot relay response
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_array, temperature, 'b-', linewidth=2, label='Temperature')
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    for switch_time in switch_times:
        plt.axvline(x=switch_time, color='g', alpha=0.3)
    plt.ylabel('Temperature (°C)')
    plt.title('Relay Auto-Tuning: Temperature Response')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_array, control_output/1000, 'r-', linewidth=2, label='Relay Output')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Control Output (kW)')
    plt.title('Relay Control Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Run all tuning method tests"""
    print("🎯 PID Auto-Tuning Methods Test Suite")
    print("Based on Ziegler-Nichols, Cohen-Coon, and Relay Auto-Tuning")
    print("=" * 70)
    
    try:
        # Test individual tuning methods
        test_ziegler_nichols()
        test_cohen_coon()
        test_process_reaction_curve()
        
        # Demonstrate relay tuning concept
        demonstrate_relay_tuning_concept()
        
        # Compare methods in simulation
        simulate_tuning_comparison()
        
        print("\n✅ All tuning method tests completed successfully!")
        print("\n🎯 Available Tuning Methods:")
        print("   • Ziegler-Nichols (Closed-Loop) - Quick, model-free")
        print("   • Cohen-Coon - Good for first-order plus delay systems")
        print("   • Relay Auto-Tuning - Automated, safe oscillation method")
        print("   • Conservative variants for less aggressive control")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
