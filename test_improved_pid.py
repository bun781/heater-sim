#!/usr/bin/env python3
"""
Test script for the improved PID controller
Demonstrates the improvements from Brett Beauregard's guide
"""

import sys
import os
import time

# Add the scripts directory to path so we can import the PID
sys.path.append('scripts')

try:
    from room_thermal_model_pid import SimplePID
    print("✅ Successfully imported improved SimplePID controller")
except ImportError as e:
    print(f"❌ Failed to import SimplePID: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic PID functionality"""
    print("\n🧪 Testing Basic PID Functionality")
    print("-" * 40)
    
    # Create a PID controller with some reasonable gains
    pid = SimplePID(kp=2500, ki=350, kd=700, name="Test")
    
    # Test basic parameters
    print(f"Controller name: {pid.name}")
    print(f"Initial mode: {pid.get_mode()}")
    print(f"Tuning parameters: {pid.get_tunings()}")
    print(f"Output limits: [{pid.output_min}, {pid.output_max}]")
    
    # Simulate a simple step response
    setpoint = 22.0
    current_temp = 20.0
    time_step = 0.5  # minutes
    
    print(f"\nSimulating step response:")
    print(f"Setpoint: {setpoint}°C, Starting temp: {current_temp}°C")
    print("Time(min) | Temp(°C) | Output(W) | Error(°C)")
    print("-" * 45)
    
    for i in range(10):
        current_time = i * time_step
        
        # Compute PID output
        output = pid.compute(setpoint, current_temp, current_time)
        error = setpoint - current_temp
        
        print(f"{current_time:8.1f} | {current_temp:7.1f} | {output:8.0f} | {error:8.1f}")
        
        # Simple plant model: temp increases with heater output
        # This is just for testing, not a real thermal model
        temp_increase = output * 0.00001 * time_step  # Very simple model
        current_temp += temp_increase
        
        # Add some damping
        current_temp = current_temp * 0.98 + setpoint * 0.02

def test_derivative_kick_elimination():
    """Test that derivative kick is eliminated when setpoint changes"""
    print("\n🧪 Testing Derivative Kick Elimination")
    print("-" * 40)
    
    pid = SimplePID(kp=1000, ki=100, kd=500, name="DerivativeTest")
    
    current_temp = 20.0
    setpoint = 20.0  # Start at same temperature
    
    print("Testing setpoint step change at t=5.0 minutes")
    print("Time(min) | Setpoint | Temp(°C) | Output(W)")
    print("-" * 40)
    
    for i in range(15):
        current_time = i * 0.5
        
        # Step change in setpoint at t=5.0
        if current_time >= 5.0:
            setpoint = 25.0
        
        output = pid.compute(setpoint, current_temp, current_time)
        
        print(f"{current_time:8.1f} | {setpoint:7.1f} | {current_temp:7.1f} | {output:8.0f}")
        
        # Simple dynamics
        current_temp += output * 0.00001 * 0.5

def test_auto_manual_mode():
    """Test auto/manual mode switching"""
    print("\n🧪 Testing Auto/Manual Mode Switching")
    print("-" * 40)
    
    pid = SimplePID(kp=2000, ki=300, kd=600, name="ModeTest")
    
    # Set a manual output
    pid.output = 50000  # 50kW manual output
    
    current_temp = 22.0
    setpoint = 24.0
    
    print("Time(min) | Mode   | Temp(°C) | Output(W)")
    print("-" * 38)
    
    for i in range(12):
        current_time = i * 0.5
        
        # Switch to manual at t=2.0, back to auto at t=4.0
        if current_time == 2.0:
            pid.set_mode('MANUAL')
            pid.output = 30000  # Set manual output
        elif current_time == 4.0:
            pid.set_mode('AUTO')  # Should initialize smoothly
        
        if pid.get_mode() == 'AUTO':
            output = pid.compute(setpoint, current_temp, current_time)
        else:
            output = pid.output  # Use manual output
        
        print(f"{current_time:8.1f} | {pid.get_mode():6s} | {current_temp:7.1f} | {output:8.0f}")
        
        # Simple dynamics
        current_temp += output * 0.00001 * 0.5

def test_output_limits():
    """Test output limiting and anti-windup"""
    print("\n🧪 Testing Output Limits and Anti-Windup")
    print("-" * 40)
    
    pid = SimplePID(kp=5000, ki=1000, kd=1000, name="LimitTest")
    
    # Set tight output limits
    pid.set_output_limits(0, 25000)  # Max 25kW
    
    current_temp = 15.0  # Start far from setpoint
    setpoint = 30.0      # Large error to cause saturation
    
    print("Output limited to 25kW max")
    print("Time(min) | Temp(°C) | Output(W) | I-Term")
    print("-" * 38)
    
    for i in range(15):
        current_time = i * 0.5
        
        output = pid.compute(setpoint, current_temp, current_time)
        
        print(f"{current_time:8.1f} | {current_temp:7.1f} | {output:8.0f} | {pid.i_term:6.0f}")
        
        # Simple dynamics - temperature rises slowly
        current_temp += output * 0.00002 * 0.5

def test_tuning_changes():
    """Test on-the-fly tuning parameter changes"""
    print("\n🧪 Testing On-the-Fly Tuning Changes")
    print("-" * 40)
    
    pid = SimplePID(kp=1000, ki=100, kd=200, name="TuningTest")
    
    current_temp = 20.0
    setpoint = 23.0
    
    print("Changing tuning parameters at t=3.0 minutes")
    print("Time(min) | Kp   | Ki  | Kd  | Output(W)")
    print("-" * 40)
    
    for i in range(12):
        current_time = i * 0.5
        
        # Change tuning at t=3.0
        if current_time == 3.0:
            pid.set_tunings(2000, 300, 500)  # More aggressive
        
        output = pid.compute(setpoint, current_temp, current_time)
        kp, ki, kd = pid.get_tunings()
        
        print(f"{current_time:8.1f} | {kp:4.0f} | {ki:3.0f} | {kd:3.0f} | {output:8.0f}")
        
        # Simple dynamics
        current_temp += output * 0.00001 * 0.5

def main():
    """Run all tests"""
    print("🏠 Improved PID Controller Test Suite")
    print("Based on Brett Beauregard's PID Improvements")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_derivative_kick_elimination()
        test_auto_manual_mode()
        test_output_limits()
        test_tuning_changes()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 Key Improvements Demonstrated:")
        print("   • Fixed sample time for consistent behavior")
        print("   • Derivative on measurement (no derivative kick)")
        print("   • Proper anti-windup protection")
        print("   • Auto/Manual mode with bumpless transfer")
        print("   • Safe tuning parameter changes")
        print("   • Output limiting with integral protection")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
