"""
PID Controller Module with Advanced Auto-Tuning Methods

Based on:
- Astrom & Hagglund "PID Controllers: Theory, Design and Tuning" (1995)
- Franklin et al. "Feedback Control of Dynamic Systems" (2019)
- Ziegler-Nichols tuning methods
- Relay feedback auto-tuning (Astrom-Hagglund method)
- Cohen-Coon tuning rules
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List, Callable
import time
from dataclasses import dataclass

@dataclass
class TuningResults:
    """Results from auto-tuning procedures"""
    method: str
    kp: float
    ki: float
    kd: float
    ultimate_gain: Optional[float] = None
    ultimate_period: Optional[float] = None
    dead_time: Optional[float] = None
    time_constant: Optional[float] = None
    steady_state_gain: Optional[float] = None
    success: bool = True
    message: str = ""

class PIDController:
    """
    Advanced PID controller implementation with multiple auto-tuning methods
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0, 
                 output_limits: Tuple[float, float] = (0, 100000),
                 derivative_filter_tau: float = 0.1):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain  
            kd: Derivative gain
            output_limits: (min, max) output limits in Watts
            derivative_filter_tau: Derivative filter time constant (minutes)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.derivative_filter_tau = derivative_filter_tau
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        self.filtered_derivative = 0.0
        
        # For debugging/monitoring
        self.last_proportional = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0
        
        # Auto-tuning state
        self.tuning_data = []
        self.is_tuning = False
        
    def compute(self, setpoint: float, measurement: float, 
                current_time: float) -> float:
        """
        Compute PID control output
        
        Args:
            setpoint: Desired temperature (°C)
            measurement: Current temperature (°C)
            current_time: Current time (minutes)
            
        Returns:
            Control output (Watts)
        """
        
        # Error calculation
        error = setpoint - measurement
        
        # Time step
        dt = current_time - self.previous_time if self.previous_time > 0 else 0.1
        dt = max(dt, 1e-6)  # Prevent division by zero
        
        # Proportional term
        self.last_proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        self.last_integral = self.ki * self.integral
        
        # Derivative term with filtering
        if dt > 0:
            raw_derivative = (error - self.previous_error) / dt
            # First-order filter for derivative
            alpha = dt / (self.derivative_filter_tau + dt)
            self.filtered_derivative = (1 - alpha) * self.filtered_derivative + alpha * raw_derivative
        else:
            self.filtered_derivative = 0.0
            
        self.last_derivative = self.kd * self.filtered_derivative
        
        # PID output
        output = self.last_proportional + self.last_integral + self.last_derivative
        
        # Apply output limits
        output_limited = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Anti-windup: adjust integral if output is saturated
        if output != output_limited and abs(self.last_integral) > 0:
            # Back-calculate integral to prevent windup
            excess = output - output_limited
            self.integral -= excess / self.ki if self.ki != 0 else 0
        
        # Store data for tuning if active
        if self.is_tuning:
            self.tuning_data.append({
                'time': current_time,
                'setpoint': setpoint,
                'measurement': measurement,
                'error': error,
                'output': output_limited
            })
        
        # Update state
        self.previous_error = error
        self.previous_time = current_time
        
        return output_limited
    
    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        self.filtered_derivative = 0.0
        self.last_proportional = 0.0
        self.last_integral = 0.0
        self.last_derivative = 0.0
        self.tuning_data = []
    
    def get_components(self) -> Tuple[float, float, float]:
        """Get the last P, I, D components for debugging"""
        return self.last_proportional, self.last_integral, self.last_derivative
    
    def set_gains(self, kp: float, ki: float, kd: float):
        """Set PID gains"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        print(f"PID gains updated: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")

class AutoTuner:
    """
    Auto-tuning class implementing multiple tuning methods
    """
    
    def __init__(self, pid_controller: PIDController):
        self.pid = pid_controller
        self.plant_simulator = None  # Will be set by simulation
        
    def set_plant_simulator(self, simulator_func: Callable):
        """Set the plant simulator function for auto-tuning"""
        self.plant_simulator = simulator_func
    
    def ziegler_nichols_closed_loop(self, setpoint: float = 22.0, 
                                   test_duration: float = 120.0) -> TuningResults:
        """
        Ziegler-Nichols closed-loop method
        
        Approach: Set I and D to zero, ramp up Kp until sustained oscillation
        
        Args:
            setpoint: Target temperature for test
            test_duration: Test duration in minutes
            
        Returns:
            TuningResults with recommended gains
        """
        print("🔧 Starting Ziegler-Nichols Closed-Loop Auto-Tuning...")
        
        if not self.plant_simulator:
            return TuningResults("Ziegler-Nichols", 0, 0, 0, success=False, 
                                message="No plant simulator available")
        
        # Start with P-only control
        original_gains = (self.pid.kp, self.pid.ki, self.pid.kd)
        self.pid.set_gains(1.0, 0.0, 0.0)
        
        # Test different Kp values to find ultimate gain
        kp_values = np.logspace(0, 3, 20)  # 1 to 1000
        ultimate_gain = None
        ultimate_period = None
        
        for kp in kp_values:
            print(f"  Testing Kp = {kp:.1f}")
            self.pid.set_gains(kp, 0.0, 0.0)
            
            # Run simulation
            oscillation_data = self._run_oscillation_test(setpoint, test_duration)
            
            # Check for sustained oscillation
            if self._detect_oscillation(oscillation_data):
                ultimate_gain = kp
                ultimate_period = self._calculate_period(oscillation_data)
                print(f"  ✓ Found ultimate gain: Ku = {ultimate_gain:.1f}")
                print(f"  ✓ Ultimate period: Tu = {ultimate_period:.1f} minutes")
                break
        
        # Restore original gains if tuning failed
        if ultimate_gain is None:
            self.pid.set_gains(*original_gains)
            return TuningResults("Ziegler-Nichols", *original_gains, success=False,
                                message="Could not find ultimate gain")
        
        # Calculate Ziegler-Nichols gains
        kp_zn = 0.6 * ultimate_gain
        ki_zn = 1.2 * ultimate_gain / ultimate_period
        kd_zn = 0.075 * ultimate_gain * ultimate_period
        
        # Apply new gains
        self.pid.set_gains(kp_zn, ki_zn, kd_zn)
        
        return TuningResults(
            method="Ziegler-Nichols Closed-Loop",
            kp=kp_zn, ki=ki_zn, kd=kd_zn,
            ultimate_gain=ultimate_gain,
            ultimate_period=ultimate_period,
            success=True,
            message=f"Tuning successful. Ku={ultimate_gain:.1f}, Tu={ultimate_period:.1f}"
        )
    
    def relay_feedback_tuning(self, setpoint: float = 22.0, 
                             relay_amplitude: float = 1000.0,
                             test_duration: float = 60.0) -> TuningResults:
        """
        Relay feedback auto-tuning (Åström-Hägglund method)
        
        Approach: Use relay (on/off) control to force oscillations
        
        Args:
            setpoint: Target temperature
            relay_amplitude: Relay output amplitude (Watts)
            test_duration: Test duration in minutes
            
        Returns:
            TuningResults with recommended gains
        """
        print("🔧 Starting Relay Feedback Auto-Tuning...")
        
        if not self.plant_simulator:
            return TuningResults("Relay Feedback", 0, 0, 0, success=False,
                                message="No plant simulator available")
        
        # Store original gains
        original_gains = (self.pid.kp, self.pid.ki, self.pid.kd)
        
        # Run relay test
        relay_data = self._run_relay_test(setpoint, relay_amplitude, test_duration)
        
        # Analyze oscillation
        if not self._detect_oscillation(relay_data):
            self.pid.set_gains(*original_gains)
            return TuningResults("Relay Feedback", *original_gains, success=False,
                                message="No sustained oscillation detected")
        
        # Calculate ultimate parameters from relay test
        period = self._calculate_period(relay_data)
        amplitude = self._calculate_amplitude(relay_data)
        
        # Ultimate gain calculation for relay feedback
        ultimate_gain = (4 * relay_amplitude) / (np.pi * amplitude)
        ultimate_period = period
        
        print(f"  ✓ Ultimate gain from relay: Ku = {ultimate_gain:.1f}")
        print(f"  ✓ Ultimate period: Tu = {ultimate_period:.1f} minutes")
        
        # Ziegler-Nichols gains from relay data
        kp_relay = 0.6 * ultimate_gain
        ki_relay = 1.2 * ultimate_gain / ultimate_period
        kd_relay = 0.075 * ultimate_gain * ultimate_period
        
        # Apply new gains
        self.pid.set_gains(kp_relay, ki_relay, kd_relay)
        
        return TuningResults(
            method="Relay Feedback (Åström-Hägglund)",
            kp=kp_relay, ki=ki_relay, kd=kd_relay,
            ultimate_gain=ultimate_gain,
            ultimate_period=ultimate_period,
            success=True,
            message=f"Relay tuning successful. Ku={ultimate_gain:.1f}, Tu={ultimate_period:.1f}"
        )
    
    def cohen_coon_tuning(self, setpoint: float = 22.0, 
                         step_size: float = 2.0,
                         test_duration: float = 60.0) -> TuningResults:
        """
        Cohen-Coon tuning method (Process Reaction Curve)
        
        Approach: Apply step change, measure reaction curve parameters
        
        Args:
            setpoint: Base temperature
            step_size: Step change magnitude (°C)
            test_duration: Test duration in minutes
            
        Returns:
            TuningResults with Cohen-Coon gains
        """
        print("🔧 Starting Cohen-Coon Process Reaction Curve Tuning...")
        
        if not self.plant_simulator:
            return TuningResults("Cohen-Coon", 0, 0, 0, success=False,
                                message="No plant simulator available")
        
        # Store original gains
        original_gains = (self.pid.kp, self.pid.ki, self.pid.kd)
        
        # Set to manual mode (open-loop test)
        self.pid.set_gains(0.0, 0.0, 0.0)
        
        # Run step response test
        step_data = self._run_step_response_test(setpoint, step_size, test_duration)
        
        # Analyze step response
        analysis = self._analyze_step_response(step_data, step_size)
        
        if not analysis['success']:
            self.pid.set_gains(*original_gains)
            return TuningResults("Cohen-Coon", *original_gains, success=False,
                                message=analysis['message'])
        
        L = analysis['dead_time']      # Dead time
        T = analysis['time_constant']  # Time constant  
        K = analysis['steady_state_gain']  # Steady-state gain
        
        print(f"  ✓ Dead time (L): {L:.2f} minutes")
        print(f"  ✓ Time constant (T): {T:.2f} minutes")
        print(f"  ✓ Steady-state gain (K): {K:.3f}")
        
        # Cohen-Coon tuning formulas
        tau = L / T  # Dimensionless dead time
        
        # Cohen-Coon PID formulas
        kp_cc = (1/K) * (1.35 + 0.25*tau) / tau
        ti_cc = L * (2.5 - 2*tau) / (1 - 0.39*tau)  # Integral time
        td_cc = L * 0.37 / (1 - 0.81*tau)           # Derivative time
        
        ki_cc = kp_cc / ti_cc if ti_cc > 0 else 0
        kd_cc = kp_cc * td_cc
        
        # Apply new gains
        self.pid.set_gains(kp_cc, ki_cc, kd_cc)
        
        return TuningResults(
            method="Cohen-Coon",
            kp=kp_cc, ki=ki_cc, kd=kd_cc,
            dead_time=L, time_constant=T, steady_state_gain=K,
            success=True,
            message=f"Cohen-Coon tuning successful. L={L:.2f}, T={T:.2f}, K={K:.3f}"
        )
    
    def _run_oscillation_test(self, setpoint: float, duration: float) -> List[Dict]:
        """Run oscillation test for Ziegler-Nichols method"""
        # This would interface with the building simulation
        # For now, return mock data - will be implemented by simulation
        return []
    
    def _run_relay_test(self, setpoint: float, amplitude: float, duration: float) -> List[Dict]:
        """Run relay feedback test"""
        # This would interface with the building simulation
        # Implementation depends on simulation interface
        return []
    
    def _run_step_response_test(self, setpoint: float, step_size: float, duration: float) -> List[Dict]:
        """Run step response test for Cohen-Coon method"""
        # This would interface with the building simulation
        # Implementation depends on simulation interface
        return []
    
    def _detect_oscillation(self, data: List[Dict]) -> bool:
        """Detect if system is oscillating"""
        if len(data) < 10:
            return False
        
        # Simple oscillation detection based on zero crossings
        measurements = [d['measurement'] for d in data]
        mean_val = np.mean(measurements)
        crossings = 0
        
        for i in range(1, len(measurements)):
            if (measurements[i-1] - mean_val) * (measurements[i] - mean_val) < 0:
                crossings += 1
        
        # Need at least 4 crossings for sustained oscillation
        return crossings >= 4
    
    def _calculate_period(self, data: List[Dict]) -> float:
        """Calculate oscillation period"""
        if len(data) < 10:
            return 1.0
        
        measurements = [d['measurement'] for d in data]
        times = [d['time'] for d in data]
        
        # Find peaks
        peaks = []
        for i in range(1, len(measurements)-1):
            if measurements[i] > measurements[i-1] and measurements[i] > measurements[i+1]:
                peaks.append(times[i])
        
        if len(peaks) < 2:
            return 1.0
        
        # Average period between peaks
        periods = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
        return np.mean(periods) if periods else 1.0
    
    def _calculate_amplitude(self, data: List[Dict]) -> float:
        """Calculate oscillation amplitude"""
        if len(data) < 10:
            return 1.0
        
        measurements = [d['measurement'] for d in data]
        return (np.max(measurements) - np.min(measurements)) / 2
    
    def _analyze_step_response(self, data: List[Dict], step_size: float) -> Dict:
        """Analyze step response for Cohen-Coon parameters"""
        if len(data) < 20:
            return {'success': False, 'message': 'Insufficient data'}
        
        measurements = [d['measurement'] for d in data]
        times = [d['time'] for d in data]
        
        # Find steady-state value
        steady_state = np.mean(measurements[-10:])  # Last 10 points
        initial_value = measurements[0]
        
        # Steady-state gain
        K = (steady_state - initial_value) / step_size
        
        # Find 28.3% and 63.2% response points (for dead time and time constant)
        target_28 = initial_value + 0.283 * (steady_state - initial_value)
        target_63 = initial_value + 0.632 * (steady_state - initial_value)
        
        t_28 = None
        t_63 = None
        
        for i, val in enumerate(measurements):
            if t_28 is None and val >= target_28:
                t_28 = times[i]
            if t_63 is None and val >= target_63:
                t_63 = times[i]
                break
        
        if t_28 is None or t_63 is None:
            return {'success': False, 'message': 'Could not find response points'}
        
        # Calculate parameters
        L = 1.5 * t_28 - 0.5 * t_63  # Dead time
        T = t_63 - t_28               # Time constant
        
        return {
            'success': True,
            'dead_time': max(L, 0.1),  # Ensure positive
            'time_constant': max(T, 0.1),
            'steady_state_gain': K,
            'message': 'Step response analysis successful'
        }

def create_pid_controller(controller_type: str = "standard") -> PIDController:
    """
    Factory function to create pre-tuned PID controllers
    
    Args:
        controller_type: "aggressive", "standard", "conservative"
        
    Returns:
        Configured PIDController instance
    """
    
    if controller_type == "aggressive":
        return PIDController(kp=800, ki=80, kd=150)
    elif controller_type == "conservative":
        return PIDController(kp=200, ki=20, kd=50)
    else:  # standard
        return PIDController(kp=500, ki=50, kd=100)

def demonstrate_tuning_methods():
    """Demonstrate the different tuning methods"""
    print("🎯 PID Auto-Tuning Methods Demonstration")
    print("=" * 50)
    
    # Create PID controller
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
    tuner = AutoTuner(pid)
    
    print("Available tuning methods:")
    print("1. Ziegler-Nichols Closed-Loop")
    print("   - Ramp up Kp until sustained oscillation")
    print("   - Calculate Ku (ultimate gain) and Tu (period)")
    print("   - Apply Z-N formulas: Kp=0.6*Ku, Ki=1.2*Ku/Tu, Kd=0.075*Ku*Tu")
    print()
    
    print("2. Relay Feedback (Åström-Hägglund)")
    print("   - Use on/off relay to force oscillations")
    print("   - Measure amplitude and period")
    print("   - Calculate ultimate parameters from relay response")
    print()
    
    print("3. Cohen-Coon (Process Reaction Curve)")
    print("   - Apply step change in open-loop")
    print("   - Measure dead time (L), time constant (T), gain (K)")
    print("   - Apply Cohen-Coon formulas for first-order plus delay systems")
    print()
    
    print("💡 These methods will be integrated with the building simulation")
    print("   to provide automatic PID tuning capabilities.")

if __name__ == "__main__":
    demonstrate_tuning_methods()
