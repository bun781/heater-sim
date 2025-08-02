"""
Advanced Room Thermal Simulation with PID Control
- Environment variable configuration via .env files
- TRUE discrete setpoint values (step changes, no interpolation)
- Simplified simulation without environmental conditions
- Modular PID-Simulation interaction at each time step
"""

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_array(env_var: str, default_values: List[float]) -> np.ndarray:
    """
    Get array from environment variable
    
    Args:
        env_var: Environment variable name
        default_values: Default values if env var not found
        
    Returns:
        NumPy array of values
    """
    env_value = os.getenv(env_var)
    if env_value:
        try:
            # Parse comma-separated values
            values = [float(x.strip()) for x in env_value.split(',')]
            return np.array(values)
        except ValueError as e:
            print(f"⚠️  Error parsing {env_var}: {e}. Using defaults.")
            return np.array(default_values)
    else:
        print(f"⚠️  {env_var} not found in .env file. Using defaults.")
        return np.array(default_values)

def get_env_float(env_var: str, default_value: float) -> float:
    """
    Get float from environment variable
    
    Args:
        env_var: Environment variable name
        default_value: Default value if env var not found
        
    Returns:
        Float value
    """
    env_value = os.getenv(env_var)
    if env_value:
        try:
            return float(env_value)
        except ValueError as e:
            print(f"⚠️  Error parsing {env_var}: {e}. Using default: {default_value}")
            return default_value
    else:
        print(f"⚠️  {env_var} not found in .env file. Using default: {default_value}")
        return default_value

# ============================================================================
# ENVIRONMENT VARIABLE CONFIGURATION
# ============================================================================

# Load arrays from environment variables
SETPOINT_ARRAY = get_env_array('SETPOINT_ARRAY', [27.0, 26.5, 26.0, 25.0, 24.5, 23.5, 23.0, 23.5, 24.5, 25.0, 26.0, 26.5, 27.0])
# SETPOINT_ARRAY = get_env_array('SETPOINT_ARRAY', [24])
AMBIENT_ARRAY = get_env_array('AMBIENT_ARRAY', [-5, -4, -3, -2, 0, 1, 3, 4, 5, 7, 8, 9, 10, 9, 8, 7, 5, 4, 3, 1, 0, -2, -3, -4, -5])
BACKGROUND_LOSS_ARRAY = get_env_array('BACKGROUND_LOSS_ARRAY', [0])

# Load simulation parameters
DURATION_HOURS = get_env_float('DURATION_HOURS', 25.0)
TIME_STEP_MINUTES = get_env_float('TIME_STEP_MINUTES', 0.5)

# Load room physical properties
ROOM_VOLUME = get_env_float('ROOM_VOLUME', 140.0)
COOLING_COEFFICIENT = get_env_float('COOLING_COEFFICIENT', 0.002)
THERMAL_CAPACITY_PER_M3 = get_env_float('THERMAL_CAPACITY_PER_M3', 1900.0)
HEATER_EFFICIENCY = get_env_float('HEATER_EFFICIENCY', 0.95)

# Load control system settings
OUTPUT_MIN = get_env_float('OUTPUT_MIN', 0.0)
OUTPUT_MAX = get_env_float('OUTPUT_MAX', 100000.0)
CONTROL_LAG_MINUTES = get_env_float('CONTROL_LAG_MINUTES', 0)
NOISE_STD = get_env_float('NOISE_STD', 50.0)

# Placeholder values
STANDARD_KP = get_env_float('STANDARD_KP', 2500)
STANDARD_KI = get_env_float('STANDARD_KI', 350)
STANDARD_KD = get_env_float('STANDARD_KD', 700)

# ============================================================================

class SimplePID:
    """
    Improved PID Controller based on Brett Beauregard's Guide
    Includes all major improvements: sample time, derivative kick elimination,
    reset windup prevention, auto/manual mode, and proper initialization
    """
    
    def __init__(self, kp: float, ki: float, kd: float, name: str):
        self.name = name
        
        # Tuning parameters (will be modified for fixed sample time)
        self.dispKp = kp  # Display/user-entered values
        self.dispKi = ki
        self.dispKd = kd
        
        # Working parameters (modified for sample time)
        self.kp = kp
        self.ki = ki * TIME_STEP_MINUTES  # Convert to per-sample-time basis
        self.kd = kd / TIME_STEP_MINUTES  # Convert to per-sample-time basis
        
        # Sample time (minutes) - fixed from environment variable
        self.sample_time = TIME_STEP_MINUTES
        
        # Working variables
        self.input = 0.0
        self.output = 0.0
        self.setpoint = 0.0
        self.i_term = 0.0
        self.last_input = 0.0
        self.last_time = 0.0
        
        # Output limits from environment variables
        self.output_min = OUTPUT_MIN
        self.output_max = OUTPUT_MAX
        
        # Auto/Manual mode
        self.in_auto = True
        
        # For debugging/monitoring
        self.errors = []
        
    def compute(self, setpoint: float, measurement: float, current_time: float) -> float:
        """
        Compute PID output with Brett Beauregard's improvements
        
        Args:
            setpoint: Desired value
            measurement: Current measurement (Input)
            current_time: Current time (minutes)
            
        Returns:
            Control output value
        """
        # Store inputs
        self.input = measurement
        self.setpoint = setpoint
        
        # Only compute if in automatic mode
        if not self.in_auto:
            return self.output
        
        # Check if enough time has passed (sample time management)
        time_change = current_time - self.last_time
        if self.last_time > 0 and time_change < self.sample_time:
            return self.output  # Not time to compute yet
        
        # Error calculation
        error = setpoint - measurement
        
        # Store error for debugging
        self.errors.append(error)
        if len(self.errors) > 1000:  # Limit memory usage
            self.errors.pop(0)
        
        # Integral term (with anti-windup handled later)
        self.i_term += self.ki * error
        
        # Derivative term - "Derivative on Measurement" to avoid derivative kick
        # This prevents spikes when setpoint changes suddenly
        d_input = measurement - self.last_input
        
        # Compute preliminary output
        output = self.kp * error + self.i_term - self.kd * d_input
        
        # Apply output limits and anti-windup (improved method)
        if output > self.output_max:
            # Anti-windup: back-calculate integral term
            self.i_term -= output - self.output_max
            output = self.output_max
        elif output < self.output_min:
            # Anti-windup: back-calculate integral term  
            self.i_term += self.output_min - output
            output = self.output_min
        
        # Store outputs
        self.output = output
        
        # Remember some variables for next time
        self.last_input = measurement
        self.last_time = current_time
        
        return output
    
    def set_tunings(self, kp: float, ki: float, kd: float):
        """
        Update PID tuning parameters
        Handles conversion to sample-time basis
        """
        if kp < 0 or ki < 0 or kd < 0:
            return  # Don't allow negative tunings
        
        # Store display values
        self.dispKp = kp
        self.dispKi = ki
        self.dispKd = kd
        
        # Convert to working values based on sample time
        self.kp = kp
        self.ki = ki * self.sample_time
        self.kd = kd / self.sample_time
    
    def set_sample_time(self, new_sample_time: float):
        """
        Update sample time and adjust tuning parameters accordingly
        """
        if new_sample_time <= 0:
            return
        
        # Calculate ratio
        ratio = new_sample_time / self.sample_time
        
        # Adjust tuning parameters
        self.ki *= ratio
        self.kd /= ratio
        
        # Update sample time
        self.sample_time = new_sample_time
    
    def set_output_limits(self, min_val: float, max_val: float):
        """Set output limits"""
        if min_val >= max_val:
            return
        
        self.output_min = min_val
        self.output_max = max_val
        
        # Clamp current output if needed
        if self.output > self.output_max:
            self.output = self.output_max
        elif self.output < self.output_min:
            self.output = self.output_min
        
        # Clamp integral term if needed
        if self.i_term > self.output_max:
            self.i_term = self.output_max
        elif self.i_term < self.output_min:
            self.i_term = self.output_min
    
    def set_mode(self, mode: str):
        """
        Set controller mode: 'AUTO' or 'MANUAL'
        Handles proper initialization when switching to AUTO
        """
        new_auto = (mode.upper() == 'AUTO')
        
        # Check for transition from MANUAL to AUTO
        if new_auto and not self.in_auto:
            self.initialize()
        
        self.in_auto = new_auto
    
    def initialize(self):
        """
        Initialize controller for bumpless transfer from MANUAL to AUTO
        This is called automatically when switching to AUTO mode
        """
        self.i_term = self.output  # Set integral to current output
        self.last_input = self.input  # Prevent derivative spike
        
        # Ensure integral term is within limits
        if self.i_term > self.output_max:
            self.i_term = self.output_max
        elif self.i_term < self.output_min:
            self.i_term = self.output_min
    
    def reset(self):
        """
        Reset PID controller state
        Required method for compatibility with existing simulation
        """
        self.i_term = 0.0
        self.last_input = 0.0
        self.last_time = 0.0
        self.output = 0.0
        self.errors = []
        self.in_auto = True  # Default to automatic mode
    
    def get_mode(self) -> str:
        """Get current mode"""
        return "AUTO" if self.in_auto else "MANUAL"
    
    def get_tunings(self) -> tuple:
        """Get current tuning parameters (display values)"""
        return (self.dispKp, self.dispKi, self.dispKd)

# ============================================================================
# PID TUNING METHODS
# ============================================================================

class PIDTuner:
    """
    PID Controller Auto-Tuning Methods
    Implements Ziegler-Nichols, Cohen-Coon, and Relay Auto-Tuning
    """
    
    @staticmethod
    def ziegler_nichols_closed_loop(ku: float, tu: float, controller_type: str = "PID") -> tuple:
        """
        Ziegler-Nichols Closed-Loop (Ultimate Gain) Method
        
        Args:
            ku: Ultimate gain (critical gain where system oscillates)
            tu: Ultimate period (oscillation period at critical gain)
            controller_type: "P", "PI", "PD", or "PID"
            
        Returns:
            Tuple of (Kp, Ki, Kd) gains
        """
        if controller_type.upper() == "P":
            kp = 0.5 * ku
            ki = 0.0
            kd = 0.0
        elif controller_type.upper() == "PI":
            kp = 0.45 * ku
            ki = 1.2 * ku / tu
            kd = 0.0
        elif controller_type.upper() == "PD":
            kp = 0.8 * ku
            ki = 0.0
            kd = 0.1 * ku * tu
        elif controller_type.upper() == "PID":
            # Classic Ziegler-Nichols PID
            kp = 0.6 * ku
            ki = 1.2 * ku / tu  
            kd = 0.075 * ku * tu
        else:
            raise ValueError("Controller type must be P, PI, PD, or PID")
        
        return (kp, ki, kd)
    
    @staticmethod
    def ziegler_nichols_conservative(ku: float, tu: float) -> tuple:
        """
        Conservative Ziegler-Nichols tuning (less aggressive)
        
        Args:
            ku: Ultimate gain
            tu: Ultimate period
            
        Returns:
            Tuple of (Kp, Ki, Kd) gains
        """
        kp = 0.33 * ku  # More conservative than 0.6
        ki = 0.66 * ku / tu  # Less aggressive integral
        kd = 0.11 * ku * tu  # Slightly more derivative
        
        return (kp, ki, kd)
    
    @staticmethod
    def cohen_coon(K: float, L: float, T: float) -> tuple:
        """
        Cohen-Coon Tuning Method
        Based on open-loop step response parameters
        
        Args:
            K: Process steady-state gain (ΔOutput/ΔInput)
            L: Dead time (delay before response starts)
            T: Time constant (time to reach 63% of final value)
            
        Returns:
            Tuple of (Kp, Ki, Kd) gains
        """
        # Cohen-Coon PID formulas
        tau = L / T  # Dimensionless ratio
        
        # Proportional gain
        kp = (1.35 / K) * (T / L) * (1 + 0.18 * tau)
        
        # Integral time constant
        Ti = L * (2.5 - 2 * tau) / (1 - 0.39 * tau)
        ki = kp / Ti
        
        # Derivative time constant  
        Td = L * (0.37 - 0.37 * tau) / (1 - 0.81 * tau)
        kd = kp * Td
        
        return (kp, ki, kd)
    
    @staticmethod
    def calculate_ultimate_from_relay(relay_amplitude: float, oscillation_period: float, 
                                    temp_amplitude: float) -> tuple:
        """
        Calculate ultimate gain and period from relay test data
        
        Args:
            relay_amplitude: Amplitude of relay signal (W)
            oscillation_period: Period of oscillation (minutes)
            temp_amplitude: Amplitude of temperature oscillation (°C)
            
        Returns:
            tuple: (Ku, Tu) ultimate gain and period
        """
        import math
        
        # Calculate ultimate gain using relay method
        Ku = (4 * relay_amplitude) / (math.pi * temp_amplitude)
        Tu = oscillation_period
        
        return (Ku, Tu)
    
    @staticmethod
    def relay_auto_tune(pid_controller, setpoint: float, measurement_func, 
                       control_func, relay_amplitude: float = 10.0, 
                       max_cycles: int = 20, dt: float = 0.1) -> tuple:
        """
        Relay (Åström-Hägglund) Auto-Tuning Method
        
        Args:
            pid_controller: PID controller instance
            setpoint: Target setpoint for tuning
            measurement_func: Function that returns current measurement
            control_func: Function that applies control output
            relay_amplitude: Amplitude of relay output
            max_cycles: Maximum number of oscillation cycles
            dt: Time step for simulation
            
        Returns:
            Tuple of (Kp, Ki, Kd, Ku, Tu) where Ku and Tu are ultimate values
        """
        import numpy as np
        
        # Store original tuning
        original_tuning = pid_controller.get_tunings()
        
        # Set controller to manual mode for relay test
        pid_controller.set_mode('MANUAL')
        
        # Data collection arrays
        time_data = []
        measurement_data = []
        output_data = []
        
        # Relay control variables
        relay_state = 1  # Start with positive relay
        last_measurement = measurement_func()
        last_switch_time = 0
        switch_times = []
        switch_measurements = []
        
        print("🔄 Starting Relay Auto-Tuning...")
        print(f"   Relay amplitude: ±{relay_amplitude}")
        print(f"   Target setpoint: {setpoint}")
        
        current_time = 0
        cycles_completed = 0
        
        try:
            while cycles_completed < max_cycles and current_time < 300:  # Max 5 minutes
                # Get current measurement
                current_measurement = measurement_func()
                
                # Relay logic: switch when crossing setpoint
                if ((current_measurement > setpoint and relay_state == 1) or 
                    (current_measurement < setpoint and relay_state == -1)):
                    
                    # Record switch
                    switch_times.append(current_time)
                    switch_measurements.append(current_measurement)
                    
                    # Switch relay
                    relay_state *= -1
                    
                    print(f"   Switch {len(switch_times)}: t={current_time:.1f}s, "
                          f"T={current_measurement:.2f}°C, Relay={relay_state:+d}")
                
                # Apply relay output
                relay_output = relay_state * relay_amplitude
                pid_controller.output = relay_output
                control_func(relay_output)
                
                # Store data
                time_data.append(current_time)
                measurement_data.append(current_measurement)
                output_data.append(relay_output)
                
                # Count complete cycles (need at least 4 switches for 2 cycles)
                if len(switch_times) >= 4:
                    cycles_completed = len(switch_times) // 2
                
                current_time += dt
        
        except Exception as e:
            print(f"❌ Relay tuning failed: {e}")
            # Restore original tuning
            pid_controller.set_tunings(*original_tuning)
            pid_controller.set_mode('AUTO')
            return original_tuning + (0, 0)
        
        # Analyze results
        if len(switch_times) < 4:
            print("❌ Insufficient oscillations for tuning")
            pid_controller.set_tunings(*original_tuning)
            pid_controller.set_mode('AUTO')
            return original_tuning + (0, 0)
        
        # Calculate ultimate gain and period
        # Use last few complete cycles for better accuracy
        periods = []
        amplitudes = []
        
        for i in range(2, len(switch_times)-1, 2):  # Every other switch = one period
            period = switch_times[i+2] - switch_times[i]
            periods.append(period)
            
            # Calculate amplitude from measurement data in this period
            start_idx = int(switch_times[i] / dt)
            end_idx = int(switch_times[i+2] / dt)
            if end_idx < len(measurement_data):
                period_data = measurement_data[start_idx:end_idx]
                amplitude = (max(period_data) - min(period_data)) / 2
                amplitudes.append(amplitude)
        
        if not periods or not amplitudes:
            print("❌ Could not calculate oscillation parameters")
            pid_controller.set_tunings(*original_tuning)
            pid_controller.set_mode('AUTO')
            return original_tuning + (0, 0)
        
        # Average the last few periods and amplitudes
        Tu = np.mean(periods[-3:]) if len(periods) >= 3 else np.mean(periods)
        amplitude_avg = np.mean(amplitudes[-3:]) if len(amplitudes) >= 3 else np.mean(amplitudes)
        
        # Calculate ultimate gain using relay method
        # Ku = 4 * relay_amplitude / (π * amplitude)
        Ku = (4 * relay_amplitude) / (np.pi * amplitude_avg)
        
        print(f"✅ Relay tuning completed:")
        print(f"   Ultimate Period (Tu): {Tu:.2f} seconds")
        print(f"   Ultimate Gain (Ku): {Ku:.2f}")
        print(f"   Oscillation Amplitude: {amplitude_avg:.2f}°C")
        
        # Apply Ziegler-Nichols tuning based on ultimate values
        kp, ki, kd = PIDTuner.ziegler_nichols_closed_loop(Ku, Tu, "PID")
        
        print(f"   Calculated PID gains: Kp={kp:.1f}, Ki={ki:.3f}, Kd={kd:.1f}")
        
        # Apply new tuning and switch back to auto
        pid_controller.set_tunings(kp, ki, kd)
        pid_controller.set_mode('AUTO')
        
        return (kp, ki, kd, Ku, Tu)
    
    @staticmethod
    def process_reaction_curve(step_input: float, time_data: list, 
                              response_data: list) -> tuple:
        """
        Extract process parameters from step response data
        
        Args:
            step_input: Magnitude of step input applied
            time_data: List of time values
            response_data: List of response measurements
            
        Returns:
            Tuple of (K, L, T) - gain, dead time, time constant
        """
        import numpy as np
        
        time_array = np.array(time_data)
        response_array = np.array(response_data)
        
        # Find steady-state gain
        initial_value = response_array[0]
        final_value = response_array[-1]
        K = (final_value - initial_value) / step_input
        
        # Find dead time (when response starts)
        threshold = initial_value + 0.05 * (final_value - initial_value)
        dead_time_idx = np.where(response_array > threshold)[0]
        L = time_array[dead_time_idx[0]] if len(dead_time_idx) > 0 else 0
        
        # Find time constant (63% of final value)
        target_63 = initial_value + 0.63 * (final_value - initial_value)
        time_63_idx = np.where(response_array > target_63)[0]
        time_63 = time_array[time_63_idx[0]] if len(time_63_idx) > 0 else time_array[-1]
        T = time_63 - L
        
        return (K, L, T)

class ArrayBackgroundLoss:
    """Background heat loss from user-defined array"""
    
    def __init__(self, loss_array: np.ndarray, noise_std: float = None):
        """
        Initialize array-based background heat loss
        
        Args:
            loss_array: Array of background heat loss values (W)
            noise_std: Standard deviation of random noise (W)
        """
        self.loss_array = loss_array
        self.noise_std = noise_std if noise_std is not None else NOISE_STD
        
    def get_heat_loss(self, time_index: int) -> float:
        """
        Get background heat loss at specific time index
        
        Args:
            time_index: Current time step index
            
        Returns:
            Heat loss in Watts (always positive = heat removal)
        """
        # Get base loss from array (with bounds checking)
        if time_index < len(self.loss_array):
            base_loss = self.loss_array[time_index]
        else:
            base_loss = self.loss_array[-1]  # Use last value if beyond array
        
        # Add small random noise
        noise = np.random.normal(0, self.noise_std)
        
        # Ensure positive heat loss
        total_loss = base_loss + noise
        return max(total_loss, 100.0)  # Minimum 100W loss
    
    def reset(self):
        """Reset to initial state"""
        pass

class ThermalSimulation:
    """
    Room thermal simulation using Newton's Law of Cooling
    Simplified version without environmental conditions
    """
    
    def __init__(self, room_volume: float = None, 
                 cooling_coefficient: float = None,
                 thermal_capacity_per_m3: float = None,
                 heater_efficiency: float = None):
        """
        Initialize thermal simulation with environment variable defaults
        
        Args:
            room_volume: Room volume in m³
            cooling_coefficient: Newton's cooling coefficient (1/min)
            thermal_capacity_per_m3: Thermal capacity per m³ (J/K/m³)
            heater_efficiency: HVAC heater efficiency (0-1)
        """
        self.room_volume = room_volume if room_volume is not None else ROOM_VOLUME
        self.cooling_coefficient = cooling_coefficient if cooling_coefficient is not None else COOLING_COEFFICIENT
        thermal_cap_per_m3 = thermal_capacity_per_m3 if thermal_capacity_per_m3 is not None else THERMAL_CAPACITY_PER_M3
        self.thermal_capacity = self.room_volume * thermal_cap_per_m3
        self.heater_efficiency = heater_efficiency if heater_efficiency is not None else HEATER_EFFICIENCY
        
        # Current state
        self.current_temperature = 20.0  # Start at 20°C
        self.current_time = 0.0
        
        # Background loss model (will be set later)
        self.background_loss = None
        
        # Control lag simulation
        self.control_history = []
        self.lag_minutes = CONTROL_LAG_MINUTES
        
    def reset(self, initial_temperature: float = 20.0):
        """Reset simulation to initial state"""
        self.current_temperature = initial_temperature
        self.current_time = 0.0
        self.control_history = []
        if self.background_loss:
            self.background_loss.reset()
    
    def set_background_loss_array(self, loss_array: np.ndarray):
        """Set background heat loss array"""
        self.background_loss = ArrayBackgroundLoss(loss_array)
    
    def step(self, setpoint: float, ambient_temp: float,
             control_input: float, dt: float = None) -> float:
        """
        Perform one simulation time step
        
        Args:
            setpoint: Current setpoint temperature (°C)
            ambient_temp: Current ambient temperature (°C)
            control_input: HVAC control input from PID (W)
            dt: Time step (minutes)
            
        Returns:
            New room temperature (°C)
        """
        
        if dt is None:
            dt = TIME_STEP_MINUTES
        
        # Apply control lag
        self.control_history.append((self.current_time, control_input))
        if len(self.control_history) > int(self.lag_minutes / dt * 2):
            self.control_history.pop(0)
        
        # Get lagged control value
        lag_time = self.current_time - self.lag_minutes
        actual_control = self._get_lagged_control(lag_time, control_input)
        
        # Calculate heat flows (simplified - no solar or internal gains)
        Q_hvac = actual_control * self.heater_efficiency
        
        time_index = int(self.current_time / dt)
        Q_background_loss = -self.background_loss.get_heat_loss(time_index)
        
        # Total heat input
        Q_total = Q_hvac + Q_background_loss
        
        # Newton's Law of Cooling: dT/dt = -k(T - T_ambient) + Q/C
        dT_dt = (-self.cooling_coefficient * (self.current_temperature - ambient_temp) + 
                Q_total / self.thermal_capacity)
        
        # Update temperature and time
        self.current_temperature += dT_dt * dt
        self.current_time += dt
        
        return self.current_temperature
    
    def _get_lagged_control(self, lag_time: float, default_control: float) -> float:
        """Get control value with lag applied"""
        if not self.control_history or lag_time <= self.control_history[0][0]:
            return default_control
        
        # Linear interpolation
        for i in range(len(self.control_history) - 1):
            t1, c1 = self.control_history[i]
            t2, c2 = self.control_history[i + 1]
            
            if t1 <= lag_time <= t2:
                if t2 == t1:
                    return c1
                alpha = (lag_time - t1) / (t2 - t1)
                return c1 + alpha * (c2 - c1)
        
        return self.control_history[-1][1]
    
    def get_current_temperature(self) -> float:
        """Get current room temperature"""
        return self.current_temperature
    
    def get_current_time(self) -> float:
        """Get current simulation time"""
        return self.current_time

def create_discrete_profile(array: np.ndarray, total_time_steps: int) -> np.ndarray:
    """
    Create TRUE discrete profile - each array value held constant for equal time periods
    NO interpolation - creates step changes
    
    Args:
        array: Input array with discrete values
        total_time_steps: Total number of time steps needed
        
    Returns:
        Discrete step profile
    """
    if len(array) == 0:
        return np.zeros(total_time_steps)
    
    # Calculate how many time steps each value should be held
    steps_per_value = total_time_steps // len(array)
    remainder = total_time_steps % len(array)
    
    # Create discrete profile
    discrete_profile = []
    
    for i, value in enumerate(array):
        # Each value gets equal time, with remainder distributed to first values
        hold_steps = steps_per_value + (1 if i < remainder else 0)
        discrete_profile.extend([value] * hold_steps)
    
    return np.array(discrete_profile[:total_time_steps])

def run_simulation(setpoint_array: np.ndarray, 
                  ambient_array: np.ndarray,
                  background_loss_array: np.ndarray,
                  pid_controller: SimplePID,
                  duration_hours: float = None,
                  dt_minutes: float = None) -> Dict[str, np.ndarray]:
    """
    Run complete simulation with TRUE discrete setpoints
    
    Args:
        setpoint_array: Array of DISCRETE setpoint temperatures (°C)
        ambient_array: Array of ambient temperatures (°C)
        background_loss_array: Array of background heat loss values (W)
        pid_controller: PID controller instance
        duration_hours: Total simulation duration (hours)
        dt_minutes: Time step (minutes)
        
    Returns:
        Dictionary containing simulation results
    """
    
    # Use environment variables for defaults
    if duration_hours is None:
        duration_hours = DURATION_HOURS
    if dt_minutes is None:
        dt_minutes = TIME_STEP_MINUTES
    
    print(f"🔄 Running simulation: {pid_controller.name} PID")
    print(f"   Duration: {duration_hours} hours, Time step: {dt_minutes} minutes")
    
    # Time setup
    time_steps = int(duration_hours * 60 / dt_minutes)
    time_minutes = np.arange(0, duration_hours * 60, dt_minutes)
    time_hours = time_minutes / 60
    
    # Create TRUE discrete profiles (step changes, no interpolation)
    setpoints = create_discrete_profile(setpoint_array, time_steps)
    ambient_temps = create_discrete_profile(ambient_array, time_steps)
    background_losses_base = create_discrete_profile(background_loss_array, time_steps)
    
    print(f"   Created discrete profiles: {len(setpoint_array)} → {time_steps} points")
    print(f"   Discrete setpoint values: {np.unique(setpoints)} °C")
    print(f"   Steps per setpoint: ~{time_steps // len(setpoint_array)} time steps")
    
    # Initialize simulation and controller
    simulation = ThermalSimulation()
    simulation.set_background_loss_array(background_losses_base)
    simulation.reset()
    pid_controller.reset()
    
    # Storage arrays
    temperatures = np.zeros(time_steps)
    control_outputs = np.zeros(time_steps)
    background_losses = np.zeros(time_steps)
    
    temperatures[0] = simulation.get_current_temperature()
    
    # Main simulation loop - PID-Simulation interaction
    for i in range(1, time_steps):
        # Current conditions
        current_temp = simulation.get_current_temperature()
        current_setpoint = setpoints[i]  # TRUE DISCRETE value (step change)
        current_ambient = ambient_temps[i]
        current_time = time_minutes[i]
        
        # PID computes control output based on current temperature
        control_output = pid_controller.compute(current_setpoint, current_temp, current_time)
        control_outputs[i] = control_output
        
        # Get background loss for logging
        background_losses[i] = simulation.background_loss.get_heat_loss(i)
        
        # Simulation takes control output and returns new temperature
        new_temp = simulation.step(
            setpoint=current_setpoint,
            ambient_temp=current_ambient,
            control_input=control_output,
            dt=dt_minutes
        )
        
        temperatures[i] = new_temp
    
    print(f"   ✅ Simulation completed: {time_steps} time steps")
    
    return {
        'name': pid_controller.name,
        'time_minutes': time_minutes,
        'time_hours': time_hours,
        'temperature': temperatures,
        'setpoint': setpoints,
        'ambient': ambient_temps,
        'control': control_outputs,
        'background_loss': background_losses,
        'room_volume': simulation.room_volume,
        'thermal_capacity': simulation.thermal_capacity
    }

def run_simulation_with_ziegler_tuning(setpoint_array,
                                       ambient_array,
                                       background_loss_array,
                                       pid_controller,
                                       duration_hours=None,
                                       dt_minutes=None):
    """Perform ZN tuning, then run the full simulation and return results."""

    # --- Perform ZN tuning ---
    def short_test(pid, kp, hours=4.0):
        pid.reset()
        pid.set_tunings(kp, 0, 0)
        sim = ThermalSimulation()
        sim.set_background_loss_array(background_loss_array)  # ✅ so it's not None
        sim.reset()

        setpoint, ambient = 22.0, 20.0
        temps = []
        for t in np.arange(0, hours * 60, TIME_STEP_MINUTES):
            temp = sim.get_current_temperature()
            out = pid.compute(setpoint, temp, t)
            sim.step(setpoint, ambient, out, TIME_STEP_MINUTES)
            temps.append(sim.get_current_temperature())
        return np.array(temps)

    def detect_sustained(temps):
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(temps)
        if len(peaks) < 2:
            return False, None
        periods = np.diff(peaks) * TIME_STEP_MINUTES
        avg_period = np.mean(periods)
        return (np.std(periods) < 0.1 * avg_period), avg_period

    def find_ku_tu(pid):
        kp = 100.0
        while kp <= 10000:
            temps = short_test(pid, kp)
            sustained, Tu = detect_sustained(temps)
            print(f"[Tuning] Kp={kp}, sustained={sustained}, Tu={Tu}")
            if sustained:
                return kp, Tu
            kp += 100.0
        raise RuntimeError("Ku not found")

    def apply_zn(pid, Ku, Tu):
        Kp = 0.6 * Ku
        Ki = 1.2 * Ku / Tu
        Kd = 0.075 * Ku * Tu
        pid.set_tunings(Kp, Ki, Kd)
        print(f"[Tuning] Applied ZN: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")

    print("\n🚀 Running Simulation with Ziegler–Nichols Tuning...")
    print("🎯 Running Ziegler–Nichols tuning...")
    Ku, Tu = find_ku_tu(pid_controller)
    apply_zn(pid_controller, Ku, Tu)
    print("✅ Tuning complete\n")

    # --- Now run your normal simulation ---
    return run_simulation(
        setpoint_array,
        ambient_array,
        background_loss_array,
        pid_controller,
        duration_hours,
        dt_minutes
    )

def run_simulation_with_cohen_coon_tuning(setpoint_array,
                                          ambient_array,
                                          background_loss_array,
                                          pid_controller,
                                          duration_hours=None,
                                          dt_minutes=None):
    """Perform Cohen–Coon tuning, then run the full simulation and return results."""

    # --- Perform Cohen–Coon tuning ---
    def step_test(pid, hours=3.0):
        """Run short open-loop step test with background loss."""
        pid.reset()
        pid.set_tunings(0, 0, 0)  # Open loop (PID output fixed)
        sim = ThermalSimulation()
        sim.set_background_loss_array(background_loss_array)  # ✅ so it's not None
        sim.reset()

        setpoint, ambient = 20, 18
        control_output = OUTPUT_MAX # Use max output for clear step response
        temps, times = [], []

        for t in np.arange(0, hours * 60, TIME_STEP_MINUTES):
            sim.step(setpoint, ambient, control_output, TIME_STEP_MINUTES)
            temps.append(sim.get_current_temperature())
            times.append(t)

        return np.array(times), np.array(temps), control_output

    def estimate_parameters(times, temps, step_power):
        """Estimate K, tau, L from temperature curve."""
        delta_temp = temps[-1] - temps[0]
        K = delta_temp / step_power  # process gain

        steady_state_temp = temps[-1]
        target_63 = temps[0] + 0.632 * (steady_state_temp - temps[0])

        idx_tau = np.where(temps >= target_63)[0][0]
        tau_time = times[idx_tau]

        idx_L = np.where(temps > temps[0] + 0.05 * delta_temp)[0][0]
        L_time = times[idx_L]

        tau = (tau_time - L_time) / 60.0  # min → hr
        L = L_time / 60.0  # min → hr
        return K, tau, L

    def apply_cohen_coon(pid, K, tau, L):
        """Apply Cohen–Coon tuning formulas."""
        if L <= 0:
            raise ValueError("Invalid dead time (L) detected.")

        # PID form
        Kp = (1 / K) * ((4 / 3) * (tau / L) + 0.25)
        Ki_hr = Kp / tau
        Kd_hr = Kp * (L / 4.0)

        # Convert to per-minute units for your simulation
        Ki = Ki_hr / 60.0
        Kd = Kd_hr * 60.0

        pid.set_tunings(Kp, Ki, Kd)
        print(f"[Cohen–Coon] Applied: Kp={Kp:.4f}, Ki={Ki:.6f}, Kd={Kd:.4f}")

    print("\n🚀 Running Simulation with Cohen–Coon Tuning...")
    print("🎯 Running Cohen–Coon tuning...")
    times, temps, step_power = step_test(pid_controller)
    K, tau, L = estimate_parameters(times, temps, step_power)
    print(f"[Cohen–Coon] Estimated: K={K:.6f}, τ={tau:.3f} hr, L={L:.3f} hr")
    apply_cohen_coon(pid_controller, K, tau, L)
    print("✅ Cohen–Coon tuning complete\n")

    # --- Now run your normal simulation ---
    return run_simulation(
        setpoint_array,
        ambient_array,
        background_loss_array,
        pid_controller,
        duration_hours,
        dt_minutes
    )


import numpy as np
from typing import Dict

def mean_step_itae_paper(time_s, sp, pv):
    """
    Compute mean ITAE across all setpoint-change intervals
    using the paper's definition with arithmetic sequence normalization.
    """
    sp = np.array(sp)
    pv = np.array(pv)
    time_s = np.array(time_s)

    # Find indices where setpoint changes
    step_indices = np.where(np.diff(sp) != 0)[0] + 1
    if len(step_indices) == 0:
        return 0.0  # no step changes

    itae_values = []
    for i, idx in enumerate(step_indices):
        start_t = time_s[idx]
        # End time = next step change or end of simulation
        if i + 1 < len(step_indices):
            end_t = time_s[step_indices[i + 1]]
        else:
            end_t = time_s[-1]

        # Mask for this interval
        mask = (time_s >= start_t) & (time_s <= end_t)
        seg_time = time_s[mask] - start_t  # relative time
        seg_error = np.abs(sp[mask] - pv[mask])

        if len(seg_time) < 2:
            continue

        # Paper ITAE: integral of t * |error| over interval
        numerator = np.trapz(seg_time * seg_error, seg_time)

        # Arithmetic sequence normalization: sum of time points
        denominator = np.sum(seg_time)
        if denominator == 0:
            continue

        itae_val = numerator / denominator
        itae_values.append(itae_val)

    return np.mean(itae_values) if itae_values else 0.0


def analyze_results(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Analyze simulation results and calculate performance metrics."""
    temp = results['temperature']
    setpoint = results['setpoint']
    control = results['control']
    background_loss = results['background_loss']
    time_minutes = results['time_minutes']

    # Convert time to seconds
    time_s = np.array(time_minutes) * 60

    # Performance metrics
    error = temp - setpoint
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    std_dev = np.std(error)
    max_error = np.max(np.abs(error))

    # Comfort analysis
    comfort_violations = np.sum(np.abs(error) > 1.0)
    comfort_percent = (1 - comfort_violations / len(error)) * 100

    # Energy consumption (convert to kWh)
    dt_hours = (time_minutes[1] - time_minutes[0]) / 60
    energy = np.sum(np.abs(control)) * dt_hours / 1000

    # Background loss analysis
    avg_background_loss = np.mean(background_loss)
    total_background_loss = np.sum(background_loss) * dt_hours / 1000

    # Mean ITAE across setpoint-change intervals (paper method)
    itae_value = mean_step_itae_paper(time_s, setpoint, temp)

    return {
        'name': results['name'],
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'comfort_percent': comfort_percent,
        'energy': energy,
        'std_dev': std_dev,
        'avg_background_loss': avg_background_loss,
        'total_background_loss': total_background_loss,
        'itae': itae_value
    }



import matplotlib.ticker as ticker


def create_plots(all_results: Dict[str, Dict[str, np.ndarray]]):
    """Create plots with a separate ambient temperature subplot"""

    print("📊 Creating analysis plots...")

    try:
        fig, axes = plt.subplots(5, 1, figsize=(15, 18))  # Now 5 plots
        colors = ['blue', 'red', 'green', 'purple', 'orange']

        # Get reference data
        ref_results = list(all_results.values())[0]
        time_hours = ref_results['time_hours']

        # Tick settings
        major_tick_spacing = 1.0  # hours
        minor_tick_spacing = 0.5  # hours

        # --- Plot 1: Temperature tracking ---
        for i, (name, results) in enumerate(all_results.items()):
            axes[0].plot(time_hours, results['temperature'],
                         color=colors[i % len(colors)], linewidth=2,
                         label=f'{name} PID', alpha=0.8)

        axes[0].plot(time_hours, ref_results['setpoint'], 'k-',
                     linewidth=2, label='Setpoint (Discrete)', alpha=0.9, dashes=(1, 1))
        axes[0].set_ylabel('Temperature (°C)', fontsize=12)
        axes[0].set_title('Room Temperature vs Setpoint', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # --- Plot 2: Ambient temperature ---
        axes[1].plot(time_hours, ref_results['ambient'], color='orange',
                     linewidth=2, alpha=0.8, label='Ambient Temperature')
        axes[1].set_ylabel('Ambient (°C)', fontsize=12)
        axes[1].set_title('Ambient Temperature Profile', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # --- Plot 3: Control effort ---
        for i, (name, results) in enumerate(all_results.items()):
            axes[2].plot(time_hours, results['control'] / 1000,
                         color=colors[i % len(colors)], linewidth=2,
                         label=f'{name} PID', alpha=0.8)
        axes[2].set_ylabel('Heater Power (kW)', fontsize=12)
        axes[2].set_title('HVAC Control Effort', fontsize=14)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        # --- Plot 4: Background heat loss ---
        background_loss_kw = ref_results['background_loss'] / 1000
        axes[3].plot(time_hours, background_loss_kw, 'purple', linewidth=2,
                     label='Array Background Loss', alpha=0.8)
        mean_loss = np.mean(background_loss_kw)
        axes[3].axhline(y=mean_loss, color='purple', linestyle='--', alpha=0.6,
                        label=f'Mean: {mean_loss:.1f}kW')
        axes[3].set_ylabel('Heat Loss (kW)', fontsize=12)
        axes[3].set_title('Background Heat Loss Array', fontsize=14)
        axes[3].legend(fontsize=11)
        axes[3].grid(True, alpha=0.3)

        # --- Plot 5: Error analysis ---
        for i, (name, results) in enumerate(all_results.items()):
            error = results['temperature'] - results['setpoint']
            axes[4].plot(time_hours, error, color=colors[i % len(colors)],
                         linewidth=2, label=f'{name} Error', alpha=0.8)
        axes[4].axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Comfort Bounds')
        axes[4].axhline(y=-1.0, color='red', linestyle=':', alpha=0.5)
        axes[4].set_xlabel('Time (hours)', fontsize=12)
        axes[4].set_ylabel('Temp Error (°C)', fontsize=12)
        axes[4].set_title('Temperature Tracking Error', fontsize=14)
        axes[4].legend(fontsize=11)
        axes[4].grid(True, alpha=0.3)

        # --- Apply ticks to all plots ---
        for ax in axes:
            ax.set_xlim(0, 25)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
            ax.grid(which='minor', color='gray', alpha=0.2, linestyle='--')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error creating plots: {e}")


def print_results(performance_data: List[Dict[str, float]], room_info: Dict[str, float]):
    """Print comprehensive results including normalized ITAE and STD"""

    print("\n" + "=" * 125)
    print("ENVIRONMENT VARIABLE CONFIGURED ROOM THERMAL SIMULATION")
    print("TRUE DISCRETE SETPOINTS & ARRAY BACKGROUND LOSS FROM .ENV FILES")
    print("=" * 125)
    print(
        f"Room Volume: {room_info['volume']:.0f} m³, Thermal Capacity: {room_info['thermal_capacity'] / 1000:.0f} kJ/K")
    print("-" * 125)

    # Added STD column to header
    print(f"{'Controller':<22} {'MAE(°C)':<8} {'RMSE(°C)':<9} {'STD(°C)':<9} {'Max Err':<8} "
          f"{'Comfort%':<9} {'Energy(kWh)':<12} {'Avg Loss(kW)':<12} {'Loss(kWh)':<10} {'ITAE(norm)':<11}")
    print("-" * 125)

    for data in performance_data:
        print(f"{data['name']:<22} {data['mae']:<8.3f} {data['rmse']:<9.3f} {data['std_dev']:<9.3f} "
              f"{data['max_error']:<8.3f} {data['comfort_percent']:<9.1f} "
              f"{data['energy']:<12.2f} {data['avg_background_loss'] / 1000:<12.2f} "
              f"{data['total_background_loss']:<10.2f} {data['itae']:<11.3f}")

    print("=" * 125)

    # Analysis
    avg_loss = np.mean([data['avg_background_loss'] for data in performance_data])
    total_loss = np.mean([data['total_background_loss'] for data in performance_data])
    avg_energy = np.mean([data['energy'] for data in performance_data])

    print(f"\n🔥 ENVIRONMENT VARIABLE CONFIGURATION:")
    print(f"• Configuration: Loaded from .env file")
    print(f"• Arrays: Comma-separated values in environment variables")
    print(f"• Average background loss: {avg_loss / 1000:.2f} kW")
    print(f"• Total background loss energy: {total_loss:.2f} kWh")
    print(f"• Background vs HVAC energy: {total_loss / avg_energy * 100:.1f}%")
    print(f"• Flexible: Modify .env file to change all parameters")

    # Best performers
    best_accuracy = min(performance_data, key=lambda x: x['mae'])
    best_comfort = max(performance_data, key=lambda x: x['comfort_percent'])
    best_energy = min(performance_data, key=lambda x: x['energy'])
    best_itae = min(performance_data, key=lambda x: x['itae'])

    print(f"\n🏆 BEST PERFORMERS:")
    print(f"Most Accurate: {best_accuracy['name']} (MAE: {best_accuracy['mae']:.3f}°C)")
    print(f"Best Comfort: {best_comfort['name']} ({best_comfort['comfort_percent']:.1f}%)")
    print(f"Most Efficient: {best_energy['name']} ({best_energy['energy']:.2f} kWh)")
    print(f"Lowest ITAE: {best_itae['name']} (ITAE(norm): {best_itae['itae']:.3f})")


def print_env_config():
    """Print current environment variable configuration"""
    print(f"\n📋 CURRENT ENVIRONMENT CONFIGURATION:")
    print(f"   • Setpoint Array: {len(SETPOINT_ARRAY)} values: {SETPOINT_ARRAY}")
    print(f"   • Ambient Array: {len(AMBIENT_ARRAY)} values: {AMBIENT_ARRAY}")
    print(f"   • Background Loss Array: {len(BACKGROUND_LOSS_ARRAY)} values: {BACKGROUND_LOSS_ARRAY}")
    print(f"   • Duration: {DURATION_HOURS} hours")
    print(f"   • Time Step: {TIME_STEP_MINUTES} minutes")
    print(f"   • Room Volume: {ROOM_VOLUME} m³")
    print(f"   • Cooling Coefficient: {COOLING_COEFFICIENT}")
    print(f"   • Thermal Capacity: {THERMAL_CAPACITY_PER_M3} J/K/m³")
    print(f"   • Heater Efficiency: {HEATER_EFFICIENCY}")

def main():
    """Main simulation with environment variable configuration"""

    print("🏠 ENVIRONMENT VARIABLE CONFIGURED ROOM THERMAL SIMULATION")
    print("=" * 60)
    print("🎯 Configuration loaded from .env files")

    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  No .env file found! Creating one from .env.example...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from .env.example")
            print("💡 You can now modify .env to customize the simulation")
        else:
            print("❌ No .env.example file found either!")
            print("💡 Using hardcoded defaults")

    # Check packages
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from dotenv import load_dotenv
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        if 'dotenv' in str(e):
            print("💡 Install python-dotenv: pip install python-dotenv")
        return

    # Print current configuration
    print_env_config()

    # Create PID controllers for tuned runs
    controllers = [
        SimplePID(kp=STANDARD_KP, ki=STANDARD_KI, kd=STANDARD_KD, name="Ziegler–Nichols Tuned"),
        SimplePID(kp=STANDARD_KP, ki=STANDARD_KI, kd=STANDARD_KD, name="Cohen–Coon Tuned")
    ]

    print(f"\n🔥 Environment Variable Features:")
    print(f"   • All parameters loaded from .env file")
    print(f"   • TRUE discrete setpoints (step changes, no interpolation)")
    print(f"   • Comma-separated arrays in environment variables")
    print(f"   • Easy configuration without code changes")
    print(f"   • Simplified thermal model (Newton's Law of Cooling)")

    # Run simulations
    print(f"\n🚀 Running Environment-Configured Simulations:")
    all_results = {}
    performance_data = []

    for controller in controllers:
        try:
            if controller.name == "Ziegler–Nichols Tuned":
                print("\n🚀 Running Simulation with Ziegler–Nichols Tuning...")
                results = run_simulation_with_ziegler_tuning(
                    SETPOINT_ARRAY,
                    AMBIENT_ARRAY,
                    BACKGROUND_LOSS_ARRAY,
                    controller,
                    DURATION_HOURS,
                    TIME_STEP_MINUTES
                )
            elif controller.name == "Cohen–Coon Tuned":
                print("\n🚀 Running Simulation with Cohen–Coon Tuning...")
                results = run_simulation_with_cohen_coon_tuning(
                    SETPOINT_ARRAY,
                    AMBIENT_ARRAY,
                    BACKGROUND_LOSS_ARRAY,
                    controller,
                    DURATION_HOURS,
                    TIME_STEP_MINUTES
                )
            else:
                results = run_simulation(
                    SETPOINT_ARRAY,
                    AMBIENT_ARRAY,
                    BACKGROUND_LOSS_ARRAY,
                    controller,
                    DURATION_HOURS,
                    TIME_STEP_MINUTES
                )

            all_results[controller.name] = results
            performance_data.append(analyze_results(results))

        except Exception as e:
            print(f"   ❌ Error with {controller.name}: {e}")

    if not all_results:
        print("❌ No simulations completed!")
        return

    # Results analysis
    first_result = list(all_results.values())[0]
    room_info = {
        'volume': first_result['room_volume'],
        'thermal_capacity': first_result['thermal_capacity']
    }

    print_results(performance_data, room_info)
    create_plots(all_results)

    print(f"\n✅ Environment variable simulation completed successfully!")
    print(f"\n🎯 Key Features:")
    print(f"   • All configuration in .env file")
    print(f"   • No code changes needed for parameter adjustments")
    print(f"   • TRUE discrete setpoints (perfect step changes)")
    print(f"   • Comma-separated array format")
    print(f"   • Modular PID-Simulation interaction")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
