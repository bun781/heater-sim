"""
Advanced Room Thermal Simulation with PID Control
- TRUE discrete setpoint values (step changes, no interpolation)
- User-editable arrays directly in code
- Simplified simulation without environmental conditions
- Modular PID-Simulation interaction at each time step
"""

import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

# ============================================================================
# USER-EDITABLE ARRAYS - MODIFY THESE VALUES
# ============================================================================

# Discrete setpoint temperatures (°C) - TRUE step changes
SETPOINT_ARRAY = np.array([22.0, 22.0, 23.0, 25.0, 24.0, 24.0, 20.0, 20.0, 26.0, 26.0, 21.0, 21.0, 23.0])

# Ambient temperatures (°C)
AMBIENT_ARRAY = np.array([25.0, 23.0, 27.0, 22.0, 28.0, 24.0, 26.0, 25.0, 29.0, 21.0, 27.0, 23.0, 25.0])

# Background heat loss values (Watts)
BACKGROUND_LOSS_ARRAY = np.array([10000])

# ============================================================================

class SimplePID:
    """Simple PID Controller with anti-windup and derivative filtering"""
    
    def __init__(self, kp: float, ki: float, kd: float, name: str):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.name = name
        
        # Internal state
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        self.errors = []
        
        # Output limits
        self.output_min = 0.0
        self.output_max = 100000.0  # 100kW max
        
    def compute(self, setpoint: float, measurement: float, current_time: float) -> float:
        """Compute PID output with proper time handling"""
        error = setpoint - measurement
        
        # Calculate time step
        if self.previous_time == 0.0:
            dt = 0.5  # Default time step in minutes
        else:
            dt = current_time - self.previous_time
            dt = max(dt, 1e-6)  # Prevent division by zero
        
        # PID calculation
        P = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        I = self.ki * self.integral
        
        # Derivative term
        if dt > 0:
            D = self.kd * (error - self.previous_error) / dt
        else:
            D = 0.0
        
        # Total output
        output = P + I + D
        
        # Apply output limits and anti-windup
        if output > self.output_max:
            output = self.output_max
            # Anti-windup: reduce integral
            self.integral -= (P + I + D - self.output_max) / self.ki if self.ki != 0 else 0
        elif output < self.output_min:
            output = self.output_min
            # Anti-windup: reduce integral
            self.integral -= (P + I + D - self.output_min) / self.ki if self.ki != 0 else 0
        
        # Update state
        self.previous_error = error
        self.previous_time = current_time
        self.errors.append(error)
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = 0.0
        self.errors = []

class ArrayBackgroundLoss:
    """Background heat loss from user-defined array"""
    
    def __init__(self, loss_array: np.ndarray, noise_std: float = 50.0):
        """
        Initialize array-based background heat loss
        
        Args:
            loss_array: Array of background heat loss values (W)
            noise_std: Standard deviation of random noise (W)
        """
        self.loss_array = loss_array
        self.noise_std = noise_std
        
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
    
    def __init__(self, room_volume: float = 140.0, 
                 cooling_coefficient: float = 0.02,
                 thermal_capacity_per_m3: float = 1900.0,
                 heater_efficiency: float = 0.95):
        """
        Initialize thermal simulation
        
        Args:
            room_volume: Room volume in m³
            cooling_coefficient: Newton's cooling coefficient (1/min)
            thermal_capacity_per_m3: Thermal capacity per m³ (J/K/m³)
            heater_efficiency: HVAC heater efficiency (0-1)
        """
        self.room_volume = room_volume
        self.cooling_coefficient = cooling_coefficient
        self.thermal_capacity = room_volume * thermal_capacity_per_m3
        self.heater_efficiency = heater_efficiency
        
        # Current state
        self.current_temperature = -2  # Start at 20°C
        self.current_time = 0.0
        
        # Background loss model (will be set later)
        self.background_loss = None
        
        # Control lag simulation
        self.control_history = []
        self.lag_minutes = 0
        
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
             control_input: float, dt: float = 0.5) -> float:
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
                  duration_hours: float = 25.0,
                  dt_minutes: float = 0.5) -> Dict[str, np.ndarray]:
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

def analyze_results(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Analyze simulation results"""
    temp = results['temperature']
    setpoint = results['setpoint']
    control = results['control']
    background_loss = results['background_loss']
    
    # Performance metrics
    error = temp - setpoint
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Comfort analysis
    comfort_violations = np.sum(np.abs(error) > 1.0)
    comfort_percent = (1 - comfort_violations / len(error)) * 100
    
    # Energy consumption (convert to kWh)
    dt_hours = (results['time_minutes'][1] - results['time_minutes'][0]) / 60
    energy = np.sum(np.abs(control)) * dt_hours / 1000
    
    # Background loss analysis
    avg_background_loss = np.mean(background_loss)
    total_background_loss = np.sum(background_loss) * dt_hours / 1000
    
    return {
        'name': results['name'],
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'comfort_percent': comfort_percent,
        'energy': energy,
        'avg_background_loss': avg_background_loss,
        'total_background_loss': total_background_loss
    }

def create_plots(all_results: Dict[str, Dict[str, np.ndarray]]):
    """Create simplified plots without environmental conditions"""
    
    print("📊 Creating analysis plots...")
    
    try:
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        # Get reference data
        ref_results = list(all_results.values())[0]
        time_hours = ref_results['time_hours']
        
        # Plot 1: Temperature tracking with TRUE discrete setpoints
        for i, (name, results) in enumerate(all_results.items()):
            axes[0].plot(time_hours, results['temperature'], 
                        color=colors[i % len(colors)], linewidth=2, 
                        label=f'{name} PID', alpha=0.8)
        
        axes[0].plot(time_hours, ref_results['setpoint'], 'k-', 
                    linewidth=4, label='Discrete Setpoint (Step Changes)', alpha=0.9)
        axes[0].set_ylabel('Temperature (°C)', fontsize=12)
        axes[0].set_title('Room Temperature Control with TRUE Discrete Setpoints', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Control effort
        for i, (name, results) in enumerate(all_results.items()):
            axes[1].plot(time_hours, results['control'] / 1000,
                        color=colors[i % len(colors)], linewidth=2, 
                        label=f'{name} PID', alpha=0.8)
        
        axes[1].set_ylabel('Heater Power (kW)', fontsize=12)
        axes[1].set_title('HVAC Control Effort', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Background heat loss (from array)
        background_loss_kw = ref_results['background_loss'] / 1000
        axes[2].plot(time_hours, background_loss_kw, 'purple', linewidth=2, 
                    label='Array Background Loss', alpha=0.8)
        
        mean_loss = np.mean(background_loss_kw)
        axes[2].axhline(y=mean_loss, color='purple', linestyle='--', alpha=0.6, 
                       label=f'Mean: {mean_loss:.1f}kW')
        
        axes[2].set_ylabel('Heat Loss (kW)', fontsize=12)
        axes[2].set_title('User-Defined Background Heat Loss Array', fontsize=14)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Error analysis
        for i, (name, results) in enumerate(all_results.items()):
            error = results['temperature'] - results['setpoint']
            axes[3].plot(time_hours, error, color=colors[i % len(colors)], 
                        linewidth=2, label=f'{name} Error', alpha=0.8)
        
        axes[3].axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Comfort Bounds')
        axes[3].axhline(y=-1.0, color='red', linestyle=':', alpha=0.5)
        axes[3].set_xlabel('Time (hours)', fontsize=12)
        axes[3].set_ylabel('Temperature Error (°C)', fontsize=12)
        axes[3].set_title('Temperature Tracking Error', fontsize=14)
        axes[3].legend(fontsize=11)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")

def print_results(performance_data: List[Dict[str, float]], room_info: Dict[str, float]):
    """Print comprehensive results"""
    
    print("\n" + "="*100)
    print("SIMPLIFIED ROOM THERMAL SIMULATION - TRUE DISCRETE SETPOINTS & ARRAY BACKGROUND LOSS")
    print("MODULAR PID-SIMULATION INTERACTION WITH USER-EDITABLE ARRAYS")
    print("="*100)
    print(f"Room Volume: {room_info['volume']:.0f} m³, Thermal Capacity: {room_info['thermal_capacity']/1000:.0f} kJ/K")
    print("-"*100)
    print(f"{'Controller':<15} {'MAE(°C)':<8} {'RMSE(°C)':<9} {'Max Err':<8} {'Comfort%':<9} {'Energy(kWh)':<12} {'Avg Loss(kW)':<12} {'Loss(kWh)':<10}")
    print("-"*100)
    
    for data in performance_data:
        print(f"{data['name']:<15} {data['mae']:<8.3f} {data['rmse']:<9.3f} "
              f"{data['max_error']:<8.3f} {data['comfort_percent']:<9.1f} "
              f"{data['energy']:<12.2f} {data['avg_background_loss']/1000:<12.2f} "
              f"{data['total_background_loss']:<10.2f}")
    
    print("="*100)
    
    # Analysis
    avg_loss = np.mean([data['avg_background_loss'] for data in performance_data])
    total_loss = np.mean([data['total_background_loss'] for data in performance_data])
    avg_energy = np.mean([data['energy'] for data in performance_data])
    
    print(f"\n🔥 ARRAY-BASED BACKGROUND HEAT LOSS ANALYSIS:")
    print(f"• Heat Loss Model: User-defined array of loss values")
    print(f"• Independent: No relationship to temperature differences")
    print(f"• Average background loss: {avg_loss/1000:.2f} kW")
    print(f"• Total background loss energy: {total_loss:.2f} kWh")
    print(f"• Background vs HVAC energy: {total_loss/avg_energy*100:.1f}%")
    print(f"• Flexible: Any loss pattern can be specified")
    
    # Best performers
    best_accuracy = min(performance_data, key=lambda x: x['mae'])
    best_comfort = max(performance_data, key=lambda x: x['comfort_percent'])
    best_energy = min(performance_data, key=lambda x: x['energy'])
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"Most Accurate: {best_accuracy['name']} (MAE: {best_accuracy['mae']:.3f}°C)")
    print(f"Best Comfort: {best_comfort['name']} ({best_comfort['comfort_percent']:.1f}%)")
    print(f"Most Efficient: {best_energy['name']} ({best_energy['energy']:.2f} kWh)")

def main():
    """Main simulation with TRUE discrete setpoints and simplified model"""
    
    print("🏠 SIMPLIFIED ROOM THERMAL SIMULATION")
    print("="*50)
    print("🎯 TRUE Discrete Setpoints + Array Background Loss + Simplified Model")
    
    # Check packages
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return
    
    # Simulation parameters
    duration_hours = 25.0
    dt_minutes = 0.5
    
    print(f"\n⚙️ Simulation Configuration:")
    print(f"   • Duration: {duration_hours} hours")
    print(f"   • Time Step: {dt_minutes} minutes")
    print(f"   • Total Points: {int(duration_hours * 60 / dt_minutes)}")
    
    # Use the user-editable arrays defined at the top of the file
    print(f"\n📊 User-Editable Arrays (modify at top of file):")
    print(f"   • Setpoint Array: {len(SETPOINT_ARRAY)} values: {SETPOINT_ARRAY} °C")
    print(f"   • Ambient Array: {len(AMBIENT_ARRAY)} values: {AMBIENT_ARRAY} °C")
    print(f"   • Background Loss Array: {len(BACKGROUND_LOSS_ARRAY)} values: {BACKGROUND_LOSS_ARRAY} W")
    
    # Create PID controllers
    controllers = [
        SimplePID(kp=1500, ki=200, kd=400, name="Conservative"),
        SimplePID(kp=2500, ki=350, kd=700, name="Standard"),
        SimplePID(kp=3500, ki=500, kd=1000, name="Aggressive")
    ]
    
    print(f"\n🎛️ PID Controllers:")
    for controller in controllers:
        print(f"   • {controller.name}: Kp={controller.kp}, Ki={controller.ki}, Kd={controller.kd}")
    
    print(f"\n🔥 Simplified Model Features:")
    print(f"   • TRUE discrete setpoints (step changes, no interpolation)")
    print(f"   • User-editable arrays at top of file")
    print(f"   • No environmental conditions (solar, internal gains)")
    print(f"   • Pure Newton's Law of Cooling + HVAC + Background Loss")
    
    # Run simulations
    print(f"\n🚀 Running Modular PID-Simulation Interactions:")
    all_results = {}
    performance_data = []
    
    for controller in controllers:
        try:
            results = run_simulation(
                setpoint_array=SETPOINT_ARRAY,
                ambient_array=AMBIENT_ARRAY,
                background_loss_array=BACKGROUND_LOSS_ARRAY,
                pid_controller=controller,
                duration_hours=duration_hours,
                dt_minutes=dt_minutes
            )
            all_results[controller.name] = results
            performance = analyze_results(results)
            performance_data.append(performance)
            
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
    
    print(f"\n✅ Simplified simulation completed successfully!")
    print(f"\n🎯 Key Features:")
    print(f"   • TRUE discrete setpoints (perfect step changes)")
    print(f"   • User-editable arrays at top of file")
    print(f"   • Simplified thermal model (no environmental conditions)")
    print(f"   • Modular PID-Simulation interaction")
    print(f"   • Newton's Law of Cooling with control lag")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
