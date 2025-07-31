import time

# This line is important. The simulation file needs these values.
from scripts.room_thermal_model_pid import TIME_STEP_MINUTES, OUTPUT_MIN, OUTPUT_MAX

class PIDBeginner:
    """
    A simple, beginner-friendly PID Controller by Anvay Ajmera.
    Based on Brett Beauregard's guide, translated from C++.
    """
    def __init__(self, Kp, Ki, Kd, setpoint, sample_time=1.0, output_limits=(OUTPUT_MIN, OUTPUT_MAX)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.setpoint = setpoint
        self.sample_time = sample_time
        self.output_min, self.output_max = output_limits
        
        self.reset()

    def reset(self):
        """
        This method is required by the simulator to reset the controller's state.
        """
        self.ITerm = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.output = 0.0

    def compute(self, current_setpoint, current_temp, current_time):
        """
        This is the core method called by the simulator at each time step.
        It calculates the required heater power output.
        """
        
        time_change = current_time - self.last_time
        
        if time_change >= self.sample_time:
            error = current_setpoint - current_temp
            
            self.ITerm += (error * time_change)
            # Anti-windup clamping
            if self.ITerm > self.output_max:
                self.ITerm = self.output_max
            elif self.ITerm < self.output_min:
                self.ITerm = self.output_min
                
            delta_error = error - self.last_error
            if time_change > 0:
                d_term = self.Kd * (delta_error / time_change)
            else:
                d_term = 0.0

            output = (self.Kp * error) + (self.Ki * self.ITerm) + d_term
            self.output = max(self.output_min, min(self.output_max, output))
            
            self.last_error = error
            self.last_time = current_time

        return self.output