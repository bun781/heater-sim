import random
# Courtesy of ChatGPT

# --- Simulation settings ---
duration_hours = 25
dt_minutes = 5
time_steps = int(duration_hours * 60 / dt_minutes)  # e.g. 300 elements

# --- Event parameters ---
students_power = 2000      # W
class_duration = 70        # min
class_steps = class_duration // dt_minutes  # 14 steps

leak_power = 5000         # W (negative for heat loss)
leak_duration = 20         # min
leak_steps = leak_duration // dt_minutes    # 4 steps at 5-min resolution

windows_per_class = 3      # number of windows per class
window_power = 50000      # W
window_duration = 4        # min
window_steps = max(1, window_duration // dt_minutes)  # 1 step at 5-min resolution

# --- Generate for class counts 1 to 10 ---
for num_classes in range(1, 11):
    background_loss_array = [0] * time_steps

    for _ in range(num_classes):
        # --- Randomly place class ---
        while True:
            start_idx = random.randint(0, time_steps - class_steps)
            if all(background_loss_array[i] == 0 for i in range(start_idx, start_idx + class_steps)):
                break

        # --- Students for the whole class ---
        for i in range(start_idx, start_idx + class_steps):
            background_loss_array[i] += students_power

        # --- Windows inside the class ---
        possible_indices = list(range(start_idx, start_idx + class_steps - window_steps))
        window_times = random.sample(possible_indices, windows_per_class)
        for w_start in window_times:
            for i in range(w_start, w_start + window_steps):
                background_loss_array[i] += window_power

        # --- Leak inside the class ---
        possible_leak_indices = list(range(start_idx, start_idx + class_steps - leak_steps))
        leak_start = random.choice(possible_leak_indices)
        for i in range(leak_start, leak_start + leak_steps):
            background_loss_array[i] += leak_power

    # Convert to comma-separated string
    env_string = ",".join(str(v) for v in background_loss_array)
    print(f"BACKGROUND_LOSS_ARRAY = {env_string}")
