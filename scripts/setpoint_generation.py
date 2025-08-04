import random

def random_temp_array(length):
    possible_values = [x * 0.5 for x in range(48, 55)]  # 24.0 to 27.0
    return [random.choice(possible_values) for _ in range(length)]

# Example: generate 13 values
temps = random_temp_array(13)
print(temps)
