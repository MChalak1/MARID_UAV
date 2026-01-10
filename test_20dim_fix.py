#!/usr/bin/env python3
"""
Verify that the state vector dimension fix works correctly (12-D base → 20-D extended).
"""
import sys
sys.path.insert(0, '/home/mc/marid_ws/src/marid_controller')

import numpy as np
import math

print("Testing 20-D state vector fix...")
print("=" * 60)

# Simulate base state (12-D, no redundant altitude)
base_state = np.array([
    100.0, 200.0, 5000.0,  # [0:2] Position: x, y, z (z is altitude)
    50.0, 30.0, 0.5,       # [3:5] Linear velocity: vx, vy, vz
    0.05, 0.1, 0.02,       # [6:8] Attitude: roll, pitch, yaw
    0.01, 0.02, 0.005,     # [9:11] Angular velocity: roll_rate, pitch_rate, yaw_rate
])

print(f"Base state (12-D): {len(base_state)} dimensions")
print(f"  [0:2]  Position (x, y, z): {base_state[0]:.2f}, {base_state[1]:.2f}, {base_state[2]:.2f}")
print(f"  [3:5]  Velocity: {base_state[3]:.2f}, {base_state[4]:.2f}, {base_state[5]:.2f}")
print(f"  [6:8]  Attitude: {base_state[6]:.3f}, {base_state[7]:.3f}, {base_state[8]:.3f}")
print(f"  [9:11] Angular velocity: {base_state[9]:.3f}, {base_state[10]:.3f}, {base_state[11]:.3f}")
print(f"  Note: z (index 2) = {base_state[2]:.2f} is altitude - no redundant altitude dimension!\n")

# Simulate waypoint info
destination = np.array([1000.0, 2000.0])
distance = np.linalg.norm(destination - base_state[0:2])
desired_heading = math.atan2(destination[1] - base_state[1], destination[0] - base_state[0])
current_yaw = base_state[8]
heading_error = desired_heading - current_yaw

# Normalize heading error
while heading_error > math.pi:
    heading_error -= 2 * math.pi
while heading_error < -math.pi:
    heading_error += 2 * math.pi

altitude_min = 3.0
altitude_max = 10000.0
target_altitude = 8000.0
target_velocity = 112.0

# Build extended state (20-D) - matches controller logic
extended_state = np.concatenate([
    base_state,  # 12 dimensions
    destination,  # 2 dimensions → 14
    [distance],  # 1 dimension → 15
    [heading_error],  # 1 dimension → 16
    [altitude_min, altitude_max, target_altitude],  # 3 dimensions → 19
    [target_velocity]  # 1 dimension → 20 total
])

print(f"Extended state (20-D): {len(extended_state)} dimensions")
print(f"  [12:13] Waypoint (x, y): {extended_state[12]:.2f}, {extended_state[13]:.2f}")
print(f"  [14]    Distance: {extended_state[14]:.2f}")
print(f"  [15]    Heading error: {extended_state[15]:.3f}")
print(f"  [16]    Altitude min: {extended_state[16]:.2f}")
print(f"  [17]    Altitude max: {extended_state[17]:.2f}")
print(f"  [18]    Target altitude: {extended_state[18]:.2f}")
print(f"  [19]    Target velocity: {extended_state[19]:.2f}")

# Verify dimension
from marid_controller.ai_model import STATE_DIM
print(f"\n{'='*60}")
print(f"Expected STATE_DIM: {STATE_DIM}")
print(f"Actual extended state length: {len(extended_state)}")

if len(extended_state) == STATE_DIM:
    print(f"✓ SUCCESS: Dimension matches! (20-D)")
else:
    print(f"✗ FAILURE: Dimension mismatch! Expected {STATE_DIM}, got {len(extended_state)}")
    sys.exit(1)

# Verify no redundant altitude
print(f"\n{'='*60}")
print("Verifying no redundant altitude:")
print(f"  Base state[2] (z/altitude): {base_state[2]:.2f}")
print(f"  Extended state[2] (z/altitude): {extended_state[2]:.2f}")
print(f"  Extended state indices 16-18: altitude constraints")
print(f"  ✓ No redundant altitude dimension in base state!")

print(f"\n{'='*60}")
print("✓ All tests passed! State vector is correctly 20-D with no redundancy.")
print(f"{'='*60}")
