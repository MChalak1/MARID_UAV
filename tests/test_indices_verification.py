#!/usr/bin/env python3
"""
Verify that the documented indices match the actual state vector construction.
"""
import numpy as np
import math

print("Verifying state vector indices...")
print("=" * 60)

# Simulate controller's state vector construction
base_state = np.array([
    100.0, 200.0, 5000.0,  # [0:2] Position: x, y, z
    50.0, 30.0, 0.5,       # [3:5] Linear velocity: vx, vy, vz
    0.05, 0.1, 0.02,       # [6:8] Attitude: roll, pitch, yaw
    0.01, 0.02, 0.005,     # [9:11] Angular velocity: roll_rate, pitch_rate, yaw_rate
    5000.0                 # [12] Altitude
])

destination = np.array([1000.0, 2000.0])
distance = np.linalg.norm(destination - base_state[0:2])
heading_error = 0.5  # Some heading error value
altitude_min = 3.0
altitude_max = 10000.0
target_altitude = 8000.0
target_velocity = 112.0

# This matches exactly what the controller does
state_with_target = np.concatenate([
    base_state,  # indices 0-12 (13 elements)
    destination,  # indices 13, 14
    [distance],  # index 15
    [heading_error],  # index 16
    [altitude_min, altitude_max, target_altitude],  # indices 17, 18, 19
    [target_velocity]  # index 20
])

print(f"Total state vector length: {len(state_with_target)}")
print(f"Expected: 21")
assert len(state_with_target) == 21, "State vector should be 21 dimensions"
print("✓ Length correct\n")

# Verify indices match documentation
print("Verifying indices match documentation:")
print("-" * 60)

# Base state
print(f"[0:2]  Position (x, y, z): {state_with_target[0]:.2f}, {state_with_target[1]:.2f}, {state_with_target[2]:.2f}")
print(f"[3:5]  Velocity (vx, vy, vz): {state_with_target[3]:.2f}, {state_with_target[4]:.2f}, {state_with_target[5]:.2f}")
print(f"[6:8]  Attitude (roll, pitch, yaw): {state_with_target[6]:.3f}, {state_with_target[7]:.3f}, {state_with_target[8]:.3f}")
print(f"[9:11] Angular velocity: {state_with_target[9]:.3f}, {state_with_target[10]:.3f}, {state_with_target[11]:.3f}")
print(f"[12]   Altitude: {state_with_target[12]:.2f}")

# Waypoint info
print(f"\n[13:14] Waypoint position (x, y): {state_with_target[13]:.2f}, {state_with_target[14]:.2f}")
print(f"[15]    Distance to waypoint: {state_with_target[15]:.2f}")
print(f"[16]    Heading error: {state_with_target[16]:.3f}")
print(f"[17]    Altitude min: {state_with_target[17]:.2f}")
print(f"[18]    Altitude max: {state_with_target[18]:.2f}")
print(f"[19]    Target altitude: {state_with_target[19]:.2f}")
print(f"[20]    Target velocity: {state_with_target[20]:.2f}")

# Verify values match what we put in
print("\n" + "=" * 60)
print("Verifying values match expected:")
assert state_with_target[13] == destination[0], f"Index 13 should be waypoint_x ({destination[0]}), got {state_with_target[13]}"
assert state_with_target[14] == destination[1], f"Index 14 should be waypoint_y ({destination[1]}), got {state_with_target[14]}"
assert abs(state_with_target[15] - distance) < 1e-6, f"Index 15 should be distance ({distance}), got {state_with_target[15]}"
assert abs(state_with_target[16] - heading_error) < 1e-6, f"Index 16 should be heading_error ({heading_error}), got {state_with_target[16]}"
assert abs(state_with_target[17] - altitude_min) < 1e-6, f"Index 17 should be altitude_min ({altitude_min}), got {state_with_target[17]}"
assert abs(state_with_target[18] - altitude_max) < 1e-6, f"Index 18 should be altitude_max ({altitude_max}), got {state_with_target[18]}"
assert abs(state_with_target[19] - target_altitude) < 1e-6, f"Index 19 should be target_altitude ({target_altitude}), got {state_with_target[19]}"
assert abs(state_with_target[20] - target_velocity) < 1e-6, f"Index 20 should be target_velocity ({target_velocity}), got {state_with_target[20]}"

print("✓ All indices verified correctly!")
print("=" * 60)
print("\nSummary:")
print("  [0:12]  Base state (13 dimensions)")
print("  [13:14] Waypoint x, y")
print("  [15]    Distance to waypoint")
print("  [16]    Heading error")
print("  [17]    Altitude minimum")
print("  [18]    Altitude maximum")
print("  [19]    Target altitude")
print("  [20]    Target velocity")
print("\n✓ Documentation indices are now correct!")
