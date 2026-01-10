#!/usr/bin/env python3
"""
Test script to verify state vector dimensions match between controller and AI model wrapper.
"""
import sys
import numpy as np
import math

# Add the workspace to the path
sys.path.insert(0, '/home/mc/marid_ws/install/marid_controller/lib/python3.12/site-packages')
sys.path.insert(0, '/home/mc/marid_ws/src/marid_controller')

try:
    from marid_controller.ai_model import MaridAIModel, STATE_DIM, ACTION_DIM
    print(f"✓ Successfully imported MaridAIModel")
    print(f"  STATE_DIM constant: {STATE_DIM}")
    print(f"  ACTION_DIM constant: {ACTION_DIM}")
except ImportError as e:
    print(f"✗ Failed to import MaridAIModel: {e}")
    sys.exit(1)

# Create a test instance
model = MaridAIModel()
print(f"✓ Created MaridAIModel instance (using placeholder model)")

# Simulate the controller's state vector construction
# Base state (13 dimensions)
base_state = np.array([
    0.0, 0.0, 5.0,      # x, y, z position
    10.0, 0.0, 0.0,     # vx, vy, vz velocity
    0.0, 0.1, 0.0,      # roll, pitch, yaw attitude
    0.0, 0.0, 0.05,     # roll_rate, pitch_rate, yaw_rate angular velocity
    5.0                  # altitude
])

# Waypoint info (8 dimensions)
destination = np.array([1000.0, 2000.0])  # waypoint x, y
distance = np.linalg.norm(destination - base_state[0:2])
heading_error = math.atan2(destination[1] - base_state[1], destination[0] - base_state[0]) - base_state[8]
# Normalize heading error
while heading_error > math.pi:
    heading_error -= 2 * math.pi
while heading_error < -math.pi:
    heading_error += 2 * math.pi

altitude_min = 3.0
altitude_max = 10000.0
target_altitude = 8000.0
target_velocity = 112.0

# Construct full state vector exactly as the controller does
state_with_target = np.concatenate([
    base_state,  # 13 dimensions
    destination,  # 2 dimensions → 15
    [distance],  # 1 dimension → 16
    [heading_error],  # 1 dimension → 17
    [altitude_min, altitude_max, target_altitude],  # 3 dimensions → 20
    [target_velocity]  # 1 dimension → 21 total
])

print(f"\nState vector construction test:")
print(f"  Base state dimension: {len(base_state)}")
print(f"  Full state dimension: {len(state_with_target)}")
print(f"  Expected dimension: {STATE_DIM}")

if len(state_with_target) == STATE_DIM:
    print(f"✓ State vector dimension matches: {len(state_with_target)} == {STATE_DIM}")
else:
    print(f"✗ State vector dimension mismatch: {len(state_with_target)} != {STATE_DIM}")
    sys.exit(1)

# Test the model's predict method
print(f"\nTesting model.predict() with {len(state_with_target)}-dimensional state vector...")
try:
    action = model.predict(state_with_target)
    print(f"✓ Model prediction succeeded")
    print(f"  Action dimension: {len(action)}")
    print(f"  Expected action dimension: {ACTION_DIM}")
    print(f"  Action values: [total_thrust={action[0]:.3f}, yaw_differential={action[1]:.3f}]")
    
    if len(action) == ACTION_DIM:
        print(f"✓ Action vector dimension matches: {len(action)} == {ACTION_DIM}")
    else:
        print(f"✗ Action vector dimension mismatch: {len(action)} != {ACTION_DIM}")
        sys.exit(1)
        
except ValueError as e:
    print(f"✗ Model prediction failed with ValueError (expected): {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Model prediction failed with unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with wrong dimension (should fail)
print(f"\nTesting validation with wrong dimension (should fail)...")
wrong_state = np.zeros(15)  # Wrong dimension
try:
    action = model.predict(wrong_state)
    print(f"✗ Validation failed: should have raised ValueError for dimension mismatch")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Validation works correctly: {e}")
except Exception as e:
    print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    sys.exit(1)

# Test with NaN values (should fail)
print(f"\nTesting validation with NaN values (should fail)...")
state_with_nan = state_with_target.copy()
state_with_nan[0] = np.nan
try:
    action = model.predict(state_with_nan)
    print(f"✗ Validation failed: should have raised ValueError for NaN values")
    sys.exit(1)
except ValueError as e:
    print(f"✓ Validation works correctly: {e}")
except Exception as e:
    print(f"✗ Unexpected error type: {type(e).__name__}: {e}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"✓ All tests passed! State vector dimensions are correctly aligned.")
print(f"{'='*60}")
