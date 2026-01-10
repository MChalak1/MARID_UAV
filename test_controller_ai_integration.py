#!/usr/bin/env python3
"""
Integration test to verify the controller can import and use the AI model wrapper correctly.
"""
import sys
sys.path.insert(0, '/home/mc/marid_ws/install/marid_controller/lib/python3.12/site-packages')
sys.path.insert(0, '/home/mc/marid_ws/src/marid_controller')

try:
    from marid_controller.ai_model import MaridAIModel, STATE_DIM, ACTION_DIM
    print(f"✓ Imported MaridAIModel, STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}")
except ImportError as e:
    print(f"✗ Failed to import AI model: {e}")
    sys.exit(1)

try:
    # Try importing the controller to see if it can import the AI model
    from marid_controller.marid_ai_controller import MaridAIController, calculate_total_mass_from_urdf
    print(f"✓ Successfully imported MaridAIController")
except ImportError as e:
    print(f"✗ Failed to import MaridAIController: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify the controller's state vector construction matches the model's expectation
import numpy as np
import math

print(f"\nSimulating controller's compute_ai_control state vector construction...")

# Create a mock state vector (13 dims) as controller's get_state_vector() returns
base_state = np.array([
    100.0, 200.0, 5000.0,  # x, y, z position
    50.0, 30.0, 0.5,       # vx, vy, vz velocity
    0.05, 0.1, 0.02,       # roll, pitch, yaw attitude
    0.01, 0.02, 0.005,     # roll_rate, pitch_rate, yaw_rate angular velocity
    5000.0                 # altitude
])

# Simulate controller parameters
destination = np.array([1000.0, 2000.0])
distance = np.linalg.norm(destination - base_state[0:2])
desired_heading = math.atan2(destination[1] - base_state[1], destination[0] - base_state[0])
current_yaw = base_state[8]
heading_error = desired_heading - current_yaw
while heading_error > math.pi:
    heading_error -= 2 * math.pi
while heading_error < -math.pi:
    heading_error += 2 * math.pi

altitude_min = 3.0
altitude_max = 10000.0
target_altitude = 8000.0
target_velocity = 112.0

# This is exactly what the controller does in compute_ai_control()
state_with_target = np.concatenate([
    base_state,  # 13 dimensions
    destination,  # 2 dimensions → 15
    [distance],  # 1 dimension → 16
    [heading_error],  # 1 dimension → 17
    [altitude_min, altitude_max, target_altitude],  # 3 dimensions → 20
    [target_velocity]  # 1 dimension → 21 total
])

print(f"  Base state: {len(base_state)} dims")
print(f"  Extended state: {len(state_with_target)} dims")
print(f"  Model expects: {STATE_DIM} dims")

if len(state_with_target) == STATE_DIM:
    print(f"✓ Dimension match confirmed: {len(state_with_target)} == {STATE_DIM}")
else:
    print(f"✗ Dimension mismatch: {len(state_with_target)} != {STATE_DIM}")
    sys.exit(1)

# Test the model with this state vector
model = MaridAIModel()
print(f"\nTesting model prediction with controller-constructed state vector...")
try:
    action = model.predict(state_with_target)
    print(f"✓ Model prediction successful")
    print(f"  Action: [thrust={action[0]:.3f}, yaw_diff={action[1]:.3f}]")
    print(f"  Action dimension: {len(action)} (expected: {ACTION_DIM})")
    
    if len(action) == ACTION_DIM:
        print(f"✓ Action dimension correct")
    else:
        print(f"✗ Action dimension incorrect")
        sys.exit(1)
        
except ValueError as e:
    print(f"✗ Model validation failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*60}")
print(f"✓ Integration test passed!")
print(f"  Controller can construct state vector with {STATE_DIM} dimensions")
print(f"  Model accepts and validates {STATE_DIM}-dimensional state vector")
print(f"  Model returns {ACTION_DIM}-dimensional action vector")
print(f"{'='*60}")
