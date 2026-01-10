# Option A Implementation: High-Level AI Control

## Summary

Implemented critical fixes and ML infrastructure for Option A (high-level AI control). The system is now ready for behavior cloning (imitation learning) training.

## ✅ Critical Fixes Completed

### 1. **Fixed Altitude Redundancy Bug**
   - **Before**: Base state was 13-D with redundant altitude (`state[2]` = z from odom, `state[12]` = baro altitude, but only `state[2]` was used)
   - **After**: Base state is 12-D, `state[2]` represents altitude (prefers baro if available, otherwise odom z)
   - **Result**: Clean state vector, no confusion for ML training
   - **Files Changed**: 
     - `marid_ai_controller.py`: `get_state_vector()` now returns 12-D base state
     - Extended state vector: 20-D (12 base + 8 waypoint info)

### 2. **Updated State Vector Dimensions**
   - **Base state**: 12-D `[x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]`
   - **Extended state**: 20-D (base 12 + waypoint 8)
   - **Waypoint info**: `[waypoint_x, waypoint_y, distance, heading_error, alt_min, alt_max, target_alt, target_vel]`
   - **Files Changed**:
     - `ai_model.py`: Updated `STATE_DIM = 20`, corrected index documentation
     - `marid_ai_controller.py`: Updated extended state construction and validation

### 3. **Added State Normalization Module**
   - **File**: `state_normalizer.py`
   - **Features**:
     - Compute mean/std from training data
     - Normalize/denormalize state vectors
     - Save/load normalization parameters (JSON)
     - Critical for ML: neural networks need normalized inputs

### 4. **Created Data Logger Node**
   - **File**: `marid_data_logger.py`
   - **Features**:
     - Subscribes to state (odom, IMU, baro) and actions (thrust, yaw_diff from PID)
     - Logs to `.npz` files in chunks (configurable size)
     - Logs metadata (timestamps, control mode)
     - Configurable logging rate (default: 50 Hz)
   - **Usage**: Run alongside controller to collect training data

## 📊 State Vector Specification (20-D)

```
Base State (12 dimensions):
  [0:2]   Position: x, y, z (m) - Note: z is altitude
  [3:5]   Linear velocity: vx, vy, vz (m/s)
  [6:8]   Attitude: roll, pitch, yaw (rad)
  [9:11]  Angular velocity: roll_rate, pitch_rate, yaw_rate (rad/s)

Extended State (20 dimensions):
  [0:11]  Base state (12-D)
  [12:13] Waypoint position: waypoint_x, waypoint_y (m)
  [14]    Distance to waypoint (m)
  [15]    Heading error: desired_heading - current_yaw (rad, normalized [-pi, pi])
  [16]    Altitude minimum constraint (m)
  [17]    Altitude maximum constraint (m)
  [18]    Target altitude (m)
  [19]    Target velocity (m/s)
```

## 🚀 Next Steps for ML Training

### Phase 1: Data Collection (Current Phase)
1. **Launch system with data logger**:
   ```bash
   ros2 launch marid_controller full_controller.launch.py
   ros2 run marid_controller marid_data_logger.py
   ```

2. **Fly scenarios**:
   - Different waypoints
   - Different altitudes/speeds
   - ~30-60 minutes of flight data total

3. **Collect data**:
   - Data saved to `~/marid_ws/data/marid_flight_data_*.npz`
   - Each file contains `states` (N×20) and `actions` (N×2) arrays

### Phase 2: Behavior Cloning (Imitation Learning)
1. **Compute normalization**:
   ```python
   from marid_controller.state_normalizer import StateNormalizer
   import numpy as np
   
   # Load all data files
   data = np.load('marid_flight_data_chunk0000.npz')
   states = data['states']  # Shape: (N, 20)
   actions = data['actions']  # Shape: (N, 2)
   
   # Fit normalizer
   normalizer = StateNormalizer()
   normalizer.fit(states)
   normalizer.save('normalizer.json')
   ```

2. **Train PyTorch model**:
   - Use `MaridPolicyNetwork` from `ai_model.py` as template
   - Input: 20-D normalized state
   - Output: 2-D actions (thrust, yaw_diff) for now (will migrate to high-level)
   - Loss: MSE between model output and PID actions
   - Save model + normalizer together

3. **Test trained model**:
   - Load model and normalizer in controller
   - Use AI mode with PID fallback
   - Compare AI vs PID behavior

### Phase 3: Migrate to High-Level Commands (Future)
- Change action space from 2-D to 3-D:
  - `[desired_heading_rate, desired_speed, desired_altitude]`
- Update PID controllers to track high-level targets
- Retrain model with high-level supervision

## 🔧 Files Modified/Created

### Modified:
- `src/marid_controller/marid_controller/marid_ai_controller.py`
  - Fixed altitude redundancy (12-D base state)
  - Updated extended state to 20-D
  - Added validation for dimensions

- `src/marid_controller/marid_controller/ai_model.py`
  - Updated `STATE_DIM = 20`
  - Fixed index documentation
  - Added TODOs for high-level migration

- `src/marid_controller/CMakeLists.txt`
  - Added `marid_data_logger.py` to install list

### Created:
- `src/marid_controller/marid_controller/state_normalizer.py`
  - State normalization class
  - Save/load functionality

- `src/marid_controller/marid_controller/marid_data_logger.py`
  - Data logging node
  - Collects state-action pairs for training

## 🧪 Testing

All fixes verified:
- ✅ State vector is correctly 20-D
- ✅ No redundant altitude dimension
- ✅ All files compile without errors
- ✅ Dimension validation works correctly

Test script: `test_20dim_fix.py` (all tests pass)

## 📝 Remaining Issues (From AI Review)

### Not Yet Fixed (Lower Priority):
1. **Thrust controller performance**: Still uses `subprocess.run("gz topic ...")` - should migrate to native transport
2. **Action interface scope**: Currently outputs low-level (thrust, yaw_diff), will migrate to high-level
3. **Time history**: State vector doesn't include previous states/actions (can add later)
4. **Safety node**: Contains "self-destruct" logic - should refactor to failsafe-only

### Architecture Decisions:
- **Option A chosen**: AI outputs high-level commands, PID handles low-level
- **Current implementation**: Still outputs low-level (for backward compatibility)
- **Migration path**: Will change to 3-D high-level actions after initial training

## 🎯 Success Criteria

- [x] State vector is clean (no redundancy)
- [x] Dimensions match between controller and model
- [x] Normalization module ready
- [x] Data logger ready
- [ ] Training script created (next step)
- [ ] First behavior-cloned model trained
- [ ] Migration to high-level commands

## 📚 References

- Original AI review highlighted critical issues
- Option A architecture: AI → high-level targets → PID → low-level actuation
- Behavior cloning approach: Supervised learning from PID demonstrations
- Safe RL future: Use cloned policy as initialization with PID safety shield
