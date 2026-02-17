# MARID Drone Test Suite

This directory contains regression tests and verification scripts for the MARID drone control system. These tests verify state vector dimensions, AI model integration, and Option A architecture implementation.

## Overview

The MARID drone uses a **20-dimensional state vector** (12-D base state + 8-D waypoint information) for AI-based guidance control. These tests ensure that:

- State vector dimensions are correct and consistent across all components
- No redundant altitude dimension exists in the base state
- AI model integration works correctly
- Option A architecture is properly implemented

## Test Files

### 1. `test_20dim_fix.py`
**Purpose:** Verifies that the state vector dimension fix works correctly (12-D base → 20-D extended, no redundant altitude).

**What it tests:**
- Base state is 12-D (position, velocity, attitude, angular velocity)
- Extended state is 20-D (base + waypoint info)
- No redundant altitude dimension in base state
- State vector construction matches `STATE_DIM` constant from `ai_model.py`

**Key assertions:**
- `len(base_state) == 12`
- `len(extended_state) == 20`
- `len(extended_state) == STATE_DIM`

**Usage:**
```bash
cd /home/mc/marid_ws/tests
python3 test_20dim_fix.py
```

---

### 2. `test_state_vector_dimensions.py`
**Purpose:** Verifies state vector dimensions match between controller and AI model wrapper.

**What it tests:**
- AI model can be imported successfully
- State vector construction matches model expectations
- Model validation works (rejects wrong dimensions, NaN values)
- Action vector dimensions are correct

**Key assertions:**
- State vector dimension matches `STATE_DIM`
- Action vector dimension matches `ACTION_DIM`
- Model rejects invalid inputs (wrong dimension, NaN values)

**Usage:**
```bash
cd /home/mc/marid_ws/tests
python3 test_state_vector_dimensions.py
```

---

### 3. `test_controller_ai_integration.py`
**Purpose:** Integration test to verify the controller can import and use the AI model wrapper correctly.

**What it tests:**
- `MaridAIModel` can be imported
- `MaridAIController` can be imported
- Controller's state vector construction matches model expectations
- Model accepts controller-constructed state vectors
- Model returns valid action vectors

**Key assertions:**
- Controller constructs 20-D (or 21-D) state vector
- Model accepts and validates the state vector
- Model returns 2-D action vector `[total_thrust, yaw_differential]`

**Note:** This test may use a 21-D state vector (with redundant altitude) if it's testing an older version of the controller.

**Usage:**
```bash
cd /home/mc/marid_ws/tests
python3 test_controller_ai_integration.py
```

---

### 4. `test_indices_verification.py`
**Purpose:** Verifies that documented state vector indices match the actual state vector construction.

**What it tests:**
- State vector index mapping is correct
- Each index contains the expected value
- Documentation matches implementation

**State vector structure (21-D version):**
- `[0:2]` Position (x, y, z)
- `[3:5]` Velocity (vx, vy, vz)
- `[6:8]` Attitude (roll, pitch, yaw)
- `[9:11]` Angular velocity (roll_rate, pitch_rate, yaw_rate)
- `[12]` Altitude (redundant - may be removed in current version)
- `[13:14]` Waypoint position (x, y)
- `[15]` Distance to waypoint
- `[16]` Heading error
- `[17]` Altitude minimum
- `[18]` Altitude maximum
- `[19]` Target altitude
- `[20]` Target velocity

**Note:** This test uses a 21-D state vector (with redundant altitude at index 12). The current implementation uses 20-D (no redundant altitude).

**Usage:**
```bash
cd /home/mc/marid_ws/tests
python3 test_indices_verification.py
```

---

### 5. `test_option_a_complete.py`
**Purpose:** Complete Option A Architecture Verification Test - tests all components to ensure Option A is correctly implemented.

**What it tests:**
1. **State Vector Dimensions (20-D)**
   - Base state is 12-D (no redundant altitude)
   - Extended state is 20-D

2. **AI Model Output (Guidance Targets)**
   - Model outputs 2-D guidance targets `[desired_heading_rate, desired_speed]`
   - Guidance targets are in reasonable ranges

3. **Guidance Node Structure**
   - File exists: `marid_ai_guidance.py`
   - Publishes to `/marid/guidance/desired_heading_rate`
   - Publishes to `/marid/guidance/desired_speed`
   - Does NOT publish actuator commands

4. **Guidance Tracker Structure**
   - File exists: `marid_guidance_tracker.py`
   - Subscribes to guidance topics
   - Publishes to `/marid/thrust/total` and `/marid/thrust/yaw_differential`
   - Only tracks guidance (does not compute guidance)

5. **Topic Name Consistency**
   - Guidance tracker and thrust controller use matching topic names

6. **State Normalizer**
   - Can fit, transform, and inverse transform 20-D states

**Usage:**
```bash
cd /home/mc/marid_ws/tests
python3 test_option_a_complete.py
```

---

## Running All Tests

To run all tests sequentially:

```bash
cd /home/mc/marid_ws/tests
for test in test_*.py; do
    echo "Running $test..."
    python3 "$test"
    echo ""
done
```

Or run them individually:

```bash
python3 test_20dim_fix.py
python3 test_state_vector_dimensions.py
python3 test_controller_ai_integration.py
python3 test_indices_verification.py
python3 test_option_a_complete.py
```

## State Vector Structure

### Current Implementation (20-D)

**Base State (12-D):**
- `[0:2]` Position: x, y, z (z is altitude - no redundancy!)
- `[3:5]` Linear velocity: vx, vy, vz
- `[6:8]` Attitude: roll, pitch, yaw
- `[9:11]` Angular velocity: roll_rate, pitch_rate, yaw_rate

**Extended State (20-D = 12 + 8):**
- Base state (12-D)
- `[12:13]` Waypoint position (x, y)
- `[14]` Distance to waypoint
- `[15]` Heading error
- `[16]` Altitude minimum
- `[17]` Altitude maximum
- `[18]` Target altitude
- `[19]` Target velocity

### Important Notes

⚠️ **Dimension Consistency:** The current implementation uses **20-D state vectors** (12-D base + 8-D waypoint info). Some older tests may reference 21-D vectors with redundant altitude - these should be updated to match the current implementation.

⚠️ **Path Dependencies:** These tests use absolute paths (`/home/mc/marid_ws/...`). If your workspace is located elsewhere, you may need to update the paths in the test files.

## Test History

These tests were created during the development of the MARID drone control system to:
- Fix state vector dimension mismatches (21-D → 20-D)
- Verify Option A architecture implementation
- Ensure AI model integration works correctly
- Document expected state vector structure

## Related Documentation

- **[Physics Formulas & Equations Reference](../PHYSICS_FORMULAS.md)** - Complete physics documentation
- **[Option A Architecture Verification](../OPTION_A_ARCHITECTURE_VERIFICATION.md)** - Architecture details
- **[Option A Implementation Summary](../OPTION_A_IMPLEMENTATION_SUMMARY.md)** - Implementation details

## Troubleshooting

**Import Errors:**
- Ensure ROS2 workspace is sourced: `source /home/mc/marid_ws/install/setup.bash`
- Check that `marid_controller` package is built: `colcon build --packages-select marid_controller`

**Dimension Mismatches:**
- Verify `STATE_DIM` constant in `src/marid_controller/marid_controller/ai_model.py`
- Check controller's state vector construction logic
- Ensure no redundant altitude dimension exists

**Path Issues:**
- Update absolute paths in test files if workspace location differs
- Ensure test files can access `src/marid_controller` directory

---

**Last Updated:** 2026-01-XX  
**Test Suite Version:** 1.0
