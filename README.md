# MARID UAV

**Modular Aerospace Robotics & Integrated Design (MARID)** is a research-oriented UAV framework developed in **ROS 2** and **Gazebo**.  
The project focuses on **modular aircraft architectures**, **physics-based simulation**, and **AI-assisted autonomous guidance** integrated with classical flight-control systems.

⚠️ **Status: Active Research & Development**  
Core simulation and control functionality is implemented and tested. Several subsystems (notably AI guidance and ML training workflows) are under active refinement.

---

## ✈️ Features

### Core Simulation
- **ROS 2 + Gazebo** integrated simulation environment
- **URDF/Xacro models** with parametric geometry, joints, and physical properties
- **Gazebo plugins** for Lift/Drag aerodynamics, IMU, GPS, barometer, and thrust application
- Modular vehicle configuration supporting experimentation with unconventional layouts

---

### Flight Control & Navigation
- **AI-Assisted Guidance Layer (Option A)**  
  Neural networks provide **high-level guidance targets**, while low-level actuation is handled by classical controllers
- **Classical Control Stack**
  - PID-based attitude control using aerodynamic surfaces
  - Thrust and speed regulation with mass-aware limits
- **Waypoint Navigation**
  - Local-frame and GPS-based waypoint tracking
- **Multi-Sensor Fusion**
  - Extended Kalman Filter (EKF) combining IMU, GPS, and barometric altitude
- **Cascaded Control Architecture**


AI / Guidance  →  PID Controllers  →  Actuators



This structure mirrors real aerospace flight stacks and ensures safety, interpretability, and robustness.

---

### Machine Learning Infrastructure
- **Guidance-Level Learning (Option A)**
- AI outputs *guidance commands* (e.g. desired heading rate and speed)
- No direct neural control of actuators
- **Pose-from-IMU+Altitude (EKF Augmentation)**
- Learned model predicts pose from IMU + barometer to assist state estimation
- **IMU-Based Position Estimation (Physics + ML)** *(In Development)*
- Hybrid approach combining physics-based double integration with ML corrections
- Learns to filter noise, compensate bias drift, and correct integration errors from acceleration sequences
- Enables GPS-denied navigation using only IMU data with known initial position
- Uses LSTM/GRU sequence models to process temporal acceleration patterns
- **Data Loggers**
- Pose estimator logger (IMU + altitude → pose for EKF training)
- IMU physics position logger *(In Development)* — acceleration sequences → position for GPS-denied navigation
- Data logger (state–guidance pairs for control learning)
- **Multi-sensor fusion (planned)** — LIDAR and camera inputs for improved state prediction and GPS-free operation
- **State Normalization**
- Mean/std normalization for stable neural-network training
- **Behavior Cloning Ready**
- Supervised imitation learning from PID-generated guidance
- **Safe Deployment**
- PID fallback and state validation during runtime inference

#### Training & deployment workflow (pose / GPS-free goal)
- **Phase 1 (4-D):** Train a model on diverse sim data (multiple flights and scenarios) to predict z, roll, pitch, yaw from IMU + altitude. Validate and save weights. Data from `pose_estimator_logger`.
- **Phase 2 (6-D):** Add x, y via sequence-based position (e.g. `imu_physics_position_logger`) or fused estimate; train on diverse sim data and save weights.
- **Phase 3 (planned):** Fuse LIDAR and camera (e.g. LIDAR odometry, visual odometry) with IMU and learned pose for drift correction and long-duration GPS-free navigation.
- **Goal:** A control architecture that can run without GPS by fusing IMU, barometer, learned pose, and (optionally) LIDAR and camera in the EKF.

---

### Safety & Monitoring
- **Failsafe Supervisor**
- State validation (NaN/Inf checks, dimension checks)
- Graceful fallback to classical control
- **Operational Safeguards**
- Altitude bounds
- Thrust and rate limits
- Emergency override logic (simulation)

---

## Control Architecture Options

MARID supports multiple control architectures as part of ongoing research and experimentation.  
Only **Option A** is used in the current primary workflow.

### Option A — AI-Assisted Guidance (Recommended)
- Neural networks operate at the **guidance level only**
- AI outputs high-level targets such as desired heading rate and speed
- Classical PID controllers handle all low-level actuation
- Ensures safety, interpretability, and stability
- Mirrors architectures used in real aerospace systems

This is the **default and recommended architecture** for development, testing, and machine learning.

### Option B — Direct Neural Control (Legacy / Experimental)
- Neural networks output actuator-level commands (e.g. thrust, yaw differential)
- Included for comparison and historical experimentation
- **Not used** in the current control stack
- Retained for research completeness only

### Control Architecture Overview

#### Option A (Primary — Guidance-Based)

```
[Sensors: IMU, baro, GPS, ...]
        ↓
[State Estimation]
 (EKF; optionally augmented by learned pose-from-IMU+altitude)
        ↓
[AI Guidance Layer]
 (heading rate, speed)
        ↓
[Classical Controllers]
 (PID / control laws)
        ↓
[Actuators]
 (thrust, surfaces)
```

#### Option B (Legacy — Direct Control)

```
[Sensors / EKF]
        ↓
[Neural Network]
 (thrust, yaw, surfaces)
        ↓
[Actuators]
```




## 📊 System Architecture

### ROS 2 Packages

- **`marid_description`**  
URDF/XACRO models, meshes, Gazebo worlds, `ros2_control` configuration

- **`marid_localization`**  
EKF nodes, sensor adapters, GPS-to-local transformation utilities

- **`marid_controller`**  
- AI guidance node (high-level targets only)
- Attitude controller (control surfaces)
- Thrust / speed controller
- Safety / failsafe supervisor

- **`marid_logging`**  
- Pose estimator logger (IMU + altitude → pose for EKF training)
- IMU physics position logger *(In Development)* — acceleration sequences → position
- Data logger (state–action pairs for control/guidance learning)
- IMU logger (raw IMU to CSV for debugging)

---

### Guidance State Vector

The AI guidance module operates on a **20-dimensional state vector**:

**Base State (12-D):**


[x, y, z,
vx, vy, vz,
roll, pitch, yaw,
roll_rate, pitch_rate, yaw_rate]



**Extended Guidance State (8-D):**


[waypoint_x,
waypoint_y,
distance_to_waypoint,
bearing_error,
altitude_min,
altitude_max,
target_altitude,
target_velocity]



> This state is intentionally **low-dimensional, interpretable, and physically meaningful**, making it suitable for supervised learning and future reinforcement learning.

Detailed rationale is provided in `OPTION_A_IMPLEMENTATION.md`.

---

## 🚀 Getting Started

### Prerequisites
- **ROS 2**: Humble or Jazzy
- **Gazebo**: Ignition Fortress or newer
- **colcon** build tools
- **Python 3.10+**
- Python dependencies: `numpy`, `tf_transformations`
- *(Optional for training)* PyTorch

---

### Build & Install

```bash
git clone https://github.com/MChalak1/MARID_UAV.git
cd MARID_UAV

colcon build
source install/setup.bash
````

---

### Launch Simulation

**Terminal 1 — Gazebo**

```bash
ros2 launch marid_description gazebo.launch.py
```

**Terminal 2 — Controllers & Localization**

```bash
ros2 launch marid_controller full_controller.launch.py
```

This launches:

* Controller manager & joint state broadcaster
* Localization EKFs
* Attitude controller
* Thrust / speed controller
* AI guidance node (with PID fallback)
* Safety supervisor

---

### Optional: Data Loggers for ML Training

| Goal | Command | Output |
|------|---------|--------|
| **Pose prediction for EKF** | `ros2 run marid_logging pose_estimator_logger` | `~/marid_ws/data/marid_pose_imu_altitude_*.npz` |
| **IMU position estimation** *(In Development)* | `ros2 run marid_logging imu_physics_position_logger` | `~/marid_ws/data/marid_imu_position_*.npz` |
| **Control/guidance learning** | `ros2 run marid_logging marid_data_logger` | `~/marid_ws/data/marid_flight_data_*.npz` |

Requires Gazebo + localization running. See `marid_logging/README.md` for when to use which logger.

---

## 📚 Documentation

### Physics Reference
- **[Physics Formulas & Equations Reference](PHYSICS_FORMULAS.md)**  
  Complete documentation of all physics formulas, aerodynamic equations, stability calculations, and simulation parameters. This document is **essential** for understanding:
  - Aerodynamic lift and drag calculations
  - Propulsion and thruster models
  - Stability analysis (aerodynamic center vs. center of gravity)
  - Control system equations
  - Sensor models and noise characteristics
  - CFD validation requirements

---

## 🤖 ML Training Workflows

### Pose-from-IMU+Altitude (EKF Augmentation)

Train a model to predict pose from IMU + altitude, to assist the EKF.

1. **Data collection:** Run `pose_estimator_logger` during simulation (Gazebo + localization).
2. **Train:** Learn `f(IMU, altitude) → pose` from `marid_pose_imu_altitude_*.npz`.
3. **Deploy:** Integrate model to augment EKF (future work).

---

### IMU-Based Position Estimation (Physics-Augmented Learning) *(In Development)*

Train a hybrid model that combines **physics-based integration** with **ML corrections** to estimate `[x, y, z]` position from IMU acceleration sequences.

**Approach:**
- **Physics Foundation:** Double-integrate acceleration → velocity → position
- **ML Augmentation:** Learn corrections for noise filtering, bias drift, and integration errors
- **Temporal Context:** Process acceleration sequences (50-200 timesteps) using LSTM/GRU

**Workflow:**

1. **Data collection:** *(Logger in development)* Log IMU acceleration sequences with ground-truth positions
   ```bash
   ros2 run marid_logging imu_physics_position_logger
   ```
   Planned output: `marid_imu_position_*.npz` with:
   - Input: `(sequence_length, 10)` — `[ax, ay, az, gx, gy, gz, qx, qy, qz, qw]` over time window
   - Target: `[x₀, y₀, z₀]` → `[x_final, y_final, z_final]` (initial + final positions)

2. **Preprocessing:** Transform body-frame acceleration to world frame using orientation quaternions, remove gravity

3. **Train:** Learn `f(accel_sequence, initial_pos) → final_pos` with physics-aware loss:
   ```python
   # Physics integration baseline
   physics_pred = integrate_acceleration(accel_sequence, dt)
   # ML corrections
   ml_correction = model(accel_sequence)
   # Combined prediction
   predicted_pos = initial_pos + physics_pred + ml_correction
   ```

4. **Deploy:** Real-time position estimation node for GPS-denied navigation (requires periodic corrections from GPS/landmarks)

**Benefits:**
- Interpretable (physics provides baseline)
- Data-efficient (leverages physics structure)
- Handles drift through learned corrections
- Enables navigation in GPS-denied environments

---

### Guidance-Level Learning (Option A — Behavior Cloning)

1. **Data collection:** Run full system with classical control; use `marid_data_logger` to log state → guidance targets.
2. **Compute normalization**

```python
from marid_controller.state_normalizer import StateNormalizer
import numpy as np

data = np.load("marid_flight_data_chunk0000.npz")
states = data["states"]      # (N, 20)
targets = data["actions"]    # (N, 2) — or guidance if logger updated for Option A

normalizer = StateNormalizer()
normalizer.fit(states)
normalizer.save("normalizer.json")
```

3. **Train:** Input normalized 20-D state → output 2-D guidance targets `(desired_heading_rate, desired_speed)`.
4. **Deploy:** Load model in guidance node; retain PID fallback.

---

## 📁 Repository Structure

```
MARID_UAV/
src/
├── marid_controller/
    ├── launch/
    ├── marid_controller/
    │   ├── marid_ai_guidance.py
    │   ├── marid_guidance_tracker.py
    │   ├── marid_ai_controller.py        (legacy Option B)
    │   ├── marid_attitude_controller.py
    │   ├── marid_thrust_controller.py
    │   ├── marid_safety_node.py
    │   ├── ai_model.py
    │   └── state_normalizer.py
    ├── package.xml
    ├── setup.py
    └── CMakeLists.txt
└── marid_logging/
    └── marid_logging/
        ├── imu_logger.py
        ├── marid_data_logger.py
        ├── pose_estimator_logger.py
        └── imu_physics_position_logger.py  (in development)

```

---

## 🔧 Known Limitations & Future Work

* **Thrust Application**

  * Currently implemented via Gazebo subprocess calls (performance bottleneck)
  * Planned: native transport or plugin-based force application
* **Temporal Context**

  * Current state is single-step
  * Future: stacked states or recurrent policies
* **Learning Scope**

  * Presently guidance-only (Option A) and pose-from-IMU+altitude (EKF augmentation)
  * **In Development:** IMU-based position estimation (physics + ML) for GPS-denied navigation
  * Future research may explore lower-level learning under strict safety constraints

---

## ⚠️ Disclaimer

This repository is intended for **research and educational use only**.
Some aspects of the MARID concept are under active patent protection; sensitive propulsion and thermal-management details are intentionally excluded.

This software is **simulation-only**. Thorough verification and validation are required before any physical deployment.

---

## Further Reading and Technical Background

Additional background, design rationale, and ongoing technical reflections related to the MARID project are documented externally:

- Technical blog: https://www.blogger.com/blog/post/edit/7941702355038936306/5673750805561614405

These materials provide supplementary context and are not required to understand or use the code in this repository.


## 📄 License

**Research License**
Use is permitted for academic and non-commercial research purposes only.
Redistribution or commercial use requires explicit permission from the author.

---

## 🙏 Acknowledgments

Built on **ROS 2**, **Gazebo**, and the broader open-source robotics, controls, and aerospace research community.

```
