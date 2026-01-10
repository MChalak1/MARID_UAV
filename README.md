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
- **Data Logger**
- Records state–guidance pairs during flight
- **State Normalization**
- Mean/std normalization for stable neural-network training
- **Behavior Cloning Ready**
- Supervised imitation learning from PID-generated guidance
- **Safe Deployment**
- PID fallback and state validation during runtime inference

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

Option A (Primary – Guidance-Based)
----------------------------------
[ Sensors / EKF ]
        |
        v
[ AI Guidance Layer ]
 (heading rate, speed)
        |
        v
[ Classical Controllers ]
 (PID / control laws)
        |
        v
[ Actuators ]
 (thrust, surfaces)


Option B (Legacy – Direct Control)
---------------------------------
[ Sensors / EKF ]
        |
        v
[ Neural Network ]
 (thrust, yaw, surfaces)
        |
        v
[ Actuators ]



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
- Data logger for ML training
- Safety / failsafe supervisor

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

### Optional: Data Logger for ML Training

```bash
ros2 run marid_controller marid_data_logger
```

Data is saved to:

```
~/marid_ws/data/marid_flight_data_*.npz
```

---

## 🤖 ML Training Workflow (Option A)

### Phase 1 — Data Collection

1. Run the full system with classical control enabled
2. Fly multiple waypoint scenarios
3. Log state → guidance targets
4. Collect ~30–60 minutes of flight data

---

### Phase 2 — Behavior Cloning (Supervised Learning)

1. **Compute Normalization**

```python
from marid_controller.state_normalizer import StateNormalizer
import numpy as np

data = np.load("marid_flight_data_chunk0000.npz")
states = data["states"]      # (N, 20)
targets = data["guidance"]   # (N, 2)

normalizer = StateNormalizer()
normalizer.fit(states)
normalizer.save("normalizer.json")
```

2. **Train PyTorch Model**

* Input: normalized 20-D state
* Output: 2-D guidance targets
  `(desired_heading_rate, desired_speed)`
* Loss: Mean-Squared Error (MSE)

3. **Deploy**

* Load model + normalizer in guidance node
* Enable AI guidance mode
* Retain PID fallback for safety

---

## 📁 Repository Structure

```
MARID_UAV/
src/
└── marid_controller/
    ├── launch/
    ├── marid_controller/
    │   ├── marid_ai_guidance.py
    │   ├── marid_guidance_tracker.py
    │   ├── marid_ai_controller.py        (legacy Option B)
    │   ├── marid_attitude_controller.py
    │   ├── marid_thrust_controller.py
    │   ├── marid_data_logger.py
    │   ├── marid_safety_node.py
    │   ├── ai_model.py
    │   └── state_normalizer.py
    ├── package.xml
    ├── setup.py
    └── CMakeLists.txt

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

  * Presently guidance-only (Option A)
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
