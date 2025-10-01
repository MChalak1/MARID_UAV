MARID_UAV

Modular Aerospace Robotics & Integrated Design (MARID) is a research UAV framework developed in ROS 2 and Gazebo. This repository showcases simulation, modeling, and control of advanced UAV concepts using modular aerodynamic surfaces, configurable propulsion, and robotics integration.

✈️ Features

ROS 2 + Gazebo Simulation: Fully integrated environment for testing UAV dynamics and control.

URDF/Xacro Models: Parametric description of UAV geometry and joints.

Custom Controllers: ROS 2 nodes in Python/C++ for wing/actuator commands and thrust management.

Gazebo Worlds & Plugins: Wind, wrench, IMU, and thruster integration for realistic testing.

Trajectory & State Estimation: Odometry, IMU, and Kalman filter examples for UAV navigation.

📂 Repository Structure
MARID_UAV/
 ├── src/
 │    ├── marid_description/     # URDF/Xacro models / gazebo and RViz launch files
 │    ├── marid_controller/      # Control nodes
 ├── README.md
 ├── .gitignore

🚀 Getting Started
Prerequisites

ROS 2 Humble or Jazzy

Gazebo (Ignition or Fortress)

colcon build tools

Build & Run
# Clone repo
git clone https://github.com/yourname/MARID_UAV.git
cd MARID_UAV

# Build
colcon build

# Source
source install/setup.bash

# Launch simulation
ros2 launch marid_gazebo gazebo.launch.py

📖 Documentation

UAV architecture diagrams

Example controller configurations

Sample simulation runs

👉 For extended technical notes and ongoing research, see my aerospace blog
. https://mchalakaerospace.blogspot.com/2024/07/multi-axis-rotary-wing-integrated.html

⚠️ Disclaimer

This repository is intended for research and educational purposes only. Some elements of the MARID UAV system are under active patent protection. Sensitive propulsion and thermal management details are not included.
