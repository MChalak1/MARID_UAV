# marid_logging

MARID UAV sensor data logging for ML training and analysis.

## When to use which logger

| Goal | Use |
|------|-----|
| **Pose prediction for EKF** (train model: IMU+altitude → pose) | `pose_estimator_logger` |
| **Control/guidance learning** (behavior cloning: state → thrust, yaw) | `marid_data_logger` |
| **IMU-only logging** (no sim, no ground truth; CSV for debugging) | `imu_logger` |

For pose-from-IMU training, `pose_estimator_logger` already includes IMU; `imu_logger` is only needed when you want standalone IMU CSV (e.g. on real hardware without ground truth).

---

## Nodes

### imu_logger

Subscribes to `sensor_msgs/Imu` and appends each message to a CSV file with columns:

`timestamp`, `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `qx`, `qy`, `qz`, `qw`

**Parameters:**

| Parameter     | Type   | Default       | Description                    |
|---------------|--------|---------------|--------------------------------|
| `imu_topic`   | string | `/imu`        | IMU topic to subscribe to     |
| `output_file` | string | `imu_log.csv` | Output CSV filename            |
| `output_dir`  | string | `""`          | Directory for output (optional)|

**Run (after `colcon build` and `source install/setup.bash`):**

```bash
ros2 run marid_logging imu_logger
```

With parameters:

```bash
ros2 run marid_logging imu_logger --ros-args \
  -p imu_topic:=/imu \
  -p output_file:=flight1_imu.csv \
  -p output_dir:=/tmp/marid_data
```

Stop with `Ctrl+C`; the file is closed cleanly.

### marid_data_logger

Collects state-action pairs for ML training (imitation learning / behavior cloning). Subscribes to odometry, IMU, barometer altitude, thrust, and yaw differential; logs to `.npz` files in chunks.

**Run (after `colcon build` and `source install/setup.bash`):**

```bash
ros2 run marid_logging marid_data_logger
```

Data is saved to `~/marid_ws/data/marid_flight_data_*.npz` by default. See `marid_data_logger.py` for parameters.

### pose_estimator_logger

Collects IMU + altitude (inputs) and ground-truth pose (target) for training a learned pose-from-IMU+altitude model to assist the EKF.

- **Input (X):** IMU orientation, angular velocity, linear acceleration, altitude (11-D)
- **Target (y_real):** [z, roll, pitch, yaw] from Gazebo (4-D; x,y not observable from IMU+altitude)

**Run:**

```bash
ros2 run marid_logging pose_estimator_logger
```

Data is saved to `~/marid_ws/data/marid_pose_imu_altitude_*.npz`. Requires Gazebo simulation with `gazebo_pose_to_odom` publishing `/gazebo/odom`.

## Use with WSL workspace

Copy or symlink this package into `~/marid_ws/src/`, then:

```bash
cd ~/marid_ws
colcon build --packages-select marid_logging
source install/setup.bash
```
