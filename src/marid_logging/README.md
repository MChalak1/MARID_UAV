# marid_logging

MARID UAV sensor data logging to CSV for ML and analysis.

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

## Use with WSL workspace

Copy or symlink this package into `~/marid_ws/src/`, then:

```bash
cd ~/marid_ws
colcon build --packages-select marid_logging
source install/setup.bash
```
