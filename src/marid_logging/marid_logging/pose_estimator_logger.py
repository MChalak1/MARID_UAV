#!/usr/bin/env python3
"""
MARID Pose Estimator Data Logger
Collects IMU + altitude (inputs) and ground-truth pose (target) for training
a learned pose-from-IMU+altitude model to assist the EKF.

Input features (X):  IMU orientation, angular velocity, linear acceleration, altitude
Target (y_real):     [z, roll, pitch, yaw] from Gazebo (4-D; x,y not observable from IMU+altitude)

Subscribes to:
    - /imu_ekf          (sensor_msgs/Imu)
    - /barometer/altitude (geometry_msgs/PoseWithCovarianceStamped, z only)
    - /gazebo/odom      (nav_msgs/Odometry, ground truth)

Saves to .npz files for training: f(IMU, altitude) -> pose
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import math


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to roll, pitch, yaw (radians)."""
    from tf_transformations import euler_from_quaternion
    roll, pitch, yaw = euler_from_quaternion([x, y, z, w])
    return roll, pitch, yaw


class PoseEstimatorLogger(Node):
    """
    Data logger for pose-from-IMU+altitude training.
    
    Logs:
        X: [qx, qy, qz, qw, gx, gy, gz, ax, ay, az, altitude]  (11-D input)
        y: [z, roll, pitch, yaw]  (4-D; x,y not observable from IMU+altitude)
    """
    
    def __init__(self):
        super().__init__('pose_estimator_logger')
        
        self.declare_parameter('imu_topic', '/imu_ekf')
        self.declare_parameter('altitude_topic', '/barometer/altitude')
        self.declare_parameter('ground_truth_topic', '/gazebo/odom')
        self.declare_parameter('log_directory', '~/marid_ws/data')
        self.declare_parameter('log_filename_prefix', 'marid_pose_imu_altitude')
        self.declare_parameter('samples_per_file', 10000)
        self.declare_parameter('log_rate', 50.0)
        self.declare_parameter('enable_logging', True)
        
        self.imu_topic_ = self.get_parameter('imu_topic').value
        self.altitude_topic_ = self.get_parameter('altitude_topic').value
        self.gt_topic_ = self.get_parameter('ground_truth_topic').value
        log_dir = Path(self.get_parameter('log_directory').value).expanduser()
        self.log_dir_ = log_dir
        self.log_dir_.mkdir(parents=True, exist_ok=True)
        self.filename_prefix_ = self.get_parameter('log_filename_prefix').value
        self.samples_per_file_ = self.get_parameter('samples_per_file').value
        log_rate = self.get_parameter('log_rate').value
        self.enable_logging_ = self.get_parameter('enable_logging').value
        
        # Buffers
        self.imu_inputs_ = []   # (N, 11): [qx,qy,qz,qw, gx,gy,gz, ax,ay,az, altitude]
        self.pose_targets_ = [] # (N, 4): [z, roll,pitch,yaw]
        
        # Latest messages (for nearest-neighbor sync)
        self.last_imu_ = None
        self.last_altitude_ = None
        self.last_odom_ = None
        
        self.file_counter_ = 0
        self.samples_in_current_file_ = 0
        self.total_samples_logged_ = 0
        self.start_time_ = time.time()
        
        # Subscriptions
        self.imu_sub_ = self.create_subscription(
            Imu, self.imu_topic_, self.imu_callback, 10
        )
        self.altitude_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            self.altitude_topic_,
            self.altitude_callback,
            10
        )
        self.odom_sub_ = self.create_subscription(
            Odometry, self.gt_topic_, self.odom_callback, 10
        )
        
        self.log_timer_ = self.create_timer(1.0 / log_rate, self.log_data)
        
        self.get_logger().info('Pose Estimator Logger initialized')
        self.get_logger().info(f'  IMU: {self.imu_topic_}')
        self.get_logger().info(f'  Altitude: {self.altitude_topic_}')
        self.get_logger().info(f'  Ground truth: {self.gt_topic_}')
        self.get_logger().info(f'  Log directory: {self.log_dir_}')
        self.get_logger().info(f'  Log rate: {log_rate} Hz')
    
    def imu_callback(self, msg):
        self.last_imu_ = msg
    
    def altitude_callback(self, msg):
        self.last_altitude_ = msg.pose.pose.position.z
    
    def odom_callback(self, msg):
        self.last_odom_ = msg
    
    def _build_input(self):
        """Build 11-D input: [qx,qy,qz,qw, gx,gy,gz, ax,ay,az, altitude]"""
        if self.last_imu_ is None:
            return None
        ori = self.last_imu_.orientation
        gyro = self.last_imu_.angular_velocity
        acc = self.last_imu_.linear_acceleration
        alt = self.last_altitude_ if self.last_altitude_ is not None else 0.0
        if (math.isnan(alt) or math.isinf(alt)):
            alt = 0.0
        return np.array([
            ori.x, ori.y, ori.z, ori.w,
            gyro.x, gyro.y, gyro.z,
            acc.x, acc.y, acc.z,
            alt
        ], dtype=np.float32)
    
    def _build_target(self):
        """Build 4-D target: [z, roll, pitch, yaw] (x,y not observable from IMU+altitude)"""
        if self.last_odom_ is None:
            return None
        pos = self.last_odom_.pose.pose.position
        ori = self.last_odom_.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)
        return np.array([
            pos.z,
            roll, pitch, yaw
        ], dtype=np.float32)
    
    def log_data(self):
        if not self.enable_logging_:
            return
        if self.last_imu_ is None or self.last_odom_ is None:
            return
        inp = self._build_input()
        tgt = self._build_target()
        if inp is None or tgt is None:
            return
        if np.any(np.isnan(inp)) or np.any(np.isnan(tgt)):
            return
        if np.any(np.isinf(inp)) or np.any(np.isinf(tgt)):
            return
        self.imu_inputs_.append(inp)
        self.pose_targets_.append(tgt)
        self.samples_in_current_file_ += 1
        self.total_samples_logged_ += 1
        if self.samples_in_current_file_ >= self.samples_per_file_:
            self._save_chunk()
    
    def _save_chunk(self):
        if len(self.imu_inputs_) == 0:
            return
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'{self.filename_prefix_}_{ts}_chunk{self.file_counter_:04d}.npz'
        fpath = self.log_dir_ / fname
        inputs = np.array(self.imu_inputs_, dtype=np.float32)
        targets = np.array(self.pose_targets_, dtype=np.float32)
        np.savez_compressed(
            fpath,
            imu_inputs=inputs,
            pose_targets=targets,
            input_dim=11,
            target_dim=4,
            num_samples=len(self.imu_inputs_)
        )
        elapsed = time.time() - self.start_time_
        rate = self.total_samples_logged_ / elapsed if elapsed > 0 else 0
        self.get_logger().info(
            f'Saved {fname} ({len(self.imu_inputs_)} samples, '
            f'total: {self.total_samples_logged_}, rate: {rate:.1f} Hz)'
        )
        self.imu_inputs_ = []
        self.pose_targets_ = []
        self.file_counter_ += 1
        self.samples_in_current_file_ = 0
    
    def shutdown_callback(self):
        if len(self.imu_inputs_) > 0:
            self.get_logger().info('Saving remaining data on shutdown...')
            self._save_chunk()
        elapsed = time.time() - self.start_time_
        self.get_logger().info(
            f'Pose Estimator Logger shutdown. Total samples: {self.total_samples_logged_}, '
            f'time: {elapsed:.1f}s, rate: {self.total_samples_logged_/elapsed:.1f} Hz'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
