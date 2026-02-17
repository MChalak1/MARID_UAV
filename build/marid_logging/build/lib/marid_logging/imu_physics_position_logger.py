#!/usr/bin/env python3
"""
MARID IMU Physics Position Data Logger
Collects IMU acceleration sequences (inputs) and ground-truth positions (targets) for training
a physics-augmented ML model to estimate [x, y, z] position from IMU data.

Input features (X):  Sequences of [ax, ay, az, gx, gy, gz, qx, qy, qz, qw] over time window
Target (y_real):     Initial position [x₀, y₀, z₀] and final position [x_final, y_final, z_final]

Subscribes to:
    - /imu_ekf          (sensor_msgs/Imu)
    - /gazebo/odom      (nav_msgs/Odometry, ground truth)

Saves to .npz files for training: f(IMU_sequence, initial_pos) -> final_pos
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import math
from collections import deque


class IMUPhysicsPositionLogger(Node):
    """
    Data logger for IMU-based position estimation (physics + ML).
    
    Logs:
        X: Sequences of shape (sequence_length, 10) - [ax, ay, az, gx, gy, gz, qx, qy, qz, qw]
        y_initial: [x₀, y₀, z₀] - Initial position at start of sequence
        y_final: [x_final, y_final, z_final] - Final position at end of sequence
        y_delta: [Δx, Δy, Δz] - Displacement over sequence (final - initial)
    """
    
    def __init__(self):
        super().__init__('imu_physics_position_logger')
        
        self.declare_parameter('imu_topic', '/imu_ekf')
        self.declare_parameter('ground_truth_topic', '/gazebo/odom')
        self.declare_parameter('log_directory', '~/marid_ws/data')
        self.declare_parameter('log_filename_prefix', 'marid_imu_position')
        self.declare_parameter('samples_per_file', 1000)  # Number of sequences per file
        self.declare_parameter('sequence_length', 100)  # Number of timesteps per sequence
        self.declare_parameter('sequence_rate', 10.0)  # Hz - how often to extract sequences
        self.declare_parameter('enable_logging', True)
        
        self.imu_topic_ = self.get_parameter('imu_topic').value
        self.gt_topic_ = self.get_parameter('ground_truth_topic').value
        log_dir = Path(self.get_parameter('log_directory').value).expanduser()
        self.log_dir_ = log_dir
        self.log_dir_.mkdir(parents=True, exist_ok=True)
        self.filename_prefix_ = self.get_parameter('log_filename_prefix').value
        self.samples_per_file_ = self.get_parameter('samples_per_file').value
        self.sequence_length_ = self.get_parameter('sequence_length').value
        sequence_rate = self.get_parameter('sequence_rate').value
        self.enable_logging_ = self.get_parameter('enable_logging').value
        
        # IMU buffer: deque of (timestamp, imu_data) tuples
        # imu_data: [ax, ay, az, gx, gy, gz, qx, qy, qz, qw]
        self.imu_buffer_ = deque(maxlen=self.sequence_length_ * 2)  # Keep extra for overlap
        
        # Latest ground truth position
        self.last_odom_ = None
        self.last_odom_timestamp_ = None
        
        # Buffers for sequences
        self.imu_sequences_ = []  # List of (sequence_length, 10) arrays
        self.initial_positions_ = []  # List of [x₀, y₀, z₀]
        self.final_positions_ = []  # List of [x_final, y_final, z_final]
        self.delta_positions_ = []  # List of [Δx, Δy, Δz]
        
        self.file_counter_ = 0
        self.sequences_in_current_file_ = 0
        self.total_sequences_logged_ = 0
        self.start_time_ = time.time()
        
        # Subscriptions
        self.imu_sub_ = self.create_subscription(
            Imu, self.imu_topic_, self.imu_callback, 10
        )
        self.odom_sub_ = self.create_subscription(
            Odometry, self.gt_topic_, self.odom_callback, 10
        )
        
        # Timer to extract sequences periodically
        self.sequence_timer_ = self.create_timer(1.0 / sequence_rate, self.extract_sequence)
        
        self.get_logger().info('IMU Physics Position Logger initialized')
        self.get_logger().info(f'  IMU: {self.imu_topic_}')
        self.get_logger().info(f'  Ground truth: {self.gt_topic_}')
        self.get_logger().info(f'  Log directory: {self.log_dir_}')
        self.get_logger().info(f'  Sequence length: {self.sequence_length_} timesteps')
        self.get_logger().info(f'  Sequence extraction rate: {sequence_rate} Hz')
        self.get_logger().info(f'  Samples per file: {self.samples_per_file_}')
    
    def imu_callback(self, msg):
        """Store IMU data in buffer with timestamp."""
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Extract IMU features: [ax, ay, az, gx, gy, gz, qx, qy, qz, qw]
        acc = msg.linear_acceleration
        gyro = msg.angular_velocity
        ori = msg.orientation
        
        # Validate data
        if (math.isnan(acc.x) or math.isnan(acc.y) or math.isnan(acc.z) or
            math.isnan(gyro.x) or math.isnan(gyro.y) or math.isnan(gyro.z) or
            math.isnan(ori.x) or math.isnan(ori.y) or math.isnan(ori.z) or math.isnan(ori.w) or
            math.isinf(acc.x) or math.isinf(acc.y) or math.isinf(acc.z) or
            math.isinf(gyro.x) or math.isinf(gyro.y) or math.isinf(gyro.z) or
            math.isinf(ori.x) or math.isinf(ori.y) or math.isinf(ori.z) or math.isinf(ori.w)):
            return
        
        imu_data = np.array([
            acc.x, acc.y, acc.z,  # Linear acceleration
            gyro.x, gyro.y, gyro.z,  # Angular velocity
            ori.x, ori.y, ori.z, ori.w  # Orientation quaternion
        ], dtype=np.float32)
        
        self.imu_buffer_.append((timestamp, imu_data))
    
    def odom_callback(self, msg):
        """Store latest ground truth position."""
        self.last_odom_ = msg
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_odom_timestamp_ = timestamp
    
    def _get_position_at_time(self, target_timestamp, tolerance=0.1):
        """
        Get position closest to target_timestamp.
        For now, returns latest position if within tolerance, None otherwise.
        In a more sophisticated implementation, could interpolate between odom messages.
        """
        if self.last_odom_ is None:
            return None
        
        # Simple approach: use latest position if timestamp is close
        if abs(self.last_odom_timestamp_ - target_timestamp) < tolerance:
            pos = self.last_odom_.pose.pose.position
            return np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        
        return None
    
    def extract_sequence(self):
        """Extract a sequence from the IMU buffer if we have enough data."""
        if not self.enable_logging_:
            return
        
        if len(self.imu_buffer_) < self.sequence_length_:
            return
        
        if self.last_odom_ is None:
            return
        
        # Extract the most recent sequence_length samples
        sequence_data = list(self.imu_buffer_)[-self.sequence_length_:]
        
        # Build IMU sequence array: (sequence_length, 10)
        imu_sequence = np.array([data for _, data in sequence_data], dtype=np.float32)
        
        # Get timestamps
        timestamps = [ts for ts, _ in sequence_data]
        initial_timestamp = timestamps[0]
        final_timestamp = timestamps[-1]
        
        # Get positions at sequence boundaries
        initial_pos = self._get_position_at_time(initial_timestamp)
        final_pos = self._get_position_at_time(final_timestamp)
        
        if initial_pos is None or final_pos is None:
            # Could not get positions for this sequence
            return
        
        # Validate sequence
        if np.any(np.isnan(imu_sequence)) or np.any(np.isinf(imu_sequence)):
            return
        if np.any(np.isnan(initial_pos)) or np.any(np.isinf(initial_pos)):
            return
        if np.any(np.isnan(final_pos)) or np.any(np.isinf(final_pos)):
            return
        
        # Calculate delta position
        delta_pos = final_pos - initial_pos
        
        # Store sequence
        self.imu_sequences_.append(imu_sequence)
        self.initial_positions_.append(initial_pos)
        self.final_positions_.append(final_pos)
        self.delta_positions_.append(delta_pos)
        
        self.sequences_in_current_file_ += 1
        self.total_sequences_logged_ += 1
        
        if self.sequences_in_current_file_ >= self.samples_per_file_:
            self._save_chunk()
    
    def _save_chunk(self):
        """Save accumulated sequences to .npz file."""
        if len(self.imu_sequences_) == 0:
            return
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'{self.filename_prefix_}_{ts}_chunk{self.file_counter_:04d}.npz'
        fpath = self.log_dir_ / fname
        
        # Convert to numpy arrays
        imu_sequences = np.array(self.imu_sequences_, dtype=np.float32)  # (N, sequence_length, 10)
        initial_positions = np.array(self.initial_positions_, dtype=np.float32)  # (N, 3)
        final_positions = np.array(self.final_positions_, dtype=np.float32)  # (N, 3)
        delta_positions = np.array(self.delta_positions_, dtype=np.float32)  # (N, 3)
        
        np.savez_compressed(
            fpath,
            imu_sequences=imu_sequences,
            initial_positions=initial_positions,
            final_positions=final_positions,
            delta_positions=delta_positions,
            sequence_length=self.sequence_length_,
            input_dim=10,  # [ax, ay, az, gx, gy, gz, qx, qy, qz, qw]
            position_dim=3,  # [x, y, z]
            num_sequences=len(self.imu_sequences_)
        )
        
        elapsed = time.time() - self.start_time_
        rate = self.total_sequences_logged_ / elapsed if elapsed > 0 else 0
        self.get_logger().info(
            f'Saved {fname} ({len(self.imu_sequences_)} sequences, '
            f'total: {self.total_sequences_logged_}, rate: {rate:.2f} seq/s)'
        )
        
        # Clear buffers
        self.imu_sequences_ = []
        self.initial_positions_ = []
        self.final_positions_ = []
        self.delta_positions_ = []
        self.file_counter_ += 1
        self.sequences_in_current_file_ = 0
    
    def shutdown_callback(self):
        """Save remaining data on shutdown."""
        if len(self.imu_sequences_) > 0:
            self.get_logger().info('Saving remaining data on shutdown...')
            self._save_chunk()
        elapsed = time.time() - self.start_time_
        self.get_logger().info(
            f'IMU Physics Position Logger shutdown. Total sequences: {self.total_sequences_logged_}, '
            f'time: {elapsed:.1f}s, rate: {self.total_sequences_logged_/elapsed:.2f} seq/s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = IMUPhysicsPositionLogger()
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
