#!/usr/bin/env python3
"""
MARID Data Logger Node
Collects state-action pairs for ML training (imitation learning / behavior cloning).
Logs state vectors and PID control outputs to .npz files for later training.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import time

# Import state vector builder (reuse from AI controller)
try:
    from marid_controller.marid_controller.ai_model import STATE_DIM
except ImportError:
    STATE_DIM = 20  # Fallback


class MaridDataLogger(Node):
    """
    Data logger for collecting state-action pairs.
    
    Subscribes to:
        - /odometry/filtered/local (state: position, velocity, orientation)
        - /imu_ekf (state: angular velocities)
        - /barometer/altitude (state: altitude)
        - /marid/thrust/total (action: PID thrust output)
        - /marid/yaw/differential (action: PID yaw differential output)
    
    Logs to .npz files in chunks for efficient storage and training.
    """
    
    def __init__(self):
        super().__init__('marid_data_logger')
        
        # Parameters
        self.declare_parameter('log_directory', '~/marid_ws/data')
        self.declare_parameter('log_filename_prefix', 'marid_flight_data')
        self.declare_parameter('samples_per_file', 10000)  # Chunk size
        self.declare_parameter('enable_logging', True)
        self.declare_parameter('log_rate', 50.0)  # Hz - should match controller update rate
        self.declare_parameter('log_metadata', True)  # Log timestamps, scenario info, etc.
        
        # Waypoint parameters (fallback if not received via topic)
        self.declare_parameter('default_destination_x', 0.0)
        self.declare_parameter('default_destination_y', 0.0)
        self.declare_parameter('default_target_altitude', 8000.0)
        self.declare_parameter('default_altitude_min', 3.0)
        self.declare_parameter('default_altitude_max', 10000.0)
        self.declare_parameter('default_target_velocity', 112.0)
        
        log_dir = self.get_parameter('log_directory').value
        self.log_dir_ = Path(log_dir).expanduser()
        self.log_dir_.mkdir(parents=True, exist_ok=True)
        
        self.filename_prefix_ = self.get_parameter('log_filename_prefix').value
        self.samples_per_file_ = self.get_parameter('samples_per_file').value
        self.enable_logging_ = self.get_parameter('enable_logging').value
        log_rate = self.get_parameter('log_rate').value
        self.log_metadata_ = self.get_parameter('log_metadata').value
        
        # Data buffers
        self.states_ = []
        self.actions_ = []
        self.metadata_ = []
        
        # Current state variables (same structure as AI controller)
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_altitude_ = None
        self.current_thrust_ = None
        self.current_yaw_diff_ = None
        
        # Waypoint info (needed for extended state vector) - initialize from parameters
        default_dest_x = self.get_parameter('default_destination_x').value
        default_dest_y = self.get_parameter('default_destination_y').value
        self.destination_ = np.array([default_dest_x, default_dest_y])
        self.altitude_min_ = self.get_parameter('default_altitude_min').value
        self.altitude_max_ = self.get_parameter('default_altitude_max').value
        self.target_altitude_ = self.get_parameter('default_target_altitude').value
        self.target_velocity_ = self.get_parameter('default_target_velocity').value
        
        # File counter
        self.file_counter_ = 0
        self.samples_in_current_file_ = 0
        
        # Subscriptions
        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/odometry/filtered/local',
            self.odom_callback,
            10
        )
        
        self.imu_sub_ = self.create_subscription(
            Imu,
            '/imu_ekf',
            self.imu_callback,
            10
        )
        
        self.altitude_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            '/barometer/altitude',
            self.altitude_callback,
            10
        )
        
        self.thrust_sub_ = self.create_subscription(
            Float64,
            '/marid/thrust/total',
            self.thrust_callback,
            10
        )
        
        self.yaw_diff_sub_ = self.create_subscription(
            Float64,
            '/marid/yaw/differential',
            self.yaw_diff_callback,
            10
        )
        
        # Control mode subscription (to log which controller is active)
        self.control_mode_sub_ = self.create_subscription(
            String,
            '/marid/control_mode',
            self.control_mode_callback,
            10
        )
        
        # Waypoint subscription (to track current destination)
        self.waypoint_sub_ = self.create_subscription(
            PoseStamped,
            '/marid/waypoint',
            self.waypoint_callback,
            10
        )
        
        self.current_control_mode_ = 'unknown'
        
        # Timer for logging (respects log rate)
        self.log_timer_ = self.create_timer(1.0 / log_rate, self.log_data)
        
        # Statistics
        self.total_samples_logged_ = 0
        self.start_time_ = time.time()
        
        self.get_logger().info(f'MARID Data Logger initialized')
        self.get_logger().info(f'  Log directory: {self.log_dir_}')
        self.get_logger().info(f'  Samples per file: {self.samples_per_file_}')
        self.get_logger().info(f'  Log rate: {log_rate} Hz')
        self.get_logger().info(f'  Logging enabled: {self.enable_logging_}')
        
        if not self.enable_logging_:
            self.get_logger().warn('Logging is DISABLED. Enable via parameter to collect data.')
    
    def odom_callback(self, msg):
        """Store current odometry"""
        self.current_odom_ = msg
    
    def imu_callback(self, msg):
        """Store current IMU data"""
        self.current_imu_ = msg
    
    def altitude_callback(self, msg):
        """Store current altitude from barometer"""
        self.current_altitude_ = msg.pose.pose.position.z
    
    def thrust_callback(self, msg):
        """Store current thrust command (PID output)"""
        self.current_thrust_ = msg.data
    
    def yaw_diff_callback(self, msg):
        """Store current yaw differential command (PID output)"""
        self.current_yaw_diff_ = msg.data
    
    def control_mode_callback(self, msg):
        """Store current control mode"""
        self.current_control_mode_ = msg.data
    
    def waypoint_callback(self, msg):
        """Update current waypoint destination"""
        self.destination_ = np.array([msg.pose.position.x, msg.pose.position.y])
        if msg.pose.position.z > 0:
            self.target_altitude_ = msg.pose.position.z
    
    def get_state_vector(self):
        """
        Build state vector (same structure as AI controller).
        Returns: numpy array (12-D base state)
        """
        state = np.zeros(12)
        
        if self.current_odom_ is not None:
            state[0] = self.current_odom_.pose.pose.position.x
            state[1] = self.current_odom_.pose.pose.position.y
            # Use barometric altitude if available, otherwise odom z
            if self.current_altitude_ is not None:
                state[2] = self.current_altitude_
            else:
                state[2] = self.current_odom_.pose.pose.position.z
            
            state[3] = self.current_odom_.twist.twist.linear.x
            state[4] = self.current_odom_.twist.twist.linear.y
            state[5] = self.current_odom_.twist.twist.linear.z
            
            from tf_transformations import euler_from_quaternion
            q = self.current_odom_.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            state[6] = roll
            state[7] = pitch
            state[8] = yaw
        
        if self.current_imu_ is not None:
            state[9] = self.current_imu_.angular_velocity.x
            state[10] = self.current_imu_.angular_velocity.y
            state[11] = self.current_imu_.angular_velocity.z
        
        return state
    
    def get_extended_state_vector(self, state):
        """
        Build extended state vector (20-D) with waypoint info.
        Same structure as AI controller's compute_ai_control.
        """
        # Compute waypoint information
        current_pos = np.array([state[0], state[1]])
        direction = self.destination_ - current_pos
        distance = np.linalg.norm(direction)
        
        current_yaw = state[8]
        desired_heading = np.arctan2(direction[1], direction[0])
        heading_error = desired_heading - current_yaw
        
        # Normalize heading error to [-pi, pi]
        import math
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Extended state (20-D): base(12) + waypoint(8)
        extended_state = np.concatenate([
            state,  # 12 dimensions
            [self.destination_[0], self.destination_[1]],  # 2 dims → 14
            [distance],  # 1 dim → 15
            [heading_error],  # 1 dim → 16
            [self.altitude_min_, self.altitude_max_, self.target_altitude_],  # 3 dims → 19
            [self.target_velocity_]  # 1 dim → 20 total
        ])
        
        return extended_state
    
    def log_data(self):
        """Main logging callback - called at log_rate frequency"""
        if not self.enable_logging_:
            return
        
        # Check if we have all required data
        if (self.current_odom_ is None or 
            self.current_imu_ is None or 
            self.current_thrust_ is None or 
            self.current_yaw_diff_ is None):
            return  # Skip this cycle if data incomplete
        
        try:
            # Build state vector
            base_state = self.get_state_vector()
            extended_state = self.get_extended_state_vector(base_state)
            
            # Build action vector (PID outputs)
            action = np.array([self.current_thrust_, self.current_yaw_diff_])
            
            # Append to buffers
            self.states_.append(extended_state)
            self.actions_.append(action)
            
            # Metadata (optional, for analysis)
            if self.log_metadata_:
                metadata = {
                    'timestamp': time.time(),
                    'control_mode': self.current_control_mode_,
                    'has_baro': self.current_altitude_ is not None
                }
                self.metadata_.append(metadata)
            
            self.samples_in_current_file_ += 1
            self.total_samples_logged_ += 1
            
            # Save file when buffer is full
            if self.samples_in_current_file_ >= self.samples_per_file_:
                self.save_current_file()
            
        except Exception as e:
            self.get_logger().error(f'Error logging data: {e}')
    
    def save_current_file(self):
        """Save current buffer to .npz file"""
        if len(self.states_) == 0:
            return
        
        # Generate filename with timestamp and counter
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.filename_prefix_}_{timestamp}_chunk{self.file_counter_:04d}.npz'
        filepath = self.log_dir_ / filename
        
        # Convert to numpy arrays
        states_array = np.array(self.states_, dtype=np.float32)
        actions_array = np.array(self.actions_, dtype=np.float32)
        
        # Save to .npz
        save_dict = {
            'states': states_array,
            'actions': actions_array,
            'state_dim': STATE_DIM,
            'action_dim': 2,
            'num_samples': len(self.states_)
        }
        
        if self.log_metadata_ and len(self.metadata_) > 0:
            # Save metadata as separate arrays for easy loading
            save_dict['timestamps'] = np.array([m['timestamp'] for m in self.metadata_])
            save_dict['control_modes'] = np.array([m['control_mode'] for m in self.metadata_])
        
        np.savez_compressed(filepath, **save_dict)
        
        elapsed_time = time.time() - self.start_time_
        rate = self.total_samples_logged_ / elapsed_time if elapsed_time > 0 else 0
        
        self.get_logger().info(
            f'Saved data chunk: {filename} ({len(self.states_)} samples, '
            f'total: {self.total_samples_logged_}, rate: {rate:.1f} Hz)'
        )
        
        # Clear buffers and increment counter
        self.states_ = []
        self.actions_ = []
        self.metadata_ = []
        self.file_counter_ += 1
        self.samples_in_current_file_ = 0
    
    def shutdown_callback(self):
        """Save remaining data on shutdown"""
        if len(self.states_) > 0:
            self.get_logger().info('Saving remaining data on shutdown...')
            self.save_current_file()
        
        elapsed_time = time.time() - self.start_time_
        self.get_logger().info(
            f'Data logger shutdown. Total samples logged: {self.total_samples_logged_}, '
            f'total time: {elapsed_time:.1f}s, avg rate: {self.total_samples_logged_/elapsed_time:.1f} Hz'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MaridDataLogger()
    
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
