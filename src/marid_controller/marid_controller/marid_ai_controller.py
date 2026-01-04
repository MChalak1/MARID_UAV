#!/usr/bin/env python3
"""
MARID AI Flight Controller with Waypoint Navigation
Neural network-based flight controller with destination, altitude range, and speed control.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import numpy as np
import math
import time

# Import your AI model
try:
    from marid_controller.marid_controller.ai_model import MaridAIModel
    AI_MODEL_AVAILABLE = True
except ImportError:
    AI_MODEL_AVAILABLE = False
    print("Warning: AI model not available. Install PyTorch/TensorFlow and train a model.")


class MaridAIController(Node):
    def __init__(self):
        super().__init__('marid_ai_controller')
        
        # Parameters
        self.declare_parameter('control_mode', 'ai')  # 'ai', 'pid', 'manual', 'hybrid'
        self.declare_parameter('model_path', '')
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('enable_ai', True)
        self.declare_parameter('enable_pid_fallback', True)
        
        # Waypoint navigation parameters
        self.declare_parameter('destination_x', 0.0)  # Target destination X (m)
        self.declare_parameter('destination_y', 0.0)  # Target destination Y (m)
        self.declare_parameter('altitude_min', 3.0)  # Minimum altitude (m)
        self.declare_parameter('altitude_max', 10.0)  # Maximum altitude (m)
        self.declare_parameter('target_altitude', 5.0)  # Preferred altitude (m)
        self.declare_parameter('target_velocity', 10.0)  # Average/target speed (m/s)
        self.declare_parameter('waypoint_tolerance', 2.0)  # Distance tolerance for waypoint (m)
        self.declare_parameter('altitude_tolerance', 1.0)  # Altitude tolerance (m)
        
        # Control limits
        self.declare_parameter('min_thrust', 0.0)
        self.declare_parameter('max_thrust', 30.0)
        self.declare_parameter('max_yaw_differential', 0.2)
        
        # Get parameters
        self.control_mode_ = self.get_parameter('control_mode').value
        self.model_path_ = self.get_parameter('model_path').value
        self.update_rate_ = self.get_parameter('update_rate').value
        self.enable_ai_ = self.get_parameter('enable_ai').value
        self.enable_pid_fallback_ = self.get_parameter('enable_pid_fallback').value
        
        # Waypoint parameters
        self.destination_ = np.array([
            self.get_parameter('destination_x').value,
            self.get_parameter('destination_y').value
        ])
        self.altitude_min_ = self.get_parameter('altitude_min').value
        self.altitude_max_ = self.get_parameter('altitude_max').value
        self.target_altitude_ = self.get_parameter('target_altitude').value
        self.target_velocity_ = self.get_parameter('target_velocity').value
        self.waypoint_tolerance_ = self.get_parameter('waypoint_tolerance').value
        self.altitude_tolerance_ = self.get_parameter('altitude_tolerance').value
        
        self.min_thrust_ = self.get_parameter('min_thrust').value
        self.max_thrust_ = self.get_parameter('max_thrust').value
        self.max_yaw_differential_ = self.get_parameter('max_yaw_differential').value
        
        # State variables
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_altitude_ = None
        self.waypoint_reached_ = False
        
        # PID controllers
        self.altitude_pid_ = PIDController(kp=2.0, ki=0.1, kd=0.5, output_limits=(0.0, 30.0))
        self.velocity_pid_ = PIDController(kp=1.0, ki=0.05, kd=0.3, output_limits=(0.0, 30.0))
        self.pitch_pid_ = PIDController(kp=1.5, ki=0.1, kd=0.4, output_limits=(-0.2, 0.2))
        self.yaw_pid_ = PIDController(kp=0.5, ki=0.05, kd=0.1, output_limits=(-0.2, 0.2))
        self.heading_pid_ = PIDController(kp=1.0, ki=0.1, kd=0.3, output_limits=(-0.2, 0.2))
        
        # AI Model
        self.ai_model_ = None
        if self.enable_ai_ and AI_MODEL_AVAILABLE:
            try:
                self.ai_model_ = MaridAIModel(model_path=self.model_path_, model_type='pytorch')
                self.get_logger().info('AI model loaded successfully')
            except Exception as e:
                self.get_logger().error(f'Failed to load AI model: {e}')
                self.get_logger().warn('Falling back to PID control')
                self.control_mode_ = 'pid'
        elif self.enable_ai_ and not AI_MODEL_AVAILABLE:
            self.get_logger().warn('AI model not available. Using PID control.')
            self.control_mode_ = 'pid'
        
        # Subscribers
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
        
        # Waypoint subscriber (for dynamic waypoint updates)
        self.waypoint_sub_ = self.create_subscription(
            PoseStamped,
            '/marid/waypoint',
            self.waypoint_callback,
            10
        )
        
        # Publishers
        self.total_thrust_pub_ = self.create_publisher(
            Float64,
            '/marid/thrust/total',
            10
        )
        
        self.yaw_differential_pub_ = self.create_publisher(
            Float64,
            '/marid/thrust/yaw_differential',
            10
        )
        
        self.control_mode_pub_ = self.create_publisher(
            String,
            '/marid/control_mode',
            10
        )
        
        self.waypoint_status_pub_ = self.create_publisher(
            Bool,
            '/marid/waypoint_reached',
            10
        )
        
        # Control loop timer
        timer_period = 1.0 / self.update_rate_
        self.control_timer_ = self.create_timer(timer_period, self.control_loop)
        
        self.get_logger().info(f'MARID AI Controller initialized')
        self.get_logger().info(f'Control mode: {self.control_mode_}')
        self.get_logger().info(f'Destination: ({self.destination_[0]:.2f}, {self.destination_[1]:.2f})')
        self.get_logger().info(f'Altitude range: {self.altitude_min_:.2f} - {self.altitude_max_:.2f} m')
        self.get_logger().info(f'Target altitude: {self.target_altitude_:.2f} m')
        self.get_logger().info(f'Target velocity: {self.target_velocity_:.2f} m/s')
    
    def odom_callback(self, msg):
        """Store current odometry"""
        self.current_odom_ = msg
    
    def imu_callback(self, msg):
        """Store current IMU data"""
        self.current_imu_ = msg
    
    def altitude_callback(self, msg):
        """Store current altitude from barometer"""
        self.current_altitude_ = msg.pose.pose.position.z
    
    def waypoint_callback(self, msg):
        """Update destination waypoint dynamically"""
        self.destination_ = np.array([msg.pose.position.x, msg.pose.position.y])
        if msg.pose.position.z > 0:
            self.target_altitude_ = msg.pose.position.z
        self.waypoint_reached_ = False
        self.get_logger().info(f'New waypoint set: ({self.destination_[0]:.2f}, {self.destination_[1]:.2f}) at {self.target_altitude_:.2f} m')
    
    def get_state_vector(self):
        """
        Extract state vector from current sensor readings.
        Returns: numpy array [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, altitude]
        """
        state = np.zeros(13)
        
        if self.current_odom_ is not None:
            state[0] = self.current_odom_.pose.pose.position.x
            state[1] = self.current_odom_.pose.pose.position.y
            state[2] = self.current_odom_.pose.pose.position.z
            
            state[3] = self.current_odom_.twist.twist.linear.x
            state[4] = self.current_odom_.twist.twist.linear.y
            state[5] = self.current_odom_.twist.twist.linear.z
            
            q = self.current_odom_.pose.pose.orientation
            from tf_transformations import euler_from_quaternion
            roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            state[6] = roll
            state[7] = pitch
            state[8] = yaw
        
        if self.current_imu_ is not None:
            state[9] = self.current_imu_.angular_velocity.x
            state[10] = self.current_imu_.angular_velocity.y
            state[11] = self.current_imu_.angular_velocity.z
        
        if self.current_altitude_ is not None:
            state[12] = self.current_altitude_
        elif self.current_odom_ is not None:
            state[12] = self.current_odom_.pose.pose.position.z
        
        return state
    
    def compute_waypoint_heading(self, state):
        """Compute desired heading to waypoint"""
        current_pos = np.array([state[0], state[1]])
        direction = self.destination_ - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < self.waypoint_tolerance_:
            return None, distance  # Waypoint reached
        
        desired_heading = math.atan2(direction[1], direction[0])
        return desired_heading, distance
    
    def compute_pid_control(self, state):
        """
        Compute control actions using PID controllers with waypoint navigation.
        Returns: (total_thrust, yaw_differential)
        """
        # Check if waypoint reached
        desired_heading, distance = self.compute_waypoint_heading(state)
        
        if desired_heading is None:
            # Waypoint reached - maintain position
            if not self.waypoint_reached_:
                self.get_logger().info('Waypoint reached!')
                self.waypoint_reached_ = True
            
            # Maintain altitude and hover
            altitude_error = self.target_altitude_ - state[2]
            altitude_thrust = self.altitude_pid_.update(altitude_error, self.get_clock().now().nanoseconds / 1e9)
            total_thrust = np.clip(altitude_thrust, self.min_thrust_, self.max_thrust_)
            yaw_differential = 0.0
            return total_thrust, yaw_differential
        
        # Altitude control (within range)
        current_altitude = state[2]
        if current_altitude < self.altitude_min_:
            altitude_error = self.altitude_min_ - current_altitude
        elif current_altitude > self.altitude_max_:
            altitude_error = self.altitude_max_ - current_altitude
        else:
            altitude_error = self.target_altitude_ - current_altitude
        
        altitude_thrust = self.altitude_pid_.update(altitude_error, self.get_clock().now().nanoseconds / 1e9)
        
        # Velocity control (maintain target speed)
        velocity = math.sqrt(state[3]**2 + state[4]**2 + state[5]**2)
        velocity_error = self.target_velocity_ - velocity
        velocity_thrust = self.velocity_pid_.update(velocity_error, self.get_clock().now().nanoseconds / 1e9)
        
        # Total thrust
        total_thrust = np.clip(altitude_thrust + velocity_thrust, self.min_thrust_, self.max_thrust_)
        
        # Heading control (navigate to waypoint)
        current_yaw = state[8]
        heading_error = desired_heading - current_yaw
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        yaw_differential = self.heading_pid_.update(heading_error, self.get_clock().now().nanoseconds / 1e9)
        yaw_differential = np.clip(yaw_differential, -self.max_yaw_differential_, self.max_yaw_differential_)
        
        return total_thrust, yaw_differential
    
    def compute_ai_control(self, state):
        """
        Compute control actions using AI model with waypoint navigation.
        Returns: (total_thrust, yaw_differential)
        """
        if self.ai_model_ is None:
            return self.compute_pid_control(state)
        
        try:
            # Compute waypoint information
            desired_heading, distance = self.compute_waypoint_heading(state)
            if desired_heading is None:
                # Waypoint reached - use PID for hovering
                return self.compute_pid_control(state)
            
            # Create extended state vector with waypoint info
            current_yaw = state[8]
            heading_error = desired_heading - current_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            state_with_target = np.concatenate([
                state,
                [self.destination_[0], self.destination_[1]],  # Waypoint position
                [distance],  # Distance to waypoint
                [heading_error],  # Heading error
                [self.altitude_min_, self.altitude_max_, self.target_altitude_],  # Altitude constraints
                [self.target_velocity_]  # Target velocity
            ])
            
            # Get action from AI model
            action = self.ai_model_.predict(state_with_target)
            
            # Extract thrust and yaw differential
            total_thrust = np.clip(action[0], self.min_thrust_, self.max_thrust_)
            yaw_differential = np.clip(action[1], -self.max_yaw_differential_, self.max_yaw_differential_)
            
            return total_thrust, yaw_differential
            
        except Exception as e:
            self.get_logger().error(f'AI model prediction failed: {e}')
            self.get_logger().warn('Falling back to PID control')
            return self.compute_pid_control(state)
    
    def control_loop(self):
        """Main control loop"""
        if self.current_odom_ is None:
            return
        
        state = self.get_state_vector()
        
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            self.get_logger().warn('Invalid state detected (NaN/inf), skipping control update')
            return
        
        # Compute control actions
        if self.control_mode_ == 'ai' and self.ai_model_ is not None:
            total_thrust, yaw_differential = self.compute_ai_control(state)
        elif self.control_mode_ == 'pid' or (self.control_mode_ == 'ai' and self.ai_model_ is None):
            total_thrust, yaw_differential = self.compute_pid_control(state)
        else:
            return
        
        # Publish control commands
        thrust_msg = Float64()
        thrust_msg.data = float(total_thrust)
        self.total_thrust_pub_.publish(thrust_msg)
        
        yaw_msg = Float64()
        yaw_msg.data = float(yaw_differential)
        self.yaw_differential_pub_.publish(yaw_msg)
        
        # Publish control mode
        mode_msg = String()
        mode_msg.data = self.control_mode_
        self.control_mode_pub_.publish(mode_msg)
        
        # Publish waypoint status
        status_msg = Bool()
        status_msg.data = self.waypoint_reached_
        self.waypoint_status_pub_.publish(status_msg)


class PIDController:
    """Simple PID controller implementation"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    
    def update(self, error, current_time):
        """Update PID controller and return output"""
        if self.last_time is None:
            self.last_time = current_time
            return 0.0
        
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0
        
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.last_error) / dt
        
        output = p_term + i_term + d_term
        
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


def main():
    rclpy.init()
    node = MaridAIController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
