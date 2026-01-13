#!/usr/bin/env python3
"""
MARID Attitude Controller
Controls roll, pitch, and yaw by actuating control surfaces (wings and tail).
Converts waypoint navigation commands into control surface deflections.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
from tf_transformations import euler_from_quaternion


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
            self.last_error = error
            return 0.0
        
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.last_error) / dt
        
        # Total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)
        
        # Update state
        self.last_time = current_time
        self.last_error = error
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


class MaridAttitudeController(Node):
    def __init__(self):
        super().__init__('marid_attitude_controller')
        
        # Parameters
        self.declare_parameter('update_rate', 50.0)  # Control loop frequency (Hz)
        
        # PID gains for attitude control
        self.declare_parameter('roll_kp', 1.0)
        self.declare_parameter('roll_ki', 0.0)
        self.declare_parameter('roll_kd', 0.3)
        
        self.declare_parameter('pitch_kp', 1.5)
        self.declare_parameter('pitch_ki', 0.0)
        self.declare_parameter('pitch_kd', 0.5)
        
        self.declare_parameter('yaw_kp', 1.0)
        self.declare_parameter('yaw_ki', 0.0)
        self.declare_parameter('yaw_kd', 0.3)
        
        # Control surface limits (radians)
        self.declare_parameter('wing_max_deflection', 0.5)  # Max wing deflection (rad)
        self.declare_parameter('tail_max_deflection', 0.5)   # Max tail deflection (rad)
        
        # Waypoint navigation parameters
        self.declare_parameter('destination_latitude', None)
        self.declare_parameter('destination_longitude', None)
        self.declare_parameter('destination_x', None)
        self.declare_parameter('destination_y', None)
        self.declare_parameter('datum_latitude', 37.45397139527321)
        self.declare_parameter('datum_longitude', -122.16791304213365)
        self.declare_parameter('waypoint_tolerance', 2.0)
        
        # Get parameters
        self.update_rate_ = self.get_parameter('update_rate').value
        
        # PID controllers
        roll_limits = (-self.get_parameter('wing_max_deflection').value, 
                      self.get_parameter('wing_max_deflection').value)
        pitch_limits = (-self.get_parameter('wing_max_deflection').value, 
                       self.get_parameter('wing_max_deflection').value)
        yaw_limits = (-self.get_parameter('tail_max_deflection').value, 
                     self.get_parameter('tail_max_deflection').value)
        
        self.roll_pid_ = PIDController(
            kp=self.get_parameter('roll_kp').value,
            ki=self.get_parameter('roll_ki').value,
            kd=self.get_parameter('roll_kd').value,
            output_limits=roll_limits
        )
        
        self.pitch_pid_ = PIDController(
            kp=self.get_parameter('pitch_kp').value,
            ki=self.get_parameter('pitch_ki').value,
            kd=self.get_parameter('pitch_kd').value,
            output_limits=pitch_limits
        )
        
        self.yaw_pid_ = PIDController(
            kp=self.get_parameter('yaw_kp').value,
            ki=self.get_parameter('yaw_ki').value,
            kd=self.get_parameter('yaw_kd').value,
            output_limits=yaw_limits
        )
        
        # Waypoint navigation
        dest_lat = self.get_parameter('destination_latitude').value
        dest_lon = self.get_parameter('destination_longitude').value
        dest_x = self.get_parameter('destination_x').value
        dest_y = self.get_parameter('destination_y').value
        self.datum_lat_ = self.get_parameter('datum_latitude').value
        self.datum_lon_ = self.get_parameter('datum_longitude').value
        self.waypoint_tolerance_ = self.get_parameter('waypoint_tolerance').value
        
        # Determine destination
        if dest_lat is not None and dest_lon is not None:
            # Use GPS coordinates
            self.destination_ = self.lat_lon_to_local(dest_lat, dest_lon, self.datum_lat_, self.datum_lon_)
            self.get_logger().info(f'Destination (GPS): ({dest_lat:.6f}°, {dest_lon:.6f}°)')
        elif dest_x is not None and dest_y is not None:
            # Use local coordinates
            self.destination_ = np.array([dest_x, dest_y])
            self.get_logger().info(f'Destination (local): ({dest_x:.2f}, {dest_y:.2f}) m')
        else:
            self.destination_ = None
            self.get_logger().warn('No destination set. Attitude control will maintain level flight.')
        
        # Current state
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_roll_ = 0.0
        self.current_pitch_ = 0.0
        self.current_yaw_ = 0.0
        self.current_roll_rate_ = 0.0
        self.current_pitch_rate_ = 0.0
        self.current_yaw_rate_ = 0.0
        
        # Desired attitude (for waypoint navigation)
        self.desired_roll_ = 0.0  # Bank angle for turns
        self.desired_pitch_ = 0.0  # Pitch for altitude changes
        self.desired_yaw_ = 0.0    # Heading to waypoint
        
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
        
        self.waypoint_sub_ = self.create_subscription(
            PoseStamped,
            '/marid/waypoint',
            self.waypoint_callback,
            10
        )
        
        # Publisher for joint commands
        # ROS2 control uses Float64MultiArray for joint group position controller
        self.joint_cmd_pub_ = self.create_publisher(
            Float64MultiArray,
            '/simple_position_controller/commands',
            10
        )
        
        # Control loop timer
        timer_period = 1.0 / self.update_rate_
        self.control_timer_ = self.create_timer(timer_period, self.control_loop)
        
        # Joint order (must match marid_controllers.yaml)
        self.joint_names_ = [
            'left_wing_joint',
            'right_wing_joint',
            'tail_left_joint',
            'tail_right_joint'
        ]
        
        self.get_logger().info('MARID Attitude Controller initialized')
        self.get_logger().info(f'Update rate: {self.update_rate_} Hz')
        if self.destination_ is not None:
            self.get_logger().info(f'Waypoint navigation enabled')
    
    def lat_lon_to_local(self, lat, lon, datum_lat, datum_lon):
        """Convert GPS coordinates to local x/y (meters)"""
        # Simple flat-earth approximation (accurate for < 100km)
        R = 6371000.0  # Earth radius in meters
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        datum_lat_rad = math.radians(datum_lat)
        datum_lon_rad = math.radians(datum_lon)
        
        dlat = lat_rad - datum_lat_rad
        dlon = lon_rad - datum_lon_rad
        
        x = R * dlon * math.cos(datum_lat_rad)
        y = R * dlat
        
        return np.array([x, y])
    
    def odom_callback(self, msg):
        """Extract attitude from odometry"""
        self.current_odom_ = msg
        
        # Extract roll, pitch, yaw from quaternion
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        self.current_roll_ = roll
        self.current_pitch_ = pitch
        self.current_yaw_ = yaw
    
    def imu_callback(self, msg):
        """Extract angular rates from IMU"""
        self.current_imu_ = msg
        self.current_roll_rate_ = msg.angular_velocity.x
        self.current_pitch_rate_ = msg.angular_velocity.y
        self.current_yaw_rate_ = msg.angular_velocity.z
    
    def waypoint_callback(self, msg):
        """Update destination waypoint"""
        self.destination_ = np.array([msg.pose.position.x, msg.pose.position.y])
        self.get_logger().info(f'New waypoint: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) m')
    
    def compute_waypoint_commands(self):
        """Compute desired roll, pitch, yaw from waypoint navigation"""
        if self.destination_ is None or self.current_odom_ is None:
            # Maintain level flight
            return 0.0, 0.0, self.current_yaw_
        
        # Current position
        current_pos = np.array([
            self.current_odom_.pose.pose.position.x,
            self.current_odom_.pose.pose.position.y
        ])
        
        # Direction to waypoint
        direction = self.destination_ - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < self.waypoint_tolerance_:
            # Waypoint reached - maintain level flight
            return 0.0, 0.0, self.current_yaw_
        
        # Desired heading to waypoint
        desired_yaw = math.atan2(direction[1], direction[0])
        
        # Compute desired roll for coordinated turn (bank angle)
        # Use a simple proportional relationship: more heading error = more bank
        heading_error = desired_yaw - self.current_yaw_
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Bank angle proportional to heading error (max 30 degrees for turns)
        max_bank_angle = math.radians(30.0)
        desired_roll = np.clip(heading_error * 0.5, -max_bank_angle, max_bank_angle)
        
        # Desired pitch (for now, maintain level flight - can be enhanced for altitude control)
        desired_pitch = 0.0
        
        return desired_roll, desired_pitch, desired_yaw
    
    def control_loop(self):
        """Main control loop - computes and publishes control surface commands"""
        if self.current_odom_ is None:
            return
        
        # Compute desired attitude from waypoint
        desired_roll, desired_pitch, desired_yaw = self.compute_waypoint_commands()
        
        # Get current time
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Compute attitude errors
        roll_error = desired_roll - self.current_roll_
        pitch_error = desired_pitch - self.current_pitch_
        
        # Yaw error (normalize to [-pi, pi])
        yaw_error = desired_yaw - self.current_yaw_
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi
        
        # Update PID controllers
        roll_command = self.roll_pid_.update(roll_error, current_time)
        pitch_command = self.pitch_pid_.update(pitch_error, current_time)
        yaw_command = self.yaw_pid_.update(yaw_error, current_time)
        
        # Convert attitude commands to control surface deflections
        # Control surface mapping:
        # - Wings (left_wing_joint, right_wing_joint): Pitch control (symmetric) + Roll control (differential)
        # - Tail (tail_left_joint, tail_right_joint): Yaw control (differential) + Pitch assist (symmetric)
        
        # Wing deflections:
        # - Symmetric deflection for pitch (both wings same direction)
        # - Differential deflection for roll (opposite directions)
        left_wing_deflection = pitch_command - roll_command  # Negative roll = left wing up
        right_wing_deflection = pitch_command + roll_command  # Positive roll = right wing up
        
        # Tail deflections:
        # - Differential deflection for yaw (opposite directions)
        # - Symmetric deflection for pitch assist (both same direction)
        tail_pitch_assist = pitch_command * 0.3  # 30% pitch assist from tail
        left_tail_deflection = tail_pitch_assist - yaw_command  # Negative yaw = left tail up
        right_tail_deflection = tail_pitch_assist + yaw_command  # Positive yaw = right tail up
        
        # Clamp deflections to joint limits
        wing_max = self.get_parameter('wing_max_deflection').value
        tail_max = self.get_parameter('tail_max_deflection').value
        
        left_wing_deflection = np.clip(left_wing_deflection, -wing_max, wing_max)
        right_wing_deflection = np.clip(right_wing_deflection, -wing_max, wing_max)
        left_tail_deflection = np.clip(left_tail_deflection, -tail_max, tail_max)
        right_tail_deflection = np.clip(right_tail_deflection, -tail_max, tail_max)
        
        # Publish joint commands
        joint_cmd = Float64MultiArray()
        joint_cmd.data = [
            float(left_wing_deflection),
            float(right_wing_deflection),
            float(left_tail_deflection),
            float(right_tail_deflection)
        ]
        
        self.joint_cmd_pub_.publish(joint_cmd)
        
        # Log periodically (every 2 seconds)
        if int(current_time * 10) % 100 == 0:  # Every 2 seconds at 50Hz
            self.get_logger().debug(
                f'Attitude: R={math.degrees(self.current_roll_):.1f}° '
                f'P={math.degrees(self.current_pitch_):.1f}° '
                f'Y={math.degrees(self.current_yaw_):.1f}° | '
                f'Commands: R={math.degrees(roll_command):.2f}° '
                f'P={math.degrees(pitch_command):.2f}° '
                f'Y={math.degrees(yaw_command):.2f}°'
            )


def main():
    rclpy.init()
    node = MaridAttitudeController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

