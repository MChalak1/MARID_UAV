#!/usr/bin/env python3
"""
MARID Guidance Tracker (Option A Architecture)
Converts high-level guidance targets into low-level actuator commands.

Subscribes to guidance targets:
    /marid/guidance/desired_heading_rate (Float64) - rad/s
    /marid/guidance/desired_speed (Float64) - m/s

Publishes actuator commands:
    /marid/thrust/total (Float64) - N
    /marid/thrust/yaw_differential (Float64) - rad/s or normalized
    
Note: Topic names match what marid_thrust_controller.py expects.
    
This node implements the PID tracking layer that converts guidance targets
into actuator commands. It does NOT compute guidance - that's done by marid_ai_guidance.py.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
import math
import xml.etree.ElementTree as ET


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


def calculate_total_mass_from_urdf(urdf_string):
    """Parse URDF and calculate total mass"""
    try:
        root = ET.fromstring(urdf_string)
        total_mass = 0.0
        
        for link in root.findall('.//link'):
            inertial = link.find('inertial')
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass_value = mass_elem.get('value')
                    if mass_value:
                        try:
                            total_mass += float(mass_value)
                        except ValueError:
                            pass
        
        return total_mass if total_mass > 0 else None
    except Exception as e:
        print(f"Error parsing URDF: {e}")
        return None


class MaridGuidanceTracker(Node):
    """
    Guidance Tracker - Tracks high-level guidance targets and outputs actuator commands.
    
    Architecture:
        Guidance Node → Guidance Targets → This Node → Actuator Commands → Thrust Controller
    """
    
    def __init__(self):
        super().__init__('marid_guidance_tracker')
        
        # Parameters
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('thrust_to_weight_ratio', 2.5)
        self.declare_parameter('max_thrust', None)  # Auto-calculate if None
        self.declare_parameter('base_thrust_override', None)
        self.declare_parameter('min_thrust', 0.0)
        self.declare_parameter('max_yaw_differential', 0.2)
        
        # Speed tracking PID gains
        self.declare_parameter('speed_kp', 1.0)
        self.declare_parameter('speed_ki', 0.05)
        self.declare_parameter('speed_kd', 0.3)
        
        # Heading rate tracking PID gains
        self.declare_parameter('heading_rate_kp', 1.0)
        self.declare_parameter('heading_rate_ki', 0.1)
        self.declare_parameter('heading_rate_kd', 0.3)
        
        # Altitude control (maintains altitude while tracking guidance)
        self.declare_parameter('altitude_kp', 2.0)
        self.declare_parameter('altitude_ki', 0.1)
        self.declare_parameter('altitude_kd', 0.5)
        self.declare_parameter('target_altitude', 8000.0)
        
        # Get parameters
        self.update_rate_ = self.get_parameter('update_rate').value
        self.thrust_to_weight_ratio_ = self.get_parameter('thrust_to_weight_ratio').value
        self.min_thrust_ = self.get_parameter('min_thrust').value
        self.max_yaw_differential_ = self.get_parameter('max_yaw_differential').value
        
        # Calculate max_thrust
        max_thrust_param = self.get_parameter('max_thrust').value
        base_thrust_override = self.get_parameter('base_thrust_override').value
        
        if max_thrust_param is None or base_thrust_override is not None:
            aircraft_mass = self.get_aircraft_mass()
            if aircraft_mass is not None:
                if base_thrust_override is not None:
                    self.max_thrust_ = float(base_thrust_override)
                else:
                    g = 9.81
                    weight = aircraft_mass * g
                    self.max_thrust_ = weight * self.thrust_to_weight_ratio_
                self.get_logger().info(f'Aircraft mass: {aircraft_mass:.2f} kg, max_thrust: {self.max_thrust_:.2f} N')
            else:
                self.max_thrust_ = 200.0 if max_thrust_param is None else float(max_thrust_param)
                self.get_logger().warn(f'Could not determine mass, using default max_thrust: {self.max_thrust_:.2f} N')
        else:
            self.max_thrust_ = float(max_thrust_param)
        
        # Current state
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_altitude_ = None
        
        # Guidance targets (from guidance node)
        self.desired_heading_rate_ = 0.0
        self.desired_speed_ = 0.0
        self.guidance_mode_ = 'pid'  # Track which mode guidance is in
        self.guidance_received_ = False  # Flag to track if guidance has been received
        self.pid_reset_on_guidance_ = False  # Flag to ensure PID reset only happens once
        
        # PID controllers for tracking guidance targets
        self.speed_pid_ = PIDController(
            kp=self.get_parameter('speed_kp').value,
            ki=self.get_parameter('speed_ki').value,
            kd=self.get_parameter('speed_kd').value,
            output_limits=(0.0, self.max_thrust_)
        )
        
        self.heading_rate_pid_ = PIDController(
            kp=self.get_parameter('heading_rate_kp').value,
            ki=self.get_parameter('heading_rate_ki').value,
            kd=self.get_parameter('heading_rate_kd').value,
            output_limits=(-self.max_yaw_differential_, self.max_yaw_differential_)
        )
        
        self.altitude_pid_ = PIDController(
            kp=self.get_parameter('altitude_kp').value,
            ki=self.get_parameter('altitude_ki').value,
            kd=self.get_parameter('altitude_kd').value,
            output_limits=(0.0, self.max_thrust_)
        )
        
        self.target_altitude_ = self.get_parameter('target_altitude').value
        
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
        
        # Subscribe to GUIDANCE TARGETS (from guidance node)
        self.desired_heading_rate_sub_ = self.create_subscription(
            Float64,
            '/marid/guidance/desired_heading_rate',
            self.desired_heading_rate_callback,
            10
        )
        
        self.desired_speed_sub_ = self.create_subscription(
            Float64,
            '/marid/guidance/desired_speed',
            self.desired_speed_callback,
            10
        )
        
        self.guidance_mode_sub_ = self.create_subscription(
            String,
            '/marid/guidance/mode',
            self.guidance_mode_callback,
            10
        )
        
        # Publishers - ACTUATOR COMMANDS (consumed by thrust controller)
        # Topic names must match what marid_thrust_controller.py subscribes to
        self.total_thrust_pub_ = self.create_publisher(
            Float64,
            '/marid/thrust/total',
            10
        )
        
        self.yaw_differential_pub_ = self.create_publisher(
            Float64,
            '/marid/thrust/yaw_differential',  # Matches thrust controller subscription
            10
        )
        
        # Control loop timer
        self.control_timer_ = self.create_timer(1.0 / self.update_rate_, self.control_loop)
        
        self.get_logger().info('MARID Guidance Tracker initialized (Option A architecture)')
        self.get_logger().info(f'  Max thrust: {self.max_thrust_:.2f} N')
        self.get_logger().info(f'  Subscribes to guidance targets from /marid/guidance/*')
        self.get_logger().info(f'  Publishes actuator commands to /marid/thrust/*')
    
    def get_aircraft_mass(self):
        """Get aircraft mass from URDF"""
        try:
            from ament_index_python.packages import get_package_share_directory
            import os
            
            package_dir = get_package_share_directory('marid_description')
            urdf_path = os.path.join(package_dir, 'urdf', 'marid.urdf')
            
            # Try to run xacro to get URDF string
            import subprocess
            result = subprocess.run(
                ['xacro', urdf_path],
                capture_output=True,
                text=True,
                timeout=5.0
            )
            
            if result.returncode == 0:
                return calculate_total_mass_from_urdf(result.stdout)
        except Exception as e:
            self.get_logger().debug(f'Could not get aircraft mass: {e}')
        
        return None
    
    def odom_callback(self, msg):
        """Store current odometry"""
        self.current_odom_ = msg
    
    def imu_callback(self, msg):
        """Store current IMU data"""
        self.current_imu_ = msg
    
    def altitude_callback(self, msg):
        """Store current altitude"""
        self.current_altitude_ = msg.pose.pose.position.z
    
    def desired_heading_rate_callback(self, msg):
        """Store desired heading rate from guidance node"""
        self.desired_heading_rate_ = msg.data
        
        # Mark guidance as received and reset PIDs once when guidance first arrives
        if not self.guidance_received_:
            self.guidance_received_ = True
            self._reset_pids_for_guidance()
    
    def desired_speed_callback(self, msg):
        """Store desired speed from guidance node"""
        self.desired_speed_ = msg.data
        
        # Mark guidance as received and reset PIDs once when guidance first arrives
        if not self.guidance_received_:
            self.guidance_received_ = True
            self._reset_pids_for_guidance()
    
    def _reset_pids_for_guidance(self):
        """Reset PID controllers when guidance first arrives (called only once)"""
        if not self.pid_reset_on_guidance_:
            self.speed_pid_.reset()
            self.heading_rate_pid_.reset()
            self.pid_reset_on_guidance_ = True
            self.get_logger().info('Guidance targets received! Resetting PID controllers and starting guidance tracking.')
    
    def guidance_mode_callback(self, msg):
        """Store guidance mode"""
        self.guidance_mode_ = msg.data
    
    def control_loop(self):
        """Main control loop - tracks guidance targets and outputs actuator commands"""
        if self.current_odom_ is None:
            return
        
        # Wait for guidance to be received before tracking
        # This prevents tracking zero guidance during initialization
        if not self.guidance_received_:
            # During initialization, maintain altitude only (safety)
            # Don't track speed/heading until guidance is active
            self.get_logger().debug('Waiting for guidance targets... Using altitude-only control.')
            
            # Only maintain altitude (prevent falling during startup)
            if current_altitude is not None:
                altitude_error = self.target_altitude_ - current_altitude
                altitude_thrust = self.altitude_pid_.update(altitude_error, current_time)
                altitude_thrust = np.clip(altitude_thrust, self.min_thrust_, self.max_thrust_)
                
                # Publish altitude-only thrust (no yaw command)
                thrust_msg = Float64()
                thrust_msg.data = float(altitude_thrust)
                self.total_thrust_pub_.publish(thrust_msg)
                
                yaw_msg = Float64()
                yaw_msg.data = 0.0  # No yaw command during initialization
                self.yaw_differential_pub_.publish(yaw_msg)
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Get current state
        current_velocity = math.sqrt(
            self.current_odom_.twist.twist.linear.x**2 +
            self.current_odom_.twist.twist.linear.y**2 +
            self.current_odom_.twist.twist.linear.z**2
        )
        
        current_yaw_rate = 0.0
        if self.current_imu_ is not None:
            current_yaw_rate = self.current_imu_.angular_velocity.z
        
        current_altitude = self.current_altitude_
        if current_altitude is None:
            current_altitude = self.current_odom_.pose.pose.position.z
        
        # Track speed target (guidance provides desired_speed)
        speed_error = self.desired_speed_ - current_velocity
        speed_thrust = self.speed_pid_.update(speed_error, current_time)
        
        # Track heading rate target (guidance provides desired_heading_rate)
        heading_rate_error = self.desired_heading_rate_ - current_yaw_rate
        yaw_differential = self.heading_rate_pid_.update(heading_rate_error, current_time)
        yaw_differential = np.clip(yaw_differential, -self.max_yaw_differential_, self.max_yaw_differential_)
        
        # Maintain altitude (independent control, doesn't come from guidance yet)
        altitude_error = self.target_altitude_ - current_altitude
        altitude_thrust = self.altitude_pid_.update(altitude_error, current_time)
        
        # Total thrust combines speed tracking and altitude maintenance
        total_thrust = speed_thrust + altitude_thrust
        total_thrust = np.clip(total_thrust, self.min_thrust_, self.max_thrust_)
        
        # Publish actuator commands
        thrust_msg = Float64()
        thrust_msg.data = float(total_thrust)
        self.total_thrust_pub_.publish(thrust_msg)
        
        yaw_msg = Float64()
        yaw_msg.data = float(yaw_differential)
        self.yaw_differential_pub_.publish(yaw_msg)
    
    def set_target_altitude(self, altitude):
        """Update target altitude (can be called from waypoint updates)"""
        self.target_altitude_ = altitude


def main(args=None):
    rclpy.init(args=args)
    node = MaridGuidanceTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
