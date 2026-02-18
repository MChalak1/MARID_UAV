#!/usr/bin/env python3
"""
MARID AI Guidance Node (Option A Architecture)
High-level guidance node that outputs guidance targets (desired_heading_rate, desired_speed).
This node does NOT output actuator commands - those are handled by guidance_tracker.py.

Architecture:
  AI Guidance Node → Guidance Targets (desired_heading_rate, desired_speed)
  Guidance Tracker → Actuator Commands (thrust, yaw_diff, control surfaces)
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
import xml.etree.ElementTree as ET

# Import AI model
try:
    from marid_controller.marid_controller.ai_model import MaridAIModel, STATE_DIM
    from marid_controller.marid_controller.state_normalizer import StateNormalizer
    AI_MODEL_AVAILABLE = True
except ImportError:
    AI_MODEL_AVAILABLE = False
    print("Warning: AI model dependencies not available. Install required packages.")


def calculate_total_mass_from_urdf(urdf_string):
    """
    Parse URDF/XACRO string and calculate total mass of all links.
    Returns total mass in kg, or None if parsing fails.
    """
    try:
        root = ET.fromstring(urdf_string)
        total_mass = 0.0
        mass_count = 0
        
        for link in root.findall('.//link'):
            inertial = link.find('inertial')
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass_value = mass_elem.get('value')
                    if mass_value:
                        try:
                            mass = float(mass_value)
                            total_mass += mass
                            mass_count += 1
                        except ValueError:
                            pass
        
        if mass_count > 0:
            return total_mass
        else:
            return None
    except Exception as e:
        print(f"Error parsing URDF: {e}")
        return None


class MaridAIGuidance(Node):
    """
    AI Guidance Node - Outputs high-level guidance targets.
    
    Publishes:
        /marid/guidance/desired_heading_rate (Float64) - rad/s
        /marid/guidance/desired_speed (Float64) - m/s
        /marid/guidance/mode (String) - 'ai' or 'pid'
        /marid/guidance/waypoint_reached (Bool)
    
    Does NOT publish actuator commands (thrust, yaw_diff).
    """
    
    def __init__(self):
        super().__init__('marid_ai_guidance')
        
        # Parameters
        self.declare_parameter('control_mode', 'ai')  # 'ai', 'pid', 'hybrid'
        self.declare_parameter('model_path', '')
        self.declare_parameter('normalizer_path', '')  # Path to normalization parameters
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('enable_ai', True)
        self.declare_parameter('enable_pid_fallback', True)
        
        # Waypoint navigation parameters
        self.declare_parameter('destination_latitude', -1.0)
        self.declare_parameter('destination_longitude', -1.0)
        self.declare_parameter('destination_x', -1.0)
        self.declare_parameter('destination_y', -1.0)
        self.declare_parameter('datum_latitude', 37.4)  # Match Gazebo world origin (wt.sdf)
        self.declare_parameter('datum_longitude', -122.1)
        
        # Guidance parameters
        self.declare_parameter('target_altitude', 8000.0)  # m
        self.declare_parameter('target_velocity', 112.0)  # m/s
        self.declare_parameter('altitude_min', 3.0)  # m
        self.declare_parameter('altitude_max', 10000.0)  # m
        self.declare_parameter('waypoint_tolerance', 2.0)  # m
        
        # Guidance limits
        self.declare_parameter('max_heading_rate', 0.5)  # rad/s
        self.declare_parameter('min_speed', 10.0)  # m/s
        self.declare_parameter('max_speed', 200.0)  # m/s
        
        # Get parameters
        self.control_mode_ = self.get_parameter('control_mode').value
        self.model_path_ = self.get_parameter('model_path').value
        self.normalizer_path_ = self.get_parameter('normalizer_path').value
        self.update_rate_ = self.get_parameter('update_rate').value
        self.enable_ai_ = self.get_parameter('enable_ai').value
        self.enable_pid_fallback_ = self.get_parameter('enable_pid_fallback').value
        
        # Get datum
        self.datum_lat_ = self.get_parameter('datum_latitude').value
        self.datum_lon_ = self.get_parameter('datum_longitude').value
        
        # Get waypoint
        dest_lat = self.get_parameter('destination_latitude').value
        dest_lon = self.get_parameter('destination_longitude').value
        dest_x = self.get_parameter('destination_x').value
        dest_y = self.get_parameter('destination_y').value
        
        if dest_lat != -1.0 and dest_lon != -1.0:
            x, y = self.lat_lon_to_local(dest_lat, dest_lon, self.datum_lat_, self.datum_lon_)
            self.destination_ = np.array([x, y])
            self.destination_gps_ = (dest_lat, dest_lon)
            self.use_gps_coords_ = True
            self.get_logger().info(f'Using GPS coordinates: ({dest_lat:.6f}°, {dest_lon:.6f}°) → local: ({x:.2f}, {y:.2f}) m')
        elif dest_x != -1.0 and dest_y != -1.0:
            self.destination_ = np.array([dest_x, dest_y])
            self.destination_gps_ = None
            self.use_gps_coords_ = False
            self.get_logger().info(f'Using local coordinates: ({dest_x:.2f}, {dest_y:.2f}) m')
        else:
            self.destination_ = np.array([0.0, 0.0])
            self.destination_gps_ = None
            self.use_gps_coords_ = False
            self.get_logger().warn('No destination specified, defaulting to origin (0, 0)')
        
        self.target_altitude_ = self.get_parameter('target_altitude').value
        self.target_velocity_ = self.get_parameter('target_velocity').value
        self.altitude_min_ = self.get_parameter('altitude_min').value
        self.altitude_max_ = self.get_parameter('altitude_max').value
        self.waypoint_tolerance_ = self.get_parameter('waypoint_tolerance').value
        
        self.max_heading_rate_ = self.get_parameter('max_heading_rate').value
        self.min_speed_ = self.get_parameter('min_speed').value
        self.max_speed_ = self.get_parameter('max_speed').value
        
        # State variables
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_altitude_ = None
        self.waypoint_reached_ = False
        
        # AI Model and Normalizer
        self.ai_model_ = None
        self.normalizer_ = None
        
        if self.enable_ai_ and AI_MODEL_AVAILABLE:
            try:
                self.ai_model_ = MaridAIModel(model_path=self.model_path_, model_type='pytorch')
                self.get_logger().info(f'AI model loaded from: {self.model_path_}')
            except Exception as e:
                self.get_logger().error(f'Failed to load AI model: {e}')
                self.get_logger().warn('Falling back to PID guidance')
                self.control_mode_ = 'pid'
            
            # Load normalizer if path provided
            if self.normalizer_path_ and len(self.normalizer_path_) > 0:
                try:
                    self.normalizer_ = StateNormalizer.load(self.normalizer_path_)
                    self.get_logger().info(f'State normalizer loaded from: {self.normalizer_path_}')
                except Exception as e:
                    self.get_logger().warn(f'Could not load normalizer from {self.normalizer_path_}: {e}')
                    self.get_logger().warn('Proceeding without normalization - this may cause poor AI performance!')
        elif self.enable_ai_ and not AI_MODEL_AVAILABLE:
            self.get_logger().warn('AI dependencies not available. Using PID guidance.')
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
        
        self.waypoint_sub_ = self.create_subscription(
            PoseStamped,
            '/marid/waypoint',
            self.waypoint_callback,
            10
        )
        
        # Publishers - GUIDANCE TARGETS (not actuator commands!)
        self.desired_heading_rate_pub_ = self.create_publisher(
            Float64,
            '/marid/guidance/desired_heading_rate',
            10
        )
        
        self.desired_speed_pub_ = self.create_publisher(
            Float64,
            '/marid/guidance/desired_speed',
            10
        )
        
        self.guidance_mode_pub_ = self.create_publisher(
            String,
            '/marid/guidance/mode',
            10
        )
        
        self.waypoint_status_pub_ = self.create_publisher(
            Bool,
            '/marid/guidance/waypoint_reached',
            10
        )
        
        # Control timer
        self.control_timer_ = self.create_timer(1.0 / self.update_rate_, self.guidance_loop)
        
        self.get_logger().info('MARID AI Guidance Node initialized (Option A architecture)')
        self.get_logger().info(f'  Control mode: {self.control_mode_}')
        self.get_logger().info(f'  Target waypoint: ({self.destination_[0]:.2f}, {self.destination_[1]:.2f}) m')
        self.get_logger().info(f'  Target speed: {self.target_velocity_:.2f} m/s')
        self.get_logger().info(f'  Target altitude: {self.target_altitude_:.2f} m')
    
    def lat_lon_to_local(self, lat, lon, datum_lat, datum_lon):
        """Convert GPS coordinates to local ENU coordinates"""
        R = 6371000.0  # Earth radius in meters
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        datum_lat_rad = math.radians(datum_lat)
        datum_lon_rad = math.radians(datum_lon)
        
        dlat = lat_rad - datum_lat_rad
        dlon = lon_rad - datum_lon_rad
        
        x = R * dlon * math.cos(datum_lat_rad)
        y = R * dlat
        
        return x, y
    
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
        """Update destination waypoint"""
        self.destination_ = np.array([msg.pose.position.x, msg.pose.position.y])
        if msg.pose.position.z > 0:
            self.target_altitude_ = msg.pose.position.z
        self.waypoint_reached_ = False
        self.get_logger().info(f'New waypoint: ({self.destination_[0]:.2f}, {self.destination_[1]:.2f}) m')
    
    def get_state_vector(self):
        """
        Build base state vector (12-D).
        Returns: numpy array [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
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
    
    def compute_waypoint_heading(self, state):
        """Compute desired heading to waypoint"""
        current_pos = np.array([state[0], state[1]])
        direction = self.destination_ - current_pos
        distance = np.linalg.norm(direction)
        
        # Safety check: avoid division issues when too close
        if distance < 1e-6:
            return None, distance
        
        if distance < self.waypoint_tolerance_:
            return None, distance
        
        # Desired heading to waypoint
        # atan2(y, x) = atan2(north, east) gives angle from East axis (Gazebo/ROS convention)
        # 0°=East, 90°=North, 180°=West, -90°=South
        desired_heading = math.atan2(direction[1], direction[0])
        
        # Safety check: ensure desired_heading is valid
        if not np.isfinite(desired_heading):
            self.get_logger().warn('Invalid desired_heading (NaN/inf), returning None')
            return None, distance
        
        return desired_heading, distance
    
    def compute_pid_guidance(self, state):
        """
        Compute guidance targets using PID logic (fallback when AI unavailable).
        Returns: (desired_heading_rate, desired_speed)
        """
        desired_heading, distance = self.compute_waypoint_heading(state)
        
        if desired_heading is None:
            # Waypoint reached - maintain current heading, slow down
            self.waypoint_reached_ = True
            return 0.0, 0.0
        
        # Compute desired heading rate from heading error
        current_yaw = state[8]
        heading_error = desired_heading - current_yaw
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # Convert heading error to desired heading rate (proportional control)
        # Positive heading rate = turn right, negative = turn left
        heading_rate_gain = 1.0  # rad/s per rad error
        desired_heading_rate = heading_rate_gain * heading_error
        desired_heading_rate = np.clip(desired_heading_rate, -self.max_heading_rate_, self.max_heading_rate_)
        
        # Desired speed based on distance to waypoint
        # Closer to waypoint = slower (for precision landing)
        distance_factor = min(1.0, distance / 100.0)  # Scale down within 100m
        desired_speed = self.target_velocity_ * distance_factor
        desired_speed = np.clip(desired_speed, self.min_speed_, self.max_speed_)
        
        return desired_heading_rate, desired_speed
    
    def compute_ai_guidance(self, state):
        """
        Compute guidance targets using AI model.
        Returns: (desired_heading_rate, desired_speed)
        """
        if self.ai_model_ is None:
            return self.compute_pid_guidance(state)
        
        try:
            desired_heading, distance = self.compute_waypoint_heading(state)
            if desired_heading is None:
                self.waypoint_reached_ = True
                return 0.0, 0.0
            
            # Build extended state vector (20-D)
            current_yaw = state[8]
            heading_error = desired_heading - current_yaw
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi
            
            extended_state = np.concatenate([
                state,  # 12 dimensions
                [self.destination_[0], self.destination_[1]],  # 2 dims → 14
                [distance],  # 1 dim → 15
                [heading_error],  # 1 dim → 16
                [self.altitude_min_, self.altitude_max_, self.target_altitude_],  # 3 dims → 19
                [self.target_velocity_]  # 1 dim → 20 total
            ])
            
            # Validate dimension
            if len(extended_state) != STATE_DIM:
                self.get_logger().error(
                    f'State vector dimension mismatch: expected {STATE_DIM}, got {len(extended_state)}'
                )
                return self.compute_pid_guidance(state)
            
            # Normalize state if normalizer available
            if self.normalizer_ is not None:
                try:
                    extended_state = self.normalizer_.transform(extended_state)
                except Exception as e:
                    self.get_logger().warn(f'Normalization failed: {e}, using unnormalized state')
            
            # Get AI prediction (should output guidance targets, not actuator commands)
            action = self.ai_model_.predict(extended_state)
            
            # AI model outputs [desired_heading_rate, desired_speed] for Option A
            desired_heading_rate = np.clip(action[0], -self.max_heading_rate_, self.max_heading_rate_)
            desired_speed = np.clip(action[1], self.min_speed_, self.max_speed_)
            
            return desired_heading_rate, desired_speed
            
        except ValueError as e:
            self.get_logger().error(f'AI guidance validation failed: {e}')
            return self.compute_pid_guidance(state)
        except Exception as e:
            self.get_logger().error(f'AI guidance prediction failed: {e}')
            return self.compute_pid_guidance(state)
    
    def guidance_loop(self):
        """Main guidance loop - computes and publishes guidance targets"""
        if self.current_odom_ is None:
            return
        
        state = self.get_state_vector()
        
        # Check for invalid state
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            self.get_logger().warn('Invalid state detected (NaN/inf), skipping guidance update')
            return
        
        # Compute guidance targets
        if self.control_mode_ == 'ai' and self.ai_model_ is not None:
            desired_heading_rate, desired_speed = self.compute_ai_guidance(state)
        else:
            desired_heading_rate, desired_speed = self.compute_pid_guidance(state)
            self.control_mode_ = 'pid'  # Ensure mode is set correctly
        
        # Publish guidance targets (NOT actuator commands!)
        heading_rate_msg = Float64()
        heading_rate_msg.data = float(desired_heading_rate)
        self.desired_heading_rate_pub_.publish(heading_rate_msg)
        
        speed_msg = Float64()
        speed_msg.data = float(desired_speed)
        self.desired_speed_pub_.publish(speed_msg)
        
        # Publish guidance mode
        mode_msg = String()
        mode_msg.data = self.control_mode_
        self.guidance_mode_pub_.publish(mode_msg)
        
        # Publish waypoint status
        status_msg = Bool()
        status_msg.data = self.waypoint_reached_
        self.waypoint_status_pub_.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MaridAIGuidance()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
