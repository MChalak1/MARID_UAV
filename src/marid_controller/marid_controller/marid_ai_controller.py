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
import xml.etree.ElementTree as ET

# Import your AI model
try:
    from marid_controller.marid_controller.ai_model import MaridAIModel
    AI_MODEL_AVAILABLE = True
except ImportError:
    AI_MODEL_AVAILABLE = False
    print("Warning: AI model not available. Install PyTorch/TensorFlow and train a model.")


def calculate_total_mass_from_urdf(urdf_string):
    """
    Parse URDF/XACRO string and calculate total mass of all links.
    Returns total mass in kg, or None if parsing fails.
    """
    try:
        root = ET.fromstring(urdf_string)
        
        total_mass = 0.0
        mass_count = 0
        
        # Find all links with inertial properties
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


class MaridAIController(Node):
    def __init__(self):
        super().__init__('marid_ai_controller')
        
        # Parameters
        self.declare_parameter('control_mode', 'ai')  # 'ai', 'pid', 'manual', 'hybrid'
        self.declare_parameter('model_path', '')
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('enable_ai', True)
        self.declare_parameter('enable_pid_fallback', True)
        
        # Waypoint navigation parameters - GPS coordinates (preferred)
        self.declare_parameter('destination_latitude', None)  # Target destination lat (degrees)
        self.declare_parameter('destination_longitude', None)  # Target destination lon (degrees)
        # Local coordinates (backward compatibility)
        self.declare_parameter('destination_x', None)  # Target destination X (m) - optional
        self.declare_parameter('destination_y', None)  # Target destination Y (m) - optional
        
        # Datum (reference point) - should match navsat_transform.yaml
        self.declare_parameter('datum_latitude', 37.45397139527321)  # Reference latitude (degrees)
        self.declare_parameter('datum_longitude', -122.16791304213365)  # Reference longitude (degrees)
        
        # Altitude and velocity parameters
        self.declare_parameter('altitude_min', 3.0)  # Minimum altitude (m)
        self.declare_parameter('altitude_max', 10000.0)  # Maximum altitude (m)
        self.declare_parameter('target_altitude', 8000.0)  # Preferred altitude (m)
        self.declare_parameter('target_velocity', 112.0)  # Average/target speed (m/s)
        self.declare_parameter('waypoint_tolerance', 2.0)  # Distance tolerance for waypoint (m)
        self.declare_parameter('altitude_tolerance', 1.0)  # Altitude tolerance (m)
        
        # Control limits - AUTO-CALCULATION SUPPORT
        self.declare_parameter('min_thrust', 0.0)
        self.declare_parameter('max_thrust', None)  # None = auto-calculate from mass
        self.declare_parameter('thrust_to_weight_ratio', 2.5)  # Thrust-to-weight ratio (e.g., 2.5 = 2.5x weight)
        self.declare_parameter('base_thrust_override', None)  # Override auto-calculation if set (N)
        self.declare_parameter('max_yaw_differential', 0.2)
        
        # Get parameters
        self.control_mode_ = self.get_parameter('control_mode').value
        self.model_path_ = self.get_parameter('model_path').value
        self.update_rate_ = self.get_parameter('update_rate').value
        self.enable_ai_ = self.get_parameter('enable_ai').value
        self.enable_pid_fallback_ = self.get_parameter('enable_pid_fallback').value
        
        # Get datum coordinates
        self.datum_lat_ = self.get_parameter('datum_latitude').value
        self.datum_lon_ = self.get_parameter('datum_longitude').value
        
        # Get waypoint - prefer lat/lon, fall back to x/y
        dest_lat = self.get_parameter('destination_latitude').value
        dest_lon = self.get_parameter('destination_longitude').value
        dest_x = self.get_parameter('destination_x').value
        dest_y = self.get_parameter('destination_y').value
        
        # Determine destination coordinates
        if dest_lat is not None and dest_lon is not None:
            # Use GPS coordinates (preferred)
            x, y = self.lat_lon_to_local(dest_lat, dest_lon, self.datum_lat_, self.datum_lon_)
            self.destination_ = np.array([x, y])
            self.destination_gps_ = (dest_lat, dest_lon)
            self.use_gps_coords_ = True
            self.get_logger().info(f'Using GPS coordinates: ({dest_lat:.6f}°, {dest_lon:.6f}°)')
            self.get_logger().info(f'Converted to local: ({x:.2f}, {y:.2f}) m')
        elif dest_x is not None and dest_y is not None:
            # Use local coordinates (backward compatibility)
            self.destination_ = np.array([dest_x, dest_y])
            self.destination_gps_ = None
            self.use_gps_coords_ = False
            self.get_logger().info(f'Using local coordinates: ({dest_x:.2f}, {dest_y:.2f}) m')
        else:
            # Default to origin if nothing specified
            self.destination_ = np.array([0.0, 0.0])
            self.destination_gps_ = None
            self.use_gps_coords_ = False
            self.get_logger().warn('No destination specified, defaulting to origin (0, 0)')
        
        self.altitude_min_ = self.get_parameter('altitude_min').value
        self.altitude_max_ = self.get_parameter('altitude_max').value
        self.target_altitude_ = self.get_parameter('target_altitude').value
        self.target_velocity_ = self.get_parameter('target_velocity').value
        self.waypoint_tolerance_ = self.get_parameter('waypoint_tolerance').value
        self.altitude_tolerance_ = self.get_parameter('altitude_tolerance').value
        
        # Calculate thrust limits based on aircraft mass
        self.aircraft_mass_ = None
        self.min_thrust_ = self.get_parameter('min_thrust').value
        max_thrust_param = self.get_parameter('max_thrust').value
        base_thrust_override = self.get_parameter('base_thrust_override').value
        thrust_to_weight_ratio = self.get_parameter('thrust_to_weight_ratio').value
        
        # Try to get aircraft mass and calculate max_thrust
        if max_thrust_param is None or base_thrust_override is not None:
            # Auto-calculate from mass
            self.aircraft_mass_ = self.get_aircraft_mass()
            
            if self.aircraft_mass_ is not None:
                if base_thrust_override is not None:
                    # Use override value directly
                    self.max_thrust_ = float(base_thrust_override)
                    self.get_logger().info(f'Aircraft mass: {self.aircraft_mass_:.2f} kg, Using override thrust: {self.max_thrust_:.2f} N')
                else:
                    # Calculate based on thrust-to-weight ratio
                    g = 9.81  # m/s²
                    weight = self.aircraft_mass_ * g  # Weight in Newtons
                    self.max_thrust_ = weight * thrust_to_weight_ratio
                    self.get_logger().info(f'Aircraft mass: {self.aircraft_mass_:.2f} kg')
                    self.get_logger().info(f'Weight: {weight:.2f} N')
                    self.get_logger().info(f'Calculated max_thrust: {self.max_thrust_:.2f} N (T/W ratio: {thrust_to_weight_ratio:.2f})')
            else:
                # Fallback to default if mass calculation fails
                self.max_thrust_ = 200.0 if max_thrust_param is None else float(max_thrust_param)
                self.get_logger().warn(f'Could not determine aircraft mass, using default max_thrust: {self.max_thrust_:.2f} N')
        else:
            # Use explicitly provided max_thrust
            self.max_thrust_ = float(max_thrust_param)
            self.get_logger().info(f'Using explicit max_thrust: {self.max_thrust_:.2f} N')
        
        self.max_yaw_differential_ = self.get_parameter('max_yaw_differential').value
        
        # State variables
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_altitude_ = None
        self.waypoint_reached_ = False
        
        # PID controllers - update output limits to match max_thrust
        self.altitude_pid_ = PIDController(kp=2.0, ki=0.1, kd=0.5, output_limits=(0.0, self.max_thrust_))
        self.velocity_pid_ = PIDController(kp=1.0, ki=0.05, kd=0.3, output_limits=(0.0, self.max_thrust_))
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
        if self.use_gps_coords_ and self.destination_gps_ is not None:
            self.get_logger().info(f'Destination (GPS): ({self.destination_gps_[0]:.6f}°, {self.destination_gps_[1]:.6f}°)')
        self.get_logger().info(f'Destination (local): ({self.destination_[0]:.2f}, {self.destination_[1]:.2f}) m')
        self.get_logger().info(f'Datum: ({self.datum_lat_:.6f}°, {self.datum_lon_:.6f}°)')
        self.get_logger().info(f'Altitude range: {self.altitude_min_:.2f} - {self.altitude_max_:.2f} m')
        self.get_logger().info(f'Target altitude: {self.target_altitude_:.2f} m')
        self.get_logger().info(f'Target velocity: {self.target_velocity_:.2f} m/s')
        self.get_logger().info(f'Thrust range: {self.min_thrust_:.2f} - {self.max_thrust_:.2f} N')
    
    def get_aircraft_mass(self):
        """
        Get total aircraft mass from URDF file.
        Reads the URDF/XACRO file directly and calculates total mass.
        Returns mass in kg, or None if unavailable.
        """
        try:
            from ament_index_python.packages import get_package_share_directory
            import os
            import subprocess
            
            marid_description_dir = get_package_share_directory('marid_description')
            urdf_path = os.path.join(marid_description_dir, 'urdf', 'marid.urdf.xacro')
            
            if not os.path.exists(urdf_path):
                self.get_logger().warn(f'URDF file not found: {urdf_path}')
                return None
            
            # Process XACRO file to get expanded URDF
            # xacro command expands all includes and macros
            try:
                result = subprocess.run(
                    ['xacro', urdf_path],
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                
                if result.returncode == 0 and result.stdout:
                    expanded_urdf = result.stdout
                    mass = calculate_total_mass_from_urdf(expanded_urdf)
                    if mass is not None and mass > 0:
                        return mass
                    else:
                        self.get_logger().warn(f'Could not parse mass from URDF (got {mass})')
                else:
                    self.get_logger().warn(f'xacro command failed: {result.stderr}')
            except FileNotFoundError:
                self.get_logger().warn('xacro command not found. Install with: sudo apt install ros-jazzy-xacro')
            except subprocess.TimeoutExpired:
                self.get_logger().warn('xacro command timed out')
            except Exception as e:
                self.get_logger().warn(f'Error running xacro: {e}')
            
            # Fallback: Try to parse XACRO directly (may not work with includes)
            try:
                with open(urdf_path, 'r') as f:
                    urdf_content = f.read()
                mass = calculate_total_mass_from_urdf(urdf_content)
                if mass is not None and mass > 0:
                    self.get_logger().info('Parsed mass from XACRO file directly (may be incomplete)')
                    return mass
            except Exception as e:
                self.get_logger().debug(f'Could not parse XACRO directly: {e}')
            
            return None
        except Exception as e:
            self.get_logger().warn(f'Could not get aircraft mass: {e}')
            return None
    
    def lat_lon_to_local(self, lat, lon, datum_lat, datum_lon):
        """
        Convert latitude/longitude to local x/y coordinates (meters) relative to datum.
        Uses simple flat-earth approximation (accurate for small areas < 100km).
        
        Args:
            lat: Target latitude (degrees)
            lon: Target longitude (degrees)
            datum_lat: Reference latitude (degrees) - from navsat_transform.yaml
            datum_lon: Reference longitude (degrees) - from navsat_transform.yaml
        
        Returns:
            (x, y) tuple in meters relative to datum
            x is East (positive = east of datum)
            y is North (positive = north of datum)
        """
        # Earth radius in meters
        R = 6371000.0
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        datum_lat_rad = math.radians(datum_lat)
        datum_lon_rad = math.radians(datum_lon)
        
        # Calculate differences
        dlat = lat_rad - datum_lat_rad
        dlon = lon_rad - datum_lon_rad
        
        # Flat-earth approximation (accurate for small distances < 100km)
        # x is East (longitude difference)
        x = R * dlon * math.cos(datum_lat_rad)
        # y is North (latitude difference)
        y = R * dlat
        
        return x, y
    
    def local_to_lat_lon(self, x, y, datum_lat, datum_lon):
        """
        Convert local x/y coordinates to latitude/longitude.
        Inverse of lat_lon_to_local().
        
        Args:
            x: East offset in meters (positive = east)
            y: North offset in meters (positive = north)
            datum_lat: Reference latitude (degrees)
            datum_lon: Reference longitude (degrees)
        
        Returns:
            (lat, lon) tuple in degrees
        """
        # Earth radius in meters
        R = 6371000.0
        
        # Convert datum to radians
        datum_lat_rad = math.radians(datum_lat)
        datum_lon_rad = math.radians(datum_lon)
        
        # Convert local coordinates to lat/lon differences
        dlat = y / R  # North component
        dlon = x / (R * math.cos(datum_lat_rad))  # East component
        
        # Add to datum
        lat_rad = datum_lat_rad + dlat
        lon_rad = datum_lon_rad + dlon
        
        # Convert back to degrees
        lat = math.degrees(lat_rad)
        lon = math.degrees(lon_rad)
        
        return lat, lon
    
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
        """
        Update destination waypoint dynamically.
        Supports both local coordinates (x, y) and GPS coordinates via custom fields.
        For GPS coordinates, use frame_id to indicate GPS mode, or extend with custom message.
        """
        # For now, assume PoseStamped uses x/y in local frame
        # You could extend this to check frame_id or use a custom message type
        self.destination_ = np.array([msg.pose.position.x, msg.pose.position.y])
        
        # If waypoint is provided in local coordinates, convert to GPS for logging if datum is available
        if self.datum_lat_ is not None and self.datum_lon_ is not None:
            lat, lon = self.local_to_lat_lon(
                msg.pose.position.x, 
                msg.pose.position.y,
                self.datum_lat_,
                self.datum_lon_
            )
            self.destination_gps_ = (lat, lon)
            self.use_gps_coords_ = False  # Waypoint came in local coords
            self.get_logger().info(f'New waypoint set (local): ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) m')
            self.get_logger().info(f'Waypoint GPS equivalent: ({lat:.6f}°, {lon:.6f}°)')
        else:
            self.get_logger().info(f'New waypoint set: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) m')
        
        if msg.pose.position.z > 0:
            self.target_altitude_ = msg.pose.position.z
        self.waypoint_reached_ = False
    
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
