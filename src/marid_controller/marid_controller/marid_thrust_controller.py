#!/usr/bin/env python3
"""
MARID Thrust Controller
Applies thrust forces to individual thruster links to counter gravity and provide propulsion.
Uses service-based approach: publishes directly to Gazebo using gz topic commands.
Keyboard control: Up arrow = +10N, Down arrow = -10N (0-300N range)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import threading
import sys
import subprocess
import os
import time
import math
import xml.etree.ElementTree as ET


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

# Try to import pynput for keyboard control
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Install with: pip install pynput")


class MaridThrustController(Node):
    def __init__(self):
        super().__init__('marid_thrust_controller')
        
        # Parameters
        self.declare_parameter('initial_thrust', 1.0)        # Initial thrust in Newtons
        self.declare_parameter('min_thrust', 0.0)             # Minimum thrust (N)
        self.declare_parameter('max_thrust', 200.0)          # Maximum thrust (N) - can be None for auto-calculation
        self.declare_parameter('thrust_to_weight_ratio', 2.5)  # Thrust-to-weight ratio (if max_thrust is None)
        self.declare_parameter('base_thrust_override', -1.0)  # Override auto-calculation if set (N), -1.0 means not set
        self.declare_parameter('thrust_increment', 1.0)       # Thrust increment per keypress (N)
        self.declare_parameter('world_name', 'wt')            # Gazebo world name
        self.declare_parameter('model_name', 'marid')         # Model name
        self.declare_parameter('link_name', 'base_link_front') # Link to apply force to
        self.declare_parameter('update_rate', 50.0)           # Update rate for thruster commands (Hz)
        self.declare_parameter('enable_keyboard', True)       # Enable keyboard control
        self.declare_parameter('enable_differential', False)  # Enable differential thrust for yaw control
        self.declare_parameter('thrust_to_angvel_gain', 50.0)  # Conversion factor: omega = gain * sqrt(thrust)
        self.declare_parameter('use_thruster_plugin', True)    # Use Gazebo Thruster plugin (True) or legacy wrench (False)
        self.declare_parameter('use_center_thruster', False)   # Use single center thruster (True) or dual left/right (False)
        self.declare_parameter('thrust_rate_limit', 50.0)       # Max change rate (N/s) for smooth transitions
        self.declare_parameter('thrust_smoothing_factor', 0.1)  # Exponential smoothing: 0.0 = no smoothing, 1.0 = full smoothing
        
        self.initial_thrust_ = self.get_parameter('initial_thrust').value
        self.min_thrust_ = self.get_parameter('min_thrust').value
        
        # Calculate max_thrust if needed (same logic as AI controller)
        max_thrust_param = self.get_parameter('max_thrust').value
        base_thrust_override = self.get_parameter('base_thrust_override').value
        thrust_to_weight_ratio = self.get_parameter('thrust_to_weight_ratio').value
        
        if max_thrust_param is None or base_thrust_override != -1.0:
            # Auto-calculate from mass
            aircraft_mass = self.get_aircraft_mass()
            if aircraft_mass is not None:
                if base_thrust_override != -1.0:
                    self.max_thrust_ = float(base_thrust_override)
                    self.get_logger().info(f'Thrust Controller: Aircraft mass: {aircraft_mass:.2f} kg, Using override thrust: {self.max_thrust_:.2f} N')
                else:
                    g = 9.81
                    weight = aircraft_mass * g
                    self.max_thrust_ = weight * thrust_to_weight_ratio
                    self.get_logger().info(f'Thrust Controller: Aircraft mass: {aircraft_mass:.2f} kg, Calculated max_thrust: {self.max_thrust_:.2f} N')
            else:
                self.max_thrust_ = 200.0 if max_thrust_param is None else float(max_thrust_param)
                self.get_logger().warn(f'Thrust Controller: Could not determine mass, using default max_thrust: {self.max_thrust_:.2f} N')
        else:
            self.max_thrust_ = float(max_thrust_param)
        
        self.thrust_increment_ = self.get_parameter('thrust_increment').value
        self.world_name_ = self.get_parameter('world_name').value
        self.model_name_ = self.get_parameter('model_name').value
        self.link_name_ = self.get_parameter('link_name').value
        self.enable_keyboard_ = self.get_parameter('enable_keyboard').value
        self.enable_differential_ = self.get_parameter('enable_differential').value
        self.thrust_to_angvel_gain_ = self.get_parameter('thrust_to_angvel_gain').value
        self.use_thruster_plugin_ = self.get_parameter('use_thruster_plugin').value
        self.use_center_thruster_ = self.get_parameter('use_center_thruster').value
        self.thrust_rate_limit_ = self.get_parameter('thrust_rate_limit').value
        self.thrust_smoothing_factor_ = self.get_parameter('thrust_smoothing_factor').value
        
        # Current thrust values in Newtons (absolute, not ratio)
        self.current_thrust_ = self.initial_thrust_
        self.left_thrust_ = self.initial_thrust_
        self.right_thrust_ = self.initial_thrust_
        
        # Target thrust values (what we want to achieve)
        self.target_left_thrust_ = self.initial_thrust_
        self.target_right_thrust_ = self.initial_thrust_
        self.target_center_thrust_ = self.initial_thrust_  # For center thruster mode
        
        # Last published values (for rate limiting and smoothing)
        self.last_published_left_ = self.initial_thrust_
        self.last_published_right_ = self.initial_thrust_
        self.last_published_center_ = self.initial_thrust_  # For center thruster mode
        
        # Lock for thread-safe access to thrust values
        self.thrust_lock_ = threading.Lock()
        
        # Differential thrust for yaw control (will be set by attitude controller)
        self.yaw_differential_ = 0.0  # -1.0 to 1.0, negative = yaw left, positive = yaw right
        
        # Publishers: Thruster plugin (force in N) for center and L/R
        if self.use_thruster_plugin_:
            self.thrust_center_pub_ = self.create_publisher(
                Float64,
                '/model/marid/joint/thruster_center_joint/cmd_thrust',
                10
            )
            self.thrust_left_pub_ = self.create_publisher(
                Float64,
                '/model/marid/joint/thruster_L_joint/cmd_thrust',
                10
            )
            self.thrust_right_pub_ = self.create_publisher(
                Float64,
                '/model/marid/joint/thruster_R_joint/cmd_thrust',
                10
            )
            if self.use_center_thruster_:
                self.get_logger().info('Using Thruster plugin - CENTER (force in N)')
            else:
                self.get_logger().info('Using Thruster plugin - DUAL (left + right, force in N)')
        else:
            # Legacy: Use ApplyLinkWrench plugin (in world) via direct gz topic publishing
            # Use persistent topic (applies continuously until cleared or updated)
            self.wrench_topic_ = f'/world/{self.world_name_}/wrench/persistent'
            self.get_logger().info('Using legacy ApplyLinkWrench plugin (wrench commands)')
        
        # Subscribe to thrust commands (in Newtons)
        self.left_thrust_sub_ = self.create_subscription(
            Float64,
            '/marid/thrust/left',
            self.left_thrust_callback,
            10
        )
        
        self.right_thrust_sub_ = self.create_subscription(
            Float64,
            '/marid/thrust/right',
            self.right_thrust_callback,
            10
        )
        
        # Subscribe to combined thrust command (applied to both thrusters equally, in Newtons)
        self.total_thrust_sub_ = self.create_subscription(
            Float64,
            '/marid/thrust/total',
            self.total_thrust_callback,
            10
        )
        
        # Subscribe to yaw differential command (for differential thrust control)
        self.yaw_differential_sub_ = self.create_subscription(
            Float64,
            '/marid/thrust/yaw_differential',
            self.yaw_differential_callback,
            10
        )
        
        # Subscribe to odometry for altitude/velocity feedback (optional, for future auto-thrust)
        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/odometry/filtered/local',
            self.odom_callback,
            10
        )
        
        # Current state (for future altitude/velocity control)
        self.current_altitude_ = 0.0
        self.current_velocity_ = 0.0
        
        # Timer to apply wrench commands via gz topic
        # Start with a delay to ensure model is spawned first
        update_period = 1.0 / self.get_parameter('update_rate').value
        
        # Flag to track if we've started applying thrust
        self.thrust_started_ = False
        self.start_time_ = self.get_clock().now()
        self.start_delay_ = 4.0  # Wait 4 seconds before starting
        
        # Simple approach: wait 4 seconds (model spawns after 3 seconds in launch file)
        # Then start applying thrust
        self.get_logger().info('Will start applying thrust after 4 seconds...')
        
        # Create a timer that checks if delay has passed (runs every 0.1s until started)
        self.create_timer(0.1, self.check_and_start_thrust)
        
        # Create the main thrust application timer (it will check the flag before applying)
        self.thrust_timer_ = self.create_timer(update_period, self.apply_thrust)
        
        # Start keyboard listener in a separate thread
        self.keyboard_listener_ = None
        if self.enable_keyboard_ and PYNPUT_AVAILABLE:
            self.start_keyboard_listener()
        elif self.enable_keyboard_ and not PYNPUT_AVAILABLE:
            self.get_logger().warn('Keyboard control requested but pynput not available. Install with: pip install pynput')
        
        self.get_logger().info('MARID Thrust Controller initialized (Service-based)')
        self.get_logger().info(f'World: {self.world_name_}, Model: {self.model_name_}, Link: {self.link_name_}')
        self.get_logger().info(f'Thruster mode: {"CENTER (single)" if self.use_center_thruster_ else "DUAL (left/right)"}')
        self.get_logger().info(f'Initial thrust: {self.initial_thrust_:.2f} N {"(center)" if self.use_center_thruster_ else "per thruster"}')
        self.get_logger().info(f'Thrust range: {self.min_thrust_:.2f} - {self.max_thrust_:.2f} N')
        self.get_logger().info(f'Thrust increment: {self.thrust_increment_:.2f} N per keypress')
        if self.enable_keyboard_ and PYNPUT_AVAILABLE:
            self.get_logger().info('Keyboard control: UP arrow = +1N, DOWN arrow = -1N')
        self.get_logger().info(f'Differential thrust: {"enabled" if self.enable_differential_ else "disabled"}')
        self.get_logger().info(f'Thrust rate limit: {self.thrust_rate_limit_:.1f} N/s')
        self.get_logger().info(f'Thrust smoothing factor: {self.thrust_smoothing_factor_:.2f}')
        if self.use_thruster_plugin_:
            self.get_logger().info('Publishing force (N) to Gazebo Thruster plugins (cmd_thrust)')
        else:
            self.get_logger().info(f'Using ApplyLinkWrench plugin via: {self.wrench_topic_}')
            self.get_logger().info(f'Entity ID: 11 (base_link_front)')
            self.get_logger().info('Publishing EntityWrench messages directly to Gazebo using gz topic commands')
        self.get_logger().info('Checking for model spawn before applying thrust...')
    
    def start_keyboard_listener(self):
        """Start keyboard listener in a separate thread"""
        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.increment_thrust()
                elif key == keyboard.Key.down:
                    self.decrement_thrust()
            except AttributeError:
                # Special keys (like arrow keys) handled above
                pass
        
        def on_release(key):
            # Don't do anything on release
            pass
        
        # Start listener in a separate thread
        self.keyboard_listener_ = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.keyboard_listener_.start()
        self.get_logger().info('Keyboard listener started')
    
    def increment_thrust(self):
        """Increment thrust target by thrust_increment_"""
        with self.thrust_lock_:
            new_thrust = min(self.current_thrust_ + self.thrust_increment_, self.max_thrust_)
            if new_thrust != self.current_thrust_:
                self.current_thrust_ = new_thrust
                if self.use_center_thruster_:
                    self.target_center_thrust_ = new_thrust
                else:
                    self.target_left_thrust_ = new_thrust
                    self.target_right_thrust_ = new_thrust
                self.get_logger().info(f'Thrust target increased to {self.current_thrust_:.2f} N')
    
    def decrement_thrust(self):
        """Decrement thrust target by thrust_increment_"""
        with self.thrust_lock_:
            new_thrust = max(self.current_thrust_ - self.thrust_increment_, self.min_thrust_)
            if new_thrust != self.current_thrust_:
                self.current_thrust_ = new_thrust
                if self.use_center_thruster_:
                    self.target_center_thrust_ = new_thrust
                else:
                    self.target_left_thrust_ = new_thrust
                    self.target_right_thrust_ = new_thrust
                self.get_logger().info(f'Thrust target decreased to {self.current_thrust_:.2f} N')
    
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
    
    def left_thrust_callback(self, msg):
        """Set left thruster target (in Newtons)"""
        with self.thrust_lock_:
            self.target_left_thrust_ = max(self.min_thrust_, min(self.max_thrust_, msg.data))
            self.get_logger().debug(f'Left thrust target: {self.target_left_thrust_:.2f} N')
    
    def right_thrust_callback(self, msg):
        """Set right thruster target (in Newtons)"""
        with self.thrust_lock_:
            self.target_right_thrust_ = max(self.min_thrust_, min(self.max_thrust_, msg.data))
            self.get_logger().debug(f'Right thrust target: {self.target_right_thrust_:.2f} N')
    
    def total_thrust_callback(self, msg):
        """Set both thrusters to the same target value (in Newtons)"""
        with self.thrust_lock_:
            thrust = max(self.min_thrust_, min(self.max_thrust_, msg.data))
            self.current_thrust_ = thrust
            if self.use_center_thruster_:
                self.target_center_thrust_ = thrust  # Set center target for center thruster mode
            else:
                self.target_left_thrust_ = thrust
                self.target_right_thrust_ = thrust
            self.get_logger().debug(f'Total thrust target: {thrust:.2f} N')
    
    def yaw_differential_callback(self, msg):
        """Set yaw differential (-1.0 to 1.0)
        Positive = more left thrust (turn right / positive yaw)
        Negative = more right thrust (turn left / negative yaw)
        """
        if self.enable_differential_:
            self.yaw_differential_ = max(-1.0, min(1.0, msg.data))
            self.get_logger().debug(f'Yaw differential: {self.yaw_differential_:.3f}')
    
    def odom_callback(self, msg):
        """Store current altitude and velocity for future auto-thrust control"""
        self.current_altitude_ = msg.pose.pose.position.z
        linear_vel = msg.twist.twist.linear
        self.current_velocity_ = (linear_vel.x**2 + linear_vel.y**2 + linear_vel.z**2)**0.5
    
    def calculate_thrust_values(self):
        """Calculate thrust target values - returns center thrust or (left, right) depending on mode"""
        with self.thrust_lock_:
            if self.use_center_thruster_:
                # Center thruster mode - return single value
                center_target = max(self.min_thrust_, min(self.max_thrust_, self.target_center_thrust_))
                return center_target, None  # Return center, None for right
            else:
                # Dual thruster mode
                if self.enable_differential_:
                    # Apply differential thrust for yaw control while maintaining total thrust
                    # Total thrust = left + right should remain constant
                    base_thrust = (self.target_left_thrust_ + self.target_right_thrust_) / 2.0
                    differential_gain = 0.3  # Max 30% differential (reduced from 100% for stability)
                    differential = self.yaw_differential_ * differential_gain * base_thrust
                    
                    # Maintain symmetry: left + right = 2 * base_thrust
                    # Positive differential → more left thrust → turn right
                    # Negative differential → more right thrust → turn left
                    left_target = base_thrust + differential
                    right_target = base_thrust - differential
                else:
                    # Equal thrust to both
                    left_target = self.target_left_thrust_
                    right_target = self.target_right_thrust_
                
                # Clamp to valid range
                left_target = max(self.min_thrust_, min(self.max_thrust_, left_target))
                right_target = max(self.min_thrust_, min(self.max_thrust_, right_target))
                
                return left_target, right_target
    
    def check_and_start_thrust(self):
        """Check if delay has passed and start applying thrust"""
        if self.thrust_started_:
            return  # Already started
        
        elapsed = (self.get_clock().now() - self.start_time_).nanoseconds / 1e9
        
        if elapsed >= self.start_delay_:
            self.get_logger().info(f'=== STARTING THRUST APPLICATION (after {elapsed:.1f}s) ===')
            self.thrust_started_ = True
            self.get_logger().info('Thrust flag set to True. Applying initial thrust...')
            # Apply initial thrust immediately
            self.apply_thrust()
            self.get_logger().info('Initial thrust call completed. Timer will continue at 10 Hz.')
    
    def apply_thrust(self):
        """Apply thrust with rate limiting and smoothing using Gazebo Thruster plugin or legacy wrench"""
        # Don't apply thrust until model is spawned
        if not self.thrust_started_:
            return
        
        # Get target values (center thruster or left/right depending on mode)
        thrust_values = self.calculate_thrust_values()
        
        if self.use_center_thruster_:
            # Center thruster mode
            target_center = thrust_values[0]
            
            # Calculate time delta for rate limiting
            if not hasattr(self, '_last_update_time'):
                self._last_update_time = self.get_clock().now()
                dt = 0.02  # Assume 50 Hz = 0.02s
            else:
                current_time = self.get_clock().now()
                dt = (current_time - self._last_update_time).nanoseconds / 1e9
                self._last_update_time = current_time
                
                # Clamp dt to prevent issues from system lag
                dt = max(0.001, min(0.1, dt))  # Between 1ms and 100ms
            
            # Apply rate limiting
            max_change = self.thrust_rate_limit_ * dt
            center_change = target_center - self.last_published_center_
            
            if abs(center_change) > max_change:
                center_change = max_change if center_change > 0 else -max_change
            
            # Apply exponential smoothing
            if self.thrust_smoothing_factor_ > 0:
                alpha = self.thrust_smoothing_factor_
                center_thrust = self.last_published_center_ + alpha * (target_center - self.last_published_center_)
            else:
                center_thrust = self.last_published_center_ + center_change
            
            # Update last published value
            self.last_published_center_ = center_thrust
            
            if self.use_thruster_plugin_:
                # Thruster plugin: publish force in N (axis 0 1 0 = force +Y forward)
                msg = Float64()
                msg.data = float(max(0.0, max(self.min_thrust_, min(self.max_thrust_, center_thrust))))
                self.thrust_center_pub_.publish(msg)
                self.thrust_left_pub_.publish(Float64(data=0.0))
                self.thrust_right_pub_.publish(Float64(data=0.0))
            return  # Exit early for center thruster mode
        
        # Dual thruster mode (original logic)
        target_left, target_right = thrust_values
        
        # Calculate time delta for rate limiting
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = self.get_clock().now()
            dt = 0.02  # Assume 50 Hz = 0.02s
        else:
            current_time = self.get_clock().now()
            dt = (current_time - self._last_update_time).nanoseconds / 1e9
            self._last_update_time = current_time
            
            # Clamp dt to prevent issues from system lag
            dt = max(0.001, min(0.1, dt))  # Between 1ms and 100ms
        
        # Apply rate limiting
        max_change = self.thrust_rate_limit_ * dt
        left_change = target_left - self.last_published_left_
        right_change = target_right - self.last_published_right_
        
        if abs(left_change) > max_change:
            left_change = max_change if left_change > 0 else -max_change
        if abs(right_change) > max_change:
            right_change = max_change if right_change > 0 else -max_change
        
        # Apply exponential smoothing (optional, can be disabled by setting factor to 0)
        if self.thrust_smoothing_factor_ > 0:
            alpha = self.thrust_smoothing_factor_  # 0.0 = no smoothing, 1.0 = full smoothing
            left_thrust = self.last_published_left_ + alpha * (target_left - self.last_published_left_)
            right_thrust = self.last_published_right_ + alpha * (target_right - self.last_published_right_)
        else:
            # Just rate limiting, no smoothing
            left_thrust = self.last_published_left_ + left_change
            right_thrust = self.last_published_right_ + right_change
        
        # Update last published values
        self.last_published_left_ = left_thrust
        self.last_published_right_ = right_thrust
        
        if self.use_thruster_plugin_:
            self.thrust_center_pub_.publish(Float64(data=0.0))
            self.thrust_left_pub_.publish(Float64(data=float(max(0.0, left_thrust))))
            self.thrust_right_pub_.publish(Float64(data=float(max(0.0, right_thrust))))
            
            if not hasattr(self, '_thruster_log_counter'):
                self._thruster_log_counter = 0
            self._thruster_log_counter += 1
            if self._thruster_log_counter == 1:
                self.get_logger().info('Thrust commands published (Thruster plugin, force in N).')
            elif self._thruster_log_counter % 50 == 0:
                self.get_logger().debug(
                    f'Thrust: L={left_thrust:.2f}N, R={right_thrust:.2f}N'
                )
        else:
            # ===== LEGACY: Use ApplyLinkWrench Plugin =====
            total_thrust = left_thrust + right_thrust
            
            # Calculate yaw torque from differential thrust
            # Differential creates a moment arm effect for steering
            # If left_thrust > right_thrust, we want positive yaw (turn right)
            # If right_thrust > left_thrust, we want negative yaw (turn left)
            # Note: thrust_differential = right - left, so we need to negate it for correct yaw direction
            thrust_differential = right_thrust - left_thrust
            # Convert differential to yaw torque
            # Assume thrusters are ~0.5m apart (adjust based on your model)
            thruster_separation = 0.5  # meters (distance between left and right thrusters)
            # Scale factor to convert force difference to torque (adjust for your model)
            # Negate because: left_thrust > right_thrust → negative differential → should give positive yaw
            yaw_torque_z = -thrust_differential * thruster_separation * 0.5
            
            # Create EntityWrench message in protobuf text format (not JSON!)
            # gz topic -p expects protobuf DebugString format
            # Use entity ID instead of name (more reliable)
            # Entity ID 11 = base_link_front (from gz model -m marid -l output)
            # If entity ID changes, we can query it dynamically, but for now use fixed ID
            entity_id = 11  # base_link_front link ID
            
            # Protobuf text format using entity ID
            # To apply force at center of mass (offset: 0, 0.1, 0.02 from link origin),
            # we need to add compensating torque: torque = offset × force
            # offset = (0, 0.1, 0.02), force = (0, total_thrust, 0)
            # torque = offset × force = (0.1*0 - 0.02*total_thrust, 0.02*0 - 0*0, 0*0 - 0.1*0)
            # torque = (-0.02*total_thrust, 0, 0)  [cross product]
            # Actually: torque_x = offset_y * force_z - offset_z * force_y = 0.1*0 - 0.02*total_thrust = -0.02*total_thrust
            #          torque_y = offset_z * force_x - offset_x * force_z = 0.02*0 - 0*0 = 0
            #          torque_z = offset_x * force_y - offset_y * force_x = 0*total_thrust - 0.1*0 = 0
            compensating_torque_x = -0.02 * total_thrust  # Negative because force is +Y, offset is +Z
            
            entity_wrench_proto = f'''entity {{
  id: {entity_id}
}}
wrench {{
  force {{
    x: 0.0
    y: {total_thrust}
    z: 0.0
  }}
  torque {{
    x: {compensating_torque_x}
    y: 0.0
    z: {yaw_torque_z}
  }}
}}'''
            
            # Use gz topic pub to publish directly to Gazebo (bypasses bridge issues)
            # Retry up to 3 times with increasing timeout
            max_retries = 3
            timeout = 5.0  # Increased from 2.0 to 5.0 seconds
            
            for attempt in range(max_retries):
                try:
                    # Ensure we have the full path to gz or use shell=True
                    env = dict(os.environ)
                    # Use shell=True to ensure proper PATH resolution and environment
                    cmd_str = f'gz topic -t {self.wrench_topic_} -m gz.msgs.EntityWrench -p \'{entity_wrench_proto}\''
                    
                    result = subprocess.run(
                        cmd_str,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=env,
                        executable='/bin/bash'
                    )
                    
                    if result.returncode == 0:
                        # Command succeeded
                        if not hasattr(self, '_success_counter'):
                            self._success_counter = 0
                        self._success_counter += 1
                        if self._success_counter == 1:
                            self.get_logger().info('gz topic command succeeded! Thrust should be applied.')
                        elif self._success_counter % 50 == 0:
                            self.get_logger().debug(f'gz topic command succeeded ({self._success_counter} times)')
                        return  # Success, exit retry loop
                    else:
                        # Command failed but didn't timeout
                        if attempt == max_retries - 1:  # Last attempt
                            self.get_logger().error(f'gz topic command failed after {max_retries} attempts (code {result.returncode})')
                            if result.stderr:
                                self.get_logger().error(f'stderr: {result.stderr[:300]}')
                            if result.stdout:
                                self.get_logger().error(f'stdout: {result.stdout[:300]}')
                            if not hasattr(self, '_cmd_logged'):
                                self.get_logger().error(f'Failed command: {cmd_str[:200]}')
                                self.get_logger().error(f'Message payload: {entity_wrench_proto[:200]}')
                                self._cmd_logged = True
                        else:
                            self.get_logger().warn(f'gz topic command failed (attempt {attempt + 1}/{max_retries}), retrying...')
                            time.sleep(0.5)  # Brief delay before retry
                            continue
                    
                except subprocess.TimeoutExpired:
                    if attempt == max_retries - 1:  # Last attempt
                        self.get_logger().error(f'gz topic command timed out after {max_retries} attempts. Is Gazebo running?')
                        if not hasattr(self, '_timeout_logged'):
                            self.get_logger().error('Make sure Gazebo is running and the model is spawned before starting the thrust controller.')
                            self._timeout_logged = True
                    else:
                        self.get_logger().warn(f'gz topic command timed out (attempt {attempt + 1}/{max_retries}), retrying with longer timeout...')
                        timeout += 2.0  # Increase timeout for next attempt
                        time.sleep(0.5)
                        continue
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        if not hasattr(self, '_exception_logged'):
                            self.get_logger().error(f'Error publishing wrench after {max_retries} attempts: {str(e)[:200]}')
                            self._exception_logged = True
                    else:
                        self.get_logger().warn(f'Error on attempt {attempt + 1}/{max_retries}: {str(e)[:100]}, retrying...')
                        time.sleep(0.5)
                        continue
            
            # If we get here, all retries failed
            # Log periodically (every 5 seconds at 10 Hz = every 50 calls)
            if hasattr(self, '_log_counter'):
                self._log_counter += 1
            else:
                self._log_counter = 0
            
            if self._log_counter % 50 == 0:
                with self.thrust_lock_:
                    current = self.current_thrust_
                self.get_logger().warn(
                    f'Failed to apply thrust via ApplyLinkWrench: L={left_thrust:.2f}N, R={right_thrust:.2f}N, Total={total_thrust:.2f}N (Current: {current:.2f}N)'
                )
                self.get_logger().warn(f'Topic: {self.wrench_topic_}, Entity ID: {entity_id} (base_link_front)')
                self.get_logger().warn('Check that Gazebo is running and the model is spawned.')
    
    def destroy_node(self):
        """Clean up keyboard listener on shutdown"""
        if self.keyboard_listener_ is not None:
            self.keyboard_listener_.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = MaridThrustController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
