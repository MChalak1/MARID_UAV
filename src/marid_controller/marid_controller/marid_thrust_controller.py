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
        self.declare_parameter('base_thrust_override', None)  # Override auto-calculation if set (N)
        self.declare_parameter('thrust_increment', 1.0)       # Thrust increment per keypress (N)
        self.declare_parameter('world_name', 'wt')            # Gazebo world name
        self.declare_parameter('model_name', 'marid')         # Model name
        self.declare_parameter('link_name', 'base_link_front') # Link to apply force to
        self.declare_parameter('update_rate', 10.0)           # Update rate for persistent wrench (Hz)
        self.declare_parameter('enable_keyboard', True)       # Enable keyboard control
        self.declare_parameter('enable_differential', False)  # Enable differential thrust for yaw control
        
        self.initial_thrust_ = self.get_parameter('initial_thrust').value
        self.min_thrust_ = self.get_parameter('min_thrust').value
        
        # Calculate max_thrust if needed (same logic as AI controller)
        max_thrust_param = self.get_parameter('max_thrust').value
        base_thrust_override = self.get_parameter('base_thrust_override').value
        thrust_to_weight_ratio = self.get_parameter('thrust_to_weight_ratio').value
        
        if max_thrust_param is None or base_thrust_override is not None:
            # Auto-calculate from mass
            aircraft_mass = self.get_aircraft_mass()
            if aircraft_mass is not None:
                if base_thrust_override is not None:
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
        
        # Current thrust values in Newtons (absolute, not ratio)
        self.current_thrust_ = self.initial_thrust_
        self.left_thrust_ = self.initial_thrust_
        self.right_thrust_ = self.initial_thrust_
        
        # Lock for thread-safe access to thrust values
        self.thrust_lock_ = threading.Lock()
        
        # Differential thrust for yaw control (will be set by attitude controller)
        self.yaw_differential_ = 0.0  # -1.0 to 1.0, negative = yaw left, positive = yaw right
        
        # Use ApplyLinkWrench plugin (in world) via direct gz topic publishing
        # Use persistent topic (applies continuously until cleared or updated)
        self.world_name_ = self.get_parameter('world_name').value
        self.wrench_topic_ = f'/world/{self.world_name_}/wrench/persistent'
        
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
        self.get_logger().info(f'Initial thrust: {self.initial_thrust_:.2f} N per thruster')
        self.get_logger().info(f'Thrust range: {self.min_thrust_:.2f} - {self.max_thrust_:.2f} N')
        self.get_logger().info(f'Thrust increment: {self.thrust_increment_:.2f} N per keypress')
        if self.enable_keyboard_ and PYNPUT_AVAILABLE:
            self.get_logger().info('Keyboard control: UP arrow = +1N, DOWN arrow = -1N')
        self.get_logger().info(f'Differential thrust: {"enabled" if self.enable_differential_ else "disabled"}')
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
        """Increment thrust by thrust_increment_"""
        with self.thrust_lock_:
            new_thrust = min(self.current_thrust_ + self.thrust_increment_, self.max_thrust_)
            if new_thrust != self.current_thrust_:
                self.current_thrust_ = new_thrust
                self.left_thrust_ = new_thrust
                self.right_thrust_ = new_thrust
                self.get_logger().info(f'Thrust increased to {self.current_thrust_:.2f} N')
                # Immediately apply the new thrust
                self.apply_thrust()
    
    def decrement_thrust(self):
        """Decrement thrust by thrust_increment_"""
        with self.thrust_lock_:
            new_thrust = max(self.current_thrust_ - self.thrust_increment_, self.min_thrust_)
            if new_thrust != self.current_thrust_:
                self.current_thrust_ = new_thrust
                self.left_thrust_ = new_thrust
                self.right_thrust_ = new_thrust
                self.get_logger().info(f'Thrust decreased to {self.current_thrust_:.2f} N')
                # Immediately apply the new thrust
                self.apply_thrust()
    
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
        """Set left thruster command (in Newtons)"""
        with self.thrust_lock_:
            self.left_thrust_ = max(self.min_thrust_, min(self.max_thrust_, msg.data))
            self.get_logger().debug(f'Left thrust command: {self.left_thrust_:.2f} N')
    
    def right_thrust_callback(self, msg):
        """Set right thruster command (in Newtons)"""
        with self.thrust_lock_:
            self.right_thrust_ = max(self.min_thrust_, min(self.max_thrust_, msg.data))
            self.get_logger().debug(f'Right thrust command: {self.right_thrust_:.2f} N')
    
    def total_thrust_callback(self, msg):
        """Set both thrusters to the same value (in Newtons)"""
        with self.thrust_lock_:
            thrust = max(self.min_thrust_, min(self.max_thrust_, msg.data))
            self.current_thrust_ = thrust
            self.left_thrust_ = thrust
            self.right_thrust_ = thrust
            self.get_logger().debug(f'Total thrust command: {thrust:.2f} N')
    
    def yaw_differential_callback(self, msg):
        """Set yaw differential (-1.0 to 1.0)
        Negative = more left thrust (yaw right)
        Positive = more right thrust (yaw left)
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
        """Calculate left and right thrust values with optional differential control"""
        with self.thrust_lock_:
            if self.enable_differential_:
                # Apply differential thrust for yaw control
                # yaw_differential > 0 means yaw left, so increase right thrust
                # yaw_differential < 0 means yaw right, so increase left thrust
                differential_gain = 0.2  # Max 20% differential
                base_thrust = self.current_thrust_
                differential = self.yaw_differential_ * differential_gain * base_thrust
                
                left_thrust = self.left_thrust_ - differential
                right_thrust = self.right_thrust_ + differential
            else:
                # Equal thrust to both (simplified mode)
                left_thrust = self.left_thrust_
                right_thrust = self.right_thrust_
            
            # Clamp to valid range
            left_thrust = max(self.min_thrust_, min(self.max_thrust_, left_thrust))
            right_thrust = max(self.min_thrust_, min(self.max_thrust_, right_thrust))
            
            return left_thrust, right_thrust
    
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
        """Apply thrust using gz topic command to publish EntityWrench to ApplyLinkWrench plugin"""
        # Don't apply thrust until model is spawned
        if not self.thrust_started_:
            return
        
        left_thrust, right_thrust = self.calculate_thrust_values()
        total_thrust = left_thrust + right_thrust
        
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
    z: 0.0
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
