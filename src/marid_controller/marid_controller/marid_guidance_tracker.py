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

Now includes physics-based thrust calculation with wind compensation.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, FluidPressure
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
        
        # Check for invalid time or error (NaN/inf protection)
        if not np.isfinite(current_time) or not np.isfinite(error):
            return 0.0
        
        dt = current_time - self.last_time
        
        # Handle timing issues from lag:
        # - dt <= 0: invalid or clock jumped backward
        # - dt too small: numerical issues (less than 1ms)
        # - dt too large: system lagged (more than 1 second)
        if dt <= 0 or dt < 0.001:
            return 0.0  # Skip update if timing is invalid
        
        if dt > 1.0:
            # System lagged significantly - reset to prevent integral windup
            self.integral = 0.0
            self.last_time = current_time
            self.last_error = error
            return 0.0
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup protection)
        self.integral += error * dt
        # Clamp integral to prevent windup
        if self.ki > 0:
            max_integral = 100.0 / self.ki
        else:
            max_integral = 1000.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i_term = self.ki * self.integral
        
        # Derivative term (with dt protection)
        d_term = self.kd * (error - self.last_error) / dt
        
        # Total output
        output = p_term + i_term + d_term
        
        # Check for NaN before applying limits
        if not np.isfinite(output):
            self.integral = 0.0
            output = 0.0
        
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
    
    Now includes physics-based thrust calculation with wind compensation.
    """
    
    def __init__(self):
        super().__init__('marid_guidance_tracker')
        
        # Parameters
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('thrust_to_weight_ratio', 2.5)
        # Use -1.0 as sentinel for "not set" (None equivalent)
        # -1.0 means "auto-calculate" or "not set"
        self.declare_parameter('max_thrust', -1.0)
        self.declare_parameter('base_thrust_override', -1.0)
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
        self.declare_parameter('target_altitude', 5.0)  # Default to 5m (will be overridden by launch file)
        
        # Physics-based thrust parameters
        self.declare_parameter('drag_coefficient', 0.1)  # Cd * A * rho / 2 (simplified drag model)
        self.declare_parameter('use_physics_thrust', True)  # Enable physics-based calculation
        self.declare_parameter('use_airspeed_sensor', True)  # Use airspeed sensor if available
        
        # Wind vector (from world file: [0, 1, 0] m/s = 1 m/s in Y direction)
        self.declare_parameter('wind_x', 0.0)  # m/s
        self.declare_parameter('wind_y', 1.0)  # m/s
        self.declare_parameter('wind_z', 0.0)  # m/s
        
        # Air density (kg/m³) - standard sea level
        self.declare_parameter('air_density', 1.225)  # kg/m³ at sea level
        
        # Get parameters
        self.update_rate_ = self.get_parameter('update_rate').value
        self.thrust_to_weight_ratio_ = self.get_parameter('thrust_to_weight_ratio').value
        self.min_thrust_ = self.get_parameter('min_thrust').value
        self.max_yaw_differential_ = self.get_parameter('max_yaw_differential').value
        self.use_physics_thrust_ = self.get_parameter('use_physics_thrust').value
        self.use_airspeed_sensor_ = self.get_parameter('use_airspeed_sensor').value
        self.drag_coeff_ = self.get_parameter('drag_coefficient').value
        self.air_density_ = self.get_parameter('air_density').value
        
        # Wind vector
        self.wind_vector_ = np.array([
            self.get_parameter('wind_x').value,
            self.get_parameter('wind_y').value,
            self.get_parameter('wind_z').value
        ])
        
        # Calculate max_thrust
        max_thrust_param = self.get_parameter('max_thrust').value
        base_thrust_override = self.get_parameter('base_thrust_override').value
        
        # Store override value (>= 0 means use fixed thrust, -1.0 means not set)
        self.base_thrust_override_ = base_thrust_override  # Store directly: -1.0 = not set, >= 0 = use this value
        
        # Treat -1.0 as "not set" (None equivalent)
        if max_thrust_param < 0 or base_thrust_override >= 0:
            aircraft_mass = self.get_aircraft_mass()
            if aircraft_mass is not None:
                self.aircraft_mass_ = aircraft_mass
                if base_thrust_override >= 0:
                    self.max_thrust_ = float(base_thrust_override)  # Set max to override value
                    if self.base_thrust_override_ >= 0:
                        self.get_logger().info(f'Fixed thrust mode enabled: {self.base_thrust_override_:.2f} N (overrides PID calculations)')
                else:
                    g = 9.81
                    weight = aircraft_mass * g
                    self.max_thrust_ = weight * self.thrust_to_weight_ratio_
                self.get_logger().info(f'Aircraft mass: {aircraft_mass:.2f} kg, max_thrust: {self.max_thrust_:.2f} N')
            else:
                self.aircraft_mass_ = 10.0  # Default fallback
                self.max_thrust_ = 200.0 if max_thrust_param < 0 else float(max_thrust_param)
                self.get_logger().warn(f'Could not determine mass, using default max_thrust: {self.max_thrust_:.2f} N')
        else:
            self.max_thrust_ = float(max_thrust_param)
            # Estimate mass from max_thrust
            g = 9.81
            self.aircraft_mass_ = (self.max_thrust_ / self.thrust_to_weight_ratio_) / g
        
        # Current state
        self.current_odom_ = None
        self.current_imu_ = None
        self.current_altitude_ = None
        self.current_airspeed_ = None  # From pitot sensor
        self.airspeed_received_ = False
        
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
        
        # Subscribe to airspeed sensor (if available)
        if self.use_airspeed_sensor_:
            self.airspeed_sub_ = self.create_subscription(
                FluidPressure,
                '/airspeed',
                self.airspeed_callback,
                10
            )
            self.get_logger().info('Subscribed to /airspeed for wind compensation')
        else:
            self.get_logger().info('Airspeed sensor disabled - will estimate from ground velocity and wind')
        
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
        self.get_logger().info(f'  Aircraft mass: {self.aircraft_mass_:.2f} kg')
        self.get_logger().info(f'  Wind vector: [{self.wind_vector_[0]:.2f}, {self.wind_vector_[1]:.2f}, {self.wind_vector_[2]:.2f}] m/s')
        self.get_logger().info(f'  Drag coefficient: {self.drag_coeff_:.4f}')
        self.get_logger().info(f'  Physics-based thrust: {"enabled" if self.use_physics_thrust_ else "disabled"}')
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
    
    def airspeed_callback(self, msg):
        """
        Store current airspeed from pitot sensor.
        Airspeed sensor publishes pressure differential (Pa).
        Convert to velocity: v = sqrt(2 * delta_p / rho)
        """
        try:
            # Pressure differential in Pascals
            # Static pressure is typically ~101325 Pa at sea level
            # Dynamic pressure = 0.5 * rho * v²
            # So: delta_p = 0.5 * rho * v² → v = sqrt(2 * delta_p / rho)
            pressure_diff = msg.fluid_pressure  # Pa (dynamic pressure)
            
            if pressure_diff > 0:
                # Convert pressure to velocity
                airspeed = math.sqrt(2.0 * pressure_diff / self.air_density_)
                self.current_airspeed_ = airspeed
                self.airspeed_received_ = True
            else:
                # Negative or zero pressure - likely stationary or sensor error
                self.current_airspeed_ = 0.0
        except Exception as e:
            self.get_logger().debug(f'Error processing airspeed: {e}')
            self.current_airspeed_ = None
    
    def estimate_airspeed(self, ground_velocity):
        """
        Estimate airspeed from ground velocity and wind vector.
        airspeed_vector = ground_velocity - wind_vector
        airspeed = ||airspeed_vector||
        """
        airspeed_vector = ground_velocity - self.wind_vector_
        return np.linalg.norm(airspeed_vector)
    
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
        """Main control loop - tracks guidance targets and outputs actuator commands with physics-based thrust"""
        if self.current_odom_ is None:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Check for invalid time (handles timing issues from system lag)
        if not np.isfinite(current_time) or current_time <= 0:
            return  # Skip this iteration
        
        # Get ground velocity (for navigation and airspeed estimation)
        ground_velocity = np.array([
            self.current_odom_.twist.twist.linear.x,
            self.current_odom_.twist.twist.linear.y,
            self.current_odom_.twist.twist.linear.z
        ])
        ground_speed = np.linalg.norm(ground_velocity)
        
        # Get airspeed (for aerodynamics and drag calculation)
        if self.use_airspeed_sensor_ and self.airspeed_received_ and self.current_airspeed_ is not None:
            airspeed = self.current_airspeed_
        else:
            # Estimate airspeed from ground velocity and wind
            airspeed = self.estimate_airspeed(ground_velocity)
            if not self.airspeed_received_ and ground_speed > 0.1:
                self.get_logger().debug(f'Using estimated airspeed: {airspeed:.2f} m/s (ground: {ground_speed:.2f} m/s)')
        
        # Check for NaN airspeed
        if not np.isfinite(airspeed) or airspeed < 0:
            airspeed = max(0.0, ground_speed)  # Fallback to ground speed
        
        # Wait for guidance to be received before tracking
        # This prevents tracking zero guidance during initialization
        if not self.guidance_received_:
            # During initialization, maintain altitude only (safety)
            # Don't track speed/heading until guidance is active
            self.get_logger().debug('Waiting for guidance targets... Using altitude-only control.')
            
            # Use odometry altitude as fallback if barometer not available
            current_altitude = self.current_altitude_
            if current_altitude is None:
                current_altitude = self.current_odom_.pose.pose.position.z
            
            # Maintain altitude (prevent falling during startup)
            altitude_error = self.target_altitude_ - current_altitude
            altitude_thrust = self.altitude_pid_.update(altitude_error, current_time)
            
            # Add weight as base offset (works for both gravity and no-gravity cases)
            g = 9.81
            weight = self.aircraft_mass_ * g
            total_thrust = weight + altitude_thrust
            
            # Safety: Ensure minimum thrust during initialization to prevent zero thrust
            # This is especially important when gravity = 0 (weight = 0)
            if total_thrust < 1.0:
                total_thrust = max(total_thrust, 1.0)
                self.get_logger().debug(f'Applied minimum thrust during initialization: {total_thrust:.2f}N')
            
            total_thrust = np.clip(total_thrust, self.min_thrust_, self.max_thrust_)
            
            # Publish altitude-only thrust (no yaw command)
            thrust_msg = Float64()
            thrust_msg.data = float(total_thrust)
            self.total_thrust_pub_.publish(thrust_msg)
            
            yaw_msg = Float64()
            yaw_msg.data = 0.0  # No yaw command during initialization
            self.yaw_differential_pub_.publish(yaw_msg)
            return
        
        # ===== PHYSICS-BASED THRUST CALCULATION =====
        g = 9.81
        weight = self.aircraft_mass_ * g
        
        # Maintain altitude first (provides base thrust)
        current_altitude = self.current_altitude_
        if current_altitude is None:
            current_altitude = self.current_odom_.pose.pose.position.z
        
        # Check for NaN altitude
        if not np.isfinite(current_altitude):
            self.get_logger().debug('Invalid altitude (NaN), using target as fallback')
            current_altitude = self.target_altitude_  # Use target as fallback
        
        altitude_error = self.target_altitude_ - current_altitude
        altitude_thrust = self.altitude_pid_.update(altitude_error, current_time)
        
        # Check for NaN in PID output
        if not np.isfinite(altitude_thrust):
            self.get_logger().debug('Altitude PID returned NaN, resetting')
            self.altitude_pid_.reset()
            altitude_thrust = 0.0
        
        # Base thrust = weight (for gravity compensation) + altitude correction
        # If gravity = 0, weight = 0, so this works for both gravity and no-gravity cases
        # This prevents initial nose-dive by providing weight compensation from the start
        base_thrust = weight + altitude_thrust
        
        if self.use_physics_thrust_:
            # Drag force = drag_coefficient * airspeed²
            # drag_coefficient = 0.5 * rho * Cd * A (pre-computed parameter)
            current_drag = self.drag_coeff_ * airspeed**2
            
            # Desired airspeed (simplified - could account for wind)
            desired_airspeed = self.desired_speed_
            desired_drag = self.drag_coeff_ * desired_airspeed**2
            
            # Drag compensation: add thrust to overcome drag at desired speed
            # Only add the difference between desired and current drag
            drag_compensation = desired_drag - current_drag
            
            # PID correction for airspeed error (fine-tuning)
            airspeed_error = desired_airspeed - airspeed
            speed_correction = self.speed_pid_.update(airspeed_error, current_time)
            
            # Speed-related thrust = drag compensation + PID correction
            # Scale down PID correction to avoid over-correction
            speed_thrust = drag_compensation + 0.3 * speed_correction
            
        else:
            # Original PID-only approach (reactive, no physics)
            speed_error = self.desired_speed_ - ground_speed
            speed_thrust = self.speed_pid_.update(speed_error, current_time)
        
        # Track heading rate target (guidance provides desired_heading_rate)
        current_yaw_rate = 0.0
        if self.current_imu_ is not None:
            current_yaw_rate = self.current_imu_.angular_velocity.z
        
        heading_rate_error = self.desired_heading_rate_ - current_yaw_rate
        
        # Add deadband to prevent integral windup from IMU noise
        # Ignore small errors (< 0.05 rad/s ≈ 2.9 deg/s) to prevent constant yaw_differential
        HEADING_RATE_DEADBAND = 0.05  # rad/s
        if abs(heading_rate_error) < HEADING_RATE_DEADBAND:
            heading_rate_error = 0.0
        
        yaw_differential = self.heading_rate_pid_.update(heading_rate_error, current_time)
        yaw_differential = np.clip(yaw_differential, -self.max_yaw_differential_, self.max_yaw_differential_)
        
        # If base_thrust_override is set, use it as fixed thrust (ignore all PID calculations)
        if self.base_thrust_override_ >= 0:
            total_thrust = float(self.base_thrust_override_)
            self.get_logger().debug(f'Using fixed thrust override: {total_thrust:.2f} N')
        else:
            # Normal PID-based calculation
            total_thrust = base_thrust + speed_thrust
            
            # Check for NaN before clipping
            if not np.isfinite(total_thrust):
                self.get_logger().debug(f'Total thrust is NaN (base: {base_thrust}, speed: {speed_thrust}), using fallback')
                total_thrust = self.min_thrust_
        
        total_thrust = np.clip(total_thrust, self.min_thrust_, self.max_thrust_)
        
        # Publish actuator commands
        thrust_msg = Float64()
        thrust_msg.data = float(total_thrust)
        self.total_thrust_pub_.publish(thrust_msg)
        
        # Also check yaw_differential for NaN
        if not np.isfinite(yaw_differential):
            self.get_logger().debug('Yaw differential is NaN, resetting')
            self.heading_rate_pid_.reset()
            yaw_differential = 0.0
        
        yaw_msg = Float64()
        yaw_msg.data = float(yaw_differential)
        self.yaw_differential_pub_.publish(yaw_msg)
        
        # Debug logging (periodic)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 50 == 0:  # Every 1 second at 50 Hz
            self.get_logger().debug(
                f'Thrust: {total_thrust:.2f}N (speed: {speed_thrust:.2f}N, alt: {altitude_thrust:.2f}N) | '
                f'Airspeed: {airspeed:.2f} m/s (desired: {self.desired_speed_:.2f} m/s) | '
                f'Ground speed: {ground_speed:.2f} m/s | Yaw diff: {yaw_differential:.3f}'
            )
    
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
