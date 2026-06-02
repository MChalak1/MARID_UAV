#!/usr/bin/env python3
"""
MARID Attitude Controller
Controls roll, pitch, and yaw by actuating control surfaces.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float64, String
import numpy as np
import math
from tf_transformations import euler_from_quaternion


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update(self, error, current_time):
        if self.last_time is None:
            self.last_time = current_time
            self.last_error = error
            return 0.0

        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0

        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.last_error) / dt

        output = p_term + i_term + d_term

        lo, hi = self.output_limits
        if lo is not None:
            output = max(lo, output)
        if hi is not None:
            output = min(hi, output)

        self.last_time = current_time
        self.last_error = error
        return output

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


class MaridAttitudeController(Node):
    def __init__(self):
        super().__init__('marid_attitude_controller')

        self.last_log_time_ = 0.0

        self.declare_parameter('update_rate', 50.0)

        self.declare_parameter('roll_kp', 1.0)
        self.declare_parameter('roll_ki', 0.0)
        self.declare_parameter('roll_kd', 0.3)

        self.declare_parameter('pitch_kp', 0.6)
        self.declare_parameter('pitch_ki', 0.0)
        self.declare_parameter('pitch_kd', 0.2)

        self.declare_parameter('yaw_kp', 1.0)
        self.declare_parameter('yaw_ki', 0.0)
        self.declare_parameter('yaw_kd', 0.3)

        self.declare_parameter('wing_max_deflection', 0.15)
        self.declare_parameter('tail_max_deflection', 0.15)

        self.declare_parameter('destination_latitude', -1.0)
        self.declare_parameter('destination_longitude', -1.0)
        self.declare_parameter('destination_x', -1.0)
        self.declare_parameter('destination_y', -1.0)
        self.declare_parameter('datum_latitude', 37.4)
        self.declare_parameter('datum_longitude', -122.1)
        self.declare_parameter('waypoint_tolerance', 2.0)

        self.declare_parameter('target_altitude', 5.0)
        self.declare_parameter('altitude_pitch_gain', 0.1)   # altitude error (m) → desired vz (m/s)
        self.declare_parameter('altitude_vz_max', 5.0)        # max target vertical speed (m/s)
        self.declare_parameter('altitude_vz_pitch_gain', 0.025)  # vz error (m/s) → pitch (rad)
        self.declare_parameter('altitude_pitch_max', 0.20)    # max pitch from altitude loop (rad)
        self.declare_parameter('climb_wing_incidence', 0.0)
        self.declare_parameter('airborne_altitude_threshold', 0.5)
        self.declare_parameter('airborne_speed_threshold', 15.0)
        self.declare_parameter('pitch_slew_rate', 0.087)
        self.declare_parameter('pitch_slew_slow_radius', math.radians(8.0))
        self.declare_parameter('pitch_slew_curve_exponent', 2.0)
        self.declare_parameter('pitch_slew_snap_threshold', math.radians(0.05))
        self.declare_parameter('roll_slew_rate', 0.0085)
        self.declare_parameter('cruise_pitch_slew_rate', math.radians(20.0))
        self.declare_parameter('cruise_roll_slew_rate', math.radians(4.0))
        self.declare_parameter('cruise_navigation_delay', 10.0)
        self.declare_parameter('cruise_pitch_trim', math.radians(-2.0))
        self.declare_parameter('climb_pitch_up_boost', 1.15)
        self.declare_parameter('cruise_pitch_up_boost', 1.15)
        self.declare_parameter('heading_to_bank_gain', 0.35)
        self.declare_parameter('max_bank_angle', math.radians(15.0))
        self.declare_parameter('roll_angle_to_rate_gain', 1.5)
        self.declare_parameter('max_roll_rate_command', math.radians(8.0))
        self.declare_parameter('min_roll_rate_command', math.radians(0.8))
        self.declare_parameter('roll_angle_deadband', math.radians(0.15))
        self.declare_parameter('roll_rate_kp', 0.04)
        self.declare_parameter('roll_gain_reference_speed', 20.0)
        self.declare_parameter('min_roll_speed_scale', 0.35)
        self.declare_parameter('fallback_rate_filter_alpha', 0.08)
        self.declare_parameter('max_fallback_attitude_rate', math.radians(25.0))
        self.declare_parameter('pitch_angle_to_rate_gain', 1.0)
        self.declare_parameter('max_pitch_rate_command', math.radians(6.0))
        self.declare_parameter('pitch_rate_kp', 0.05)
        self.declare_parameter('pitch_gain_reference_speed', 30.0)
        self.declare_parameter('min_pitch_speed_scale', 0.35)
        self.declare_parameter('pitch_deflection_slow_radius', math.radians(8.0))
        self.declare_parameter('pitch_deflection_curve_exponent', 2.0)
        self.declare_parameter('min_pitch_deflection_scale', 0.35)
        self.declare_parameter('yaw_angle_to_rate_gain', 1.0)
        self.declare_parameter('max_yaw_rate_command', math.radians(8.0))
        self.declare_parameter('yaw_rate_kp', 0.05)
        self.declare_parameter('yaw_rate_deadband', math.radians(0.5))

        self.update_rate_ = self.get_parameter('update_rate').value

        wing_max = self.get_parameter('wing_max_deflection').value
        tail_max = self.get_parameter('tail_max_deflection').value

        self.roll_pid_ = PIDController(
            self.get_parameter('roll_kp').value,
            self.get_parameter('roll_ki').value,
            self.get_parameter('roll_kd').value,
            (-wing_max, wing_max),
        )

        self.pitch_pid_ = PIDController(
            self.get_parameter('pitch_kp').value,
            self.get_parameter('pitch_ki').value,
            self.get_parameter('pitch_kd').value,
            (-tail_max, tail_max),
        )

        self.yaw_pid_ = PIDController(
            self.get_parameter('yaw_kp').value,
            self.get_parameter('yaw_ki').value,
            self.get_parameter('yaw_kd').value,
            (-tail_max, tail_max),
        )

        dest_lat = self.get_parameter('destination_latitude').value
        dest_lon = self.get_parameter('destination_longitude').value
        dest_x = self.get_parameter('destination_x').value
        dest_y = self.get_parameter('destination_y').value

        self.datum_lat_ = self.get_parameter('datum_latitude').value
        self.datum_lon_ = self.get_parameter('datum_longitude').value
        self.waypoint_tolerance_ = self.get_parameter('waypoint_tolerance').value

        self.target_altitude_ = self.get_parameter('target_altitude').value
        self.altitude_pitch_gain_ = self.get_parameter('altitude_pitch_gain').value
        self.altitude_vz_max_ = self.get_parameter('altitude_vz_max').value
        self.altitude_vz_pitch_gain_ = self.get_parameter('altitude_vz_pitch_gain').value
        self.altitude_pitch_max_ = self.get_parameter('altitude_pitch_max').value
        self.climb_wing_incidence_ = self.get_parameter('climb_wing_incidence').value
        self.airborne_altitude_threshold_ = self.get_parameter('airborne_altitude_threshold').value
        self.airborne_speed_threshold_ = self.get_parameter('airborne_speed_threshold').value
        self.pitch_slew_rate_ = self.get_parameter('pitch_slew_rate').value
        self.pitch_slew_slow_radius_ = self.get_parameter('pitch_slew_slow_radius').value
        self.pitch_slew_curve_exponent_ = self.get_parameter('pitch_slew_curve_exponent').value
        self.pitch_slew_snap_threshold_ = self.get_parameter('pitch_slew_snap_threshold').value
        self.roll_slew_rate_ = self.get_parameter('roll_slew_rate').value
        self.cruise_pitch_slew_rate_ = self.get_parameter('cruise_pitch_slew_rate').value
        self.cruise_roll_slew_rate_ = self.get_parameter('cruise_roll_slew_rate').value
        self.cruise_navigation_delay_ = self.get_parameter('cruise_navigation_delay').value
        self.cruise_pitch_trim_ = self.get_parameter('cruise_pitch_trim').value
        self.climb_pitch_up_boost_ = self.get_parameter('climb_pitch_up_boost').value
        self.cruise_pitch_up_boost_ = self.get_parameter('cruise_pitch_up_boost').value
        self.heading_to_bank_gain_ = self.get_parameter('heading_to_bank_gain').value
        self.max_bank_angle_ = self.get_parameter('max_bank_angle').value
        self.roll_angle_to_rate_gain_ = self.get_parameter('roll_angle_to_rate_gain').value
        self.max_roll_rate_command_ = self.get_parameter('max_roll_rate_command').value
        self.min_roll_rate_command_ = self.get_parameter('min_roll_rate_command').value
        self.roll_angle_deadband_ = self.get_parameter('roll_angle_deadband').value
        self.roll_rate_kp_ = self.get_parameter('roll_rate_kp').value
        self.roll_gain_reference_speed_ = self.get_parameter('roll_gain_reference_speed').value
        self.min_roll_speed_scale_ = self.get_parameter('min_roll_speed_scale').value
        self.fallback_rate_filter_alpha_ = self.get_parameter('fallback_rate_filter_alpha').value
        self.max_fallback_attitude_rate_ = self.get_parameter('max_fallback_attitude_rate').value
        self.pitch_angle_to_rate_gain_ = self.get_parameter('pitch_angle_to_rate_gain').value
        self.max_pitch_rate_command_ = self.get_parameter('max_pitch_rate_command').value
        self.pitch_rate_kp_ = self.get_parameter('pitch_rate_kp').value
        self.pitch_gain_reference_speed_ = self.get_parameter('pitch_gain_reference_speed').value
        self.min_pitch_speed_scale_ = self.get_parameter('min_pitch_speed_scale').value
        self.pitch_deflection_slow_radius_ = self.get_parameter('pitch_deflection_slow_radius').value
        self.pitch_deflection_curve_exponent_ = self.get_parameter('pitch_deflection_curve_exponent').value
        self.min_pitch_deflection_scale_ = self.get_parameter('min_pitch_deflection_scale').value
        self.yaw_angle_to_rate_gain_ = self.get_parameter('yaw_angle_to_rate_gain').value
        self.max_yaw_rate_command_ = self.get_parameter('max_yaw_rate_command').value
        self.yaw_rate_kp_ = self.get_parameter('yaw_rate_kp').value
        self.yaw_rate_deadband_ = self.get_parameter('yaw_rate_deadband').value

        self.current_speed_ = 0.0
        self.slewed_pitch_ = 0.0
        self.slewed_roll_cmd_ = 0.0

        if dest_lat != -1.0 and dest_lon != -1.0:
            self.destination_ = self.lat_lon_to_local(dest_lat, dest_lon, self.datum_lat_, self.datum_lon_)
            self.get_logger().info(f'Destination GPS: ({dest_lat:.6f}, {dest_lon:.6f})')
        elif dest_x != -1.0 and dest_y != -1.0:
            self.destination_ = np.array([dest_x, dest_y])
            self.get_logger().info(f'Destination local: ({dest_x:.2f}, {dest_y:.2f}) m')
        else:
            self.destination_ = None
            self.get_logger().warn('No destination set. Maintaining level flight.')

        self.flight_phase_ = 'climb'

        self.current_odom_ = None
        self.current_altitude_ = None

        self.current_roll_ = 0.0
        self.current_pitch_ = 0.0
        self.current_yaw_ = 0.0

        self.current_roll_rate_ = 0.0
        self.current_pitch_rate_ = 0.0
        self.current_yaw_rate_ = 0.0
        self.filtered_roll_rate_ = 0.0
        self.filtered_pitch_rate_ = 0.0
        self.filtered_yaw_rate_ = 0.0
        self.last_attitude_time_ = None
        self.last_roll_for_rate_ = None
        self.last_pitch_for_rate_ = None
        self.last_yaw_for_rate_ = None
        self.cruise_enter_altitude_ = float('nan')
        self.climb_reentry_altitude_ = float('nan')
        self.cruise_start_time_ = None

        self.odom_sub_ = self.create_subscription(
            Odometry,
            '/gazebo/odom',
            self.odom_callback,
            10,
        )

        self.waypoint_sub_ = self.create_subscription(
            PoseStamped,
            '/marid/waypoint',
            self.waypoint_callback,
            10,
        )

        self.altitude_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            '/barometer/altitude',
            self.altitude_callback,
            10,
        )

        self.guidance_mode_sub_ = self.create_subscription(
            String,
            '/marid/guidance/mode',
            self.guidance_mode_callback,
            10,
        )

        self.left_wing_pub_ = self.create_publisher(
            Float64,
            '/model/marid/joint/left_wing_joint/cmd_pos',
            10,
        )
        self.right_wing_pub_ = self.create_publisher(
            Float64,
            '/model/marid/joint/right_wing_joint/cmd_pos',
            10,
        )
        self.tail_left_pub_ = self.create_publisher(
            Float64,
            '/model/marid/joint/tail_left_joint/cmd_pos',
            10,
        )
        self.tail_right_pub_ = self.create_publisher(
            Float64,
            '/model/marid/joint/tail_right_joint/cmd_pos',
            10,
        )

        self.control_timer_ = self.create_timer(
            1.0 / self.update_rate_,
            self.control_loop,
        )

        self.get_logger().info('MARID Attitude Controller initialized')
        self.get_logger().info(f'Update rate: {self.update_rate_} Hz')
        self.get_logger().info(f'Altitude target={self.target_altitude_} m, gain={self.altitude_pitch_gain_}')

    def lat_lon_to_local(self, lat, lon, datum_lat, datum_lon):
        R = 6371000.0
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        datum_lat_rad = math.radians(datum_lat)
        datum_lon_rad = math.radians(datum_lon)

        dlat = lat_rad - datum_lat_rad
        dlon = lon_rad - datum_lon_rad

        x = R * dlon * math.cos(datum_lat_rad)
        y = R * dlat

        return np.array([x, y])

    def wrap_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def odom_callback(self, msg):
        self.current_odom_ = msg

        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_attitude_time_ is not None:
            dt = now - self.last_attitude_time_
            if dt > 1e-4:
                alpha = self.fallback_rate_filter_alpha_
                measured_roll_rate = self.wrap_angle(roll - self.last_roll_for_rate_) / dt
                measured_pitch_rate = (pitch - self.last_pitch_for_rate_) / dt
                measured_yaw_rate = self.wrap_angle(yaw - self.last_yaw_for_rate_) / dt
                measured_roll_rate = float(np.clip(
                    measured_roll_rate,
                    -self.max_fallback_attitude_rate_,
                    self.max_fallback_attitude_rate_,
                ))
                measured_pitch_rate = float(np.clip(
                    measured_pitch_rate,
                    -self.max_fallback_attitude_rate_,
                    self.max_fallback_attitude_rate_,
                ))
                measured_yaw_rate = float(np.clip(
                    measured_yaw_rate,
                    -self.max_fallback_attitude_rate_,
                    self.max_fallback_attitude_rate_,
                ))
                self.filtered_roll_rate_ += alpha * (measured_roll_rate - self.filtered_roll_rate_)
                self.filtered_pitch_rate_ += alpha * (measured_pitch_rate - self.filtered_pitch_rate_)
                self.filtered_yaw_rate_ += alpha * (measured_yaw_rate - self.filtered_yaw_rate_)

        self.last_attitude_time_ = now
        self.last_roll_for_rate_ = roll
        self.last_pitch_for_rate_ = pitch
        self.last_yaw_for_rate_ = yaw

        self.current_roll_ = roll
        self.current_pitch_ = pitch
        self.current_yaw_ = yaw

        raw_roll_rate = msg.twist.twist.angular.x
        raw_pitch_rate = msg.twist.twist.angular.y
        raw_yaw_rate = msg.twist.twist.angular.z
        self.current_roll_rate_ = raw_roll_rate if abs(raw_roll_rate) > 1e-5 else self.filtered_roll_rate_
        self.current_pitch_rate_ = raw_pitch_rate if abs(raw_pitch_rate) > 1e-5 else self.filtered_pitch_rate_
        self.current_yaw_rate_ = raw_yaw_rate if abs(raw_yaw_rate) > 1e-5 else self.filtered_yaw_rate_

        vx = msg.twist.twist.linear.x
        self.current_speed_ = abs(vx) if math.isfinite(vx) else 0.0

    def waypoint_callback(self, msg):
        new_xy = np.array([msg.pose.position.x, msg.pose.position.y])

        if self.destination_ is not None and np.allclose(new_xy, self.destination_, atol=1.0):
            return

        self.destination_ = new_xy
        self.get_logger().info(f'New waypoint: ({new_xy[0]:.2f}, {new_xy[1]:.2f}) m')

    def altitude_callback(self, msg):
        self.current_altitude_ = msg.pose.pose.position.z

    def guidance_mode_callback(self, msg):
        # Only allow explicit climb/cruise phase overrides from external topic.
        # 'pid'/'ai' guidance mode strings must not overwrite the internal flight phase —
        # those are guidance algorithm selectors, not aircraft flight phases.
        if msg.data in ('climb', 'cruise'):
            self.flight_phase_ = msg.data

    def _compute_altitude_pitch(self):
        current_altitude = self.current_altitude_

        if current_altitude is None and self.current_odom_ is not None:
            current_altitude = self.current_odom_.pose.pose.position.z

        if current_altitude is None or not np.isfinite(current_altitude):
            return 0.0

        altitude_error = current_altitude - self.target_altitude_

        # Altitude error → desired vertical speed (clipped to ±altitude_vz_max)
        desired_vz = float(np.clip(
            self.altitude_pitch_gain_ * altitude_error,
            -self.altitude_vz_max_,
            self.altitude_vz_max_,
        ))

        # Current vertical speed from odometry
        current_vz = 0.0
        if self.current_odom_ is not None:
            current_vz = float(self.current_odom_.twist.twist.linear.z)

        # vz error → pitch angle. Settling: when vz=0 at target altitude, pitch=0.
        altitude_pitch = float(np.clip(
            self.altitude_vz_pitch_gain_ * (desired_vz - current_vz),
            -self.altitude_pitch_max_,
            self.altitude_pitch_max_,
        ))
        return altitude_pitch

    def compute_waypoint_commands(self):
        altitude_pitch = self._compute_altitude_pitch()

        if self.destination_ is None or self.current_odom_ is None:
            return 0.0, altitude_pitch, self.current_yaw_

        current_pos = np.array([
            self.current_odom_.pose.pose.position.x,
            self.current_odom_.pose.pose.position.y,
        ])

        direction = self.destination_ - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6 or distance < self.waypoint_tolerance_:
            return 0.0, altitude_pitch, self.current_yaw_

        desired_yaw = math.atan2(direction[1], direction[0])

        if not np.isfinite(desired_yaw):
            desired_yaw = self.current_yaw_

        heading_error = self.wrap_angle(desired_yaw - self.current_yaw_)

        # Positive roll is the bank direction that reduces negative heading error
        # for this airframe/sim convention.
        desired_roll = float(np.clip(
            -heading_error * self.heading_to_bank_gain_,
            -self.max_bank_angle_,
            self.max_bank_angle_,
        ))

        pitch_compensation_factor = 0.1
        turn_pitch = abs(desired_roll) * pitch_compensation_factor

        desired_pitch = turn_pitch + altitude_pitch

        return desired_roll, desired_pitch, desired_yaw

    def control_loop(self):
        if self.current_odom_ is None:
            return

        desired_roll, desired_pitch_target, desired_yaw = self.compute_waypoint_commands()

        if not (
            np.isfinite(desired_roll)
            and np.isfinite(desired_pitch_target)
            and np.isfinite(desired_yaw)
        ):
            self.get_logger().warn('Invalid desired attitude, skipping update.')
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = 1.0 / self.update_rate_

        current_altitude = self.current_altitude_
        if current_altitude is None and self.current_odom_ is not None:
            current_altitude = self.current_odom_.pose.pose.position.z

        truly_on_ground = (
            current_altitude is None
            or not math.isfinite(current_altitude)
            or current_altitude < 0.3
        )

        airborne = (
            current_altitude is not None
            and math.isfinite(current_altitude)
            and current_altitude >= self.airborne_altitude_threshold_
            and self.current_speed_ >= self.airborne_speed_threshold_
        )

        # =========================================================
        # 1. PHASE-DEPENDENT DESIRED ATTITUDE
        # =========================================================

        if self.flight_phase_ == 'climb':
            desired_roll = 0.0
            desired_yaw = self.current_yaw_
            

            if truly_on_ground :
                # Intentional nose-up scheduling during ground roll.
                desired_pitch = -math.radians(30.0)
                self.slewed_pitch_ = desired_pitch
                self.slewed_roll_cmd_ = 0.0
                self.roll_pid_.reset()

            elif airborne:
                self.cruise_enter_altitude_ = self.target_altitude_ * 1.00
                self.climb_reentry_altitude_ = self.target_altitude_ * 0.95
                entering_cruise = False
                # Stay at climb pitch until target altitude reached
                if self.current_altitude_ < self.cruise_enter_altitude_:
                    desired_pitch_target = -math.radians(30.0)  # -30 deg
                else:
                    self.flight_phase_ = "cruise"
                    self.cruise_start_time_ = current_time
                    entering_cruise = True
                    desired_pitch_target = self.cruise_pitch_trim_
                diff_pitch = desired_pitch_target - self.slewed_pitch_
                if abs(diff_pitch) <= self.pitch_slew_snap_threshold_:
                    self.slewed_pitch_ = desired_pitch_target
                elif entering_cruise:
                    step_pitch = self.cruise_pitch_slew_rate_ * dt
                    self.slewed_pitch_ += math.copysign(
                        min(abs(diff_pitch), step_pitch),
                        diff_pitch,
                    )
                else:
                    slew_fraction = min(
                        abs(diff_pitch) / max(self.pitch_slew_slow_radius_, 1e-6),
                        1.0,
                    )
                    shaped_slew_rate = self.pitch_slew_rate_ * (
                        slew_fraction ** self.pitch_slew_curve_exponent_
                    )
                    step_pitch = shaped_slew_rate * dt

                    self.slewed_pitch_ += math.copysign(
                        min(abs(diff_pitch), step_pitch),
                        diff_pitch,
                    )

                desired_pitch = self.slewed_pitch_

            else:
                desired_pitch = self.slewed_pitch_
                self.slewed_roll_cmd_ = 0.0
                self.roll_pid_.reset()

        else:
            desired_pitch_target += self.cruise_pitch_trim_
            diff_pitch = desired_pitch_target - self.slewed_pitch_
            if abs(diff_pitch) <= self.pitch_slew_snap_threshold_:
                self.slewed_pitch_ = desired_pitch_target
            else:
                step_pitch = self.cruise_pitch_slew_rate_ * dt
                self.slewed_pitch_ += math.copysign(
                    min(abs(diff_pitch), step_pitch),
                    diff_pitch,
                )
            desired_pitch = self.slewed_pitch_

        if (
            self.flight_phase_ == "cruise"
            and current_altitude is not None
            and current_altitude < self.climb_reentry_altitude_
        ):
            self.flight_phase_ = "climb"
            self.cruise_start_time_ = None

        if self.flight_phase_ == "cruise":
            if self.cruise_start_time_ is None:
                self.cruise_start_time_ = current_time
            if current_time - self.cruise_start_time_ < self.cruise_navigation_delay_:
                desired_roll = 0.0
                desired_yaw = self.current_yaw_

        # =========================================================
        # 2. ATTITUDE ERRORS
        # =========================================================

        roll_error = self.wrap_angle(desired_roll - self.current_roll_)
        pitch_error = desired_pitch - self.current_pitch_
        yaw_error = self.wrap_angle(desired_yaw - self.current_yaw_)

        # =========================================================
        # 3. PID OUTPUTS
        # =========================================================

        speed_for_gain = max(abs(self.current_speed_), 1.0)
        roll_speed_scale = (self.roll_gain_reference_speed_ / speed_for_gain) ** 2
        roll_speed_scale = float(np.clip(
            roll_speed_scale,
            self.min_roll_speed_scale_,
            1.0,
        ))

        desired_roll_rate = self.roll_angle_to_rate_gain_ * roll_error
        if abs(roll_error) < self.roll_angle_deadband_:
            desired_roll_rate = 0.0
        elif abs(desired_roll_rate) < self.min_roll_rate_command_:
            desired_roll_rate = math.copysign(self.min_roll_rate_command_, roll_error)
        desired_roll_rate *= roll_speed_scale
        desired_roll_rate = float(np.clip(
            desired_roll_rate,
            -self.max_roll_rate_command_,
            self.max_roll_rate_command_,
        ))
        roll_rate_error = desired_roll_rate - self.current_roll_rate_
        roll_command_raw = self.roll_rate_kp_ * roll_speed_scale * roll_rate_error
        roll_command_raw = float(np.clip(
            roll_command_raw,
            -self.get_parameter('wing_max_deflection').value,
            self.get_parameter('wing_max_deflection').value,
        ))
        pitch_speed_scale = (self.pitch_gain_reference_speed_ / speed_for_gain) ** 2
        pitch_speed_scale = float(np.clip(
            pitch_speed_scale,
            self.min_pitch_speed_scale_,
            1.0,
        ))
        tail_max = self.get_parameter('tail_max_deflection').value

        desired_pitch_rate = self.pitch_angle_to_rate_gain_ * pitch_error
        desired_pitch_rate = float(np.clip(
            desired_pitch_rate,
            -self.max_pitch_rate_command_,
            self.max_pitch_rate_command_,
        ))
        pitch_rate_error = desired_pitch_rate - self.current_pitch_rate_
        pitch_deflection_fraction = min(
            abs(pitch_error) / max(self.pitch_deflection_slow_radius_, 1e-6),
            1.0,
        )
        pitch_deflection_scale = self.min_pitch_deflection_scale_ + (
            1.0 - self.min_pitch_deflection_scale_
        ) * (pitch_deflection_fraction ** self.pitch_deflection_curve_exponent_)
        pitch_command = self.pitch_rate_kp_ * pitch_speed_scale * pitch_rate_error * pitch_deflection_scale
        if self.flight_phase_ == 'climb' and pitch_command < 0.0:
            pitch_command *= self.climb_pitch_up_boost_
        elif self.flight_phase_ == 'cruise' and pitch_command < 0.0:
            pitch_command *= self.cruise_pitch_up_boost_
        pitch_command = float(np.clip(
            pitch_command,
            -tail_max,
            tail_max,
        ))
        desired_yaw_rate = self.yaw_angle_to_rate_gain_ * yaw_error
        desired_yaw_rate = float(np.clip(
            desired_yaw_rate,
            -self.max_yaw_rate_command_,
            self.max_yaw_rate_command_,
        ))
        yaw_rate_error = desired_yaw_rate - self.current_yaw_rate_
        if abs(yaw_rate_error) < self.yaw_rate_deadband_:
            yaw_rate_error = 0.0
        yaw_command = -self.yaw_rate_kp_ * yaw_rate_error
        yaw_command = float(np.clip(
            yaw_command,
            -tail_max,
            tail_max,
        ))

        # =========================================================
        # 4. ACTUATOR AUTHORITY / SLEW / MIXING
        # =========================================================

        if self.flight_phase_ == 'climb':
            wing_pitch = self.climb_wing_incidence_
            active_yaw = 0.0

            if truly_on_ground:
                active_roll = 0.0
                roll_command = 0.0
                active_yaw = 0.0
                self.slewed_roll_cmd_ = 0.0
                self.roll_pid_.reset()

            else:
                active_yaw = yaw_command
                step_roll = self.roll_slew_rate_ * dt
                diff_roll_cmd = roll_command_raw - self.slewed_roll_cmd_

                self.slewed_roll_cmd_ += math.copysign(
                    min(abs(diff_roll_cmd), step_roll),
                    diff_roll_cmd,
                )

                roll_command = self.slewed_roll_cmd_
                active_roll = roll_command

        else:
            wing_pitch = 0.0
            step_roll = self.cruise_roll_slew_rate_ * dt
            diff_roll_cmd = roll_command_raw - self.slewed_roll_cmd_
            self.slewed_roll_cmd_ += math.copysign(
                min(abs(diff_roll_cmd), step_roll),
                diff_roll_cmd,
            )
            roll_command = self.slewed_roll_cmd_
            active_roll = roll_command
            active_yaw = yaw_command

        left_wing_deflection = wing_pitch - active_roll
        right_wing_deflection = wing_pitch + active_roll

        tail_pitch_assist = -pitch_command
        left_tail_deflection = tail_pitch_assist - active_yaw
        right_tail_deflection = tail_pitch_assist + active_yaw

        wing_max = self.get_parameter('wing_max_deflection').value
        left_wing_deflection = float(np.clip(left_wing_deflection, -wing_max, wing_max))
        right_wing_deflection = float(np.clip(right_wing_deflection, -wing_max, wing_max))
        left_tail_deflection = float(np.clip(left_tail_deflection, -tail_max, tail_max))
        right_tail_deflection = float(np.clip(right_tail_deflection, -tail_max, tail_max))

        # =========================================================
        # 5. LOGGING
        # =========================================================

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_log_time_ >= 1.0:
            self.get_logger().info(
                f"\n"
                f"desired_pitch={math.degrees(desired_pitch):.2f}\n"
                f"pitch_error={math.degrees(pitch_error):.2f}\n"
                f"desired_pitch_rate={math.degrees(desired_pitch_rate):.2f}\n"
                f"pitch_rate_error={math.degrees(pitch_rate_error):.2f}\n"
                f"pitch_speed_scale={pitch_speed_scale:.2f}\n"
                f"pitch_deflection_scale={pitch_deflection_scale:.2f}\n"
                f"pitch_cmd={math.degrees(pitch_command):.2f}\n"
                f"pitch_rate={math.degrees(self.current_pitch_rate_):.4f}\n"
                f"tail_L={math.degrees(left_tail_deflection):.2f}\n"
                f"tail_R={math.degrees(right_tail_deflection):.2f}\n"
                f"***********************************************\n"
                f"desired_roll={math.degrees(desired_roll):.2f}\n"
                f"current_roll={math.degrees(self.current_roll_):.2f}\n"
                f"roll_error={math.degrees(roll_error):.2f}\n"
                f"desired_roll_rate={math.degrees(desired_roll_rate):.2f}\n"
                f"roll_rate_error={math.degrees(roll_rate_error):.2f}\n"
                f"roll_speed_scale={roll_speed_scale:.2f}\n"
                f"roll_rate={math.degrees(self.current_roll_rate_):.4f}\n"
                f"roll_cmd_raw={math.degrees(roll_command_raw):.2f}\n"
                f"roll_cmd_slewed={math.degrees(roll_command):.2f}\n"
                f"left_wing={math.degrees(left_wing_deflection):.2f}\n"
                f"right_wing={math.degrees(right_wing_deflection):.2f}\n"
                f"***********************************************\n"
                f"desired_yaw={math.degrees(desired_yaw):.2f}\n"
                f"current_yaw={math.degrees(self.current_yaw_):.2f}\n"
                f"yaw_error={math.degrees(yaw_error):.2f}\n"
                f"tail_pitch_assist={math.degrees(tail_pitch_assist):.2f}\n"
                f"active_yaw={math.degrees(active_yaw):.2f}\n"
                f"desired_yaw_rate={math.degrees(desired_yaw_rate):.2f}\n"
                f"yaw_rate_error={math.degrees(yaw_rate_error):.2f}\n"
                f"yaw_rate={math.degrees(self.current_yaw_rate_):.4f}\n"
                f"speed={self.current_speed_:.2f}\n"
                f"altitude={current_altitude if current_altitude is not None else float('nan'):.2f}\n"
                f"target_altitude={self.target_altitude_:.2f}\n"
                f"cruise_enter_altitude={self.cruise_enter_altitude_ if hasattr(self, 'cruise_enter_altitude_') else float('nan'):.2f}\n"
                f"climb_reentry_altitude={self.climb_reentry_altitude_ if hasattr(self, 'climb_reentry_altitude_') else float('nan'):.2f}\n"
                f"truly_on_ground={truly_on_ground}\n"
                f"airborne={airborne}\n"
                f"phase={self.flight_phase_}\n"
                "===============================================================================\n\n"
            )
            self.last_log_time_ = now

        # =========================================================
        # 6. PUBLISH COMMANDS
        # =========================================================

        left_wing_msg = Float64()
        left_wing_msg.data = left_wing_deflection
        self.left_wing_pub_.publish(left_wing_msg)

        right_wing_msg = Float64()
        right_wing_msg.data = right_wing_deflection
        self.right_wing_pub_.publish(right_wing_msg)

        tail_left_msg = Float64()
        tail_left_msg.data = left_tail_deflection
        self.tail_left_pub_.publish(tail_left_msg)

        tail_right_msg = Float64()
        tail_right_msg.data = right_tail_deflection
        self.tail_right_pub_.publish(tail_right_msg)


def main():
    rclpy.init()
    node = MaridAttitudeController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
