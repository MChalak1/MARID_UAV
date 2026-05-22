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
        self.declare_parameter('altitude_pitch_gain', 0.04)
        self.declare_parameter('climb_wing_incidence', 0.0)
        self.declare_parameter('airborne_altitude_threshold', 0.5)
        self.declare_parameter('airborne_speed_threshold', 15.0)
        self.declare_parameter('pitch_slew_rate', 0.087)
        self.declare_parameter('roll_slew_rate', 0.0085)

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
        self.climb_wing_incidence_ = self.get_parameter('climb_wing_incidence').value
        self.airborne_altitude_threshold_ = self.get_parameter('airborne_altitude_threshold').value
        self.airborne_speed_threshold_ = self.get_parameter('airborne_speed_threshold').value
        self.pitch_slew_rate_ = self.get_parameter('pitch_slew_rate').value
        self.roll_slew_rate_ = self.get_parameter('roll_slew_rate').value

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

        self.current_roll_ = roll
        self.current_pitch_ = pitch
        self.current_yaw_ = yaw

        self.current_roll_rate_ = msg.twist.twist.angular.x
        self.current_pitch_rate_ = msg.twist.twist.angular.y
        self.current_yaw_rate_ = msg.twist.twist.angular.z

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
        self.flight_phase_ = msg.data

    def _compute_altitude_pitch(self):
        current_altitude = self.current_altitude_

        if current_altitude is None and self.current_odom_ is not None:
            current_altitude = self.current_odom_.pose.pose.position.z

        if current_altitude is None or not np.isfinite(current_altitude):
            return 0.0

        altitude_error = current_altitude - self.target_altitude_
        altitude_pitch = self.altitude_pitch_gain_ * altitude_error

        return float(np.clip(altitude_pitch, -0.12, 0.12))

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

        max_bank_angle = math.radians(30.0)
        desired_roll = float(np.clip(heading_error * 0.5, -max_bank_angle, max_bank_angle))

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
                self.cruise_enter_altitude_ = self.target_altitude_ * 1.15
                self.climb_reentry_altitude_ = self.target_altitude_ * 1.1
                # Stay at climb pitch until target altitude reached
                if self.current_altitude_ < self.cruise_enter_altitude_:
                    desired_pitch_target = -math.radians(30.0)  # -30 deg
                else:
                    self.flight_phase_ = "cruise"
                    desired_pitch_target = 0.0
                step_pitch = self.pitch_slew_rate_ * dt
                diff_pitch = desired_pitch_target - self.slewed_pitch_

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
            desired_pitch = desired_pitch_target
            self.slewed_pitch_ = desired_pitch
            self.slewed_roll_cmd_ = 0.0

        if (
            self.flight_phase_ == "cruise"
            and current_altitude is not None
            and current_altitude < self.climb_reentry_altitude_
        ):
            self.flight_phase_ = "climb"

        # =========================================================
        # 2. ATTITUDE ERRORS
        # =========================================================

        roll_error = self.wrap_angle(desired_roll - self.current_roll_)
        pitch_error = desired_pitch - self.current_pitch_
        yaw_error = self.wrap_angle(desired_yaw - self.current_yaw_)

        # =========================================================
        # 3. PID OUTPUTS
        # =========================================================

        roll_command_raw = self.roll_pid_.update(roll_error, current_time)
        pitch_command = self.pitch_pid_.update(pitch_error, current_time)
        yaw_command = -self.yaw_pid_.update(yaw_error, current_time)

        # =========================================================
        # 4. ACTUATOR AUTHORITY / SLEW / MIXING
        # =========================================================

        if self.flight_phase_ == 'climb':
            wing_pitch = self.climb_wing_incidence_
            active_yaw = 0.0

            if truly_on_ground:
                active_roll = 0.0
                roll_command = 0.0
                self.slewed_roll_cmd_ = 0.0
                self.roll_pid_.reset()

            else:
                step_roll = self.roll_slew_rate_ * dt
                diff_roll_cmd = roll_command_raw - self.slewed_roll_cmd_

                self.slewed_roll_cmd_ += math.copysign(
                    min(abs(diff_roll_cmd), step_roll),
                    diff_roll_cmd,
                )

                roll_command = self.slewed_roll_cmd_
                active_roll = roll_command

        else:
            wing_pitch = pitch_command
            roll_command = roll_command_raw
            active_roll = roll_command
            active_yaw = yaw_command

        left_wing_deflection = wing_pitch - active_roll
        right_wing_deflection = wing_pitch + active_roll

        tail_pitch_assist = -pitch_command
        left_tail_deflection = tail_pitch_assist - active_yaw
        right_tail_deflection = tail_pitch_assist + active_yaw

        wing_max = self.get_parameter('wing_max_deflection').value
        tail_max = self.get_parameter('tail_max_deflection').value

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
                f"pitch_cmd={math.degrees(pitch_command):.2f}\n"
                f"pitch_rate={math.degrees(self.current_pitch_rate_):.2f}\n"
                f"tail_L={math.degrees(left_tail_deflection):.2f}\n"
                f"tail_R={math.degrees(right_tail_deflection):.2f}\n"
                f"***********************************************\n"
                f"desired_roll={math.degrees(desired_roll):.2f}\n"
                f"roll_error={math.degrees(roll_error):.2f}\n"
                f"roll_cmd_raw={math.degrees(roll_command_raw):.2f}\n"
                f"roll_cmd_slewed={math.degrees(roll_command):.2f}\n"
                f"left_wing={math.degrees(left_wing_deflection):.2f}\n"
                f"right_wing={math.degrees(right_wing_deflection):.2f}\n"
                f"***********************************************\n"
                f"tail_pitch_assist={math.degrees(tail_pitch_assist):.2f}\n"
                f"active_yaw={math.degrees(active_yaw):.2f}\n"
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
