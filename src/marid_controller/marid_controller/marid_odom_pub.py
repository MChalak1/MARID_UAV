#!/usr/bin/env python3
import math
import numpy as np

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, TwistStamped
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

from std_srvs.srv import Empty
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from tf_transformations import quaternion_matrix


class MaridOdomPublisher(Node):
    def __init__(self):
        super().__init__("marid_odom_node")

        # -----------------------
        # Parameters
        # -----------------------
        self.declare_parameter("imu_outputs_specific_force", True)
        self.declare_parameter("gravity", 9.81)
        self.declare_parameter("publish_rate", 50.0)

        self.declare_parameter("max_velocity", 200.0)          # m/s safety limit
        self.declare_parameter("max_acceleration", 350.0)      # m/s^2 safety limit

        # Damping interpreted as "fraction kept per second" (0.95 => keep 95% after 1 second)
        self.declare_parameter("velocity_decay_rate", 0.95)

        self.declare_parameter("base_position_variance", 0.01)
        self.declare_parameter("base_velocity_variance", 0.1)
        self.declare_parameter("variance_growth_rate", 0.001)  # per second

        self.declare_parameter("child_frame_id", "base_link_front_imu")
        self.declare_parameter("publish_tf", True)

        # Bias calibration
        self.declare_parameter("calibration_required", 100)    # samples
        self.declare_parameter("calibration_max_dt", 0.05)     # ignore large dt during calibration

        # Barometer fusion
        self.declare_parameter("use_barometer", True)
        self.declare_parameter("baro_weight", 0.98)            # fraction of baro in fused z (0..1)

        # FAST-LIO fusion
        self.declare_parameter("use_fastlio", True)
        self.declare_parameter("fastlio_topic", "/Odometry")
        # How much FAST-LIO nudges position each update (0=ignore, 1=hard reset).
        # Low value lets airspeed dead reckoning own position; FAST-LIO corrects slow drift.
        # Increase toward 1.0 in feature-rich environments where FAST-LIO is reliable.
        self.declare_parameter("fastlio_position_weight", 0.3)

        # Airspeed complementary filter
        self.declare_parameter("use_airspeed", True)
        self.declare_parameter("airspeed_weight", 0.8)
        self.declare_parameter("min_airspeed_for_fusion", 0.5)   # m/s — only skip when truly static
        self.declare_parameter("wind_topic", "/wind/velocity")
        # Disable wind correction by default: in simulation there is no actual wind, and the
        # wind estimator uses EKF/Gazebo ground truth which leaks into GPS-denied estimation.
        # Enable only for real flight where wind is genuine and estimated independently.
        self.declare_parameter("use_wind_correction", False)

        # -----------------------
        # Get parameters
        # -----------------------
        self.accel_is_sf_ = bool(self.get_parameter("imu_outputs_specific_force").value)
        self.g_ = float(self.get_parameter("gravity").value)
        self.publish_rate_ = float(self.get_parameter("publish_rate").value)

        self.max_velocity_ = float(self.get_parameter("max_velocity").value)
        self.max_acceleration_ = float(self.get_parameter("max_acceleration").value)

        self.velocity_decay_per_sec_ = float(self.get_parameter("velocity_decay_rate").value)

        self.base_pos_var_ = float(self.get_parameter("base_position_variance").value)
        self.base_vel_var_ = float(self.get_parameter("base_velocity_variance").value)
        self.var_growth_rate_ = float(self.get_parameter("variance_growth_rate").value)

        self.child_frame_id_ = str(self.get_parameter("child_frame_id").value)
        self.publish_tf_ = bool(self.get_parameter("publish_tf").value)

        self.calibration_required_ = int(self.get_parameter("calibration_required").value)
        self.calibration_max_dt_ = float(self.get_parameter("calibration_max_dt").value)

        self.use_barometer_ = bool(self.get_parameter("use_barometer").value)
        self.baro_weight_ = float(self.get_parameter("baro_weight").value)
        self.baro_weight_ = max(0.0, min(1.0, self.baro_weight_))
        self.imu_weight_ = 1.0 - self.baro_weight_

        self.use_fastlio_ = bool(self.get_parameter("use_fastlio").value)
        self.fastlio_topic_ = str(self.get_parameter("fastlio_topic").value)
        self.fastlio_pos_weight_ = max(0.0, min(1.0, float(
            self.get_parameter("fastlio_position_weight").value)))

        self.use_airspeed_ = bool(self.get_parameter("use_airspeed").value)
        self.airspeed_weight_ = max(0.0, min(1.0, float(self.get_parameter("airspeed_weight").value)))
        self.min_airspeed_ = float(self.get_parameter("min_airspeed_for_fusion").value)
        self.wind_topic_ = str(self.get_parameter("wind_topic").value)
        self.use_wind_correction_ = bool(self.get_parameter("use_wind_correction").value)

        # -----------------------
        # TF broadcaster + listener
        # -----------------------
        self.tf_broadcaster_ = TransformBroadcaster(self)
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        # -----------------------
        # State variables
        # -----------------------
        self.x_ = 0.0
        self.y_ = 0.0

        # Keep IMU-only vertical state separately
        self.z_imu_ = 0.0
        self.vz_imu_ = 0.0

        # Horizontal velocities
        self.vx_ = 0.0
        self.vy_ = 0.0

        # Published fused z (baro + IMU)
        self.z_fused_ = 0.0

        # Angular velocity (just passing through one component you used)
        self.wz_ = 0.0

        # Orientation
        self.qx_, self.qy_, self.qz_, self.qw_ = 0.0, 0.0, 0.0, 1.0

        # Bias estimation: average of (measured - expected_stationary_measurement)
        self.accel_bias_ = np.zeros(3, dtype=float)
        self.bias_samples_ = []
        self.calibrating_ = True
        self.calibration_samples_ = 0

        # Statistics
        self.integration_count_ = 0
        self.dropped_samples_ = 0
        self.warning_count_ = 0

        # Barometer fusion state
        self.baro_z_ = 0.0
        self.baro_ready_ = False

        # FAST-LIO fusion state
        self.fastlio_ready_ = False

        # Airspeed fusion state
        self.last_airspeed_ = 0.0
        self.airspeed_ready_ = False
        self.wind_x_ = 0.0
        self.wind_y_ = 0.0
        self.wind_z_ = 0.0

        # Time tracking
        self.start_time_ = self.get_clock().now()
        self.prev_imu_time_ = None

        # -----------------------
        # Subscriptions & publishers
        # -----------------------
        self.pose_sub_ = self.create_subscription(
            Imu, "/imu_ekf", self.imu_callback, 10
        )

        if self.use_barometer_:
            self.baro_sub_ = self.create_subscription(
                PoseWithCovarianceStamped,
                "/barometer/altitude",
                self.baro_callback,
                10,
            )

        if self.use_fastlio_:
            self.fastlio_sub_ = self.create_subscription(
                Odometry, self.fastlio_topic_, self.fastlio_callback, 10
            )

        if self.use_airspeed_:
            self.airspeed_sub_ = self.create_subscription(
                Float64, "/airspeed/velocity", self.airspeed_callback, 10
            )
            self.wind_sub_ = self.create_subscription(
                TwistStamped, self.wind_topic_, self.wind_callback, 10
            )

        self.odom_pub_ = self.create_publisher(Odometry, "/marid/odom", 10)
        self.diag_pub_ = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        # Timer
        timer_period = 1.0 / max(1e-3, self.publish_rate_)
        self.timer_ = self.create_timer(timer_period, self.publish_odom)

        # Services
        self.reset_srv_ = self.create_service(
            Empty, "~/reset_odometry", self.reset_odometry_callback
        )

        # Odometry message template
        self.odom_msg_ = Odometry()
        self.odom_msg_.header.frame_id = "odom"
        self.odom_msg_.child_frame_id = self.child_frame_id_

        self.get_logger().info("IMU Odometry Node initialized (fixed math).")
        self.get_logger().info(
            f"IMU specific_force={self.accel_is_sf_}, gravity={self.g_}, "
            f"calibration_required={self.calibration_required_} samples"
        )

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _is_finite(x: float) -> bool:
        return not (math.isnan(x) or math.isinf(x))

    def normalize_quaternion(self, qx, qy, qz, qw):
        """Return normalized quaternion; identity if invalid."""
        if not (self._is_finite(qx) and self._is_finite(qy) and self._is_finite(qz) and self._is_finite(qw)):
            return 0.0, 0.0, 0.0, 1.0
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if not self._is_finite(norm) or norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return qx/norm, qy/norm, qz/norm, qw/norm

    def body_to_world_R(self):
        """Rotation matrix Body -> World using current quaternion."""
        R = quaternion_matrix([self.qx_, self.qy_, self.qz_, self.qw_])[:3, :3]
        return R

    # -----------------------
    # Callbacks
    # -----------------------
    def reset_odometry_callback(self, request, response):
        self.x_ = 0.0
        self.y_ = 0.0
        self.z_imu_ = 0.0
        self.z_fused_ = 0.0

        self.vx_ = 0.0
        self.vy_ = 0.0
        self.vz_imu_ = 0.0

        self.integration_count_ = 0
        self.dropped_samples_ = 0
        self.warning_count_ = 0

        self.start_time_ = self.get_clock().now()
        self.prev_imu_time_ = None

        self.get_logger().info("Odometry reset to zero")
        return response


    def baro_callback(self, msg: PoseWithCovarianceStamped):
        z = msg.pose.pose.position.z
        if not self._is_finite(z):
            return
        self.baro_z_ = float(z)
        self.baro_ready_ = True

        # Initialize fused z once
        if self.integration_count_ == 0:
            self.z_imu_ = self.baro_z_
            self.z_fused_ = self.baro_z_

    def fastlio_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        vel = msg.twist.twist.linear

        if not (self._is_finite(pos.x) and self._is_finite(pos.y) and self._is_finite(pos.z)):
            return
        if not (self._is_finite(vel.x) and self._is_finite(vel.y) and self._is_finite(vel.z)):
            return

        # Transform FAST-LIO position from its frame (camera_init) to odom.
        # Use TF lookup so the offset is always correct regardless of static transform config.
        try:
            tf = self.tf_buffer_.lookup_transform(
                'odom',
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            tz = tf.transform.translation.z
        except Exception:
            # Static transform not yet available — skip this update
            return

        x_odom = float(pos.x) + tx
        y_odom = float(pos.y) + ty
        z_odom = float(pos.z) + tz

        # Soft-correct XY position toward FAST-LIO — airspeed dead reckoning owns
        # position between updates; FAST-LIO nudges it to prevent long-term drift.
        # Weight 0 = ignore FAST-LIO position, 1 = hard reset (old behaviour).
        w = self.fastlio_pos_weight_
        self.x_ = w * x_odom + (1.0 - w) * self.x_
        self.y_ = w * y_odom + (1.0 - w) * self.y_

        # Velocity: let FAST-LIO seed vx/vy only on first fix, then let
        # airspeed blend own it. Hard-resetting velocity kills the airspeed correction.
        if not self.fastlio_ready_:
            self.vx_ = float(vel.x)
            self.vy_ = float(vel.y)

        # Always anchor z_imu_ to FAST-LIO to prevent IMU Z drift.
        # Baro still dominates z_fused via its weight, but now its IMU term
        # tracks FAST-LIO Z instead of drifting double-integrated acceleration.
        self.z_imu_ = z_odom
        self.vz_imu_ = float(vel.z)

        self.fastlio_ready_ = True

    def airspeed_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val >= 0.0:
            self.last_airspeed_ = val
            self.airspeed_ready_ = True

    def wind_callback(self, msg: TwistStamped):
        wx = msg.twist.linear.x
        wy = msg.twist.linear.y
        wz = msg.twist.linear.z
        if math.isfinite(wx) and math.isfinite(wy) and math.isfinite(wz):
            self.wind_x_ = wx
            self.wind_y_ = wy
            self.wind_z_ = wz

    def imu_callback(self, msg: Imu):
        try:
            # Timestamp
            t = Time.from_msg(msg.header.stamp)

            # First IMU message init
            if self.prev_imu_time_ is None:
                self.prev_imu_time_ = t
                self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
                )
                return

            dt = (t - self.prev_imu_time_).nanoseconds / 1e9

            if dt <= 0.0:
                self.dropped_samples_ += 1
                return

            # Clamp huge dt
            if dt > 1.0:
                dt = 1.0
                self.dropped_samples_ += 1

            self.prev_imu_time_ = t

            # Update and normalize orientation
            self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
            )

            # Angular velocity (pass-through)
            wz = msg.angular_velocity.z
            self.wz_ = wz if self._is_finite(wz) else 0.0

            # Raw body acceleration reading
            a_b = np.array(
                [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                dtype=float
            )

            if not np.all(np.isfinite(a_b)):
                self.dropped_samples_ += 1
                return

            # Clamp extreme accel
            accel_mag = float(np.linalg.norm(a_b))
            if accel_mag > self.max_acceleration_:
                a_b *= (self.max_acceleration_ / max(1e-9, accel_mag))
                self.warning_count_ += 1

            # Rotation matrices
            R_bw = self.body_to_world_R()            # body -> world
            R_wb = R_bw.T                            # world -> body
            g_w = np.array([0.0, 0.0, self.g_], dtype=float)
            g_b = R_wb @ g_w

            # -----------------------
            # Bias calibration (tilt-aware)
            # -----------------------
            if self.calibrating_:
                # Ignore large dt during calibration (timing spikes)
                if dt > self.calibration_max_dt_:
                    self.dropped_samples_ += 1
                    return

                a_expected_b = -g_b

                bias_sample = a_b - a_expected_b
                self.bias_samples_.append(bias_sample)

                self.calibration_samples_ += 1
                if self.calibration_samples_ >= self.calibration_required_:
                    self.accel_bias_ = np.mean(np.vstack(self.bias_samples_), axis=0)
                    self.calibrating_ = False
                    self.get_logger().info(
                        "Calibration complete. accel_bias_ = "
                        f"[{self.accel_bias_[0]:.5f}, {self.accel_bias_[1]:.5f}, {self.accel_bias_[2]:.5f}]"
                    )
                return

            # Bias correct
            a_b_corr = a_b - self.accel_bias_

            # Rotate to world
            a_w = R_bw @ a_b_corr

            # Gravity correction (FIXED)
            if self.accel_is_sf_:
                # a = f + g
                a_w = a_w + g_w
            else:
                # a = a_meas - g
                a_w = a_w - g_w

            # Integrate IMU-based acceleration
            self.vx_ += float(a_w[0]) * dt
            self.vy_ += float(a_w[1]) * dt
            self.vz_imu_ += float(a_w[2]) * dt


            # Airspeed complementary filter for X/Y:
            # blend IMU-integrated velocity (good short-term dynamics) with
            # airspeed dead reckoning (no long-term drift) weighted by airspeed_weight_.
            # Only active above min_airspeed_ to avoid corrupting static estimates.
            airspeed_active = (self.use_airspeed_ and self.airspeed_ready_
                               and self.last_airspeed_ >= self.min_airspeed_)

            if airspeed_active:
                v_air_world = R_bw @ np.array([self.last_airspeed_, 0.0, 0.0])
                if self.use_wind_correction_:
                    # v_ground = v_air_in_world + v_wind  (wind estimator: v_wind = v_ground - v_air)
                    v_ground_x = v_air_world[0] + self.wind_x_
                    v_ground_y = v_air_world[1] + self.wind_y_
                else:
                    # Pure airspeed dead reckoning — correct for GPS-denied sim (no actual wind)
                    v_ground_x = v_air_world[0]
                    v_ground_y = v_air_world[1]
                alpha = self.airspeed_weight_
                self.vx_ = alpha * v_ground_x + (1.0 - alpha) * self.vx_
                self.vy_ = alpha * v_ground_y + (1.0 - alpha) * self.vy_

            # Time-consistent damping — skip X/Y when airspeed is active since decay
            # introduces a systematic negative velocity bias that causes position to trail.
            keep = self.velocity_decay_per_sec_ ** dt
            if not airspeed_active:
                self.vx_ *= keep
                self.vy_ *= keep
            self.vz_imu_ *= keep

            # Velocity limit
            v_mag = math.sqrt(self.vx_**2 + self.vy_**2 + self.vz_imu_**2)
            if v_mag > self.max_velocity_:
                scale = self.max_velocity_ / max(1e-9, v_mag)
                self.vx_ *= scale
                self.vy_ *= scale
                self.vz_imu_ *= scale
                self.warning_count_ += 1

            # Integrate position
            self.x_ += self.vx_ * dt
            self.y_ += self.vy_ * dt
            self.z_imu_ += self.vz_imu_ * dt

            # Fuse barometer into published z
            if self.use_barometer_ and self.baro_ready_:
                self.z_fused_ = self.imu_weight_ * self.z_imu_ + self.baro_weight_ * self.baro_z_
            else:
                self.z_fused_ = self.z_imu_

            self.integration_count_ += 1

        except Exception as e:
            self.get_logger().error(f"Error in imu_callback: {e}")

    # -----------------------
    # Publishing
    # -----------------------
    def publish_odom(self):
        now = self.get_clock().now()

        # Validate state
        vals = [self.x_, self.y_, self.z_fused_, self.vx_, self.vy_, self.vz_imu_,
                self.qx_, self.qy_, self.qz_, self.qw_]
        if not all(self._is_finite(v) for v in vals):
            return

        # Re-normalize quaternion before publishing
        self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
            self.qx_, self.qy_, self.qz_, self.qw_
        )

        # Covariances — FAST-LIO bounds drift so variance doesn't grow when it's active
        elapsed_time = (now - self.start_time_).nanoseconds / 1e9
        if self.fastlio_ready_:
            pos_variance = self.base_pos_var_
        else:
            pos_variance = self.base_pos_var_ + self.var_growth_rate_ * max(0.0, elapsed_time)
        vel_variance = self.base_vel_var_

        # Fill odometry
        self.odom_msg_.header.stamp = now.to_msg()

        self.odom_msg_.pose.pose.position.x = self.x_
        self.odom_msg_.pose.pose.position.y = self.y_
        self.odom_msg_.pose.pose.position.z = self.z_fused_

        self.odom_msg_.pose.pose.orientation.x = self.qx_
        self.odom_msg_.pose.pose.orientation.y = self.qy_
        self.odom_msg_.pose.pose.orientation.z = self.qz_
        self.odom_msg_.pose.pose.orientation.w = self.qw_

        self.odom_msg_.twist.twist.linear.x = self.vx_
        self.odom_msg_.twist.twist.linear.y = self.vy_
        self.odom_msg_.twist.twist.linear.z = self.vz_imu_
        self.odom_msg_.twist.twist.angular.z = self.wz_

        # Pose covariance (x,y,z,roll,pitch,yaw)
        self.odom_msg_.pose.covariance = [
            pos_variance, 0, 0, 0, 0, 0,
            0, pos_variance, 0, 0, 0, 0,
            0, 0, pos_variance, 0, 0, 0,
            0, 0, 0, 0.01, 0, 0,
            0, 0, 0, 0, 0.01, 0,
            0, 0, 0, 0, 0, 0.01
        ]

        # Twist covariance (vx,vy,vz,wx,wy,wz)
        self.odom_msg_.twist.covariance = [
            vel_variance, 0, 0, 0, 0, 0,
            0, vel_variance, 0, 0, 0, 0,
            0, 0, vel_variance, 0, 0, 0,
            0, 0, 0, 0.1, 0, 0,
            0, 0, 0, 0, 0.1, 0,
            0, 0, 0, 0, 0, 0.1
        ]

        self.odom_pub_.publish(self.odom_msg_)

        # Diagnostics every ~2s
        if self.integration_count_ > 0 and (self.integration_count_ % int(max(1, self.publish_rate_ * 2)) == 0):
            self.publish_diagnostics(now)

        # TF
        if self.publish_tf_:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = "odom"
            t.child_frame_id = self.child_frame_id_

            t.transform.translation.x = self.x_
            t.transform.translation.y = self.y_
            t.transform.translation.z = self.z_fused_
            t.transform.rotation.x = self.qx_
            t.transform.rotation.y = self.qy_
            t.transform.rotation.z = self.qz_
            t.transform.rotation.w = self.qw_
            self.tf_broadcaster_.sendTransform(t)

    def publish_diagnostics(self, now):
        diag_array = DiagnosticArray()
        diag_array.header.stamp = now.to_msg()

        status = DiagnosticStatus()
        status.name = "IMU Odometry"
        status.hardware_id = "marid_odom_node"

        if self.dropped_samples_ > 100 or self.warning_count_ > 50:
            status.level = DiagnosticStatus.WARN
            status.message = "High error/warning rate detected"
        else:
            status.level = DiagnosticStatus.OK
            status.message = "Operating normally"

        status.values.append(KeyValue(key="Integration Count", value=str(self.integration_count_)))
        status.values.append(KeyValue(key="Dropped Samples", value=str(self.dropped_samples_)))
        status.values.append(KeyValue(key="Warnings", value=str(self.warning_count_)))
        status.values.append(KeyValue(key="Position X", value=f"{self.x_:.3f}"))
        status.values.append(KeyValue(key="Position Y", value=f"{self.y_:.3f}"))
        status.values.append(KeyValue(key="Position Z (fused)", value=f"{self.z_fused_:.3f}"))

        vmag = math.sqrt(self.vx_**2 + self.vy_**2 + self.vz_imu_**2)
        status.values.append(KeyValue(key="Velocity Magnitude", value=f"{vmag:.3f}"))

        status.values.append(KeyValue(key="Accel Bias X", value=f"{self.accel_bias_[0]:.5f}"))
        status.values.append(KeyValue(key="Accel Bias Y", value=f"{self.accel_bias_[1]:.5f}"))
        status.values.append(KeyValue(key="Accel Bias Z", value=f"{self.accel_bias_[2]:.5f}"))

        status.values.append(KeyValue(key="IMU specific force", value=str(self.accel_is_sf_)))
        status.values.append(KeyValue(key="Barometer ready", value=str(self.baro_ready_)))
        status.values.append(KeyValue(key="Baro weight", value=f"{self.baro_weight_:.2f}"))
        status.values.append(KeyValue(key="FAST-LIO ready", value=str(self.fastlio_ready_)))
        status.values.append(KeyValue(key="Airspeed ready", value=str(self.airspeed_ready_)))
        status.values.append(KeyValue(key="Airspeed (m/s)", value=f"{self.last_airspeed_:.2f}"))
        status.values.append(KeyValue(key="Wind X", value=f"{self.wind_x_:.2f}"))
        status.values.append(KeyValue(key="Wind Y", value=f"{self.wind_y_:.2f}"))

        diag_array.status.append(status)
        self.diag_pub_.publish(diag_array)


def main():
    rclpy.init()
    node = MaridOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
