#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster

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

        # -----------------------
        # TF broadcaster
        # -----------------------
        self.tf_broadcaster_ = TransformBroadcaster(self)

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


            # Time-consistent damping
            keep = self.velocity_decay_per_sec_ ** dt
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

        # Covariances
        elapsed_time = (now - self.start_time_).nanoseconds / 1e9
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
