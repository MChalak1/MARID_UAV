#!/usr/bin/env python3
import math
import numpy as np

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped, TwistStamped, TwistWithCovarianceStamped
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
        # 0.7 works well in feature-rich environments (poles/walls); lower in open fields.
        self.declare_parameter("fastlio_position_weight", 0.7)
        # Blend weight for FAST-LIO yaw correction (0=ignore, 1=hard override).
        # Uses FAST-LIO orientation yaw to correct heading drift when scan-matching is reliable.
        self.declare_parameter("fastlio_yaw_weight", 0.3)

        # IMU X/Y integration toggle.
        # When False, IMU accelerations are NOT integrated into vx/vy.
        # Velocity then comes only from airspeed, optical flow, or FAST-LIO seeding,
        # and decays toward zero between updates via velocity_decay_rate.
        # Recommended when FAST-LIO is the primary position source: disabling IMU X/Y
        # integration eliminates double-integration drift that fights the FAST-LIO correction
        # and causes the characteristic sawtooth drift pattern.
        self.declare_parameter("use_imu_xy", True)

        # Airspeed complementary filter
        self.declare_parameter("use_airspeed", True)
        self.declare_parameter("airspeed_weight", 0.8)
        self.declare_parameter("min_airspeed_for_fusion", 0.5)   # m/s — only skip when truly static
        self.declare_parameter("wind_topic", "/wind/velocity")
        # Disable wind correction by default: in simulation there is no actual wind, and the
        # wind estimator uses EKF/Gazebo ground truth which leaks into GPS-denied estimation.
        # Enable only for real flight where wind is genuine and estimated independently.
        self.declare_parameter("use_wind_correction", False)

        # Optical flow fusion
        self.declare_parameter("use_optical_flow", True)
        self.declare_parameter("optical_flow_topic", "/optical_flow/velocity")
        # Weight for OF velocity vs IMU velocity (0=ignore OF, 1=hard OF override)
        self.declare_parameter("optical_flow_weight", 0.7)
        # OF is suppressed above this altitude (sonar valid range)
        self.declare_parameter("max_of_altitude", 4.5)

        # Forward camera optical flow fusion
        # Provides lateral (vy) and vertical (vz) body-frame velocity in flight.
        # Complements downward OF (which covers vx+vy below max_of_altitude).
        self.declare_parameter("use_forward_camera", True)
        self.declare_parameter("forward_camera_topic", "/forward_camera/velocity")
        # Blend weight for forward camera vy (0=ignore, 1=hard override)
        self.declare_parameter("forward_camera_weight", 0.5)
        # Only fuse above this altitude — below it the downward camera is preferred
        self.declare_parameter("min_forward_camera_altitude", 1.0)

        # Sonar altitude anchor
        self.declare_parameter("use_sonar", True)
        self.declare_parameter("sonar_range_topic", "/sonar/range")
        # Weight to anchor z_imu_ toward sonar AGL when sonar is valid
        self.declare_parameter("sonar_weight", 0.95)
        self.declare_parameter("max_sonar_altitude", 5.0)

        # Wheel odometry fusion (ground taxiing)
        self.declare_parameter("use_wheel_odom", True)
        self.declare_parameter("wheel_odom_topic", "/wheel/odometry")
        # Sonar AGL below which wheel odometry owns XY position and velocity.
        # Must match (or be slightly larger than) ground_threshold in wheel_odometry.py.
        self.declare_parameter("ground_threshold", 0.30)

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
        self.fastlio_yaw_weight_ = max(0.0, min(1.0, float(
            self.get_parameter("fastlio_yaw_weight").value)))

        self.use_imu_xy_ = bool(self.get_parameter("use_imu_xy").value)

        self.use_airspeed_ = bool(self.get_parameter("use_airspeed").value)
        self.airspeed_weight_ = max(0.0, min(1.0, float(self.get_parameter("airspeed_weight").value)))
        self.min_airspeed_ = float(self.get_parameter("min_airspeed_for_fusion").value)
        self.wind_topic_ = str(self.get_parameter("wind_topic").value)
        self.use_wind_correction_ = bool(self.get_parameter("use_wind_correction").value)

        self.use_optical_flow_  = bool(self.get_parameter("use_optical_flow").value)
        self.of_topic_          = str(self.get_parameter("optical_flow_topic").value)
        self.of_weight_         = max(0.0, min(1.0, float(self.get_parameter("optical_flow_weight").value)))
        self.max_of_altitude_   = float(self.get_parameter("max_of_altitude").value)

        self.use_sonar_         = bool(self.get_parameter("use_sonar").value)
        self.sonar_range_topic_ = str(self.get_parameter("sonar_range_topic").value)
        self.sonar_weight_      = max(0.0, min(1.0, float(self.get_parameter("sonar_weight").value)))
        self.max_sonar_alt_     = float(self.get_parameter("max_sonar_altitude").value)

        self.use_forward_camera_      = bool(self.get_parameter("use_forward_camera").value)
        self.fwd_cam_topic_           = str(self.get_parameter("forward_camera_topic").value)
        self.fwd_cam_weight_          = max(0.0, min(1.0, float(
            self.get_parameter("forward_camera_weight").value)))
        self.min_fwd_cam_altitude_    = float(self.get_parameter("min_forward_camera_altitude").value)

        self.use_wheel_odom_    = bool(self.get_parameter("use_wheel_odom").value)
        self.wheel_odom_topic_  = str(self.get_parameter("wheel_odom_topic").value)
        self.ground_threshold_  = float(self.get_parameter("ground_threshold").value)

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
        self.last_fastlio_stamp_ = None      # rclpy.Time of last FAST-LIO message
        self.last_fastlio_x_ = 0.0          # previous FAST-LIO position for finite-diff velocity
        self.last_fastlio_y_ = 0.0
        self.last_fastlio_z_ = 0.0
        self.fastlio_timeout_ = 0.5         # seconds — declare FAST-LIO stale after this
        # Rotation matrix from camera_init to odom.
        # camera_init is FAST-LIO's inertial frame, aligned with the drone's body frame
        # at the moment of the first LiDAR scan. Its orientation relative to odom equals
        # the drone's initial IMU attitude — captured once and held fixed.
        # None until the first FAST-LIO message sets it.
        self.camera_init_to_odom_R_ = None

        # Airspeed fusion state
        self.last_airspeed_ = 0.0
        self.airspeed_ready_ = False
        self.wind_x_ = 0.0
        self.wind_y_ = 0.0
        self.wind_z_ = 0.0

        # Optical flow fusion state (body-frame velocity from OF estimator)
        self.of_vx_body_ = 0.0
        self.of_vy_body_ = 0.0
        self.of_ready_   = False
        self.of_seeded_  = False   # True after first OF activation seeds vx_/vy_

        # Forward camera fusion state (body-frame vy, vz from forward flow estimator)
        self.fwd_cam_vy_body_ = 0.0
        self.fwd_cam_vz_body_ = 0.0
        self.fwd_cam_ready_   = False

        # Sonar altitude state
        self.sonar_range_ = None
        self.sonar_ready_ = False

        # Wheel odometry fusion state
        self.wheel_odom_ready_ = False
        self.wheel_x_    = 0.0
        self.wheel_y_    = 0.0
        self.wheel_vfwd_ = 0.0   # body-frame forward speed from wheel_odometry node

        # Time tracking
        self.start_time_ = self.get_clock().now()
        self.prev_imu_time_ = None
        self.prev_publish_time_ = None

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

        if self.use_optical_flow_:
            self.of_sub_ = self.create_subscription(
                TwistWithCovarianceStamped, self.of_topic_, self.of_callback, 10
            )

        if self.use_sonar_:
            self.sonar_sub_ = self.create_subscription(
                Range, self.sonar_range_topic_, self.sonar_callback, 10
            )

        if self.use_forward_camera_:
            self.fwd_cam_sub_ = self.create_subscription(
                TwistWithCovarianceStamped, self.fwd_cam_topic_, self.fwd_cam_callback, 10
            )

        if self.use_wheel_odom_:
            self.wheel_odom_sub_ = self.create_subscription(
                Odometry, self.wheel_odom_topic_, self.wheel_odom_callback, 10
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

    def _update_z_fused(self):
        """Recompute z_fused_ from FAST-LIO z_imu_ and barometer."""
        if self.use_barometer_ and self.baro_ready_:
            self.z_fused_ = (1.0 - self.baro_weight_) * self.z_imu_ + self.baro_weight_ * self.baro_z_
        else:
            self.z_fused_ = self.z_imu_

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
        self.prev_publish_time_ = None

        # Re-capture camera_init rotation on next FAST-LIO message
        self.fastlio_ready_ = False
        self.camera_init_to_odom_R_ = None
        self.last_fastlio_stamp_ = None

        self.wheel_odom_ready_ = False
        self.wheel_x_    = 0.0
        self.wheel_y_    = 0.0
        self.wheel_vfwd_ = 0.0

        self.get_logger().info("Odometry reset to zero")
        return response


    def baro_callback(self, msg: PoseWithCovarianceStamped):
        z = msg.pose.pose.position.z
        if not self._is_finite(z):
            return
        self.baro_z_ = float(z)
        self.baro_ready_ = True
        # Recompute fused Z whenever baro updates
        self._update_z_fused()

    def fastlio_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        if not (self._is_finite(pos.x) and self._is_finite(pos.y) and self._is_finite(pos.z)):
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

        # Use the full TF (rotation + translation) to transform FAST-LIO positions into odom.
        # The static TF defines the true camera_init→odom relationship. Using the IMU body
        # orientation here (the old camera_init_to_odom_R_ approach) was wrong: Madgwick
        # without a magnetometer has arbitrary initial yaw, so it baked a random rotation
        # into every FAST-LIO position and caused the drift/reversal behaviour.
        tf_q = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        R_tf = quaternion_matrix(tf_q)[:3, :3]
        p_camera = np.array([float(pos.x), float(pos.y), float(pos.z)])
        p_odom = R_tf @ p_camera + np.array([tx, ty, tz])
        x_odom = float(p_odom[0])
        y_odom = float(p_odom[1])
        z_odom = float(p_odom[2])

        now = self.get_clock().now()

        # Velocity from finite-differencing consecutive FAST-LIO positions.
        # FAST-LIO does not populate the twist field of its Odometry message — the
        # velocity in msg.twist.twist is always zero. Differencing positions gives
        # world-frame velocity directly with no frame rotation needed.
        if self.fastlio_ready_ and self.last_fastlio_stamp_ is not None:
            dt_fl = (now - self.last_fastlio_stamp_).nanoseconds / 1e9
            if 0.005 < dt_fl < self.fastlio_timeout_:
                self.vx_ = (x_odom - self.last_fastlio_x_) / dt_fl
                self.vy_ = (y_odom - self.last_fastlio_y_) / dt_fl
                vz_fl    = (z_odom - self.last_fastlio_z_) / dt_fl
                # Blend FAST-LIO Z velocity with IMU Z velocity (IMU is better for short intervals)
                self.vz_imu_ = 0.5 * vz_fl + 0.5 * self.vz_imu_

        self.last_fastlio_stamp_ = now
        self.last_fastlio_x_ = x_odom
        self.last_fastlio_y_ = y_odom
        self.last_fastlio_z_ = z_odom

        # Position: hard-reset from FAST-LIO once scan-matching has converged.
        # Suppressed on the ground — wheel odometry owns XY there and fires faster.
        if not (self.use_wheel_odom_ and self._on_ground()):
            if not self.fastlio_ready_:
                # First lock: soft blend in case FAST-LIO hasn't converged yet
                w = self.fastlio_pos_weight_
                self.x_ = w * x_odom + (1.0 - w) * self.x_
                self.y_ = w * y_odom + (1.0 - w) * self.y_
            else:
                # Converged: use FAST-LIO position directly as ground truth
                self.x_ = x_odom
                self.y_ = y_odom

        # Anchor z_imu_ to FAST-LIO first, then refine with sonar when in range.
        # Sonar gives a direct AGL reading that is more accurate than LiDAR scan-match Z
        # at low altitudes (< max_sonar_alt_), especially over flat ground.
        self.z_imu_ = z_odom
        if (self.use_sonar_ and self.sonar_ready_
                and self.sonar_range_ is not None
                and self.sonar_range_ <= self.max_sonar_alt_):
            self.z_imu_ = (self.sonar_weight_ * self.sonar_range_
                           + (1.0 - self.sonar_weight_) * self.z_imu_)
        self._update_z_fused()

        # Blend FAST-LIO yaw into current heading when scan-matching is reliable.
        # Skip on the very first message (fastlio_ready_ still False): FAST-LIO orientation
        # is also unreliable before the first scan-match completes and injecting a bad yaw
        # would corrupt the rotation matrix used by the IMU integration loop.
        if (self.fastlio_ready_
                and self.fastlio_yaw_weight_ > 0.0
                and self._is_finite(ori.x) and self._is_finite(ori.y)
                and self._is_finite(ori.z) and self._is_finite(ori.w)):
            # Extract yaw from FAST-LIO quaternion
            fl_yaw = math.atan2(
                2.0 * (ori.w * ori.z + ori.x * ori.y),
                1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
            )
            # Extract current yaw from IMU quaternion
            imu_yaw = math.atan2(
                2.0 * (self.qw_ * self.qz_ + self.qx_ * self.qy_),
                1.0 - 2.0 * (self.qy_ * self.qy_ + self.qz_ * self.qz_)
            )
            # Interpolate yaw (handle wrap-around)
            dyaw = fl_yaw - imu_yaw
            if dyaw > math.pi:
                dyaw -= 2.0 * math.pi
            elif dyaw < -math.pi:
                dyaw += 2.0 * math.pi
            blended_yaw = imu_yaw + self.fastlio_yaw_weight_ * dyaw
            # Extract roll/pitch from current IMU quaternion, rebuild with blended yaw
            sinr_cosp = 2.0 * (self.qw_ * self.qx_ + self.qy_ * self.qz_)
            cosr_cosp = 1.0 - 2.0 * (self.qx_ * self.qx_ + self.qy_ * self.qy_)
            imu_roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_)
            imu_pitch = math.asin(max(-1.0, min(1.0, sinp)))
            cr, cp, cy = math.cos(imu_roll / 2), math.cos(imu_pitch / 2), math.cos(blended_yaw / 2)
            sr, sp, sy = math.sin(imu_roll / 2), math.sin(imu_pitch / 2), math.sin(blended_yaw / 2)
            self.qw_ = cr * cp * cy + sr * sp * sy
            self.qx_ = sr * cp * cy - cr * sp * sy
            self.qy_ = cr * sp * cy + sr * cp * sy
            self.qz_ = cr * cp * sy - sr * sp * cy
            self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
                self.qx_, self.qy_, self.qz_, self.qw_)

        self.fastlio_ready_ = True

    def airspeed_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val >= 0.0:
            self.last_airspeed_ = val
            self.airspeed_ready_ = True

    def of_callback(self, msg: TwistWithCovarianceStamped):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if math.isfinite(vx) and math.isfinite(vy):
            self.of_vx_body_ = vx
            self.of_vy_body_ = vy
            self.of_ready_   = True

    def fwd_cam_callback(self, msg: TwistWithCovarianceStamped):
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        if math.isfinite(vy) and math.isfinite(vz):
            self.fwd_cam_vy_body_ = vy
            self.fwd_cam_vz_body_ = vz
            self.fwd_cam_ready_   = True

    def sonar_callback(self, msg: Range):
        r = float(msg.range)
        if math.isfinite(r) and msg.min_range <= r <= msg.max_range:
            self.sonar_range_ = r
            self.sonar_ready_ = True
        else:
            self.sonar_ready_ = False

    def _on_ground(self) -> bool:
        """True when sonar confirms the drone is on the ground."""
        return (self.sonar_ready_
                and self.sonar_range_ is not None
                and self.sonar_range_ <= self.ground_threshold_)

    def wheel_odom_callback(self, msg: Odometry):
        """Consume /wheel/odometry — own XY position and velocity on the ground."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x   # body-frame forward speed
        if not (self._is_finite(x) and self._is_finite(y) and self._is_finite(v)):
            return
        self.wheel_x_    = x
        self.wheel_y_    = y
        self.wheel_vfwd_ = v
        self.wheel_odom_ready_ = True
        # While on the ground, wheel odometry is the primary XY position source.
        # Update here (at ~50 Hz) so the published odometry stays tight between
        # FAST-LIO corrections (which only fire at ~10 Hz and are suppressed on ground).
        if self._on_ground():
            self.x_ = x
            self.y_ = y

    def wind_callback(self, msg: TwistStamped):
        wx = msg.twist.linear.x
        wy = msg.twist.linear.y
        wz = msg.twist.linear.z
        if math.isfinite(wx) and math.isfinite(wy) and math.isfinite(wz):
            self.wind_x_ = wx
            self.wind_y_ = wy
            self.wind_z_ = wz

    def imu_callback(self, msg: Imu):
        # Orientation and angular velocity only — zero integration.
        # Position is owned entirely by fastlio_callback.
        self.qx_, self.qy_, self.qz_, self.qw_ = self.normalize_quaternion(
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        wz = msg.angular_velocity.z
        self.wz_ = wz if self._is_finite(wz) else 0.0
        self.integration_count_ += 1

    # -----------------------
    # Publishing
    # -----------------------
    def publish_odom(self):
        now = self.get_clock().now()

        # Dead-reckon XY from optical flow between FAST-LIO scans.
        # FAST-LIO updates at ~10 Hz; this timer fires at 50 Hz. Integrating OF velocity
        # here propagates position at full rate and keeps XY valid during brief scan gaps.
        # FAST-LIO hard-resets x_/y_ in fastlio_callback, so OF drift can't accumulate
        # beyond one scan interval (~100 ms).
        if (self.prev_publish_time_ is not None
                and self.fastlio_ready_
                and self.use_optical_flow_ and self.of_ready_
                and self.z_fused_ <= self.max_of_altitude_
                and not (self.use_wheel_odom_ and self._on_ground())):
            dt = (now - self.prev_publish_time_).nanoseconds / 1e9
            if 0.0 < dt < 0.5:
                R = self.body_to_world_R()
                v_body = np.array([self.of_vx_body_, self.of_vy_body_, 0.0])
                v_world = R @ v_body
                self.x_ += v_world[0] * dt
                self.y_ += v_world[1] * dt
        self.prev_publish_time_ = now

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

        # Velocity for the published message.
        # Priority: wheel odometry (ground) > optical flow (low altitude) > FAST-LIO finite-diff
        vx_pub = self.vx_
        vy_pub = self.vy_
        if self.use_wheel_odom_ and self.wheel_odom_ready_ and self._on_ground():
            # Rotate body-frame forward speed into world frame using current IMU orientation.
            R = self.body_to_world_R()
            v_body = np.array([self.wheel_vfwd_, 0.0, 0.0])
            v_world = R @ v_body
            vx_pub = v_world[0]
            vy_pub = v_world[1]
        elif (self.use_optical_flow_ and self.of_ready_
                and self.z_fused_ <= self.max_of_altitude_):
            # Blend optical-flow world-frame velocity into vx/vy when valid and below max altitude.
            R = self.body_to_world_R()
            v_body = np.array([self.of_vx_body_, self.of_vy_body_, 0.0])
            v_world = R @ v_body
            vx_pub = self.of_weight_ * v_world[0] + (1.0 - self.of_weight_) * self.vx_
            vy_pub = self.of_weight_ * v_world[1] + (1.0 - self.of_weight_) * self.vy_

        # Forward camera vy correction — active in flight above min_fwd_cam_altitude_.
        # Blends lateral velocity regardless of which source provided vx/vy above,
        # constraining lateral drift where the downward camera is inactive.
        if (self.use_forward_camera_ and self.fwd_cam_ready_
                and not self._on_ground()
                and self.z_fused_ >= self.min_fwd_cam_altitude_):
            R = self.body_to_world_R()
            vy_world_fwd = (R @ np.array([0.0, self.fwd_cam_vy_body_, 0.0]))[1]
            vy_pub = self.fwd_cam_weight_ * vy_world_fwd + (1.0 - self.fwd_cam_weight_) * vy_pub

        self.odom_msg_.twist.twist.linear.x = vx_pub
        self.odom_msg_.twist.twist.linear.y = vy_pub
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
        status.values.append(KeyValue(key="IMU XY integration", value=str(self.use_imu_xy_)))
        status.values.append(KeyValue(key="Barometer ready", value=str(self.baro_ready_)))
        status.values.append(KeyValue(key="Baro weight", value=f"{self.baro_weight_:.2f}"))
        status.values.append(KeyValue(key="FAST-LIO ready", value=str(self.fastlio_ready_)))
        status.values.append(KeyValue(key="FAST-LIO pos weight", value=f"{self.fastlio_pos_weight_:.2f}"))
        status.values.append(KeyValue(key="FAST-LIO yaw weight", value=f"{self.fastlio_yaw_weight_:.2f}"))
        status.values.append(KeyValue(key="Airspeed ready", value=str(self.airspeed_ready_)))
        status.values.append(KeyValue(key="Airspeed (m/s)", value=f"{self.last_airspeed_:.2f}"))
        status.values.append(KeyValue(key="Wind X", value=f"{self.wind_x_:.2f}"))
        status.values.append(KeyValue(key="Wind Y", value=f"{self.wind_y_:.2f}"))
        status.values.append(KeyValue(key="OF ready", value=str(self.of_ready_)))
        status.values.append(KeyValue(key="OF vx_body", value=f"{self.of_vx_body_:.3f}"))
        status.values.append(KeyValue(key="OF vy_body", value=f"{self.of_vy_body_:.3f}"))
        sonar_str = f"{self.sonar_range_:.3f}" if self.sonar_range_ is not None else "N/A"
        status.values.append(KeyValue(key="Sonar AGL (m)", value=sonar_str))
        status.values.append(KeyValue(key="On ground", value=str(self._on_ground())))
        status.values.append(KeyValue(key="Wheel odom ready", value=str(self.wheel_odom_ready_)))
        status.values.append(KeyValue(key="Wheel vfwd (m/s)", value=f"{self.wheel_vfwd_:.3f}"))
        status.values.append(KeyValue(key="Fwd cam ready", value=str(self.fwd_cam_ready_)))
        status.values.append(KeyValue(key="Fwd cam vy_body", value=f"{self.fwd_cam_vy_body_:.3f}"))
        status.values.append(KeyValue(key="Fwd cam vz_body", value=f"{self.fwd_cam_vz_body_:.3f}"))

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
