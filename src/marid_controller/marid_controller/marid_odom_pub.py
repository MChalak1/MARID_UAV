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
        # Altitude above which scan matching degrades (open sky, no nearby features).
        # FAST-LIO is treated as stale above this and thrust DR takes over.
        self.declare_parameter("max_fastlio_altitude", 30.0)  # m AGL
        # EMA coefficient for FAST-LIO finite-diff velocity (α ∈ (0,1]).
        # Raw finite-diff at ~10 Hz introduces ~0.5 m/s noise; α=0.3 gives τ ≈ 0.23 s
        # smoothing without meaningful lag at cruise.  α=1.0 disables smoothing.
        self.declare_parameter("fastlio_velocity_alpha", 0.3)

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
        # Airspeed sensor staleness timeout: if no message arrives within this window the
        # last reading is discarded and the next priority source is used instead.
        self.declare_parameter("airspeed_timeout", 1.0)    # seconds

        # Optical flow fusion
        self.declare_parameter("use_optical_flow", True)
        self.declare_parameter("optical_flow_topic", "/optical_flow/velocity")
        # Weight for OF velocity vs IMU velocity (0=ignore OF, 1=hard OF override)
        self.declare_parameter("optical_flow_weight", 0.7)
        # OF is suppressed above this altitude (sonar valid range)
        self.declare_parameter("max_of_altitude", 4.5)
        # OF is suppressed below this altitude (too close to ground for valid flow)
        self.declare_parameter("min_of_altitude", 0.3)
        # Maximum body pitch angle (degrees) at which OF vx is still trusted.
        # Above this threshold the downward camera tilts enough to mix pitch
        # rotation into the flow reading, corrupting the forward velocity estimate.
        self.declare_parameter("max_of_pitch_deg", 5.0)
        self.declare_parameter("of_timeout", 0.5)          # seconds

        # Forward camera optical flow fusion
        # Provides lateral (vy) and vertical (vz) body-frame velocity in flight.
        # Complements downward OF (which covers vx+vy below max_of_altitude).
        self.declare_parameter("use_forward_camera", True)
        self.declare_parameter("forward_camera_topic", "/forward_camera/velocity")
        # Blend weight for forward camera vy (0=ignore, 1=hard override)
        self.declare_parameter("forward_camera_weight", 0.5)
        # Only fuse above this altitude — below it the downward camera is preferred
        self.declare_parameter("min_forward_camera_altitude", 1.0)
        self.declare_parameter("fwd_cam_timeout", 0.5)     # seconds

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

        # Thrust dead reckoning + online drag calibration via FAST-LIO
        self.declare_parameter("use_thrust_dr", True)
        self.declare_parameter("thrust_topic", "/model/marid/joint/thruster_center_joint/cmd_thrust")
        self.declare_parameter("drone_mass", 190.0)           # kg
        self.declare_parameter("drag_coeff_init", 0.5)        # k_d initial guess
        self.declare_parameter("thrust_dr_weight", 0.7)       # blend weight for vx when FAST-LIO stale
        self.declare_parameter("rls_forgetting_factor", 0.99)
        self.declare_parameter("min_speed_for_drag_id", 2.0)  # m/s — skip RLS at near-zero speed
        self.declare_parameter("air_density", 1.225)          # kg/m³ ISA sea-level default
        # If no thrust message arrives within this window, assume glide (thrust = 0).
        # Physics integration keeps running — drag still decelerates the estimate.
        self.declare_parameter("thrust_timeout", 0.5)         # seconds

        # H₂ mass tracking — updates drone_mass_ as fuel is consumed.
        # Primary estimator: thrust-based SFC integration (always available).
        # Optional correction: ideal gas law from tank pressure + temperature sensors.
        self.declare_parameter("track_h2_mass", True)
        self.declare_parameter("drone_empty_mass", 160.0)       # kg — structural mass without fuel
        self.declare_parameter("h2_initial_mass", 30.0)         # kg — full tank at takeoff
        # Thrust-specific fuel consumption: kg of H₂ per Newton per second.
        # Approximate for a hydrogen fuel cell + electric motor at ~55% system efficiency.
        self.declare_parameter("specific_fuel_consumption", 5.0e-6)  # kg/(N·s)
        self.declare_parameter("tank_pressure_topic", "/tank/pressure")      # std_msgs/Float64, Pa
        self.declare_parameter("tank_temperature_topic", "/tank/temperature") # std_msgs/Float64, K
        self.declare_parameter("tank_volume", 0.5)              # m³ — usable gas-phase volume
        # How strongly tank sensor reading corrects the integrated estimate (0=ignore, 1=hard reset).
        self.declare_parameter("tank_sensor_weight", 0.01)

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
        self.max_fastlio_alt_    = float(self.get_parameter("max_fastlio_altitude").value)
        self.fastlio_vel_alpha_  = max(0.0, min(1.0, float(
            self.get_parameter("fastlio_velocity_alpha").value)))

        self.use_imu_xy_ = bool(self.get_parameter("use_imu_xy").value)

        self.use_airspeed_ = bool(self.get_parameter("use_airspeed").value)
        self.airspeed_weight_ = max(0.0, min(1.0, float(self.get_parameter("airspeed_weight").value)))
        self.min_airspeed_ = float(self.get_parameter("min_airspeed_for_fusion").value)
        self.wind_topic_ = str(self.get_parameter("wind_topic").value)
        self.use_wind_correction_ = bool(self.get_parameter("use_wind_correction").value)
        self.airspeed_timeout_   = float(self.get_parameter("airspeed_timeout").value)

        self.use_optical_flow_  = bool(self.get_parameter("use_optical_flow").value)
        self.of_topic_          = str(self.get_parameter("optical_flow_topic").value)
        self.of_weight_         = max(0.0, min(1.0, float(self.get_parameter("optical_flow_weight").value)))
        self.max_of_altitude_   = float(self.get_parameter("max_of_altitude").value)
        self.min_of_altitude_   = float(self.get_parameter("min_of_altitude").value)
        self.max_of_pitch_rad_  = math.radians(float(self.get_parameter("max_of_pitch_deg").value))
        self.of_timeout_        = float(self.get_parameter("of_timeout").value)

        self.use_sonar_         = bool(self.get_parameter("use_sonar").value)
        self.sonar_range_topic_ = str(self.get_parameter("sonar_range_topic").value)
        self.sonar_weight_      = max(0.0, min(1.0, float(self.get_parameter("sonar_weight").value)))
        self.max_sonar_alt_     = float(self.get_parameter("max_sonar_altitude").value)

        self.use_forward_camera_      = bool(self.get_parameter("use_forward_camera").value)
        self.fwd_cam_topic_           = str(self.get_parameter("forward_camera_topic").value)
        self.fwd_cam_weight_          = max(0.0, min(1.0, float(
            self.get_parameter("forward_camera_weight").value)))
        self.min_fwd_cam_altitude_    = float(self.get_parameter("min_forward_camera_altitude").value)
        self.fwd_cam_timeout_         = float(self.get_parameter("fwd_cam_timeout").value)

        self.use_wheel_odom_    = bool(self.get_parameter("use_wheel_odom").value)
        self.wheel_odom_topic_  = str(self.get_parameter("wheel_odom_topic").value)
        self.ground_threshold_  = float(self.get_parameter("ground_threshold").value)

        self.use_thrust_dr_     = bool(self.get_parameter("use_thrust_dr").value)
        self.thrust_topic_      = str(self.get_parameter("thrust_topic").value)
        self.drone_mass_        = float(self.get_parameter("drone_mass").value)
        self.drag_coeff_        = float(self.get_parameter("drag_coeff_init").value)
        self.thrust_dr_weight_  = max(0.0, min(1.0, float(self.get_parameter("thrust_dr_weight").value)))
        self.rls_lambda_        = float(self.get_parameter("rls_forgetting_factor").value)
        self.min_speed_drag_id_ = float(self.get_parameter("min_speed_for_drag_id").value)
        self.air_density_       = float(self.get_parameter("air_density").value)
        self.thrust_timeout_    = float(self.get_parameter("thrust_timeout").value)

        self.track_h2_mass_    = bool(self.get_parameter("track_h2_mass").value)
        self.drone_empty_mass_ = float(self.get_parameter("drone_empty_mass").value)
        self.h2_initial_mass_  = float(self.get_parameter("h2_initial_mass").value)
        self.sfc_              = float(self.get_parameter("specific_fuel_consumption").value)
        self.tank_volume_      = float(self.get_parameter("tank_volume").value)
        self.tank_sensor_w_    = float(self.get_parameter("tank_sensor_weight").value)
        self.R_H2_             = 8314.0 / 2.016   # J/(kg·K) specific gas constant for H₂

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
        self.fastlio_vel_initialized_ = False  # True after first valid finite-diff velocity
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
        self.last_airspeed_stamp_ = None
        self.wind_x_ = 0.0
        self.wind_y_ = 0.0
        self.wind_z_ = 0.0

        # Optical flow fusion state (body-frame velocity from OF estimator)
        self.of_vx_body_ = 0.0
        self.of_vy_body_ = 0.0
        self.of_ready_   = False
        self.of_seeded_  = False   # True after first OF activation seeds vx_/vy_
        self.last_of_stamp_   = None

        # Forward camera fusion state (body-frame vy, vz from forward flow estimator)
        self.fwd_cam_vy_body_ = 0.0
        self.fwd_cam_vz_body_ = 0.0
        self.fwd_cam_ready_   = False
        self.last_fwd_cam_stamp_ = None

        # Sonar altitude state
        self.sonar_range_ = None
        self.sonar_ready_ = False
        self.last_sonar_stamp_ = None
        self.sonar_timeout_    = 1.0   # seconds — sonar declared stale after this

        # Debounce counter for on-ground detection used in vx source switching.
        # A single noisy sonar reading below ground_threshold_ must not snap vx to 0.
        # Physical location and wheel-odom position use raw _on_ground() for fast response;
        # only the vx source decision is debounced.
        self.on_ground_ticks_ = 0
        self.on_ground_debounce_ticks_ = 3   # ~60 ms at 50 Hz

        # Wheel odometry fusion state
        self.wheel_odom_ready_ = False
        self.wheel_x_    = 0.0
        self.wheel_y_    = 0.0
        self.wheel_vfwd_ = 0.0   # body-frame forward speed from wheel_odometry node

        # Thrust dead reckoning state.
        # thrust_dr_ready_ starts True: with thrust_N_=0 the model runs as a pure glide
        # from the first publish tick, applying drag and gravity even before the first
        # thrust command arrives.
        self.thrust_N_          = 0.0
        self.thrust_dr_vx_      = 0.0   # body-frame forward velocity estimate
        self.thrust_dr_ready_   = True
        self.last_thrust_stamp_ = None
        self.prev_on_ground_    = True  # tracks ground→air transition for liftoff seed
        self.liftoff_seeded_    = False  # one-shot: prevents re-seeding on sonar flicker

        # Last published velocities — used as hold/decay fallback so a single bad sensor
        # tick doesn't snap vx_pub to 0 (self.vx_ is always 0 when FAST-LIO is off).
        self.last_vx_pub_ = 0.0
        self.last_vy_pub_ = 0.0
        self.rls_P_            = 1.0   # RLS scalar covariance for drag_coeff_
        self.rls_vx_prev_      = None  # previous FAST-LIO body-frame vx for finite-diff
        self.rls_time_prev_    = None

        # H₂ mass tracking state
        self.h2_remaining_     = self.h2_initial_mass_
        self.tank_pressure_    = None  # Pa, from sensor (optional)
        self.tank_temperature_ = None  # K, from sensor (optional)

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

        if self.use_thrust_dr_:
            self.thrust_sub_ = self.create_subscription(
                Float64, self.thrust_topic_, self.thrust_callback, 10
            )

        if self.track_h2_mass_:
            tank_p_topic = self.get_parameter("tank_pressure_topic").value
            tank_t_topic = self.get_parameter("tank_temperature_topic").value
            self.tank_p_sub_ = self.create_subscription(
                Float64, tank_p_topic, self.tank_pressure_callback, 10
            )
            self.tank_t_sub_ = self.create_subscription(
                Float64, tank_t_topic, self.tank_temperature_callback, 10
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
        """Recompute z_fused_ with altitude-gated source priority.

        Priority order:
          1. Sonar (AGL ≤ max_sonar_alt_) — direct AGL measurement, most accurate at low altitude.
             z_fused = w_sonar * z_sonar + (1 - w_sonar) * z_imu
          2. Barometer — absolute pressure altitude, dominates above sonar range.
             z_fused = (1 - w_baro) * z_imu + w_baro * z_baro
          3. FAST-LIO / IMU z alone — if neither sonar nor baro available.
        """
        sonar_valid = (
            self.use_sonar_ and self.sonar_ready_
            and self.sonar_range_ is not None
            and self.sonar_range_ <= self.max_sonar_alt_
        )
        if sonar_valid:
            # Sonar dominates at low altitude — direct AGL measurement outperforms
            # both baro (1-2 m drift) and LiDAR scan-match Z over flat ground.
            self.z_fused_ = (self.sonar_weight_ * self.sonar_range_
                             + (1.0 - self.sonar_weight_) * self.z_imu_)
        elif self.use_barometer_ and self.baro_ready_:
            self.z_fused_ = ((1.0 - self.baro_weight_) * self.z_imu_
                             + self.baro_weight_ * self.baro_z_)
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
        self.fastlio_vel_initialized_ = False
        self.camera_init_to_odom_R_ = None
        self.last_fastlio_stamp_ = None

        self.wheel_odom_ready_ = False
        self.wheel_x_    = 0.0
        self.wheel_y_    = 0.0
        self.wheel_vfwd_ = 0.0

        self.last_airspeed_stamp_ = None
        self.last_of_stamp_       = None
        self.last_fwd_cam_stamp_  = None

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
                vx_new = (x_odom - self.last_fastlio_x_) / dt_fl
                vy_new = (y_odom - self.last_fastlio_y_) / dt_fl
                vz_fl  = (z_odom - self.last_fastlio_z_) / dt_fl
                # EMA smoothing: v_k = α·v_new + (1-α)·v_{k-1}
                # First valid measurement initialises directly to avoid cold-start bias.
                if not self.fastlio_vel_initialized_:
                    self.vx_ = vx_new
                    self.vy_ = vy_new
                    self.fastlio_vel_initialized_ = True
                else:
                    alpha = self.fastlio_vel_alpha_
                    self.vx_ = alpha * vx_new + (1.0 - alpha) * self.vx_
                    self.vy_ = alpha * vy_new + (1.0 - alpha) * self.vy_
                # Blend FAST-LIO Z velocity with IMU Z velocity (IMU is better for short intervals)
                self.vz_imu_ = 0.5 * vz_fl + 0.5 * self.vz_imu_
                # RLS drag calibration + thrust DR re-seed using FAST-LIO as the velocity reference.
                # FAST-LIO is the best truth available in this GPS-denied node — calibrate while
                # it's active, run dead reckoning when it goes stale.
                if self.use_thrust_dr_:
                    R = self.body_to_world_R()
                    vx_body = float((R.T @ np.array([self.vx_, self.vy_, self.vz_imu_]))[0])
                    now_sec = float(now.nanoseconds) * 1e-9
                    if (self.rls_vx_prev_ is not None and self.rls_time_prev_ is not None
                            and abs(vx_body) >= self.min_speed_drag_id_):
                        dt_rls = now_sec - self.rls_time_prev_
                        if 0.005 < dt_rls < self.fastlio_timeout_:
                            a_fl  = (vx_body - self.rls_vx_prev_) / dt_rls
                            pitch = math.asin(max(-1.0, min(1.0,
                                2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_))))
                            y   = (self.thrust_N_ - self.drone_mass_ * a_fl
                                   - self.drone_mass_ * self.g_ * math.sin(pitch))
                            phi = self.air_density_ * vx_body * abs(vx_body)
                            if abs(phi) > 1e-3:
                                lam = self.rls_lambda_
                                K = self.rls_P_ * phi / (lam + phi * self.rls_P_ * phi)
                                self.drag_coeff_ = max(0.0, min(10.0,
                                    self.drag_coeff_ + K * (y - phi * self.drag_coeff_)))
                                self.rls_P_ = max(1e-6, min(100.0,
                                    (self.rls_P_ - K * phi * self.rls_P_) / lam))
                    self.rls_vx_prev_   = vx_body
                    self.rls_time_prev_ = now_sec
                    self.thrust_dr_vx_  = vx_body

        self.last_fastlio_stamp_ = now
        self.last_fastlio_x_ = x_odom
        self.last_fastlio_y_ = y_odom
        self.last_fastlio_z_ = z_odom

        # Position: suppressed on the ground — wheel odometry owns XY there.
        # Above ground, FAST-LIO corrects position when scan matching is valid.
        if not (self.use_wheel_odom_ and self._on_ground()):
            if not self.fastlio_ready_:
                w = self.fastlio_pos_weight_
                self.x_ = w * x_odom + (1.0 - w) * self.x_
                self.y_ = w * y_odom + (1.0 - w) * self.y_
            else:
                self.x_ = x_odom
                self.y_ = y_odom

        # Anchor z_imu_ to FAST-LIO z; sonar and baro priority weighting
        # is handled centrally in _update_z_fused().
        self.z_imu_ = z_odom
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
            self.last_airspeed_stamp_ = self.get_clock().now()

    def of_callback(self, msg: TwistWithCovarianceStamped):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if math.isfinite(vx) and math.isfinite(vy):
            self.of_vx_body_ = vx
            self.of_vy_body_ = vy
            self.of_ready_   = True
            self.last_of_stamp_ = self.get_clock().now()

    def fwd_cam_callback(self, msg: TwistWithCovarianceStamped):
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        if math.isfinite(vy) and math.isfinite(vz):
            self.fwd_cam_vy_body_ = vy
            self.fwd_cam_vz_body_ = vz
            self.fwd_cam_ready_   = True
            self.last_fwd_cam_stamp_ = self.get_clock().now()

    def sonar_callback(self, msg: Range):
        r = float(msg.range)
        if math.isfinite(r) and msg.min_range <= r <= msg.max_range:
            self.sonar_range_ = r
            self.sonar_ready_ = True
            self.last_sonar_stamp_ = self.get_clock().now()
            self._update_z_fused()
        # Don't clear sonar_ready_ on a bad packet — use timeout in _on_ground() instead.
        # A single out-of-range reading caused intermittent drops to vx=0 on the ground.

    def _on_ground(self) -> bool:
        """True when a recent valid sonar reading confirms AGL <= ground_threshold."""
        if not self.sonar_ready_ or self.sonar_range_ is None or self.last_sonar_stamp_ is None:
            return False
        if (self.get_clock().now() - self.last_sonar_stamp_).nanoseconds / 1e9 > self.sonar_timeout_:
            return False
        return self.sonar_range_ <= self.ground_threshold_

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

    def thrust_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val >= 0.0:
            self.thrust_N_ = val
            self.last_thrust_stamp_ = self.get_clock().now()

    def tank_pressure_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val > 0.0:
            self.tank_pressure_ = val

    def tank_temperature_callback(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val) and val > 0.0:
            self.tank_temperature_ = val

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

        dt_pub = 0.0
        if self.prev_publish_time_ is not None:
            _dt = (now - self.prev_publish_time_).nanoseconds / 1e9
            if 0.0 < _dt < 0.5:
                dt_pub = _dt

        fastlio_stale = (
            not self.fastlio_ready_
            or self.last_fastlio_stamp_ is None
            or (now - self.last_fastlio_stamp_).nanoseconds / 1e9 > self.fastlio_timeout_
            or self.z_fused_ > self.max_fastlio_alt_
        )
        airspeed_stale = (
            not self.airspeed_ready_
            or self.last_airspeed_stamp_ is None
            or (now - self.last_airspeed_stamp_).nanoseconds / 1e9 > self.airspeed_timeout_
        )
        of_stale = (
            not self.of_ready_
            or self.last_of_stamp_ is None
            or (now - self.last_of_stamp_).nanoseconds / 1e9 > self.of_timeout_
        )
        fwd_cam_stale = (
            not self.fwd_cam_ready_
            or self.last_fwd_cam_stamp_ is None
            or (now - self.last_fwd_cam_stamp_).nanoseconds / 1e9 > self.fwd_cam_timeout_
        )

        # If thrust topic has gone silent, treat as glide: thrust_N_ = 0 so drag and
        # gravity still decelerate the estimate naturally via the physics equation.
        if (self.use_thrust_dr_ and self.last_thrust_stamp_ is not None
                and (now - self.last_thrust_stamp_).nanoseconds / 1e9 > self.thrust_timeout_):
            self.thrust_N_ = 0.0

        # H₂ mass tracking: consume fuel proportional to thrust, then correct with
        # tank sensors when available. Updates drone_mass_ for thrust DR accuracy.
        if self.track_h2_mass_ and dt_pub > 0.0:
            # Integrate SFC-based consumption
            dm = self.sfc_ * self.thrust_N_ * dt_pub
            self.h2_remaining_ = max(0.0, self.h2_remaining_ - dm)

            # Ideal gas law correction from tank sensors (slow blend to avoid noise).
            # p·V = m·R_H2·T  →  m = p·V / (R_H2·T)
            if self.tank_pressure_ is not None and self.tank_temperature_ is not None:
                m_sensor = (self.tank_pressure_ * self.tank_volume_) / (self.R_H2_ * self.tank_temperature_)
                m_sensor = max(0.0, min(self.h2_initial_mass_, m_sensor))
                w = self.tank_sensor_w_
                self.h2_remaining_ = (1.0 - w) * self.h2_remaining_ + w * m_sensor

            self.drone_mass_ = self.drone_empty_mass_ + self.h2_remaining_

        # On liftoff, seed thrust_dr_vx_ from airspeed so there's no drop-to-zero
        # when the velocity source switches from wheel odometry to thrust DR.
        # liftoff_seeded_ is a one-shot flag: prevents re-seeding every time sonar
        # noise causes _on_ground() to flicker True→False at the threshold.
        # Guard with use_sonar_: without sonar, _on_ground() is always False so
        # prev_on_ground_ (init=True) would fire a spurious seed on the first tick.
        currently_on_ground = self._on_ground()

        # Debounce for vx source selection only — prevents a single noisy sonar reading
        # below ground_threshold_ from snapping vx_pub to wheel_vfwd_ (typically 0).
        if currently_on_ground:
            self.on_ground_ticks_ = min(self.on_ground_ticks_ + 1, self.on_ground_debounce_ticks_ + 1)
        else:
            self.on_ground_ticks_ = 0
        on_ground_for_vx = (self.on_ground_ticks_ >= self.on_ground_debounce_ticks_)

        if (self.use_sonar_ and self.prev_on_ground_ and not currently_on_ground
                and self.use_thrust_dr_ and not self.liftoff_seeded_):
            seed = self.last_airspeed_ if (not airspeed_stale and self.last_airspeed_ >= self.min_airspeed_) \
                   else (self.wheel_vfwd_ if self.wheel_odom_ready_ else self.thrust_dr_vx_)
            self.thrust_dr_vx_ = seed
            self.liftoff_seeded_ = True
        # Reset seed flag only when drone is clearly on the ground (well below threshold),
        # not just at the noisy boundary.
        if self.sonar_range_ is not None and self.sonar_range_ < self.ground_threshold_ * 0.5:
            self.liftoff_seeded_ = False
        self.prev_on_ground_ = currently_on_ground

        # Integrate thrust DR body-frame forward velocity at publish rate.
        if (self.use_thrust_dr_ and self.thrust_dr_ready_
                and not self._on_ground() and dt_pub > 0.0):
            pitch = math.asin(max(-1.0, min(1.0,
                2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_))))
            drag = self.drag_coeff_ * self.air_density_ * self.thrust_dr_vx_ * abs(self.thrust_dr_vx_)
            a_x  = (self.thrust_N_ - drag - self.drone_mass_ * self.g_ * math.sin(pitch)) / self.drone_mass_
            self.thrust_dr_vx_ = max(-self.max_velocity_,
                                      min(self.max_velocity_, self.thrust_dr_vx_ + a_x * dt_pub))

        # Dead-reckon XY position between sensor corrections.
        # Priority: optical flow (FAST-LIO active, low altitude) > thrust DR (FAST-LIO stale, airborne).
        # FAST-LIO hard-resets x_/y_ each scan, capping OF drift to one scan interval (~100 ms).
        if dt_pub > 0.0 and not (self.use_wheel_odom_ and self._on_ground()):
            if (self.fastlio_ready_
                    and self.use_optical_flow_ and not of_stale
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_):
                R = self.body_to_world_R()
                v_body = np.array([self.of_vx_body_, self.of_vy_body_, 0.0])
                v_world = R @ v_body
                self.x_ += v_world[0] * dt_pub
                self.y_ += v_world[1] * dt_pub
            elif (self.use_thrust_dr_ and self.thrust_dr_ready_ and fastlio_stale):
                R = self.body_to_world_R()
                v_world = R @ np.array([self.thrust_dr_vx_, 0.0, 0.0])
                self.x_ += v_world[0] * dt_pub
                self.y_ += v_world[1] * dt_pub
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

        # -----------------------------------------------------------------------
        # Velocity estimation
        # -----------------------------------------------------------------------
        # vx — two phases, no elif between them:
        #   Ground : airspeed (primary) → wheel_vfwd (fallback)
        #   Air    : airspeed always; thrust DR when airspeed unavailable
        #
        # vy — layered corrections applied on top of vx phase result:
        #   FAST-LIO lateral finite-diff (active, below max_fastlio_altitude)
        #   → OF vy post-blend (low altitude)
        #   → forward camera vy post-blend
        #   → hold/decay
        #
        # FAST-LIO is used only for position correction (fastlio_callback) and
        # vy; it is NOT a vx source. Optical flow is NOT a vx source — with
        # positive pitch the downward camera cannot cleanly isolate forward motion.
        # -----------------------------------------------------------------------
        decay = self.velocity_decay_per_sec_ ** dt_pub if dt_pub > 0.0 else 1.0
        vx_pub = self.last_vx_pub_ * decay
        vy_pub = self.last_vy_pub_ * decay

        # When sonar is disabled fall back to z_fused_ as ground proxy (vx only).
        ground_for_vx = on_ground_for_vx or (
            not self.use_sonar_ and self.z_fused_ < self.ground_threshold_
        )

        R = self.body_to_world_R()

        if ground_for_vx:
            # Ground: wheel odometry primary (reliable down to 0 m/s),
            # airspeed as fallback (pitot unreliable below ~5 m/s).
            if self.use_wheel_odom_ and self.wheel_odom_ready_:
                v_fwd = self.wheel_vfwd_
            elif self.use_airspeed_ and not airspeed_stale and self.last_airspeed_ >= self.min_airspeed_:
                v_fwd = self.last_airspeed_
            else:
                v_fwd = 0.0
            v_world = R @ np.array([v_fwd, 0.0, 0.0])
            vx_pub = v_world[0]
            vy_pub = v_world[1]

        else:
            # Air vx priority:
            #   1. Airspeed — direct pitot measurement, pitch-independent (preferred).
            #      Note: airspeed = ground speed only in zero wind. Wind correction is
            #      disabled in simulation; enable use_wind_correction for real flight.
            #   2. FAST-LIO finite-diff — position-derived vx, EMA-smoothed.
            #   3. Thrust DR — physics model fallback when both above are unavailable.
            if self.use_airspeed_ and not airspeed_stale and self.last_airspeed_ >= self.min_airspeed_:
                v_world = R @ np.array([self.last_airspeed_, 0.0, 0.0])
                vx_pub = self.airspeed_weight_ * v_world[0] + (1.0 - self.airspeed_weight_) * vx_pub

            elif not fastlio_stale:
                # Airspeed absent or stale: FAST-LIO position-derived vx as fallback.
                vx_pub = self.vx_

            elif (self.use_thrust_dr_ and self.thrust_dr_ready_ and self.fastlio_ready_):
                # No airspeed, no active LiDAR: physics dead reckoning as last resort.
                v_world = R @ np.array([self.thrust_dr_vx_, 0.0, 0.0])
                vx_pub = self.thrust_dr_weight_ * v_world[0] + (1.0 - self.thrust_dr_weight_) * vx_pub
                vy_pub = self.thrust_dr_weight_ * v_world[1] + (1.0 - self.thrust_dr_weight_) * vy_pub

            # OF vx correction — pitch-gated blend on top of the primary vx source.
            # Downward camera is NOT a standalone vx source: at non-zero pitch the camera
            # normal tilts away from vertical, mixing pitch rotation into the flow field
            # and corrupting the forward velocity estimate. The pitch gate (max_of_pitch_deg)
            # limits application to near-level flight where contamination is negligible.
            pitch = math.asin(max(-1.0, min(1.0,
                2.0 * (self.qw_ * self.qy_ - self.qz_ * self.qx_))))
            if (self.use_optical_flow_ and not of_stale
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_
                    and abs(pitch) <= self.max_of_pitch_rad_):
                vx_of = (R @ np.array([self.of_vx_body_, self.of_vy_body_, 0.0]))[0]
                vx_pub = self.of_weight_ * vx_of + (1.0 - self.of_weight_) * vx_pub

            # vy — layered corrections applied in priority order.
            # 1. FAST-LIO lateral velocity (EMA-smoothed finite-diff) when scan-match active.
            if not fastlio_stale:
                vy_pub = self.vy_

            # 2. OF vy blend at low altitude.  vy is less sensitive to pitch contamination
            #    than vx (lateral flow is orthogonal to the pitch rotation axis), so no
            #    pitch gate is required here.
            if (self.use_optical_flow_ and not of_stale
                    and self.min_of_altitude_ <= self.z_fused_ <= self.max_of_altitude_):
                vy_of = (R @ np.array([self.of_vx_body_, self.of_vy_body_, 0.0]))[1]
                vy_pub = self.of_weight_ * vy_of + (1.0 - self.of_weight_) * vy_pub

            # 3. Forward camera vy above min_fwd_cam_altitude_.
            if (self.use_forward_camera_ and not fwd_cam_stale
                    and self.z_fused_ >= self.min_fwd_cam_altitude_):
                vy_world_fwd = (R @ np.array([0.0, self.fwd_cam_vy_body_, 0.0]))[1]
                vy_pub = self.fwd_cam_weight_ * vy_world_fwd + (1.0 - self.fwd_cam_weight_) * vy_pub

        self.last_vx_pub_ = vx_pub
        self.last_vy_pub_ = vy_pub

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
        now_diag = self.get_clock().now()
        def _age(stamp):
            if stamp is None:
                return float('inf')
            return (now_diag - stamp).nanoseconds / 1e9

        status.values.append(KeyValue(key="Airspeed ready", value=str(self.airspeed_ready_)))
        status.values.append(KeyValue(key="Airspeed (m/s)", value=f"{self.last_airspeed_:.2f}"))
        status.values.append(KeyValue(key="Airspeed age (s)", value=f"{_age(self.last_airspeed_stamp_):.2f}"))
        status.values.append(KeyValue(key="Wind X", value=f"{self.wind_x_:.2f}"))
        status.values.append(KeyValue(key="Wind Y", value=f"{self.wind_y_:.2f}"))
        status.values.append(KeyValue(key="OF ready", value=str(self.of_ready_)))
        status.values.append(KeyValue(key="OF age (s)", value=f"{_age(self.last_of_stamp_):.2f}"))
        status.values.append(KeyValue(key="OF vx_body", value=f"{self.of_vx_body_:.3f}"))
        status.values.append(KeyValue(key="OF vy_body", value=f"{self.of_vy_body_:.3f}"))
        sonar_str = f"{self.sonar_range_:.3f}" if self.sonar_range_ is not None else "N/A"
        status.values.append(KeyValue(key="Sonar AGL (m)", value=sonar_str))
        status.values.append(KeyValue(key="On ground", value=str(self._on_ground())))
        status.values.append(KeyValue(key="Wheel odom ready", value=str(self.wheel_odom_ready_)))
        status.values.append(KeyValue(key="Wheel vfwd (m/s)", value=f"{self.wheel_vfwd_:.3f}"))
        status.values.append(KeyValue(key="Fwd cam ready", value=str(self.fwd_cam_ready_)))
        status.values.append(KeyValue(key="Fwd cam age (s)", value=f"{_age(self.last_fwd_cam_stamp_):.2f}"))
        status.values.append(KeyValue(key="Fwd cam vy_body", value=f"{self.fwd_cam_vy_body_:.3f}"))
        status.values.append(KeyValue(key="Fwd cam vz_body", value=f"{self.fwd_cam_vz_body_:.3f}"))
        status.values.append(KeyValue(key="Thrust DR ready", value=str(self.thrust_dr_ready_)))
        status.values.append(KeyValue(key="Thrust (N)", value=f"{self.thrust_N_:.1f}"))
        status.values.append(KeyValue(key="Thrust DR vx_body", value=f"{self.thrust_dr_vx_:.3f}"))
        status.values.append(KeyValue(key="Drag coeff k_d", value=f"{self.drag_coeff_:.4f}"))
        status.values.append(KeyValue(key="RLS covariance P", value=f"{self.rls_P_:.6f}"))
        status.values.append(KeyValue(key="H2 remaining (kg)", value=f"{self.h2_remaining_:.3f}"))
        status.values.append(KeyValue(key="Drone mass (kg)", value=f"{self.drone_mass_:.2f}"))
        tank_p_str = f"{self.tank_pressure_:.1f}" if self.tank_pressure_ is not None else "N/A"
        tank_t_str = f"{self.tank_temperature_:.2f}" if self.tank_temperature_ is not None else "N/A"
        status.values.append(KeyValue(key="Tank pressure (Pa)", value=tank_p_str))
        status.values.append(KeyValue(key="Tank temperature (K)", value=tank_t_str))

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
