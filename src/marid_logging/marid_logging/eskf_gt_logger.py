#!/usr/bin/env python3
"""
MARID ESKF Ground-Truth Logger
Logs (ESKF estimate, Gazebo ground truth) pairs for training an ML correction model.

Input  (X, 12-D): ESKF state from /marid/odom
    [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
Target (y,  7-D): ground-truth pose+velocity from /gazebo/odom
    [x, y, roll, pitch, yaw, vx_world, vy_world]
    z/vz excluded (barometer-accurate). Gazebo body-frame twist (physics-engine velocity,
    not finite-differenced) is rotated to world (ENU) to match the ESKF convention.
Thrust (1-D): actual Gazebo center-thruster command in N from
    /model/marid/joint/thruster_center_joint/cmd_thrust (Float64).
    This captures joystick, Option A after marid_thrust_controller smoothing, and
    manual overrides because they all converge at the Gazebo thruster command.
Heading sidecars (1-D): horizontal-projected body-forward heading for ESKF and GT.
Time sidecars:
    t              (N,): seconds since this flight/logger started
    dt             (N,): seconds since previous logged sample; dt[0] = 0
    stamp_ros_sec  (N,): absolute ROS time at the logger tick
    eskf_stamp_sec (N,): source header stamp from the ESKF odometry message
    gt_stamp_sec   (N,): source header stamp from the Gazebo GT odometry message

Extended arrays (saved to data_extended only):
    imu_acc    (N, 3): [ax, ay, az] body-frame specific force from /imu
    yaw_madgwick (N,): raw Madgwick yaw from /imu_ekf (before ESKF fusion)
    airspeed   (N,):   body-frame forward airspeed from /airspeed/velocity
    sun_yaw    (N,):   sun-derived heading (rad), 0 when invalid
    sun_valid  (N,):   1 when sun elevation >= sun_el_min_deg, else 0

Training scripts choose between data/ (12-col base) and data_extended/ (full feature set)
via a DATA_FOLDER toggle at the top of each training script.

Each session gets a unique flight_id (startup timestamp) embedded in the filename.
"""
import fcntl
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float64, String
from pathlib import Path
from datetime import datetime
import time
from tf_transformations import euler_from_quaternion

_LOCK_PATH = Path('/tmp/eskf_gt_logger.lock')


class ESKFGTLogger(Node):
    def __init__(self):
        super().__init__('eskf_gt_logger')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('eskf_topic',         '/marid/odom')
        self.declare_parameter('ground_truth_topic', '/gazebo/odom')
        self.declare_parameter(
            'thrust_topic',
            '/model/marid/joint/thruster_center_joint/cmd_thrust',
        )
        self.declare_parameter('raw_imu_topic',      '/imu')
        self.declare_parameter('imu_ekf_topic',      '/imu_ekf')
        self.declare_parameter('airspeed_topic',     '/airspeed/velocity')
        self.declare_parameter('sun_vector_topic',   '/sun_sensor/sun_vector_body')
        self.declare_parameter('sun_azimuth_topic',  '/sun_sensor/sun_azimuth_enu_rad')
        self.declare_parameter('sun_elevation_topic','/sun_sensor/sun_elevation_deg')
        self.declare_parameter('sun_el_min_deg',     10.0)
        self.declare_parameter('use_velocity_lstm',  False)
        self.declare_parameter('log_directory',      '~/marid_ws/data_sync')
        self.declare_parameter('log_directory_vel_mod', '~/marid_ws/data_vel_mod')
        self.declare_parameter('log_filename_prefix','marid_eskf_gt')
        self.declare_parameter('samples_per_file',   10000)
        self.declare_parameter('enable_logging',     True)
        self.declare_parameter('flight_id',          '')

        if self.get_parameter('use_velocity_lstm').value:
            log_dir = Path(self.get_parameter('log_directory_vel_mod').value).expanduser()
        else:
            log_dir = Path(self.get_parameter('log_directory').value).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir_          = log_dir
        self.prefix_           = self.get_parameter('log_filename_prefix').value
        self.samples_per_file_ = self.get_parameter('samples_per_file').value
        self.enable_logging_   = self.get_parameter('enable_logging').value
        self.sun_el_min_deg_   = float(self.get_parameter('sun_el_min_deg').value)

        # Exclusive lock — only one instance logs at a time.
        self._lock_file = open(_LOCK_PATH, 'w')
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            self.get_logger().warn('eskf_gt_logger already running — this instance will not log.')
            self.enable_logging_ = False

        flight_id = self.get_parameter('flight_id').value
        self.flight_id_ = flight_id if flight_id else datetime.now().strftime('flight_%Y%m%d_%H%M%S')

        # ── Base sensor state ──────────────────────────────────────────────
        self.last_eskf_   = None
        self.last_gt_     = None
        self.last_thrust_ = 0.0

        # ── Extended sensor state ──────────────────────────────────────────
        self.last_imu_acc_      = np.zeros(3, np.float32)   # [ax, ay, az] body-frame
        self.last_yaw_madgwick_ = 0.0
        self.last_airspeed_     = 0.0
        self.last_sun_yaw_      = 0.0
        self.last_sun_valid_    = 0.0
        self.last_sun_azimuth_  = 0.0

        # ── Chunk buffers (base) ───────────────────────────────────────────
        self.eskf_inputs_    = []
        self.pose_targets_   = []
        self.thrust_buf_     = []
        self.eskf_heading_   = []
        self.target_heading_ = []

        # ── Chunk buffers (extended) ───────────────────────────────────────
        self.imu_acc_buf_      = []
        self.yaw_madgwick_buf_ = []
        self.airspeed_buf_     = []
        self.sun_yaw_buf_      = []
        self.sun_valid_buf_    = []
        self.t_buf_            = []
        self.dt_buf_           = []
        self.stamp_ros_buf_    = []
        self.eskf_stamp_buf_   = []
        self.gt_stamp_buf_     = []

        # ── Full-flight buffers (for mirror saves) ─────────────────────────
        self.all_eskf_           = []
        self.all_targets_        = []
        self.all_thrust_         = []
        self.all_eskf_heading_   = []
        self.all_target_heading_ = []
        self.all_imu_acc_        = []
        self.all_yaw_madgwick_   = []
        self.all_airspeed_       = []
        self.all_sun_yaw_        = []
        self.all_sun_valid_      = []
        self.all_t_              = []
        self.all_dt_             = []
        self.all_stamp_ros_      = []
        self.all_eskf_stamp_     = []
        self.all_gt_stamp_       = []

        self.file_counter_  = 0
        self.total_samples_ = 0
        self.start_time_    = time.time()
        self.first_ros_stamp_     = None
        self.last_ros_stamp_      = None
        self.first_gt_stamp_logged_ = None
        self.last_gt_stamp_logged_ = None   # GT stamp of previous logged sample, for dt

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(Odometry, self.get_parameter('eskf_topic').value,
                                 self._eskf_cb, 10)
        self.create_subscription(Odometry, self.get_parameter('ground_truth_topic').value,
                                 self._gt_cb, 10)
        self.create_subscription(Float64, self.get_parameter('thrust_topic').value,
                                 self._thrust_cb, 10)
        self.create_subscription(Imu, self.get_parameter('raw_imu_topic').value,
                                 self._raw_imu_cb, 10)
        self.create_subscription(Imu, self.get_parameter('imu_ekf_topic').value,
                                 self._imu_ekf_cb, 10)
        self.create_subscription(Float64, self.get_parameter('airspeed_topic').value,
                                 self._airspeed_cb, 10)
        self.create_subscription(Vector3Stamped, self.get_parameter('sun_vector_topic').value,
                                 self._sun_vector_cb, 10)
        self.create_subscription(Float64, self.get_parameter('sun_azimuth_topic').value,
                                 self._sun_azimuth_cb, 10)
        self.create_subscription(Float64, self.get_parameter('sun_elevation_topic').value,
                                 self._sun_elevation_cb, 10)

        self._status_pub_ = self.create_publisher(String, '/eskf_gt_logger/status', 10)

        self.get_logger().info('ESKF-GT Logger ready (extended mode, GT-callback driven)')
        self.get_logger().info(f'  Output: {self.log_dir_}  |  Flight: {self.flight_id_}')

    # ── Base callbacks ─────────────────────────────────────────────────────

    def _eskf_cb(self, msg: Odometry):
        self.last_eskf_ = msg

    def _gt_cb(self, msg: Odometry):
        self.last_gt_ = msg
        self._log()

    def _thrust_cb(self, msg: Float64):
        self.last_thrust_ = msg.data

    # ── Extended callbacks ─────────────────────────────────────────────────

    def _raw_imu_cb(self, msg: Imu):
        a = msg.linear_acceleration
        if math.isfinite(a.x) and math.isfinite(a.y) and math.isfinite(a.z):
            self.last_imu_acc_ = np.array([a.x, a.y, a.z], np.float32)

    def _imu_ekf_cb(self, msg: Imu):
        o = msg.orientation
        if not all(math.isfinite(v) for v in (o.x, o.y, o.z, o.w)):
            return
        _, _, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.last_yaw_madgwick_ = float(yaw)

    def _airspeed_cb(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val):
            self.last_airspeed_ = val

    def _sun_vector_cb(self, msg: Vector3Stamped):
        sx, sy, sz = msg.vector.x, msg.vector.y, msg.vector.z
        if not all(math.isfinite(v) for v in (sx, sy, sz)):
            return
        if sz <= 0.0:
            return
        if self.last_sun_valid_ < 0.5:
            return
        bh_x, bh_y = self._tilt_compensate(sx, sy, sz)
        sun_yaw = self.last_sun_azimuth_ - math.atan2(bh_y, bh_x)
        self.last_sun_yaw_ = float(math.atan2(math.sin(sun_yaw), math.cos(sun_yaw)))

    def _sun_azimuth_cb(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val):
            self.last_sun_azimuth_ = val

    def _sun_elevation_cb(self, msg: Float64):
        val = float(msg.data)
        if math.isfinite(val):
            self.last_sun_valid_ = 1.0 if val >= self.sun_el_min_deg_ else 0.0

    # ── Tilt compensation ──────────────────────────────────────────────────

    def _tilt_compensate(self, vx: float, vy: float, vz: float):
        """Project a body-frame vector onto the horizontal plane (same formula as marid_odom_pub)."""
        if self.last_eskf_ is None:
            return vx, vy
        o = self.last_eskf_.pose.pose.orientation
        roll, pitch, _ = euler_from_quaternion([o.x, o.y, o.z, o.w])
        cr, sr = math.cos(roll),  math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        bh_x = vx*cp + (vy*sr + vz*cr)*sp
        bh_y = vy*cr - vz*sr
        return bh_x, bh_y

    # ── Static extractors ──────────────────────────────────────────────────

    @staticmethod
    def _stamp_to_sec(msg: Odometry):
        stamp = msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    @staticmethod
    def _odom_to_state(msg: Odometry):
        """Extract [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r] from Odometry."""
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        t = msg.twist.twist
        roll, pitch, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        return np.array([p.x, p.y, p.z,
                         roll, pitch, yaw,
                         t.linear.x, t.linear.y, t.linear.z,
                         t.angular.x, t.angular.y, t.angular.z], dtype=np.float32)

    @staticmethod
    def _odom_to_pose(msg: Odometry, vel_world=None):
        """Extract [x, y, roll, pitch, yaw, vx_world, vy_world] from Odometry."""
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        t = msg.twist.twist
        roll, pitch, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        if vel_world is not None:
            vx_world = float(vel_world[0])
            vy_world = float(vel_world[1])
        else:
            cy, sy = math.cos(yaw), math.sin(yaw)
            vx_world = t.linear.x * cy - t.linear.y * sy
            vy_world = t.linear.x * sy + t.linear.y * cy
        return np.array([p.x, p.y,
                         roll, pitch, yaw,
                         vx_world, vy_world], dtype=np.float32)

    @staticmethod
    def _horizontal_heading_from_quat(qx, qy, qz, qw):
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        return math.atan2(fy, fx)

    @classmethod
    def _odom_to_horizontal_heading(cls, msg: Odometry):
        o = msg.pose.pose.orientation
        return np.float32(cls._horizontal_heading_from_quat(o.x, o.y, o.z, o.w))

    @staticmethod
    def _wrap_pi(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    # ── Log timer ─────────────────────────────────────────────────────────

    def _log(self):
        if not self.enable_logging_:
            return
        if self.last_eskf_ is None or self.last_gt_ is None:
            return

        eskf_state = self._odom_to_state(self.last_eskf_)
        gt_pose    = self._odom_to_pose(self.last_gt_)
        eskf_heading   = self._odom_to_horizontal_heading(self.last_eskf_)
        target_heading = self._odom_to_horizontal_heading(self.last_gt_)

        if np.any(np.isnan(eskf_state)) or np.any(np.isinf(eskf_state)):
            return
        if np.any(np.isnan(gt_pose)) or np.any(np.isinf(gt_pose)):
            return
        if not (np.isfinite(eskf_heading) and np.isfinite(target_heading)):
            return

        stamp_ros  = self.get_clock().now().nanoseconds * 1.0e-9
        eskf_stamp = self._stamp_to_sec(self.last_eskf_)
        gt_stamp   = self._stamp_to_sec(self.last_gt_)
        if self.first_ros_stamp_ is None:
            self.first_ros_stamp_ = stamp_ros
            self.first_gt_stamp_logged_ = gt_stamp
            dt_ros = 0.0
        else:
            dt_ros = max(0.0, gt_stamp - self.last_gt_stamp_logged_)
        self.last_ros_stamp_       = stamp_ros
        self.last_gt_stamp_logged_ = gt_stamp
        t_rel = gt_stamp - self.first_gt_stamp_logged_

        self.eskf_inputs_.append(eskf_state)
        self.pose_targets_.append(gt_pose)
        self.thrust_buf_.append(np.float32(self.last_thrust_))
        self.eskf_heading_.append(eskf_heading)
        self.target_heading_.append(target_heading)
        self.t_buf_.append(np.float64(t_rel))
        self.dt_buf_.append(np.float32(dt_ros))
        self.stamp_ros_buf_.append(np.float64(stamp_ros))
        self.eskf_stamp_buf_.append(np.float64(eskf_stamp))
        self.gt_stamp_buf_.append(np.float64(gt_stamp))

        self.imu_acc_buf_.append(self.last_imu_acc_.copy())
        self.yaw_madgwick_buf_.append(np.float32(self.last_yaw_madgwick_))
        self.airspeed_buf_.append(np.float32(self.last_airspeed_))
        self.sun_yaw_buf_.append(np.float32(self.last_sun_yaw_))
        self.sun_valid_buf_.append(np.float32(self.last_sun_valid_))

        self.all_eskf_.append(eskf_state)
        self.all_targets_.append(gt_pose)
        self.all_thrust_.append(np.float32(self.last_thrust_))
        self.all_eskf_heading_.append(eskf_heading)
        self.all_target_heading_.append(target_heading)
        self.all_t_.append(np.float64(t_rel))
        self.all_dt_.append(np.float32(dt_ros))
        self.all_stamp_ros_.append(np.float64(stamp_ros))
        self.all_eskf_stamp_.append(np.float64(eskf_stamp))
        self.all_gt_stamp_.append(np.float64(gt_stamp))
        self.all_imu_acc_.append(self.last_imu_acc_.copy())
        self.all_yaw_madgwick_.append(np.float32(self.last_yaw_madgwick_))
        self.all_airspeed_.append(np.float32(self.last_airspeed_))
        self.all_sun_yaw_.append(np.float32(self.last_sun_yaw_))
        self.all_sun_valid_.append(np.float32(self.last_sun_valid_))

        self.total_samples_ += 1
        if len(self.eskf_inputs_) >= self.samples_per_file_:
            self._save_chunk()

        # Publish status for HUD (~1 Hz)
        if self.total_samples_ % 50 == 0:
            chunk_samples = len(self.eskf_inputs_)
            msg = String()
            msg.data = f'Chunk {self.file_counter_}  [{chunk_samples:>5}/{self.samples_per_file_}]'
            self._status_pub_.publish(msg)

    # ── Save helpers ───────────────────────────────────────────────────────

    def _ext_arrays(self, imu_acc, yaw_madgwick, airspeed, sun_yaw, sun_valid):
        return dict(
            imu_acc       = imu_acc.astype(np.float32),
            yaw_madgwick  = yaw_madgwick.astype(np.float32),
            airspeed      = airspeed.astype(np.float32),
            sun_yaw       = sun_yaw.astype(np.float32),
            sun_valid     = sun_valid.astype(np.float32),
        )

    def _time_arrays(self, t, dt, stamp_ros, eskf_stamp, gt_stamp):
        return dict(
            t             = t.astype(np.float64),
            dt            = dt.astype(np.float32),
            stamp_ros_sec = stamp_ros.astype(np.float64),
            eskf_stamp_sec = eskf_stamp.astype(np.float64),
            gt_stamp_sec   = gt_stamp.astype(np.float64),
        )

    def _save_chunk(self):
        if not self.eskf_inputs_:
            return
        fname = f'{self.prefix_}_{self.flight_id_}_chunk{self.file_counter_:04d}.npz'
        np.savez_compressed(
            self.log_dir_ / fname,
            eskf_inputs  = np.array(self.eskf_inputs_,  dtype=np.float32),
            pose_targets = np.array(self.pose_targets_, dtype=np.float32),
            thrust       = np.array(self.thrust_buf_,   dtype=np.float32),
            eskf_horizontal_heading   = np.array(self.eskf_heading_,   dtype=np.float32),
            target_horizontal_heading = np.array(self.target_heading_, dtype=np.float32),
            **self._time_arrays(
                np.array(self.t_buf_, dtype=np.float64),
                np.array(self.dt_buf_, dtype=np.float32),
                np.array(self.stamp_ros_buf_, dtype=np.float64),
                np.array(self.eskf_stamp_buf_, dtype=np.float64),
                np.array(self.gt_stamp_buf_, dtype=np.float64),
            ),
            flight_id    = self.flight_id_,
            input_dim    = 12,
            target_dim   = 7,
            heading_dim  = 1,
            num_samples  = len(self.eskf_inputs_),
            **self._ext_arrays(
                np.array(self.imu_acc_buf_),
                np.array(self.yaw_madgwick_buf_),
                np.array(self.airspeed_buf_),
                np.array(self.sun_yaw_buf_),
                np.array(self.sun_valid_buf_),
            ),
        )
        elapsed = max(float(self.all_t_[-1]), 1.0e-6) if self.all_t_ else time.time() - self.start_time_
        self.get_logger().info(
            f'Saved {fname}  ({len(self.eskf_inputs_)} samples, '
            f'total: {self.total_samples_},  rate: {self.total_samples_/elapsed:.1f} Hz)'
        )
        self.eskf_inputs_    = []
        self.pose_targets_   = []
        self.thrust_buf_     = []
        self.eskf_heading_   = []
        self.target_heading_ = []
        self.t_buf_          = []
        self.dt_buf_         = []
        self.stamp_ros_buf_  = []
        self.eskf_stamp_buf_ = []
        self.gt_stamp_buf_   = []
        self.imu_acc_buf_    = []
        self.yaw_madgwick_buf_ = []
        self.airspeed_buf_   = []
        self.sun_yaw_buf_    = []
        self.sun_valid_buf_  = []
        self.file_counter_  += 1

    def _save_flight(self, fid, X, Y, T, Hx, Hy, t, dt, stamp_ros, eskf_stamp, gt_stamp,
                     imu_acc, yaw_madgwick, airspeed, sun_yaw, sun_valid):
        """Write one or more chunk files for a (possibly mirrored) flight array."""
        n   = len(X)
        spf = self.samples_per_file_
        for i, start in enumerate(range(0, n, spf)):
            sl = slice(start, start + spf)
            fname = f'{self.prefix_}_{fid}_chunk{i:04d}.npz'
            np.savez_compressed(
                self.log_dir_ / fname,
                eskf_inputs  = X[sl],
                pose_targets = Y[sl],
                thrust       = T[sl],
                eskf_horizontal_heading   = Hx[sl],
                target_horizontal_heading = Hy[sl],
                **self._time_arrays(t[sl], dt[sl], stamp_ros[sl], eskf_stamp[sl], gt_stamp[sl]),
                flight_id    = fid,
                input_dim    = 12,
                target_dim   = 7,
                heading_dim  = 1,
                num_samples  = len(X[sl]),
                **self._ext_arrays(
                    imu_acc[sl], yaw_madgwick[sl], airspeed[sl],
                    sun_yaw[sl], sun_valid[sl],
                ),
            )
        self.get_logger().info(
            f'Mirror saved: {fid}  ({n} samples, {math.ceil(n / spf)} chunk(s))'
        )

    def _base_arrays(self):
        X  = np.array(self.all_eskf_,          dtype=np.float32)
        Y  = np.array(self.all_targets_,        dtype=np.float32)
        T  = np.array(self.all_thrust_,         dtype=np.float32)
        Hx = np.array(self.all_eskf_heading_,   dtype=np.float32)
        Hy = np.array(self.all_target_heading_, dtype=np.float32)
        A  = np.array(self.all_imu_acc_,        dtype=np.float32)   # (N, 3)
        Ym = np.array(self.all_yaw_madgwick_,   dtype=np.float32)
        As = np.array(self.all_airspeed_,       dtype=np.float32)
        Syw= np.array(self.all_sun_yaw_,        dtype=np.float32)
        Svl= np.array(self.all_sun_valid_,      dtype=np.float32)
        t  = np.array(self.all_t_,              dtype=np.float64)
        dt = np.array(self.all_dt_,             dtype=np.float32)
        Rs = np.array(self.all_stamp_ros_,      dtype=np.float64)
        Es = np.array(self.all_eskf_stamp_,     dtype=np.float64)
        Gs = np.array(self.all_gt_stamp_,       dtype=np.float64)
        return X, Y, T, Hx, Hy, t, dt, Rs, Es, Gs, A, Ym, As, Syw, Svl

    # ── Mirror saves ───────────────────────────────────────────────────────

    def _save_mirror(self):
        """Left-right mirror: negate y(1), roll(3), yaw(5), vy(7), p(9), r(11)."""
        if not self.all_eskf_:
            return
        X, Y, T, Hx, Hy, t, dt, Rs, Es, Gs, A, Ym, As, Syw, Svl = self._base_arrays()

        Xm = X.copy(); Ym_ = Y.copy()
        for col in (1, 3, 5, 7, 9, 11):
            Xm[:, col] = -X[:, col]
        for col in (1, 2, 4, 6):
            Ym_[:, col] = -Y[:, col]
        Hxm = self._wrap_pi(-Hx).astype(np.float32)
        Hym = self._wrap_pi(-Hy).astype(np.float32)

        # Extended: ay_body flips (body-y axis reversal); yaw quantities negate
        Am   = A.copy();   Am[:, 1] = -A[:, 1]
        YmM  = self._wrap_pi(-Ym).astype(np.float32)
        AsM  = As.copy()   # airspeed: body-x symmetric
        SywM = self._wrap_pi(-Syw).astype(np.float32)

        fid = f'{self.flight_id_}_mirror'
        self._save_flight(fid, Xm, Ym_, T, Hxm, Hym, t, dt, Rs, Es, Gs,
                          Am, YmM, AsM, SywM, Svl.copy())

    def _save_fore_aft_mirror(self):
        """Fore-aft mirror: negate x(0), roll(3), vx(6), p(9), r(11); yaw → π−yaw."""
        if not self.all_eskf_:
            return
        X, Y, T, Hx, Hy, t, dt, Rs, Es, Gs, A, Ym, As, Syw, Svl = self._base_arrays()

        Xm = X.copy(); Ym_ = Y.copy()
        for col in (0, 3, 6, 9, 11):
            Xm[:, col] = -X[:, col]
        Xm[:, 5] = self._wrap_pi(np.pi - X[:, 5])
        for col in (0, 2, 5):
            Ym_[:, col] = -Y[:, col]
        Ym_[:, 4] = self._wrap_pi(np.pi - Y[:, 4])
        Hxm = self._wrap_pi(np.pi - Hx).astype(np.float32)
        Hym = self._wrap_pi(np.pi - Hy).astype(np.float32)

        # Extended: ax_body flips (x-axis reversal); ay_body flips (roll negation)
        Am   = A.copy(); Am[:, 0] = -A[:, 0]; Am[:, 1] = -A[:, 1]
        YmM  = self._wrap_pi(np.pi - Ym).astype(np.float32)
        AsM  = As.copy()   # airspeed: body-x forward remains positive
        SywM = self._wrap_pi(np.pi - Syw).astype(np.float32)

        fid = f'{self.flight_id_}_fore_aft_mirror'
        self._save_flight(fid, Xm, Ym_, T, Hxm, Hym, t, dt, Rs, Es, Gs,
                          Am, YmM, AsM, SywM, Svl.copy())

    def _save_combined_mirror(self):
        """Combined 180° mirror: negate x(0),y(1),vx(6),vy(7); yaw → yaw+π."""
        if not self.all_eskf_:
            return
        X, Y, T, Hx, Hy, t, dt, Rs, Es, Gs, A, Ym, As, Syw, Svl = self._base_arrays()

        Xm = X.copy(); Ym_ = Y.copy()
        for col in (0, 1, 6, 7):
            Xm[:, col] = -X[:, col]
        Xm[:, 5] = self._wrap_pi(X[:, 5] + np.pi)
        for col in (0, 1, 5, 6):
            Ym_[:, col] = -Y[:, col]
        Ym_[:, 4] = self._wrap_pi(Y[:, 4] + np.pi)
        Hxm = self._wrap_pi(Hx + np.pi).astype(np.float32)
        Hym = self._wrap_pi(Hy + np.pi).astype(np.float32)

        # Extended: lateral(-ay) + fore-aft(-ax,-ay) = ax negated, ay unchanged
        Am   = A.copy(); Am[:, 0] = -A[:, 0]
        YmM  = self._wrap_pi(Ym + np.pi).astype(np.float32)
        AsM  = As.copy()
        SywM = self._wrap_pi(Syw + np.pi).astype(np.float32)

        fid = f'{self.flight_id_}_combined_mirror'
        self._save_flight(fid, Xm, Ym_, T, Hxm, Hym, t, dt, Rs, Es, Gs,
                          Am, YmM, AsM, SywM, Svl.copy())

    # ── Shutdown ───────────────────────────────────────────────────────────

    def shutdown_callback(self):
        if self.eskf_inputs_:
            self.get_logger().info('Saving remaining data on shutdown...')
            self._save_chunk()
        if self.total_samples_ > 0:
            self._save_mirror()
            self._save_fore_aft_mirror()
            self._save_combined_mirror()
        elapsed = max(float(self.all_t_[-1]), 1.0e-6) if self.all_t_ else time.time() - self.start_time_
        self.get_logger().info(
            f'Shutdown. Total samples: {self.total_samples_}, elapsed: {elapsed:.1f}s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ESKFGTLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_callback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
