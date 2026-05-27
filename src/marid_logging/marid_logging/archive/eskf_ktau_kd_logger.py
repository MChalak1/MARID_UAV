#!/usr/bin/env python3
"""
MARID ESKF k_tau / kd Diagnostic Logger
========================================
Logs the ESKF internal propeller-bias (k_tau) and drag-coefficient (kd) states
alongside ground-truth yaw for offline diagnosis of thrust-proportional yaw drift.

Subscribes
----------
/marid/eskf_internal  (Float32MultiArray, 10 fields at 50 Hz from marid_odom_pub)
/gazebo/odom          (nav_msgs/Odometry)  — ground-truth pose for yaw reference

CSV columns (one row per /marid/eskf_internal message)
-------------------------------------------------------
  time_sec       — ROS2 header stamp in seconds (float64)
  k_tau_est      — running propeller-torque coefficient estimate  [rad/s/N]
  k_tau_corr     — k_tau_est × thrust_N = actual applied yaw correction  [rad/s]
  kd             — ESKF physics-mode drag coefficient  (0 if IMU mode)
  P_k_tau        — k_tau variance P[k_tau_idx, k_tau_idx]
  P_k_tau_yaw    — cross-covariance P[k_tau_idx, 8] (k_tau ↔ yaw)
  P_kd           — kd variance P[12, 12]  (0 if IMU mode)
  sun_innov_deg  — last sun heading innovation before ESKF update  [deg]  (NaN until first sun)
  thrust_N       — current thruster command  [N]
  is_airborne    — 1.0 in flight, 0.0 on ground
  eskf_yaw_deg   — ESKF horizontal heading  [deg]
  gt_yaw_deg     — Gazebo ground-truth horizontal heading  [deg]  (NaN until first GT message)
  yaw_err_deg    — eskf_yaw − gt_yaw (wrapped to ±180°)  [deg]

Typical usage
-------------
  ros2 run marid_logging eskf_ktau_kd_logger

Output file: ~/marid_ws/data_ktau/<flight_id>.csv   (one file per session)
"""

import fcntl
import math
import csv
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from pathlib import Path
from datetime import datetime
import time
from tf_transformations import euler_from_quaternion

_LOCK_PATH = Path('/tmp/eskf_ktau_kd_logger.lock')

# ── Field indices in the /marid/eskf_internal Float32MultiArray ──────────────
_I_K_TAU_EST   = 0
_I_K_TAU_CORR  = 1
_I_KD          = 2
_I_P_KTAU      = 3
_I_P_KTAU_YAW  = 4
_I_P_KD        = 5
_I_SUN_INNOV   = 6
_I_THRUST_N    = 7
_I_IS_AIRBORNE = 8
_I_ESKF_YAW    = 9
_N_FIELDS      = 10


class ESKFKtauKdLogger(Node):
    def __init__(self):
        super().__init__('eskf_ktau_kd_logger')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('internal_topic',     '/marid/eskf_internal')
        self.declare_parameter('ground_truth_topic', '/gazebo/odom')
        self.declare_parameter('log_directory',      '~/marid_ws/data_ktau')
        self.declare_parameter('flight_id',          '')
        self.declare_parameter('flush_every_n_rows', 50)  # flush to disk every N rows

        internal_topic = self.get_parameter('internal_topic').value
        gt_topic       = self.get_parameter('ground_truth_topic').value
        log_dir        = Path(self.get_parameter('log_directory').value).expanduser()
        flight_id      = self.get_parameter('flight_id').value
        self.flush_n_  = int(self.get_parameter('flush_every_n_rows').value)

        # Exclusive lock — only one instance logs at a time.
        # If both joystick_teleop_wings and option_a_controller are launched together,
        # whichever acquires the lock first wins; the other silently disables itself.
        self._lock_file = open(_LOCK_PATH, 'w')
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.enable_logging_ = True
        except IOError:
            self.get_logger().warn(
                'eskf_ktau_kd_logger already running — this instance will not log.'
            )
            self.enable_logging_ = False

        log_dir.mkdir(parents=True, exist_ok=True)
        self.flight_id_ = flight_id if flight_id else datetime.now().strftime('flight_%Y%m%d_%H%M%S')

        # ── State cache (always initialised — even disabled instances need valid attrs) ──
        self.last_gt_yaw_deg_ = float('nan')
        self.row_count_       = 0
        self.start_time_      = time.time()
        self.csv_file_        = None
        self.csv_writer_      = None

        if self.enable_logging_:
            csv_path = log_dir / f'{self.flight_id_}.csv'
            self.csv_file_   = open(csv_path, 'w', newline='', buffering=1)  # line-buffered
            self.csv_writer_ = csv.writer(self.csv_file_)
            self.csv_writer_.writerow([
                'time_sec',
                'k_tau_est',
                'k_tau_corr',
                'kd',
                'P_k_tau',
                'P_k_tau_yaw',
                'P_kd',
                'sun_innov_deg',
                'thrust_N',
                'is_airborne',
                'eskf_yaw_deg',
                'gt_yaw_deg',
                'yaw_err_deg',
            ])
            self.get_logger().info(
                f'ESKF k_tau/kd logger started — flight: {self.flight_id_}'
            )
            self.get_logger().info(f'  Output: {csv_path}')

        # ── Subscriptions (always subscribe — callbacks gate on enable_logging_) ──
        self.create_subscription(
            Float32MultiArray, internal_topic, self._internal_cb, 10
        )
        self.create_subscription(
            Odometry, gt_topic, self._gt_cb, 10
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _gt_cb(self, msg: Odometry):
        if not self.enable_logging_:
            return
        o = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([o.x, o.y, o.z, o.w])
        self.last_gt_yaw_deg_ = math.degrees(yaw)

    def _internal_cb(self, msg: Float32MultiArray):
        if not self.enable_logging_:
            return
        d = msg.data
        if len(d) < _N_FIELDS:
            self.get_logger().warn(
                f'eskf_internal has {len(d)} fields, expected {_N_FIELDS} — skipping'
            )
            return

        # Extract fields
        k_tau_est  = float(d[_I_K_TAU_EST])
        k_tau_corr = float(d[_I_K_TAU_CORR])
        kd         = float(d[_I_KD])
        P_k_tau    = float(d[_I_P_KTAU])
        P_k_tau_y  = float(d[_I_P_KTAU_YAW])
        P_kd       = float(d[_I_P_KD])
        sun_innov  = float(d[_I_SUN_INNOV])
        thrust_N   = float(d[_I_THRUST_N])
        is_air     = float(d[_I_IS_AIRBORNE])
        eskf_yaw   = float(d[_I_ESKF_YAW])

        # Derived
        sun_innov_deg = math.degrees(sun_innov) if math.isfinite(sun_innov) else float('nan')
        eskf_yaw_deg  = math.degrees(eskf_yaw)
        gt_yaw_deg    = self.last_gt_yaw_deg_
        if math.isfinite(gt_yaw_deg):
            err = eskf_yaw_deg - gt_yaw_deg
            # wrap to ±180
            yaw_err_deg = (err + 180.0) % 360.0 - 180.0
        else:
            yaw_err_deg = float('nan')

        # Timestamp from ROS clock
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        self.csv_writer_.writerow([
            f'{now_sec:.6f}',
            f'{k_tau_est:.6e}',
            f'{k_tau_corr:.6e}',
            f'{kd:.6f}',
            f'{P_k_tau:.6e}',
            f'{P_k_tau_y:.6e}',
            f'{P_kd:.6e}',
            f'{sun_innov_deg:.4f}' if math.isfinite(sun_innov_deg) else 'nan',
            f'{thrust_N:.2f}',
            f'{is_air:.1f}',
            f'{eskf_yaw_deg:.4f}',
            f'{gt_yaw_deg:.4f}' if math.isfinite(gt_yaw_deg) else 'nan',
            f'{yaw_err_deg:.4f}' if math.isfinite(yaw_err_deg) else 'nan',
        ])

        self.row_count_ += 1
        # Flush explicitly every flush_n_ rows (belt-and-suspenders over line buffering)
        if self.row_count_ % self.flush_n_ == 0:
            self.csv_file_.flush()

    # ── Shutdown ───────────────────────────────────────────────────────────

    def shutdown_callback(self):
        if self.enable_logging_ and self.csv_file_ is not None:
            self.csv_file_.flush()
            self.csv_file_.close()
            elapsed = time.time() - self.start_time_
            self.get_logger().info(
                f'Shutdown — {self.row_count_} rows logged in {elapsed:.1f}s '
                f'({self.row_count_ / max(elapsed, 1e-3):.1f} Hz avg)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ESKFKtauKdLogger()
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
