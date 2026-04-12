#!/usr/bin/env python3
"""
Forward Camera Velocity Estimator

Estimates body-frame lateral (vy) and vertical (vz) velocity from the
forward-facing camera using gyro-compensated Lucas-Kanade sparse optical flow.

Why gyro compensation?
  A rotating camera produces apparent pixel motion even with zero translation.
  Subtracting the rotation-induced flow isolates the translational component,
  which is the only part that carries velocity information.

Depth estimate:
  For a forward-looking camera, feature depth Z ≈ depth_scale × altitude.
  depth_scale is tunable (default 3.0). In urban/feature-rich scenes lower it;
  over open terrain raise it.

Velocity formulas (after gyro compensation):
  vy_body = depth_scale × alt × mean_horizontal_flow_px_per_s / focal_px
  vz_body = depth_scale × alt × mean_vertical_flow_px_per_s   / focal_px

Forward velocity (vx) is NOT estimated — feature depth is unknown and airspeed
already provides a reliable vx measurement.

Body→Camera axis mapping (camera rpy=0,0,0, aligned with base_link_front):
  Body:   +X=forward, +Y=left, +Z=up
  Camera: +Z=forward, +X=right, +Y=down  (standard pinhole convention)
  Therefore: ωc_x = -ωb_y,  ωc_y = -ωb_z,  ωc_z = ωb_x
"""

import math
import numpy as np
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Range, Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped


class ForwardFlowEstimator(Node):
    def __init__(self):
        super().__init__('forward_flow_estimator')

        # --- Parameters ---
        self.declare_parameter('camera_topic',   '/camera/image_raw')
        self.declare_parameter('output_topic',   '/forward_camera/velocity')
        self.declare_parameter('imu_topic',      '/imu_ekf')
        self.declare_parameter('sonar_topic',    '/sonar/range')
        self.declare_parameter('baro_topic',     '/barometer/altitude')

        self.declare_parameter('image_width',    640)
        self.declare_parameter('horizontal_fov', 1.047)   # 60 deg

        # Only active above this AGL — below it the downward camera is preferred
        self.declare_parameter('min_altitude',   1.0)     # m

        # Mean feature depth = depth_scale × altitude. Tune per environment.
        self.declare_parameter('depth_scale',    3.0)

        # LK tracking
        self.declare_parameter('max_features',   150)
        self.declare_parameter('min_features',   40)      # redetect below this
        self.declare_parameter('lk_winsize',     21)
        self.declare_parameter('lk_max_level',   3)

        # Translational flow magnitude gate (px/frame) — rejects spurious vectors
        self.declare_parameter('max_flow_magnitude', 50.0)

        # Sign corrections: flip if vy/vz direction is inverted in your setup
        self.declare_parameter('sign_vy', 1.0)
        self.declare_parameter('sign_vz', 1.0)

        self.declare_parameter('velocity_variance', 0.1)
        self.declare_parameter('frame_id', 'base_link_front')

        # --- Read parameters ---
        cam_topic   = self.get_parameter('camera_topic').value
        out_topic   = self.get_parameter('output_topic').value
        imu_topic   = self.get_parameter('imu_topic').value
        sonar_topic = self.get_parameter('sonar_topic').value
        baro_topic  = self.get_parameter('baro_topic').value

        img_w              = int(self.get_parameter('image_width').value)
        hfov               = float(self.get_parameter('horizontal_fov').value)
        self.focal_px_     = (img_w / 2.0) / math.tan(hfov / 2.0)
        self.min_alt_      = float(self.get_parameter('min_altitude').value)
        self.depth_scale_  = float(self.get_parameter('depth_scale').value)
        self.max_features_ = int(self.get_parameter('max_features').value)
        self.min_features_ = int(self.get_parameter('min_features').value)
        lk_win             = int(self.get_parameter('lk_winsize').value)
        lk_lvl             = int(self.get_parameter('lk_max_level').value)
        self.max_flow_     = float(self.get_parameter('max_flow_magnitude').value)
        self.sign_vy_      = float(self.get_parameter('sign_vy').value)
        self.sign_vz_      = float(self.get_parameter('sign_vz').value)
        self.vel_var_      = float(self.get_parameter('velocity_variance').value)
        self.frame_id_     = str(self.get_parameter('frame_id').value)

        self.lk_params_ = dict(
            winSize=(lk_win, lk_win),
            maxLevel=lk_lvl,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )
        self.gftt_params_ = dict(
            maxCorners=self.max_features_,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

        # --- State ---
        self.bridge_     = CvBridge()
        self.prev_gray_  = None
        self.prev_pts_   = None   # (N,1,2) float32, LK format
        self.prev_stamp_ = None

        # Body angular velocity from IMU (rad/s)
        self.wb_x_ = 0.0
        self.wb_y_ = 0.0
        self.wb_z_ = 0.0

        # Best altitude: sonar preferred when valid, baro as fallback
        self.sonar_alt_ = None
        self.baro_alt_  = None

        # --- Subscriptions ---
        self.create_subscription(Image,                     cam_topic,   self.image_cb,  5)
        self.create_subscription(Imu,                       imu_topic,   self.imu_cb,   10)
        self.create_subscription(Range,                     sonar_topic, self.sonar_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, baro_topic,  self.baro_cb,  10)

        # --- Publisher ---
        self.vel_pub_ = self.create_publisher(TwistWithCovarianceStamped, out_topic, 10)

        self.get_logger().info(
            f'ForwardFlowEstimator ready  '
            f'(focal={self.focal_px_:.1f}px, depth_scale={self.depth_scale_}, '
            f'min_alt={self.min_alt_}m)'
        )

    # ------------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        self.wb_x_ = msg.angular_velocity.x
        self.wb_y_ = msg.angular_velocity.y
        self.wb_z_ = msg.angular_velocity.z

    def sonar_cb(self, msg: Range):
        if math.isfinite(msg.range) and msg.min_range <= msg.range <= msg.max_range:
            self.sonar_alt_ = float(msg.range)
        else:
            self.sonar_alt_ = None

    def baro_cb(self, msg: PoseWithCovarianceStamped):
        z = msg.pose.pose.position.z
        if math.isfinite(z) and z >= 0.0:
            self.baro_alt_ = float(z)

    def _altitude(self):
        """Sonar AGL when valid, barometric altitude otherwise."""
        if self.sonar_alt_ is not None:
            return self.sonar_alt_
        return self.baro_alt_

    # ------------------------------------------------------------------
    def _rotational_flow(self, pts_c: np.ndarray, dt: float) -> np.ndarray:
        """
        Pixel flow per frame caused by body rotation (no translation).
        pts_c : (N,2) feature positions relative to principal point (u, v).
        Returns (N,2) flow in pixels.

        Standard rotational flow model (Longuet-Higgins):
          δu/dt = (u·v/f)·ωc_x − (f + u²/f)·ωc_y + v·ωc_z
          δv/dt = (f + v²/f)·ωc_x − (u·v/f)·ωc_y − u·ωc_z

        Body→Camera: ωc_x = −ωb_y,  ωc_y = −ωb_z,  ωc_z = ωb_x
        """
        f    = self.focal_px_
        wc_x = -self.wb_y_
        wc_y = -self.wb_z_
        wc_z =  self.wb_x_

        u = pts_c[:, 0]
        v = pts_c[:, 1]

        du = ((u * v / f) * wc_x - (f + u * u / f) * wc_y + v * wc_z) * dt
        dv = ((f + v * v / f) * wc_x - (u * v / f) * wc_y - u * wc_z) * dt

        return np.stack([du, dv], axis=1)

    # ------------------------------------------------------------------
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}', throttle_duration_sec=5.0)
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stamp = msg.header.stamp

        alt = self._altitude()
        if alt is None or alt < self.min_alt_:
            # Below minimum altitude — reset tracking so it starts fresh on ascent
            self.prev_gray_  = gray
            self.prev_pts_   = None
            self.prev_stamp_ = stamp
            return

        # (Re-)detect features when count is low
        if self.prev_pts_ is None or len(self.prev_pts_) < self.min_features_:
            pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.gftt_params_)
            self.prev_pts_   = pts          # None if no features found
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        if self.prev_gray_ is None:
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        # Frame interval
        t_now  = stamp.sec + stamp.nanosec * 1e-9
        t_prev = self.prev_stamp_.sec + self.prev_stamp_.nanosec * 1e-9
        dt = t_now - t_prev
        if dt <= 0.0 or dt > 0.5:
            self.prev_gray_  = gray
            self.prev_pts_   = None
            self.prev_stamp_ = stamp
            return

        # Lucas-Kanade tracking
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_, gray, self.prev_pts_, None, **self.lk_params_
        )
        if curr_pts is None:
            self.prev_gray_  = gray
            self.prev_pts_   = None
            self.prev_stamp_ = stamp
            return

        tracked = status.ravel() == 1
        if not np.any(tracked):
            self.prev_gray_  = gray
            self.prev_pts_   = None
            self.prev_stamp_ = stamp
            return

        prev_ok = self.prev_pts_[tracked].reshape(-1, 2)
        curr_ok = curr_pts[tracked].reshape(-1, 2)

        # Coordinates relative to principal point
        h, w = gray.shape
        cx, cy = w / 2.0, h / 2.0
        prev_c = prev_ok - np.array([cx, cy])

        # Raw pixel flow
        raw_flow = curr_ok - prev_ok          # (N,2) px/frame

        # Subtract rotational component → isolate translation
        rot_flow   = self._rotational_flow(prev_c, dt)
        trans_flow = raw_flow - rot_flow      # (N,2) px/frame

        # Outlier rejection
        mag   = np.linalg.norm(trans_flow, axis=1)
        valid = mag < self.max_flow_
        if not np.any(valid):
            self.prev_gray_  = gray
            self.prev_pts_   = curr_ok[valid].reshape(-1, 1, 2) if np.any(valid) else None
            self.prev_stamp_ = stamp
            return

        mean_flow_per_s = np.mean(trans_flow[valid], axis=0) / dt   # (u,v) px/s

        # Velocity formula:
        #   vy_body =  Z_est × mean_u_per_s / f   (rightward pixel shift = +vy)
        #   vz_body =  Z_est × mean_v_per_s / f   (downward pixel shift  = -vz, hence sign_vz)
        Z_est   = self.depth_scale_ * alt
        vy_body = self.sign_vy_ * Z_est * mean_flow_per_s[0] / self.focal_px_
        vz_body = self.sign_vz_ * Z_est * mean_flow_per_s[1] / self.focal_px_

        # Covariance grows with altitude (depth uncertainty) and flow spread
        flow_std = float(np.std(mag[valid]))
        variance = self.vel_var_ * (1.0 + alt * 0.2) * max(1.0, flow_std / 10.0)

        out = TwistWithCovarianceStamped()
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = self.frame_id_

        out.twist.twist.linear.x = 0.0       # vx not estimated
        out.twist.twist.linear.y = vy_body
        out.twist.twist.linear.z = vz_body

        c_vy = variance
        c_vz = variance * 2.0   # vertical slightly less reliable
        out.twist.covariance = [
            9999, 0,    0,    0,    0,    0,
            0,    c_vy, 0,    0,    0,    0,
            0,    0,    c_vz, 0,    0,    0,
            0,    0,    0,    9999, 0,    0,
            0,    0,    0,    0,    9999, 0,
            0,    0,    0,    0,    0,    9999,
        ]
        self.vel_pub_.publish(out)

        # Advance state — keep only inlier points for next frame
        self.prev_gray_  = gray
        self.prev_pts_   = curr_ok[valid].reshape(-1, 1, 2)
        self.prev_stamp_ = stamp


def main():
    rclpy.init()
    node = ForwardFlowEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
