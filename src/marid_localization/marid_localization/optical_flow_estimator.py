#!/usr/bin/env python3
"""
Optical Flow Velocity Estimator

Computes body-frame (base_link_front) XY velocity from:
  - Downward camera (/optical_flow/camera) — Farneback dense optical flow
  - Sonar rangefinder (/sonar/scan)        — altitude scaling + AGL publishing

Publishes:
  /optical_flow/velocity  (TwistWithCovarianceStamped, frame=base_link_front)
  /sonar/range            (Range)

Velocity formula:
  v_body = flow_mean_px_per_frame * altitude_m * fps / focal_length_px

Active only when sonar altitude is within [min_altitude, max_altitude].
"""
import math
import numpy as np
import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Range
from geometry_msgs.msg import TwistWithCovarianceStamped


class OpticalFlowEstimator(Node):
    def __init__(self):
        super().__init__('optical_flow_estimator')

        # --- Parameters ---
        self.declare_parameter('camera_topic', '/optical_flow/camera')
        self.declare_parameter('sonar_topic', '/sonar/scan')
        self.declare_parameter('output_topic', '/optical_flow/velocity')
        self.declare_parameter('sonar_range_topic', '/sonar/range')

        self.declare_parameter('image_width', 320)
        self.declare_parameter('image_height', 240)
        self.declare_parameter('horizontal_fov', 1.047)   # 60 deg

        # Body-frame sign conventions for pixel flow → velocity:
        #   camera +X = body -Z (down); camera looks at ground from below
        #   image row flow (flow_y > 0 = frame shifts down) → drone moving forward (+X body)
        #   image col flow (flow_x > 0 = frame shifts right) → drone moving left (-Y body)? Tune as needed.
        # sign_vx: sign applied to row-direction flow to get vx_body
        # sign_vy: sign applied to col-direction flow to get vy_body
        self.declare_parameter('sign_vx', 1.0)
        self.declare_parameter('sign_vy', 1.0)

        self.declare_parameter('min_altitude', 0.3)   # m — sonar valid range min
        self.declare_parameter('max_altitude', 5.0)    # m — OF inactive above this
        self.declare_parameter('frame_id', 'base_link_front')

        # Farneback OF parameters
        self.declare_parameter('fb_pyr_scale', 0.5)
        self.declare_parameter('fb_levels', 3)
        self.declare_parameter('fb_winsize', 15)
        self.declare_parameter('fb_iterations', 3)
        self.declare_parameter('fb_poly_n', 5)
        self.declare_parameter('fb_poly_sigma', 1.2)

        # Outlier rejection: discard flow vectors with magnitude > this (px/frame)
        self.declare_parameter('max_flow_magnitude', 300.0)

        # Base velocity covariance (m/s)^2 — scales with altitude and flow noise
        self.declare_parameter('velocity_variance', 0.05)

        # --- Read parameters ---
        camera_topic   = self.get_parameter('camera_topic').value
        sonar_topic    = self.get_parameter('sonar_topic').value
        output_topic   = self.get_parameter('output_topic').value
        sonar_r_topic  = self.get_parameter('sonar_range_topic').value

        img_w  = int(self.get_parameter('image_width').value)
        hfov   = float(self.get_parameter('horizontal_fov').value)
        self.focal_px_   = (img_w / 2.0) / math.tan(hfov / 2.0)
        self.sign_vx_    = float(self.get_parameter('sign_vx').value)
        self.sign_vy_    = float(self.get_parameter('sign_vy').value)
        self.min_alt_    = float(self.get_parameter('min_altitude').value)
        self.max_alt_    = float(self.get_parameter('max_altitude').value)
        self.frame_id_   = str(self.get_parameter('frame_id').value)
        self.max_flow_   = float(self.get_parameter('max_flow_magnitude').value)
        self.vel_var_    = float(self.get_parameter('velocity_variance').value)

        self.fb_pyr_scale_  = float(self.get_parameter('fb_pyr_scale').value)
        self.fb_levels_     = int(self.get_parameter('fb_levels').value)
        self.fb_winsize_    = int(self.get_parameter('fb_winsize').value)
        self.fb_iterations_ = int(self.get_parameter('fb_iterations').value)
        self.fb_poly_n_     = int(self.get_parameter('fb_poly_n').value)
        self.fb_poly_sigma_ = float(self.get_parameter('fb_poly_sigma').value)

        # --- State ---
        self.bridge_      = CvBridge()
        self.prev_gray_   = None
        self.prev_stamp_  = None
        self.altitude_    = None
        self.sonar_ready_ = False

        # --- Subscriptions ---
        self.create_subscription(Image,     camera_topic, self.image_callback, 5)
        self.create_subscription(LaserScan, sonar_topic,  self.sonar_callback, 10)

        # --- Publishers ---
        self.vel_pub_   = self.create_publisher(TwistWithCovarianceStamped, output_topic, 10)
        self.range_pub_ = self.create_publisher(Range, sonar_r_topic, 10)

        self.get_logger().info(
            f'OpticalFlowEstimator ready: focal={self.focal_px_:.1f}px, '
            f'OF range {self.min_alt_:.2f}–{self.max_alt_:.2f} m'
        )

    # ------------------------------------------------------------------
    # Sonar callback: convert LaserScan → Range and store altitude
    # ------------------------------------------------------------------
    def sonar_callback(self, msg: LaserScan):
        if not msg.ranges:
            return
        r = float(msg.ranges[0])
        if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
            # Out of range or inf/nan — invalid reading
            self.sonar_ready_ = False
            return

        self.altitude_    = r
        self.sonar_ready_ = True

        # Re-publish as sensor_msgs/Range for downstream consumers
        out = Range()
        out.header          = msg.header
        out.radiation_type  = Range.ULTRASOUND
        out.field_of_view   = 0.1    # ~6 deg beam cone
        out.min_range       = msg.range_min
        out.max_range       = msg.range_max
        out.range           = r
        self.range_pub_.publish(out)

    # ------------------------------------------------------------------
    # Image callback: compute Farneback flow → body velocity
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge_.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}', throttle_duration_sec=5.0)
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stamp = msg.header.stamp

        if self.prev_gray_ is None:
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        # Compute frame interval
        t_now  = stamp.sec + stamp.nanosec * 1e-9
        t_prev = self.prev_stamp_.sec + self.prev_stamp_.nanosec * 1e-9
        dt     = t_now - t_prev

        if dt <= 0.0 or dt > 0.5:
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        # Only process when sonar gives valid altitude inside OF range
        if not self.sonar_ready_ or self.altitude_ is None:
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        alt = self.altitude_
        if alt < self.min_alt_ or alt > self.max_alt_:
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        # --- Farneback dense optical flow ---
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray_, gray, None,
            self.fb_pyr_scale_, self.fb_levels_, self.fb_winsize_,
            self.fb_iterations_, self.fb_poly_n_, self.fb_poly_sigma_,
            0
        )
        # flow[...,0] = horizontal pixel displacement (column / X direction in image)
        # flow[...,1] = vertical   pixel displacement (row    / Y direction in image)

        flow_col = flow[..., 0]   # cols: corresponds to body Y
        flow_row = flow[..., 1]   # rows: corresponds to body X

        # Outlier rejection: discard high-magnitude vectors (e.g. from edges/noise)
        mag   = np.sqrt(flow_col**2 + flow_row**2)
        valid = mag < self.max_flow_

        if not np.any(valid):
            self.prev_gray_  = gray
            self.prev_stamp_ = stamp
            return

        mean_col = float(np.mean(flow_col[valid]))
        mean_row = float(np.mean(flow_row[valid]))

        # --- Pixel flow → body velocity ---
        # scale [m/s per px/frame] = altitude_m * fps / focal_px
        fps   = 1.0 / dt
        scale = alt * fps / self.focal_px_

        vx_body = self.sign_vx_ * mean_row * scale
        vy_body = self.sign_vy_ * mean_col * scale

        # Covariance: grows with altitude and pixel-flow standard deviation
        flow_std = float(np.std(mag[valid]))
        variance = self.vel_var_ * (1.0 + alt * 0.3) * max(1.0, flow_std / 20.0)

        # --- Publish velocity ---
        out = TwistWithCovarianceStamped()
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = self.frame_id_

        out.twist.twist.linear.x = vx_body
        out.twist.twist.linear.y = vy_body
        out.twist.twist.linear.z = 0.0   # Z not estimated from 2-D flow

        c = variance
        # Covariance is 6×6 row-major: (vx, vy, vz, wx, wy, wz)
        out.twist.covariance = [
            c,    0,    0,    0,    0,    0,
            0,    c,    0,    0,    0,    0,
            0,    0, 9999,    0,    0,    0,
            0,    0,    0, 9999,    0,    0,
            0,    0,    0,    0, 9999,    0,
            0,    0,    0,    0,    0, 9999,
        ]
        self.vel_pub_.publish(out)

        self.prev_gray_  = gray
        self.prev_stamp_ = stamp


def main():
    rclpy.init()
    node = OpticalFlowEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
