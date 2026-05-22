#!/usr/bin/env python3
import math

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu


class CameraHudOverlay(Node):
    def __init__(self):
        super().__init__("camera_hud_overlay")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("imu_topic", "/imu_ekf")
        self.declare_parameter("altitude_topic", "/barometer/altitude")
        self.declare_parameter("eskf_odom_topic", "/marid/odom")
        self.declare_parameter("gt_odom_topic", "/gazebo/odom")
        self.declare_parameter("output_topic", "/camera/image_hud")
        self.declare_parameter("smoothing_alpha", 0.15)

        self.image_topic_ = str(self.get_parameter("image_topic").value)
        self.imu_topic_ = str(self.get_parameter("imu_topic").value)
        self.altitude_topic_ = str(self.get_parameter("altitude_topic").value)
        self.eskf_odom_topic_ = str(self.get_parameter("eskf_odom_topic").value)
        self.gt_odom_topic_ = str(self.get_parameter("gt_odom_topic").value)
        self.output_topic_ = str(self.get_parameter("output_topic").value)
        self.alpha_ = float(self.get_parameter("smoothing_alpha").value)
        self.alpha_ = max(0.0, min(1.0, self.alpha_))

        self.bridge_ = CvBridge()
        self.roll_deg_ = 0.0
        self.pitch_deg_ = 0.0
        self.yaw_deg_ = 0.0
        self.altitude_m_ = 0.0
        self.have_attitude_ = False
        self.have_altitude_ = False

        self.g_force_ = 0.0

        self.roll_gt_deg_  = 0.0
        self.pitch_gt_deg_ = 0.0
        self.yaw_gt_deg_   = 0.0
        self.have_att_gt_  = False

        self.vx_est_ = 0.0
        self.vy_est_ = 0.0
        self.vx_gt_ = 0.0
        self.vy_gt_ = 0.0
        self.have_vel_est_ = False
        self.have_vel_gt_ = False

        self.image_sub_ = self.create_subscription(
            Image, self.image_topic_, self.image_callback, 10
        )
        self.imu_sub_ = self.create_subscription(
            Imu, self.imu_topic_, self.imu_callback, 20
        )

        self.odom_sub_ = self.create_subscription(
            Odometry,
            "/marid/odom",
            self.odom_callback,
            20
        )

        self.altitude_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            self.altitude_topic_,
            self.altitude_callback,
            10,
        )
        self.eskf_sub_ = self.create_subscription(
            Odometry, self.eskf_odom_topic_, self.eskf_callback, 10
        )
        self.gt_sub_ = self.create_subscription(
            Odometry, self.gt_odom_topic_, self.gt_callback, 10
        )
        self.image_pub_ = self.create_publisher(Image, self.output_topic_, 10)

        self.get_logger().info(
            f"Camera HUD overlay started: {self.image_topic_} + {self.imu_topic_} -> {self.output_topic_}"
        )

    @staticmethod
    def quaternion_to_euler_deg(x: float, y: float, z: float, w: float):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation

        roll_raw, pitch_raw, yaw_raw = self.quaternion_to_euler_deg(
            q.x, q.y, q.z, q.w
        )

        self.roll_deg_  = roll_raw
        self.pitch_deg_ = pitch_raw
        self.yaw_deg_   = yaw_raw

        self.have_attitude_ = True

    def imu_callback(self, msg: Imu):

        a = msg.linear_acceleration

        if (
            any(math.isnan(v) or math.isinf(v) for v in [a.x, a.y, a.z])
        ):
            return

        
        a_norm = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        self.g_force_ = a_norm / 9.81


    def altitude_callback(self, msg: PoseWithCovarianceStamped):
        z = msg.pose.pose.position.z
        if not (math.isnan(z) or math.isinf(z)):
            self.altitude_m_ = z
            self.have_altitude_ = True

    def eskf_callback(self, msg: Odometry):
        # /marid/odom twist is already world-frame (ESKF integrates in ENU)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        if not (math.isnan(vx) or math.isinf(vx) or math.isnan(vy) or math.isinf(vy)):
            self.vx_est_ = vx
            self.vy_est_ = vy
            self.have_vel_est_ = True

    def gt_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        roll_deg, pitch_deg, yaw_deg = self.quaternion_to_euler_deg(q.x, q.y, q.z, q.w)
        self.roll_gt_deg_  = roll_deg
        self.pitch_gt_deg_ = pitch_deg
        self.yaw_gt_deg_   = yaw_deg
        self.have_att_gt_  = True

        # /gazebo/odom twist is body-frame; rotate to world using GT yaw
        yaw = math.radians(yaw_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        self.vx_gt_ = vx_b * cy - vy_b * sy
        self.vy_gt_ = vx_b * sy + vy_b * cy
        self.have_vel_gt_ = True

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge_.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except CvBridgeError as err:
            self.get_logger().error(f"Could not convert image: {err}")
            return

        h, w = frame.shape[:2]

        x0 = 20
        y0 = 30
        green = (0, 255, 0)
        red = (255, 0, 0)      # OpenCV red
        shadow = (0, 0, 0)

        blue = (100, 149, 237)   # GT attitude colour
        col2_x = x0 + 220        # second column x (shared with velocity GT)

        text_lines = []
        gt_att_lines = []        # parallel list of GT strings for col2 ('' = no GT yet)

        if self.have_attitude_:
            text_lines = [
                (f"Roll:  {self.roll_deg_:6.1f} deg", green),
                (f"Pitch: {self.pitch_deg_:6.1f} deg", green),
                (f"Yaw:   {self.yaw_deg_:6.1f} deg", green),
            ]
            if self.have_att_gt_:
                gt_att_lines = [
                    f"gt {self.roll_gt_deg_:+6.1f}",
                    f"gt {self.pitch_gt_deg_:+6.1f}",
                    f"gt {self.yaw_gt_deg_:+6.1f}",
                ]
            else:
                gt_att_lines = ['', '', '']

            if self.have_altitude_:
                alt_text = f"Alt:   {self.altitude_m_:6.2f} m"
                alt_color = red if self.altitude_m_ <= 10.0 else green
                text_lines.append((alt_text, alt_color))
                gt_att_lines.append('')

            gf = self.g_force_
            gf_color = green if gf < 1.5 else (255, 165, 0) if gf < 2.5 else red
            text_lines.append((f"G-force: {gf:.2f} g", gf_color))
            gt_att_lines.append('')

        else:
            text_lines    = [("Waiting for IMU attitude...", green)]
            gt_att_lines  = ['']

        y = y0
        for (text, line_color), gt_str in zip(text_lines, gt_att_lines):
            cv2.putText(frame, text, (x0 + 1, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, shadow, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)
            if gt_str:
                cv2.putText(frame, gt_str, (col2_x + 1, y + 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, shadow, 3, cv2.LINE_AA)
                cv2.putText(frame, gt_str, (col2_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, 2, cv2.LINE_AA)
            y += 30

        if self.have_vel_est_:
            gt_color = (0, 220, 220)   # cyan for GT velocity
            vel_rows = [
                (f"Vx: {self.vx_est_:+6.2f} m/s", f"gt {self.vx_gt_:+6.2f}" if self.have_vel_gt_ else ""),
                (f"Vy: {self.vy_est_:+6.2f} m/s", f"gt {self.vy_gt_:+6.2f}" if self.have_vel_gt_ else ""),
            ]
            y_vel0 = y0 + len(text_lines) * 30
            for i, (est_str, gt_str) in enumerate(vel_rows):
                y = y_vel0 + i * 30
                cv2.putText(frame, est_str, (x0 + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow, 3, cv2.LINE_AA)
                cv2.putText(frame, est_str, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, green, 2, cv2.LINE_AA)
                if gt_str:
                    cv2.putText(frame, gt_str, (col2_x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow, 3, cv2.LINE_AA)
                    cv2.putText(frame, gt_str, (col2_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_color, 2, cv2.LINE_AA)

        cx = w // 2
        cy = h // 2
        horizon_half_width = min(320, w // 2)
        pitch_pixels_per_deg = 3.0
        y_horizon = int(cy - self.pitch_deg_ * pitch_pixels_per_deg) if self.have_attitude_ else cy
        roll_rad = math.radians(self.roll_deg_) if self.have_attitude_ else 0.0

        dx = int(horizon_half_width * math.cos(roll_rad))
        dy = int(horizon_half_width * math.sin(roll_rad))
        p1 = (cx - dx, y_horizon + dy)
        p2 = (cx + dx, y_horizon - dy)

        # Horizon line: dark outline then bold red line
        cv2.line(frame, p1, p2, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.line(frame, p1, p2, (255, 0, 0), 6, cv2.LINE_AA)

        # Pitch bars: short vertical bars left/right of center, aligned with roll
        vx = float(p2[0] - p1[0])
        vy = float(p2[1] - p1[1])
        norm_v = math.hypot(vx, vy)
        if norm_v > 1e-3:
            ux = vx / norm_v
            uy = vy / norm_v
            # Normal vector (perpendicular to horizon)
            nx = -uy
            ny = ux

            bar_offset = min(120.0, horizon_half_width * 0.75)
            bar_length = 70.0

            # Base points along the horizon (left and right of center)
            left_base = (cx - ux * bar_offset, y_horizon - uy * bar_offset)
            right_base = (cx + ux * bar_offset, y_horizon + uy * bar_offset)

            # Endpoints of bars (extend equally above/below horizon)
            half_len = bar_length * 0.5
            for base in (left_base, right_base):
                bx, by = base
                top = (int(bx - nx * half_len), int(by - ny * half_len))
                bottom = (int(bx + nx * half_len), int(by + ny * half_len))
                cv2.line(frame, top, bottom, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.drawMarker(frame, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 24, 4)
        cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        try:
            hud_msg = self.bridge_.cv2_to_imgmsg(frame, encoding="rgb8")
        except CvBridgeError as err:
            self.get_logger().error(f"Could not encode HUD image: {err}")
            return

        hud_msg.header = msg.header
        self.image_pub_.publish(hud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraHudOverlay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
