#!/usr/bin/env python3
import math

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, Imu


class CameraHudOverlay(Node):
    def __init__(self):
        super().__init__("camera_hud_overlay")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("imu_topic", "/imu_ekf")
        self.declare_parameter("altitude_topic", "/barometer/altitude")
        self.declare_parameter("output_topic", "/camera/image_hud")
        self.declare_parameter("smoothing_alpha", 0.15)

        self.image_topic_ = str(self.get_parameter("image_topic").value)
        self.imu_topic_ = str(self.get_parameter("imu_topic").value)
        self.altitude_topic_ = str(self.get_parameter("altitude_topic").value)
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

        self.image_sub_ = self.create_subscription(
            Image, self.image_topic_, self.image_callback, 10
        )
        self.imu_sub_ = self.create_subscription(
            Imu, self.imu_topic_, self.imu_callback, 20
        )
        self.altitude_sub_ = self.create_subscription(
            PoseWithCovarianceStamped,
            self.altitude_topic_,
            self.altitude_callback,
            10,
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

    def imu_callback(self, msg: Imu):
        q = msg.orientation
        if (
            any(math.isnan(v) or math.isinf(v) for v in [q.x, q.y, q.z, q.w])
            or abs(q.x) + abs(q.y) + abs(q.z) + abs(q.w) < 1e-9
        ):
            return

        roll_raw, pitch_raw, yaw_raw = self.quaternion_to_euler_deg(q.x, q.y, q.z, q.w)

        if not self.have_attitude_:
            self.roll_deg_ = roll_raw
            self.pitch_deg_ = pitch_raw
            self.yaw_deg_ = yaw_raw
            self.have_attitude_ = True
            return

        beta = 1.0 - self.alpha_
        self.roll_deg_ = beta * self.roll_deg_ + self.alpha_ * roll_raw
        self.pitch_deg_ = beta * self.pitch_deg_ + self.alpha_ * pitch_raw
        self.yaw_deg_ = beta * self.yaw_deg_ + self.alpha_ * yaw_raw

    def altitude_callback(self, msg: PoseWithCovarianceStamped):
        z = msg.pose.pose.position.z
        if not (math.isnan(z) or math.isinf(z)):
            self.altitude_m_ = z
            self.have_altitude_ = True

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge_.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except CvBridgeError as err:
            self.get_logger().error(f"Could not convert image: {err}")
            return

        h, w = frame.shape[:2]
        color = (0, 255, 0)
        shadow = (0, 0, 0)

        text_lines = []
        if self.have_attitude_:
            text_lines = [
                f"Roll:  {self.roll_deg_:6.1f} deg",
                f"Pitch: {self.pitch_deg_:6.1f} deg",
                f"Yaw:   {self.yaw_deg_:6.1f} deg",
            ]
            if self.have_altitude_:
                text_lines.append(f"Alt:   {self.altitude_m_:6.1f} m")
        else:
            text_lines = ["Waiting for IMU attitude..."]

        x0 = 18
        y0 = 34
        for i, line in enumerate(text_lines):
            y = y0 + i * 30
            cv2.putText(
                frame,
                line,
                (x0 + 1, y + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                shadow,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

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
