#!/usr/bin/env python3
"""
VX Error Monitor

Compares ground-truth body-frame forward velocity (/gazebo/odom) against the
node estimate (/marid/odom) and publishes percentage error and raw signals for
PlotJuggler visualisation.

Published topics:
  /debug/vx_gt          std_msgs/Float64  — ground truth vx (body frame)
  /debug/vx_est         std_msgs/Float64  — estimated vx (body frame)
  /debug/vx_error_pct   std_msgs/Float64  — |gt - est| / max(|gt|, min_speed) × 100
  /debug/vx_error_abs   std_msgs/Float64  — |gt - est| in m/s
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


class VxErrorMonitor(Node):
    def __init__(self):
        super().__init__('vx_error_monitor')

        self.declare_parameter('gt_topic',  '/gazebo/odom')
        self.declare_parameter('est_topic', '/marid/odom')
        self.declare_parameter('min_speed',  0.5)   # m/s — suppress % error below this

        self.gt_topic_  = self.get_parameter('gt_topic').value
        self.est_topic_ = self.get_parameter('est_topic').value
        self.min_speed_ = float(self.get_parameter('min_speed').value)

        self.vx_gt_  = None
        self.vx_est_ = None

        self.create_subscription(Odometry, self.gt_topic_,  self.gt_cb,  10)
        self.create_subscription(Odometry, self.est_topic_, self.est_cb, 10)

        self.pub_gt_      = self.create_publisher(Float64, '/debug/vx_gt',        10)
        self.pub_est_     = self.create_publisher(Float64, '/debug/vx_est',       10)
        self.pub_err_pct_ = self.create_publisher(Float64, '/debug/vx_error_pct', 10)
        self.pub_err_abs_ = self.create_publisher(Float64, '/debug/vx_error_abs', 10)

        self.get_logger().info(
            f'VxErrorMonitor ready  (gt={self.gt_topic_}, est={self.est_topic_}, '
            f'min_speed={self.min_speed_} m/s)'
        )

    def gt_cb(self, msg: Odometry):
        self.vx_gt_ = float(msg.twist.twist.linear.x)
        self._publish()

    def est_cb(self, msg: Odometry):
        self.vx_est_ = float(msg.twist.twist.linear.x)
        self._publish()

    def _publish(self):
        if self.vx_gt_ is None or self.vx_est_ is None:
            return
        if not (math.isfinite(self.vx_gt_) and math.isfinite(self.vx_est_)):
            return

        err_abs = abs(self.vx_gt_ - self.vx_est_)
        denom   = max(abs(self.vx_gt_), self.min_speed_)
        err_pct = (err_abs / denom) * 100.0

        self.pub_gt_.publish(Float64(data=self.vx_gt_))
        self.pub_est_.publish(Float64(data=self.vx_est_))
        self.pub_err_abs_.publish(Float64(data=err_abs))

        # Only publish % error when drone is moving fast enough to be meaningful
        if abs(self.vx_gt_) >= self.min_speed_:
            self.pub_err_pct_.publish(Float64(data=err_pct))


def main():
    rclpy.init()
    node = VxErrorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
