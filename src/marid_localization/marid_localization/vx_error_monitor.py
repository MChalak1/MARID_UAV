#!/usr/bin/env python3
"""
State Error Monitor

Compares ground-truth state (/gazebo/odom) against the node estimate
(/marid/odom) and publishes absolute and percentage errors for position
(x, y, z) and velocity (vx, vy, vz).

Published topics for each channel <ch> in {x, y, z, vx, vy, vz}:
  /debug/<ch>_gt          std_msgs/Float64  — ground truth
  /debug/<ch>_est         std_msgs/Float64  — estimate
  /debug/<ch>_error_abs   std_msgs/Float64  — |gt − est|
  /debug/<ch>_error_pct   std_msgs/Float64  — |gt − est| / max(|gt|, min_ref) × 100
                                              published only when |gt| ≥ min_ref

min_ref is min_pos for position channels and min_speed for velocity channels.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64


_CHANNELS = ('x', 'y', 'z', 'vx', 'vy', 'vz')


def _extract(msg: Odometry):
    """Return (x, y, z, vx, vy, vz) from an Odometry message."""
    p = msg.pose.pose.position
    v = msg.twist.twist.linear
    return (p.x, p.y, p.z, v.x, v.y, v.z)


class VxErrorMonitor(Node):
    def __init__(self):
        super().__init__('vx_error_monitor')

        self.declare_parameter('gt_topic',  '/gazebo/odom')
        self.declare_parameter('est_topic', '/marid/odom')
        self.declare_parameter('min_speed',  0.5)   # m/s — velocity % denominator floor
        self.declare_parameter('min_pos',    1.0)   # m   — position % denominator floor

        self.gt_topic_  = self.get_parameter('gt_topic').value
        self.est_topic_ = self.get_parameter('est_topic').value
        min_speed       = float(self.get_parameter('min_speed').value)
        min_pos         = float(self.get_parameter('min_pos').value)

        # min_ref indexed by channel: first 3 are position, last 3 are velocity
        self._min_ref = [min_pos, min_pos, min_pos,
                         min_speed, min_speed, min_speed]

        # Latest ground-truth and estimate values (None until first message)
        self._gt  = [None] * 6
        self._est = [None] * 6

        # Build publisher lists in channel order
        self._pub_gt  = []
        self._pub_est = []
        self._pub_abs = []
        self._pub_pct = []
        for ch in _CHANNELS:
            self._pub_gt.append( self.create_publisher(Float64, f'/debug/{ch}_gt',        10))
            self._pub_est.append(self.create_publisher(Float64, f'/debug/{ch}_est',       10))
            self._pub_abs.append(self.create_publisher(Float64, f'/debug/{ch}_error_abs', 10))
            self._pub_pct.append(self.create_publisher(Float64, f'/debug/{ch}_error_pct', 10))

        self.create_subscription(Odometry, self.gt_topic_,  self._gt_cb,  10)
        self.create_subscription(Odometry, self.est_topic_, self._est_cb, 10)

        self.get_logger().info(
            f'StateErrorMonitor ready  gt={self.gt_topic_}  est={self.est_topic_}  '
            f'min_pos={min_pos} m  min_speed={min_speed} m/s'
        )

    def _gt_cb(self, msg: Odometry):
        for i, v in enumerate(_extract(msg)):
            self._gt[i] = float(v)
        self._publish()

    def _est_cb(self, msg: Odometry):
        for i, v in enumerate(_extract(msg)):
            self._est[i] = float(v)
        self._publish()

    def _publish(self):
        for i in range(6):
            gt  = self._gt[i]
            est = self._est[i]
            if gt is None or est is None:
                continue
            if not (math.isfinite(gt) and math.isfinite(est)):
                continue

            err_abs = abs(gt - est)
            denom   = max(abs(gt), self._min_ref[i])
            err_pct = (err_abs / denom) * 100.0

            self._pub_gt[i].publish( Float64(data=gt))
            self._pub_est[i].publish(Float64(data=est))
            self._pub_abs[i].publish(Float64(data=err_abs))
            if abs(gt) >= self._min_ref[i]:
                self._pub_pct[i].publish(Float64(data=err_pct))


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
