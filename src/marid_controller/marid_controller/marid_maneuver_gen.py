#!/usr/bin/env python3
"""
MARID Maneuver Generator
Publishes random nearby waypoints on a random timer to drive the attitude
controller through aggressive heading changes and banks for LSTM training data.

Subscribes : /gazebo/odom  (nav_msgs/Odometry)  — current position
Publishes  : /marid/waypoint (geometry_msgs/PoseStamped) — next target

Parameters
----------
timer_min_s      : float  — minimum seconds between waypoint changes (default 15)
timer_max_s      : float  — maximum seconds between waypoint changes (default 45)
wp_min_dist_m    : float  — minimum waypoint distance from current pos (default 150)
wp_max_dist_m    : float  — maximum waypoint distance from current pos (default 500)
startup_delay_s  : float  — seconds after node start before first waypoint (default 5)
"""

import math
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class ManeuverGenerator(Node):
    def __init__(self):
        super().__init__('marid_maneuver_gen')

        self.declare_parameter('timer_min_s',     15.0)
        self.declare_parameter('timer_max_s',     45.0)
        self.declare_parameter('wp_min_dist_m',  12000.0)
        self.declare_parameter('wp_max_dist_m',  24000.0)
        self.declare_parameter('startup_delay_s',  5.0)

        self._t_min   = self.get_parameter('timer_min_s').value
        self._t_max   = self.get_parameter('timer_max_s').value
        self._d_min   = self.get_parameter('wp_min_dist_m').value
        self._d_max   = self.get_parameter('wp_max_dist_m').value
        startup       = self.get_parameter('startup_delay_s').value

        self._x = 0.0
        self._y = 0.0
        self._has_odom = False
        self._timer = None

        self._odom_sub = self.create_subscription(
            Odometry, '/gazebo/odom', self._odom_cb, 10)
        self._wp_pub = self.create_publisher(
            PoseStamped, '/marid/waypoint', 10)

        # Initial delay before first waypoint
        self._startup_timer = self.create_timer(startup, self._on_startup)

        self.get_logger().info(
            f'Maneuver generator ready — '
            f'timer=[{self._t_min:.0f},{self._t_max:.0f}]s  '
            f'dist=[{self._d_min:.0f},{self._d_max:.0f}]m  '
            f'startup={startup:.0f}s')

    def _odom_cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._has_odom = True

    def _on_startup(self):
        self._startup_timer.cancel()
        self._startup_timer = None
        self._publish_waypoint()
        self._schedule_next()

    def _on_timer(self):
        self._timer.cancel()
        self._timer = None
        self._publish_waypoint()
        self._schedule_next()

    def _schedule_next(self):
        delay = random.uniform(self._t_min, self._t_max)
        self._timer = self.create_timer(delay, self._on_timer)
        self.get_logger().info(f'Next waypoint in {delay:.1f}s')

    def _publish_waypoint(self):
        if not self._has_odom:
            self.get_logger().warn('No odom yet — skipping waypoint')
            return

        bearing = random.uniform(0.0, 2.0 * math.pi)
        dist    = random.uniform(self._d_min, self._d_max)
        wp_x    = self._x + dist * math.cos(bearing)
        wp_y    = self._y + dist * math.sin(bearing)

        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = wp_x
        msg.pose.position.y = wp_y
        msg.pose.position.z = 0.0

        self._wp_pub.publish(msg)
        self.get_logger().info(
            f'Waypoint → ({wp_x:.1f}, {wp_y:.1f})  '
            f'dist={dist:.0f}m  bearing={math.degrees(bearing):.0f}°')


def main(args=None):
    rclpy.init(args=args)
    node = ManeuverGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
