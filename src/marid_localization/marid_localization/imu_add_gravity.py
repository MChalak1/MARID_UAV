#!/usr/bin/env python3
"""
Add world gravity (in body frame) to Gazebo IMU for Madgwick filter.

Gazebo publishes specific force (gravity removed). Madgwick needs ~9.81 magnitude
when level. We use last orientation from /imu/data (filter output) to rotate
world gravity (0, 0, +9.81) into body frame and add to raw accel. Works in sim and real.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf_transformations import quaternion_matrix
import numpy as np


# World-frame "reaction" gravity for Madgwick (ENU: up = +Z)
G_WORLD = np.array([0.0, 0.0, 9.81])


class ImuAddGravity(Node):
    def __init__(self):
        super().__init__('imu_add_gravity')
        self.pub = self.create_publisher(Imu, '/imu/with_gravity', 10)
        self.sub_raw = self.create_subscription(Imu, '/imu', self.cb_raw, 10)
        self.sub_orientation = self.create_subscription(
            Imu, '/imu/data', self.cb_orientation, 10
        )
        self.last_q = None  # [x, y, z, w] from /imu/data

    def cb_orientation(self, msg):
        self.last_q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ]

    def cb_raw(self, msg):
        if self.last_q is not None:
            R = quaternion_matrix(self.last_q)[:3, :3]  # body-to-world
            g_body = R.T @ G_WORLD
        else:
            g_body = G_WORLD.copy()

        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance
        out.angular_velocity = msg.angular_velocity
        out.angular_velocity_covariance = msg.angular_velocity_covariance
        out.linear_acceleration.x = msg.linear_acceleration.x + g_body[0]
        out.linear_acceleration.y = msg.linear_acceleration.y + g_body[1]
        out.linear_acceleration.z = msg.linear_acceleration.z + g_body[2]
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImuAddGravity()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
