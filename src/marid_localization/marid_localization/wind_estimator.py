#!/usr/bin/env python3
"""
Wind Estimator Node

Estimates 3D wind vector in the odom/world frame:
    v_wind = v_ground - v_air_in_odom

Subscriptions:
  - /odometry/filtered/local (nav_msgs/Odometry)   [configurable: odom_topic]
  - /airspeed/velocity        (std_msgs/Float64)   [configurable: airspeed_topic]

Publications:
  - /wind/velocity (geometry_msgs/TwistStamped)  # linear = wind vector in odom frame
  - /wind/speed    (std_msgs/Float64)           # |v_wind|
"""

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped

import numpy as np
from tf_transformations import quaternion_matrix


class WindEstimator(Node):
    def __init__(self) -> None:
        super().__init__("wind_estimator")

        # Parameters
        self.declare_parameter("odom_topic", "/odometry/filtered/local")
        self.declare_parameter("airspeed_topic", "/airspeed/velocity")
        self.declare_parameter("wind_frame_id", "odom")

        self.odom_topic_ = self.get_parameter("odom_topic").value
        self.airspeed_topic_ = self.get_parameter("airspeed_topic").value
        self.wind_frame_id_ = self.get_parameter("wind_frame_id").value

        # State
        self.last_airspeed_: float | None = None

        # Subscribers
        self.odom_sub_ = self.create_subscription(
            Odometry, self.odom_topic_, self.odom_callback, 10
        )
        self.airspeed_sub_ = self.create_subscription(
            Float64, self.airspeed_topic_, self.airspeed_callback, 10
        )

        # Publishers
        self.wind_twist_pub_ = self.create_publisher(
            TwistStamped, "/wind/velocity", 10
        )
        self.wind_speed_pub_ = self.create_publisher(
            Float64, "/wind/speed", 10
        )

        self.get_logger().info("WindEstimator started.")
        self.get_logger().info(f"  Odom topic:     {self.odom_topic_}")
        self.get_logger().info(f"  Airspeed topic: {self.airspeed_topic_}")
        self.get_logger().info(f"  Wind frame_id:  {self.wind_frame_id_}")

    def airspeed_callback(self, msg: Float64) -> None:
        """Store latest airspeed from pitot tube (m/s)."""
        self.last_airspeed_ = msg.data

    def odom_callback(self, msg: Odometry) -> None:
        """On each odom update, estimate wind vector if airspeed is available."""
        if self.last_airspeed_ is None:
            # No airspeed yet, skip
            return

        try:
            # Ground velocity in odom frame
            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.linear.y
            vz = msg.twist.twist.linear.z
            v_ground_odom = np.array([vx, vy, vz], dtype=float)

            # Orientation (odom -> body) from odometry
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w

            # Rotation matrix from quaternion (odom -> body)
            T_odom_body = quaternion_matrix([qx, qy, qz, qw])  # 4x4
            R_odom_body = T_odom_body[:3, :3]

            # We need rotation body->odom to express body-frame vectors in odom frame:
            R_body_odom = R_odom_body.T

            # Airspeed vector in body frame: along +X_body
            airspeed = float(self.last_airspeed_)
            v_air_body = np.array([airspeed, 0.0, 0.0], dtype=float)

            # Airspeed in odom frame
            v_air_odom = R_body_odom @ v_air_body

            # Wind vector in odom frame: v_wind = v_ground - v_air
            v_wind_odom = v_ground_odom - v_air_odom

            wind_speed = float(np.linalg.norm(v_wind_odom))

            # Publish as TwistStamped (linear = wind vector)
            wind_twist = TwistStamped()
            wind_twist.header.stamp = msg.header.stamp
            # Use configured frame or odom frame from incoming message
            wind_twist.header.frame_id = self.wind_frame_id_ or msg.header.frame_id

            wind_twist.twist.linear.x = float(v_wind_odom[0])
            wind_twist.twist.linear.y = float(v_wind_odom[1])
            wind_twist.twist.linear.z = float(v_wind_odom[2])

            # Angular part unused for wind; keep zero
            wind_twist.twist.angular.x = 0.0
            wind_twist.twist.angular.y = 0.0
            wind_twist.twist.angular.z = 0.0

            self.wind_twist_pub_.publish(wind_twist)

            # Publish scalar wind speed
            speed_msg = Float64()
            speed_msg.data = wind_speed
            self.wind_speed_pub_.publish(speed_msg)

        except Exception as e:
            self.get_logger().error(f"Error estimating wind: {e}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WindEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

