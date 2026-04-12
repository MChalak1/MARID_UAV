#!/usr/bin/env python3
"""
Wheel odometry node for MARID ground taxiing.

Uses front wheel joint velocities + IMU heading to dead-reckon XY position
while the drone is on the ground. Only active when sonar AGL <= ground_threshold.
FAST-LIO is gated the same way, so on the ground this node is the primary
position source; in the air FAST-LIO + optical flow take over.

Wheel geometry (from URDF):
  left_front_wheel_joint  / right_front_wheel_joint : radius 0.089 m, axis Y
  rear_wheel_joint                                  : radius 0.080 m, axis Y

Positive angular velocity around Y = forward motion (right-hand rule, wheel
contact point moves in +X when omega_y > 0).
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__('wheel_odometry_node')

        # Parameters
        self.declare_parameter('front_wheel_radius', 0.089)
        self.declare_parameter('rear_wheel_radius',  0.080)
        self.declare_parameter('ground_threshold',   0.30)   # sonar AGL, metres
        self.declare_parameter('publish_tf',         False)   # don't fight EKF TF
        self.declare_parameter('odom_frame',         'odom')
        self.declare_parameter('base_frame',         'base_link_front')

        self.r_front_ = self.get_parameter('front_wheel_radius').value
        self.r_rear_  = self.get_parameter('rear_wheel_radius').value
        self.gnd_thr_ = self.get_parameter('ground_threshold').value
        self.pub_tf_  = self.get_parameter('publish_tf').value
        self.odom_fr_ = self.get_parameter('odom_frame').value
        self.base_fr_ = self.get_parameter('base_frame').value

        # Joint names expected in /joint_states
        self.LEFT_JOINT  = 'left_front_wheel_joint'
        self.RIGHT_JOINT = 'right_front_wheel_joint'
        self.REAR_JOINT  = 'rear_wheel_joint'

        # State
        self.x_   = 0.0
        self.y_   = 0.0
        self.yaw_ = 0.0          # from IMU (Madgwick-filtered)
        self.on_ground_ = False
        self.prev_time_ = None

        # Subscriptions
        self.create_subscription(JointState, '/joint_states',
                                 self.joint_cb, 10)
        self.create_subscription(Imu, '/imu_ekf',
                                 self.imu_cb, 10)
        self.create_subscription(Range, '/sonar/range',
                                 self.sonar_cb, 10)

        # Publishers
        self.odom_pub_ = self.create_publisher(Odometry, '/wheel/odometry', 10)
        self.tf_br_    = TransformBroadcaster(self)

        self.get_logger().info(
            f'Wheel odometry ready  '
            f'(r_front={self.r_front_} m, r_rear={self.r_rear_} m, '
            f'ground_thr={self.gnd_thr_} m)'
        )

    # ------------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        """Extract yaw from Madgwick-filtered IMU."""
        q = msg.orientation
        # yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw_ = math.atan2(siny_cosp, cosy_cosp)

    def sonar_cb(self, msg: Range):
        """Gate integration on sonar altitude."""
        if math.isfinite(msg.range) and msg.min_range <= msg.range <= msg.max_range:
            self.on_ground_ = msg.range <= self.gnd_thr_
        else:
            self.on_ground_ = False

    # ------------------------------------------------------------------
    def joint_cb(self, msg: JointState):
        """Main integration step, driven by joint state updates (~50 Hz in Gazebo)."""
        now = self.get_clock().now()

        # Parse wheel velocities by joint name
        vel = {name: v for name, v in zip(msg.name, msg.velocity)}

        v_left  = vel.get(self.LEFT_JOINT,  0.0) * self.r_front_
        v_right = vel.get(self.RIGHT_JOINT, 0.0) * self.r_front_
        v_rear  = vel.get(self.REAR_JOINT,  0.0) * self.r_rear_

        # Forward velocity: average all three wheels.
        # All three should agree when going straight; averaging suppresses
        # transient slip on any single wheel.
        v_fwd = (v_left + v_right + v_rear) / 3.0

        # Only integrate while on the ground
        if self.on_ground_ and self.prev_time_ is not None:
            dt = (now - self.prev_time_).nanoseconds / 1e9
            if 0.0 < dt < 0.5:   # ignore stale or sim-pause gaps
                self.x_ += v_fwd * math.cos(self.yaw_) * dt
                self.y_ += v_fwd * math.sin(self.yaw_) * dt

        self.prev_time_ = now
        self._publish(now, v_fwd)

    # ------------------------------------------------------------------
    def _publish(self, now, v_fwd: float):
        msg = Odometry()
        msg.header.stamp    = now.to_msg()
        msg.header.frame_id = self.odom_fr_
        msg.child_frame_id  = self.base_fr_

        msg.pose.pose.position.x = self.x_
        msg.pose.pose.position.y = self.y_
        msg.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, self.yaw_)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        msg.twist.twist.linear.x = v_fwd
        msg.twist.twist.linear.y = 0.0

        # Pose covariance: tight on ground, large when airborne
        p_var = 0.01 if self.on_ground_ else 9999.0
        msg.pose.covariance[0]  = p_var   # x
        msg.pose.covariance[7]  = p_var   # y
        msg.pose.covariance[35] = 0.05    # yaw (from IMU)
        msg.twist.covariance[0] = 0.05    # vx

        self.odom_pub_.publish(msg)

        if self.pub_tf_:
            t = TransformStamped()
            t.header.stamp    = now.to_msg()
            t.header.frame_id = self.odom_fr_
            t.child_frame_id  = self.base_fr_
            t.transform.translation.x = self.x_
            t.transform.translation.y = self.y_
            t.transform.translation.z = 0.0
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_br_.sendTransform(t)


def main():
    rclpy.init()
    node = WheelOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
