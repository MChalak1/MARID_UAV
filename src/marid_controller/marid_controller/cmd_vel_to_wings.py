#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class CmdVelToWings(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_to_wings")

        # Parameters to tune mapping from cmd_vel to joints
        self.declare_parameter("front_gain", 1.0)
        self.declare_parameter("tail_gain", 1.0)
        self.declare_parameter("roll_gain", 0.5)
        self.declare_parameter("yaw_gain", 0.5)
        self.declare_parameter("thrust_gain", 900.0)

        self.front_gain = float(self.get_parameter("front_gain").value)
        self.tail_gain = float(self.get_parameter("tail_gain").value)
        self.roll_gain = float(self.get_parameter("roll_gain").value)
        self.yaw_gain = float(self.get_parameter("yaw_gain").value)
        self.thrust_gain = float(self.get_parameter("thrust_gain").value)

        qos = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            "cmd_vel",
            self.cmd_vel_callback,
            qos,
        )

        # Directly command Gazebo joints
        self.left_wing_pub = self.create_publisher(
            Float64,
            "/model/marid/joint/left_wing_joint/cmd_pos",
            qos,
        )
        self.right_wing_pub = self.create_publisher(
            Float64,
            "/model/marid/joint/right_wing_joint/cmd_pos",
            qos,
        )
        self.tail_left_pub = self.create_publisher(
            Float64,
            "/model/marid/joint/tail_left_joint/cmd_pos",
            qos,
        )
        self.tail_right_pub = self.create_publisher(
            Float64,
            "/model/marid/joint/tail_right_joint/cmd_pos",
            qos,
        )

        # Center thruster only
        self.center_thrust_pub = self.create_publisher(
            Float64,
            "/model/marid/joint/thruster_center_joint/cmd_thrust",
            qos,
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        # Symmetric components from vertical stick motion
        front_avg = self.front_gain * msg.linear.x      # left stick vertical
        tail_avg = self.tail_gain * msg.angular.z       # right stick vertical

        # Differential components from horizontal stick motion
        roll_cmd = self.roll_gain * msg.linear.y        # left stick horizontal
        yaw_cmd = self.yaw_gain * msg.angular.y         # right stick horizontal

        # Front wings: average ± roll differential
        front_left = front_avg + roll_cmd
        front_right = front_avg - roll_cmd

        # Tail wings: average ± yaw differential
        tail_left = tail_avg + yaw_cmd
        tail_right = tail_avg - yaw_cmd

        # Center thrust from linear.z (e.g. right trigger)
        # Assume joystick axis in [-1, 1]; map to [0, 1]
        raw_throttle = msg.linear.z
        throttle_01 = 0.5 * (raw_throttle + 1.0)
        thrust = self.thrust_gain * throttle_01

        front_left_msg = Float64()
        front_left_msg.data = front_left
        self.left_wing_pub.publish(front_left_msg)

        front_right_msg = Float64()
        front_right_msg.data = front_right
        self.right_wing_pub.publish(front_right_msg)

        tail_left_msg = Float64()
        tail_left_msg.data = tail_left
        self.tail_left_pub.publish(tail_left_msg)

        tail_right_msg = Float64()
        tail_right_msg.data = tail_right
        self.tail_right_pub.publish(tail_right_msg)

        thrust_msg = Float64()
        thrust_msg.data = thrust
        self.center_thrust_pub.publish(thrust_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelToWings()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

