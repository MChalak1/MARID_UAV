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
        self.declare_parameter("thrust_gain", 500.0)

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

        self.teleop_vtail = 0.0
        self.teleop_front_wing = 0.0

        self.create_subscription(Float64, '/marid/teleop/vtail', self._vtail_cb, qos)
        self.create_subscription(Float64, '/marid/teleop/front_wing', self._wing_cb, qos)

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

    def _vtail_cb(self, msg: Float64) -> None:
        self.teleop_vtail = msg.data

    def _wing_cb(self, msg: Float64) -> None:
        self.teleop_front_wing = msg.data

    def cmd_vel_callback(self, msg: Twist) -> None:
        # Symmetric components from vertical stick motion
        front_avg = self.front_gain * msg.linear.x      # left stick vertical
        tail_avg = self.tail_gain * msg.angular.z       # right stick vertical

        # Differential components from horizontal stick motion
        roll_cmd = self.roll_gain * msg.linear.y        # left stick horizontal
        yaw_cmd = self.yaw_gain * msg.angular.y         # right stick horizontal

        # Front wings: average ± roll differential + symmetric AOA offset
        front_left = front_avg + roll_cmd + self.teleop_front_wing
        front_right = front_avg - roll_cmd + self.teleop_front_wing

        # Tail wings: average ± yaw differential + symmetric vtail offset
        tail_left = tail_avg + yaw_cmd + self.teleop_vtail
        tail_right = tail_avg - yaw_cmd + self.teleop_vtail

        self.left_wing_pub.publish(Float64(data=front_left))
        self.right_wing_pub.publish(Float64(data=front_right))
        self.tail_left_pub.publish(Float64(data=tail_left))
        self.tail_right_pub.publish(Float64(data=tail_right))


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

