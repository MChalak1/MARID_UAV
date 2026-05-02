#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64

BUTTON_A = 0   # thrust -10 N
BUTTON_B = 1   # front wing AOA down
BUTTON_X = 2   # front wing AOA up
BUTTON_Y = 3   # thrust +10 N
BUTTON_6 = 6   # thrust halve
BUTTON_7 = 7   # thrust preset 2165 N
AXIS_DPAD_V = 7  # D-pad vertical: up=+1, down=-1  (axes[], not buttons[])

THRUST_STEP = 10.0    # N
THRUST_PRESET = 2165.0  # N
SURFACE_STEP = 0.02   # rad


class JoyIncrementer(Node):
    def __init__(self):
        super().__init__('joy_incrementer')

        self.declare_parameter('thrust_min', 0.0)
        self.declare_parameter('thrust_max', 5000.0)
        self.declare_parameter('surface_min', -0.5)
        self.declare_parameter('surface_max', 0.5)

        self.thrust = 0.0
        self.vtail = 0.0
        self.front_wing = 0.0

        self.prev_buttons = []
        self.prev_dpad_v = 0.0

        qos = rclpy.qos.QoSProfile(depth=1)

        # Thrust goes directly to the Gazebo joint — independent of cmd_vel pipeline
        self.thrust_pub = self.create_publisher(
            Float64, '/model/marid/joint/thruster_center_joint/cmd_thrust', qos)
        self.vtail_pub = self.create_publisher(Float64, '/marid/teleop/vtail', qos)
        self.wing_pub = self.create_publisher(Float64, '/marid/teleop/front_wing', qos)

        self.create_subscription(Joy, '/joy', self._joy_cb, 10)
        # Publish thrust at 20 Hz so Gazebo keeps the setpoint even without button activity
        self.create_timer(0.05, self._thrust_timer)
        self.get_logger().info('joy_incrementer ready')

    def _thrust_timer(self):
        self.thrust_pub.publish(Float64(data=self.thrust))

    def _joy_cb(self, msg: Joy):
        buttons = msg.buttons
        axes = msg.axes

        if not self.prev_buttons:
            self.prev_buttons = [0] * len(buttons)
            return  # skip first message to avoid spurious edges

        changed = False
        thrust_min = self.get_parameter('thrust_min').value
        thrust_max = self.get_parameter('thrust_max').value
        surf_min = self.get_parameter('surface_min').value
        surf_max = self.get_parameter('surface_max').value

        def rising(idx):
            return len(buttons) > idx and buttons[idx] and not self.prev_buttons[idx]

        if rising(BUTTON_Y):
            self.thrust = min(self.thrust + THRUST_STEP, thrust_max)
            changed = True
        if rising(BUTTON_A):
            self.thrust = max(self.thrust - THRUST_STEP, thrust_min)
            changed = True
        if rising(BUTTON_6):
            self.thrust = max(self.thrust * 0.5, thrust_min)
            self.vtail = 0.0
            self.front_wing = 0.0
            changed = True
        if rising(BUTTON_7):
            self.thrust = THRUST_PRESET
            changed = True

        dpad_v = axes[AXIS_DPAD_V] if len(axes) > AXIS_DPAD_V else 0.0
        if dpad_v > 0.5 and self.prev_dpad_v <= 0.5:
            self.vtail = min(self.vtail + SURFACE_STEP, surf_max)
            changed = True
        elif dpad_v < -0.5 and self.prev_dpad_v >= -0.5:
            self.vtail = max(self.vtail - SURFACE_STEP, surf_min)
            changed = True
        self.prev_dpad_v = dpad_v

        if rising(BUTTON_X):
            self.front_wing = min(self.front_wing + SURFACE_STEP, surf_max)
            changed = True
        if rising(BUTTON_B):
            self.front_wing = max(self.front_wing - SURFACE_STEP, surf_min)
            changed = True

        if changed:
            self.vtail_pub.publish(Float64(data=self.vtail))
            self.wing_pub.publish(Float64(data=self.front_wing))
            self.get_logger().info(
                f'thrust={self.thrust:.1f}N  vtail={self.vtail:.3f}rad  wing={self.front_wing:.3f}rad'
            )

        self.prev_buttons = list(buttons)


def main(args=None):
    rclpy.init(args=args)
    node = JoyIncrementer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
