import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import math
import numpy as np
from .robot import Robot

class ForwardKinematicsSubscriber(Node):

    def __init__(self):
        super().__init__('forward_kinematics_subscriber')

        # Subscribe to qvals channel for joint values. Must be 3 joint values
        self.subscription = self.create_subscription(
            JointState,
            'Joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning   
        self.publisher = self.create_publisher(Float32MultiArray, '/end_effector_pose', 10)
        self.robot = Robot()
        print("ji")

    def listener_callback(self, msg):
        # Assuming msg.data contains joint values for q1, q2, q3, q4
        print(msg)
        if len(msg.position) == 4:
            q1, q2, q3, q4 = msg.position

            # Compute forward kinematics
            H_4 = self.robot.forward(q1, q2, q3, q4)

            # Extract the end effector position (x, y, z)
            x, y, z, _ = np.dot(H_4, [0, 0, 0, 1])

            # Log and publish the end effector position
            self.get_logger().info(f"The end effector is at: [{x}, {y}, {z}]")    

            pos_msg = Float32MultiArray()
            pos_msg.data = [x, y, z]
            self.publisher.publish(pos_msg)
        else:
            self.get_logger().warn('Received joint states with incorrect number of elements.')


def main(args=None):
    rclpy.init(args=args)

    subscriber = ForwardKinematicsSubscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
