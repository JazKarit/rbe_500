import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
import numpy as np

class InverseKinematicsSubscriber(Node):

    def __init__(self):
        super().__init__('forward_kinematics_subscriber')

        # Subscribe to pose channel for desired robot pose
        self.subscription = self.create_subscription(
            Pose,
            'pose',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1

    def listener_callback(self, msg):
        # Get desired x,y,z positions
        # Any orientation given by the msg is ignored since our arm has only 3 Dof 
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z

        # First find theta 1 which is defined by y and x
        theta_1 = np.arctan2(y,x)

        # Find D using the law of cosines, D = cos(theta_3)
        # Law of cosines c^2 = a^2 + b^2 - 2ab*cos(theta_3)
        # x^2 + y^2 + (z-l1)^2 = c^2
        # l2 = a, l3 = b
        D = (np.power(y,2) + np.power(x,2) + np.power(z-self.l1,2) - np.power(self.l2,2) - np.power(self.l3,2)) / (2*self.l2*self.l3)
        
        # Use arctan2 to get theta 3 to get the correct quadrant
        # Add a negative in front since theta_3 is cw for the robot but ccw in the slides
        # Using the positive square root means we choose the elbow down position
        theta_3 = -np.arctan2(np.sqrt(1-np.power(D,2)),D)

        # Find theta_2 from theta_3 using geometry. Again use arctan2 to preserve the quadrant
        # theta 3 is replaced by negative theta 3 since it is backwards from the calculations in the slides
        # Add a negative in the front since theta 2 is cw for the robot and ccw in the slides
        theta_2 = -(np.arctan2(z-self.l1,np.sqrt(np.power(x,2)+np.power(y,2))) - np.arctan2(self.l3*np.sin(-theta_3),self.l2+self.l3*np.cos(-theta_3)))
        print(f"\nTheta 1: {np.round(theta_1,3)}")
        print(f"Theta 2: {np.round(theta_2,3)}")
        print(f"Theta 3: {np.round(theta_3,3)}")

        
def main(args=None):
    rclpy.init(args=args)

    subscriber = InverseKinematicsSubscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
