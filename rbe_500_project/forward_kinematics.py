import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
import math
import numpy as np

class ForwardKinematicsSubscriber(Node):

    def __init__(self):
        super().__init__('forward_kinematics_subscriber')

        # Subscribe to qvals channel for joint values. Must be 3 joint values
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'qvals',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1
        
    def A(self,theta,alpha,a,d):
        """
        Finds the Homogenous Transform between two consecutive coordinate frames for a robot arm. 
        Uses the Denavit–Hartenberg parameters.

        Args:
            theta (float): radians 
            alpha (float): radians
            a (float):
            d (float);
        Returns:
            4x4 np.array representing the Homogenous Transform from frame i+1 to frame i"""
        return np.array([[np.cos(theta),-np.sin(theta)*np.cos(alpha),np.sin(theta)*np.sin(alpha),a*np.cos(theta)],
                         [np.sin(theta),np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                         [0,np.sin(alpha),np.cos(alpha),d],
                         [0,0,0,1]])

    def A_1(self,q1):
        """
        Find the Homogenous Transform from frame 1 to frame 0
        """
        return self.A(q1,-np.pi/2,0,self.l1)
    
    def A_2(self,q2):
        """
        Find the Homogenous Transform from frame 2 to frame 1
        """
        return self.A(q2,0,self.l2,0)
    
    def A_3(self,q3):
        """
        Find the Homogenous Transform from frame 3 to frame 2
        """
        return self.A(q3,0,self.l3,0)

    def listener_callback(self, msg):
        q_vals = msg.data
        q_1, q_2, q_3 = q_vals

        # Multiply our A matrices to find the transform from end effector frame (frame 3) to base frame (frame 0)
        H_3 = self.A_1(q_1) @ self.A_2(q_2) @ self.A_3(q_3)     
        
        # Round the values in the matrix so they appear nicely in the terminal
        H_3 = np.round(H_3,3)

        # Transfom the 0 vector in the end effector frame to the base frame to find its position
        x,y,z,_ = np.dot(H_3,[0,0,0,1])
        print(f"The end effector is at: [{x},{y},{z}]")

        # Take the transpose of the H matrix so it is a rotation from base frame to end effector.
        # This is the more logical matrix to display along with the former position vector.
        # "End effector is at pos [x,y,z] and to get to its orientation we need to use this rotation matrix"
        H_3T = H_3.T
        print(f"The rotation matrix from the base frame to the end effector frame is:\n\
                [{H_3T[0][0]},{H_3T[0][1]},{H_3T[0][2]}]\n\
                [{H_3T[1][0]},{H_3T[1][1]},{H_3T[1][2]}]\n\
                [{H_3T[2][0]},{H_3T[2][1]},{H_3T[2][2]}]\n")

def main(args=None):
    rclpy.init(args=args)

    subscriber = ForwardKinematicsSubscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
