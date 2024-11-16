import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
import math
import numpy as np

class TwistSubscriber(Node):

    def __init__(self):
        super().__init__('twist_to_joint_vel')

        # Subscribe to channel for twist
        self.subscription = self.create_subscription(
            Twist,
            'twist',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        self.l1 = 1
        self.l2 = 1
        self.l3 = 1

        # Arm configuration
        #
        #  o--o
        #  |  |
        #  o  v

        self.q1 = 0
        self.q2 = 0
        self.q3 = np.pi/2
        
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
    
    def Jv(self,q1,q2,q3):
        """
        Find the 3x3 Jacobian for linear velocity
        """

        # zn_ is the vector zn with an extra 0 as the 4th element
        # We can use our A matrices to find these vectors
        z0_ = np.array([[0],
                        [0],
                        [1],
                        [0]])
        z1_ = self.A_1(self.q1) @ z0_
        z2_ = self.A_2(q2) @ z1_
        
        # zn has 3 elements
        z0 = z0_[:3].flatten()
        z1 = z1_[:3].flatten()
        z2 = z2_[:3].flatten()
        
        # Find H_n^0 matrices
        H_1 = self.A_1(q1)
        H_2 = self.A_1(q1) @ self.A_2(q2)
        H_3 = self.A_1(q1) @ self.A_2(q2) @ self.A_3(q3) 

        # Find o_n the origin of the nth frame using H
        zero = np.array([[0],[0],[0],[1]])
        o0 = zero[:3].flatten()
        o1 = (H_1 @ zero)[:3].flatten()
        o2 = (H_2 @ zero)[:3].flatten()
        o3 = (H_3 @ zero)[:3].flatten()
        
        # Calculate J_v_i = z_{i-1} x (o_n-o_{i-1}) for revolute joint
        # Concatenate into Jv (3x3)
        Jv = np.concatenate((np.cross(z0,(o3-o0)).reshape((3,1)), \
                            np.cross(z1,(o3-o1)).reshape((3,1)), \
                            np.cross(z2,(o3-o2)).reshape((3,1))), axis=1)

        return Jv

    def listener_callback(self, msg):
        linear_vel = msg.linear
        angular_vel = msg.angular
        
        x, y, z = linear_vel.x, linear_vel.y, linear_vel.z
        x_,y_,z_ = angular_vel.x, angular_vel.y, angular_vel.z

        # twist = J q'
        # J^{-1} twist = q'
        # Since we only have 3 DoF we will focus on the linear velocity component of the twist
        # Jv^{-1} twist_v = q'

        dq = np.linalg.inv(self.Jv(self.q1,self.q2,self.q3)) @  np.matrix([[x],[y],[z]])
        
        dq = np.round(dq,3)

        print("")
        print(f"Joint 1 Velocity: {dq[0][0]} radians/sec")
        print(f"Joint 2 Velocity: {dq[1][0]} radians/sec")
        print(f"Joint 3 Velocity: {dq[2][0]} radians/sec")

def main(args=None):
    rclpy.init(args=args)

    subscriber = TwistSubscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
