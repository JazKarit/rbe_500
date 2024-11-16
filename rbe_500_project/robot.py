import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
import math
import numpy as np


class Robot:
    def __init__(self):
        self.l0l1 = 96.326
        self.l20  = 128.
        self.l21  = 24.
        self.l2   = math.sqrt(128.**2 + 24.**2)
        self.l3   = 124.
        self.l4   = 133.4

        self.beta = math.atan2(self.l20,self.l21)

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
        return np.array([[math.cos(theta),-math.sin(theta)*math.cos(alpha),math.sin(theta)*math.sin(alpha),a*math.cos(theta)],
                         [math.sin(theta),math.cos(theta)*math.cos(alpha),-math.cos(theta)*math.sin(alpha),a*math.sin(theta)],
                         [0,math.sin(alpha),math.cos(alpha),d],
                         [0,0,0,1]])

    def A_1(self,q1):
        """
        Find the Homogenous Transform from frame 1 to frame 0
        """
        return self.A(q1,-math.pi/2,0,self.l0l1)
    
    def A_2(self,q2):
        """
        Find the Homogenous Transform from frame 2 to frame 1
        """
        return self.A(-self.beta+q2,0,self.l2,0)
    
    def A_3(self,q3):
        """
        Find the Homogenous Transform from frame 3 to frame 2
        """
        return self.A(self.beta+q3,0,self.l3,0)
    
    def A_4(self,q4):
        """
        Find the Homogenous Transform from frame 4 to frame 3
        """
        return self.A(q4,0,self.l4,0)
    
    def forward(self,q1,q2,q3,q4):
        # Multiply our A matrices to find the transform from end effector frame (frame 3) to base frame (frame 0)
        print(np.round((self.A_1(q1) @ self.A_2(q2)),1))
        
        H_3 = self.A_1(q1) @ self.A_2(q2) @ self.A_3(q3) @ self.A_4(q4)
        return H_3

    def inverse(self,x,y,z,theta):
        z_double_tick = math.sin(theta) * self.l4
        z_tick = z + z_double_tick - self.l0l1
        

        x_double_tick = math.cos(theta) * self.l4
        x_tick = math.sqrt(x**2 + y**2) - x_double_tick
        

        a = math.sqrt(x_tick**2 + z_tick**2)
        

        
        cos_q3_tick = (self.l3**2 + self.l2**2  - a**2) / (2*self.l2*self.l3)

        # todo use atan2?
        q3_tick =  math.acos(cos_q3_tick)
        

        gamma = np.pi/2-self.beta

        M = self.l3/a * math.sin(q3_tick)
        b = math.atan2(z_tick,x_tick)
        q2_tick = M + b + math.pi/2

        q1 = math.atan2(y,x)
        q2 = q2_tick - (math.pi/2 + self.beta)
        q3 = q3_tick - (math.pi/2 + gamma)
        
        q4 = ((np.pi - q2_tick) + (np.pi - q3_tick)) - np.pi/2

        return q1, q2, q3, q4 
    
robot = Robot()

print(np.round(robot.forward(0,0,0,0),2))

print(np.round(np.multiply(robot.inverse(281.4,0,224.33,0),180/np.pi),2) )


