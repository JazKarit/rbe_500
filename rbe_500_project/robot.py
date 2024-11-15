import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
import math
import numpy as np


class Robot:
    def __init__(self):
        self.l0l1 = 96.326
        self.l20  = 128
        self.l21  = 24
        self.l2   = np.sqrt(np.power(128,2), np.power(24,2))
        self.l3   = 124
        self.l4   = 133.4
        self.beta = np.arctan2(self.l21,self.l20)

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
        return self.A(q1,-np.pi/2,0,self.l0l1)
    
    def A_2(self,q2):
        """
        Find the Homogenous Transform from frame 2 to frame 1
        """
        return self.A(-np.pi/2+self.beta+q2,0,self.l2,0)
    
    def A_3(self,q3):
        """
        Find the Homogenous Transform from frame 3 to frame 2
        """
        return self.A(np.pi/2-self.beta+q3,0,self.l3,0)
    
    def A_4(self,q3):
        """
        Find the Homogenous Transform from frame 4 to frame 3
        """
        return self.A(self.q4,0,self.l4,0)
    
    def forward(self,q1,q2,q3,q4):
        # Multiply our A matrices to find the transform from end effector frame (frame 3) to base frame (frame 0)
        H_3 = self.A_1(q1) @ self.A_2(q2) @ self.A_3(q3) @ self.A_3(q4)

    def inverse(self,x,y,z,theta):
        z_double_tick = np.sin(theta) * self.l4
        z_tick = z + z_double_tick - self.l0l1

        x_double_tick = np.cos(theta) * self.l4
        x_tick = np.sqrt(x**2 + y**2) - x_double_tick

        a = np.sqrt(x_tick**2 + z_tick**2)

        
        cos_q3_tick = (self.l3**2 + self.l2**2  - a**2) / (2*self.l3*self.l4)

        # todo use atan2?
        q3_tick =  np.arccos(cos_q3_tick)

        M = self.l3/a * np.sin(q3_tick)
        b = np.arctan2(z_tick,x_tick)

        q1 = np.arctan2(y,x)
        q2 = M + b - np.pi / 2
        q3 = q3_tick-(np.pi/2+self.beta)
        q4 = -theta


