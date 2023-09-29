#!/usr/bin/env python

## Communication Class
#This class has functions in order to receive and send information
#from and to gazebo

#from paramiko import Agent
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan
import time
from Agent import AgentClass
import numpy as np

class CommunicationP3DX():
    
    def __init__(self,A=None):
        self.freq=10#frequency in Hz
        self.P3DX_vel_publisher = rospy.Publisher('/p3dx/cmd_vel', Twist, queue_size=1)
        self.P3DX_state_publisher=rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.StateMsg=ModelState()
        self.cmd = Twist()
        self.ctrl_c = False
        self.rate = rospy.Rate(self.freq) # 10hz
        self.laser_subscriber=rospy.Subscriber('/p3dx/laser/scan', LaserScan, self.laser_callback)
        self.odom_subscriber=rospy.Subscriber('/p3dx/odom', Odometry, self.odom_callback)
        self.laser_scan_full=[]#full laser information
        self.laser_scan=[] # laser .range information
        self.odom_full=[]# full odometry information
        self.odom=[]# x,y,z position and orientation
        self.forward_vel=[0.4 ,0]
        self.left_vel=[0.2 ,0.4]
        self.right_vel=[0.2 , -0.4]
        if (type(A) is AgentClass):
            self.Ag=A
        #A.__init__(self)
        
        
        #rospy.on_shutdown(self.shutdownhook)
        #Custom action
    
    def odom_callback(self,msg):
       self.odom_full=msg
       self.odom=[msg.pose.pose.position.x ,msg.pose.pose.position.y, msg.pose.pose.position.z,
       msg.pose.pose.orientation.x , msg.pose.pose.orientation.y,
       msg.pose.pose.orientation.z , msg.pose.pose.orientation.w]
       
       
       try:
            if (type(self.Ag) is AgentClass):
                
                eulerOri=self.Ag.get_euler_from_quaternion(msg.pose.pose.orientation.x , msg.pose.pose.orientation.y,msg.pose.pose.orientation.z , msg.pose.pose.orientation.w)
                
                self.Ag.Pos=[self.odom[0],self.odom[1],self.odom[2],eulerOri[0],eulerOri[1],eulerOri[2]]
                self.Ag.Vel=[msg.twist.twist.linear.x ,msg.twist.twist.linear.y, msg.twist.twist.linear.z,
            msg.twist.twist.angular.x , msg.twist.twist.angular.y,
            msg.twist.twist.angular.z]
       except:
           pass
       rospy.logdebug("Odometry Data Updated")

    def laser_callback(self,msg):
        #This function is executed every time a message is published in the topic /p3dx/laser/scan
        
        #So this function should update the parameter self.laser_scan to the newest laser scan
        self.laser_scan_full=msg
        self.laser_scan=msg.ranges
        
        try:
            if (type(self.Ag) is AgentClass):
                self.Ag.laser_scan=msg.ranges
        except:
           pass
        if len(self.laser_scan)==0:
            self.laser_scan=list(np.zeros(727))
            print("Veio com Zeroooooooooo")
        rospy.logdebug("Laser Scans Updated")
    def StateDef(self,x,y,oz):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuous publishing systems, this is no big deal, but in systems that publish only
        once, it IS very important.
        """
        self.StateMsg.model_name='p3dx'
        self.StateMsg.pose.position.x=x
        self.StateMsg.pose.position.y=y
        self.StateMsg.pose.orientation.x=oz[0]
        self.StateMsg.pose.orientation.y=oz[1]
        self.StateMsg.pose.orientation.z=oz[2]
        self.StateMsg.pose.orientation.w=oz[3]
        while not self.ctrl_c:
            connections = self.P3DX_vel_publisher.get_num_connections()
            if connections > 0:
                self.P3DX_state_publisher.publish(self.StateMsg)
                rospy.logdebug("State changed")
                rospy.sleep(0.1)
                break
            else:
                self.rate.sleep()
        
    
   
    def custom_action(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuous publishing systems, this is no big deal, but in systems that publish only
        once, it IS very important.
        """
        while not self.ctrl_c:
            connections = self.P3DX_vel_publisher.get_num_connections()
            if connections > 0:
                self.P3DX_vel_publisher.publish(self.cmd)
                rospy.logdebug("Custom Action Published")
                break
            else:
                self.rate.sleep()
    #Move Forward
    def action_forward(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuous publishing systems, this is no big deal, but in systems that publish only
        once, it IS very important.
        """
        self.cmd.linear.x=self.forward_vel[0]
        self.cmd.angular.z=self.forward_vel[1]
        while not self.ctrl_c:
            connections = self.P3DX_vel_publisher.get_num_connections()
            if connections > 0:
                self.P3DX_vel_publisher.publish(self.cmd)
                rospy.logdebug("Forward Action Published")
                #rospy.sleep(0.1)
                time.sleep(0.1) 
                break
            else:
                self.rate.sleep()    
    #Move Left
    def action_left(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuous publishing systems, this is no big deal, but in systems that publish only
        once, it IS very important.
        """
        self.cmd.linear.x=self.left_vel[0]
        self.cmd.angular.z=self.left_vel[1]
        while not self.ctrl_c:
            connections = self.P3DX_vel_publisher.get_num_connections()
            if connections > 0:
                self.P3DX_vel_publisher.publish(self.cmd)
                rospy.logdebug("Left Action Published")
                time.sleep(0.1) 
                break
            else:
                self.rate.sleep()  

    #Move Right
    def action_right(self):
        """
        This is because publishing in topics sometimes fails the first time you publish.
        In continuous publishing systems, this is no big deal, but in systems that publish only
        once, it IS very important.
        """
        self.cmd.linear.x=self.right_vel[0]
        self.cmd.angular.z=self.right_vel[1]
        self.cmd.linear.x=0.3
        self.cmd.angular.z=-0.6
        while not self.ctrl_c:
            connections = self.P3DX_vel_publisher.get_num_connections()
            if connections > 0:
                self.P3DX_vel_publisher.publish(self.cmd)
                rospy.logdebug("Right Action Published")
                time.sleep(0.1) 
                break
            else:
                self.rate.sleep()
    
    def shutdownhook(self):
        #works better than rospy.is_shut_down()
        self.stop_P3DX()
        self.ctrl_c = True
    
    def stop_P3DX(self):
        rospy.logdebug("shutdown time! Stop the robot")
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.custom_action()
