#!/usr/bin/env python2
## Agent Class
#This class has the functions to calculate state, rewards and such

from re import S
import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import time
import numpy as np
from operator import add


class AgentClass():
    def __init__(self):
        self.Pos=[0,0,0,0,0,0,0]
        self.laser_scan=list(np.zeros(27))
        self.Vel=[]
        self.state=list(np.zeros(7))
        self.past_state=list(np.zeros(31))
        self.goal=[-8,8] # Gambiarra, pq isso aqui fica bem estranho
    def n_min(self,data,n=0.2):
        sorted=np.sort(data)
        index=int(len(data)*n)
        return sorted[index]



      
    def get_scan_sections(self,theta=36,angle=180):
        #Separates the laser scans in sections of theta degrees
        #angle is the angle range of scans
        # Currently using 180
        laser_sections=[]
        if len(self.laser_scan)<727:
            units_per_degree=(727/270)#270 being the full range of the laser scan
        else:
            units_per_degree=(len(self.laser_scan)/270)#270 being the full range of the laser scan
        step=int(np.round(units_per_degree*theta))
        #Defining the start angle in case not using the full 270
        if angle !=270:
            discount_degrees=int((270-angle)) #how many degrees to be discounted from each side
            discount_index=int(discount_degrees*units_per_degree/2)#how many index to skip from each side
            start=0+discount_index
            end=len(self.laser_scan)-discount_index
        else:
            start=0
            end=len(self.laser_scan)
        laser_scan=np.array(self.laser_scan)
        laser_scan[laser_scan>4]=4
        for i in range(start,end,step):
            #Getting the mean reading THIS SHOULD BE CHECKED

            #laser_sections.append(np.mean(laser_scan[i:(i+step)]))

            # Getting n-st lower item
            laser_sections.append(self.n_min(laser_scan[i:(i+step)],0.2))
        
        return laser_sections
    
    def get_state_discrete(self,cl=0.5,md=1.3):
        #Currently state is Section readings+robot coordinates+goal coordinates
        #Gathers information relevant to form the state of the robot
        #The state is treated as 4 discrete groups: very close(3), close(2),ok(1), safe(4)
        laser_sections=np.array(self.get_scan_sections())
        
        #treating for discrete groups:
        laser_sections_cl=laser_sections<cl
        laser_sections_cl=laser_sections_cl.astype(int)
        laser_sections_md=laser_sections<md
        laser_sections_md=laser_sections_md.astype(int)
        #laser_sections_saf=laser_sections>=md
        #laser_sections_saf=laser_sections_saf.astype(int)
        
        laser_sections=laser_sections_cl+laser_sections_md
        #Pegar região do destino
        agent_coord=[self.Pos[0],self.Pos[1]]#xy coordinates
        goal_coord=self.goal
        agent_ori=self.Pos[5]
        agent_pos=(self.Pos[0],self.Pos[1])
        goal_pos=(goal_coord[0],goal_coord[1])
        resultante=(-(agent_pos[0]-goal_pos[0]),-(agent_pos[1]-goal_pos[1]))
        angle=math.atan2(resultante[1],resultante[0])-agent_ori
        #print(angle)
        #angle=(-(agent_ori-np.angle(goal_coord[0]+goal_coord[1]*1j)))
        #print(agent_ori,np.angle((goal_coord[0]-self.Pos[0])+goal_coord[1]-(self.Pos[1]*1j)))
        #print(np.rad2deg(angle),agent_ori)
        #
        if angle>-np.deg2rad(18) and angle<np.deg2rad(18):
            angle=2
        elif angle<-np.deg2rad(18) and angle>-np.deg2rad(54):
            angle=1
        elif angle<-np.deg2rad(54) and angle>=-np.deg2rad(90):
            angle=0
        elif angle>np.deg2rad(18) and angle<np.deg2rad(54):
            angle=3
        elif angle>np.deg2rad(54) and angle<=np.deg2rad(90):
            angle=4
        elif angle<=-np.deg2rad(90) or angle>=np.deg2rad(90):
            angle=5        
        #updates past state before updating the current state
        self.past_state=self.state
        #Concatenate all informations relevant to the state        
        self.state=list(laser_sections)
        self.state.append(angle)
        
        #self.state=''.join(map(str,self.state))
        return self.state

        
    def get_reward(self,Kclose=-0.3,Kobs=-1,Kgoal=20,Ktime=-0.1,very_close=0.8,close=1.5,dist_crash=0.30):
        #Reward Structure
        #1:reward for getting closer to goal(Kcloser)
        #2:penalty for close to obstacle(Kobs)
        #3:Reward for reaching goal(Kgoal)
        #4 Penalty for time passing(Ktime)
        #1
        r1=[]
        #print(np.array((self.past_state[-4:-3])))
        
        # new_distance=((self.state[-4]-self.goal[0])**2+
        #               (self.state[-3]-self.goal[1])**2)**(0.5)
        # print(len(self.state),self.state,(self.state))
        # print()
        # print(len(self.goal),self.goal)
        
        new_distance=np.linalg.norm([self.Pos[0]-self.goal[0],
                                     self.Pos[1]-self.goal[1]])
        # if new_distance<old_distance:
        #     #got closer
        #     r1=Kclose
        # else:
        #     #no penalty for getting away. This can be changed
        #     r1=-0.5*Kclose
        #r1=Kclose*new_distance/(np.linalg.norm([self.goal]))    
        #2
        r2=0
        for i in range(0,len(self.state)-1):
            if int(self.state[i])==2:
                r2=r2+100*Kobs
            if int(self.state[i])==1:
                r2=r2+2*Kobs
            if int(self.state[i])==0:
                r2=r2+1*Kobs
        # for i in range(0,len(self.state)-1):
        #     #Analyzing each section
        #     if i==0 or i==4:
        #         if int(self.state[i])== 2:
        #             r2=r2+100*Kobs
        #         elif int(self.state[i])==1:
        #             r2=r2+3*Kobs
        #         elif int(self.state[i])==0:
        #             r2=r2+Kobs
        #     if i==1 or i==3:
        #         if int(self.state[i])== 2:
        #             r2=r2+100*Kobs*2
        #         elif int(self.state[i])==1:
        #             r2=r2+3*Kobs*2
        #         elif int(self.state[i])==0:
        #             r2=r2+Kobs*2
        #     if i==2:
        #         if int(self.state[i])== 2:
        #             r2=r2+100*Kobs*3
        #         elif int(self.state[i])==1:
        #             r2=r2+3*Kobs*3
        #         elif int(self.state[i])==0:
        #             r2=r2+Kobs*3
        

        #3
        r3=0
        if new_distance<0.3:
            #close enough
            r3=Kgoal
            
        #4
        r4=Ktime
        #print(r1,r2,r3,r4)

        r5=0
        #print(self.state,'sda')
        if len(self.state)>=5 and int(self.state[5])!=2:
            r5=-3
        else:
            r5=10
        
        reward=r2+r5
        r3=0
        if new_distance<0.3:
            #close enough
            reward=Kgoal
            
        return reward
    
 
    def get_euler_from_quaternion(self,x, y, z, w):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll_x = math.atan2(t0, t1)
        
            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = math.asin(t2)
        
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw_z = math.atan2(t3, t4)
        
            return roll_x, pitch_y, yaw_z # in radians
    def get_quaternion_from_euler(self,roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
    def is_done(self,number_iterations=-1,max_iterations=200,reach_dist=0.3):
        done=False
        status=-1 # -1 for not done, 0 for crash, 1 for goal 2 for time 
        
        new_distance=((self.Pos[0]-self.goal[0])**2+
                      (self.Pos[1]-self.goal[1])**2)**(0.5)
        if new_distance<reach_dist:
            done = True
            status=1
        if len(self.laser_scan)>0:# avoiding the first check
            if min(self.laser_scan)<0.2:
                #Crash ! 
                done = True
                status=0
        if number_iterations>max_iterations:
            done=True
            status=2
        if number_iterations==-1:
            return done
        else:
            return done,status
            



