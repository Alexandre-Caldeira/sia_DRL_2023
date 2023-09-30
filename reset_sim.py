
from CommunicationP3DX import CommunicationP3DX
from Agent import AgentClass
from std_srvs.srv import Empty
import rospy
import time

rospy.init_node('learning_loop')
rospy.wait_for_service('/gazebo/reset_world')
pause_physics_client=rospy.ServiceProxy('/gazebo/pause_physics',Empty)
unpause_physics_client=rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

#Placeholder functions for script without NN
actions_linear=[0, 0.2, 0.6] # random linear speed for sampling
actions_angular=[0, 0.2, 0.6] # random angular speed for sampling
goal_zones_x=[(0,2),(3,4.8),(5,7),(5,7)]
goal_zones_y=[(2,6),(3,8),(0,4),(6.5,9)]
#Initialize relevant objects
# Agent=AgentClass()
# Com=CommunicationP3DX(Agent)

reset_simulation()