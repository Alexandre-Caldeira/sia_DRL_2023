
from CommunicationP3DX import CommunicationP3DX
from Agent import AgentClass
from std_srvs.srv import Empty
import rospy
import time
from datetime import datetime
import shelve
import os

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
Agent=AgentClass()
Com=CommunicationP3DX(Agent)


# Agent.get_state_discrete()
Agent.get_reward()
time.sleep(1)
# Agent.get_state_discrete()
Agent.get_reward()
time.sleep(1)
# Agent.get_state_discrete()
Agent.get_reward()
reset_simulation()



laser_scan_state_type_atual = 'min'
theta_atual =46
str_hora_inicio_treino = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]


print(str(Agent.get_scan_sections()))
print(str(Agent.Pos))
print(str_hora_inicio_treino+' state> '+str(Agent.get_state_discrete(laser_scan_state_type='min', theta = theta_atual)))
print(str_hora_inicio_treino+' state> '+str(Agent.get_state_discrete(laser_scan_state_type='mean', theta = theta_atual)))
print(str_hora_inicio_treino+' state> '+str(Agent.get_state_discrete(laser_scan_state_type='mode', theta = theta_atual)))
print(str_hora_inicio_treino+' state> '+str(Agent.goal))
print(len(Agent.state))


dd = {}
dd['novo'] = 0
dd['novo_2'] = {}
dd['novo_2'][1] = [5]
dd['novo_2'][1].append(10)
print(dd)
print(dd['novo_2'][1][1])

str_hora_inicio_treino = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]

path = '/media/nero-ia/ADATA UFD/sim_data/wsh_'+str(laser_scan_state_type_atual)+str(theta_atual)+'_'+str_hora_inicio_treino

if not os.path.exists(path):
    os.makedirs(path)

filename=path+'/wsh_'+str_hora_inicio_treino+'.out'
my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()
