import shelve
import warnings
warnings.filterwarnings('ignore')

import pickle
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR

"""
Adapted from https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/blob/master/DDQN_discrete.py#L21
for a gazebo application.

Implementation of Double DQN for gym environments with discrete action space.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from CommunicationP3DX import CommunicationP3DX
from Agent import AgentClass
from std_srvs.srv import Empty
import rospy
import time
import shelve
import os
from datetime import datetime

str_hora_inicio_treino = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]

"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1


"""
If the observations are images we use CNNs.
"""
class QNetworkCNN(nn.Module):
    def __init__(self, action_dim):
        super(QNetworkCNN, self).__init__()

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(8960, 512)
        self.fc_2 = nn.Linear(512, action_dim)

    def forward(self, inp):
        inp = inp.view((1, 3, 210, 160))
        x1 = F.relu(self.conv_1(inp))
        x1 = F.relu(self.conv_2(x1))
        x1 = F.relu(self.conv_3(x1))
        x1 = torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc_1(x1))
        x1 = self.fc_2(x1)

        return x1


"""
memory to save the state, action, reward sequence from the current episode.
"""
class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


def select_action(model, action_space_size, state, eps):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, action_space_size)
    else:
        action = np.argmax(values.cpu().numpy())

    return action


def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

def evaluate(Qmodel, horizon_eval, repeats,episode):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for repeat_i in range(repeats):

        #env.reset()
        state = Agent.get_state_discrete(laser_scan_state_type=laser_scan_state_type_atual, theta=theta_atual)
        reset_simulation()

        done = False
        i= 0
        while not done:
            i += 1

            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())

            if action==0:
                Com.action_forward()
            elif action==1:
                Com.action_right()
            else:
                Com.action_left()

            state=Agent.get_state_discrete(laser_scan_state_type=laser_scan_state_type_atual, theta=theta_atual)
            reward, new_distance, r1, r4 = Agent.get_reward(number_iterations=i)
            done,status_done = Agent.is_done(number_iterations=i,max_iterations=horizon_eval, reach_dist=0.5)

            if i > horizon_eval:
                done = True

        if repeat_i==0:
            hist_dict['rewards'][episode+1] = [[reward, new_distance, r1, r4]]
            hist_dict['epresult'][episode+1] = [done, status_done]

            hist_dict['pos_eval'][episode+1] = [Agent.Pos]
            hist_dict['scan_eval'][episode+1] = [Agent.laser_scan]
            hist_dict['state_eval'][episode+1] = [state]
        else:
            hist_dict['rewards'][episode+1].append([reward, new_distance, r1, r4])
            hist_dict['epresult'][episode+1].append([done, status_done])

            hist_dict['pos_eval'][episode+1].append(Agent.Pos)
            hist_dict['scan_eval'][episode+1].append(Agent.laser_scan)
            hist_dict['state_eval'][episode+1].append(state)


            perform += reward
    Qmodel.train()
    return perform/repeats, new_distance, r1, r4


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())



if __name__ == '__main__':
    save_path='/media/nero-ia/7A309A87309A49D1/sia_23/25k/tests/'

    #for agg_method, n_sectors in training_list:
    import pickle
    with open('/home/nero-ia/Documents/Caldeira/sia_DRL_2023/q1nets_25k.pickle', 'rb') as f:
        dict_Q_1 = pickle.load(f)

    hist_method_goal = {}
    #hist_dict = {'pos':{}, 'scan':{}, 'rewards':{}, 'state':{}, 'epresult':{}}
    #dict_Q_1.items():
    for method in [
        #'min_5'
        #,
        'min_6'
        ,'min_10'
        ,'mode_4'
        ,'mode_5'
        ,'mode_6'
        ,'mode_10'
        ]:
        q_network =dict_Q_1[method]

        for current_goal in [0,1,2,3,4]:

            hist_method_goal['goal_'+str(current_goal)+method] = {'pos':{}, 'scan':{}, 'rewards':{}, 'state':{}, 'epresult':{}}

            rospy.init_node('learning_loop')
            rospy.wait_for_service('/gazebo/reset_world')
            pause_physics_client=rospy.ServiceProxy('/gazebo/pause_physics',Empty)
            unpause_physics_client=rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
            reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

            #Initialize relevant objects
            Agent=AgentClass()
            Com=CommunicationP3DX(Agent)

            Agent.get_state_discrete()
            Agent.get_reward(1)
            rospy.sleep(1)
            Agent.get_state_discrete()
            Agent.get_reward(1)
            rospy.sleep(1)
            Agent.get_state_discrete()
            Agent.get_reward(1)
            reset_simulation()

            Agent.goal = {0:[-8,8],
                        1:[-2,8],
                        2:[8,-3],
                        3:[-8,-8],
                        4:[8,8]}[current_goal]

            starting_pos = {
                    0:[-4,2,[0,0,0,0]],
                    1:[-4,2,[0,0,0,0]],
                    2:[0,-5,[0,0,0,0]],
                    3:[0,-5,[0,0,0,0]],
                    4:[3,2,[0,0,0,0]]
            }[current_goal]

            for _ in [1,2]:
                Com.StateDef(x = starting_pos[0],y =starting_pos[1], oz=starting_pos[2])

            time.sleep(1)

            if len(Agent.state)>0 and len(Agent.past_state)>0:
                rospy.loginfo('States Initialized Succesfuly')
                print('Current goal ->' +str(Agent.goal))
            else:
                rospy.loginfo('States not received!!')

            print('Started method'+str(method)+' for goal ='+str(Agent.goal))

            laser_scan_state_type_atual = method.split('_')[0]
            theta_atual = {
                    '4':46,
                    '5':36.1,
                    '6':30.1,
                    '10': 18.1
                }[method.split('_')[1]]

            horizon_eval = 60
            repeats = 1000 #1000

            q_network.eval()
            perform = 0
            for episode in range(repeats):

                #env.reset()
                #print(theta_atual)
                state = Agent.get_state_discrete(laser_scan_state_type=laser_scan_state_type_atual, theta=theta_atual)
                reset_simulation()
                for _ in [1,2]:
                    Com.StateDef(x = starting_pos[0],y =starting_pos[1], oz=starting_pos[2])


                done = False
                i= 0
                while not done:
                    i += 1

                    state = torch.Tensor(state).to(device)
                    with torch.no_grad():
                        values = q_network(state)
                    action = np.argmax(values.cpu().numpy())

                    if action==0:
                        Com.action_forward()
                    elif action==1:
                        Com.action_right()
                    else:
                        Com.action_left()

                    state=Agent.get_state_discrete(laser_scan_state_type=laser_scan_state_type_atual, theta=theta_atual)
                    reward, new_distance, r1, r4 = Agent.get_reward(number_iterations=i)
                    done,status_done = Agent.is_done(number_iterations=i,max_iterations=horizon_eval, reach_dist=0.5)

                    if i==1:
                        hist_method_goal['goal_'+str(current_goal)+method]['pos'][episode+1] = [Agent.Pos]
                        hist_method_goal['goal_'+str(current_goal)+method]['scan'][episode+1] = [Agent.laser_scan]
                        hist_method_goal['goal_'+str(current_goal)+method]['state'][episode+1] = [state]
                        hist_method_goal['goal_'+str(current_goal)+method]['rewards'][episode+1] = [[reward, new_distance, r1, r4]]
                        hist_method_goal['goal_'+str(current_goal)+method]['epresult'][episode+1] = [done, status_done]
                    else:
                        hist_method_goal['goal_'+str(current_goal)+method]['pos'][episode+1].append(Agent.Pos)
                        hist_method_goal['goal_'+str(current_goal)+method]['scan'][episode+1].append(Agent.laser_scan)
                        hist_method_goal['goal_'+str(current_goal)+method]['state'][episode+1].append(state)
                        hist_method_goal['goal_'+str(current_goal)+method]['rewards'][episode+1].append([[reward, new_distance, r1, r4]])
                        hist_method_goal['goal_'+str(current_goal)+method]['epresult'][episode+1].append([done, status_done])

                    if i > horizon_eval:

                        print([reward,new_distance,status_done])

                        done = True

                    perform += reward

            print('Finished with method '+str(method)+' for goal ='+str(Agent.goal))


            with open(save_path+'tstres_goal_'+str(current_goal)+method+'.pickle', 'wb') as handle:
                current_dict = hist_method_goal['goal_'+str(current_goal)+method].copy()
                pickle.dump(current_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_path+'tstres_all_hist_method_goal.pickle', 'wb') as handle:
            pickle.dump(hist_method_goal, handle, protocol=pickle.HIGHEST_PROTOCOL)