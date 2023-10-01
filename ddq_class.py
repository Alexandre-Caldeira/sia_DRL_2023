import warnings
warnings.filterwarnings('ignore')

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

def evaluate(Qmodel, horizon_eval, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):

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

            #state, reward, done, _ = env.step(action)
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


            perform += reward
    Qmodel.train()
    return perform/repeats, new_distance, r1, r4


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def main(state_space_size, action_space_size=3, gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995,
         eps_min=0.01, update_step=10, batch_size=64, update_repeats=50, num_episodes=3000, max_memory_size=50000,
         lr_gamma=0.9, lr_step=100, measure_step=100, measure_repeats=100, hidden_dim=64, cnn=False, horizon=200000,horizon_eval=50,
         theta_atual:int= 36,laser_scan_state_type_atual:str = 'min',checkpoint_inter=1000):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    #env = gym.make(env_name)
    #torch.manual_seed(seed)
    #env.seed(seed)

    if cnn:
        Q_1 = QNetworkCNN(action_dim=action_space_size).to(device)
        Q_2 = QNetworkCNN(action_dim=action_space_size).to(device)
    else:
        Q_1 = QNetwork(action_dim=action_space_size, state_dim=state_space_size,
                                        hidden_dim=hidden_dim).to(device)
        Q_2 = QNetwork(action_dim=action_space_size, state_dim=state_space_size,
                                        hidden_dim=hidden_dim).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    # https://douglasrizzo.com.br/step-decay-lr/
    initial_lr = lr
    final_lr = lr*1e-3
    n_updates = num_episodes*0.7
    lr_gamma =  (final_lr / initial_lr)**(1 / n_updates) # gamma

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in range(num_episodes):
        # display the performance
        if (episode % measure_step == 0) and episode >= min_episodes:
            print('eval...')
            reward_eval, new_distance, r1, r4 = evaluate(Q_1, horizon_eval, measure_repeats)
            performance.append([episode, reward_eval])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_lr()[0])
            print("eps: ", eps)

            # hist_dict = {'pos':{}, 'scan':{}, 'rewards':{}, 'rates':{}, 'state':{}, 'epresult':{}}
            hist_dict['rewards'][episode+1] = [performance[-1][1], new_distance, r1, r4]
            hist_dict['rates'][episode+1] = [eps, scheduler.get_lr()[0]]
            hist_dict['epresult'][episode+1] = [done, status_done]

        reset_simulation() #env.reset()
        Agent.get_state_discrete(laser_scan_state_type=laser_scan_state_type_atual, theta=theta_atual)
        state = Agent.state
        #print(state)
        memory.state.append(state)

        done = False
        i = 0
        while not done:
            i += 1
            #if state is None: break
            action = select_action(Q_2, action_space_size, state, eps)

            #TODO take action here, calculate rewards, check if done, refresh state variable
            #state, reward, done, _ = env.step(action)

            if action==0:
                Com.action_forward()
            elif action==1:
                Com.action_right()
            else:
                Com.action_left()


            state  = Agent.get_state_discrete(laser_scan_state_type=laser_scan_state_type_atual, theta=theta_atual)
            #print(state)
            #reward, _, _, _ = Agent.get_reward(number_iterations=i)
            reward, new_distance, r1, r4 = Agent.get_reward(number_iterations=i)
            done,status_done = Agent.is_done(number_iterations=i,max_iterations=horizon, reach_dist=0.5)


            # hist_dict = {'pos':{}, 'scan':{}, 'rewards':{}, 'rates':{}, 'state':{}, 'epresult':{}}
            if i==1:
                hist_dict['pos'][episode+1] = [Agent.Pos]
                hist_dict['scan'][episode+1] = [Agent.laser_scan]
                hist_dict['state'][episode+1] = [state]
            else:
                hist_dict['pos'][episode+1].append(Agent.Pos)
                hist_dict['scan'][episode+1].append(Agent.laser_scan)
                hist_dict['state'][episode+1].append(state)

            if i > horizon:
                done = True

            if (episode % (measure_step) == 0) and i%5==0:
                if i<=5:
                    str_hora_agr = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]
                    print('Epsisode '+str(episode)+' @'+str_hora_agr)

                str_hora_agr = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]
                print('i: '+str(i)+' \tr:'+str([reward, new_distance, r1, r4]))
            # render the environment if render == True
            #if render and episode % render_step == 0:
            #    env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)


        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):

                pause_physics_client() # pause simulation to train
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)
                unpause_physics_client() # unpause simulation after training
                reset_simulation()

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)
            # print(Agent.goal)

        # update learning rate and eps
        scheduler.step()
        eps = max(eps*eps_decay, eps_min)

        # checkpoints @ every 3k episodes
        if episode % checkpoint_inter ==0:

            str_hora_agr = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]
            path = '/media/nero-ia/ADATA UFD/sim_data/wsh_'+str(laser_scan_state_type_atual)+str(n_sectors)+'_'+str_hora_inicio_treino
            #path ='./checkpoints/'+'kv_wsh_'+str(laser_scan_state_type_atual)+str(n_sectors)+'_'+str_hora_inicio_treino

            if not os.path.exists(path):
                os.makedirs(path)

            filename=path+'/wsh_'+str_hora_agr+'.out'
            my_shelf = shelve.open(filename,flag = 'n') # 'n' for new
            for key in dir():
                try:
                    my_shelf['hist_dict'] = hist_dict
                    my_shelf[key] = globals()[key]
                except:
                    #
                    # __builtins__, my_shelf, and imported modules can not be shelved.
                    #
                    print('ERROR shelving: {0}'.format(key))
            my_shelf.close()
            print('checkpoint successfull @'+str_hora_agr)



    return Q_1, performance

if __name__ == '__main__':


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

    Agent.get_state_discrete()
    Agent.get_reward(1)
    rospy.sleep(1)
    Agent.get_state_discrete()
    Agent.get_reward(1)
    rospy.sleep(1)
    Agent.get_state_discrete()
    Agent.get_reward(1)
    reset_simulation()

    time.sleep(1)
    if len(Agent.state)>0 and len(Agent.past_state)>0:
        rospy.loginfo('States Initialized Succesfuly')
    else:
        rospy.loginfo('States not received!!')

    max_episodes= 10000
    checkpoint_inter= 500
    max_iterations = 60
    # action_time=0.2
    # memory_capacity=15000

    #[ (46,'mean'),(46,'min'),(46,'mode')
     #   ,(36.1,'mean'),(36.1,'min'),(36.1,'mode')
      #  ,(30.1,'mean'),(30.1,'min'),(30.1,'mode')
       # ,(18.1,'mean'),(18.1,'min'),(18.1,'mode')]

    for theta_atual,laser_scan_state_type_atual in [
        (46,'min')
        ,(30.1,'mean')
        ,(30.1,'min')
        ,(18.1,'mode')


        ,(46,'mean')
        ,(46,'mode')

        ,(30.1,'mode')

        ,(18.1,'mean')
        ,(18.1,'min')

        ,(36.1,'min')
        ,(36.1,'mean')
        ,(36.1,'mode')
        ]:

        print('-'*120)
        print('''
            Started training for:
                \tlaser_scan_state_type_atual = {laser_scan_state_type_atual}
                \ttheta_atual = {theta_atual}'
        '''.format(laser_scan_state_type_atual=laser_scan_state_type_atual,theta_atual=theta_atual))

        theta_atual = theta_atual # 46, 36.1, 30.1, 18.1
        laser_scan_state_type_atual = laser_scan_state_type_atual # min, mean, mode
        hist_dict = {'pos':{}, 'scan':{}, 'rewards':{}, 'rates':{}, 'state':{}, 'epresult':{}}

        n_sectors ={
                46:4+1,
                36.1:5+1,
                30.1:6+1,
                18.1:10+1
            }[theta_atual]-1

        path = '/media/nero-ia/ADATA UFD/sim_data/wsh_'+str(laser_scan_state_type_atual)+str(n_sectors)+'_'+str_hora_inicio_treino
        #path ='./checkpoints/'+'kv_wsh_'+str(laser_scan_state_type_atual)+str(n_sectors)+'_'+str_hora_inicio_treino

        if not os.path.exists(path):
            os.makedirs(path)

        Q_1, performance = main(
            state_space_size={
                46:4+1,
                36.1:5+1,
                30.1:6+1,
                18.1:10+1
            }[theta_atual],
            theta_atual = theta_atual,
            laser_scan_state_type_atual = laser_scan_state_type_atual,
            checkpoint_inter=checkpoint_inter,
            horizon= max_iterations,
            measure_step=100,
            measure_repeats= 10,
            horizon_eval=25,
            num_episodes=max_episodes, eps = 0.8, eps_decay=0.99975) #visualize eps decay=> geogebra 80*0.99975^(1000 x)

        str_hora_agr = str(datetime.now()).replace(' ','_').replace(':','').replace('-','')[0:15]
        path = '/media/nero-ia/ADATA UFD/sim_data/wsh_'+str(laser_scan_state_type_atual)+str(n_sectors)+'_'+str_hora_inicio_treino
        #path = './checkpoints/'+'kv_wsh_'+str(laser_scan_state_type_atual)+str(n_sectors)+'_'+str_hora_inicio_treino

        if not os.path.exists(path):
            os.makedirs(path)

        filename=path+'/wsh_'+str_hora_agr+'.out'
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