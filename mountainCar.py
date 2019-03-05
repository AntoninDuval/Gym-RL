
# coding: utf-8

# In[1]:


import random
import math
import os.path

import numpy as np
import pandas as pd
import gym
import time
env = gym.make('MountainCar-v0')
actions = [0,1,2]
NUM_BUCKETS = (6, 8)  # (x, x', theta, theta')
NUM_ACTIONS = env.action_space.n # (left, right)
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1


# In[2]:


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))


# In[3]:


class QLearningTable:
    def __init__(self, actions,reward_decay=0.99):
        self.actions = actions  # a list
        self.gamma = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, explore_rate):
        self.check_state_exist(observation)       
        if np.random.uniform() > explore_rate:
            # choose best action
            state_action = self.q_table.loc[observation, :]              
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)          
        return action

    def learn(self, s, a, r, s_,lr):
        self.check_state_exist(s_)
        self.check_state_exist(s)       
        q_predict = self.q_table.loc[s, a]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r #si la partie se fini, on donne un full reward
            
        # update
        self.q_table.loc[s, a] += lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


# In[4]:


env = gym.make('MountainCar-v0')
qlearn = QLearningTable(actions=list(range(len(actions))))
learning_rate = get_learning_rate(0)
explore_rate = get_explore_rate(0)

for episode in range(200):
    
    obv = env.reset()
    previous_state = state_to_bucket(obv)
    
    v_max=obv[1] #vitesse max atteinte
    x_max=obv[0] #distance max atteinte
    
    for t in range(199):
        reward=0
        env.render()       
        #select an action
        rl_action = qlearn.choose_action(str(previous_state),explore_rate)
        #take an action 
        observation, _, done, info = env.step(rl_action)
        
        #observe the result
        bucket=state_to_bucket(observation)
        
        if observation[1]>v_max: #si la voiture dépasse la dernière limite de vitesse, elle a un reward
            v_max=observation[1]
            reward=1
        if observation[0]>x_max: #si la voiture est aller plus loin que la dernière fois, elle gagne un reward
            x_max=observation[0]
            reward=1
        
        
        previous_state=bucket
        
        if done:
            reward=100
            print('YEAH FINISH LINE')
            qlearn.learn(str(previous_state), rl_action, reward,'terminal',learning_rate)
            break
            
        #learn from previous action
        qlearn.learn(str(previous_state), rl_action, reward,str(bucket),learning_rate)        
        previous_state=bucket

    # Update parameters
    explore_rate = get_explore_rate(episode)
    learning_rate = get_learning_rate(episode)
    print("Episode ",episode," finished after {} timesteps".format(t+1))
            
env.close()


# In[5]:


qlearn.q_table


# In[6]:


'''env = gym.make('MountainCar-v0')
env.reset()
done=False
env.render() 
while not done:
        env.render()
        reward=0    
        #select an action
        rl_action = qlearn.choose_action(str(previous_state),0.01)
        #take an action 
        observation, _, done, info = env.step(rl_action)
        
        #observe the result
        bucket=state_to_bucket(observation)
env.close()
'''

