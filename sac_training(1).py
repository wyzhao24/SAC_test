import gym
import numpy as np
import torch
from sac_agent_test import SACAgent
import os
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


senario='Pendulum-v1'
env = gym.make(id=senario)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MEMERY_SIZE = 1000000

agent=SACAgent(state_dim=STATE_DIM, action_dim = ACTION_DIM,memory_capacity=MEMERY_SIZE,
                 alpha=3e-4,beta=3e-4,gamma=0.99,tau=0.005,
                 layer1_dim=256,layer2_dim=256,batch_size=256*2) ##

NUM_EPISODES = 100*5
NUM_STEPS = 200
reward_buffer=[]
best_reward = -np.inf

current_dir = os.path.dirname(os.path.abspath(__file__))
model=current_dir+'/model/'
timestamp=time.strftime("%Y%m%d-%H%M%S")
PLOT=True

for episode in range(NUM_EPISODES):
    state = env.reset()
    eposide_reward=0
    for step in range(NUM_STEPS):
        action = agent.select_action(state) ##
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, done) ##
        eposide_reward+=reward
        state = next_state
        agent.update() ##
        if done:
            break
    reward_buffer.append(eposide_reward)
    avg_reward = np.mean(reward_buffer)

    #save model
    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(agent.actor.state_dict(),model+f'best_actor_{timestamp}.pth')
        print(f'...saved best actor model with reward {best_reward}')
    
    print(f'episode: {episode}, reward: {eposide_reward}, avg_reward: {avg_reward}')


env.close()

if PLOT:
    plt.plot(np.arange(len(reward_buffer)), reward_buffer,color='blue',alpha=0.5,label='reward')
    plt.plot(np.arange(len(reward_buffer)),gaussian_filter1d(reward_buffer, sigma=5),color='blue',linewidth=2)
    plt.title('SAC')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.savefig(f'sac_{timestamp}.png')
    plt.show()