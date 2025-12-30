import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

class ReplayMemory:
    def __init__(self, capacity,state_dim, action_dim):
        self.capacity = capacity
        self.state_memory=np.zeros((self.capacity, state_dim))
        self.action_memory=np.zeros((self.capacity, action_dim))
        self.reward_memory=np.zeros(self.capacity)
        self.next_state_memory=np.zeros((self.capacity, state_dim))
        self.done_memory=np.zeros(self.capacity)
        self.counter = 0

    def add_memory(self, state, action, reward, next_state, done):
        index=self.counter%self.capacity
        self.state_memory[index]=state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.next_state_memory[index]=next_state
        self.done_memory[index]=done
        self.counter+=1
        

    def sample(self, batch_size):
        current_batch_size = min(batch_size, self.counter)
        batch = np.random.choice(self.counter, current_batch_size, replace=False)
        batch_state=self.state_memory[batch]
        batch_action=self.action_memory[batch]
        batch_reward=self.reward_memory[batch]
        batch_next_state=self.next_state_memory[batch]
        batch_done=self.done_memory[batch]
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

class CriticNetwork(nn.Module):
        def __init__(self, beta,state_dim, action_dim,fc1_dim,fec2_dim):
            super(CriticNetwork, self).__init__()
            self.state_dim=state_dim
            self.action_dim=action_dim
            self.fc1_dim=fc1_dim
            self.fec2_dim=fec2_dim
            
            self.fc1=nn.Linear(self.state_dim+self.action_dim, self.fc1_dim)
            self.fc2=nn.Linear(self.fc1_dim, self.fec2_dim)
            self.q3=nn.Linear(self.fec2_dim, 1)
            
            self.optimizer=optim.Adam(self.parameters(), lr=beta)

        def forward(self, state, action):
            x=F.relu(self.fc1(torch.cat([state, action], 1)))
            #假设state.shape  = [64, 10]，action.shape = [64, 3]
            #则torch.cat([state, action], dim=1)得到x.shape = [64, 13]

            x=F.relu(self.fc2(x))
            q=self.q3(x)
            return q
        
class ValueNetwork(nn.Module):
        def __init__(self, beta,state_dim,fc1_dim,fec2_dim):
            super(ValueNetwork, self).__init__()
            self.state_dim=state_dim
            self.fc1_dim=fc1_dim
            self.fec2_dim=fec2_dim
            
            self.fc1=nn.Linear(self.state_dim, self.fc1_dim)
            self.fc2=nn.Linear(self.fc1_dim, self.fec2_dim)
            self.v=nn.Linear(self.fec2_dim, 1)
            
            self.optimizer=optim.Adam(self.parameters(), lr=beta)

        def forward(self, state,):
            x=F.relu(self.fc1(torch.cat([state], 1)))
            x=F.relu(self.fc2(x))
            v=self.v(x)
            return v

class ActorNetwork(nn.Module):
        def __init__(self, alpha,state_dim, action_dim,fc1_dim,fc2_dim,max_action):
            super(ActorNetwork, self).__init__()
            self.state_dim=state_dim
            self.action_dim=action_dim
            self.fc1_dim=fc1_dim
            self.fc2_dim=fc2_dim
            self.max_action=max_action
            self.fc1=nn.Linear(self.state_dim, self.fc1_dim)
            self.fc2=nn.Linear(self.fc1_dim, self.fc2_dim)
            self.mu=nn.Linear(self.fc2_dim, self.action_dim)##√
            self.sigma=nn.Linear(self.fc2_dim, self.action_dim)###√
            #这里mu和sigma都是策略化参数，mu是均值，sigma是方差
            #输出维度为action_dim，即动作的维度，这是因为即使动作是连续的，每个动作分量都遵循各自的分布
            #举个例子，电网中[Pg, Qg]2个动作，Pg和Qg是连续的，但是他们的分布是不同的
            self.optimizer=optim.Adam(self.parameters(), lr=alpha)

            self.tiny_positive=1e-6

        def forward(self, state):
            x=F.relu(self.fc1(state))
            x=F.relu(self.fc2(x))
            mu=self.mu(x)
            sigma=self.sigma(x)
            sigma=F.softplus(sigma)+self.tiny_positive
            sigma=torch.clamp(sigma, min=self.tiny_positive, max=1)

            return mu, sigma
        
        def sample_normal(self, state,reparameterize):
            mu, sigma=self.forward(state)
            probability=Normal(mu, sigma)

            if reparameterize:
                raw_action=probability.rsample()
            else:
                raw_action=probability.sample()

            tanh_action=torch.tanh(raw_action)
            scaled_action=tanh_action*self.max_action
            log_prob=probability.log_prob(raw_action)
            log_prob-=torch.log(1-tanh_action.pow(2)+self.tiny_positive)
            
            if log_prob.dim()==1:
                log_prob=log_prob.unsqueeze(0)
            log_prob=log_prob.sum(1, keepdim=True)

            return scaled_action, log_prob



class SACAgent():
    def __init__(self, state_dim, action_dim,memory_capacity,
                 alpha,beta,gamma,tau,
                 layer1_dim,layer2_dim,batch_size):
        self.memory_capacity=ReplayMemory(capacity=memory_capacity,state_dim=state_dim, action_dim=action_dim)
        self.gamma=gamma
        self.tau=tau
        self.batch_size=batch_size
        self.critic1=CriticNetwork(beta=beta,state_dim=state_dim, 
                                   action_dim=action_dim,fc1_dim=layer1_dim,
                                   fec2_dim=layer2_dim).to(device)
        self.critic2=CriticNetwork(beta=beta,state_dim=state_dim, 
                                   action_dim=action_dim,fc1_dim=layer1_dim,
                                   fec2_dim=layer2_dim).to(device)
        self.value=ValueNetwork(beta=beta,state_dim=state_dim,
                                        fc1_dim=layer1_dim,fec2_dim=layer2_dim).to(device)
        self.target_value=ValueNetwork(beta=beta,state_dim=state_dim,
                                        fc1_dim=layer1_dim,fec2_dim=layer2_dim).to(device)
        self.actor=ActorNetwork(alpha=alpha,state_dim=state_dim,
                                        action_dim=action_dim,fc1_dim=layer1_dim,
                                        fc2_dim=layer2_dim,max_action=2).to(device)
        
    def select_action(self, state):
            state=torch.tensor(state, dtype=torch.float).to(device)
            action,_=self.actor.sample_normal(state,reparameterize=False)
            return action.cpu().detach().numpy()
        
    def store_transition(self, state, action, reward, next_state, done):
            self.memory_capacity.add_memory(state, action, reward, next_state, done)

    def update(self):
            if self.memory_capacity.counter < self.batch_size:
                return
            state,action,reward,next_state,done=self.memory_capacity.sample(self.batch_size)
            state=torch.tensor(state, dtype=torch.float).to(device)
            action=torch.tensor(action, dtype=torch.float).to(device)
            reward=torch.tensor(reward, dtype=torch.float).to(device)
            next_state=torch.tensor(next_state, dtype=torch.float).to(device)
            done=torch.tensor(done, dtype=torch.bool).to(device)

            value=self.value(state).view(-1)
            with torch.no_grad():
                value_=self.target_value(next_state).view(-1)
                value_[done]=0.0

            actions,log_probs=self.actor.sample_normal(state,reparameterize=False)
            log_probs=log_probs.view(-1)

            q1=self.critic1.forward(state,actions)
            q2=self.critic2.forward(state,actions)
            q=torch.min(q1,q2)
            q=q.view(-1)
            self.value.optimizer.zero_grad()
            value_target=q-log_probs

            critic_loss=F.mse_loss(value_target,value)
            critic_loss.backward()
            self.value.optimizer.step()

            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            actions,log_probs=self.actor.sample_normal(state,reparameterize=True)
            log_probs=log_probs.view(-1)
            q1=self.critic1.forward(state,actions)
            q2=self.critic2.forward(state,actions)
            q=torch.min(q1,q2)
            q=q.view(-1)
            actor_loss=log_probs-q
            actor_loss=torch.mean(actor_loss)
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()


            with torch.no_grad():
                q_hat=reward+self.gamma*value_
            q1_old_policy=self.critic1.forward(state,action).view(-1)
            q2_old_policy=self.critic2.forward(state,action).view(-1)

            critic1_loss=F.mse_loss(q1_old_policy,q_hat)
            critic2_loss=F.mse_loss(q2_old_policy,q_hat)

            self.critic1.optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1.optimizer.step()

            self.critic2.optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2.optimizer.step()