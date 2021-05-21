import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
from ddpg_agent import DDPG, ReplayBuffer


BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA=0.99


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    def __init__(self, action_size=2, seed=42, n_agents=2):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            n_agents (int): number of distinct agents
        """

        self.n_agents = n_agents
        self.timestep = 0

        self.agents = [DDPG(i) for i in range(n_agents)]
        
        # common buffer for both the agents
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        self.timestep += 1     
        if self.timestep % 2 == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # sample from the replay buffer for each agent
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                self.learn(experiences, GAMMA)

    def act(self, all_states, add_noise=True):
        # calculate each agents action
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise_weight=0.5, add_noise=True)
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) 

    def learn(self, experiences, gamma):
        # each agent uses its own actor to calculate next_actions
        all_next_actions = []
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # extract agent i's state and get action via actor network
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            # extract agent i's next state and get action via target actor network
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)
                       
        # each agent learns from its experience sample
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
            
    def save_agents(self):
        # save models 
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"checkpoint_actor_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_{i}.pth")