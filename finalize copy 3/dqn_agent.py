import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from typing import Tuple, Dict
from amazing_game_env import AmazingGameEnv
from dqn import DQN
from experience_replay import ReplayMemory
import itertools

config = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.05,
    'learning_rate': 0.001,
    'batch_size': 16,
    'memory_size': 100000,
    'soft_update_tau': 0.01  # Soft update parameter
}


class DQNAgent:        
    def __init__(self, config: Dict):
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.batch_size = config['batch_size']
        self.memory = deque(maxlen=config['memory_size'])

    def run(self, is_training = True, render= False):
        env = AmazingGameEnv()
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        
        epsilon_history = []
        total_rewards_for_episode = []
        
        policy_dqn = DQN(num_states, num_actions)
        
        if is_training:
            memory = ReplayMemory(10000)
            epsilon = self.epsilon

            step_count = 0
 
        
        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)

            done = False
            total_rewards = 0.0
            
            while not done:
                if is_training and random.random() < epsilon:
                    memory.append((state, action, new_state, reward, done))
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                    
                new_state, reward, done, _ = env.step(action.item())
                
                new_state = torch.tensor(new_state, dtype=torch.float)
                total_rewards += reward 
                
                reward = torch.tensor(reward, dtype=torch.float)                
                
                if is_training:
                    memory.append((state, action, new_state, reward, done))
                    step_count += 1
                
                state = new_state
            
            total_rewards_for_episode.append(total_rewards)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)
            
            if len(memory)>self.batch_size:
                batch = memory.sample(self.batch_size)
                self.optimize(batch, policy_dqn, tar)
            
    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
