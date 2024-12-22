import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict
from amazing_game_env import AmazingGameEnv
from experience_replay import ReplayMemory

class DQNAgent:
    def __init__(self, env: AmazingGameEnv, config: Dict):
        self.env = env
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.memory = ReplayMemory(config['memory_size'])
        self.soft_update_tau = config['soft_update_tau']

        # Networks
        self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return self.policy_net(state).argmax().item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q-Learning target
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss and optimization
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of target network
        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.soft_update_tau * local_param.data + (1.0 - self.soft_update_tau) * target_param.data)

    def save(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())
