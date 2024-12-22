import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from typing import Tuple, Dict


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class DQNAgent:
    def __init__(self, env, config: Dict):
        self.env = env
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.memory = deque(maxlen=config['memory_size'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n

        self.model = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_model = DQN(self.input_dim, self.output_dim).to(self.device)
        self._update_target_network(hard_update=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.soft_update_tau = config['soft_update_tau']

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for the current state-action pairs
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Use the main model to choose the action for the next state
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)

            # Use the target model to evaluate the Q-value of the chosen action
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values
        targets = rewards + self.gamma * (1 - dones) * next_q_values

        # Compute the loss and optimize
        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self._update_target_network()

    def _update_target_network(self, hard_update: bool = False):
        if hard_update:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.soft_update_tau * local_param.data + (1.0 - self.soft_update_tau) * target_param.data)

    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
