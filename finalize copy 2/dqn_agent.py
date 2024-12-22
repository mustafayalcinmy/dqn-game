import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from typing import Tuple, Dict


class DQNLSTM(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_dim: int):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[1], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the flattened size of convolutional output
        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected layer after the convolutional layers
        self.fc1 = nn.Linear(conv_out_size, 512)

        # LSTM layer
        self.lstm = nn.LSTM(512, 256, batch_first=True)

        # Q-Network (output layer)
        self.fc2 = nn.Linear(256, output_dim)

        self._initialize_weights()

    def _get_conv_out(self, shape: Tuple[int]) -> int:
        """
        Calculate the output size of the convolutional layers.
        Assumes input shape includes (time_steps, channels, height, width).
        """
        channels, height, width = shape[1:]  # Skip time_steps dimension
        dummy_input = torch.zeros(1, channels, height, width)  # Single frame input
        out = self.conv3(self.conv2(self.conv1(dummy_input)))  # Pass through conv layers
        return int(np.prod(out.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DQNLSTM model.
        Expects input shape: (batch_size, time_steps, channels, height, width).
        """
        batch_size, time_steps, c, h, w = x.size()
        x = x.view(batch_size * time_steps, c, h, w)  # Merge batch and time for conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, time_steps, -1)  # Reshape for LSTM
        x = F.relu(self.fc1(x))
        x, _ = self.lstm(x)  # LSTM output
        return self.fc2(x[:, -1, :])  # Use the output from the last time step

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DQNAgentWithLSTM:
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

        # Ensure input shape includes stacked frames
        raw_obs_shape = env.observation_space.shape  # Should be (Height, Width, Channels)
        frame_stack = 4  # Number of frames to stack
        self.input_shape = (frame_stack, raw_obs_shape[2], raw_obs_shape[0], raw_obs_shape[1])  # (time_steps, channels, height, width)
        self.output_dim = env.action_space.n

        # Initialize the model
        self.model = DQNLSTM(self.input_shape, self.output_dim).to(self.device)
        self.target_model = DQNLSTM(self.input_shape, self.output_dim).to(self.device)
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

        # Reshape for LSTM: (batch_size, time_steps, channels, height, width)
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values and targets
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * (1 - dones) * next_q_values

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
