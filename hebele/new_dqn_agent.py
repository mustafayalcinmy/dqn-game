import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_dim, action_dim, memory_capacity=10000, gamma=0.99, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state, evaluate=False):
        """Select an action using epsilon-greedy policy. If evaluate=True, always use the best action."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-values for current states
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        """Copy weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save the model's parameters and optimizer state."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load the model's parameters and optimizer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        print(f"Model loaded from {filepath}")
