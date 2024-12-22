import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from gameen import AmazingGameEnv
import gym

# Import your environment
env = AmazingGameEnv()

# Hyperparameters
GAMMA = 0.99  # Discount factor
LR = 1e-3  # Learning rate
BATCH_SIZE = 64  # Batch size for training
MEMORY_SIZE = 10000  # Replay buffer size
TARGET_UPDATE = 10  # Update the target network every 10 episodes
EPSILON_START = 1.0  # Initial epsilon for epsilon-greedy strategy
EPSILON_END = 0.01  # Minimum epsilon
EPSILON_DECAY = 500  # Decay rate for epsilon
MAX_EPISODES = 50
MAX_STEPS = 200

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

# Initialize environment and network
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network is not trained directly

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Epsilon-greedy strategy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).item()  # Exploit

# Train the Q-network
def train():
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Compute current Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q-values
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # Loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main training loop
epsilon = EPSILON_START
for episode in range(MAX_EPISODES):
    state = env.reset()
    total_reward = 0

    for t in range(MAX_STEPS):
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        train()

        if done:
            break

    # Decay epsilon
    epsilon = max(EPSILON_END, EPSILON_START - episode / EPSILON_DECAY)

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
