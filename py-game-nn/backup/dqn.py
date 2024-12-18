import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from game import reset_game, step, get_state  # Import necessary functions from game.py

# Define the DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 3)  # Output for 3 possible actions (left, right, stay)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Parameters
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
target_update = 10

# Initialize DQN
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=2000)

# Training loop
for episode in range(1000):
    state = reset_game()
    for t in range(200):
        action = np.random.choice(3) if np.random.rand() <= epsilon else np.argmax(policy_net(torch.from_numpy(state).float()).detach().numpy())
        next_state, reward, done = step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = np.array(states)
            print(f"Shape of states: {states.shape}")
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.int64)

            q_values = policy_net(states).gather(1, actions.view(-1, 1))
            next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * next_q_values

            loss = nn.functional.mse_loss(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training completed!")
