import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from cube_game_env import CubeGameEnv

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

env = CubeGameEnv()
input_dim = len(env.get_observation())
action_dim = 3
model = DQN(input_dim, action_dim)
target_model = DQN(input_dim, action_dim)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return model(state).argmax().item()

for episode in range(500):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon=0.1)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if episode % 10 == 0:
            env.render()

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            q_values = model(states).gather(1, actions)
            with torch.no_grad():
                max_next_q_values = target_model(next_states).max(1, keepdim=True)[0]
                targets = rewards + gamma * max_next_q_values * (1 - dones)

            loss = loss_fn(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")
