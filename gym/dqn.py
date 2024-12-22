import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import pygame
import random
from amazing_game_env import AmazingGameEnv  # Save your game environment in amazing_game_env.py

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Synchronize target network with policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return torch.argmax(action_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-learning update
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reduce epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval()

# Main training loop
if __name__ == "__main__":
    env = AmazingGameEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episodes = 10000
    target_update_freq = 10
    max_steps = 200

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {e + 1}/{episodes}, Score: {total_reward}")
                break

        agent.replay()

        if e % target_update_freq == 0:
            agent.update_target_network()

    # Save the trained model
    agent.save("dqn_trained_model.pth")
    print("Model saved!")

    def render_trained_model():
        state = env.reset()
        done = False
        total_reward = 0
        clock = pygame.time.Clock()  # Initialize the clock for FPS regulation

        while not done:
            env.render()  # Render the environment

            # Get action from the trained model
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_values = agent.policy_net(state_tensor)
            action = torch.argmax(action_values).item()

            # Perform the action in the environment
            state, reward, done, _ = env.step(action)
            total_reward += reward

            clock.tick(60)  # Limit to 60 FPS

        print(f"Final Score: {total_reward}")
        env.close()


    # Load the model and render
    agent.load("dqn_trained_model.pth")
    print("Rendering trained model...")
    render_trained_model()
