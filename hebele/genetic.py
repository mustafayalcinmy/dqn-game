import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from amazing_game_env import AmazingGameEnv

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HybridDQNAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = []
        self.memory_size = 10000

        self.model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q-value updates
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss and optimization
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evolve(self, population_size=10, mutation_rate=0.1, elite_fraction=0.2):
        # Flatten weights
        def flatten_weights(model):
            return torch.cat([param.view(-1) for param in model.parameters()])

        def unflatten_weights(model, flat_weights):
            pointer = 0
            for param in model.parameters():
                num_params = param.numel()
                param.data = flat_weights[pointer:pointer + num_params].view(param.shape)
                pointer += num_params

        # Evaluate fitness
        def evaluate(individual):
            unflatten_weights(self.model, individual)
            total_reward = 0
            for _ in range(3):  # Evaluate over 3 episodes
                state = self.env.reset()
                done = False
                while not done:
                    action = self.act(state)
                    state, reward, done, _ = self.env.step(action)
                    total_reward += reward
            return total_reward

        # Initialize population
        current_weights = flatten_weights(self.model)
        population = [current_weights + torch.randn_like(current_weights) * 0.1 for _ in range(population_size)]

        # Evaluate population
        scores = [(ind, evaluate(ind)) for ind in population]
        scores.sort(key=lambda x: x[1], reverse=True)
        elites = [x[0] for x in scores[:int(elite_fraction * population_size)]]

        # Generate next generation
        next_population = elites[:]
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(elites, 2)
            crossover_point = random.randint(0, len(parent1) - 1)
            child = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))
            if random.random() < mutation_rate:
                mutation_indices = torch.randint(0, len(child), (int(len(child) * 0.1),))
                child[mutation_indices] += torch.randn(len(mutation_indices)) * 0.1
            next_population.append(child)

        # Set weights of the best individual
        best_weights = scores[0][0]
        unflatten_weights(self.model, best_weights)
        self.update_target_model()

