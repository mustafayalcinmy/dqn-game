import gym
from gym import spaces
import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class AmazingGameEnv(gym.Env):
    def __init__(self):
        super(AmazingGameEnv, self).__init__()
        pygame.init()
        pygame.font.init()  # Initialize the font module
        self.SCREEN_WIDTH = 405
        self.SCREEN_HEIGHT = 720
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -float("inf")]),
            high=np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, float("inf")]),
            dtype=np.float32
        )
        
        self.colors = {
            "bg": (40, 40, 40),
            "cube": (235, 219, 180),
            "obstacle": (204, 36, 29)
        }
        self.cube = {"x": 100, "y": 400, "size": 20, "speed": 0, "gravity": 350, "jump": -250, "xSpeed": 0}
        self.camera = {"y": 0, "speed": 100}
        self.gapWidth = 100
        self.blockHeight = 10
        self.obstacleFrequency = 300
        self.gaps = []
        self.score = 0
        self.running = True
        
        self.seed()  # Initialize the random seed
        self.generate_obstacles()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def generate_obstacles(self):
        self.gaps = []
        startY = self.cube["y"] + 200
        for i in range(5):
            blockY = startY - i * self.obstacleFrequency
            gapX = random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": gapX, "y": blockY, "passed": False})

    def reset(self):
        self.cube["x"] = 100
        self.cube["y"] = 400
        self.cube["speed"] = 0
        self.cube["xSpeed"] = 0
        self.camera["y"] = 0
        self.score = 0
        self.generate_obstacles()
        return self.get_state()

    def get_state(self):
        gap = self.gaps[0] if self.gaps else {"x": self.SCREEN_WIDTH / 2, "y": 0}
        return np.array([self.cube["x"], self.cube["y"], gap["x"], gap["y"]], dtype=np.float32)

    def step(self, action):
        """
        Perform the action and update the game state.
        Action: 0 = Move left and jump, 1 = Move right and jump, 2 = Do nothing.
        """
        if action == 0:  # Move left and jump
            self.cube["xSpeed"] -= self.cube["gravity"] * 0.02
            self.cube["speed"] = self.cube["jump"]  # Apply jump force
        elif action == 1:  # Move right and jump
            self.cube["xSpeed"] += self.cube["gravity"] * 0.02
            self.cube["speed"] = self.cube["jump"]  # Apply jump force
        elif action == 2:  # Do nothing
            pass

        # Update cube and environment
        self.update_cube(1 / 60)
        self.update_camera(1 / 60)
        self.update_obstacles()

        # Check for scoring
        for gap in self.gaps:
            # If the cube passes a gap and it hasn't been counted yet
            if not gap.get("passed") and self.cube["y"] > gap["y"] + self.blockHeight:
                gap["passed"] = True
                self.score += 1  # Increment score for fully passing a gap

        # Get the updated state
        state = self.get_state()

        # Reward and done conditions
        reward = self.score  # Reward is proportional to the score
        done = self.cube["y"] > self.SCREEN_HEIGHT or self.cube["y"] < 0  # Game over conditions
        if done:
            reward = -10  # Penalize game over

        return state, reward, done, {}


    def update_cube(self, dt):
        self.cube["speed"] += self.cube["gravity"] * dt
        self.cube["y"] += self.cube["speed"] * dt
        self.cube["x"] += self.cube["xSpeed"] * dt
        self.cube["x"] = max(0, min(self.cube["x"], self.SCREEN_WIDTH - self.cube["size"]))
        self.running = True  # Set running to true

        for gap in self.gaps:
            if self.cube["y"] + self.cube["size"] > gap["y"] and self.cube["y"] < gap["y"] + self.blockHeight:
                if self.cube["x"] + self.cube["size"] < gap["x"] or self.cube["x"] > gap["x"] + self.gapWidth:
                    self.running = False

        if self.cube["y"] > self.SCREEN_HEIGHT or self.cube["y"] < 0:
            self.running = False

    def update_camera(self, dt):
        targetY = self.cube["y"] - self.SCREEN_HEIGHT / 2 + self.cube["size"] / 2
        self.camera["y"] += (targetY - self.camera["y"]) * min(self.camera["speed"] * 0.01 * dt, 1)

    def update_obstacles(self):
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera["y"] + self.SCREEN_HEIGHT]
        while len(self.gaps) < 5:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacleFrequency
            newGapX = random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": newGapX, "y": newY, "passed": False})

    def render(self, mode="human"):
        if mode == "human":
            if not hasattr(self, "window"):
                self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Amazing Game")

            self.screen.fill(self.colors["bg"])

            # Draw the cube
            pygame.draw.rect(self.screen, self.colors["cube"], 
                            (self.cube["x"], self.cube["y"] - self.camera["y"], self.cube["size"], self.cube["size"]))

            # Draw gaps/obstacles
            for gap in self.gaps:
                pygame.draw.rect(self.screen, self.colors["obstacle"], 
                                (0, gap["y"] - self.camera["y"], gap["x"], self.blockHeight))
                pygame.draw.rect(self.screen, self.colors["obstacle"], 
                                (gap["x"] + self.gapWidth, gap["y"] - self.camera["y"], 
                                self.SCREEN_WIDTH - gap["x"] - self.gapWidth, self.blockHeight))

            # Display the score
            font = pygame.font.Font(None, 36)
            score_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_surface, (10, 10))

            # Blit the surface onto the display window
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()

            # Handle player input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                return 0  # Move left
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                return 1  # Move right
            elif keys[pygame.K_SPACE]:
                self.cube["speed"] = self.cube["jump"]  # Jump
            else:
                return 2  # Do nothing

    def close(self):
        pygame.quit()

        
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

def train_dqn(env, num_episodes=500, batch_size=32, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=10000)

    epsilon = epsilon_start

    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randrange(output_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                return policy_net(state).argmax().item()

    def optimize_model():
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        max_next_q_values = target_net(next_states).max(1)[0]
        expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(current_q_values, expected_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            optimize_model()
            if done:
                target_net.load_state_dict(policy_net.state_dict())
                break

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return policy_net

# Save the DQN agent
def save_model(agent, path="dqn_model.pth"):
    torch.save(agent.state_dict(), path)
    print(f"Model saved to {path}")

# Train the DQN agent
env = AmazingGameEnv()
dqn_agent = train_dqn(env, num_episodes=50)

# Save the trained model
save_model(dqn_agent)

# Use the trained DQN agent to play the game
def dqn_play(env, agent, path="dqn_model.pth"):
    agent.load_state_dict(torch.load(path))
    clock = pygame.time.Clock()
    state = env.reset()
    total_reward = 0
    while True:
        state = torch.FloatTensor(state).unsqueeze(0)
        action = agent(state).argmax().item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        env.render()
        clock.tick(60)  # Add frame rate control here
        if done:
            break
    env.close()
    print(f"Total Reward: {total_reward}")

# Play the game using the trained agent
dqn_play(env, dqn_agent)
