import gym
import torch
import random
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

# Configuration dictionary
config = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.05,
    'learning_rate': 0.001,
    'batch_size': 256,
    'memory_size': 200000,
    'update_target_steps': 2000
}

env = AmazingGameEnv()
agent = DQNAgent(env, config)
clock = pygame.time.Clock()  # Create a Clock object to control the frame rate

# Training hyperparameters
training_config = {
    'episodes': 2000,
    'min_exploration_episodes': 500,
    'max_decay_episodes': 1400,
    'render': False,
    'save_interval': 100,
    'log_file': "training_log.txt"
}

# Load hyperparameters
episodes = training_config['episodes']
min_exploration_episodes = training_config['min_exploration_episodes']
max_decay_episodes = training_config['max_decay_episodes']
render = training_config['render']
save_interval = training_config['save_interval']
log_file = training_config['log_file']

# Clear previous logs
with open(log_file, "w") as f:
    f.write("Episode, Total Reward, Epsilon, Best Reward\n")

best_reward = float('-inf')  # Initialize to negative infinity

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # Ensure epsilon follows the custom schedule
    if episode < min_exploration_episodes:
        agent.epsilon = 1.0  # Full exploration during initial episodes
    else:
        agent.epsilon = max(agent.epsilon_min, 1.0 - (episode - min_exploration_episodes) / max_decay_episodes)

    while not done:
        if render:
            env.render()

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()  # Train on past experiences

        state = next_state
        total_reward += reward

        if render:
            print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")
            clock.tick(60)  # Limit to 60 frames per second

    # Check if this episode's total reward is the best so far
    if total_reward > best_reward:
        best_reward = total_reward
        model_path = f'./models/dqn_model_best_{best_reward:.2f}.pth'
        agent.save(model_path)
        print(f"New best reward {best_reward:.2f} achieved! Model saved as '{model_path}'")

    # Log episode details
    with open(log_file, "a") as f:
        f.write(f"{episode + 1}, {total_reward:.2f}, {agent.epsilon:.4f}, {best_reward:.2f}\n")

    # Print episode stats
    print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Save the model periodically
    if (episode + 1) % save_interval == 0:
        model_path = f'./models/dqn_model_{episode + 1}.pth'
        agent.save(model_path)
        print(f"Model saved at episode {episode + 1} as '{model_path}'")

agent.save('./models/dqn_model_final.pth')
print("Training complete. Final model saved as 'dqn_model_final.pth'")
