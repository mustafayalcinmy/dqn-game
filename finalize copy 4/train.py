import gym
import torch
import random
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

# Define the configuration for the DQNAgent
config = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.05,
    'learning_rate': 0.001,
    'batch_size': 16,
    'memory_size': 100000,
    'soft_update_tau': 0.01  # Soft update parameter
}

# Initialize environment and agent
env = AmazingGameEnv()
agent = DQNAgent(env, config)
clock = pygame.time.Clock()  # Create a Clock object to control the frame rate

# Hyperparameters
episodes = 6000
min_exploration_episodes = 200  # Minimum number of episodes with full exploration
max_decay_episodes = 1000  # Episodes over which epsilon decays to epsilon_min
render = False
save_interval = 100

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
            clock.tick(120)  # Limit to 60 frames per second

    # Print episode stats
    print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Save the model periodically
    if (episode + 1) % save_interval == 0:
        torch.save(agent.model.state_dict(), f'./models/dqn_model_{episode + 1}.pth')
        print(f"Model saved at episode {episode + 1}")

# Save the final model
torch.save(agent.model.state_dict(), './models/dqn_model_final.pth')
print("Training complete. Final model saved as 'dqn_model_final.pth'")
