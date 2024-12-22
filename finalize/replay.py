import torch
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

def load_agent(env, model_path):
    # Define the configuration for the DQNAgent
    config = {
        'gamma': 0.99,
        'epsilon': 1.0,  # Initial value; will be overridden
        'epsilon_decay': 0.9995,
        'epsilon_min': 0.05,
        'learning_rate': 0.001,
        'batch_size': 64,
        'memory_size': 100000,
        'soft_update_tau': 0.01
    }

    # Initialize the agent with the environment and config
    agent = DQNAgent(env, config)
    agent.load(model_path)  # Use the refactored load method
    agent.epsilon = 0  # Set epsilon to 0 for deterministic behavior during replay
    return agent

def play_game(env, agent):
    clock = pygame.time.Clock()
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

        clock.tick(60)  # Limit the frame rate

    print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")
    env.close()

if __name__ == "__main__":
    env = AmazingGameEnv()
    model_path = './models/dqn_model_best.pth'
    agent = load_agent(env, model_path)
    play_game(env, agent)
