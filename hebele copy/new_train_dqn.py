import gym
import torch
import random
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent
from new_dqn_agent

if __name__ == "__main__":
    env = AmazingGameEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    # Training
    try:
        rewards = train_dqn(env, agent, num_episodes=500, target_update_freq=10)
        agent.save("amazing_game_dqn.pth")  # Save the trained model
    except KeyboardInterrupt:
        print("Training interrupted! Saving model...")
        agent.save("amazing_game_dqn_interrupted.pth")

    # Play the game with the trained model
    agent.load("amazing_game_dqn.pth")
    play_game(env, agent)

    env.close()
