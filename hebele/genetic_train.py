import gym
import torch
import random
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from genetic import  HybridDQNAgent

env = AmazingGameEnv()
agent = HybridDQNAgent(env)

episodes = 200
evolve_interval = 20  # Apply evolutionary updates every 20 episodes

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    # Apply evolutionary updates
    if (episode + 1) % evolve_interval == 0:
        agent.evolve()



env.close()
