import gym
import torch
import random
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

env = AmazingGameEnv()
agent = DQNAgent(env)

episodes = 500

clock = pygame.time.Clock()  # Create a Clock object to control the frame rate

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        clock.tick(60)  # Limit to 60 frames per second


        #print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")


    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

torch.save(agent.model.state_dict(), 'dqn_model.pth')

