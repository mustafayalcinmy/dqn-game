import torch
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

def load_agent(env, model_path):
    agent = DQNAgent(env)
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0 # Set epsilon to 0 to ensure deterministic behavior
    return agent

def play_game(env, agent):
    clock = pygame.time.Clock()
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

        clock.tick(60)

    print(f"State: {state}, Action: {action}, Reward: {reward}, Done: {done}")
    env.close()

if __name__ == "__main__":
    env = AmazingGameEnv()
    model_path = './models/dqn_model_2500.pth'
    agent = load_agent(env, model_path)
    play_game(env, agent)
