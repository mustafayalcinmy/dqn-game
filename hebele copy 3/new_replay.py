from amazing_game_env import AmazingGameEnv
from new_dqn_agent import DQNAgent
import torch
import pygame

def load_agent(env, model_path):
    """
    Load a trained DQN agent with the saved model weights.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # Ensure deterministic behavior
    print(f"Agent loaded from {model_path}")
    return agent

def play_game(env, agent):
    """
    Play the game using the trained DQN agent.
    """
    clock = pygame.time.Clock()
    state = env.reset()
    done = False

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        clock.tick(60)

    print(f"Game over! Final State: {state}, Last Action: {action}, Final Reward: {reward}")
    env.close()

if __name__ == "__main__":
    env = AmazingGameEnv()
    model_path = './models/dqn_model_final.pth'
    agent = load_agent(env, model_path)
    play_game(env, agent)
