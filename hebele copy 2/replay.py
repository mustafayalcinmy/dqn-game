import torch
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

# Configuration for the DQN agent
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

def load_agent(env, model_path, config):
    """
    Load a trained DQN agent with a configuration dictionary.
    """
    agent = DQNAgent(env, config)
    agent.model.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.epsilon = 0  # Set epsilon to 0 to ensure deterministic behavior
    print(f"Loaded model from {model_path}")
    return agent

def play_game(env, agent, render_fps=60):
    """
    Play a single game using the trained agent.
    """
    clock = pygame.time.Clock()
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        clock.tick(render_fps)

    print(f"Game Over! Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    env = AmazingGameEnv()
    model_path = './models/dqn_model_best_2740.00.pth'
    
    # Load the agent with the configuration
    agent = load_agent(env, model_path, config)
    
    # Play the game
    play_game(env, agent)
