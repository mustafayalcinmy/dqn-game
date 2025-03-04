import os
import argparse
import torch
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgent

class TrainingConfig:
    def __init__(self):
        self.episodes = 1650
        self.min_exploration_episodes = 350
        self.max_decay_episodes = 1200
        self.save_interval = 200
        self.epsilon_min = 0.01
        self.render = False
        self.model_dir = "./models"
        self.default_model_name = "dqn_model"
        self.target_fps = 60

def save_model(agent, model_path, episode=None, best=False):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.model.state_dict(), model_path)
    if episode:
        print(f"Model saved at episode {episode} as '{os.path.basename(model_path)}'")
    if best:
        print(f"New best model saved as '{os.path.basename(model_path)}'")

def training_loop(env, agent, config):
    best_reward = float('-inf')
    pygame.init()  
    
    try:
        for episode in range(config.episodes):
            state = env.reset()
            done = False
            total_reward = 0

            if episode < config.min_exploration_episodes:
                agent.epsilon = 1.0
            else:
                decay_episodes = max(1, config.max_decay_episodes)
                progress = (episode - config.min_exploration_episodes) / decay_episodes
                agent.epsilon = max(config.epsilon_min, 1.0 - progress)

            while not done:
                if config.render:
                    env.render()
                    pygame.event.pump()  

                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)

                agent.remember(state, action, reward, next_state, done)
                agent.replay()

                state = next_state
                total_reward += reward

                if config.render:
                    pygame.time.Clock().tick(config.target_fps)

            model_base = f"{config.default_model_name}_{episode+1}.pth"
            
            if total_reward > best_reward:
                best_reward = total_reward
                save_model(agent, 
                         os.path.join(config.model_dir, f"best_{model_base}"),
                         best=True)

            if (episode + 1) % config.save_interval == 0:
                save_model(agent,
                         os.path.join(config.model_dir, f"checkpoint_{model_base}"),
                         episode=episode+1)

            print(f"Episode {episode+1}/{config.episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.4f}")

    finally:
        save_model(agent, 
                 os.path.join(config.model_dir, f"{config.default_model_name}_final.pth"))
        env.close()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    parser.add_argument('--episodes', type=int, help='Total training episodes')
    parser.add_argument('--model-dir', type=str, help='Output directory for models')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    config = TrainingConfig()
    
    if args.episodes:
        config.episodes = args.episodes
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.render:
        config.render = True
    if args.debug:
        config.save_interval = 100  
        config.target_fps = 30  

    env = AmazingGameEnv()
    agent = DQNAgent(env)
    
    try:
        print(f"Starting training with config:\n"
            f"Episodes: {config.episodes}\n"
            f"Exploration episodes: {config.min_exploration_episodes}\n"
            f"Epsilon decay episodes: {config.max_decay_episodes}\n"
            f"Save interval: {config.save_interval}")
        
        training_loop(env, agent, config)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final model...")
    except Exception as e:
        print(f"Training failed: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    main()