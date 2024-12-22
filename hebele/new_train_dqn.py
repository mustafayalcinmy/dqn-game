from amazing_game_env import AmazingGameEnv
from new_dqn_agent import DQNAgent
import torch
import pygame

# Initialize the environment
env = AmazingGameEnv()

# Extract dimensions from the environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the agent
agent = DQNAgent(state_dim, action_dim)

# Training parameters
episodes = 1050
min_exploration_episodes = 100
max_decay_episodes = 900
render = False
save_interval = 100

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    # Adjust epsilon
    if episode < min_exploration_episodes:
        agent.epsilon = 1.0
    else:
        agent.epsilon = max(agent.epsilon_min, 1.0 - (episode - min_exploration_episodes) / max_decay_episodes)

    while not done:
        if render:
            env.render()

        # Select action and perform a step
        action = agent.select_action(state)  # Updated to use correct method
        next_state, reward, done, _ = env.step(action)

        # Store experience and train
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

    print(f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    if (episode + 1) % save_interval == 0:
        torch.save(agent.model.state_dict(), f'./models/dqn_model_{episode + 1}.pth')
        print(f"Model saved at episode {episode + 1}")

# Save final model
torch.save(agent.model.state_dict(), './models/dqn_model_final.pth')
print("Training complete. Final model saved as 'dqn_model_final.pth'")
