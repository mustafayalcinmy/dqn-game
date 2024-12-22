import gym
import torch
import numpy as np
import pygame
from amazing_game_env import AmazingGameEnv
from dqn_agent import DQNAgentWithLSTM

# Configuration for DQNAgentWithLSTM
config = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.05,
    'learning_rate': 0.001,
    'batch_size': 12,
    'memory_size': 1000,
    'soft_update_tau': 0.01  # Soft update parameter
}

# Initialize environment and agent
env = AmazingGameEnv(use_pixels=True)  # Ensure pixel-based observations
agent = DQNAgentWithLSTM(env, config)
clock = pygame.time.Clock()  # Clock object for controlling frame rate

# Training parameters
episodes = 400
min_exploration_episodes = 20  # Full exploration during initial episodes
max_decay_episodes = 350  # Episodes for epsilon decay
render = False  # Set to True for visualizing the game
save_interval = 100
frame_stack = 4  # Number of frames to stack for LSTM

# Preprocess observation space to include frame stacking
raw_obs_shape = env.observation_space.shape  # Shape: (Height, Width, Channels)
state_shape = (frame_stack, raw_obs_shape[2], raw_obs_shape[0], raw_obs_shape[1])  # (time_steps, channels, height, width)
state_buffer = np.zeros(state_shape, dtype=np.uint8)  # Buffer for stacked frames

# Helper function to stack frames
def stack_frames(state_buffer, new_frame):
    """
    Add a new frame to the stack and remove the oldest frame.
    """
    state_buffer = np.roll(state_buffer, shift=-1, axis=0)
    state_buffer[-1] = new_frame
    return state_buffer

for episode in range(episodes):
    # Reset the environment and initialize state buffer
    raw_state = env.reset().transpose(2, 0, 1)  # Transpose to (Channels, Height, Width)
    state_buffer.fill(0)  # Clear buffer for new episode
    state_buffer[-1] = raw_state  # Add the first frame to the stack

    done = False
    total_reward = 0

    # Update epsilon based on the custom schedule
    if episode < min_exploration_episodes:
        agent.epsilon = 1.0  # Full exploration
    else:
        agent.epsilon = max(
            agent.epsilon_min,
            1.0 - (episode - min_exploration_episodes) / max_decay_episodes,
        )

    while not done:
        if render:
            env.render()

        # Select an action using the current policy
        action = agent.act(state_buffer)

        # Perform the action in the environment
        next_raw_state, reward, done, _ = env.step(action)
        next_raw_state = next_raw_state.transpose(2, 0, 1)  # Transpose to (Channels, Height, Width)

        # Update the frame stack with the new state
        next_state_buffer = stack_frames(state_buffer, next_raw_state)

        # Store the transition in memory
        agent.remember(state_buffer, action, reward, next_state_buffer, done)

        # Train the agent using replay
        agent.replay()

        # Update state
        state_buffer = next_state_buffer
        total_reward += reward

        if render:
            clock.tick(120)  # Limit frame rate to 120 FPS

    # Print episode statistics
    print(
        f"Episode: {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}"
    )

    # Save the model periodically
    if (episode + 1) % save_interval == 0:
        torch.save(agent.model.state_dict(), f'./models/dqn_lstm_model_{episode + 1}.pth')
        print(f"Model saved at episode {episode + 1}")

# Save the final model after training
torch.save(agent.model.state_dict(), './models/dqn_lstm_model_final.pth')
print("Training complete. Final model saved as 'dqn_lstm_model_final.pth'")
