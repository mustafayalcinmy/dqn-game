import gymnasium as gym

# Initialize the environment with the correct render mode
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment
state, info = env.reset()

# Interact with the environment
for _ in range(100):
    action = env.action_space.sample()  # Sample a random action
    state, reward, done, truncated, info = env.step(action)

    # Render the environment
    env.render()

    if done or truncated:
        break

env.close()
