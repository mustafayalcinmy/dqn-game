import torch
from cube_game_env import CubeGameEnv
from dqn_training import DQN  # Ensure this matches the location of your DQN class
from model_utils import load_model

# Set up the environment and load the model
env = CubeGameEnv()
input_dim = len(env.get_observation())
action_dim = 3

# Initialize the model and load the saved weights
model = DQN(input_dim, action_dim)
model = load_model(model, "saved_models/dqn_model_generation_85.pth")  # Update path as needed
model.eval()  # Set model to evaluation mode

# Function to select the best action
def select_best_action(state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return model(state).argmax().item()  # Return the action with the highest Q-value

# Play the game
state = env.reset()
done = False
total_reward = 0

print("Playing the game with the trained model. Press Ctrl+C to quit.")
while True:  # Continuous gameplay
    action = select_best_action(state)  # Choose the best action
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()  # Render the game visually
    if done:
        print(f"Game Over! Total Reward: {total_reward}")
        state = env.reset()
        total_reward = 0
