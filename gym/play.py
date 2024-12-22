import time
from stable_baselines3 import DQN
from gym.amazing_game_env import AmazingGameEnv

model = DQN.load("amazing_game_dqn")


def basic_agent(obs):
    """
    A simple agent that attempts to navigate gaps.
    obs: [cube_x, cube_y, gap_x, gap_y]
    """
    cube_x, cube_y, gap_x, gap_y = obs

    # Action logic
    if cube_x < gap_x + 20:  # Move right to center in the gap
        return 1  # Move right
    elif cube_x > gap_x + 80:  # Move left to center in the gap
        return 0  # Move left
    else:
        return 2  # Do nothing


env = AmazingGameEnv()
obs = env.reset()

for _ in range(1000):
    # Use the basic agent for action selection
    action = basic_agent(obs)
    
    # Step through the environment
    obs, reward, done, info = env.step(action)
    env.render()  # Render the game
    time.sleep(1 / 60)  # 60 FPS
    
    if done:
        print("Game Over! Resetting environment.")
        obs = env.reset()  # Reset the environment if the game is over

env.close()
