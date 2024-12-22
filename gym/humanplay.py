import time
from game_env import AmazingGameEnv


env = AmazingGameEnv()
obs = env.reset()

for _ in range(1000):
    # Render the game and capture player input
    action = None
    while action is None:
        action = env.render()  # Render the game and return player input

    # Step through the environment
    obs, reward, done, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Score: {env.score}")

    # Limit frame rate
    time.sleep(1 / 60)

    if done:
        print("Game Over! Final Score:", env.score)
        obs = env.reset()  # Reset the environment for another game

env.close()
