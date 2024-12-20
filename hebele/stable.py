from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from amazing_game_env import AmazingGameEnv
import gymnasium as gym


env = AmazingGameEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_hobbalaaa")