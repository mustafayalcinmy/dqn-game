import pygame
import random
import numpy as np
from cube_game_env import CubeGameEnv

# ... (Your CubeGameEnv class code from the previous response)

def play_game(env, policy=None, render=True, delay=30):
    """Plays the CubeGameEnv, optionally using a policy.

    Args:
        env: An instance of the CubeGameEnv.
        policy: A function that takes an observation and returns an action (0, 1, or 2).
                If None, the player controls the cube with arrow keys.
        render: Whether to render the game.
        delay: Milliseconds to delay between frames (adjusts speed).
    """

    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        if policy:
            action = policy(observation)
        else:  # Human control
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 2
                    else:
                        action = 0  # No jump if other keys are pressed
                    break
                else:
                    action = 0 #no action if no event
            else:
                action = 0

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if render:
            env.render()
            pygame.time.delay(delay)  # Control game speed

    print(f"Game Over. Total Reward: {total_reward}")
    pygame.quit()


# Example usage (human control):
if __name__ == "__main__":
    env = CubeGameEnv()
    play_game(env)

    # Example usage (with a random policy - for demonstration):
    def random_policy(observation):
        return random.randint(0, 2)

    env2 = CubeGameEnv()
    play_game(env2, policy=random_policy, delay=50) #make it slower to see what the random policy does