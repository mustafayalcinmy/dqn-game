import gym
from gym import spaces
import pygame
import numpy as np
import random
from typing import Tuple, Dict, Any
from pprint import pprint

class AmazingGameEnv(gym.Env):
    def __init__(self):
        super().__init__()
        pygame.init()
        self.SCREEN_WIDTH = 405
        self.SCREEN_HEIGHT = 720
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),  # 4 dimensions
            high=np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT], dtype=np.float32)  # 4 dimensions
        )

        
        self.colors = {"bg": (40, 40, 40), "cube": (235, 219, 180), "obstacle": (204, 36, 29)}
        self.cube = {"x": 0, "y": 400, "size": 20, "speed": 0, "gravity": 300, "jump": -150, "xSpeed": 0}
        self.camera = {"y": 0, "speed": 100}
        self.gapWidth = 100
        self.blockHeight = 10
        self.obstacleFrequency = 350
        self.gaps = []
        self.score = 0
        self.running = True

        self.seed()
        self.generate_obstacles()
        self.reset()

    def seed(self, seed: int = None):
        random.seed(seed)
        np.random.seed(seed)

    def generate_obstacles(self):
        startY = self.cube["y"] - 300
        self.gaps = [{"x": random.randint(0, self.SCREEN_WIDTH - self.gapWidth), "y": startY - i * self.obstacleFrequency, "passed": False} for i in range(3)]

    def reset(self) -> np.ndarray:
        self.cube.update({"x": random.randint(10, self.SCREEN_WIDTH - 10), "y": 400, "speed": 0, "xSpeed": 0})
        self.camera["y"] = 0
        self.score = 0
        self.generate_obstacles()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        gap = None
        for tempGap in self.gaps:
            if tempGap["passed"] != True:
                gap = tempGap
                break
            
        return np.array([self.cube["x"], self.cube["y"], gap["x"] + self.gapWidth / 2, gap["y"]], dtype=np.float32)

 def step(self, action):
        """
        Perform the action and update the game state.
        Action: 0 = Move left and jump, 1 = Move right and jump, 2 = Do nothing.
        """
        if action == 0:  # Move left and jump
            self.cube["xSpeed"] = 0
            self.cube["xSpeed"] -= 100
            self.cube["speed"] = self.cube["jump"]  # Apply jump force
        elif action == 1:  # Move right and jump
            self.cube["xSpeed"] = 0
            self.cube["xSpeed"] += 100
            self.cube["speed"] = self.cube["jump"]  # Apply jump force
        elif action == 2:  # Do nothing
            pass

        self.update_cube(1 / 60)
        self.update_camera(1 / 60)
        self.update_obstacles()

        reward = 0
        # Check for scoring
        for gap in self.gaps:
            if not gap.get("passed") and self.cube["y"] < gap["y"]:
                gap["passed"] = True
                self.score += 1  
                reward = self.score * 10

        state = self.get_state()

        # Check if the cube hits an obstacle
        done = self.cube["y"] > self.SCREEN_HEIGHT
        if done:
            reward -= 100

        for gap in self.gaps:
            if self.cube["y"] + self.cube["size"] > gap["y"] and self.cube["y"] < gap["y"] + self.blockHeight:
                if self.cube["x"] + self.cube["size"] < gap["x"] or self.cube["x"] > gap["x"] + self.gapWidth:
                    done = True
                    reward -= 20
        

        return state, reward, done, {}


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if action == 0:
            self.cube.update({"xSpeed": -100, "speed": self.cube["jump"]})
        elif action == 1:
            self.cube.update({"xSpeed": 100, "speed": self.cube["jump"]})
        elif action == 2:
            self.cube["xSpeed"] = 0

        self.update_cube(1 / 60)
        self.update_camera(1 / 60)
        self.update_obstacles()

        state = self.get_state()
        reward, done = self.compute_reward()

        return state, reward, done, {}

    def compute_reward(self) -> Tuple[float, bool]:
        reward = 0
        done = False

        reward -= 0.1  

        for gap in self.gaps:
            if not gap["passed"] and self.cube["y"] < gap["y"]:
                gap["passed"] = True
                self.score += 1
                reward += self.score * 2  # Higher reward for passing gaps to encourage progress

        if self.cube["y"] > self.SCREEN_HEIGHT:
            done = True
            reward -= 20
            
        
        if self.cube["y"] < -1050:
            done = True

        for gap in self.gaps:
            if self.cube["y"] + self.cube["size"] > gap["y"] and self.cube["y"] < gap["y"] + self.blockHeight:
                if self.cube["x"] + self.cube["size"] < gap["x"] or self.cube["x"] > gap["x"] + self.gapWidth:
                    done = True
                    reward -= 5

        if not done:
            reward += 0.1

        return reward, done


    def update_cube(self, dt: float):
        self.cube["speed"] += self.cube["gravity"] * dt
        self.cube["y"] += self.cube["speed"] * dt
        self.cube["x"] = np.clip(self.cube["x"] + self.cube["xSpeed"] * dt, 0, self.SCREEN_WIDTH - self.cube["size"])

    def update_camera(self, dt: float):
        targetY = self.cube["y"] - self.SCREEN_HEIGHT / 2 + self.cube["size"] / 2
        self.camera["y"] += (targetY - self.camera["y"]) * min(self.camera["speed"] * 0.01 * dt, 1)

    def update_obstacles(self):
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera["y"] + self.SCREEN_HEIGHT]
        while len(self.gaps) < 4:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacleFrequency
            self.gaps.append({"x": random.randint(0, self.SCREEN_WIDTH - self.gapWidth), "y": newY, "passed": False})

    def render(self, mode: str = "human"):
        if mode == "human":
            if not hasattr(self, "window"):
                self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Amazing Game")

            self.screen.fill(self.colors["bg"])
            pygame.draw.rect(self.screen, self.colors["cube"], (self.cube["x"], self.cube["y"] - self.camera["y"], self.cube["size"], self.cube["size"]))
            for gap in self.gaps:
                pygame.draw.rect(self.screen, self.colors["obstacle"], (0, gap["y"] - self.camera["y"], gap["x"], self.blockHeight))
                pygame.draw.rect(self.screen, self.colors["obstacle"], (gap["x"] + self.gapWidth, gap["y"] - self.camera["y"], self.SCREEN_WIDTH - gap["x"] - self.gapWidth, self.blockHeight))

            font = pygame.font.Font(None, 36)
            score_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_surface, (10, 10))
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()

    def close(self):
        pygame.quit()
