import pygame
import random
import numpy as np

class CubeGameEnv:
    def __init__(self):
        pygame.init()
        self.screen_width = 405
        self.screen_height = 720
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Cube Game Training Visualization")
        self.clock = pygame.time.Clock()
        self.colors = {
            "bg": (40, 40, 40),
            "cube": (235, 219, 180),
            "obstacle": (204, 36, 29),
            "gap": (69, 133, 136)
        }
        self.gap_width = 100
        self.block_height = 10
        self.gravity = 350
        self.jump_strength = -250
        self.x_acceleration = 100
        self.friction = 10
        self.obstacle_frequency = 300
        self.max_steps = 1000  # Optional step limit
        self.reset()

    def reset(self):
        self.cube = {
            "x": 100,
            "y": 400,
            "speed": 0,
            "xSpeed": 0
        }
        self.camera_y = 0
        self.steps = 0
        self.generate_obstacles()
        return self.get_observation()

    def generate_obstacles(self):
        self.gaps = []
        startY = self.cube["y"] + 200
        for i in range(5):
            blockY = startY - i * self.obstacle_frequency
            gapX = random.randint(0, self.screen_width - self.gap_width)
            self.gaps.append({"x": gapX, "y": blockY})

    def update_obstacles(self):
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera_y + self.screen_height]
        while len(self.gaps) < 5:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacle_frequency
            newGapX = random.randint(0, self.screen_width - self.gap_width)
            self.gaps.append({"x": newGapX, "y": newY})

    def get_closest_gap(self):
        for gap in self.gaps:
            if gap["y"] > self.cube["y"]:
                return gap
        return self.gaps[-1]

    def step(self, action):
        dt = 1 / 60
        self.steps += 1
        if action == 1:
            self.cube["xSpeed"] = -self.x_acceleration
            self.cube["speed"] = self.jump_strength
        elif action == 2:
            self.cube["xSpeed"] = self.x_acceleration
            self.cube["speed"] = self.jump_strength

        self.cube["speed"] += self.gravity * dt
        self.cube["y"] += self.cube["speed"] * dt

        if self.cube["xSpeed"] > 0:
            self.cube["xSpeed"] = max(0, self.cube["xSpeed"] - self.friction * dt)
        elif self.cube["xSpeed"] < 0:
            self.cube["xSpeed"] = min(0, self.cube["xSpeed"] + self.friction * dt)

        self.cube["x"] += self.cube["xSpeed"] * dt
        self.cube["x"] = max(0, min(self.cube["x"], self.screen_width))

        self.update_obstacles()
        self.camera_y = self.cube["y"] - self.screen_height / 2

        closest_gap = self.get_closest_gap()
        reward = self.calculate_reward(closest_gap)

        done = self.is_done(closest_gap)
        return self.get_observation(), reward, done, {}

    def calculate_reward(self, closest_gap):
        cube_center_x = self.cube["x"] + 10
        gap_center_x = closest_gap["x"] + self.gap_width / 2
        horizontal_penalty = -abs(cube_center_x - gap_center_x) / (self.screen_width / 2)
        vertical_reward = max(0, (self.screen_height - abs(self.cube["y"] - closest_gap["y"])) / self.screen_height)
        return vertical_reward + horizontal_penalty

    def is_done(self, closest_gap):
        if self.cube["y"] > self.camera_y + self.screen_height or self.cube["y"] < 0:
            return True
        if (self.cube["y"] + 20 > closest_gap["y"] and
            (self.cube["x"] + 20 < closest_gap["x"] or self.cube["x"] > closest_gap["x"] + self.gap_width)):
            return True
        return self.steps >= self.max_steps

    def get_observation(self):
        closest_gap = self.get_closest_gap()
        return np.array([
            self.cube["x"] / self.screen_width,
            self.cube["y"] / self.screen_height,
            self.cube["speed"] / self.gravity,
            closest_gap["x"] / self.screen_width,
            closest_gap["y"] / self.screen_height
        ], dtype=np.float32)

    def render(self):
        self.screen.fill(self.colors["bg"])
        pygame.draw.rect(self.screen, self.colors["cube"], (self.cube["x"], self.cube["y"] - self.camera_y, 20, 20))
        for gap in self.gaps:
            gap_y = gap["y"] - self.camera_y
            pygame.draw.rect(self.screen, self.colors["obstacle"], (0, gap_y, gap["x"], self.block_height))
            pygame.draw.rect(self.screen, self.colors["obstacle"], (gap["x"] + self.gap_width, gap_y, self.screen_width - gap["x"] - self.gap_width, self.block_height))
        pygame.display.flip()
        self.clock.tick(30)
