import gym
from gym import spaces
import pygame
import numpy as np
import random

class AmazingGameEnv(gym.Env):
    def __init__(self):
        super(AmazingGameEnv, self).__init__()
        pygame.init()
        self.SCREEN_WIDTH = 405
        self.SCREEN_HEIGHT = 720
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -float("inf")]),
            high=np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, float("inf")]),
            dtype=np.float32
        )
        
        self.colors = {
            "bg": (40, 40, 40),
            "cube": (235, 219, 180),
            "obstacle": (204, 36, 29)
        }
        self.cube = {"x": 100, "y": 400, "size": 20, "speed": 0, "gravity": 350, "jump": -250, "xSpeed": 0}
        self.camera = {"y": 0, "speed": 100}
        self.gapWidth = 100
        self.blockHeight = 10
        self.obstacleFrequency = 300
        self.gaps = []
        self.score = 0
        self.running = True
        
        self.seed()  # Initialize the random seed
        self.generate_obstacles()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def generate_obstacles(self):
        self.gaps = []
        startY = self.cube["y"] + 200
        for i in range(5):
            blockY = startY - i * self.obstacleFrequency
            gapX = random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": gapX, "y": blockY, "passed": False})

    def reset(self):
        self.cube["x"] = 100
        self.cube["y"] = 400
        self.cube["speed"] = 0
        self.cube["xSpeed"] = 0
        self.camera["y"] = 0
        self.score = 0
        self.generate_obstacles()
        return self.get_state()

    def get_state(self):
        gap = self.gaps[0] if self.gaps else {"x": self.SCREEN_WIDTH / 2, "y": 0}
        return np.array([self.cube["x"], self.cube["y"], gap["x"], gap["y"]], dtype=np.float32)

    def step(self, action):
        """
        Perform the action and update the game state.
        Action: 0 = Move left and jump, 1 = Move right and jump, 2 = Do nothing.
        """
        if action == 0:  # Move left and jump
            self.cube["xSpeed"] = 0
            self.cube["xSpeed"] -= self.cube["gravity"] * 0.3
            self.cube["speed"] = self.cube["jump"]  # Apply jump force
        elif action == 1:  # Move right and jump
            self.cube["xSpeed"] = 0
            self.cube["xSpeed"] += self.cube["gravity"] * 0.3
            self.cube["speed"] = self.cube["jump"]  # Apply jump force
        elif action == 2:  # Do nothing
            pass

        self.update_cube(1 / 60)
        self.update_camera(1 / 60)
        self.update_obstacles()

        # Check for scoring
        for gap in self.gaps:
            if not gap.get("passed") and self.cube["y"] < gap["y"]:
                gap["passed"] = True
                self.score += 1  

        state = self.get_state()

        reward = 1  
        done = self.cube["y"] > self.SCREEN_HEIGHT 
        if done:
            reward = -10  

        return state, reward, done, {}


    def update_cube(self, dt):
        self.cube["speed"] += self.cube["gravity"] * dt
        self.cube["y"] += self.cube["speed"] * dt
        self.cube["x"] += self.cube["xSpeed"] * dt
        self.cube["x"] = max(0, min(self.cube["x"], self.SCREEN_WIDTH - self.cube["size"]))
        for gap in self.gaps:
            if self.cube["y"] + self.cube["size"] > gap["y"] and self.cube["y"] < gap["y"] + self.blockHeight:
                if self.cube["x"] + self.cube["size"] < gap["x"] or self.cube["x"] > gap["x"] + self.gapWidth:
                    self.running = False

    def update_camera(self, dt):
        targetY = self.cube["y"] - self.SCREEN_HEIGHT / 2 + self.cube["size"] / 2
        self.camera["y"] += (targetY - self.camera["y"]) * min(self.camera["speed"] * 0.01 * dt, 1)

    def update_obstacles(self):
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera["y"] + self.SCREEN_HEIGHT]
        while len(self.gaps) < 5:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacleFrequency
            newGapX = random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": newGapX, "y": newY, "passed": False})

    def render(self, mode="human"):
        if mode == "human":
            if not hasattr(self, "window"):
                self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Amazing Game")

            self.screen.fill(self.colors["bg"])

            # Draw the cube
            pygame.draw.rect(self.screen, self.colors["cube"], 
                            (self.cube["x"], self.cube["y"] - self.camera["y"], self.cube["size"], self.cube["size"]))

            # Draw gaps/obstacles
            for gap in self.gaps:
                pygame.draw.rect(self.screen, self.colors["obstacle"], 
                                (0, gap["y"] - self.camera["y"], gap["x"], self.blockHeight))
                pygame.draw.rect(self.screen, self.colors["obstacle"], 
                                (gap["x"] + self.gapWidth, gap["y"] - self.camera["y"], 
                                self.SCREEN_WIDTH - gap["x"] - self.gapWidth, self.blockHeight))

            # Display the score
            font = pygame.font.Font(None, 36)
            score_surface = font.render(f"Score: {self.score}", True, (255, 255, 255))
            self.screen.blit(score_surface, (10, 10))

            # Blit the surface onto the display window
            self.window.blit(self.screen, (0, 0))
            pygame.display.flip()

            # Handle player input
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                return 0  # Move left
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                return 1  # Move right
            elif keys[pygame.K_SPACE]:
                self.cube["speed"] = self.cube["jump"]  # Jump
            else:
                return 2  # Do nothing

    def close(self):
        pygame.quit()

    def human_play(self):
        clock = pygame.time.Clock()
        self.reset()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            action = self.render()
            state, reward, done, _ = self.step(action)
            if done:
                self.reset()

            clock.tick(60)

        self.close()

