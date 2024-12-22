import pygame
import sys
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 405
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Amazing Game")

# Color definitions
colors = {
    "bg": (40, 40, 40),  # Background
    "fg": (235, 219, 180),  # Foreground
    "red": (204, 36, 29),
    "green": (152, 151, 26),
    "yellow": (215, 153, 33),
    "blue": (69, 133, 136),
}


class AmazingGameEnv:
    def __init__(self):
        self.cube = {
            "x": 100,
            "y": 400,
            "size": 20,
            "speed": 0,
            "gravity": 350,
            "jump": -250,
            "xSpeed": 0,
            "xAcceleration": 100,
            "friction": 10,
        }
        self.camera = {"y": 0, "speed": 100}
        self.gapWidth = 100
        self.blockHeight = 10
        self.obstacleFrequency = 300
        self.gaps = []
        self.score = 0
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.reset()

    def reset(self):
        """Reset the environment to the initial state."""
        self.cube["x"] = 100
        self.cube["y"] = 400
        self.cube["speed"] = 0
        self.cube["xSpeed"] = 0
        self.camera["y"] = 0
        self.score = 0
        self.generate_obstacles()
        return self.get_state()

    def step(self, action):
        """Take a step in the environment with the given action."""
        if action == 1:  # Jump right
            self.cube["speed"] = self.cube["jump"]
            self.cube["xSpeed"] = self.cube["xAcceleration"]
        elif action == 2:  # Jump left
            self.cube["speed"] = self.cube["jump"]
            self.cube["xSpeed"] = -self.cube["xAcceleration"]

        self.update_cube(1 / 60)
        self.update_camera(1 / 60)
        self.update_obstacles()

        state = self.get_state()
        reward = self.compute_reward()
        done = self.cube["y"] > SCREEN_HEIGHT or self.check_collision() or self.score >= 50
        return state, reward, done

    def render(self):
        """Render the current state of the environment."""
        screen.fill(colors["bg"])
        pygame.draw.rect(
            screen,
            colors["green"],
            (self.cube["x"], self.cube["y"] - self.camera["y"], self.cube["size"], self.cube["size"]),
        )
        for gap in self.gaps:
            pygame.draw.rect(
                screen,
                colors["red"],
                (0, gap["y"] - self.camera["y"], gap["x"], self.blockHeight),
            )
            pygame.draw.rect(
                screen,
                colors["red"],
                (
                    gap["x"] + self.gapWidth,
                    gap["y"] - self.camera["y"],
                    SCREEN_WIDTH - gap["x"] - self.gapWidth,
                    self.blockHeight,
                ),
            )
        score_text = self.font.render(f"Score: {self.score}", True, colors["fg"])
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def get_state(self):
        """Get the current state representation."""
        gap = self.gaps[0] if self.gaps else {"x": SCREEN_WIDTH / 2, "y": 0}
        return np.array([self.cube["x"], self.cube["y"], gap["x"], gap["y"]])

    def generate_obstacles(self):
        """Generate initial obstacles."""
        self.gaps = []
        startY = self.cube["y"] + 200
        for i in range(5):
            blockY = startY - i * self.obstacleFrequency
            gapX = random.randint(0, SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": gapX, "y": blockY, "passed": False})

    def compute_reward(self):
        """Calculate the reward for the current step."""
        reward = 0
        for gap in self.gaps:
            if not gap["passed"] and self.cube["y"] < gap["y"]:
                gap["passed"] = True
                self.score += 1
                reward += 10  # Positive reward for passing an obstacle
        return reward

    def update_cube(self, dt):
        """Update cube position."""
        self.cube["speed"] += self.cube["gravity"] * dt
        self.cube["y"] += self.cube["speed"] * dt
        self.cube["x"] += self.cube["xSpeed"] * dt

        # Apply friction
        if self.cube["xSpeed"] > 0:
            self.cube["xSpeed"] = max(0, self.cube["xSpeed"] - self.cube["friction"] * dt)
        elif self.cube["xSpeed"] < 0:
            self.cube["xSpeed"] = min(0, self.cube["xSpeed"] + self.cube["friction"] * dt)

        # Keep cube within bounds
        self.cube["x"] = max(0, min(self.cube["x"], SCREEN_WIDTH - self.cube["size"]))

    def update_camera(self, dt):
        """Update the camera position."""
        targetY = self.cube["y"] - SCREEN_HEIGHT / 2 + self.cube["size"] / 2
        self.camera["y"] += (targetY - self.camera["y"]) * min(self.camera["speed"] * 0.01 * dt, 1)

    def update_obstacles(self):
        """Update the obstacle positions."""
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera["y"] + SCREEN_HEIGHT]
        while len(self.gaps) < 5:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacleFrequency
            newGapX = random.randint(0, SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": newGapX, "y": newY, "passed": False})

    def check_collision(self):
        """Check if the cube collides with any obstacle."""
        for gap in self.gaps:
            if self.cube["y"] + self.cube["size"] > gap["y"] and self.cube["y"] < gap["y"] + self.blockHeight:
                if self.cube["x"] + self.cube["size"] < gap["x"] or self.cube["x"] > gap["x"] + self.gapWidth:
                    return True
        return False


# Human Play Mode
def human_play():
    env = AmazingGameEnv()
    running = True
    while running:
        dt = env.clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            env.step(2)
        elif keys[pygame.K_RIGHT]:
            env.step(1)
        else:
            env.step(0)
        env.render()
    pygame.quit()


human_play()
