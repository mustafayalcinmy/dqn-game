import gym
import pygame
import numpy as np
from gym import spaces

class AmazingGameEnv(gym.Env):
    def __init__(self, manual_mode=False):
        super(AmazingGameEnv, self).__init__()
        self.manual_mode = manual_mode

        # Screen settings
        self.SCREEN_WIDTH = 405
        self.SCREEN_HEIGHT = 720
        self.gapWidth = 100
        self.blockHeight = 10
        self.obstacleFrequency = 300

        # Cube properties
        self.cube = {
            "x": 100,
            "y": 400,
            "size": 20,
            "speed": 0,
            "gravity": 350,
            "jump": -250,
            "xSpeed": 0,
            "xAcceleration": 100,
            "friction": 10
        }

        # Camera properties
        self.camera = {
            "y": 0,
            "speed": 100
        }

        # Obstacles and scoring
        self.gaps = []
        self.score = 0
        self.max_score = 100

        # Action space: 0 for no movement, 1 for move left, 2 for move right
        self.action_space = spaces.Discrete(3)

        # Observation space: [cube_x, cube_y, gap_x, gap_y]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT]),
            dtype=np.float32
        )

        # Pygame setup for rendering and manual play
        if self.manual_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Amazing Game - Manual Play")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 32)

        self.reset()

    def reset(self):
        """Reset the game environment."""
        self.cube["x"] = 100
        self.cube["y"] = 400
        self.cube["speed"] = 0
        self.cube["xSpeed"] = 0
        self.camera["y"] = 0
        self.score = 0
        self.generate_obstacles()
        return self.get_state()

    def step(self, action):
        """Perform an action and update the environment."""
        if action == 1:  # Move left
            self.cube["xSpeed"] -= self.cube["xAcceleration"]
        elif action == 2:  # Move right
            self.cube["xSpeed"] += self.cube["xAcceleration"]

        # Update game logic
        dt = 1 / 60
        self.update_cube(dt)
        self.update_camera(dt)
        self.update_obstacles()

        # Calculate state, reward, and done flag
        state = self.get_state()
        reward = self.score if self.score <= self.max_score else 0
        done = self.cube["y"] > self.SCREEN_HEIGHT or self.score >= self.max_score

        return state, reward, done, {}

    def render(self, mode='human'):
        """Render the game."""
        if not self.manual_mode:
            return

        self.screen.fill((40, 40, 40))  # Background color

        # Draw cube
        pygame.draw.rect(self.screen, (152, 151, 26), (self.cube["x"], self.cube["y"] - self.camera["y"], self.cube["size"], self.cube["size"]))

        # Draw obstacles
        for gap in self.gaps:
            pygame.draw.rect(self.screen, (204, 36, 29), (0, gap["y"] - self.camera["y"], gap["x"], self.blockHeight))
            pygame.draw.rect(self.screen, (204, 36, 29), (gap["x"] + self.gapWidth, gap["y"] - self.camera["y"], self.SCREEN_WIDTH - gap["x"] - self.gapWidth, self.blockHeight))

        # Display score
        score_text = self.font.render(f"Score: {self.score}", True, (235, 219, 180))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def generate_obstacles(self):
        """Generate initial obstacles."""
        self.gaps = []
        startY = self.cube["y"] + 200
        for i in range(5):
            blockY = startY - i * self.obstacleFrequency
            gapX = np.random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": gapX, "y": blockY})

    def update_obstacles(self):
        """Update obstacle positions."""
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera["y"] + self.SCREEN_HEIGHT]
        while len(self.gaps) < 5:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacleFrequency
            newGapX = np.random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": newGapX, "y": newY})

    def update_cube(self, dt):
        """Update the cube's position."""
        self.cube["speed"] += self.cube["gravity"] * dt
        self.cube["y"] += self.cube["speed"] * dt
        if self.cube["xSpeed"] > 0:
            self.cube["xSpeed"] = max(0, self.cube["xSpeed"] - self.cube["friction"] * dt)
        elif self.cube["xSpeed"] < 0:
            self.cube["xSpeed"] = min(0, self.cube["xSpeed"] + self.cube["friction"] * dt)
        self.cube["x"] += self.cube["xSpeed"] * dt
        self.cube["x"] = max(0, min(self.cube["x"], self.SCREEN_WIDTH - self.cube["size"]))

    def update_camera(self, dt):
        """Update the camera position."""
        targetY = self.cube["y"] - self.SCREEN_HEIGHT / 2 + self.cube["size"] / 2
        self.camera["y"] += (targetY - self.camera["y"]) * min(self.camera["speed"] * 0.01 * dt, 1)

    def get_state(self):
        """Get the current state of the environment."""
        gap = self.gaps[0] if self.gaps else {"x": self.SCREEN_WIDTH / 2, "y": 0}
        return np.array([self.cube["x"], self.cube["y"], gap["x"], gap["y"]])

    def manual_play(self):
        """Play the game manually using keyboard controls."""
        running = True
        action = 0  # Default action: No movement

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:  # Move left
                        action = 1
                    elif event.key == pygame.K_RIGHT:  # Move right
                        action = 2
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        action = 0  # Stop moving when the key is released

            # Update the game logic using the selected action
            self.step(action)

            # Render the game
            self.render()

            # Maintain a consistent frame rate
            self.clock.tick(60)

        pygame.quit()



# Example usage for manual play
if __name__ == "__main__":
    env = AmazingGameEnv(manual_mode=True)
    env.manual_play()
