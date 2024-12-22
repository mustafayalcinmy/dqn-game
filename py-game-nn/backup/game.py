class AmazingGameEnv(Env):
    def __init__(self):
        super(AmazingGameEnv, self).__init__()
        # Define action and observation space
        self.action_space = Discrete(3)  # Actions: 0=left, 1=right, 2=do nothing
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0]),  # Min values: cube x, cube y, gap x, gap y
            high=np.array([405, 720, 405, 720]),  # Max values
            dtype=np.float32
        )
        
        # Game-specific settings
        self.SCREEN_WIDTH = 405
        self.SCREEN_HEIGHT = 720
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
        self.gapWidth = 100
        self.blockHeight = 10
        self.obstacleFrequency = 300
        self.camera = {"y": 0, "speed": 100}
        self.gaps = []
        self.score = 0
        self.max_score = 10
        self.done = False

        # Initialize Pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Amazing Game Training Environment")
        self.clock = pygame.time.Clock()
        
        self.colors = {
            "bg": (40, 40, 40),
            "cube": (235, 219, 180),
            "obstacle": (204, 36, 29)
        }

        self.reset()

    def reset(self):
        self.cube["x"] = 100
        self.cube["y"] = 400
        self.cube["speed"] = 0
        self.cube["xSpeed"] = 0
        self.camera["y"] = 0
        self.score = 0
        self.done = False
        self.gaps = []
        self._generate_obstacles()
        return self._get_state()

    def step(self, action):
        if action == 0:  # Move left
            self.cube["xSpeed"] -= self.cube["xAcceleration"]
        elif action == 1:  # Move right
            self.cube["xSpeed"] += self.cube["xAcceleration"]
        elif action == 2:  # Do nothing
            pass

        dt = 1 / 60  # Fixed time step
        self._update_cube(dt)
        self._update_camera(dt)
        self._update_obstacles()

        # Calculate reward
        reward = 1  # Reward for surviving
        if self.done:
            reward = -10  # Penalty for hitting an obstacle or exceeding bounds

        # Check if game ends
        if self.score >= self.max_score:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self, mode="human"):
        self.screen.fill(self.colors["bg"])
        pygame.draw.rect(
            self.screen,
            self.colors["cube"],
            (self.cube["x"], self.cube["y"] - self.camera["y"], self.cube["size"], self.cube["size"]),
        )
        for gap in self.gaps:
            pygame.draw.rect(self.screen, self.colors["obstacle"], (0, gap["y"] - self.camera["y"], gap["x"], self.blockHeight))
            pygame.draw.rect(self.screen, self.colors["obstacle"], (gap["x"] + self.gapWidth, gap["y"] - self.camera["y"], self.SCREEN_WIDTH - gap["x"] - self.gapWidth, self.blockHeight))
        score_text = pygame.font.Font(None, 36).render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def _get_state(self):
        gap = self.gaps[0] if self.gaps else {"x": self.SCREEN_WIDTH / 2, "y": 0}
        return np.array([self.cube["x"], self.cube["y"], gap["x"], gap["y"]])

    def _generate_obstacles(self):
        startY = self.cube["y"] + 200
        for i in range(5):
            blockY = startY - i * self.obstacleFrequency
            gapX = random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": gapX, "y": blockY, "passed": False})

    def _update_obstacles(self):
        self.gaps = [gap for gap in self.gaps if gap["y"] <= self.camera["y"] + self.SCREEN_HEIGHT]
        while len(self.gaps) < 5:
            lastY = self.gaps[-1]["y"] if self.gaps else 0
            newY = lastY - self.obstacleFrequency
            newGapX = random.randint(0, self.SCREEN_WIDTH - self.gapWidth)
            self.gaps.append({"x": newGapX, "y": newY, "passed": False})

    def _update_cube(self, dt):
        self.cube["speed"] += self.cube["gravity"] * dt
        self.cube["y"] += self.cube["speed"] * dt
        if self.cube["xSpeed"] > 0:
            self.cube["xSpeed"] = max(0, self.cube["xSpeed"] - self.cube["friction"] * dt)
        elif self.cube["xSpeed"] < 0:
            self.cube["xSpeed"] = min(0, self.cube["xSpeed"] + self.cube["friction"] * dt)
        self.cube["x"] += self.cube["xSpeed"] * dt
        self.cube["x"] = max(0, min(self.cube["x"], self.SCREEN_WIDTH - self.cube["size"]))
        for gap in self.gaps:
            if self.cube["y"] + self.cube["size"] > gap["y"] and self.cube["y"] < gap["y"] + self.blockHeight:
                if self.cube["x"] + self.cube["size"] < gap["x"] or self.cube["x"] > gap["x"] + self.gapWidth:
                    self.done = True  # Collision detected, end the game
            if not gap.get("passed") and self.cube["y"] < gap["y"] + self.blockHeight:
                gap["passed"] = True
                self.score += 1

    def _update_camera(self, dt):
        targetY = self.cube["y"] - self.SCREEN_HEIGHT / 2 + self.cube["size"] / 2
        self.camera["y"] += (targetY - self.camera["y"]) * min(self.camera["speed"] * 0.01 * dt, 1)

    def play_human(self):
        """Allows a human to play the game for debugging."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.step(0)  # Move left
                    elif event.key == pygame.K_RIGHT:
                        self.step(1)  # Move right
                    elif event.key == pygame.K_SPACE:
                        self.step(2)  # Do nothing

            if not self.done:
                self.render()
            else:
                print(f"Game Over! Final Score: {self.score}")
                running = False

        self.close()


env = AmazingGameEnv()
env.play_human()
