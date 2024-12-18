import pygame
import random

class Button:
    def __init__(self, y, width, height, text, color, text_color, on_click=None):
        self.x = None  # To be set after screen initialization
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.text_color = text_color
        self.on_click = on_click if on_click else lambda: None  # Default to a no-op function

    def set_position(self, screen_width):
        # Calculate horizontal position to center the button
        self.x = (screen_width - self.width) / 2

    def draw(self, screen, font):
        # Draw the button
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

        # Draw the button text, vertically and horizontally centered
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(
            self.x + self.width / 2, self.y + self.height / 2
        ))
        screen.blit(text_surface, text_rect)

    def is_hovered(self, mx, my):
        # Check if the mouse is over the button
        return self.x < mx < self.x + self.width and self.y < my < self.y + self.height

    def click(self, mx, my):
        # If hovered, execute the button's on_click function
        if self.is_hovered(mx, my):
            self.on_click()

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
COLORS = {
    "bg": (40, 40, 40),
    "fg": (235, 219, 180),
    "red": (204, 36, 29),
    "green": (152, 151, 26),
    "yellow": (215, 153, 33),
    "blue": (69, 133, 136),
    "purple": (177, 98, 134),
    "aqua": (104, 157, 106),
    "orange": (214, 93, 14)
}

# Cube properties
cube = {
    "x": 100,
    "y": 400,
    "size": 20,
    "speed": 0,
    "gravity": 350,
    "jump": -250,
    "x_speed": 0,
    "x_acceleration": 100,
    "friction": 10
}

# Camera properties
camera = {
    "y": 0,
    "speed": 100
}

# Game state
state = "menu"
score = 0
gaps = []
obstacle_frequency = 300
gap_width = 100
block_height = 10

# Button instance
def start_game():
    global state
    state = "play"

button = Button(250, 200, 50, "Start Game", COLORS["green"], COLORS["fg"], start_game)

def generate_obstacles():
    global gaps
    start_y = cube["y"] + 200
    for i in range(5):
        block_y = start_y - i * obstacle_frequency
        gap_x = random.randint(0, SCREEN_WIDTH - gap_width)
        gaps.append({"x": gap_x, "y": block_y, "passed": False})

def update_obstacles():
    global gaps
    gaps = [gap for gap in gaps if gap["y"] <= camera["y"] + SCREEN_HEIGHT]
    while len(gaps) < 5:
        last_y = gaps[-1]["y"] if gaps else 0
        new_y = last_y - obstacle_frequency
        new_gap_x = random.randint(0, SCREEN_WIDTH - gap_width)
        gaps.append({"x": new_gap_x, "y": new_y, "passed": False})

def reset_game():
    global cube, camera, gaps, score
    cube.update({"x": 100, "y": 400, "speed": 0})
    camera["y"] = 0
    gaps = []
    generate_obstacles()

def main():
    global state, score
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    # Set button position
    button.set_position(SCREEN_WIDTH)
    
    reset_game()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if state == "menu":
                    button.click(mx, my)

        screen.fill(COLORS["bg"])

        if state == "menu":
            button.draw(screen, font)

        elif state == "play":
            # Apply gravity and movement
            cube["speed"] += cube["gravity"] * dt
            cube["y"] += cube["speed"] * dt
            cube["x"] += cube["x_speed"] * dt

            # Update camera position
            target_y = cube["y"] - SCREEN_HEIGHT / 2
            camera["y"] += (target_y - camera["y"]) * camera["speed"] * dt

            # Draw cube
            pygame.draw.rect(
                screen, COLORS["green"],
                pygame.Rect(cube["x"], cube["y"] - camera["y"], cube["size"], cube["size"])
            )

            # Draw obstacles
            for gap in gaps:
                pygame.draw.rect(
                    screen, COLORS["red"],
                    pygame.Rect(0, gap["y"] - camera["y"], gap["x"], block_height)
                )
                pygame.draw.rect(
                    screen, COLORS["red"],
                    pygame.Rect(
                        gap["x"] + gap_width, gap["y"] - camera["y"],
                        SCREEN_WIDTH - gap["x"] - gap_width, block_height
                    )
                )

            update_obstacles()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
