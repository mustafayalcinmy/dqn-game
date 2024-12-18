import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 405
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Amazing Game")

# Color definitions
colors = {
    "bg": (40, 40, 40),       # Dark background
    "fg": (235, 219, 180),    # Light foreground
    "red": (204, 36, 29),     # Red
    "green": (152, 151, 26),  # Green
    "yellow": (215, 153, 33), # Yellow
    "blue": (69, 133, 136),   # Blue
    "purple": (177, 98, 134), # Purple
    "aqua": (104, 157, 106),  # Aqua
    "orange": (214, 93, 14)   # Orange
}

# Cube properties
cube = {
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
camera = {
    "y": 0,    # Camera's vertical position
    "speed": 100  # Smooth camera movement speed
}

# Game state variables
state = "intro"
timer = 0
buttons = {}
gapWidth = 100  # Width of the gap in each block
blockHeight = 10  # Height of each rectangle
obstacleFrequency = 300  # Vertical spacing between new obstacles
gaps = []  # Table to store obstacle positions
score = 0  # Player's score
keyCooldown = 0.05
lastKeyPressTime = 0

# Button class
class Button:
    def __init__(self, y, width, height, text, color, textColor, onClick):
        self.x = (SCREEN_WIDTH - width) // 2
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.textColor = textColor
        self.onClick = onClick

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        font = pygame.font.Font(None, 32)
        text_surface = font.render(self.text, True, self.textColor)
        text_rect = text_surface.get_rect(center=(self.x + self.width / 2, self.y + self.height / 2))
        screen.blit(text_surface, text_rect)

    def is_hovered(self, mx, my):
        return self.x < mx < self.x + self.width and self.y < my < self.y + self.height

    def click(self, mx, my):
        if self.is_hovered(mx, my):
            self.onClick()

# Generate initial obstacles
def generate_obstacles():
    global gaps
    gaps = []
    startY = cube["y"] + 200
    for i in range(5):
        blockY = startY - i * obstacleFrequency
        gapX = random.randint(0, SCREEN_WIDTH - gapWidth)
        gaps.append({"x": gapX, "y": blockY})

def create_buttons():
    global buttons
    buttons["play"] = Button(350, 200, 50, "Play", colors["blue"], colors["fg"], lambda: set_state("play"))
    buttons["exit"] = Button(450, 200, 50, "Exit", colors["red"], colors["fg"], sys.exit)
    buttons["restart"] = Button(350, 200, 50, "Restart", colors["green"], colors["fg"], lambda: set_state("play") or reset_game())
    buttons["mainmenu"] = Button(450, 200, 50, "Main Menu", colors["yellow"], colors["fg"], lambda: set_state("menu"))

def set_state(new_state):
    global state
    state = new_state

create_buttons()
generate_obstacles()
# Main game loop
running = True
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)

# Reset the game
def reset_game():
    global score
    cube["x"] = 100
    cube["y"] = 400
    cube["speed"] = 0
    cube["xSpeed"] = 0
    camera["y"] = 0
    score = 0
    generate_obstacles()

# Update obstacles dynamically
def update_obstacles():
    global gaps
    gaps = [gap for gap in gaps if gap["y"] <= camera["y"] + SCREEN_HEIGHT]
    while len(gaps) < 5:
        lastY = gaps[-1]["y"] if gaps else 0
        newY = lastY - obstacleFrequency
        newGapX = random.randint(0, SCREEN_WIDTH - gapWidth)
        gaps.append({"x": newGapX, "y": newY})

# Update the cube's position
def update_cube(dt):
    global score, state
    cube["speed"] += cube["gravity"] * dt
    cube["y"] += cube["speed"] * dt
    if cube["xSpeed"] > 0:
        cube["xSpeed"] = max(0, cube["xSpeed"] - cube["friction"] * dt)
    elif cube["xSpeed"] < 0:
        cube["xSpeed"] = min(0, cube["xSpeed"] + cube["friction"] * dt)
    cube["x"] += cube["xSpeed"] * dt
    cube["x"] = max(0, min(cube["x"], SCREEN_WIDTH - cube["size"]))
    for gap in gaps:
        if cube["y"] + cube["size"] > gap["y"] and cube["y"] < gap["y"] + blockHeight:
            if cube["x"] + cube["size"] < gap["x"] or cube["x"] > gap["x"] + gapWidth:
                state = "gameover"
        if not gap.get("passed") and cube["y"] < gap["y"] + blockHeight:
            gap["passed"] = True
            score += 1
    if cube["y"] > camera["y"] + SCREEN_HEIGHT:
        state = "gameover"

# Update camera position
def update_camera(dt):
    targetY = cube["y"] - SCREEN_HEIGHT / 2 + cube["size"] / 2
    camera["y"] += (targetY - camera["y"]) * min(camera["speed"] * 0.01 * dt, 1)

# Draw the game elements
def draw_play():
    screen.fill(colors["bg"])
    pygame.draw.rect(screen, colors["green"], (cube["x"], cube["y"] - camera["y"], cube["size"], cube["size"]))
    for gap in gaps:
        pygame.draw.rect(screen, colors["red"], (0, gap["y"] - camera["y"], gap["x"], blockHeight))
        pygame.draw.rect(screen, colors["red"], (gap["x"] + gapWidth, gap["y"] - camera["y"], SCREEN_WIDTH - gap["x"] - gapWidth, blockHeight))
    score_text = font.render(str(score), True, colors["fg"])
    screen.blit(score_text, (SCREEN_WIDTH - 25, 0))

# Main game loop
while running:
    dt = clock.tick(60) / 1000
    mx, my = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if state == "menu":
                for button in buttons.values():
                    button.click(mx, my)
            elif state == "gameover":
                buttons["restart"].click(mx, my)
                buttons["mainmenu"].click(mx, my)
        elif event.type == pygame.KEYDOWN:
            if state == "play":
                if lastKeyPressTime < keyCooldown:
                    continue
                lastKeyPressTime = 0
                if event.key in [pygame.K_RIGHT, pygame.K_d]:
                    cube["xSpeed"] = 0
                    cube["speed"] = cube["jump"]
                    cube["xSpeed"] += cube["xAcceleration"]
                elif event.key in [pygame.K_LEFT, pygame.K_a]:
                    cube["xSpeed"] = 0
                    cube["speed"] = cube["jump"]
                    cube["xSpeed"] -= cube["xAcceleration"]

    if state == "intro":
        timer += dt
        if timer > 1:
            state = "menu"
    elif state == "play":
        lastKeyPressTime += dt
        update_cube(dt)
        update_camera(dt)
        update_obstacles()
    screen.fill(colors["bg"])
    if state == "intro":
        alpha = min(timer / 0.2, 1) * 255
        text_surface = font.render("Kaplan Games", True, colors["fg"])
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
        screen.blit(text_surface, text_rect)
    elif state == "menu":
        text_surface = font.render("Hobbalaaa", True, colors["fg"])
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, 200))
        screen.blit(text_surface, text_rect)
        buttons["play"].draw()
        buttons["exit"].draw()
    elif state == "play":
        draw_play()
    elif state == "gameover":
        text_surface = font.render("Game Over", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, 50))
        screen.blit(text_surface, text_rect)
        buttons["restart"].draw()
        buttons["mainmenu"].draw()
    pygame.display.flip()

pygame.quit()
sys.exit()
