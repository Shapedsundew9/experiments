"""https://chat.openai.com/share/7c69b276-1226-4a99-8703-e7e96068111a"""
import pygame
import random
from time import sleep

# Initialize Pygame
pygame.init()

# Set up the display
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Evolution Simulation")

# Define colors and constants
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BROWN = (165, 42, 42)
BLUE = (0, 0, 255)
GRAY: tuple[int, int, int] = (
    128,
    128,
    128,
)  # Added by Sam (Github copilot suggested this)
GRID_SIZE = 20
SMELL_RANGE = 10


# Plant class
class Plant:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.stage = "unripe"  # unripe, ripe, rotten

    def update(self):
        # Randomly change the stage of the plant
        if random.random() < 0.1:
            if self.stage == "unripe":
                self.stage = "ripe"
            elif self.stage == "ripe":
                self.stage = "rotten"
            elif self.stage == "rotten":  # Added by Sam (Github copilot suggested this)
                self.stage = "unripe"
            elif self.stage == "eaten":
                self.stage = "growing"  # Added by Sam
                self.x = random.randint(
                    0, window_size[0] // GRID_SIZE - 1
                )  # Added by Sam
                self.y = random.randint(
                    0, window_size[1] // GRID_SIZE - 1
                )  # Added by Sam  (Github copilot suggested this)
            elif random.random() < 0.1:
                self.stage = "unripe"  # Added by Sam (Github copilot suggested this)

    def draw(self, screen):
        if self.stage == "unripe":
            color = GREEN
        elif self.stage == "ripe":
            color = YELLOW
        elif self.stage == "rotten":
            color = BROWN
        elif self.stage == "eaten":
            color = BLACK  # Added by Sam (Github copilot suggested this)
        else:
            color = GRAY  # Added by Sam
        pygame.draw.rect(
            screen,
            color,
            (self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE),
        )


# Creature class
class Creature:
    def __init__(self, x, y, energy, smell_range):
        self.x = x
        self.y = y
        self.energy = energy
        self.smell_range = smell_range

    def move_towards_food(self, plants):
        closest_ripe_fruit = None
        min_distance = float("inf")
        for plant in plants:
            if plant.stage == "ripe":
                distance = abs(plant.x - self.x) + abs(plant.y - self.y)
                if distance < min_distance and distance <= self.smell_range:
                    closest_ripe_fruit = plant
                    min_distance = distance

        if closest_ripe_fruit:
            dx = (
                1
                if closest_ripe_fruit.x > self.x
                else -1
                if closest_ripe_fruit.x < self.x
                else 0
            )
            dy = (
                1
                if closest_ripe_fruit.y > self.y
                else -1
                if closest_ripe_fruit.y < self.y
                else 0
            )
            self.x += dx
            self.y += dy
        else:
            self.random_move()

    def random_move(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dx, dy = random.choice(directions)
        self.x = (self.x + dx) % (window_size[0] // GRID_SIZE)
        self.y = (self.y + dy) % (window_size[1] // GRID_SIZE)
        self.energy -= 1  ## Added by Sam (Github copilot suggested this)
        if self.energy < 0:
            self.x = random.randint(0, window_size[0] // GRID_SIZE - 1)  # Added by Sam
            self.y = random.randint(0, window_size[1] // GRID_SIZE - 1)  # Added by Sam
            self.energy = 20  # Added by Sam

    def eat(self, plants):
        for plant in plants:
            if plant.x == self.x and plant.y == self.y and plant.stage == "ripe":
                self.energy += 10  # Energy gained from eating a ripe fruit
                plant.stage = "eaten"

    def breed(self, other):
        if self.energy > 20 and other.energy > 20:
            x = (self.x + other.x) // 2
            y = (self.y + other.y) // 2
            new_smell_range = random.choice(
                [self.smell_range, other.smell_range]
            ) + random.choice([-1, 0, 1])
            new_creature = Creature(x, y, 10, max(1, new_smell_range))
            self.energy -= 10
            other.energy -= 10
            return new_creature
        return None

    def draw(self, screen):
        color = self.energy * 8 if self.energy < 32 else 255
        pygame.draw.rect(
            screen,
            (0, 0, color),
            (self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE),
        )


# Initialize plants and creatures
plants = [
    Plant(
        random.randint(0, window_size[0] // GRID_SIZE - 1),
        random.randint(0, window_size[1] // GRID_SIZE - 1),
    )
    for _ in range(50)
]
creatures = [
    Creature(
        random.randint(0, window_size[0] // GRID_SIZE - 1),
        random.randint(0, window_size[1] // GRID_SIZE - 1),
        20,
        SMELL_RANGE,
    )
    for _ in range(10)
]

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(BLACK)

    # Update and draw plants
    for plant in plants:
        plant.update()
        plant.draw(screen)

    # Move, eat, breed and draw creatures
    for i, creature in enumerate(creatures):
        creature.move_towards_food(plants)
        creature.eat(plants)
        for other in creatures[i + 1 :]:
            offspring = creature.breed(other)
            if offspring:
                creatures.append(offspring)
        creature.draw(screen)

    # Update the display
    pygame.display.flip()
    # sleep(0.1)  # Added by Sam

# Quit Pygame
pygame.quit()
