import pygame
import random

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
GRID_SIZE = 40
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("DQN Snake")
        self.clock = pygame.time.Clock()
        self.successes = 0
        self.failures = 0
        self.steps = 0
        self.stupidCount = 0
        self.reset()

    def reset(self):
        self.steps = 0
        self.stupidCount = 0
        self.player_pos = [random.randint(0, (SCREEN_WIDTH//GRID_SIZE)-1) * GRID_SIZE,
                           random.randint(0, (SCREEN_HEIGHT//GRID_SIZE)-1) * GRID_SIZE]
        self.goal_pos = [random.randint(0, (SCREEN_WIDTH//GRID_SIZE)-1) * GRID_SIZE,
                         random.randint(0, (SCREEN_HEIGHT//GRID_SIZE)-1) * GRID_SIZE]
        self.done = False

    def step(self, action):
        self.steps += 1
        if action == 0:  # up
            self.player_pos[1] -= GRID_SIZE
        elif action == 1:  # down
            self.player_pos[1] += GRID_SIZE
        elif action == 2:  # left
            self.player_pos[0] -= GRID_SIZE
        elif action == 3:  # right
            self.player_pos[0] += GRID_SIZE

        # Check boundaries and update position
        if self.player_pos[0] < 0 or self.player_pos[0] >= SCREEN_WIDTH or self.player_pos[1] < 0 or self.player_pos[1] >= SCREEN_HEIGHT:
            self.stupidCount += 1
            self.done = False
            reward = -1  # Strong penalty for hitting the wall
        else:
            self.done = False
            reward = -0.1  # Small penalty for each step to encourage reaching the goal

        # Check if the goal is reached
        if self.player_pos == self.goal_pos:
            reward = 10  # Reward for reaching the goal
            self.successes += 1
            self.stupidCount = -1
            self.done = True
        elif self.steps == 500:
            self.failures += 1
            self.done = True

        # Check boundaries
        self.player_pos[0] = max(
            0, min(self.player_pos[0], SCREEN_WIDTH-GRID_SIZE))
        self.player_pos[1] = max(
            0, min(self.player_pos[1], SCREEN_HEIGHT-GRID_SIZE))

        return self.get_state(), reward, self.done

    def get_state(self):
        return [self.player_pos[0], self.player_pos[1], self.goal_pos[0], self.goal_pos[1]]

    def render(self):
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, RED,
                         (*self.player_pos, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(self.screen, GREEN,
                         (*self.goal_pos, GRID_SIZE, GRID_SIZE))
        pygame.display.flip()

    def close(self):
        pygame.quit()
