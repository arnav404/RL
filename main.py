import pygame
import numpy as np
import random
from collections import deque

pygame.init()


PIXEL_WIDTH = 40
PIXEL_HEIGHT = 20
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800 * PIXEL_HEIGHT / PIXEL_WIDTH

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

state = np.zeros((PIXEL_WIDTH, PIXEL_HEIGHT))

snake = deque()

directions = {
    'w': [0, -1],
    's': [0, 1],
    'a': [-1, 0],
    'd': [1, 0]
}

currentDirection = [0, -1]
directionInput = [0, -1]
fruitPosition = [0, 0]


def restart():
    global state
    state = np.zeros((PIXEL_WIDTH, PIXEL_HEIGHT))
    global currentDirection
    global fruitPosition
    currentDirection = [0, -1]
    snake.clear()
    snake.append([10, 10])
    snake.append([10, 11])
    snake.append([10, 12])
    for segment in snake:
        state[segment[0]][segment[1]] = 1
    fruitPosition = [15, 5]
    state[fruitPosition[0]][fruitPosition[1]] = 2


isRunning = True
fruitEaten = False
frameCounter = 0

restart()

# Game loop
while isRunning:

    # Check to see if we should move snake
    if frameCounter % 30 == 0:

        # Check if current direction is possible
        if [directionInput[0]+currentDirection[0], directionInput[1]+currentDirection[1]] != [0, 0]:
            currentDirection = directionInput

        # Pop the snake's tail
        if not fruitEaten:
            state[snake[-1][0]][snake[-1][1]] = 0
            snake.pop()
        else:
            fruitEaten = False

        # Move the snake's head
        head = snake[0]
        snake.appendleft([head[0]+currentDirection[0],
                          head[1]+currentDirection[1]])

        # Check if we're out of bounds
        head = snake[0]
        if (head[0] >= PIXEL_WIDTH or head[0] < 0 or head[1] >= PIXEL_HEIGHT or head[1] < 0):
            restart()

        # Check if we've collided with ourself
        elif head in list(snake)[1:]:
            restart()

        else:

            # Check if we ate the fruit
            if fruitPosition in snake:
                fruitEaten = True
                state[fruitPosition[0]][fruitPosition[1]] = 1
                fruitPosition = [random.randint(
                    0, PIXEL_WIDTH-1), random.randint(0, PIXEL_HEIGHT-1)]

            # Change State
            for segment in snake:
                state[segment[0]][segment[1]] = 1

            state[fruitPosition[0]][fruitPosition[1]] = 2

    # Increase frame count
    frameCounter += 1

    # Draw state onto screen
    for indi, i in enumerate(state):
        for indj, j in enumerate(i):
            if state[indi][indj] == 1:
                color = (255, 255, 255)
            elif state[indi][indj] == 2:
                color = (255, 0, 0)
            else:
                color = (0, 0, 128)
            rect = pygame.Rect(SCREEN_WIDTH/PIXEL_WIDTH*indi, SCREEN_HEIGHT /
                               PIXEL_HEIGHT*indj, SCREEN_WIDTH/PIXEL_WIDTH-1, SCREEN_HEIGHT/PIXEL_HEIGHT-1)
            pygame.draw.rect(screen, color, rect)

    # Input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            isRunning = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                directionInput = directions['a']
            elif event.key == pygame.K_s:
                directionInput = directions['s']
            elif event.key == pygame.K_d:
                directionInput = directions['d']
            elif event.key == pygame.K_w:
                directionInput = directions['w']

    pygame.display.update()

pygame.quit()
