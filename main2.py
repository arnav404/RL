import matplotlib.pyplot
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

from brain import DQNAgent
from game import Game

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Initialize the environment and the agent
env = Game()
# 4 states and 4 possible actions
agent = DQNAgent(state_dim=4, action_dim=4, device=device)

n_episodes = 10000
accuracies = []
plt = matplotlib.pyplot
plt.plot([])
plt.ion()
plt.show()

step_count = 0

for e in range(n_episodes):

    # Get the (x,y) of the player and the goal
    state = env.get_state()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                break

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        step_count += 1
        if step_count % 10 == 0:
            agent.replay()

        env.render()
        env.clock.tick(1000)  # Control the speed

    print(f"Episode {e+1}/{n_episodes}, Total Reward: {total_reward}")
    if env.stupidCount != -1:
        accuracies.append(env.stupidCount)
    print(
        f"{env.successes}/{env.successes+env.failures}:{env.successes/(env.successes+env.failures)}")

    plt.pause(0.1)
    plt.plot(accuracies)

    env.reset()

# Save the model after training
torch.save(agent.model.state_dict(), "dqn_model.pth")

env.close()
