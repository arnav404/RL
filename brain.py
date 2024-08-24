import matplotlib.pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib

# This is a feedforward neural network


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.05
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.losses = []

        # Main Q-Network
        self.model = DQN(state_dim, action_dim)

        # Target Q-Network (a clone of the main network)
        self.target_model = DQN(state_dim, action_dim)
        # Initialize with the same weights
        self.target_model.load_state_dict(self.model.state_dict())
        # Set to evaluation mode since we don't train this model directly
        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_frequency = 100  # Update target network every 1000 steps
        self.step_count = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):

        losses = []

        if len(self.memory) < self.batch_size:
            return

        self.step_count += 1

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * \
                    torch.max(self.target_model(next_state)).item()

            target_f = self.model(state)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            losses.append(loss.item())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        # Update the target network at regular intervals
        if self.step_count % self.update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(sum(losses)/len(losses))
        self.losses.append(sum(losses)/len(losses))

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
