# ===============================
# Deep Q-Network (DQN) - Taxi-v3
# ===============================


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import plot_utility as pu


# -------------------------------
# Environment
# -------------------------------
env = gym.make("Taxi-v3")
STATE_SIZE = env.observation_space.n   # 500
ACTION_SIZE = env.action_space.n       # 6

# -------------------------------
# Neural Network
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------------
# DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self,
                 gamma=0.99,
                 lr=5e-4,
                 epsilon=1.0,
                 epsilon_min=0.05,
                 epsilon_decay_steps=50000,
                 memory_size=10000,
                 batch_size=128):

        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=memory_size)
        self.steps_done = 0

    def one_hot(self, state):
        vec = np.zeros(self.state_size, dtype=np.float32)
        vec[state] = 1.0
        return vec

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state = torch.tensor(self.one_hot(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store(self, transition):
        self.memory.append(transition)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(
            np.array([self.one_hot(s) for s in states], dtype=np.float32)
        ).to(self.device)
        next_states = torch.from_numpy(
            np.array([self.one_hot(s) for s in next_states], dtype=np.float32)
        ).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0].detach()

        target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        self._decay_epsilon()

        return loss.item()

    def _decay_epsilon(self):
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - \
                           (self.epsilon_start - self.epsilon_min) * \
                           (self.steps_done / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_min


# -------------------------------
# Training
# -------------------------------
def train_dqn(episodes=500):
    agent = DQNAgent()
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store((state, action, reward, next_state, done))
            agent.train_step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if ep % 100 == 0:
            avg_reward = np.mean(rewards[-50:])
            min_reward = np.min(rewards[-50:])
            max_reward = np.max(rewards[-50:])
            print(f"Episode {ep} | Avg: {avg_reward:.2f} | Min: {min_reward} | Max: {max_reward} | Epsilon: {agent.epsilon:.3f}")

    pu.plot_rewards(rewards, title="DQN Training")
    return agent


# -------------------------------
# Test
# -------------------------------
def test_dqn(agent, episodes=100):
    agent.epsilon = 0.0  # Disable Exploration
    results = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_vec = torch.tensor(agent.one_hot(state)).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = torch.argmax(agent.model(state_vec)).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results.append(total_reward)
        print(f"Test episode {ep} | Reward: {total_reward}")

    pu.plot_rewards(results, title="DQN Test")
