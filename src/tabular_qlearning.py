# ===============================
# Tabular Q-Learning - Taxi-v3
# ===============================


import gymnasium as gym
import numpy as np
import random
import plot_utility as pu


# -------------------------------
# Environment
# -------------------------------
env = gym.make("Taxi-v3")
STATE_SIZE = env.observation_space.n   # 500
ACTION_SIZE = env.action_space.n       # 6


class TabularQLearning:
    def __init__(self,
                 learning_rate=0.7,
                 learning_rate_min=0.3,
                 learning_rate_decay=0.99998,
                 discount_factor=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.9995):

        self.lr = learning_rate
        self.lr_min = learning_rate_min
        self.lr_decay = learning_rate_decay

        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((STATE_SIZE, ACTION_SIZE))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(ACTION_SIZE)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_learning_rate(self):
        self.lr = max(self.lr * self.lr_decay, self.lr_min)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# -------------------------------
# Training Q-learning
# -------------------------------
def train_tabular(episodes=50000):
    agent = TabularQLearning()
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        agent.decay_learning_rate()
        rewards.append(total_reward)

        if ep % 3000 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {ep} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Learning "
                  f"Rate: {agent.lr:.3f}")

    pu.plot_rewards(rewards, title="Tabular Q-Learning Training")
    return agent


# -------------------------------
# Test Q-learning
# -------------------------------
def test_tabular(agent, episodes=100):
    results = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            agent.epsilon = 0.0  # Disable Exploration
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results.append(total_reward)
        print(f"Test episode {ep} | Reward: {total_reward}")

    pu.plot_rewards(results, title="Tabular Q-Learning Test")
