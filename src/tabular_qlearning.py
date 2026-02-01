# ===============================
# Tabular Q-Learning - Taxi-v3
# ===============================


import gymnasium as gym
import numpy as np
import random
import plot_utility as pu


ENV_ID = "Taxi-v3"
LR = 0.7
LR_MIN = 0.3
LR_DECAY = 0.99998
GAMMA = 0.99  # Discount: how much we value future rewards
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99988
DECAY_LR = False  # LR stays fixed at 0.7 in current setup


# -------------------------------
# Agent: Q-table + epsilon-greedy + Bellman update
# -------------------------------
class TabularQLearning:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = LR
        self.lr_min = LR_MIN
        self.lr_decay = LR_DECAY
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.use_lr_decay = DECAY_LR
        # Q(s,a): one value per (state, action). Initialized to 0.
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, action_mask=None):
        """
        Epsilon-greedy: with prob epsilon explore (random), else exploit (best Q).
        Action mask (Taxi-v3): restrict to actions that actually change the state.
        """
        if action_mask is not None and action_mask.sum() > 0:
            valid_actions = np.where(action_mask)[0]
            if random.random() < self.epsilon:
                return int(random.choice(valid_actions))
            q_valid = self.q_table[state, valid_actions]
            return int(valid_actions[np.argmax(q_valid)])
        # Fallback when no mask: original behavior (environments without action_mask).
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        # Bellman update: Q(s,a) <- Q(s,a) + lr * (TD_target - Q(s,a))
        # If done: target = r. Else: target = r + gamma * max_a' Q(s',a').
        if done:
            td_target = reward
        else:
            best_next = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_learning_rate(self):
        if self.use_lr_decay:
            self.lr = max(self.lr * self.lr_decay, self.lr_min)

    def decay_epsilon(self):
        # After each episode: reduce exploration (exponential decay toward epsilon_min).
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# -------------------------------
# Training: interact with env, update Q every step, decay epsilon per episode
# -------------------------------
def train_tabular(episodes=50000):
    env = gym.make(ENV_ID)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = TabularQLearning(state_size, action_size)
    rewards = []
    epsilon_floor_episode = None

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_mask = info.get("action_mask", None)
            action = agent.choose_action(state, action_mask=action_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        agent.decay_learning_rate()
        if epsilon_floor_episode is None and agent.epsilon <= agent.epsilon_min:
            epsilon_floor_episode = ep
        rewards.append(total_reward)

        if ep % 3000 == 0:
            avg_reward = np.mean(rewards[-3000:])
            print(
                f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | "
                f"Learning Rate: {agent.lr:.3f}"
            )

    pu.plot_training_curve(
        rewards,
        title="Tabular Q-Learning Training",
        epsilon_floor_episode=epsilon_floor_episode,
    )
    env.close()
    return agent


# -------------------------------
# Test: greedy policy (epsilon=0), no exploration, report mean/std
# -------------------------------
def test_tabular(agent, episodes=100):
    env = gym.make(ENV_ID)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation for evaluation
    results = []

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_mask = info.get("action_mask", None)
            action = agent.choose_action(state, action_mask=action_mask)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results.append(total_reward)

    results = np.array(results)
    print(
        f"Tabular Q-Learning Test (n={episodes}): "
        f"mean={results.mean():.2f}, std={results.std():.2f}, min={results.min()}, max={results.max()}"
    )
    pu.plot_test_distribution(results, title="Tabular Q-Learning Test")
    env.close()
    agent.epsilon = original_epsilon
    return results
