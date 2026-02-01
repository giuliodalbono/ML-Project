# ===============================
# Tabular Q-Learning - Taxi-v3
# ===============================


import gymnasium as gym
import numpy as np
import random
import plot_utility as pu


# -------------------------------
# Agent
# -------------------------------
class TabularQLearning:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.7,
        learning_rate_min=0.3,
        learning_rate_decay=0.99998,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99988,
        decay_learning_rate=True,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.lr_min = learning_rate_min
        self.lr_decay = learning_rate_decay
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_lr_decay = decay_learning_rate

        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, action_mask=None):
        """
        Epsilon-greedy action selection. If action_mask is provided (e.g. from
        Taxi-v3 info), restrict choices to actions that change the state,
        avoiding no-ops and speeding up learning.
        """
        if action_mask is not None and action_mask.sum() > 0:
            # Changed: restrict to actions that actually change the state (Taxi action_mask)
            valid_actions = np.where(action_mask)[0]
            if random.random() < self.epsilon:
                return int(random.choice(valid_actions))
            # Changed: take best Q among valid actions only (exploitation respects mask).
            q_valid = self.q_table[state, valid_actions]
            return int(valid_actions[np.argmax(q_valid)])
        # Fallback when no mask: original behavior (environments without action_mask).
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_learning_rate(self):
        if self.use_lr_decay:
            self.lr = max(self.lr * self.lr_decay, self.lr_min)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# -------------------------------
# Training
# -------------------------------
def train_tabular(
    episodes=50000,
    seed=None,
    decay_learning_rate=True,
    env_id="Taxi-v3",
    plot_save_path=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = gym.make(env_id)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = TabularQLearning(
        state_size,
        action_size,
        decay_learning_rate=decay_learning_rate,
    )
    rewards = []

    for ep in range(episodes):
        # Changed: reset with seed only on first episode so initial state is reproducible;
        # later episodes stay stochastic.
        state, info = env.reset(seed=seed if ep == 0 else None)
        done = False
        total_reward = 0

        while not done:
            # Changed: use action_mask from info when available (Taxi-v3) so only pick actions that change the state.
            action_mask = info.get("action_mask", None)
            action = agent.choose_action(state, action_mask=action_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        agent.decay_learning_rate()
        rewards.append(total_reward)

        if ep % 3000 == 0:
            avg_reward = np.mean(rewards[-3000:])
            print(
                f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f} | "
                f"Learning Rate: {agent.lr:.3f}"
            )

    pu.plot_rewards(
        rewards,
        title="Tabular Q-Learning Training",
        save_path=plot_save_path,
        show=(plot_save_path is None),
    )
    env.close()
    return agent


# -------------------------------
# Test
# -------------------------------
def test_tabular(
    agent,
    episodes=100,
    seed=None,
    env_id="Taxi-v3",
    plot_save_path=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = gym.make(env_id)
    agent.epsilon = 0.0  # Disable exploration during test
    results = []

    for ep in range(episodes):
        state, info = env.reset(seed=seed if ep == 0 else None)
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

    pu.plot_rewards(
        results,
        title="Tabular Q-Learning Test",
        save_path=plot_save_path,
        show=(plot_save_path is None),
    )

    env.close()
