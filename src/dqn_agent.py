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


ENV_ID = "Taxi-v3"
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_STEPS = 50000
MEMORY_SIZE = 50000
BATCH_SIZE = 64
WARM_UP_STEPS = 1000   # No training until buffer has enough samples
GRAD_CLIP_NORM = 1.0   # Stabilize training
TARGET_UPDATE_STEPS = 500  # Copy policy -> target network periodically
TRAIN_FREQUENCY = 1
PLOT_TRAINING = True


# -------------------------------
# Q-network: state (one-hot 500) -> hidden 128 -> hidden 128 -> 6 Q-values
# -------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        hidden1 = 128
        hidden2 = 128
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_size),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# DQN Agent
# -------------------------------
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_start = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay_steps = EPSILON_DECAY_STEPS
        self.batch_size = BATCH_SIZE
        self.warm_up_steps = WARM_UP_STEPS
        self.gradient_clip_norm = GRAD_CLIP_NORM
        self.target_update_steps = TARGET_UPDATE_STEPS

        self.device = torch.device("cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()  # Huber: robust to outliers

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.steps_done = 0
        self.learn_steps = 0

    def one_hot(self, state):
        # Discrete state index -> 500-dim one-hot for the network input
        vec = np.zeros(self.state_size, dtype=np.float32)
        vec[state] = 1.0
        return vec

    def select_action(self, state, action_mask=None):
        # Epsilon-greedy; if mask given, only valid actions.
        if action_mask is not None and action_mask.sum() > 0:
            valid_actions = np.where(action_mask)[0]
            if random.random() < self.epsilon:
                return int(random.choice(valid_actions))
            state_t = torch.tensor(self.one_hot(state)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_t).squeeze(0).cpu().numpy()
            q_masked = np.where(action_mask, q_values, -np.inf)
            return int(np.argmax(q_masked))
        # Fallback when no mask (environments without action_mask).
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_t = torch.tensor(self.one_hot(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def store(self, transition):
        # Append (s,a,r,s',done,next_mask) to replay buffer.
        self.memory.append(transition)

    def update_epsilon(self):
        # Linear decay over steps (not per episode): epsilon decreases as we see more data
        if self.epsilon_decay_steps <= 0:
            return
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - (self.epsilon_start - self.epsilon_min) * frac,
        )

    def train_step(self):
        """
        One gradient step: sample batch from replay, compute TD target with TARGET
        network, minimize Huber loss, optional grad clip, sync target every N steps.
        """
        if len(self.memory) < self.batch_size or self.steps_done < self.warm_up_steps:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, next_action_masks = zip(*batch)

        # Batch of one-hot states
        states = torch.from_numpy(
            np.array([self.one_hot(s) for s in states], dtype=np.float32)
        ).to(self.device)
        next_states = torch.from_numpy(
            np.array([self.one_hot(s) for s in next_states], dtype=np.float32)
        ).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Q(s,a) from current policy network
        q_values = self.model(states).gather(1, actions).squeeze(1)
        # TD target: r + gamma * max_a' Q_target(s',a')  (0 if done)
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            use_masks = all(mask is not None for mask in next_action_masks)
            if use_masks:
                mask = torch.from_numpy(
                    np.array(next_action_masks, dtype=np.float32)
                ).to(self.device)
                zero_rows = mask.sum(dim=1, keepdim=True) == 0
                masked_next_q = next_q_values.masked_fill(mask == 0, -1e9)
                next_q_values = masked_next_q.max(1)[0]
                if zero_rows.any():
                    next_q_values = torch.where(
                        zero_rows.squeeze(1),
                        torch.zeros_like(next_q_values),
                        next_q_values,
                    )
            else:
                next_q_values = next_q_values.max(1)[0]

        target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clip_norm
            )
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()


# -------------------------------
# Training: step env -> store transition -> update epsilon -> train_step each step
# -------------------------------
def train_dqn(episodes=10000, run=None):
    env = gym.make(ENV_ID)
    state_size = env.observation_space.n
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    rewards = []
    total_steps = 0
    epsilon_floor_episode = None

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action_mask = info.get("action_mask", None)
            action = agent.select_action(state, action_mask=action_mask)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_action_mask = info.get("action_mask", None)
            agent.store((state, action, reward, next_state, done, next_action_mask))
            total_steps += 1
            agent.steps_done += 1
            agent.update_epsilon()
            if (
                epsilon_floor_episode is None
                and agent.epsilon <= agent.epsilon_min + 1e-12
            ):
                epsilon_floor_episode = ep
            if total_steps % TRAIN_FREQUENCY == 0:
                agent.train_step()

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        if ep % 100 == 0:
            avg_reward = np.mean(rewards[-50:])
            min_reward = np.min(rewards[-50:])
            max_reward = np.max(rewards[-50:])
            print(f"Episode {ep} | Avg Reward: {avg_reward:.2f} | Min: {min_reward} | Max: {max_reward} | Epsilon: {agent.epsilon:.3f}")

    if PLOT_TRAINING:
        pu.plot_training_curve(
            rewards,
            title="DQN Training run " + str(run),
            epsilon_floor_episode=epsilon_floor_episode,
        )
    env.close()
    return agent


# -------------------------------
# Test: epsilon=0 (greedy), collect rewards over episodes, plot distribution
# -------------------------------
def test_dqn(agent, episodes=100, run=None):
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
            action = agent.select_action(state, action_mask=action_mask)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results.append(total_reward)

    results = np.array(results)
    mean = results.mean()
    std = results.std()
    print(
        f"DQN Test (n={episodes}): "
        f"mean={mean:.2f}, std={std:.2f}, min={results.min()}, max={results.max()}"
    )

    test_title = f"DQN Test run {run}" if run is not None else "DQN Test"
    pu.plot_test_distribution(results, title=test_title)
    env.close()
    agent.epsilon = original_epsilon
    return results
