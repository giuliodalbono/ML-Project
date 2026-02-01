import numpy as np

import plot_utility as pu
from tabular_qlearning import train_tabular, test_tabular
from dqn_agent import train_dqn, test_dqn


TABULAR_TRAINING_EPISODES = 30_000
TABULAR_TESTING_EPISODES = 100

DQN_TRAINING_EPISODES = 10_000
DQN_TESTING_EPISODES = 100
DQN_RUNS = 3  # Multiple runs to assess DQN stability (no fixed seed)


def main():
    # --- Tabular Q-Learning: single train + test ---
    print("=== Training Tabular Q-Learning ===")
    tab_agent = train_tabular(episodes=TABULAR_TRAINING_EPISODES)
    print("\n=== Testing Tabular Q-Learning ===")
    test_tabular(tab_agent, episodes=TABULAR_TESTING_EPISODES)

    dqn_test_means = []
    dqn_all_results = []

    # --- DQN: multiple independent runs for reproducibility/stability ---
    for run_idx in range(DQN_RUNS):
        print(f"\n=== Training DQN (run {run_idx + 1}/{DQN_RUNS}) ===")
        dqn_agent = train_dqn(episodes=DQN_TRAINING_EPISODES, run=run_idx + 1)
        print(f"\n=== Testing DQN (run {run_idx + 1}/{DQN_RUNS}) ===")
        results = test_dqn(dqn_agent, episodes=DQN_TESTING_EPISODES, run=run_idx + 1)
        dqn_test_means.append(results.mean())
        dqn_all_results.append(results)

    # --- Aggregate DQN results and summary plots ---
    dqn_all_results = np.concatenate(dqn_all_results)
    pu.plot_test_distribution(
        dqn_all_results,
        title="DQN Test (All Runs)",
    )
    pu.plot_multirun_summary(
        dqn_test_means,
        title="DQN Multi-Run Summary",
    )

    print(
        "\nDQN Multi-Run Summary: "
        f"mean={np.mean(dqn_test_means):.2f}, "
        f"std={np.std(dqn_test_means):.2f} "
        f"(per-run means={['{:.2f}'.format(m) for m in dqn_test_means]})"
    )


if __name__ == "__main__":
    main()
