import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(rewards, title="", window=100, save_path=None, show=True):
    rewards = np.asarray(rewards)
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5, label="Raw")

    smoothed = np.array(
        [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    )
    plt.plot(smoothed, label=f"Mean (w={window})", linewidth=2)

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()

    plt.close()
