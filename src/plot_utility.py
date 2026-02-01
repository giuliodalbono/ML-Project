import matplotlib.pyplot as plt
import numpy as np


# -------------------------------
# Plot
# -------------------------------
def plot_rewards(rewards, title=""):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.5)
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()
