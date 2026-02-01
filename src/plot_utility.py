"""
Plotting utilities for training curves, test reward distributions, and multi-run summary.
Figures saved in resources/graphs/; window size for smoothing scales with data length.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_SHOW = True
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "resources" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_WINDOW_DIVISOR = 50
TRAIN_WINDOW_MIN = 50
TRAIN_WINDOW_MAX = 1000
TEST_HIST_BINS = 20


def _safe_filename(title, fallback):
    """Sanitize title for use as filename (no special chars, single spaces)."""
    base = title.strip() if title else fallback
    base = re.sub(r"[^\w\s\-().]", "_", base)
    base = re.sub(r"\s+", " ", base).strip()
    return f"{base}.png"


def _save_current_plot(title, fallback):
    filename = _safe_filename(title, fallback)
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150)


def _save_figure(fig, title, fallback):
    filename = _safe_filename(title, fallback)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150)


def plot_training_curve(rewards, title="", epsilon_floor_episode=None):
    """
    Raw episode rewards + trailing mean (adaptive window ~2% of length).
    Optional vertical line at episode where epsilon hits minimum (exploration floor).
    """
    rewards = np.asarray(rewards)
    window = int(
        np.clip(len(rewards) // TRAIN_WINDOW_DIVISOR, TRAIN_WINDOW_MIN, TRAIN_WINDOW_MAX)
    )
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Raw")

    smoothed = np.array(
        [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    )
    plt.plot(smoothed, label=f"Mean (w={window})", linewidth=2)

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")

    if epsilon_floor_episode is not None:
        plt.axvline(
            epsilon_floor_episode,
            color="purple",
            linestyle=":",
            linewidth=1,
            label="Epsilon floor",
        )
    plt.legend()
    plt.tight_layout()

    _save_current_plot(title, "Training Curve")
    if PLOT_SHOW:
        plt.show()

    plt.close()


def plot_test_distribution(results, title=""):
    """Histogram + box plot of test episode returns; mean and std in box."""
    results = np.asarray(results)
    mean = results.mean()
    std = results.std()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(results, bins=TEST_HIST_BINS, edgecolor="black", alpha=0.8)
    axes[0].axvline(mean, color="C1", linewidth=1)
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(results, vert=False)
    axes[1].set_title("Box Plot")
    axes[1].set_xlabel("Reward")

    stats_text = f"mean={mean:.2f}\nstd={std:.2f}"
    axes[1].text(
        0.98,
        0.95,
        stats_text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(title)
    fig.tight_layout()

    _save_figure(fig, title, "Test Distribution")
    if PLOT_SHOW:
        plt.show()

    plt.close(fig)


def plot_multirun_summary(run_means, title=""):
    """Per-run test mean rewards + overall mean and ±1 std band (DQN stability)."""
    run_means = np.asarray(run_means, dtype=np.float32)
    x = np.arange(len(run_means))
    mean = run_means.mean()
    std = run_means.std()

    plt.figure(figsize=(8, 4))
    plt.scatter(x, run_means, label="Run mean")
    plt.axhline(mean, color="C1", label="Mean")
    plt.fill_between(
        [-0.5, len(run_means) - 0.5],
        mean - std,
        mean + std,
        color="C1",
        alpha=0.2,
        label="±1 std",
    )

    plt.title(title if title else "Multi-Run Summary")
    plt.xlabel("Run index")
    plt.ylabel("Test mean reward")
    plt.xticks(x, [str(i) for i in x])
    plt.text(
        0.98,
        0.95,
        f"mean={mean:.2f}\nstd={std:.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.legend()
    plt.tight_layout()

    _save_current_plot(title, "Multi-Run Summary")
    if PLOT_SHOW:
        plt.show()

    plt.close()
