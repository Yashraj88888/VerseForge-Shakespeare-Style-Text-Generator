"""
utils.py
VerseForge: Shakespeare-Style Autoregressive Text Generation
Utility functions for reproducibility, vocabulary I/O, quantitative metrics
(loss, perplexity, accuracy), qualitative n-gram diversity metrics
(Distinct-1, Distinct-2, repetition ratio), and publication-grade visualization.
"""

import os
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Ensure headless matplotlib backend
plt.switch_backend("Agg")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and tensorflow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_vocab(token_to_id: Dict[str, int], filepath: str) -> None:
    """Saves the vocabulary mapping to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, indent=2, ensure_ascii=False)


def load_vocab(filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Loads vocabulary mapping from a JSON file and returns (token_to_id, id_to_token)."""
    with open(filepath, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    id_to_token = {int(idx): tok for tok, idx in token_to_id.items()}
    return token_to_id, id_to_token


def compute_perplexity(loss: float) -> float:
    """
    Computes Perplexity = exp(loss).
    For categorical cross-entropy with natural log base, perplexity is exp(CE).
    Caps at 1e6 to avoid overflow in early high-loss states.
    """
    try:
        return float(min(np.exp(loss), 1e6))
    except OverflowError:
        return 1e6


def compute_distinct_n(tokens: List[str], n: int = 1) -> float:
    """
    Computes Distinct-N ratio:
      Distinct-N = Number of unique n-grams / Total number of n-grams.
    Higher values signify greater lexical diversity.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_repetition_ratio(tokens: List[str], n: int = 2) -> float:
    """
    Computes n-gram repetition ratio:
      Repetition Ratio = 1.0 - Distinct-N
    Higher values indicate repetitive, looping text degeneration.
    """
    return 1.0 - compute_distinct_n(tokens, n=n)


def plot_training_history(
    histories: Dict[str, Any],
    output_dir: str = "results/plots",
) -> None:
    """
    Generates and saves comprehensive training curves:
      1. Training & Validation Loss for all four models
      2. Training & Validation Accuracy for all four models
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = {
        "Simple_RNN": "#1f77b4",
        "Bidirectional_RNN": "#ff7f0e",
        "LSTM": "#2ca02c",
        "GRU": "#d62728",
    }

    # 1. Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in histories.items():
        color = colors.get(name, None)
        epochs = range(1, len(hist["loss"]) + 1)
        ax1.plot(epochs, hist["loss"], label=f"{name} (Train)", color=color, linewidth=2)
        ax2.plot(
            epochs,
            hist["val_loss"],
            label=f"{name} (Val)",
            color=color,
            linestyle="--",
            linewidth=2,
        )

    ax1.set_title("Training Loss across Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_title("Validation Loss across Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2. Accuracy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in histories.items():
        color = colors.get(name, None)
        epochs = range(1, len(hist["accuracy"]) + 1)
        ax1.plot(epochs, hist["accuracy"], label=f"{name} (Train)", color=color, linewidth=2)
        ax2.plot(
            epochs,
            hist["val_accuracy"],
            label=f"{name} (Val)",
            color=color,
            linestyle="--",
            linewidth=2,
        )

    ax1.set_title("Training Accuracy across Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Next-Word Accuracy", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_title("Validation Accuracy across Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Next-Word Accuracy", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_curves.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()


def plot_model_comparisons(
    results: List[Dict[str, Any]],
    output_dir: str = "results/plots",
) -> None:
    """
    Plots comparison bar charts:
      1. Test Perplexity (lower is better)
      2. Test Accuracy (higher is better)
      3. Training Time (seconds)
    """
    os.makedirs(output_dir, exist_ok=True)
    models = [r["model"] for r in results]
    perplexities = [r["test_perplexity"] for r in results]
    accuracies = [r["test_accuracy"] * 100 for r in results]
    train_times = [r["training_time_sec"] for r in results]
    palette = ["#4A90E2", "#F5A623", "#7ED321", "#D0021B"]

    # 1. Perplexity Bar Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, perplexities, color=palette, width=0.55, edgecolor="black", alpha=0.85)
    plt.title("Held-out Test Perplexity Comparison (Lower is Better)", fontsize=12, fontweight="bold")
    plt.ylabel("Perplexity = exp(Loss)", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_perplexity_comparison.png"), dpi=300)
    plt.close()

    # 2. Next-Word Accuracy Bar Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=palette, width=0.55, edgecolor="black", alpha=0.85)
    plt.title("Held-out Test Next-Word Accuracy Comparison (Higher is Better)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_accuracy_comparison.png"), dpi=300)
    plt.close()

    # 3. Training Time Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, train_times, color=palette, width=0.55, edgecolor="black", alpha=0.85)
    plt.title("Training Time Comparison (Seconds)", fontsize=12, fontweight="bold")
    plt.ylabel("Elapsed Time (s)", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"), dpi=300)
    plt.close()


def plot_temperature_diversity(
    temp_metrics: Dict[str, Dict[float, Dict[str, float]]],
    output_dir: str = "results/plots",
) -> None:
    """
    Plots Distinct-1, Distinct-2, and Repetition Ratio across sampling temperatures (0.5, 0.8, 1.2).
    """
    os.makedirs(output_dir, exist_ok=True)
    temperatures = [0.5, 0.8, 1.2]
    colors = {
        "Simple_RNN": "#1f77b4",
        "Bidirectional_RNN": "#ff7f0e",
        "LSTM": "#2ca02c",
        "GRU": "#d62728",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("distinct_1", "Distinct-1 (Unigram Diversity)", axes[0]),
               ("distinct_2", "Distinct-2 (Bigram Diversity)", axes[1]),
               ("repetition_2", "Bigram Repetition Ratio", axes[2])]

    for key, title, ax in metrics:
        for model_name, temps in temp_metrics.items():
            vals = [temps[t][key] for t in temperatures]
            color = colors.get(model_name, None)
            ax.plot(temperatures, vals, marker="o", label=model_name, color=color, linewidth=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Sampling Temperature", fontsize=10)
        ax.set_xticks(temperatures)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temperature_diversity_comparison.png"), dpi=300)
    plt.close()
