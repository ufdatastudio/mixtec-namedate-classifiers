"""Helper functions for training visualization and reproducibility.

MIT License
Copyright (c) 2021 Daniel Bourke
"""

import matplotlib.pyplot as plt
import torch


def plot_loss_curves(results: dict, filename_prefix: str = "loss_curves") -> None:
    """Plots training curves of a results dictionary and saves them to files.

    Args:
        results: Dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        filename_prefix: Prefix for the output files.
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylim(0, max(max(loss), max(test_loss)) * 1.1)
    plt.savefig(f"{filename_prefix}_loss.png")

    plt.clf()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"{filename_prefix}_accuracy.png")

    plt.close()


def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for torch operations for reproducibility.

    Args:
        seed: Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
