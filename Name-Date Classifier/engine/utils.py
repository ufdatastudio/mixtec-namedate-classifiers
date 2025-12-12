"""Contains utility functions for PyTorch model saving and loading."""

from pathlib import Path

import torch
from loguru import logger


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """Loads a PyTorch model from a file.

    Args:
        model_path: Path to the saved model file.
        device: Device to load the model onto (e.g. "cuda" or "cpu").

    Returns:
        The loaded PyTorch model in evaluation mode.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str,
) -> None:
    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.

    Example:
        save_model(
            model=model,
            target_dir="models",
            model_name="classifier_vit.pth"
        )
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name
    logger.info(f"Saving model to: {model_save_path}")
    torch.save(obj=model, f=model_save_path)
