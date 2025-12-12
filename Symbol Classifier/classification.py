
# Import required packages
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tomli
import torch
import torchvision
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import datasets, transforms

from engine import engine, utils
from helper_functions import plot_loss_curves, set_seeds

# Load configuration from config.toml
CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    config = tomli.load(f)

# Extract configuration values
train_dir = config["paths"]["train_dir"]
test_dir = config["paths"]["test_dir"]
model_dir = config["paths"]["model_dir"]
class_names = config["classes"]["names"]


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int
    ):

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader


if __name__ == '__main__':

    # Determine device and number of workers from config
    if torch.cuda.is_available():
        device = "cuda"
        NUM_WORKERS = config["workers"]["cuda"]
        logger.info(f"CUDA GPU available for training with {NUM_WORKERS} workers")
    elif torch.backends.mps.is_available():
        device = "mps"
        NUM_WORKERS = config["workers"]["mps"]
        logger.info(f"Apple MPS available for training with {NUM_WORKERS} workers")
    else:
        device = "cpu"
        logger.warning("You are working in CPU mode. If you intended to use GPU then abort immediately by pressing ctrl + c")
        NUM_WORKERS = config["workers"]["cpu"]
        logger.info(f"CPU based training with {NUM_WORKERS} workers")

    # 1. Get pretrained weights for ViT-Base
    logger.info("Step 1: Getting pretrained weights for ViT-Base")
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

    # 2. Setup a ViT model instance with pretrained weights
    logger.info("Step 2: Setting up a ViT model instance with pretrained weights")
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

    # 3. Freeze the base parameters
    logger.info("Step 3: Freezing the base parameters")
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # 4. Setting Random Seeds
    logger.info("Step 4: Setting Random Seeds")
    set_seeds(config["model"]["seed"])

    # 5. Setting the in_features and out_features for vit
    logger.info("Step 5: Setting the in_features and out_features for vit")
    pretrained_vit.heads = nn.Linear(
        in_features=config["model"]["in_features"],
        out_features=len(class_names)
    ).to(device)

    # 6. Print a summary using torchinfo
    logger.info("Step 6: Printing a summary using torchinfo")
    summary(model=pretrained_vit,
            input_size=(config["training"]["batch_size"], 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    # 7. Setup directory paths
    logger.info(f"Step 7: Setting up train and test directories, train:{train_dir}, test:{test_dir}")

    # 8. Get automatic transforms from pretrained ViT weights
    logger.info("Step 8: Getting automatic transforms from pretrained ViT weights")
    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    logger.debug(f"Pretrained VIT transforms: {pretrained_vit_transforms}")

    # 9. Setup dataloaders
    logger.info("Step 9: Setting up data loaders")
    train_dataloader_pretrained, test_dataloader_pretrained = create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=pretrained_vit_transforms,
        batch_size=config["training"]["batch_size"],
        num_workers=NUM_WORKERS
    )

    # 10. Create optimizer and loss function
    logger.info("Step 10: Setting up Adam Optimiser and CrossEntropyLoss functions")
    optimizer = torch.optim.Adam(
        params=pretrained_vit.parameters(),
        lr=config["training"]["learning_rate"]
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # 11. Train the classifier head of the pretrained ViT feature extractor model
    logger.info("Step 11: Beginning Model Training, This may take a while....")
    set_seeds(config["model"]["seed"])
    pretrained_vit_results = engine.train(
        model=pretrained_vit,
        train_dataloader=train_dataloader_pretrained,
        test_dataloader=test_dataloader_pretrained,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config["training"]["epochs"],
        device=device,
        class_names=class_names
    )

    # 12. Plotting Loss Curves
    logger.info("Step 12: Plotting Loss Curves")
    plot_loss_curves(pretrained_vit_results)

    # 13. Save the trained Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"symbol_classifier_vit_{timestamp}.pth"

    utils.save_model(
        model=pretrained_vit,
        target_dir=model_dir,
        model_name=model_name
    )

    logger.success(f"Model saved to {model_dir}/{model_name}")
