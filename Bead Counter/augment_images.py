"""Image augmentation module for data augmentation with rotation, color jitter, and masking.

This module reads augmentation parameters from config.toml and applies various
augmentations to training images to increase dataset diversity.
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

import tomli
from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance

# Default configuration values (used if config.toml is not found)
DEFAULT_CONFIG = {
    "num_rotations": 16,
    "rotation_angles": list(range(15, 360, 15)),
    "brightness_range": [0.8, 1.2],
    "contrast_range": [0.8, 1.2],
    "color_range": [0.0, 2.0],
    "mask_width_fraction": [0.125, 0.25],
    "mask_height_fraction": [0.125, 0.25],
}


def load_augmentation_config(config_path: Path) -> Dict:
    """Load augmentation configuration from config.toml.

    Args:
        config_path: Path to the config.toml file.

    Returns:
        Dictionary containing augmentation parameters.
    """
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomli.load(f)
            aug_config = config.get("augmentation", {})
            # Merge with defaults for any missing keys
            return {**DEFAULT_CONFIG, **aug_config}
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return DEFAULT_CONFIG


def apply_color_jitter(
    img: Image.Image,
    brightness_range: List[float],
    contrast_range: List[float],
    color_range: List[float],
) -> Image.Image:
    """Apply random color jitter to an image.

    Args:
        img: PIL Image to apply color jitter to.
        brightness_range: [min, max] range for brightness adjustment.
        contrast_range: [min, max] range for contrast adjustment.
        color_range: [min, max] range for color saturation adjustment.

    Returns:
        Color-jittered PIL Image.
    """
    brightness = ImageEnhance.Brightness(img).enhance(
        random.uniform(brightness_range[0], brightness_range[1])
    )
    contrast = ImageEnhance.Contrast(brightness).enhance(
        random.uniform(contrast_range[0], contrast_range[1])
    )
    color = ImageEnhance.Color(contrast).enhance(
        random.uniform(color_range[0], color_range[1])
    )
    return color


def apply_random_mask(
    img: Image.Image,
    mask_width_fraction: List[float],
    mask_height_fraction: List[float],
) -> Image.Image:
    """Apply a random white rectangular mask to an image.

    Args:
        img: PIL Image to apply mask to.
        mask_width_fraction: [min, max] fraction of image width for mask.
        mask_height_fraction: [min, max] fraction of image height for mask.

    Returns:
        PIL Image with random mask applied.
    """
    img_with_mask = img.copy()
    draw = ImageDraw.Draw(img_with_mask)

    width, height = img.size
    mask_width = random.randint(
        int(width * mask_width_fraction[0]), int(width * mask_width_fraction[1])
    )
    mask_height = random.randint(
        int(height * mask_height_fraction[0]), int(height * mask_height_fraction[1])
    )

    top_left_x = random.randint(0, max(0, width - mask_width))
    top_left_y = random.randint(0, max(0, height - mask_height))

    draw.rectangle(
        [(top_left_x, top_left_y), (top_left_x + mask_width, top_left_y + mask_height)],
        fill="white",
    )

    return img_with_mask


def apply_rotations(
    image_directory: str,
    rotation_angles: List[int],
    num_rotations: int,
) -> None:
    """Apply random rotations to all PNG images in directory.

    Args:
        image_directory: Directory containing images to augment.
        rotation_angles: List of angles to sample from.
        num_rotations: Number of random rotation angles to apply per image.
    """
    logger.info("Data Augmentation: Applying random rotation to images")
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    # Sample up to num_rotations angles (or all if fewer available)
                    k = min(num_rotations, len(rotation_angles))
                    for angle in random.sample(rotation_angles, k=k):
                        rotated_img = img.rotate(angle, expand=True)

                        rotated_filename = f"rot_{angle}_{filename}"
                        rotated_img_path = os.path.join(root, rotated_filename)
                        rotated_img.save(rotated_img_path)


def apply_jitter_to_images(
    image_directory: str,
    brightness_range: List[float],
    contrast_range: List[float],
    color_range: List[float],
) -> None:
    """Apply color jitter to all PNG images in directory.

    Args:
        image_directory: Directory containing images to augment.
        brightness_range: [min, max] range for brightness adjustment.
        contrast_range: [min, max] range for contrast adjustment.
        color_range: [min, max] range for color saturation adjustment.
    """
    logger.info("Data Augmentation: Applying random color jitter to images")
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    jittered_img = apply_color_jitter(
                        img, brightness_range, contrast_range, color_range
                    )

                    jittered_filename = f"jitter_{filename}"
                    jittered_img_path = os.path.join(root, jittered_filename)
                    jittered_img.save(jittered_img_path)


def apply_masks_to_images(
    image_directory: str,
    mask_width_fraction: List[float],
    mask_height_fraction: List[float],
) -> None:
    """Apply random masks to all PNG images in directory.

    Args:
        image_directory: Directory containing images to augment.
        mask_width_fraction: [min, max] fraction of image width for mask.
        mask_height_fraction: [min, max] fraction of image height for mask.
    """
    logger.info("Data Augmentation: Applying random masking to images")
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    masked_img = apply_random_mask(
                        img, mask_width_fraction, mask_height_fraction
                    )

                    masked_filename = f"mask_{filename}"
                    masked_img_path = os.path.join(root, masked_filename)
                    masked_img.save(masked_img_path)


def augment_images(image_directory: str, config: Dict) -> None:
    """Run the full augmentation pipeline on images in a directory.

    Applies rotation, color jitter, and masking augmentations sequentially.

    Args:
        image_directory: Directory containing subdirectories of images to augment.
        config: Dictionary containing augmentation configuration.
    """
    apply_rotations(
        image_directory,
        rotation_angles=config["rotation_angles"],
        num_rotations=config["num_rotations"],
    )
    apply_jitter_to_images(
        image_directory,
        brightness_range=config["brightness_range"],
        contrast_range=config["contrast_range"],
        color_range=config["color_range"],
    )
    apply_masks_to_images(
        image_directory,
        mask_width_fraction=config["mask_width_fraction"],
        mask_height_fraction=config["mask_height_fraction"],
    )
    logger.success(
        "Data Augmentation: Rotation, color jitter, and masking augmentation completed"
    )


def main() -> None:
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Augment images by performing random rotations, color jitter, and masking."
    )
    parser.add_argument(
        "image_directory",
        type=str,
        help="Directory containing the subdirectories of images to augment",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to config.toml file (default: config.toml in current directory)",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    config = load_augmentation_config(config_path)

    logger.info(f"Augmentation config: {config}")
    augment_images(args.image_directory, config)


if __name__ == "__main__":
    main()
