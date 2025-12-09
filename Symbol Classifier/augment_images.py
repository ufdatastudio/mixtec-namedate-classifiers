"""Image augmentation module for data augmentation with rotation, color jitter, and masking."""

import argparse
import os
import random

from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance

ROTATION_ANGLES = range(15, 360, 15)


def apply_color_jitter(img: Image.Image) -> Image.Image:
    """Apply random color jitter to an image.

    Args:
        img: PIL Image to apply color jitter to.

    Returns:
        Color-jittered PIL Image.
    """
    brightness = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    contrast = ImageEnhance.Contrast(brightness).enhance(random.uniform(0.8, 1.2))
    color = ImageEnhance.Color(contrast).enhance(random.uniform(0, 2))
    return color


def apply_random_mask(img: Image.Image) -> Image.Image:
    """Apply a random white rectangular mask to an image.

    Args:
        img: PIL Image to apply mask to.

    Returns:
        PIL Image with random mask applied.
    """
    img_with_mask = img.copy()
    draw = ImageDraw.Draw(img_with_mask)

    width, height = img.size
    mask_width = random.randint(width // 8, width // 4)
    mask_height = random.randint(height // 8, height // 4)

    top_left_x = random.randint(0, width - mask_width)
    top_left_y = random.randint(0, height - mask_height)

    draw.rectangle(
        [(top_left_x, top_left_y), (top_left_x + mask_width, top_left_y + mask_height)],
        fill="white"
    )

    return img_with_mask


def apply_rotations(image_directory: str, num_rotations: int = 16) -> None:
    """Apply random rotations to all PNG images in directory.

    Args:
        image_directory: Directory containing images to augment.
        num_rotations: Number of random rotation angles to apply per image.
    """
    logger.info("Data Augmentation: Applying random rotation to images")
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    for angle in random.sample(list(ROTATION_ANGLES), k=num_rotations):
                        rotated_img = img.rotate(angle, expand=True)

                        rotated_filename = f'rot_{angle}_{filename}'
                        rotated_img_path = os.path.join(root, rotated_filename)
                        rotated_img.save(rotated_img_path)


def apply_jitter_to_images(image_directory: str) -> None:
    """Apply color jitter to all PNG images in directory.

    Args:
        image_directory: Directory containing images to augment.
    """
    logger.info("Data Augmentation: Applying random color jitter to images")
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    jittered_img = apply_color_jitter(img)

                    jittered_filename = f'jitter_{filename}'
                    jittered_img_path = os.path.join(root, jittered_filename)
                    jittered_img.save(jittered_img_path)


def apply_masks_to_images(image_directory: str) -> None:
    """Apply random masks to all PNG images in directory.

    Args:
        image_directory: Directory containing images to augment.
    """
    logger.info("Data Augmentation: Applying random masking to images")
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith('.png'):
                image_path = os.path.join(root, filename)

                with Image.open(image_path) as img:
                    masked_img = apply_random_mask(img)

                    masked_filename = f'mask_{filename}'
                    masked_img_path = os.path.join(root, masked_filename)
                    masked_img.save(masked_img_path)


def augment_images(image_directory: str) -> None:
    """Run the full augmentation pipeline on images in a directory.

    Applies rotation, color jitter, and masking augmentations sequentially.

    Args:
        image_directory: Directory containing subdirectories of images to augment.
    """
    apply_rotations(image_directory)
    apply_jitter_to_images(image_directory)
    apply_masks_to_images(image_directory)
    logger.success("Data Augmentation: Rotation, color jitter, and masking augmentation completed successfully")


def main() -> None:
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Augment images by performing random rotations and color jitter.'
    )
    parser.add_argument(
        'image_directory',
        type=str,
        help='Directory containing the subdirectories of images to augment'
    )
    args = parser.parse_args()

    augment_images(args.image_directory)


if __name__ == "__main__":
    main()