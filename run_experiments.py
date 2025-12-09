"""Runner script for Mixtec classifier experiments.

This script orchestrates the full experiment pipeline for one or more classifiers:
1. Dataset setup (download from Huggingface)
2. Data augmentation (rotation, color jitter, masking)
3. Model training (ViT-B-16 transfer learning)

Example usage:
    # Run all experiments with default settings
    uv run --with loguru --with tomli --with tomli-w python run_experiments.py

    # Run only the Symbol Classifier
    uv run --with loguru --with tomli --with tomli-w python run_experiments.py --datasets symbol

    # Run multiple specific classifiers
    uv run --with loguru --with tomli --with tomli-w python run_experiments.py --datasets bead namedate

    # Skip augmentation (use existing augmented data)
    uv run --with loguru --with tomli --with tomli-w python run_experiments.py --skip-augment

    # Override training parameters
    uv run --with loguru --with tomli --with tomli-w python run_experiments.py --epochs 20 --batch-size 64

    # Dry run to see what would be executed
    uv run --with loguru --with tomli --with tomli-w python run_experiments.py --dry-run

For convenience, you can create a shell alias:
    alias run-experiments='uv run --with loguru --with tomli --with tomli-w python run_experiments.py'
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import tomli
import tomli_w
from loguru import logger

SCRIPT_DIR = Path(__file__).parent.resolve()

DATASETS = {
    "bead": {
        "name": "Bead Counter",
        "dir": SCRIPT_DIR / "Bead Counter",
        "description": "Classifies bead counts (0-15) in Mixtec manuscripts",
    },
    "namedate": {
        "name": "Name-Date Classifier",
        "dir": SCRIPT_DIR / "Name-Date Classifier",
        "description": "Classifies name-date vs year glyphs",
    },
    "symbol": {
        "name": "Symbol Classifier",
        "dir": SCRIPT_DIR / "Symbol Classifier",
        "description": "Classifies 20 Mixtec day sign symbols",
    },
}


def get_train_dir_from_config(classifier_dir: Path) -> Optional[Path]:
    """Extract the training directory path from a classifier's config.toml.

    Args:
        classifier_dir: Path to the classifier directory.

    Returns:
        Resolved path to the training directory, or None if config not found.
    """
    config_path = classifier_dir / "config.toml"
    if not config_path.exists():
        return None

    with open(config_path, "rb") as f:
        config = tomli.load(f)

    train_dir = config.get("paths", {}).get("train_dir", "")
    if train_dir:
        train_path = classifier_dir / train_dir
        return train_path.parent

    return None


def run_command(
    cmd: list[str],
    cwd: Optional[Path] = None,
    dry_run: bool = False,
    description: str = "",
) -> bool:
    """Execute a shell command with logging.

    Args:
        cmd: Command and arguments as a list.
        cwd: Working directory for the command.
        dry_run: If True, only log what would be executed.
        description: Human-readable description of the command.

    Returns:
        True if command succeeded (or dry_run), False otherwise.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    cwd_str = str(cwd) if cwd else "."

    if description:
        logger.info(f"{description}")
    logger.debug(f"Command: {cmd_str}")
    logger.debug(f"Working directory: {cwd_str}")

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {cmd_str}")
        return True

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        return False


def update_config_temporarily(
    classifier_dir: Path,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """Update config.toml with override values and return original config.

    Args:
        classifier_dir: Path to the classifier directory.
        epochs: Override for number of training epochs.
        batch_size: Override for batch size.
        learning_rate: Override for learning rate.
        seed: Override for random seed.

    Returns:
        Original config dict if changes were made, None otherwise.
    """
    config_path = classifier_dir / "config.toml"
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return None

    with open(config_path, "rb") as f:
        original_config = tomli.load(f)

    modified = False
    new_config = {
        "paths": dict(original_config.get("paths", {})),
        "model": dict(original_config.get("model", {})),
        "training": dict(original_config.get("training", {})),
        "workers": dict(original_config.get("workers", {})),
        "classes": dict(original_config.get("classes", {})),
    }

    if epochs is not None:
        new_config["training"]["epochs"] = epochs
        modified = True
    if batch_size is not None:
        new_config["training"]["batch_size"] = batch_size
        modified = True
    if learning_rate is not None:
        new_config["training"]["learning_rate"] = learning_rate
        modified = True
    if seed is not None:
        new_config["model"]["seed"] = seed
        modified = True

    if modified:
        with open(config_path, "wb") as f:
            tomli_w.dump(new_config, f)
        return original_config

    return None


def restore_config(classifier_dir: Path, original_config: dict) -> None:
    """Restore original config.toml contents.

    Args:
        classifier_dir: Path to the classifier directory.
        original_config: Original configuration dictionary to restore.
    """
    config_path = classifier_dir / "config.toml"
    with open(config_path, "wb") as f:
        tomli_w.dump(original_config, f)


def run_setup(classifier_dir: Path, dry_run: bool = False) -> bool:
    """Run the dataset setup script for a classifier.

    Args:
        classifier_dir: Path to the classifier directory.
        dry_run: If True, only log what would be executed.

    Returns:
        True if setup succeeded, False otherwise.
    """
    setup_script = classifier_dir / "setup_dataset.sh"
    if not setup_script.exists():
        logger.warning(f"Setup script not found: {setup_script}")
        return False

    return run_command(
        ["bash", str(setup_script)],
        cwd=classifier_dir,
        dry_run=dry_run,
        description="Running dataset setup script",
    )


def run_augmentation(
    classifier_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Run data augmentation for a classifier.

    Args:
        classifier_dir: Path to the classifier directory.
        dry_run: If True, only log what would be executed.

    Returns:
        True if augmentation succeeded, False otherwise.
    """
    image_dir = get_train_dir_from_config(classifier_dir)
    if image_dir is None:
        logger.warning(f"Could not determine image directory for {classifier_dir}")
        return False

    if not image_dir.exists():
        logger.warning(f"Image directory does not exist: {image_dir}")
        logger.info("You may need to run --setup first to download the dataset")
        return False

    return run_command(
        ["uv", "run", "python", "augment_images.py", str(image_dir)],
        cwd=classifier_dir,
        dry_run=dry_run,
        description=f"Running data augmentation on {image_dir}",
    )


def run_training(
    classifier_dir: Path,
    dry_run: bool = False,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> bool:
    """Run model training for a classifier.

    Args:
        classifier_dir: Path to the classifier directory.
        dry_run: If True, only log what would be executed.
        epochs: Override for number of training epochs.
        batch_size: Override for batch size.
        learning_rate: Override for learning rate.
        seed: Override for random seed.

    Returns:
        True if training succeeded, False otherwise.
    """
    original_config = None
    try:
        original_config = update_config_temporarily(
            classifier_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        if original_config and not dry_run:
            logger.info("Applied training parameter overrides to config.toml")

        success = run_command(
            ["uv", "run", "python", "classification.py"],
            cwd=classifier_dir,
            dry_run=dry_run,
            description="Running model training",
        )
        return success
    finally:
        if original_config and not dry_run:
            restore_config(classifier_dir, original_config)
            logger.debug("Restored original config.toml")


def run_experiment(
    dataset_key: str,
    setup: bool = False,
    augment: bool = True,
    train: bool = True,
    dry_run: bool = False,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> bool:
    """Run the full experiment pipeline for a single dataset.

    Args:
        dataset_key: Key identifying the dataset (bead, namedate, symbol).
        setup: Whether to run dataset setup.
        augment: Whether to run data augmentation.
        train: Whether to run model training.
        dry_run: If True, only log what would be executed.
        epochs: Override for number of training epochs.
        batch_size: Override for batch size.
        learning_rate: Override for learning rate.
        seed: Override for random seed.

    Returns:
        True if all requested steps succeeded, False otherwise.
    """
    dataset_info = DATASETS.get(dataset_key)
    if dataset_info is None:
        logger.error(f"Unknown dataset: {dataset_key}")
        return False

    classifier_dir = dataset_info["dir"]
    logger.info(f"{'=' * 60}")
    logger.info(f"Running experiment: {dataset_info['name']}")
    logger.info(f"Description: {dataset_info['description']}")
    logger.info(f"Directory: {classifier_dir}")
    logger.info(f"{'=' * 60}")

    if not classifier_dir.exists():
        logger.error(f"Classifier directory does not exist: {classifier_dir}")
        return False

    success = True

    if setup:
        logger.info("Step 1/3: Dataset Setup")
        if not run_setup(classifier_dir, dry_run=dry_run):
            logger.error("Dataset setup failed")
            success = False
    else:
        logger.info("Step 1/3: Dataset Setup [SKIPPED]")

    if augment:
        logger.info("Step 2/3: Data Augmentation")
        if not run_augmentation(classifier_dir, dry_run=dry_run):
            logger.warning("Data augmentation failed or skipped")
            if train:
                logger.info("Continuing to training step...")
    else:
        logger.info("Step 2/3: Data Augmentation [SKIPPED]")

    if train:
        logger.info("Step 3/3: Model Training")
        if not run_training(
            classifier_dir,
            dry_run=dry_run,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        ):
            logger.error("Model training failed")
            success = False
    else:
        logger.info("Step 3/3: Model Training [SKIPPED]")

    return success


def list_datasets() -> None:
    """Print information about available datasets."""
    logger.info("Available datasets:")
    logger.info("")
    for key, info in DATASETS.items():
        exists = "exists" if info["dir"].exists() else "NOT FOUND"
        logger.info(f"  {key:10} - {info['name']}")
        logger.info(f"             {info['description']}")
        logger.info(f"             Directory: {info['dir']} [{exists}]")
        logger.info("")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Mixtec classifier experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Run all experiments with defaults
  %(prog)s --datasets symbol            Run only Symbol Classifier
  %(prog)s --datasets bead namedate     Run Bead Counter and Name-Date
  %(prog)s --skip-augment               Skip data augmentation step
  %(prog)s --only-train                 Only run training (skip setup/augment)
  %(prog)s --epochs 20 --batch-size 64  Override training parameters
  %(prog)s --dry-run                    Show what would be executed
  %(prog)s --list                       List available datasets
        """,
    )

    dataset_group = parser.add_argument_group("Dataset Selection")
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        metavar="DATASET",
        help="Datasets to run experiments on. Choices: %(choices)s. Default: all",
    )
    dataset_group.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    pipeline_group = parser.add_argument_group("Pipeline Steps")
    pipeline_group.add_argument(
        "--setup",
        action="store_true",
        help="Run dataset setup (download from Huggingface). Off by default.",
    )
    pipeline_group.add_argument(
        "--skip-augment",
        action="store_true",
        help="Skip data augmentation step",
    )
    pipeline_group.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training step",
    )
    pipeline_group.add_argument(
        "--only-augment",
        action="store_true",
        help="Only run data augmentation (skip setup and training)",
    )
    pipeline_group.add_argument(
        "--only-train",
        action="store_true",
        help="Only run model training (skip setup and augmentation)",
    )

    training_group = parser.add_argument_group("Training Parameters (overrides config.toml)")
    training_group.add_argument(
        "--epochs",
        type=int,
        metavar="N",
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        metavar="N",
        help="Batch size for training and evaluation",
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        metavar="LR",
        help="Learning rate for Adam optimizer",
    )
    training_group.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Random seed for reproducibility",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running anything",
    )
    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    output_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress info messages, only show warnings and errors",
    )

    return parser.parse_args()


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure loguru logging based on verbosity settings.

    Args:
        verbose: Enable debug-level logging.
        quiet: Suppress info-level messages.
    """
    logger.remove()

    if quiet:
        level = "WARNING"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"

    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def main() -> int:
    """Main entry point for the experiment runner.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.list:
        list_datasets()
        return 0

    datasets_to_run = (
        list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    )

    setup = args.setup
    augment = not args.skip_augment
    train = not args.skip_train

    if args.only_augment:
        setup = False
        augment = True
        train = False
    elif args.only_train:
        setup = False
        augment = False
        train = True

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE - No commands will be executed")
        logger.info("=" * 60)

    logger.info(f"Datasets to process: {', '.join(datasets_to_run)}")
    logger.info(f"Pipeline steps: setup={setup}, augment={augment}, train={train}")

    if args.epochs or args.batch_size or args.learning_rate or args.seed:
        logger.info("Training parameter overrides:")
        if args.epochs:
            logger.info(f"  epochs: {args.epochs}")
        if args.batch_size:
            logger.info(f"  batch_size: {args.batch_size}")
        if args.learning_rate:
            logger.info(f"  learning_rate: {args.learning_rate}")
        if args.seed:
            logger.info(f"  seed: {args.seed}")

    results = {}
    for dataset_key in datasets_to_run:
        success = run_experiment(
            dataset_key=dataset_key,
            setup=setup,
            augment=augment,
            train=train,
            dry_run=args.dry_run,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        results[dataset_key] = success

    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    all_success = True
    for dataset_key, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        if not success:
            all_success = False
        logger.info(f"  {DATASETS[dataset_key]['name']}: {status}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
