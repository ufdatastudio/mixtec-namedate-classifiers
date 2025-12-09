# MIXTEC-CLASSIFIERS

Code repository for classifying Name-Date and Year figures in Mixtec Codices.

## Classifiers

This repository contains three classifiers:

| Classifier | Description | Classes |
|------------|-------------|---------|
| Bead Counter | Classifies bead counts in Mixtec manuscripts | 0-15 |
| Name-Date Classifier | Distinguishes name-date vs year glyphs | 2 |
| Symbol Classifier | Classifies 20 Mixtec day sign symbols | 20 |

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{salunke2025classifying,
  title = {Classifying Name-Date and Year Figures in Mixtec Codices},
  author = {Girish Salunke and Christopher Driggers-Ellis and Christan Grant},
  year = {2025},
  journal = {Anthology of Computers and the Humanities},
  volume = {3},
  pages = {1211--1219},
  editor = {Taylor Arnold, Margherita Fantoli, and Ruben Ros},
  doi = {10.63744/eNge3Gj1LSHb}
}
```

## Prerequisites

Before you begin, make sure that you have the following installed:

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Git
- GPU drivers (if using GPU for training)
- Huggingface credentials (for dataset access)
- Git LFS: You might need this to ensure that the images downloaded from Huggingface are not just references but actual files.

## Quick Start (Experiment Runner)

The `run_experiments.py` script automates the full pipeline for all classifiers.

### Run All Experiments

```bash
# Download datasets, augment images, and train all classifiers
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --setup
```

### Run Without Setup (if datasets already downloaded)

```bash
uv run --with loguru --with tomli --with tomli-w python run_experiments.py
```

### Run Specific Classifiers

```bash
# Only Symbol Classifier
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --datasets symbol

# Bead Counter and Name-Date only
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --datasets bead namedate
```

### Skip Steps

```bash
# Skip augmentation (use existing augmented data)
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --skip-augment

# Only run training
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --only-train

# Only run augmentation
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --only-augment
```

### Override Training Parameters

```bash
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --epochs 20 --batch-size 64 --learning-rate 0.0001
```

### Dry Run

Preview commands without executing:

```bash
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --dry-run
```

### List Available Datasets

```bash
uv run --with loguru --with tomli --with tomli-w python run_experiments.py --list
```

## Manual Steps (Individual Classifiers)

### 1. Setting up the Image Dataset

To clone the dataset from Huggingface, use the following command. You will be prompted to provide your Huggingface credentials:

```bash
./setup_dataset.sh
```

This will download the dataset to your local system.

### 2. Setting up the Python Environment

Each project has its own `pyproject.toml` file with dependencies. Navigate to the project directory and sync the environment using uv:

```bash
cd "Name-Date Classifier"
uv sync
```

This will create a virtual environment and install all dependencies.

**Alternative (conda):** You can also use the provided `environment.yml` file. Note that you may need to adjust the file for the GPU version of PyTorch installed on your system.

```bash
conda env create -f environment.yml
conda activate name-date
```

### 3. Perform Data Augmentation

Once the environment is set up, you can perform data augmentation on the images in the dataset. To do so, run the `augment_images.py` script and specify the folder path to your images dataset as a command-line argument:

```bash
uv run python augment_images.py <path_to_image_dataset>
```

This will perform data augmentation on the images and save the results in the same directory.

### 4. Train the Model

To train the classification model on the augmented dataset, run the `classification.py` script:

```bash
uv run python classification.py
```

This script will:

- Train the models on the image dataset.  
- Save the trained models in the `models` folder.  
- Export accuracy and loss curves.  
- Write precision, recall, and F1 score to a text file.  

## Configuration

Each classifier has a `config.toml` file with training parameters:

```toml
[training]
epochs = 10
batch_size = 32
learning_rate = 0.001

[model]
seed = 42
```

Command-line arguments to `run_experiments.py` override these settings temporarily during execution.

## Output

Training produces:
- Model checkpoints in classifier directory
- Training/validation metrics
- Confusion matrices and classification reports

## Hardware Acceleration

The training script automatically detects:
- Apple MPS (Metal Performance Shaders) on macOS
- CUDA on NVIDIA GPUs
- Falls back to CPU if neither available

## Acknowledgements

Special thanks to the following authors for helping with the code snippets to build vision Transformer classifiers:  

- Daniel Bourke, *[pytorch-deep-learning](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular)*  
- Aarohi Singla, *[Image-Classification-Using-Vision-transformer](https://github.com/AarohiSingla/Image-Classification-Using-Vision-transformer/tree/main)*  

We gratefully acknowledge their contributions to the open-source community.
