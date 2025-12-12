# Bead Counter - Code Walkthrough

This document provides an annotated walkthrough of the Bead Counter classifier, which classifies bead counts (0-15, excluding 14) in Mixtec manuscripts using transfer learning with Vision Transformer (ViT-B-16).

## Overview

The Bead Counter is an image classification system that:
- Uses a pretrained ViT-B-16 model from torchvision
- Freezes the base model and trains only the classification head
- Classifies images into 15 classes representing bead counts
- Supports CUDA, Apple MPS, and CPU training

## Quick Start

```bash
# 1. Set up the dataset (download from Huggingface)
bash setup_dataset.sh

# 2. Run data augmentation
uv run python augment_images.py ./sign_images

# 3. Train the model
uv run python classification.py
```

## Project Structure

```
Bead Counter/
├── classification.py      # Main training script
├── augment_images.py      # Data augmentation module
├── helper_functions.py    # Plotting and seed utilities
├── config.toml            # All configuration parameters
├── pyproject.toml         # Python dependencies (uv)
├── setup_dataset.sh       # Dataset download script
├── walkthrough.md         # This documentation
├── engine/
│   ├── __init__.py        # Module exports
│   ├── engine.py          # Training loop functions
│   └── utils.py           # Model save/load utilities
├── models/                # Saved model checkpoints
├── confusion_matrix/      # Training metrics & confusion matrices
└── sign_images/           # Dataset directory
    ├── train/             # Training images by class
    └── test/              # Test images by class
```

## Configuration (config.toml)

The `config.toml` file contains all configurable parameters:

```toml
[paths]
train_dir = "./sign_images/train"    # Training images location
test_dir = "./sign_images/test"      # Test images location
model_dir = "models"                  # Where to save trained models

[model]
in_features = 768                     # ViT-B-16 output features (fixed)
seed = 42                             # Random seed for reproducibility

[training]
epochs = 10                           # Number of training epochs
batch_size = 32                       # Batch size for DataLoaders
learning_rate = 1e-3                  # Adam optimizer learning rate

[workers]
cuda = 16                             # DataLoader workers for CUDA
mps = 8                               # DataLoader workers for Apple MPS
cpu = 6                               # DataLoader workers for CPU

[classes]
names = ["0", "1", "2", ...]          # Class labels (bead counts)

[augmentation]
num_rotations = 16                    # Rotated versions per image
rotation_angles = [15, 30, ...]       # Angles to sample from
brightness_range = [0.8, 1.2]         # Brightness adjustment range
contrast_range = [0.8, 1.2]           # Contrast adjustment range
color_range = [0.0, 2.0]              # Color saturation range
mask_width_fraction = [0.125, 0.25]   # Mask width as image fraction
mask_height_fraction = [0.125, 0.25]  # Mask height as image fraction
```

## Code Walkthrough

### 1. Data Augmentation (augment_images.py)

The augmentation module applies three transformations to increase dataset diversity:

#### Loading Configuration

```python
def load_augmentation_config(config_path: Path) -> Dict:
    """Load augmentation configuration from config.toml."""
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomli.load(f)
            aug_config = config.get("augmentation", {})
            # Merge with defaults for any missing keys
            return {**DEFAULT_CONFIG, **aug_config}
    else:
        logger.warning(f"Config file not found, using defaults")
        return DEFAULT_CONFIG
```

This function loads augmentation parameters from `config.toml` and falls back to sensible defaults if the file is missing.

#### Rotation Augmentation

```python
def apply_rotations(image_directory, rotation_angles, num_rotations):
    """Apply random rotations to all PNG images in directory."""
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith(".png"):
                with Image.open(image_path) as img:
                    # Sample num_rotations angles from the available angles
                    k = min(num_rotations, len(rotation_angles))
                    for angle in random.sample(rotation_angles, k=k):
                        rotated_img = img.rotate(angle, expand=True)
                        rotated_img.save(f"rot_{angle}_{filename}")
```

Creates multiple rotated versions of each image. The `expand=True` parameter ensures the entire rotated image is preserved.

#### Color Jitter

```python
def apply_color_jitter(img, brightness_range, contrast_range, color_range):
    """Apply random color jitter to an image."""
    # Adjust brightness randomly within range
    brightness = ImageEnhance.Brightness(img).enhance(
        random.uniform(brightness_range[0], brightness_range[1])
    )
    # Adjust contrast
    contrast = ImageEnhance.Contrast(brightness).enhance(
        random.uniform(contrast_range[0], contrast_range[1])
    )
    # Adjust color saturation
    color = ImageEnhance.Color(contrast).enhance(
        random.uniform(color_range[0], color_range[1])
    )
    return color
```

Applies sequential brightness, contrast, and color saturation adjustments using PIL's ImageEnhance module.

#### Random Masking

```python
def apply_random_mask(img, mask_width_fraction, mask_height_fraction):
    """Apply a random white rectangular mask to an image."""
    img_with_mask = img.copy()
    draw = ImageDraw.Draw(img_with_mask)

    width, height = img.size
    # Calculate mask dimensions as fraction of image size
    mask_width = random.randint(
        int(width * mask_width_fraction[0]),
        int(width * mask_width_fraction[1])
    )
    # ... draw white rectangle at random position
    return img_with_mask
```

Simulates occlusion by adding random white rectangles, helping the model learn to classify partially obscured images.

### 2. Training Pipeline (classification.py)

The main training script orchestrates the entire training process:

#### Configuration Loading

```python
# Load configuration from config.toml
CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    config = tomli.load(f)

# Extract values
train_dir = config["paths"]["train_dir"]
test_dir = config["paths"]["test_dir"]
class_names = config["classes"]["names"]
```

Configuration is loaded at module level, making all parameters easily accessible.

#### Device Detection

```python
if torch.cuda.is_available():
    device = "cuda"
    NUM_WORKERS = config["workers"]["cuda"]
elif torch.backends.mps.is_available():
    device = "mps"
    NUM_WORKERS = config["workers"]["mps"]
else:
    device = "cpu"
    NUM_WORKERS = config["workers"]["cpu"]
```

Automatically selects the best available compute device and adjusts worker count accordingly.

#### Model Setup (Transfer Learning)

```python
# 1. Get pretrained weights
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# 2. Create model with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze all base parameters (no gradient updates)
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# 4. Replace classification head with new trainable layer
pretrained_vit.heads = nn.Linear(
    in_features=config["model"]["in_features"],  # 768 for ViT-B-16
    out_features=len(class_names)                 # Number of classes
).to(device)
```

This is the core transfer learning approach:
- The pretrained ViT-B-16 acts as a feature extractor
- Only the final linear layer is trained
- This dramatically reduces training time and data requirements

#### DataLoader Creation

```python
def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    """Create PyTorch DataLoaders for training and testing."""
    # ImageFolder expects subdirectories named by class
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,           # Randomize training order
        num_workers=num_workers,
        pin_memory=True,        # Faster GPU transfer
    )
    # ... similar for test_dataloader with shuffle=False
    return train_dataloader, test_dataloader
```

Uses torchvision's ImageFolder which automatically infers class labels from subdirectory names.

#### Training Execution

```python
# Get transforms from pretrained weights (includes normalization)
pretrained_vit_transforms = pretrained_vit_weights.transforms()

# Create DataLoaders
train_dataloader, test_dataloader = create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=pretrained_vit_transforms,
    batch_size=config["training"]["batch_size"],
    num_workers=NUM_WORKERS
)

# Setup optimizer and loss
optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=config["training"]["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
results = engine.train(
    model=pretrained_vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=config["training"]["epochs"],
    device=device,
    class_names=class_names
)
```

### 3. Training Engine (engine/engine.py)

The engine module contains the core training loop logic:

#### Single Epoch Training

```python
def train_step(model, dataloader, loss_fn, optimizer, device):
    """Train model for one epoch."""
    model.train()  # Enable training mode (dropout, batch norm)
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Return average metrics
    return train_loss / len(dataloader), train_acc / len(dataloader)
```

#### Single Epoch Evaluation

```python
def test_step(model, dataloader, loss_fn, device):
    """Evaluate model for one epoch, collecting predictions for confusion matrix."""
    model.eval()  # Disable dropout, use running stats for batch norm
    all_preds, all_labels = [], []

    with torch.inference_mode():  # Disable gradient computation
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)

            # Collect for confusion matrix
            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return test_loss, test_acc, all_preds, all_labels
```

#### Full Training Loop

```python
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs, device, class_names):
    """Complete training loop with metrics logging."""
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        # Train and evaluate
        train_loss, train_acc = train_step(model, train_dataloader, ...)
        test_loss, test_acc, all_preds, all_labels = test_step(model, test_dataloader, ...)

        # Log metrics
        logger.info(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | ...")

        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
        plt.savefig(f"confusion_matrix/epoch_{epoch+1}.png")

        # Log precision, recall, F1
        precision = precision_score(all_labels, all_preds, average='macro')
        # ... write to metrics_log.txt

    return results
```

### 4. Model Utilities (engine/utils.py)

```python
def save_model(model, target_dir, model_name):
    """Save complete model to disk."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name
    torch.save(obj=model, f=model_save_path)
```

Saves the entire model (not just state_dict) for easy loading and inference.

### 5. Helper Functions (helper_functions.py)

```python
def plot_loss_curves(results, filename_prefix="loss_curves"):
    """Generate and save training/validation curves."""
    # Plot loss curve
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.savefig(f"{filename_prefix}_loss.png")

    # Plot accuracy curve
    plt.plot(epochs, results["train_acc"], label="train_accuracy")
    plt.plot(epochs, results["test_acc"], label="test_accuracy")
    plt.savefig(f"{filename_prefix}_accuracy.png")

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

## Outputs

After training, you'll find:

- `models/bead_counter_vit_YYYYMMDD_HHMMSS.pth` - Trained model checkpoint
- `loss_curves_loss.png` - Training/validation loss over epochs
- `loss_curves_accuracy.png` - Training/validation accuracy over epochs
- `confusion_matrix/epoch_N.png` - Confusion matrix for each epoch
- `confusion_matrix/metrics_log.txt` - Precision, recall, F1 per epoch

## Adapting for Your Project

To adapt this classifier for a different task:

- [ ] **Update class names** in `config.toml` under `[classes].names`
- [ ] **Organize your dataset** with subdirectories matching class names:
  ```
  your_images/
  ├── train/
  │   ├── class_a/
  │   ├── class_b/
  │   └── class_c/
  └── test/
      ├── class_a/
      ├── class_b/
      └── class_c/
  ```
- [ ] **Update paths** in `config.toml` under `[paths]`
- [ ] **Adjust hyperparameters** in `[training]` section:
  - Increase `epochs` for larger datasets
  - Adjust `batch_size` based on GPU memory
  - Lower `learning_rate` for fine-tuning
- [ ] **Modify augmentation** in `[augmentation]` section based on your domain
- [ ] **Update model name** in `classification.py` (line ~158)

## Dependencies

All dependencies are managed via `pyproject.toml` and `uv`:

```
loguru          - Logging
matplotlib      - Plotting
numpy           - Numerical operations
pillow          - Image processing
scikit-learn    - Metrics (confusion matrix, precision, recall, F1)
seaborn         - Heatmap visualization
tomli           - TOML config parsing
torch           - PyTorch framework
torchinfo       - Model summaries
torchvision     - Pretrained models and transforms
tqdm            - Progress bars
```
