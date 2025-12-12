# Name-Date Classifier - Code Walkthrough

This document provides an annotated walkthrough of the Name-Date Classifier, which distinguishes between name-date glyphs and year glyphs in Mixtec manuscripts using transfer learning with Vision Transformer (ViT-B-16).

## Overview

The Name-Date Classifier is a binary image classification system that:
- Uses a pretrained ViT-B-16 model from torchvision
- Freezes the base model and trains only the classification head
- Classifies images into 2 classes: `name_date` and `year`
- Supports CUDA, Apple MPS, and CPU training

## Quick Start

```bash
# 1. Set up the dataset (download from Huggingface)
bash setup_dataset.sh

# 2. Run data augmentation
uv run python augment_images.py ./name_date_images

# 3. Train the model
uv run python classification.py
```

## Project Structure

```
Name-Date Classifier/
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
└── name_date_images/      # Dataset directory
    ├── train/
    │   ├── name_date/     # Name-date glyph images
    │   └── year/          # Year glyph images
    └── test/
        ├── name_date/
        └── year/
```

## Configuration (config.toml)

The `config.toml` file contains all configurable parameters:

```toml
[paths]
train_dir = "./name_date_images/train"  # Training images location
test_dir = "./name_date_images/test"    # Test images location
model_dir = "models"                     # Where to save trained models

[model]
in_features = 768                        # ViT-B-16 output features (fixed)
seed = 42                                # Random seed for reproducibility

[training]
epochs = 10                              # Number of training epochs
batch_size = 32                          # Batch size for DataLoaders
learning_rate = 1e-3                     # Adam optimizer learning rate

[workers]
cuda = 16                                # DataLoader workers for CUDA
mps = 8                                  # DataLoader workers for Apple MPS
cpu = 6                                  # DataLoader workers for CPU

[classes]
names = ["name_date", "year"]            # Binary classification classes

[augmentation]
num_rotations = 16                       # Rotated versions per image
rotation_angles = [15, 30, ...]          # Angles to sample from
brightness_range = [0.8, 1.2]            # Brightness adjustment range
contrast_range = [0.8, 1.2]              # Contrast adjustment range
color_range = [0.0, 2.0]                 # Color saturation range
mask_width_fraction = [0.125, 0.25]      # Mask width as image fraction
mask_height_fraction = [0.125, 0.25]     # Mask height as image fraction
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
            return {**DEFAULT_CONFIG, **aug_config}
    else:
        return DEFAULT_CONFIG
```

#### Rotation Augmentation

```python
def apply_rotations(image_directory, rotation_angles, num_rotations):
    """Apply random rotations to all PNG images."""
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith(".png"):
                with Image.open(image_path) as img:
                    k = min(num_rotations, len(rotation_angles))
                    for angle in random.sample(rotation_angles, k=k):
                        rotated_img = img.rotate(angle, expand=True)
                        rotated_img.save(f"rot_{angle}_{filename}")
```

Creates rotated versions to help the model generalize across orientations.

#### Color Jitter

```python
def apply_color_jitter(img, brightness_range, contrast_range, color_range):
    """Apply random color adjustments."""
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
```

Varies color properties to handle different scan qualities and lighting conditions.

#### Random Masking

```python
def apply_random_mask(img, mask_width_fraction, mask_height_fraction):
    """Apply random white rectangular mask."""
    # Simulates damage or occlusion in manuscript images
    draw = ImageDraw.Draw(img_with_mask)
    draw.rectangle([...], fill="white")
    return img_with_mask
```

### 2. Training Pipeline (classification.py)

#### Configuration Loading

```python
CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    config = tomli.load(f)

train_dir = config["paths"]["train_dir"]
class_names = config["classes"]["names"]  # ["name_date", "year"]
```

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

#### Model Setup (Transfer Learning)

```python
# Get pretrained ViT-B-16
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Freeze base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# Replace head for binary classification
pretrained_vit.heads = nn.Linear(
    in_features=768,      # ViT-B-16 feature dimension
    out_features=2        # name_date vs year
).to(device)
```

#### DataLoader Creation

```python
def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader
```

#### Training Execution

```python
# Use transforms from pretrained model
pretrained_vit_transforms = pretrained_vit_weights.transforms()

# Setup optimizer and loss
optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=config["training"]["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()

# Train
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

#### Single Epoch Training

```python
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)
```

#### Evaluation with Confusion Matrix

```python
def test_step(model, dataloader, loss_fn, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            test_pred_labels = test_pred_logits.argmax(dim=1)

            all_preds.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return test_loss, test_acc, all_preds, all_labels
```

#### Full Training Loop

```python
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs, device, class_names):
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(...)
        test_loss, test_acc, all_preds, all_labels = test_step(...)

        # Generate 2x2 confusion matrix (binary classification)
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names)
        plt.savefig(f"confusion_matrix/epoch_{epoch+1}.png")

        # Log binary classification metrics
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

    return results
```

### 4. Model Utilities (engine/utils.py)

```python
def save_model(model, target_dir, model_name):
    """Save complete model for inference."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model, f=target_dir_path / model_name)
```

### 5. Helper Functions (helper_functions.py)

```python
def plot_loss_curves(results, filename_prefix="loss_curves"):
    """Generate training curves."""
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.savefig(f"{filename_prefix}_loss.png")

def set_seeds(seed=42):
    """Ensure reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

## Outputs

After training:

- `models/namedate_year_classifier_vit_YYYYMMDD_HHMMSS.pth` - Trained model
- `loss_curves_loss.png` - Loss over epochs
- `loss_curves_accuracy.png` - Accuracy over epochs
- `confusion_matrix/epoch_N.png` - 2x2 confusion matrix per epoch
- `confusion_matrix/metrics_log.txt` - Precision, recall, F1 per epoch

## Adapting for Your Project

- [ ] **Update class names** in `config.toml` under `[classes].names`
- [ ] **Organize dataset** with matching subdirectory names:
  ```
  your_images/
  ├── train/
  │   ├── class_a/
  │   └── class_b/
  └── test/
      ├── class_a/
      └── class_b/
  ```
- [ ] **Update paths** in `[paths]` section
- [ ] **Adjust hyperparameters** in `[training]`:
  - For binary classification, may need fewer epochs
  - Adjust `learning_rate` if overfitting
- [ ] **Update model name** in `classification.py` (line ~158)

## Dependencies

Managed via `pyproject.toml`:

```
loguru, matplotlib, numpy, pillow, scikit-learn, seaborn,
tomli, torch, torchinfo, torchvision, tqdm
```
