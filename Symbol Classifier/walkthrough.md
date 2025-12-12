# Symbol Classifier - Code Walkthrough

This document provides an annotated walkthrough of the Symbol Classifier, which identifies 20 Mixtec day sign symbols from manuscript images using transfer learning with Vision Transformer (ViT-B-16).

## Overview

The Symbol Classifier is a multi-class image classification system that:
- Uses a pretrained ViT-B-16 model from torchvision
- Freezes the base model and trains only the classification head
- Classifies images into 20 classes representing Mixtec day signs
- Supports CUDA, Apple MPS, and CPU training

## The 20 Mixtec Day Signs

| Symbol | Meaning | Symbol | Meaning |
|--------|---------|--------|---------|
| jaguar | ocelotl | grass | malinalli |
| movement | ollin | crocodile | cipactli |
| eagle | cuauhtli | serpent | coatl |
| flint | tecpatl | monkey | ozomatli |
| flower | xochitl | deer | mazatl |
| wind | ehecatl | vulture | cozcacuauhtli |
| rain | quiahuitl | house | calli |
| dog | itzcuintli | death | miquiztli |
| rabbit | tochtli | water | atl |
| reed | acatl | lizard | cuetzpalin |

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
Symbol Classifier/
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
    ├── train/
    │   ├── jaguar/
    │   ├── movement/
    │   ├── eagle/
    │   └── ... (20 class subdirectories)
    └── test/
        ├── jaguar/
        ├── movement/
        └── ...
```

## Configuration (config.toml)

```toml
[paths]
train_dir = "./sign_images/train"    # Training images
test_dir = "./sign_images/test"      # Test images
model_dir = "models"                  # Model output directory

[model]
in_features = 768                     # ViT-B-16 output features
seed = 42                             # Random seed

[training]
epochs = 10                           # Training epochs
batch_size = 32                       # Batch size
learning_rate = 1e-3                  # Learning rate

[workers]
cuda = 16                             # CUDA workers
mps = 8                               # MPS workers
cpu = 6                               # CPU workers

[classes]
names = [
    "jaguar", "movement", "eagle", "flint", "flower",
    "wind", "rain", "dog", "rabbit", "reed",
    "grass", "crocodile", "serpent", "monkey", "deer",
    "vulture", "house", "death", "water", "lizard"
]

[augmentation]
num_rotations = 16
rotation_angles = [15, 30, 45, ...]
brightness_range = [0.8, 1.2]
contrast_range = [0.8, 1.2]
color_range = [0.0, 2.0]
mask_width_fraction = [0.125, 0.25]
mask_height_fraction = [0.125, 0.25]
```

## Code Walkthrough

### 1. Data Augmentation (augment_images.py)

Augmentation is critical for this 20-class problem to prevent overfitting:

#### Configuration Loading

```python
def load_augmentation_config(config_path: Path) -> Dict:
    """Load augmentation parameters from config.toml."""
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = tomli.load(f)
            aug_config = config.get("augmentation", {})
            return {**DEFAULT_CONFIG, **aug_config}
    return DEFAULT_CONFIG
```

#### Rotation Augmentation

```python
def apply_rotations(image_directory, rotation_angles, num_rotations):
    """Create rotated versions of each image."""
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            if filename.endswith(".png"):
                with Image.open(image_path) as img:
                    k = min(num_rotations, len(rotation_angles))
                    for angle in random.sample(rotation_angles, k=k):
                        rotated_img = img.rotate(angle, expand=True)
                        rotated_img.save(f"rot_{angle}_{filename}")
```

Symbols appear at various orientations in manuscripts, so rotation augmentation is essential.

#### Color Jitter

```python
def apply_color_jitter(img, brightness_range, contrast_range, color_range):
    """Vary color properties for robustness."""
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

Handles variation in ink fading, scanning conditions, and manuscript preservation.

#### Random Masking

```python
def apply_random_mask(img, mask_width_fraction, mask_height_fraction):
    """Simulate occlusion and damage."""
    draw = ImageDraw.Draw(img_with_mask)
    # Random white rectangle placement
    draw.rectangle([...], fill="white")
    return img_with_mask
```

Helps model handle partially visible symbols and damaged manuscript areas.

### 2. Training Pipeline (classification.py)

#### Configuration Loading

```python
CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(CONFIG_PATH, "rb") as f:
    config = tomli.load(f)

class_names = config["classes"]["names"]  # 20 symbol classes
```

#### Model Setup for 20-Class Classification

```python
# Get pretrained ViT-B-16
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Freeze base (feature extractor)
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# New 20-class classification head
pretrained_vit.heads = nn.Linear(
    in_features=768,       # ViT-B-16 feature dimension
    out_features=20        # 20 day sign symbols
).to(device)
```

#### DataLoader Creation

```python
def create_dataloaders(train_dir, test_dir, transform, batch_size, num_workers):
    """Create loaders for 20-class ImageFolder dataset."""
    # ImageFolder maps subdirectory names to class indices
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    # train_data.classes = ['crocodile', 'death', 'deer', ...]

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
# Use ViT's default transforms (resize, normalize)
pretrained_vit_transforms = pretrained_vit_weights.transforms()

optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=config["training"]["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()

results = engine.train(
    model=pretrained_vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=config["training"]["epochs"],
    device=device,
    class_names=class_names  # All 20 symbol names
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

        # Forward pass through frozen ViT + trainable head
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Multi-class accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    return train_loss / len(dataloader), train_acc / len(dataloader)
```

#### Evaluation with 20x20 Confusion Matrix

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

#### Full Training Loop with Multi-Class Metrics

```python
def train(model, train_dataloader, test_dataloader, optimizer, loss_fn,
          epochs, device, class_names):
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(...)
        test_loss, test_acc, all_preds, all_labels = test_step(...)

        # Generate 20x20 confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))  # Larger figure for 20 classes
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,  # 20 labels
            yticklabels=class_names
        )
        plt.savefig(f"confusion_matrix/epoch_{epoch+1}.png")

        # Macro-averaged metrics across all 20 classes
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

    return results
```

### 4. Model Utilities (engine/utils.py)

```python
def save_model(model, target_dir, model_name):
    """Save complete model with 20-class head."""
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model, f=target_dir_path / model_name)

def load_model(model_path, device):
    """Load saved model for inference."""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model
```

### 5. Helper Functions (helper_functions.py)

```python
def plot_loss_curves(results, filename_prefix="loss_curves"):
    """Visualize training progress."""
    # Loss plot
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.savefig(f"{filename_prefix}_loss.png")

    # Accuracy plot (challenging with 20 classes)
    plt.plot(epochs, results["train_acc"], label="train_accuracy")
    plt.plot(epochs, results["test_acc"], label="test_accuracy")
    plt.savefig(f"{filename_prefix}_accuracy.png")

def set_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

## Outputs

After training:

- `models/symbol_classifier_vit_YYYYMMDD_HHMMSS.pth` - Trained model
- `loss_curves_loss.png` - Loss curves over epochs
- `loss_curves_accuracy.png` - Accuracy curves over epochs
- `confusion_matrix/epoch_N.png` - 20x20 confusion matrix per epoch
- `confusion_matrix/metrics_log.txt` - Macro precision, recall, F1 per epoch

## Interpreting 20-Class Results

With 20 classes:
- **Random chance accuracy**: 5% (1/20)
- **Macro-averaged metrics**: Equal weight to each class
- **Confusion matrix patterns**: Look for symbol pairs that are commonly confused
- **Class imbalance**: Some symbols may appear more frequently than others

## Adapting for Your Project

- [ ] **Update class names** in `config.toml` - can reduce or increase classes
- [ ] **Organize dataset** with one subdirectory per class:
  ```
  your_images/
  ├── train/
  │   ├── class_1/
  │   ├── class_2/
  │   └── ... (N classes)
  └── test/
      ├── class_1/
      └── ...
  ```
- [ ] **Update paths** in `[paths]` section
- [ ] **Adjust hyperparameters** for multi-class:
  - May need more `epochs` for many classes
  - Lower `learning_rate` if overfitting
  - Larger `batch_size` if GPU memory allows
- [ ] **Consider class weights** in loss function if imbalanced
- [ ] **Update model name** in `classification.py` (line ~159)

## Dependencies

Managed via `pyproject.toml`:

```
loguru          - Structured logging
matplotlib      - Plotting (large confusion matrices)
numpy           - Array operations
pillow          - Image augmentation
scikit-learn    - Multi-class metrics (macro averaging)
seaborn         - Heatmap visualization
tomli           - TOML configuration
torch           - PyTorch framework
torchinfo       - Model architecture summaries
torchvision     - ViT-B-16 and transforms
tqdm            - Progress tracking
```
