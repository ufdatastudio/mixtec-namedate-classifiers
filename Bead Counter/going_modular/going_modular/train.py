"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_setup, engine, model_builder, utils

from torchvision import transforms


class OrdinalLoss(nn.Module):
    """
    Ordinal loss function that penalizes predictions based on distance from true value.
    Better suited for counting tasks where order matters.
    """
    def __init__(self, num_classes, alpha=1.0):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Create class indices tensor
        class_indices = torch.arange(self.num_classes, device=predictions.device).float()
        
        # Calculate expected class for each prediction
        expected_class = torch.sum(probs * class_indices.unsqueeze(0), dim=1)
        
        # Calculate L1 distance between expected and true class
        distance_loss = F.l1_loss(expected_class, targets.float())
        
        # Add cross-entropy component for classification
        ce_loss = F.cross_entropy(predictions, targets)
        
        # Combine losses with weighting
        return ce_loss + self.alpha * distance_loss

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = OrdinalLoss(num_classes=len(class_names), alpha=2.0)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
