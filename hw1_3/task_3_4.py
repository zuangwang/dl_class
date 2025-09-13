import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CifarCNN

# --- Configuration ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZES = [32, 1024]  # Batch sizes for model 1 and model 2
MODEL_CONFIG = {"base_channels": 32, "fc_layers": [512, 256]}
INTERPOLATION_STEPS = 21 # Number of points to check between models (e.g., 21 for steps of 0.05)

# --- Setup Results Directory ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results', f'hw1_3_interpolation_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# --- Data Loading ---
print("Loading CIFAR-10 data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
# Use a fixed loader for evaluation
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
print("Data loading complete.")

# --- Helper Functions ---
def compute_metrics(loader, model, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def train_model(batch_size, lr, epochs):
    print(f"\n--- Training model with Batch Size: {batch_size} ---")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = CifarCNN(**MODEL_CONFIG)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        test_loss, test_acc = compute_metrics(test_loader, model, criterion)
        print(f"Epoch {epoch+1}/{epochs} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("Training complete.")
    return model

# --- Main Execution ---

# 1. Train the two models
model_1 = train_model(batch_size=BATCH_SIZES[0], lr=LEARNING_RATE, epochs=NUM_EPOCHS)
model_2 = train_model(batch_size=BATCH_SIZES[1], lr=LEARNING_RATE, epochs=NUM_EPOCHS)

# Get the state dictionaries
theta_1 = model_1.state_dict()
theta_2 = model_2.state_dict()

# 2. Interpolate and Evaluate
print("\n--- Performing linear interpolation and evaluation ---")
alphas = np.linspace(0, 1, INTERPOLATION_STEPS)
interpolation_results = {"alphas": [], "losses": [], "accuracies": []}
criterion = nn.CrossEntropyLoss()

# Create a model to load the interpolated weights into
model_interp = CifarCNN(**MODEL_CONFIG)

for alpha in alphas:
    print(f"Evaluating for alpha = {alpha:.2f}")
    
    # Create the interpolated state dictionary
    theta_interp = {}
    for key in theta_1:
        theta_interp[key] = (1 - alpha) * theta_1[key] + alpha * theta_2[key]
    
    # Load the new weights and evaluate
    model_interp.load_state_dict(theta_interp)
    loss, acc = compute_metrics(test_loader, model_interp, criterion)
    
    interpolation_results["alphas"].append(alpha)
    interpolation_results["losses"].append(loss)
    interpolation_results["accuracies"].append(acc)

# 3. Visualization
print("\nGenerating interpolation plot...")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Loss on the primary y-axis
color = 'tab:red'
ax1.set_xlabel('Interpolation Ratio (α)')
ax1.set_ylabel('Test Loss', color=color)
ax1.plot(interpolation_results["alphas"], interpolation_results["losses"], color=color, marker='o', label='Test Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which="both", ls="--")

# Create a second y-axis for Accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Accuracy (%)', color=color)
ax2.plot(interpolation_results["alphas"], interpolation_results["accuracies"], color=color, marker='x', label='Test Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Add annotations for the start and end models
plt.title(f'Linear Interpolation Between Models (BS={BATCH_SIZES[0]} vs BS={BATCH_SIZES[1]})')
fig.tight_layout()
plt.savefig(os.path.join(results_dir, 'interpolation_path.png'))
plt.close()
