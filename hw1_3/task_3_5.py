import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.autograd.functional import jacobian

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CifarCNN

# --- Configuration ---
NUM_EPOCHS = 2
LEARNING_RATE = 0.001 # Use a fixed learning rate
# Use different batch sizes as the "different training approaches"
BATCH_SIZES = [32, 64, 128, 256, 512]
MODEL_CONFIG = {"base_channels": 32, "fc_layers": [512, 256]}
SENSITIVITY_SAMPLES = 50 # Number of test images to use for sensitivity calculation

# --- Setup Results Directory ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results', f'hw1_3_sensitivity_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# --- Data Loading ---
print("Loading CIFAR-10 data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False) # Use a large batch for stable evaluation

# Create a smaller subset for the expensive sensitivity calculation
sensitivity_subset = Subset(test_dataset, range(SENSITIVITY_SAMPLES))
sensitivity_loader = DataLoader(sensitivity_subset, batch_size=1) # Process one image at a time
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

def calculate_jacobian_sensitivity(model, data_loader):
    """
    Calculates sensitivity as the average Frobenius norm of the Jacobian of the 
    class probabilities with respect to the input.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    total_norm = 0.0
    count = 0

    # Define the function whose Jacobian we want: p(x) = softmax(model(x))
    def model_prob_func(x_input):
        return softmax(model(x_input))

    print(f"  Calculating Jacobian sensitivity over {len(data_loader)} samples...")
    with torch.no_grad(): # We don't need gradients for the overall calculation loop
        for i, (x, y) in enumerate(data_loader):
            # jacobian function requires a function that takes a tensor and returns a tensor
            # and the input tensor. It computes J at the point x.
            # We need to enable gradients temporarily for the jacobian calculation itself.
            with torch.enable_grad():
                x.requires_grad_()
                jac = jacobian(model_prob_func, x)
            
            # jac shape is (num_classes, batch_size, C, H, W). Here batch_size=1.
            # Squeeze out the batch dimension: (10, 1, 3, 32, 32) -> (10, 3, 32, 32)
            # Reshape the 6D Jacobian (1, 10, 1, 3, 32, 32) into a 2D matrix (10, 3072)
            jac_2d = jac.reshape(10, -1)
            norm = torch.linalg.norm(jac_2d, "fro")
            total_norm += norm.item()
            count += 1
            if (i + 1) % 10 == 0:
                print(f"    ...processed {i+1}/{len(data_loader)} samples")

    return total_norm / count if count > 0 else 0

# --- Main Execution ---
all_results = []
criterion = nn.CrossEntropyLoss()

for batch_size in BATCH_SIZES:
    print(f"\n--- Training with Batch Size: {batch_size} ---")
    
    # Create a new train loader for the current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = CifarCNN(**MODEL_CONFIG)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            test_loss, test_acc = compute_metrics(test_loader, model, criterion)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Final evaluation
    final_loss, final_acc = compute_metrics(test_loader, model, criterion)
    print(f"Final Test Loss: {final_loss:.4f}, Final Test Accuracy: {final_acc:.2f}%")

    # Calculate sensitivity
    sensitivity = calculate_jacobian_sensitivity(model, sensitivity_loader)
    print(f"Calculated Sensitivity (Avg. Jacobian Norm): {sensitivity:.4f}")
    
    all_results.append({
        "batch_size": batch_size,
        "loss": final_loss,
        "accuracy": final_acc,
        "sensitivity": sensitivity
    })

# --- Visualization ---
print("\nGenerating summary plots...")
batch_sizes = [res['batch_size'] for res in all_results]
accuracies = [res['accuracy'] for res in all_results]
sensitivities = [res['sensitivity'] for res in all_results]
losses = [res['loss'] for res in all_results]

# Plot Accuracy vs. Batch Size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, accuracies, marker='o', linestyle='-')
plt.title('Final Test Accuracy vs. Batch Size')
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Final Test Accuracy (%)')
plt.xscale('log')
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(results_dir, 'accuracy_vs_batch_size.png'))
plt.close()

# Plot Loss vs. Batch Size
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, losses, marker='s', linestyle='-', color='orange')
plt.title('Final Test Loss vs. Batch Size')
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Final Test Loss')
plt.xscale('log')
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(results_dir, 'loss_vs_batch_size.png'))
plt.close()

# Combined plot for Accuracy and Sensitivity
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Batch Size (log scale)')
ax1.set_ylabel('Final Test Accuracy (%)', color=color)
ax1.plot(batch_sizes, accuracies, color=color, marker='o', label='Test Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')
ax1.grid(True, which="both", ls="--")

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Sensitivity (Avg. Jacobian Norm)', color=color)
ax2.plot(batch_sizes, sensitivities, color=color, marker='x', linestyle='--', label='Sensitivity')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Test Accuracy and Input Sensitivity vs. Batch Size')
fig.tight_layout()
plt.savefig(os.path.join(results_dir, 'accuracy_and_sensitivity_vs_batch_size.png'))
plt.close()

print(f"\nExperiment complete. Results are saved in: {results_dir}")