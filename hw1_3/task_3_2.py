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
import csv

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CifarCNN

# --- Configuration ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 2 # Increased epochs for a more complex dataset
BATCH_SIZE = 128

MODEL_CONFIGS = [
    {"base_channels": 1, "fc_layers": [128]},
    {"base_channels": 5, "fc_layers": [128]},     
    {"base_channels": 10, "fc_layers": [128]},     
    {"base_channels": 14, "fc_layers": [128]},        
    {"base_channels": 16, "fc_layers": [128]},        
    {"base_channels": 18, "fc_layers": [128]},        
    {"base_channels": 20, "fc_layers": [128]},        
    {"base_channels": 16, "fc_layers": [256]},        
    {"base_channels": 22, "fc_layers": [128]},       
    {"base_channels": 18, "fc_layers": [256]},    
    {"base_channels": 24, "fc_layers": [128]},        
    {"base_channels": 20, "fc_layers": [256]},      
    {"base_channels": 22, "fc_layers": [256]},      
]

# --- Setup Results Directory ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results', f'hw1_3_capacity_cifar_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# --- Data Loading ---
print("Loading CIFAR-10 data...")
# Normalization for 3-channel CIFAR-10 images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

# --- Main Execution ---
all_model_results = []

for i, config in enumerate(MODEL_CONFIGS):
    model = CifarCNN(**config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Training Model {i+1}/{len(MODEL_CONFIGS)} ---")
    print(f"Configuration: {config}, Parameters: {num_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Compute metrics at the end of the epoch
        train_loss, train_acc = compute_metrics(train_loader, model, criterion)
        test_loss, test_acc = compute_metrics(test_loader, model, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    all_model_results.append({
        "config": config,
        "params": num_params,
        "history": history,
        "model_state_dict": model.state_dict()
    })

# --- Detailed Saving and Plotting ---
print("\nSaving detailed results for each model...")
for i, result in enumerate(all_model_results):
    config_dir = os.path.join(results_dir, f"model_{i+1}")
    os.makedirs(config_dir, exist_ok=True)

    # Save model state
    torch.save(result['model_state_dict'], os.path.join(config_dir, 'model.pth'))

    # Save config info
    with open(os.path.join(config_dir, 'info.md'), 'w') as f:
        f.write(f"# Model {i+1} Information\n\n")
        f.write(f"**Base Channels:** `{result['config']['base_channels']}`\n")
        f.write(f"**FC Layer Config:** `{result['config']['fc_layers']}`\n")
        f.write(f"**Parameters:** {result['params']:,}\n")

    # Save metrics to CSV
    with open(os.path.join(config_dir, 'metrics.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc'])
        for epoch in range(NUM_EPOCHS):
            writer.writerow([
                epoch + 1,
                result['history']['train_loss'][epoch],
                result['history']['train_acc'][epoch],
                result['history']['test_loss'][epoch],
                result['history']['test_acc'][epoch]
            ])

    # Save individual plots
    plt.figure(figsize=(8, 5))
    plt.plot(result["history"]["train_loss"], label='Training Loss')
    plt.plot(result["history"]["test_loss"], label='Test Loss', linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Model {i+1} - Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(config_dir, 'loss_comparison.png')); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(result["history"]["train_acc"], label='Training Accuracy')
    plt.plot(result["history"]["test_acc"], label='Test Accuracy', linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title(f'Model {i+1} - Accuracy'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(config_dir, 'accuracy_comparison.png')); plt.close()

# --- Combined Visualization ---
print("Generating summary plots...")

# Extract final metrics for plotting
param_counts = [res['params'] for res in all_model_results]
final_train_losses = [res['history']['train_loss'][-1] for res in all_model_results]
final_test_losses = [res['history']['test_loss'][-1] for res in all_model_results]
final_train_accs = [res['history']['train_acc'][-1] for res in all_model_results]
final_test_accs = [res['history']['test_acc'][-1] for res in all_model_results]

# Plot Final Accuracy (Train vs. Test) vs. Number of Parameters
plt.figure(figsize=(10, 6))
plt.plot(param_counts, final_train_accs, marker='o', linestyle='--', label='Final Train Accuracy')
plt.plot(param_counts, final_test_accs, marker='o', linestyle='-', label='Final Test Accuracy')
plt.title('Final Accuracy vs. Model Capacity')
plt.xlabel('Number of Parameters'); plt.ylabel('Final Accuracy (%)'); plt.xscale('log'); plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(results_dir, 'accuracy_vs_capacity.png')); plt.close()

# Plot Final Loss (Train vs. Test) vs. Number of Parameters
plt.figure(figsize=(10, 6))
plt.plot(param_counts, final_train_losses, marker='s', linestyle='--', label='Final Train Loss')
plt.plot(param_counts, final_test_losses, marker='s', linestyle='-', label='Final Test Loss')
plt.title('Final Loss vs. Model Capacity'); plt.xlabel('Number of Parameters'); plt.ylabel('Final Loss')
plt.xscale('log'); plt.yscale('log'); plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(results_dir, 'loss_vs_capacity.png')); plt.close()

print(f"\nExperiment complete. Results are saved in: {results_dir}")