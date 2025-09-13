import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import CifarCNN

# --- Configuration ---
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZES = [32, 1024]  # Small and Large batch sizes for comparison
MODEL_CONFIG = {"base_channels": 32, "fc_layers": [512, 256]} # A reasonably sized model

# --- Setup Results Directory ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results', f'hw1_3_batch_size_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# --- Data Loading ---
print("Loading CIFAR-10 data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
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

def train_model(model, train_loader, test_loader, lr, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        train_loss, train_acc = compute_metrics(train_loader, model, criterion)
        test_loss, test_acc = compute_metrics(test_loader, model, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    return history

# --- Main Execution ---
all_results = {}

for batch_size in BATCH_SIZES:
    print(f"\n--- Training with Batch Size: {batch_size} ---")
    
    # Create new DataLoaders with the specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize a new model for a fair comparison
    model = CifarCNN(**MODEL_CONFIG)
    
    history = train_model(model, train_loader, test_loader, LEARNING_RATE, NUM_EPOCHS)
    all_results[batch_size] = history

# --- Visualization ---
print("\nGenerating comparison plots...")

# Plot Loss Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for bs, history in all_results.items():
    plt.plot(history['train_loss'], label=f'Train Loss (BS={bs})')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
for bs, history in all_results.items():
    plt.plot(history['test_loss'], label=f'Test Loss (BS={bs})')
plt.title('Test Loss Comparison')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'loss_comparison.png'))
plt.close()

# Plot Accuracy Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for bs, history in all_results.items():
    plt.plot(history['train_acc'], label=f'Train Acc (BS={bs})')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
for bs, history in all_results.items():
    plt.plot(history['test_acc'], label=f'Test Acc (BS={bs})')
plt.title('Test Accuracy Comparison')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'))
plt.close()

print(f"\nExperiment complete. Results are saved in: {results_dir}")