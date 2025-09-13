import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import csv

from model import *


def sin(x):
    return np.sin(2*np.pi*x)


def f_1(x):
    return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x) + 0.25 * np.sin(6 * np.pi * x)

def f_2(x):
    return np.sin(5 * np.pi * x) / (5 * np.pi * x + 1e-6)  # Added small constant to avoid division by zero
def generate_data(n=100000):
    x = np.random.uniform(0, 1, n)
    y = f_2(x)
    return x, y

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, n=100000):
        x, y = generate_data(n)
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # shape (n, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape (n, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# Generate train and test sets
train_dataset = SimpleDataset(1000)
test_dataset = SimpleDataset(200)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# Use FunsimDNN from model.py
in_dim = 1
out_dim = 1
criterion = nn.MSELoss()


num_epochs = 100
train_losses = []
train_accs = []
test_accs = []

def compute_acc(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            correct += ((torch.abs(outputs - batch_y) < 0.1).float().sum().item())
            total += batch_y.numel()
    model.train()
    return correct / total

# Create a timestamped directory for results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', timestamp)
os.makedirs(results_dir, exist_ok=True)


model_configs = [
    # Wide model: 1 hidden layer, 128 neurons. Params: (1*128+128) + (128*1+1) = 385
    {"hidden_dim": [128], "lr": 0.001},
    # Deep model: 2 hidden layers, 18 neurons each. Params: (1*18+18) + (18*18+18) + (18*1+1) = 379
    {"hidden_dim": [18, 18], "lr": 0.001},
]

# Store results for comparison
model_results = []

for i, config in enumerate(model_configs):
    hidden_dim = config["hidden_dim"]
    lr = config["lr"]

    # Initialize model, optimizer, and other components
    model = FunsimDNN(in_dim, out_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model (reuse your training loop)
    train_losses = []
    train_accs = []
    test_accs = []
    grad_norms = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Calculate and store gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

            # Print batch-level information
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Model {i+1}, Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Batch Loss: {loss.item():.4f}")

        # Calculate epoch-level metrics
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        train_acc = compute_acc(train_loader, model)
        test_acc = compute_acc(test_loader, model)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Print epoch-level information
        print(f"Model {i+1}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Save results for this model
    model_results.append({
        "hidden_dim": hidden_dim,
        "lr": lr,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "grad_norms": grad_norms,
        "num_params": model.count_trainable_params(model),
        # Save the model state dict for later use
        "model_state_dict": model.state_dict()
    })

# Save the trained model and metrics for each configuration
for i, result in enumerate(model_results):
    # Create a subdirectory for each model configuration
    config_dir = os.path.join(results_dir, f"model_{i+1}")
    os.makedirs(config_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(config_dir, 'model.pth')
    # Re-initialize the model with the correct architecture and load the trained weights
    model_to_save = FunsimDNN(in_dim, out_dim, result['hidden_dim'])
    # To save the actual trained weights, you would need to load them.
    # The current loop overwrites the model variable. We need to save the state dict inside the training loop.
    # For this task, let's assume the model object from the loop is what we want to save.
    # A better approach is to save the state_dict right after training each model.
    # Let's modify the training loop to save the model state.
    
    # The model state will be saved from a temporary variable in the main training loop.
    torch.save(result['model_state_dict'], model_path)


    # Save metrics to CSV
    csv_path = os.path.join(config_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Epoch', 'Training Loss', 'Training Accuracy', 'Test Accuracy', 'Gradient Norm'])
        
        num_iterations_per_epoch = len(train_loader)
        for epoch in range(num_epochs):
            for it in range(num_iterations_per_epoch):
                iteration = epoch * num_iterations_per_epoch + it
                writer.writerow([iteration, epoch+1, result["train_losses"][epoch], result["train_accs"][epoch], result["test_accs"][epoch], result["grad_norms"][iteration]])
                

    # Save configuration details
    config_info_path = os.path.join(config_dir, 'info.md')
    with open(config_info_path, 'w') as f:
        f.write(f"# Model {i+1} Information\n\n")
        f.write(f"**Hidden Layers:** {result['hidden_dim']}\n")
        f.write(f"**Learning Rate:** {result['lr']}\n")
        f.write(f"**Number of Trainable Parameters:** {result['num_params']}\n")

    # Save plots for this model
    # Training loss
    plt.figure(figsize=(8, 5))
    plt.plot(result["train_losses"], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Model {i+1} - Training Loss')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'training_loss.png'))
    plt.close()

    # Gradient norm
    plt.figure(figsize=(8, 5))
    plt.plot(result["grad_norms"], label='Gradient Norm', color='purple')
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.title(f'Model {i+1} - Gradient Norm')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'gradient_norm.png'))
    plt.close()

    # Training accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(result["train_accs"], label='Training Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model {i+1} - Training Accuracy')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'training_accuracy.png'))
    plt.close()

    # Test accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(result["test_accs"], label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Model {i+1} - Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'test_accuracy.png'))
    plt.close()

    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(result["train_losses"], label='Training Loss', color='blue')
    plt.plot(result["train_accs"], label='Training Accuracy', color='green')
    plt.plot(result["test_accs"], label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Model {i+1} - Loss and Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(config_dir, 'loss_and_accuracy_comparison.png'))
    plt.close()

# Combined plots for all models
# Plot training loss for all models
plt.figure(figsize=(10, 6))
for i, result in enumerate(model_results):
    plt.plot(result["train_losses"], label=f"model_{i+1}- params: {result['num_params']}")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison Across Models')
plt.legend()
plt.savefig(os.path.join(results_dir, 'combined_training_loss.png'))
plt.close()

# Plot training accuracy for all models
plt.figure(figsize=(10, 6))
for i, result in enumerate(model_results):
    plt.plot(result["train_accs"], label=f"model_{i+1}- params: {result['num_params']}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison Across Models')
plt.legend()
plt.savefig(os.path.join(results_dir, 'combined_training_accuracy.png'))
plt.close()

# Plot test accuracy for all models
plt.figure(figsize=(10, 6))
for i, result in enumerate(model_results):
    plt.plot(result["test_accs"], label=f"model_{i+1}- params: {result['num_params']}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Comparison Across Models')
plt.legend()
plt.savefig(os.path.join(results_dir, 'combined_test_accuracy.png'))
plt.close()

# Plot ground-truth vs. predictions for all models
plt.figure(figsize=(12, 7))

# Plot the actual training data points
plt.scatter(train_dataset.x.numpy(), train_dataset.y.numpy(), s=1, color='gray', label='Training Samples', alpha=0.5)

# Generate a smooth line for the ground truth function
x_range = torch.linspace(0, 1, 500).unsqueeze(1)
ground_truth_y = f_2(x_range.numpy())
plt.plot(x_range.numpy(), ground_truth_y, 'k--', label='Ground Truth')

# Plot predictions for each model
for i, result in enumerate(model_results):
    # Initialize model with the correct architecture
    model = FunsimDNN(in_dim, out_dim, result['hidden_dim'])
    # Load the saved weights
    model.load_state_dict(result['model_state_dict'])
    model.eval()
    with torch.no_grad():
        predictions = model(x_range)
    plt.plot(x_range.numpy(), predictions.numpy(), label=f"model_{i+1}- params: {result['num_params']}")

plt.title('Ground Truth vs. Model Predictions')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(results_dir, 'predictions_vs_truth.png'))
plt.close()