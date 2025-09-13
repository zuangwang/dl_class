import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime

# Add the parent directory to the system path to find the 'model' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import FunsimDNN

# --- Configuration ---
# Use a simple, reproducible function for consistent experiments
def target_function(x):
    return np.sin(x * 2 * np.pi)

def generate_data(n=1000):
    x = np.random.uniform(0, 1, n)
    y = target_function(x)
    return torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y).float().unsqueeze(1)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000):
        self.x, self.y = generate_data(n)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --- Experiment Settings ---
IN_DIM = 1
OUT_DIM = 1
HIDDEN_DIM = [32, 32]
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
BATCH_SIZE = 64
NUM_TRAINING_RUNS_VIZ = 8
NUM_TRAINING_RUNS_SHARPNESS = 100

# Create a timestamped directory for results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# Construct a robust path to the project's results directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(project_root, 'results', f'hw1_2_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

# --- Helper Functions ---
def get_params(model, layer_name=None):
    """Extracts flattened parameters from a model or a specific layer."""
    if layer_name:
        layer = dict(model.named_parameters())[layer_name]
        return layer.detach().cpu().numpy().flatten()
    else:
        params = [p.detach().cpu().numpy().flatten() for p in model.parameters()]
        return np.concatenate(params)

def run_training_and_collect_params(num_epochs):
    """Trains a model and collects parameter trajectories and gradient norms."""
    train_dataset = SimpleDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FunsimDNN(IN_DIM, OUT_DIM, HIDDEN_DIM)
    print([name for name, _ in model.named_parameters()])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Parameter trajectories
    whole_model_params_traj = []
    first_layer_params_traj = []
    
    # Gradient and loss per iteration
    grad_norms_per_iter = []
    loss_per_iter = []

    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
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
            grad_norms_per_iter.append(total_norm ** 0.5)
            loss_per_iter.append(loss.item())
            
            optimizer.step()
        
        # Collect parameters at the end of each epoch
        whole_model_params_traj.append(get_params(model))
        first_layer_params_traj.append(get_params(model, 'net.0.weight'))

    final_loss = loss_per_iter[-1]
    return model, final_loss, whole_model_params_traj, first_layer_params_traj, grad_norms_per_iter, loss_per_iter

def calculate_minimal_ratio_by_sampling(model, criterion, original_loss, num_samples=100, epsilon=1e-4):
    """
    Calculates the minimal ratio by sampling points around a minimum.
    The ratio is the proportion of sampled points with a higher loss.
    """
    dataset = SimpleDataset(n=500)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    x, y = next(iter(loader))

    model.eval()
    
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    higher_loss_count = 0
    for _ in range(num_samples):
        # Create a new state dict for the perturbed model
        perturbed_state_dict = {k: v.clone() for k, v in original_state_dict.items()}
        
        # Add a random perturbation to all parameters
        for param_name in perturbed_state_dict:
            param = perturbed_state_dict[param_name]
            perturbation = torch.randn_like(param) * epsilon
            perturbed_state_dict[param_name] += perturbation
            
        # Load the perturbed weights and calculate loss
        model.load_state_dict(perturbed_state_dict)
        with torch.no_grad():
            perturbed_loss = criterion(model(x), y).item()
        
        if perturbed_loss > original_loss:
            higher_loss_count += 1
            
    # Restore the original model weights
    model.load_state_dict(original_state_dict)
    
    return higher_loss_count / num_samples

# --- Main Execution ---

# 1. Visualize the optimization process
print("Running experiment 1: Visualizing optimization trajectories...")
all_runs_whole_model_traj = []
all_runs_first_layer_traj = []
for i in range(NUM_TRAINING_RUNS_VIZ):
    print(f"  Training run {i+1}/{NUM_TRAINING_RUNS_VIZ}...")
    _, _, whole_traj, layer_traj, _, _ = run_training_and_collect_params(NUM_EPOCHS)
    all_runs_whole_model_traj.append(np.array(whole_traj))
    all_runs_first_layer_traj.append(np.array(layer_traj))

# PCA and Plotting
pca_whole = PCA(n_components=2)
pca_layer = PCA(n_components=2)

# Fit PCA on all trajectories combined to have a consistent projection
pca_whole.fit(np.vstack(all_runs_whole_model_traj))
pca_layer.fit(np.vstack(all_runs_first_layer_traj))

# Plot whole model trajectories
plt.figure(figsize=(10, 8))
for traj in all_runs_whole_model_traj:
    projected_traj = pca_whole.transform(traj)
    plt.plot(projected_traj[:, 0], projected_traj[:, 1], marker='o', markersize=2, alpha=0.7)
    plt.scatter(projected_traj[0, 0], projected_traj[0, 1], marker='x', s=100, label='Start' if plt.gca().get_legend() is None else "")
    plt.scatter(projected_traj[-1, 0], projected_traj[-1, 1], marker='*', s=100, label='End' if plt.gca().get_legend() is None else "")
plt.title('PCA of Whole Model Parameter Trajectories (8 Runs)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig(os.path.join(results_dir, 'pca_whole_model.png'))
plt.close()

# Plot first layer trajectories
plt.figure(figsize=(10, 8))
for traj in all_runs_first_layer_traj:
    projected_traj = pca_layer.transform(traj)
    plt.plot(projected_traj[:, 0], projected_traj[:, 1], marker='o', markersize=2, alpha=0.7)
    plt.scatter(projected_traj[0, 0], projected_traj[0, 1], marker='x', s=100, label='Start' if plt.gca().get_legend() is None else "")
    plt.scatter(projected_traj[-1, 0], projected_traj[-1, 1], marker='*', s=100, label='End' if plt.gca().get_legend() is None else "")
plt.title('PCA of First Layer Parameter Trajectories (8 Runs)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig(os.path.join(results_dir, 'pca_first_layer.png'))
plt.close()

# 2. Observe gradient norm during training
print("\nRunning experiment 2: Observing gradient norm...")
_, _, _, _, grad_norms, losses = run_training_and_collect_params(NUM_EPOCHS)

fig, ax1 = plt.subplots(figsize=(12, 6))
iterations = np.arange(len(grad_norms))
# Plot Loss
color = 'tab:blue'
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss', color=color)
ax1.plot(iterations, losses, color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
# Plot Gradient Norm
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Gradient Norm', color=color)
ax2.plot(iterations, grad_norms, color=color, alpha=0.7, label='Gradient Norm')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')
plt.title('Loss and Gradient Norm vs. Iterations')
fig.tight_layout()
plt.savefig(os.path.join(results_dir, 'loss_vs_grad_norm.png'))
plt.close()

# 3. Minimal Ratio vs. Loss
print(f"\nRunning experiment 3: Minimal Ratio vs. Loss ({NUM_TRAINING_RUNS_SHARPNESS} runs)...")
final_losses = []
minimal_ratios = []
criterion = nn.MSELoss()
for i in range(NUM_TRAINING_RUNS_SHARPNESS):
    print(f"  Training run {i+1}/{NUM_TRAINING_RUNS_SHARPNESS}...")
    model, final_loss, _, _, _, _ = run_training_and_collect_params(num_epochs=50) # Shorter training for more runs
    
    # Get the true loss at the minimum, not just the last batch loss
    dataset = SimpleDataset(n=1000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    x_full, y_full = next(iter(loader))
    model.eval()
    with torch.no_grad():
        loss_at_minimum = criterion(model(x_full), y_full).item()

    ratio = calculate_minimal_ratio_by_sampling(model, criterion, loss_at_minimum)
    
    final_losses.append(loss_at_minimum)
    minimal_ratios.append(ratio)

plt.figure(figsize=(10, 6))
plt.scatter(minimal_ratios, final_losses, alpha=0.6)
plt.title('Loss vs. Minimal Ratio')
plt.xlabel('Minimal Ratio (Sampling Method)')
plt.ylabel('Final Loss')
plt.yscale('log')
# Minimal ratio is between 0 and 1, so no log scale on x-axis
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(results_dir, 'loss_vs_minimal_ratio.png'))
plt.close()

print(f"\nAll experiments complete. Results are saved in: {results_dir}")