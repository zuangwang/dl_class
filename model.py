import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch

class FunsimDNN(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: list[int]
        ):
        
        super().__init__()
        
        self.activation = nn.ReLU(inplace=True)
        layers: list[nn.Module] = []
        dims = [in_dim] + hidden_dim + [out_dim]
        
        # layers
        for i in range(len(dims)-1):
            in_f, out_f = dims[i], dims[i+1]
            layers.append(nn.Linear(in_f, out_f, bias=True))
            if i < len(dims)-2:
                layers.append(self.activation)
        self.net = nn.Sequential(*layers)
            
    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.net(x)

    
    def count_trainable_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
class CifarCNN(nn.Module):
    def __init__(self, base_channels=16, fc_layers=[128]):
        super(CifarCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1)

        # Calculate the size of the flattened features after conv layers
        # CIFAR-10 images are 32x32. After 3 pooling layers (2x2), size becomes 32 -> 16 -> 8 -> 4.
        # So, the feature map size is 4x4.
        self.flattened_features = (base_channels * 8) * 4 * 4
        
        # Dynamically create fully connected layers
        fc_net = []
        in_features = self.flattened_features
        for hidden_size in fc_layers:
            fc_net.append(nn.Linear(in_features, hidden_size))
            fc_net.append(nn.ReLU())
            in_features = hidden_size
        
        # Add the final output layer
        fc_net.append(nn.Linear(in_features, 10)) # 10 classes for CIFAR-10
        
        self.fc = nn.Sequential(*fc_net)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        # Conv block 2
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        # Conv block 3 (no pooling after this one in this example, but let's add one for size reduction)
        x = self.pool(x) # Final pool to get to 4x4
        
        # Flatten and pass to fully connected layers
        x = x.view(-1, self.flattened_features)
        x = self.fc(x)
        return x

class MnistCNN(nn.Module):
    def __init__(self, hidden_dim: list[int]):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        
        # Flattened size after two pooling layers (32 * 7 * 7 for MNIST)
        flattened_dim = 32 * 7 * 7
        
        # Create fully connected layers based on hidden_dim
        layers = []
        dims = [flattened_dim] + hidden_dim + [10]  # Output layer has 10 classes
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Add activation for hidden layers
                layers.append(nn.ReLU())
        
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def count_trainable_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ImageNetCNN(nn.Module):
    def __init__(self, hidden_dim: list[int], classes=1000):
        super().__init__()
        
        # A simplified VGG-style convolutional base for 224x224 images
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
        )

        # Calculate flattened dimension after conv layers
        flattened_dim = 512 * 7 * 7

        # Create fully connected layers based on hidden_dim
        layers = []
        dims = [flattened_dim] + hidden_dim + [classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Add activation and dropout for hidden layers
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))
        
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def count_trainable_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)