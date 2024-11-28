import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SimpleCNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Create a dummy input to get the flattened size after conv layers
        self.dummy_input = torch.zeros(1, *input_shape)
        self._get_flattened_size()  # Calculate flattened size dynamically

        # Fully connected layers
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, output_shape)

    def _get_flattened_size(self):
        # Pass the dummy input through the conv layers to calculate the flattened size
        x = self.dummy_input
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the output of the conv layers
        x = x.view(x.size(0), -1)  # Flatten
        self.flattened_size = x.size(1)  # Get the size after flattening

        # Print the flattened size for debugging purposes
        print(f"Flattened size after conv layers: {self.flattened_size}")

    def forward(self, x):
        # Forward pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        print(x.shape)  # After flattening

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x