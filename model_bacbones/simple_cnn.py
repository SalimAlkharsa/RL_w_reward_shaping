import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(SimpleCNN, self).__init__()
        
        # Extract properties of the input shape
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Extract the number of channels from the input shape
        n_channels = input_shape[0]
        # Extract the height and width from the input shape
        height, width = input_shape[1], input_shape[2]

        # Define a 3 layer CNN with adjusted kernel sizes and strides
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, stride=2)
        conv1_output_shape = self.get_conv_output_shape(input_shape, self.conv1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2)
        conv2_output_shape = self.get_conv_output_shape(conv1_output_shape, self.conv2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=2)
        conv3_output_shape = self.get_conv_output_shape(conv2_output_shape, self.conv3)
        
        # Calculate the size of the flattened feature map
        n_flatten = conv3_output_shape[0] * conv3_output_shape[1] * conv3_output_shape[2]
        
        # Define the fully connected layer
        self.fc1 = nn.Linear(n_flatten, 512)
        self.fc2 = nn.Linear(512, output_shape)
        
    def forward(self, x):
        print('input shape:', x.shape)
        x = torch.relu(self.conv1(x))
        print('conv1 shape:', x.shape)
        x = torch.relu(self.conv2(x))
        print('conv2 shape:', x.shape)
        x = torch.relu(self.conv3(x))
        print('conv3 shape:', x.shape)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print('flatten shape:', x.shape)
        
        x = torch.relu(self.fc1(x))
        print('fc1 shape:', x.shape)

        x = self.fc2(x)
        print('fc2 shape:', x.shape)

        return x
        
    @staticmethod
    def get_conv_output_shape(input_shape, conv_layer):
        # Extract the number of channels from the input shape
        n_channels = input_shape[0]
        # Extract the height and width from the input shape
        height, width = input_shape[1], input_shape[2]
        
        # Get the output shape of the convolutional layer
        output_channels, output_height, output_width = conv_layer(
            torch.zeros(1, n_channels, height, width)
        ).shape[1:]
        
        return output_channels, output_height, output_width
