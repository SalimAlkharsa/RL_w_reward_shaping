from collections import deque
import torch


# Define the device to be used
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use MPS if available
    DEVICE = torch.device("cpu") # Default to CPU even w MPS due to time constraints
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA if available
else:
    DEVICE = torch.device("cpu")  # Default to CPU if no GPU is available

class LastVisitedElements:
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.elements = deque(maxlen=max_size)

    def add_element(self, element):
        self.elements.append(element)

    def get_elements(self):
        return set(self.elements)
