import torch.nn as nn


class SignDigitsCNN(nn.Module):
    """
    CNN model used for sign digits image classification
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ##
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ##
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ##
            nn.Flatten(),
            nn.Linear(16384, 8192),
            nn.Linear(8192, 1024),
            nn.Linear(1024, 10),
        )

    def forward(self, xs):
        """
        Forward pass through the network
        """

        return self.network(xs)
