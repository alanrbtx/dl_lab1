import torch

class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, kernel_size=11, stride=4),
            torch.nn.MaxPool2d(stride=2, kernel_size=3),
            torch.nn.Conv2d(96, 256, 5, stride=1, padding=2),
            torch.nn.MaxPool2d(stride=2, kernel_size=3),
            torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(stride=2, kernel_size=3)
        )
        
        self.flatten = torch.nn.Flatten()
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.Linear(4096, 4096),
            torch.nn.Linear(4096, 5)
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.classification_head(x)
        return x
    
AlexNet()(torch.randn(size=(1, 3, 227, 227))).shape