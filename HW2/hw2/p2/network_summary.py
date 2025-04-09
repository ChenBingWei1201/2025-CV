import torch
import sys
from torchsummary import summary

# Add the P2 directory to the path so we can import from it
sys.path.append('/home/joe/code/2025-CV/HW2/hw2/p2')

# Import the models
from model import MyNet, ResNet18

# Function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mynet = MyNet().to(device)
resnet18 = ResNet18().to(device)

# Count parameters
mynet_params = count_parameters(mynet)
resnet18_params = count_parameters(resnet18)

# Print parameter counts
print(f"MyNet parameters: {mynet_params:,}")
print(f"ResNet18 parameters: {resnet18_params:,}")

# Print model architectures using summary
print("\nMyNet Architecture:")
summary(mynet, (3, 32, 32))

print("\nResNet18 Architecture:")
summary(resnet18, (3, 32, 32))