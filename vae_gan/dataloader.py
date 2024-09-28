import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dataloader(batch_size):
    dataroot = r"D:\brain_dataset\Training"  # Replace this with the path to your local dataset
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
        transforms.Resize(64),  # Resize images to 64x64
        transforms.CenterCrop(64),  # Crop images to 64x64
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
    ])

    # Load dataset using ImageFolder
    dataset = ImageFolder(root=dataroot, transform=transform)

    # Create DataLoader for your dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

