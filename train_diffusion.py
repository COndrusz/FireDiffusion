"""
Christopher Ondrusz
GitHub: acse_cro23
"""
from fireDiff.Models import UNet
from fireDiff.Models import DiffusionModel
from fireDiff.Datasets import VideoFramePairsDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

print("imports done")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

num_epochs = 200  # Recommended minimum number of epochs for training


dataset = VideoFramePairsDataset("../train", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print("dataset loaded")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

unet = UNet()
model = DiffusionModel(unet, device=device)
print("model loaded")

model.train_model(dataloader, epochs=num_epochs)
print("model trained")
model.save_model(f"./Diffusion_{num_epochs}")
print("model saved")
