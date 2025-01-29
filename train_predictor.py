"""
Christopher Ondrusz
GitHub: acse_cro23
"""
from fireDiff.Models import PredictionModel
from fireDiff.Models import PredictorUNet
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

num_epochs = 100  # Recommended minimum number of epochs for training


train_dataset = VideoFramePairsDataset("../train_copy", transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = VideoFramePairsDataset("../val", transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

unet = PredictorUNet()
model = PredictionModel(unet, device=device)


model.train_model(train_dataloader, val_dataloader, epochs=1)

model.save_model(f"./predictor_{num_epochs}.pt")
