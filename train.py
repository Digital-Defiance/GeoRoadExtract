import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.optim as optim
import os
import torch.nn.functional as F
from glob import glob

import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from model import UNet
from customDataset import CustomDataset

def get_image_mask_pairs(dir):
    sat_images = [f for f in os.listdir(dir) if f.endswith('_sat.jpg')]
    pairs = [(os.path.join(dir, f), os.path.join(dir, f.replace('_sat.jpg', '_mask.png'))) for f in sat_images]
    return pairs

all_pairs = get_image_mask_pairs('./dataset/train')

train_pairs, temp_pairs = train_test_split(all_pairs, test_size=0.25, random_state=42)
valid_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.4, random_state=42)

print(f"Total samples: {len(all_pairs)}")
print(f"Train samples: {len(train_pairs)} ({len(train_pairs)/len(all_pairs):.2%})")
print(f"Validation samples: {len(valid_pairs)} ({len(valid_pairs)/len(all_pairs):.2%})")
print(f"Test samples: {len(test_pairs)} ({len(test_pairs)/len(all_pairs):.2%})")

    # Define your transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = CustomDataset(train_pairs, transform=transform)
valid_dataset = CustomDataset(valid_pairs, transform=transform)
test_dataset = CustomDataset(test_pairs, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = UNet(n_class=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            valid_loss += loss.item() * images.size(0)
            
    valid_loss = valid_loss / len(valid_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs} | Train_loss: {train_loss:.4f} | Validation loss: {valid_loss:.4f}')

