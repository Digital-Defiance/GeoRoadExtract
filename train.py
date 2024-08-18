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

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e31 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e41 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e51 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.upconv2 = nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(96, 48, kernel_size=3, padding=1)  # 48 + 48 channels
        self.d22 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 + 32 channels
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 16 + 16 channels
        self.d42 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        
        # Output layer
        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    


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

class CustomDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    
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