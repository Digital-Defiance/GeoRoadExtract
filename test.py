import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet 
from customDataset import CustomDataset  
import numpy as np


model_path = 'best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(n_class=1) 
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def predict_single_image(img_path, model, transform, device, threshold = 0.1):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(output, cmap='gray') 
    
    binary_mask = np.where(output > threshold, 255, 0).astype(np.uint8)
    return binary_mask

test_image_path = 'random.jpg' 
predicted_mask = predict_single_image(test_image_path, model, transform, device)

# Visualize the result
original_image = Image.open(test_image_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()

