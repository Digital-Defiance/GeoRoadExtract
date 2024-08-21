import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.e11_bn = nn.BatchNorm2d(16)
        self.e12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.e12_bn = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.2)  # 20% dropout on input layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.e21_bn = nn.BatchNorm2d(32)
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.e22_bn = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(p=0.5)  # 50% dropout on hidden layers
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e31 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.e31_bn = nn.BatchNorm2d(48)
        self.e32 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.e32_bn = nn.BatchNorm2d(48)
        self.dropout3 = nn.Dropout(p=0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e41 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.e41_bn = nn.BatchNorm2d(64)
        self.e42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.e42_bn = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p=0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.e51 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e51_bn = nn.BatchNorm2d(128)
        self.e52 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.e52_bn = nn.BatchNorm2d(128)
        self.dropout5 = nn.Dropout(p=0.5)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d11_bn = nn.BatchNorm2d(64)
        self.dropout6 = nn.Dropout(p=0.5)
        self.d12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.d12_bn = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.upconv2 = nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(96, 48, kernel_size=3, padding=1)  # 48 + 48 channels
        self.d21_bn = nn.BatchNorm2d(48)
        self.dropout8 = nn.Dropout(p=0.5)
        self.d22 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.d22_bn = nn.BatchNorm2d(48)
        self.dropout9 = nn.Dropout(p=0.5)
        
        self.upconv3 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 + 32 channels
        self.d31_bn = nn.BatchNorm2d(32)
        self.dropout10 = nn.Dropout(p=0.5)
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.d32_bn = nn.BatchNorm2d(32)
        self.dropout11 = nn.Dropout(p=0.5)
        
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(32, 16, kernel_size=3, padding=1)  # 16 + 16 channels
        self.d41_bn = nn.BatchNorm2d(16)
        self.dropout12 = nn.Dropout(p=0.5)
        self.d42 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.d42_bn = nn.BatchNorm2d(16)
        self.dropout13 = nn.Dropout(p=0.5)
        
        # Output layer
        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.e11_bn(self.e11(x)))
        e1 = F.relu(self.e12_bn(self.e12(e1)))
        e1 = self.dropout1(e1)  
        p1 = self.pool1(e1)
        
        e2 = F.relu(self.e21_bn(self.e21(p1)))
        e2 = F.relu(self.e22_bn(self.e22(e2)))
        e2 = self.dropout2(e2) 
        p2 = self.pool2(e2)
        
        e3 = F.relu(self.e31_bn(self.e31(p2)))
        e3 = F.relu(self.e32_bn(self.e32(e3)))
        e3 = self.dropout3(e3) 
        p3 = self.pool3(e3)
        
        e4 = F.relu(self.e41_bn(self.e41(p3)))
        e4 = F.relu(self.e42_bn(self.e42(e4)))
        e4 = self.dropout4(e4)  
        p4 = self.pool4(e4)
        
        e5 = F.relu(self.e51_bn(self.e51(p4)))
        e5 = F.relu(self.e52_bn(self.e52(e5)))
        e5 = self.dropout5(e5)  

        # Decoder
        d1 = self.upconv1(e5)
        d1 = torch.cat((d1, e4), dim=1) 
        d1 = F.relu(self.d11_bn(self.d11(d1)))
        d1 = self.dropout6(d1)  
        d1 = F.relu(self.d12_bn(self.d12(d1)))
        d1 = self.dropout7(d1)
        
        d2 = self.upconv2(d1)
        d2 = torch.cat((d2, e3), dim=1) 
        d2 = F.relu(self.d21_bn(self.d21(d2)))
        d2 = self.dropout8(d2)  
        d2 = F.relu(self.d22_bn(self.d22(d2)))
        d2 = self.dropout9(d2)  
        d3 = self.upconv3(d2)
        d3 = torch.cat((d3, e2), dim=1) 
        d3 = F.relu(self.d31_bn(self.d31(d3)))
        d3 = self.dropout10(d3) 
        d3 = F.relu(self.d32_bn(self.d32(d3)))
        d3 = self.dropout11(d3)  
        
        d4 = self.upconv4(d3)
        d4 = torch.cat((d4, e1), dim=1) 
        d4 = F.relu(self.d41_bn(self.d41(d4)))
        d4 = self.dropout12(d4)  
        d4 = F.relu(self.d42_bn(self.d42(d4)))
        d4 = self.dropout13(d4)  
        
   
        out = self.outconv(d4)
        
        return out