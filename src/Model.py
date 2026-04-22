import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,stride=1)
        self.enc_bn1 = nn.BatchNorm2d(64)
        
##########################################################################################################################
        self.enc_l1_conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.enc_l1_bn1 = nn.BatchNorm2d(64)
        self.enc_l1_conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.enc_l1_bn2 = nn.BatchNorm2d(64)

        self.enc_l1_conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.enc_l1_bn3 = nn.BatchNorm2d(64)
        self.enc_l1_conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.enc_l1_bn4 = nn.BatchNorm2d(64)

##########################################################################################################################
        self.enc_l2_conv1 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.enc_l2_bn1 = nn.BatchNorm2d(128)
        self.enc_l2_conv2 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.enc_l2_bn2 = nn.BatchNorm2d(128)
        self.enc_l2_skip = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=2,padding=0)

        self.enc_l2_conv3 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.enc_l2_bn3 = nn.BatchNorm2d(128)
        self.enc_l2_conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.enc_l2_bn4 = nn.BatchNorm2d(128)

##########################################################################################################################
        self.enc_l3_conv1 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)
        self.enc_l3_bn1 = nn.BatchNorm2d(256)
        self.enc_l3_conv2 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.enc_l3_bn2 = nn.BatchNorm2d(256)
        self.enc_l3_skip = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=2,padding=0)

        self.enc_l3_conv3 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.enc_l3_bn3 = nn.BatchNorm2d(256)
        self.enc_l3_conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.enc_l3_bn4 = nn.BatchNorm2d(256)

##########################################################################################################################
        self.enc_l4_conv1 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1)
        self.enc_l4_bn1 = nn.BatchNorm2d(512)
        self.enc_l4_conv2 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.enc_l4_bn2 = nn.BatchNorm2d(512)
        self.enc_l4_skip = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=2,padding=0)

        self.enc_l4_conv3 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.enc_l4_bn3 = nn.BatchNorm2d(512)
        self.enc_l4_conv4 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.enc_l4_bn4 = nn.BatchNorm2d(512)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))  # op dimension = 64,28,28
        
        # Layer 1
             # Block 1
        y = F.relu(self.enc_l1_bn1(self.enc_l1_conv1(x)))  # 64,28,28
        y = self.enc_l1_bn2(self.enc_l1_conv2(y)) # 64,28,28
        x = F.relu(y+x)  # 64,28,28
            # Block 2
        y = F.relu(self.enc_l1_bn3(self.enc_l1_conv3(x)))  # 64,28,28
        y = self.enc_l1_bn4(self.enc_l1_conv4(y)) # 64,28,28
        x = F.relu(y+x)  # 64,28,28
        
        # Layer 2
            # Block 1
        y = F.relu(self.enc_l2_bn1(self.enc_l2_conv1(x)))  # 128,14,14
        y = self.enc_l2_bn2(self.enc_l2_conv2(y))  # 128,14,14
        x = F.relu(y + self.enc_l2_skip(x))  # 128,14,14

            # Block 2
        y = F.relu(self.enc_l2_bn3(self.enc_l2_conv3(x)))  # 128,14,14
        y = self.enc_l2_bn4(self.enc_l2_conv4(y))  # 128,14,14
        x = F.relu(y + x)  # 128,14,14

        # Layer 3
            # Block 1
        y = F.relu(self.enc_l3_bn1(self.enc_l3_conv1(x)))  # 256,7,7
        y = self.enc_l3_bn2(self.enc_l3_conv2(y))  # 256,7,7
        x = F.relu(y + self.enc_l3_skip(x))  # 256,7,7

            # Block 2
        y = F.relu(self.enc_l3_bn3(self.enc_l3_conv3(x)))  # 256,7,7
        y = self.enc_l3_bn4(self.enc_l3_conv4(y))  # 256,7,7
        x = F.relu(y + x)  # 256,7,7    

        # Layer 4
            # Block 1
        y = F.relu(self.enc_l4_bn1(self.enc_l4_conv1(x)))  # 512,4,4
        y = self.enc_l4_bn2(self.enc_l4_conv2(y))  # 512,4,4
        x = F.relu(y + self.enc_l4_skip(x))  # 512,4,4

            # Block 2
        y = F.relu(self.enc_l4_bn3(self.enc_l4_conv3(x)))  # 512,4,4
        y = self.enc_l4_bn4(self.enc_l4_conv4(y))  # 512,4,4
        x = F.relu(y + x)  # 512,4,4 

        x = self.pool(x)   # 512,1,1

        x = torch.flatten(x,1) # 512
        return x



class SSL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder() 
        self.proj_Linear1 = nn.Linear(in_features=512,out_features=512)
        self.proj_bn = nn.BatchNorm1d(512)
        self.proj_Linear2 = nn.Linear(in_features=512,out_features=128)

    def projection(self,x):
        x = F.relu(self.proj_bn(self.proj_Linear1(x)))
        x = self.proj_Linear2(x)
        return x
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.projection(x)
        z = F.normalize(x,dim=1)
        return z
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSL().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)

    x = torch.randn((64,3,28,28))
    encoder = model.encoder
    h = encoder(x)
    print(h.shape)