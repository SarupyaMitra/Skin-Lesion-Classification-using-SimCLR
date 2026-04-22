import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from Model import Encoder,SSL

class LinearClassifier(nn.Module):
    def __init__(self,encoder,num_classes=7):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(512,num_classes)
        # freeze the encoder layer
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
    
    def forward(self,x):
        # with torch.no_grad():
        h = self.encoder(x)
        return self.fc(h)
    
if __name__ == "__main__":
    # Load checkpoint
    checkpoint = torch.load("../checkpoints/ssl_chkpt.pth")
    ssl_model = SSL()
    model_state = checkpoint["model_state_dict"]
    ssl_model.load_state_dict(model_state)

# Extract ONLY the encoder — discard the projection head
    encoder = ssl_model.encoder   # outputs h of shape (512,)
    x = torch.randn((64,3,28,28))
    model = LinearClassifier(encoder)
    h = model(x)

    print(h.shape)