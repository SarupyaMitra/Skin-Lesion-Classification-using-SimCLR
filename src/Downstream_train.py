import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt


from Model import Encoder,SSL
from Loss import NXT_ent_loss
from Data import trainloader,batch_size,testloader
from Augment import SimCLRAugment
from Downstream_model import LinearClassifier


def save_checkpoint(epoch, model, optimizer, loss, path="../checkpoints/down_chkpt.pth"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path)

def load_checkpoint(model, optimizer, path="../checkpoints/down_chkpt.pth"):
    if not os.path.exists(path):
        print("No checkpoint found, starting from scratch.")
        return 0                        # start from epoch 0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1    # resume from next epoch
    print(f"Resumed from epoch {checkpoint['epoch']+1}, loss was {checkpoint['loss']:.6f}")
    return start_epoch

def downstream_training(model, trainloader, optimizer, device, criterion,epochs=100):
    losses = []
    model.train()
    start_epoch = load_checkpoint(model,optimizer)
    best_loss = float("inf")
    for epoch in range(start_epoch,epochs):
        
        running_loss  = 0.0
        for _, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()
            
            
            pred = model(x)
            loss = criterion(pred, label.squeeze())

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)  
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.6f}")
        if avg_loss<best_loss:
            best_loss = avg_loss
            save_checkpoint(epoch,model,optimizer,avg_loss)
            print(f"New best saved at epoch {epoch+1}")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Downstream Training Loss")
    plt.savefig("../Downstream_Loss.png")

    return model


if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")
    main_checkpoint = torch.load("../checkpoints/ssl_chkpt.pth")
    main_model = SSL()
    main_model.load_state_dict(main_checkpoint["model_state_dict"])

    encoder = main_model.encoder    # Extracting the pre-trained encoder
    model = LinearClassifier(encoder, num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    # num_classes = 7
    # counts = torch.zeros(num_classes)

    # for _, labels in trainloader:
    #     for l in labels:
    #         counts[l] += 1

    # N = counts.sum()
    # class_weights = N / (num_classes * counts)
    # class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss()
    downstream_training(model,trainloader,optimizer,device,criterion)