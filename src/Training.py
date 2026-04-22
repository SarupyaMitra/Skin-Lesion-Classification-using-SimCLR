import numpy as np
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt


from Model import SSL
from Loss import NXT_ent_loss
from Data import trainloader,batch_size
from Augment import SimCLRAugment

def save_checkpoint(epoch, model, optimizer, loss, path="../checkpoints/ssl_chkpt.pth"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path)

def load_checkpoint(model, optimizer, path="../checkpoints/ssl_chkpt.pth"):
    if not os.path.exists(path):
        print("No checkpoint found, starting from scratch.")
        return 0                        # start from epoch 0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1    # resume from next epoch
    print(f"Resumed from epoch {checkpoint['epoch']+1}, loss was {checkpoint['loss']:.6f}")
    return start_epoch

def training_loop(model,trainloader,optimizer,device,epochs=1000):
    losses = []
    model = model.to(device)
    model.train()
    augment = SimCLRAugment()
    start_epoch = load_checkpoint(model,optimizer)
    best_loss = float("inf")
    for epoch in range(start_epoch,epochs):
        epoch_loss = 0.0
        for _,(x,label) in enumerate(trainloader):
            
            xi = torch.stack([augment(img) for img in x])
            xj = torch.stack([augment(img) for img in x])

            x = torch.cat([xi,xj],dim=0).to(device)

            z = model(x)

            loss = NXT_ent_loss(z,batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss/len(trainloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} Avg Loss: {avg_loss:.6f}")
        if avg_loss<best_loss:
            best_loss = avg_loss
            save_checkpoint(epoch,model,optimizer,avg_loss)
            print(f"New best saved at epoch = {epoch+1}")
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.savefig("../Training_Loss.png")
    

if __name__ == "__main__":
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")
    model = SSL()
    optm = optim.AdamW(model.parameters() , lr  = 3e-4,weight_decay=1e-4)
    training_loop(model,trainloader,optm,device)