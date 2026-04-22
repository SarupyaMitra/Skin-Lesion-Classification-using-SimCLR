import numpy
from medmnist import DermaMNIST
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

my_transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = DermaMNIST(split = 'train',transform = my_transforms, download=True)
test_dataset = DermaMNIST(split = 'test',transform = my_transforms, download=True)
val_dataset = DermaMNIST(split='val',transform = my_transforms,download = True)
#print(train_dataset)

batch_size = 64
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=True)

print(len(trainloader))

def visualise():
    # imgs , labels = next(iter(trainloader))
    # print(imgs.shape,labels.shape)

    num_images = 8
    data_iter = iter(testloader)
    x,_=next(data_iter)    # It returns a batch of images. So x will have batch_size no of images.E.g: x.shape = torch.size([32,3,28,28])
    for i in range(num_images):  # Out of those images we are only printing 8 of them
        plt.subplot(1, num_images, i + 1)
        img = x[i].reshape(28,28,3)
        plt.imshow(img)
        plt.axis('off')
        #print(x[i].shape)
    plt.show()

if __name__=="__main__":
    visualise()