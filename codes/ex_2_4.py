import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
partition = {'train': trainset, 'val':valset, 'test':testset}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

img = partition['train'].dataset.data[4]
x = transform(img)

conv = nn.Conv2d(3, 3, 3, stride=1, padding=1)
relu = nn.ReLU()
pooling = nn.MaxPool2d(2,2)
# add batch dim
x = x.unsqueeze(0)
x = relu(conv(x))
imshow(x.squeeze().detach().cpu())
x = pooling(relu(conv(x)))
imshow(x.squeeze().detach().cpu())
x = relu(conv(x))
imshow(x.squeeze().detach().cpu())