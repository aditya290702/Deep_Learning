import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


#for downloading the test data and the train data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#initialising the batch size
batch_size = 64

#creating data loaders
Data_loader_train = DataLoader(training_data,batch_size)
Data_loader_test = DataLoader(test_data,batch_size)

for x,y in Data_loader_test:
    print(f"Shape of X [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



#Building models
#selecting the device
#(CPU or GPU)

device = ( "cuda" if torch.cuda.is_available() else
           "cpu")
print()
print(f"Using {device} device")


#specifying the model architecture
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,10)
    ).to(device)

print(model)

#Generating random values
x = torch.randn(1,28,28).to(device)

#feeding the values in the architecture
logits = model(x)
print(logits)
