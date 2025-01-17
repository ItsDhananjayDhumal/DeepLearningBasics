!pip install opendatasets																												# opendatasets is a library used to download dataset from kaggle
import opendatasets as od
dataset_url = 'https://www.kaggle.com/competitions/induction-task/data'
od.download('https://www.kaggle.com/competitions/induction-task/data')


import os																																				# importing libraries
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose
import numpy as np


data_dir = "./induction-task/Data/Train/"																					# creating train and validation datasets
dataset = ImageFolder(data_dir, transform=Compose([ToTensor(),
																									 Resize(size = 512)
																									]))
random_seed = 2217828																															# set 50 images as validation images. these images will not be included in
torch.manual_seed(random_seed)																										# train dataset, so that model doesnt learn on these images.
val_size = 50
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

from torch.utils.data.dataloader import DataLoader																# making the dataloader
batch_size=20
train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
val_dl = DataLoader(val_ds, batch_size = batch_size*2, shuffle = False)

#defining the model
import torch.nn as nn
import torch.nn.functional as F																										# model has 10 convolution layers and 8 feedforward layers

leaky_param = 0.2

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.MaxPool2d(2, 2), # output: 64 x 192 x 192

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.MaxPool2d(2, 2), # output: 128 x 96 x 96

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.MaxPool2d(2, 2), # output: 256 x 48 x 48

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.MaxPool2d(2, 2), # output: 512 x 24 x 24

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_param),
            nn.MaxPool2d(2, 2), # output: 512 x 12 x 12

            nn.Flatten(),
            nn.Linear(512*12*12, 512*12),
            nn.LeakyReLU(leaky_param),
            nn.Linear(512*12, 512),
            nn.LeakyReLU(leaky_param),
            nn.Linear(512, 128),
            nn.LeakyReLU(leaky_param),
            nn.Linear(128, 64),
            nn.LeakyReLU(leaky_param),
            nn.Linear(64, 32),
            nn.LeakyReLU(leaky_param),
            nn.Linear(32, 16),
            nn.LeakyReLU(leaky_param),
            nn.Linear(16, 4),
            nn.LeakyReLU(leaky_param),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


model = ConvModel()

######################################## This code is to shift data from cpu to gpu. i copied it from google ###################################
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
model = model.to(device = device)
#######################################################################################################################3


def accuracy(outputs, labels):																							# defining accuracy function
    preds = torch.round(outputs)
    return torch.tensor(torch.sum(preds.float() == labels.float()).item() / len(preds))

def fit(epochs, lr, model, traindataloader, valdataloader, opt_func = torch.optim.SGD):					# training loop. i used binary cross entropy loss function.
    loss_history = []																																						# accuracy and losses and stored and returned when execution is completed
    acc_history = []																																						# there is bad coding involved because i wrote it and im not experienced with python
    optimizer = opt_func(model.parameters(), lr)																								# later i changed the optimizer function to Adam with betas 0.5 and 0.9
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accuracy = []

        for batch in traindataloader:
            images, labels = batch
            out = model(images)
            out = torch.flatten(out)
            loss = F.binary_cross_entropy(out.float(), labels.float())
            train_losses.append(loss.item())
            train_accuracy.append(accuracy(out, labels))


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        acc_history.append(np.mean(train_accuracy))
        loss_history.append(np.mean(train_losses))
        print("Train_Loss: ", np.round(np.mean(train_losses), 3), "  Train_Acc: ", np.round(np.mean(train_accuracy), 3))
        evaluate(model, valdataloader)

    return loss_history, acc_history

@torch.no_grad()																																	# evaluate function. it takes images from validation set. model is not trained here
def evaluate(model, valdataloader):
    model.eval()
    val_losses = []
    val_accuracy = []
    for batch in valdataloader:
        images, labels = batch
        out = model(images)
        out = torch.flatten(out)
        loss = F.binary_cross_entropy(out.float(), labels.float())
        acc = accuracy(out, labels)
        val_losses.append(loss.item())
        val_accuracy.append(acc.item())
    print("Val_Loss:   ", np.round(np.mean(val_losses), 3), "   Val_Acc:   ", np.round(np.mean(val_accuracy), 3))

######################################################## Copied from ChatGPT #############################################################
import gc																								# this code is to empty gpu ram. i frequently got out of memory error so i used this to make some space	
torch.cuda.empty_cache()
gc.collect()
#############################################################################################################################################


loss_history, acc_history= fit(epochs = 10, lr = 1e-4, model = model, traindataloader = train_dl, valdataloader = val_dl, opt_func = torch.optim.Adam)  # executing the training loop



