import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import glob
import cv2
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o
import tqdm
import torch.utils.data as tud
from utils import Dataset
import numpy as np

def process_data():
    spectrogram_file = "spectrogram"
    wavelets_file = "wavelets"
    spectrogram_train = []
    wavelets_train = []
    spectrogram_test = []
    wavelets_test = []

    spectrogram_train_file = os.path.join(spectrogram_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_train.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1
    spectrogram_test_file = os.path.join(spectrogram_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_test.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1

    wavelets_train_file = os.path.join(wavelets_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_train.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1

    wavelets_test_file = os.path.join(wavelets_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_test.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1

    train = list(zip(wavelets_train, spectrogram_train))
    test = list(zip(wavelets_test, spectrogram_test))
    # train is a list of tuples, each tuple has two baby tuples, each baby tuple has (image (wavelets/spectrogram), class_idx)
    return train, test


def train_model(model, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        # data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)

def train_main():
    train, test = process_data()
    train = tud.Dataset()
    test = t.tensor(test)

    net = MusicGenreNet() # .cuda()

    epochs = 1
    lr = 0.01
    momentum = 0.65
    decay = 0
    optim = o.SGD(net.parameters(), lr=lr, momentum=momentum)
    train_loader = Dataset(train, batch_size=128)
    train_losses = []

    for epoch in range(epochs):
        train_loss = train_model(net, train_loader, optim, epoch, 1)
        train_losses.append(train_loss)
        # collect test loss too



TEMPERATURE = 0.35


class MusicGenreNet(nn.Module):
    def __init__(self):
        super(MusicGenreNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, (3, 3), stride=(2, 2))
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.linear(x)
        x = F.softmax(x)
        return x

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val

# process_data()