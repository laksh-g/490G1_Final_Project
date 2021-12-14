import librosa
import librosa.display
import os
import glob
import cv2
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o
import torch.utils.data
import tqdm
import torch.utils.data as tud
from torch.utils.data import TensorDataset
import torchvision.io
import torchvision.transforms

from utils import Dataset
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

IMG_SIZE = 256

def process_data():
    spectrogram_file = "spectrogram"
    wavelets_file = "wavelets"
    spectrogram_train = []
    wavelets_train = []
    spectrogram_test = []
    wavelets_test = []

    print("Started Procesing Data")

    spectrogram_train_file = os.path.join(spectrogram_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            # print(type(torch.from_numpy(cv2.resize(cv2.imread(image), (IMG_SIZE, IMG_SIZE)))))
            spectrogram_train.append((torchvision.io.read_image(image), class_idx)) # fix image read
        class_idx+=1
    spectrogram_test_file = os.path.join(spectrogram_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_test.append((torchvision.io.read_image(image), class_idx)) # fix image read
        class_idx+=1

    wavelets_train_file = os.path.join(wavelets_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_train.append((torchvision.io.read_image(image), class_idx))# fix image read
        class_idx+=1

    wavelets_test_file = os.path.join(wavelets_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_test.append((torchvision.io.read_image(image), class_idx)) # fix image read
        class_idx+=1

    train = spectrogram_train
    tr_x = [torch.tensor(item[0]) for item in train]
    tr_y = torch.tensor([int(item[1]) for item in train])
    b = torch.Tensor((len(tr_y), 256, 256, 3))
    train_x = torch.stack(tr_x, out=b)
    my_dataset = TensorDataset(train_x, tr_y)
    dataloader_train = torch.utils.data.DataLoader(my_dataset)

    test = spectrogram_test
    ts_x = [torch.tensor(item[0]) for item in test]
    ts_y = torch.tensor([int(item[1]) for item in test])
    b = torch.Tensor((len(ts_y), 256, 256, 3))
    test_x = torch.stack(ts_x, out=b)
    my_dataset = TensorDataset(test_x, ts_y)
    dataloader_test = torch.utils.data.DataLoader(my_dataset)
    # train = list(zip(wavelets_train, spectrogram_train))
    # test = list(zip(wavelets_test, spectrogram_test))
    # train is a list of tuples, each tuple has two baby tuples, each baby tuple has (image (wavelets/spectrogram), class_idx)
    # [((), ()),((),()),...]
    print("Finished Loading data")
    return dataloader_train, dataloader_test


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
    # Play around with these constants, you may find a better setting.
    # BATCH_SIZE = 256
    # TEST_BATCH_SIZE = 10
    # EPOCHS = 20
    # LEARNING_RATE = 0.01
    # MOMENTUM = 0.65
    USE_CUDA = False
    # SEED = 0
    # PRINT_INTERVAL = 100
    # WEIGHT_DECAY = 0.000

    # Now the actual training code
    use_cuda = USE_CUDA and torch.cuda.is_available()

    # torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': 6,
              'pin_memory': False} if use_cuda else {}



    train, test = process_data()
    net = MusicGenreNet().to(device)

    epochs = 1
    lr = 0.01
    momentum = 0.65
    decay = 0
    BATCH_SIZE = 2
    optim = o.SGD(net.parameters(), lr=lr, momentum=momentum)
    train_loader = train
    train_losses = []

    for epoch in range(epochs):
        print(f'Starting Epoch : {epoch}')
        train_loss = train_model(net, train_loader, optim, epoch, 1)
        train_losses.append(train_loss)
        print(f"Epoch {epoch} completed: training loss was {train_loss}")

        # collect test loss too

    print(f'train losses = {train_losses}')

TEMPERATURE = 0.35


class MusicGenreNet(nn.Module):
    def __init__(self):
        super(MusicGenreNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 64, (3, 3))
        self.linear = nn.Linear(65536, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.softmax(x)
        return x

    def loss(self, prediction, label, reduction='mean'):
        loss_val = F.cross_entropy(prediction.view(-1, self.vocab_size), label.view(-1), reduction=reduction)
        return loss_val


if __name__ == '__main__':
    #freeze_support()
    train_main()