import os
import glob
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as o
import torch.utils.data
from torch.utils.data import TensorDataset
import torchvision.io
import torchvision.transforms
import pt_util
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import torchvision.transforms as T

IMG_SIZE = 256

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
img_size = 256

def process_data(batch_size):
    spectrogram_file = "spectrogram"
    wavelets_file = "wavelets"
    spectrogram_train = []
    wavelets_train = []
    spectrogram_test = []
    wavelets_test = []

    transform = T.Compose([
        T.Resize((288, 432)),
    ])

    print("Started Procesing Data")

    spectrogram_train_file = os.path.join(spectrogram_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_train.append((transform(torchvision.io.read_image(image)), class_idx)) # fix image read
        class_idx+=1
    spectrogram_test_file = os.path.join(spectrogram_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_test.append((transform(torchvision.io.read_image(image)), class_idx)) # fix image read
        class_idx+=1

    wavelets_train_file = os.path.join(wavelets_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_train.append((transform(torchvision.io.read_image(image)), class_idx))# fix image read
        class_idx+=1

    wavelets_test_file = os.path.join(wavelets_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_test.append((transform(torchvision.io.read_image(image)), class_idx)) # fix image read
        class_idx+=1

    train = spectrogram_train
    tr_x = [torch.tensor(item[0]) for item in train]
    tr_y = torch.tensor([int(item[1]) for item in train])
    print(f"try_y shape: {tr_y.shape}")
    b = torch.Tensor((len(tr_y), 3, 288, 432))
    train_x = torch.stack(tr_x, out=b)
    my_dataset = TensorDataset(train_x, tr_y)
    dataloader_train = torch.utils.data.DataLoader(my_dataset)
    for (x, y) in dataloader_train:
        print(f"Sample data from data_loader: {type(x)}, {y}")
        print(x.shape)
        break


    test = spectrogram_test
    ts_x = [torch.tensor(item[0]) for item in test]
    ts_y = torch.tensor([int(item[1]) for item in test])
    b = torch.Tensor((len(ts_y), 3, 288, 432))
    test_x = torch.stack(ts_x, out=b)
    my_dataset = TensorDataset(test_x, ts_y)
    dataloader_test = torch.utils.data.DataLoader(my_dataset)
    # train is a list of tuples, each tuple has two baby tuples, each baby tuple has (image (wavelets/spectrogram), class_idx)
    # [((), ()),((),()),...]
    print("Finished Loading data")
    return dataloader_train, dataloader_test

def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = torch.tensor(label)
            label = label.to(device)
            output = model(data)
            output = output.reshape(10)
            output = output.type(torch.FloatTensor)
            output = torch.reshape(output, (1, 10))
            loss = F.cross_entropy(output, label.view(-1))
            test_loss += loss
            pred = output.max(1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def train_model(model, train_loader, optimizer, epoch, log_interval, device):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        data = data.float()
        label = torch.tensor(label)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.reshape(10)
        output = output.type(torch.FloatTensor)
        output = torch.reshape(output, (1, 10))
        loss = F.cross_entropy(output, label.view(-1))
        loss.backward()
        optimizer.step()
        a = loss.item()
        losses.append(a)
    return np.mean(losses)


def train_main(version):
    USE_CUDA = False
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())
    kwargs = {'num_workers': 6,
              'pin_memory': False} if use_cuda else {}

    BATCH_SIZE = 64
    epochs = 500
    lr = 2e-5
    momentum = 0.65
    decay = 0
    train_loader, test_loader = process_data(BATCH_SIZE)
    net = MusicGenreNet().to(device)
    # optimizer = o.SGD(net.parameters(), lr=lr)
    optimizer = o.Adam(net.parameters(), lr=lr)
    train_losses = []

    for epoch in range(epochs):
        train_loss = train_model(net, train_loader, optimizer, epoch, 64, device)
        train_losses.append(train_loss)
        print(f"Epoch {epoch} completed: training loss = {train_loss}")

    test_loss, test_accuracy = test_model(net, test_loader, device)
    models_path = os.path.join(os.getcwd(), "saved_models", f"model_{version}")
    net.save_model(models_path)
    print(f'Final train loss = {train_losses[-1]}')
    print(f'Final test accuracy: {test_accuracy}')
    return train_losses


TEMPERATURE = 0.35


class MusicGenreNet(nn.Module):
    def __init__(self):
        super(MusicGenreNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, (3, 3), (2, 2))
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 16, (4, 4), (2, 2))
        self.maxpool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(16, 24, (5, 5), (2, 2))
        self.maxpool3 = nn.MaxPool2d(3, 2)
        self.linear1 = nn.Linear(240, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        # x = F.softmax(x, dim=1)
        return x

    def loss(self, prediction, label, reduction='mean'):
        return F.mse_loss(prediction, label.type(torch.FloatTensor))

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def classify_single_input(self, model, data, device):
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            output = model(data.to(torch.float32))
            output = output.reshape(10)
            output = torch.reshape(output, (1, 10))
            pred = output.max(1)[1]

        return pred, labels[pred]

if __name__ == '__main__':
    version = "1.7"
    train_losses = train_main(version)
    plt.xlabel = "Epochs"
    plt.ylabel = "Cross Entropy Loss"
    plot_path = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_path = os.path.join(plot_path, f"training_loss_{version}.PNG")
    plt.savefig(plot_path)
