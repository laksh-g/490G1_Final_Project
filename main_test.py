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
import pt_util
from utils import Dataset
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

IMG_SIZE = 256

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
img_size = 256
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                resized_arr = resized_arr.reshape((3, 256, 256))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

def tf_process_data():
    train = get_data('spectrogram/train')
    val = get_data('spectrogram/test')

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    my_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    dataloader_train = torch.utils.data.DataLoader(my_dataset)

    my_dataset_test = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))
    dataloader_test = torch.utils.data.DataLoader(my_dataset_test)

    return dataloader_train, dataloader_test



def process_data(batch_size):
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
            # print(f"data while processing: {torchvision.io.read_image(image).shape}")
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
    # print(tr_x[0].shape)
    tr_y = torch.tensor([int(item[1]) for item in train])
    print(f"try_y shape: {tr_y.shape}")
    b = torch.Tensor((len(tr_y), 3, 288, 432))
    train_x = torch.stack(tr_x, out=b)
    my_dataset = TensorDataset(train_x, tr_y)
    dataloader_train = torch.utils.data.DataLoader(my_dataset)
    for (x, y) in dataloader_train:
        print(f"Sample data from data_loader: {x.shape}, {y}")
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
    # for param in model.parameters():
    #     print(param.data)
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        label = torch.tensor(label)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.reshape(10)
        output = output.type(torch.FloatTensor)
        output = torch.reshape(output, (1, 10))
        # print(output.shape)
        # print(label.view(-1).shape)
        loss = F.cross_entropy(output, label.view(-1))
        loss.backward()
        optimizer.step()
        # print(model.conv2.weight.grad)
        a = loss.item()
        losses.append(a)
    return np.mean(losses)


def train_main():
    USE_CUDA = False
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)
    print('num cpus:', multiprocessing.cpu_count())
    kwargs = {'num_workers': 6,
              'pin_memory': False} if use_cuda else {}

    BATCH_SIZE = 64
    epochs = 50
    lr = 5e-5
    momentum = 0.65
    decay = 0
    train_loader, test_loader = process_data(BATCH_SIZE)
    # train, test = tf_process_data()
    net = MusicGenreNet().to(device)
    # optimizer = o.SGD(net.parameters(), lr=lr)
    optimizer = o.Adam(net.parameters(), lr=lr, momentum=momentum)
    train_losses = []

    for epoch in range(epochs):
        train_loss = train_model(net, train_loader, optimizer, epoch, 64, device)
        train_losses.append(train_loss)
        print(f"Epoch {epoch} completed: training loss = {train_loss}")

    test_loss, test_accuracy = test_model(net, test_loader, device)
    print(f'Final train loss = {train_losses[-1]}')
    print(f'Final test accuracy: {test_accuracy}')


TEMPERATURE = 0.35


class MusicGenreNet(nn.Module):
    def __init__(self):
        super(MusicGenreNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 8, (3, 3), (2, 2))
        # self.maxpool1 = nn.MaxPool2d(3, 2)
        # self.conv2 = nn.Conv2d(8, 12, (4, 4), (2, 2))
        # self.maxpool2 = nn.MaxPool2d(3, 2)
        # self.conv3 = nn.Conv2d(12, 16, (5, 5), (2, 2))
        # self.maxpool3 = nn.MaxPool2d(3, 2)
        # self.linear1 = nn.Linear(160, 64)
        # self.linear2 = nn.Linear(64, 10)

        self.conv1 = nn.Conv2d(4, 8, (3, 3), (2, 2))
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 12, (4, 4), (2, 2))
        self.maxpool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(12, 16, (5, 5), (2, 2))
        self.maxpool3 = nn.MaxPool2d(3, 2)
        self.linear1 = nn.Linear(160, 64)
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
        # print(f"prediction shape: {prediction.shape}")
        # print(f"label shape: {label.type(torch.LongTensor).shape}")
        # print(f"prediction: {prediction}")
        # print(f"label: {label.type(torch.LongTensor)}")
        # print(f"type1 {type(prediction[0])}")
        # print(f"type2 {type(label[0])}")
        return F.mse_loss(prediction, label.type(torch.FloatTensor))

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

if __name__ == '__main__':
    train_main()