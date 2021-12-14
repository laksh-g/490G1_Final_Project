import torch
import torch.utils.data
import numpy as np

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, train, device):
        'Initialization'
        self.device = device
        self.train = np.asarray(train, dtype=object)
        self.labels = [x[1][1] for x in self.train]
        self.list_IDs = np.arange(1, len(train)+1)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        # print(type(self.train[index][0][0]))
        # X = np.concatenate(self.train[index][0][0], self.train[index][1][0], axis=2)
        #X = torch.LongTensor(self.train[index][0][0]).reshape(3, 288, 432)
        X = torch.LongTensor(self.train[index][0][0].reshape(3, 288, 432))
        X = X.type(torch.LongTensor)
        # X = X.to(self.device)


        # X = np.concatenate((self.train[index][0][0], self.train[index][1][0]), axis=2).reshape(6, 124416)
        print(X.shape)
        y = self.labels[ID]
        return X, y