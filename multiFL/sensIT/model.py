from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import flwr as fl
from sklearn.metrics import roc_auc_score
import ray

torch.manual_seed(1)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)



def load_data(num_clients: int):
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/sensIT/seismic.pickle', 'rb') as f:
        data1 = pickle.load(f)
    data1 = data1.iloc[:2500]

    X = data1[[str(x) for x in range(50)]]
    y = data1['class']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # X = X.values
    y = y.values
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 34)

    torchvisionType_for_trainVal = []
    for i in range(len(X_train)):
        torchvisionType_for_trainVal.append((X_train[i], y_train[i]))

    torchvisionType_for_test = []
    for i in range(len(X_test) // 10):
        torchvisionType_for_test.append((X_test[i], y_test[i]))

    partition_size = len(torchvisionType_for_trainVal) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(torchvisionType_for_trainVal, lengths, torch.Generator().manual_seed(42))

    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size = 32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size = 32))
    testloader = DataLoader(torchvisionType_for_test, batch_size = 32)

    return trainloaders, valloaders, testloader
    
class FCNet16(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(50, 32)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(32, 16)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout(x)
        x = self.act2(self.layer2(x))
        x = self.dropout(x)
        x = self.act3(self.layer3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        return x


def train(net, trainloader, epochs: int, lr):
    """Train the network on the training set."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y = y.unsqueeze(1)
            output = net(x)
            loss = criterion(output.to(torch.float32), y.to(torch.float32))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += y.size(0)
            predicted = (output > 0.5).float()
            correct += (predicted == y).sum()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}, correct {correct} total {total}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    y_label = []
    y_pred = []
    net.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y = y.unsqueeze(1)
            outputs = net(x)
            loss += criterion(outputs.to(torch.float32), y.to(torch.float32)).item()
            predicted = (outputs > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()
            y_label.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    auc = roc_auc_score(y_label, y_pred)

    print(f'total : {total}, correct: {correct}, AUC: {auc}')
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


if __name__ == "__main__":

    trainloaders, valloaders, testloader = load_data(5)

    net = FCNet16().to(DEVICE)
    train(net, trainloaders[0], 3, lr = 0.003)
    test(net, valloaders[0])