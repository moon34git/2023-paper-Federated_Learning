import torch 
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def custom():
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/acoustic.pickle', 'rb') as f:
        data1 = pickle.load(f)

    data1['class'] = data1['class'].apply(lambda x: 0 if x == 1 else (1 if x == 2 else 2)) 

    X = data1[[str(x) for x in range(50)]]
    y = data1['class']
    X = X.values
    y = y.values
    # y = torch.tensor(y.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=34)

    class customDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.len = len(self.X)
        def __getitem__(self, index):
            return self.X[index], self.y[index]
        def __len__(self):
            return self.len

    train_data = customDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = customDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_data, batch_size = 128, shuffle= True)
    test_loader = DataLoader(test_data, batch_size = 128, shuffle= True)

    class CustomModel(nn.Module):
        def __init__(self):
            super(CustomModel,self).__init__()
            self.fc1 = nn.Linear(50, 32)
            self.fc2 = nn.Linear(32,32)
            self.fc3 = nn.Linear(32,3)
            self.droput = nn.Dropout(0.2)

        def forward(self,x):

            x = x.view(-1,50)
            x = F.relu(self.fc1(x))
            x = self.droput(x)
            x = F.relu(self.fc2(x))
            x = self.droput(x)
            x = self.fc3(x)

            return x


    model = CustomModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    for epoch in tqdm(range(50)):
        correct, total, epoch_loss = 0, 0, 0.0

        for x, y in train_loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()   
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            total += y.size(0)
            correct += (output.argmax(1) == y).sum().item()
        epoch_loss /= len(train_loader)
        epoch_acc = correct / total

        if (epoch + 1) % 10 == 0:
            print(f"Epoch : {epoch+1:4d}, Loss : {epoch_loss:.3f}, Acc : {epoch_acc:.3f}")



    """Evaluate the network on the entire test set."""

    correct1, total1, loss1 = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss1 += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
    loss1 /= len(test_loader)
    accuracy1 = correct1 / total1

    print(f'Loss: {loss1:.3f}, Accuracy: {accuracy1:.3f}')



    # print(f"Model structure: {model}\n\n")

    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


def fcn():
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/acoustic.pickle', 'rb') as f:
        data1 = pickle.load(f)

    data1['class'] = data1['class'].apply(lambda x: 0 if x == 1 else (1 if x == 2 else 2)) 

    X = data1[[str(x) for x in range(50)]]
    y = data1['class']
    X = X.values
    y = y.values
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=34)

    torchvisionType_for_trainVal = []
    for i in range(len(X_train)):
        torchvisionType_for_trainVal.append((X_train[i], y_train[i]))

    torchvisionType_for_test = []
    for i in range(len(X_test)):
        torchvisionType_for_test.append((X_test[i], y_test[i]))

    partition_size = len(torchvisionType_for_trainVal) // 2
    lengths = [partition_size] * 2
    datasets = random_split(torchvisionType_for_trainVal, lengths, torch.Generator().manual_seed(42))

    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size = 128, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size = 128))
    testloader = DataLoader(torchvisionType_for_test, batch_size = 128)



    class FCNet(nn.Module):
        def __init__(self):
            super(FCNet,self).__init__()
            self.fc1 = nn.Linear(50, 32)
            self.fc2 = nn.Linear(32,32)
            self.fc3 = nn.Linear(32,3)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
        
    net = FCNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in tqdm(range(50)):
        correct, total, epoch_loss = 0, 0, 0.0
        for x, y in trainloaders[0]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += y.size(0)
            correct += (output.argmax(1) == y).sum().item()
        epoch_loss /= len(trainloaders[0].dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

if __name__ == '__main__':
    fcn()
    # custom()

