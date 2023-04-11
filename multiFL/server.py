import torch 
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('2023-paper-Federated_Learning/Data/acoustic.pickle', 'rb') as f:
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

train_loader = DataLoader(train_data, batch_size = 64, shuffle= True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle= True)

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

model = CustomModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in tqdm(range(100)):
    cost = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch : {epoch+1:4d}, Cost : {cost:.3f}")


# with torch.no_grad():
#     model.eval()
#     classes = {0: "acute triangle", 1: "right triangle", 2: "obtuse triangle"}
#     inputs = torch.FloatTensor([
#         [9.02, 9.77, 9.96], # 0 | acute triangle
#         [8.01, 8.08, 8.32], # 0 | acute triangle
#         [3.55, 5.15, 6.26], # 1 | right triangle
#         [3.32, 3.93, 5.14], # 1 | right triangle
#         [4.39, 5.56, 9.99], # 2 | obtuse triangle
#         [3.01, 3.08, 9.98], # 2 | obtuse triangle
#         [5.21, 5.38, 5.39], # 0 | acute triangle
#         [3.85, 6.23, 7.32], # 1 | right triangle
#         [4.16, 4.98, 8.54], # 2 | obtuse triangle
#     ]).to(device)
#     outputs = model(inputs)
    
#     print('---------')
#     print(outputs)
#     print(torch.round(F.softmax(outputs, dim=1), decimals=2))
#     print(outputs.argmax(1))
#     print(list(map(classes.get, outputs.argmax(1).tolist())))