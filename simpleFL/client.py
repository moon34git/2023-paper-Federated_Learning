import flwr as fl
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from typing import Callable, Dict, Optional, Tuple
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from collections import OrderedDict
import pickle
import time
import os

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = (5, 5), stride = (1, 1), padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size = (2, 2), padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = (2, 2))

        self.drop1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3136, 1000)
        self.drop2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.drop1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)

        return F.softmax(x, dim = 0)
    
def load_data(start_tr, end_tr, start_ts, end_ts, batch_size):
    with open('../Data/trainset5000.pickle', 'rb') as trs:
        train = pickle.load(trs)
    with open('../Data/testset.pickle', 'rb') as tts:
        test = pickle.load(tts)

    print("-----------INDEX INFO-----------")
    print(start_tr, end_tr, start_ts, end_ts)
# python server.py 2 72 & python client.py 2 0 72 & python client.py 2 1 72 && fg
# python server1.py 2 72 & python client.py 2 0 72 & python client.py 2 1 72 && fg
# python server.py 3 72 & python client.py 3 0 72 & python client.py 3 1 72 & python client.py 3 2 72 && fg
# python server.py 6 36 & python client.py 6 0 36 & python client.py 6 1 36 & python client.py 6 2 36 & python client.py 6 3 36 & python client.py 6 4 36 & python client.py 6 5 36 && fg
# python server.py 9 24 & python client.py 9 0 24 & python client.py 9 1 24 & python client.py 9 2 24 & python client.py 9 3 24 & python client.py 9 4 24 & python client.py 9 5 24 & python client.py 9 6 24 & python client.py 9 7 24 & python client.py 9 8 24 && fg
# python server.py 12 & python client.py 12 0 23 & python client.py 12 1 23 & python client.py 12 2 23 & python client.py 12 3 23 & python client.py 12 4 23 & python client.py 12 5 23 & python client.py 12 6 23 & python client.py 12 7 23 & python client.py 12 8 23 & python client.py 12 9 23 & python client.py 12 10 23 & python client.py 12 11 23 && fg
    
    trainset = torch.utils.data.Subset(train, range(start_tr, end_tr))
    testset = torch.utils.data.Subset(test, range(start_ts, end_ts))

    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
    testloader = DataLoader(testset, batch_size = batch_size)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}

    return trainloader, testloader, num_examples

def train(net, trainloader, epochs, DEVICE):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

        print(f"Training Loss: {loss:>7f}\n")
        
def test(net, testloader, DEVICE):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Loss: {loss}\n")
    print(f"Test Accuracy {accuracy}\n")
    return loss, accuracy


def main():
    time.sleep(5)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_size = int(sys.argv[1])
    client_num = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    tr_data_size = 5000
    ts_data_size = 10000

    client_data_size_tr = int(tr_data_size / client_size)
    client_data_size_ts = int(ts_data_size / client_size)

    start_index_tr = client_num * client_data_size_tr
    start_index_ts = client_num * client_data_size_ts

    end_index_tr = client_data_size_tr * (client_num + 1)
    end_index_ts = client_data_size_ts * (client_num + 1)

    net = model().to(DEVICE)

    trainloader, testloader, num_examples = load_data(start_index_tr, end_index_tr, start_index_ts, end_index_ts, batch_size)

    # ORDER : flClient - get_parameters - fit - get_parameters - evaluate

    class flClient(fl.client.NumPyClient):
        def __init__(self):
            print("-----------------------FLOWER CLIENT EXECUTED---------------------")
        
        def get_parameters(self, config):   # `get_parameters`: 현재 로컬 모델 매개변수 반환
            result = [val.cpu().numpy() for _, val in net.state_dict().items()]
            # print(f"Get on client (0, get_parameter): {result[0][0][0][0][0]:>4f}")
            # print(f"Get on client (7, get_parameter): {result[7][0]:>4f}")
            return result

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict = True)

        def fit(self, parameters, config):
            # print(f"Before set_parameter in FIT fn(0): {parameters[0][0][0][0][0]:>4f}")
            # print(f"Before set_parameter in FIT fn(7): {parameters[7][0]:>4f}")
            self.set_parameters(parameters)
            train(net, trainloader, epochs = 1)
            # print(f"After set_parameter in FIT fn (0): {self.get_parameters(config={})[0][0][0][0][0]:>4f}")
            # print(f"After set_parameter in FIT fn (7): {self.get_parameters(config={})[7][0]:>4f}")
            return self.get_parameters(config={}), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

    client_start = time.time()
    fl.client.start_numpy_client(server_address="117.17.189.210:8080", client=flClient())
    client_end = time.time()
    print(f"Client Execution Time: {client_end - client_start}")
    
if __name__ == "__main__":
    main()
