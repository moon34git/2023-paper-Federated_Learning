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
import pickle

import flwr as fl

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS = 2

def load_data(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
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


# class FCNet(nn.Module):
#     def __init__(self):
#         super(FCNet,self).__init__()
#         self.fc1 = nn.Linear(50, 32)
#         self.fc2 = nn.Linear(32,32)
#         self.fc3 = nn.Linear(32,3)
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet,self).__init__()
        self.fc1 = nn.Linear(in_features=50,out_features=32)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=32,out_features=32)
        self.fc3 = nn.Linear(in_features=32,out_features=3)

    def forward(self,input_data):
        out = input_data.view(-1, 50)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for x, y in trainloader:
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
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = net(x)
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):

        # print(f"[Client {self.cid}] fit, config: {config}")
        # set_parameters(self.net, parameters)
        # train(self.net, self.trainloader, epochs=1)

        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)

        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def client_fn(cid) -> FlowerClient:
    net = FCNet().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = FCNet().to(DEVICE)
    valloader = valloaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config

# trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
trainloaders, valloaders, testloader = load_data(NUM_CLIENTS)
# Create an instance of the model and get the parameters
params = get_parameters(FCNet())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=2,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(params),
    evaluate_fn=evaluate,
    on_fit_config_fn = fit_config,
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
    strategy=strategy,
    client_resources=client_resources,
)
