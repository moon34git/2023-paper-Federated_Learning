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

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 5

def load_data(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/acoustic.pickle', 'rb') as f:
        data1 = pickle.load(f)
    data1 = data1.iloc[:5000]

    X = data1[[str(x) for x in range(50)]]
    y = data1['class']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # X = X.values
    y = y.values
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.long)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 34)

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
#         super(FCNet, self).__init__()
#         self.fc1 = nn.Linear(50, 32)
#         self.fc2 = nn.Linear(32, 32)
#         self.fc3 = nn.Linear(32, 1)
        # self.dropout = nn.Dropout(0.25)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x
    
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(50, 32)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout(x)
        x = self.act2(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output(x))
        # x = self.output(x)
        return x
    
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# def train(net, trainloader, epochs: int):
#     """Train the network on the training set."""
#     # criterion = torch.nn.CrossEntropyLoss()
#     criterion = torch.nn.BCELoss()
#     optimizer = torch.optim.Adam(net.parameters())
#     net.train()
#     for epoch in range(epochs):
#         correct, total, epoch_loss = 0, 0, 0.0
#         for x, y in trainloader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             optimizer.zero_grad()
#             y = y.unsqueeze(1)
#             output = net(x)
#             loss = criterion(output.to(torch.float32), y.to(torch.float32))
#             loss.backward()
#             optimizer.step()
#             # Metrics
#             epoch_loss += loss
#             total += y.size(0)
#             correct += (output.argmax(1) == y).sum().item()
#         epoch_loss /= len(trainloader.dataset)
#         epoch_acc = correct / total
#         print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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
            # Metrics
            epoch_loss += loss
            total += y.size(0)
            predicted = (output > 0.5).float()
            correct += (predicted == y).sum()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


# def test(net, testloader):
#     """Evaluate the network on the entire test set."""
#     # criterion = torch.nn.CrossEntropyLoss()
#     criterion = torch.nn.BCELoss()
#     correct, total, loss = 0, 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for x, y in testloader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             y = y.unsqueeze(1)
#             outputs = net(x)
#             loss += criterion(outputs.to(torch.float32), y.to(torch.float32)).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += y.size(0)
#             correct += (predicted == y).sum().item()
#     loss /= len(testloader.dataset)
#     accuracy = correct / total
#     return loss, accuracy

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
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
    
    # config = {
    #     "server_round": server_round,  # The current round of federated learning
    #     "local_epochs": 1 if server_round < 2 else 2,  #
    # }

    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1,  #
    }
    return config

# trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
trainloaders, valloaders, testloader = load_data(NUM_CLIENTS)
# Create an instance of the model and get the parameters
params = get_parameters(FCNet())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    fraction_evaluate= 3,
    min_fit_clients= 3,
    min_evaluate_clients=3,
    min_available_clients=3,
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
