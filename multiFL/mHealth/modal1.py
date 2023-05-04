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
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import random
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 3

def create_dataset(X, y, time_steps, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(x)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)
    
def load_data(num_clients: int):
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_tr_modal1.pickle', 'rb') as f:
        X_tr_modal1 = pickle.load(f)

    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_ts_modal1.pickle', 'rb') as f:
        X_ts_modal1 = pickle.load(f)

    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/y_tr_modal.pickle', 'rb') as f:
        y_tr_modal = pickle.load(f)

    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/y_ts_modal.pickle', 'rb') as f:
        y_ts_modal = pickle.load(f)

    X_tr_modal1, y_tr_modal = create_dataset(X_tr_modal1, y_tr_modal, 64, 32) #Batch : 100, Step : 50
    X_ts_modal1, y_ts_modal = create_dataset(X_ts_modal1, y_ts_modal, 64, 32)  

    X_train = np.transpose(X_tr_modal1, (0, 2, 1))
    X_test = np.transpose(X_ts_modal1, (0, 2, 1))

    X_train = torch.from_numpy(X_train).float().to(DEVICE)
    y_train = torch.from_numpy(y_tr_modal).long().to(DEVICE)

    X_test = torch.from_numpy(X_test).float().to(DEVICE)
    y_test = torch.from_numpy(y_ts_modal).long().to(DEVICE)

    yvalues = pd.Series(y_train.squeeze().cpu().numpy())
    yvaluess = pd.Series(y_test.squeeze().cpu().numpy())
    ytr = []
    yts = []
    for i in range(12):
        ytr.append(yvalues[yvalues == i].index.to_list())
        yts.append(yvaluess[yvaluess == i].index.to_list())

    m1tr_index = []
    m2tr_index = []
    m3tr_index = []

    m1v_index = []
    m2v_index = []
    m3v_index = []

    m1ts_index = []
    m2ts_index = []
    m3ts_index = []

    a = 30
    b = 30 // 3
    for i in range(12):
        m1tr_index += np.random.choice(ytr[i], a, replace=False).tolist()
        m2tr_index += np.random.choice(ytr[i], a, replace=False).tolist()
        m3tr_index += np.random.choice(ytr[i], a, replace=False).tolist()

        m1v_index += np.random.choice(ytr[i], b, replace=False).tolist()
        m2v_index += np.random.choice(ytr[i], b, replace=False).tolist()
        m3v_index += np.random.choice(ytr[i], b, replace=False).tolist()

        m1ts_index += np.random.choice(yts[i], b, replace=False).tolist()
        m2ts_index += np.random.choice(yts[i], b, replace=False).tolist()
        m3ts_index += np.random.choice(yts[i], b, replace=False).tolist()

    m1tr_index = random.sample(m1tr_index, len(m1tr_index))
    m2tr_index = random.sample(m2tr_index, len(m2tr_index))
    m3tr_index = random.sample(m3tr_index, len(m3tr_index))

    m1v_index = random.sample(m1v_index, len(m1v_index))
    m2v_index = random.sample(m2v_index, len(m2v_index))
    m3v_index = random.sample(m3v_index, len(m3v_index))

    m1ts_index = random.sample(m1ts_index, len(m1ts_index))
    m2ts_index = random.sample(m2ts_index, len(m2ts_index))
    m3ts_index = random.sample(m3ts_index, len(m3ts_index))

    X_trains = [X_train[m1tr_index], X_train[m2tr_index], X_train[m3tr_index]]
    y_trains = [y_train[m1tr_index], y_train[m2tr_index], y_train[m3tr_index]]
    X_vals = [X_train[m1v_index], X_train[m2v_index], X_train[m3v_index]]
    y_vals = [y_train[m1v_index], y_train[m2v_index], y_train[m3v_index]]
    X_test = X_test[m1ts_index]
    y_test = y_test[m1ts_index]


    return X_trains, y_trains, X_vals, y_vals, X_test, y_test

        
# class ConvLSTM6(nn.Module):
#     def __init__(self):
#         super(ConvLSTM6, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.relu2 = nn.ReLU()
#         self.pool = nn.MaxPool1d(kernel_size=2)
#         self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
#         self.fc1 = nn.Linear(in_features=64, out_features=128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=128, out_features=13)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         # print(f"Round {self.round} initialized")
#         # Convolutional layers
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.pool(x)

#         # LSTM layer
#         x = x.permute(0, 2, 1)  # Change from (batch_size, seq_len, num_features) to (batch_size, num_features, seq_len)
#         x, _ = self.lstm(x)
#         x = x[:, -1, :]  # Extract the last timestep output


#         # Fully connected layers
#         x = self.fc1(x)
#         x = self.relu3(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x

class ConvLSTM6(nn.Module):
    def __init__(self):
        super(ConvLSTM6, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=13)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(f"Round {self.round} initialized")
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)

        # LSTM layer
        x = x.permute(0, 2, 1)  # Change from (batch_size, seq_len, num_features) to (batch_size, num_features, seq_len)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Extract the last timestep output


        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict['bn1.num_batches_tracked'] = torch.tensor(0)
    state_dict['bn2.num_batches_tracked'] = torch.tensor(0)
    net.load_state_dict(state_dict, strict=True)

def train(net, X_train, y_train, epochs: int):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in tqdm(range(epochs)):
        net.train()
        net.zero_grad()
        X_train = X_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        y_pred = net(X_train)
        loss = criterion(y_pred, y_train.squeeze())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}: train loss {loss:.4f}")

def test(net, X_test, y_test):
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    # test_acc = []
    # test_loss = []
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    with torch.no_grad():
        y_pred_test = net(X_test)
        loss = criterion(y_pred_test, y_test.squeeze())
        # test_loss.append(loss)
        test_acc = accuracy_score(y_test.cpu(), y_pred_test.cpu().argmax(1))
    return float(loss), test_acc
        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, X_train, y_train, X_test, y_test):
        self.cid = cid
        self.net = net
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

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
        train(self.net, self.X_train, self.y_train, epochs=local_epochs)

        return get_parameters(self.net), len(self.X_train), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.X_val, self.y_val)
        return float(loss), len(self.X_val), {"accuracy": float(accuracy)}

def client_fn(cid) -> FlowerClient:
    net = ConvLSTM6().to(DEVICE)
    X_train, y_train = X_trains[int(cid)], y_trains[int(cid)]
    X_val, y_val = X_vals[int(cid)], y_vals[int(cid)]
    return FlowerClient(cid, net, X_train, y_train, X_val, y_val)

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = ConvLSTM6().to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, X_test, y_test)
    print(f"Server-side evaluation loss {loss:.4f} / accuracy {accuracy:.4f}")

    if server_round == 3:
        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/multiFL/mHealth/balanced_weights/modal1_weights.pickle', 'wb') as f:
            pickle.dump(parameters, f)
    
    return float(loss), {"accuracy": accuracy}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 30 
    }
    return config

# trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
X_trains, y_trains, X_vals, y_vals, X_test, y_test = load_data(NUM_CLIENTS)
# Create an instance of the model and get the parameters
params = get_parameters(ConvLSTM6())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
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

