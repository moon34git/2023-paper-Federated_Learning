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
import random
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy import stats


def create_dataset(X, y, time_steps, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(x)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)
    
def load_data(type):
    if type == 'modal1':
        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_tr_modal1.pickle', 'rb') as f:
            X_tr_modal1 = pickle.load(f)

        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_ts_modal1.pickle', 'rb') as f:
            X_ts_modal1 = pickle.load(f)
    elif type == 'modal2':
        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_tr_modal2.pickle', 'rb') as f:
            X_tr_modal1 = pickle.load(f)

        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_ts_modal2.pickle', 'rb') as f:
            X_ts_modal1 = pickle.load(f)
    else:
        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_tr_modal3.pickle', 'rb') as f:
            X_tr_modal1 = pickle.load(f)

        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/X_ts_modal3.pickle', 'rb') as f:
            X_ts_modal1 = pickle.load(f)

    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/y_tr_modal.pickle', 'rb') as f:
        y_tr_modal = pickle.load(f)

    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/y_ts_modal.pickle', 'rb') as f:
        y_ts_modal = pickle.load(f)

    X_tr_modal1, y_tr_modal = create_dataset(X_tr_modal1, y_tr_modal, 32, 16)
    X_ts_modal1, y_ts_modal = create_dataset(X_ts_modal1, y_ts_modal, 32, 16)  

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

    for i in range(12):
        m1tr_index += np.random.choice(ytr[i], 72, replace=False).tolist() # Each client(3) trains 144 samples
        m2tr_index += np.random.choice(ytr[i], 72, replace=False).tolist()
        m3tr_index += np.random.choice(ytr[i], 72, replace=False).tolist()

        m1v_index += np.random.choice(ytr[i], 24, replace=False).tolist()   # Each client validates 48 samples
        m2v_index += np.random.choice(ytr[i], 24, replace=False).tolist()
        m3v_index += np.random.choice(ytr[i], 24, replace=False).tolist()

        m1ts_index += np.random.choice(yts[i], 24, replace=False).tolist()  # Each client tests 60 samples
        m2ts_index += np.random.choice(yts[i], 24, replace=False).tolist()
        m3ts_index += np.random.choice(yts[i], 24, replace=False).tolist()

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


class ConvLSTM12(nn.Module):
    def __init__(self):
        super(ConvLSTM12, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
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
    
class ConvLSTM6(nn.Module):
    def __init__(self):
        super(ConvLSTM6, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
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
        print(f"Test accuracy: {test_acc}")
    return float(loss), test_acc

def get_parameter():
    parameters = []
    for i in range(1, 4):
        with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/multiFL/mHealth/unbalanced_weights/modal'+str(i)+'_weights.pickle', 'rb') as f:
            data = pickle.load(f)
            parameters.append(data)

    avg_params = []

    for i in range(22, 26):
        avg_params.append((parameters[0][i] + parameters[1][i] + parameters[2][i]) / 3)
    
    for i in range(22, 26):
        parameters[0][i] = avg_params[i-22]
        parameters[1][i] = avg_params[i-22]
        parameters[2][i] = avg_params[i-22]

    return parameters

def set_parameters(net, net_State_dict ,parameters: List[np.ndarray]):
    params_dict = zip(net_State_dict, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    state_dict['bn1.num_batches_tracked'] = torch.tensor(0)
    state_dict['bn2.num_batches_tracked'] = torch.tensor(0)
    net.load_state_dict(state_dict, strict = True)

def model(type, DEVICE):
    if type == 'modal1':
        net = ConvLSTM6().to(DEVICE)
    elif type == 'modal2':
        net = ConvLSTM6().to(DEVICE)
    else:
        net = ConvLSTM12().to(DEVICE)

    X_trains, y_trains, X_vals, y_vals, X_test, y_test = load_data(type)
    agg = get_parameter()
    net_State_dict = net.state_dict().keys()

    if type == 'modal1':
        set_parameters(net, net_State_dict, agg[0])
    elif type == 'modal2':
        set_parameters(net, net_State_dict, agg[1])
    else:
        set_parameters(net, net_State_dict, agg[2])

    test(net, X_test, y_test)

if __name__ == "__main__":
    torch.manual_seed(1)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    DEVICE = torch.device("cuda") 
    for i in ['modal1', 'modal2', 'modal3']:
        model(i, DEVICE)