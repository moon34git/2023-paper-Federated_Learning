from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
import pickle
import torch.optim as optim
from tqdm import tqdm
import flwr as fl

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

def load_data(num_clients: int):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/trainset5000.pickle', 'rb') as trs:
        trainset = pickle.load(trs)
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/testset.pickle', 'rb') as tts:
        testset = pickle.load(tts)

    trainset.transform = transform
    testset.transform = transform

    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
   
    return trainloaders, valloaders, testloader

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
  
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        # self.layers = []
        # for d in range(len(dims) - 1):
        #     self.layers += [Layer(dims[d], dims[d + 1])]

        self.layer1 = Layer(dims[0], dims[1])
        self.layer2 = Layer(dims[1], dims[2])
   

    # def test(self, x):
    #     goodness_per_label = []
    #     for label in range(10):
    #         h = overlay_y_on_x(x, label)
    #         goodness = []
    #         for layer in self.layers:
    #             h = layer(h)
    #             goodness += [h.pow(2).mean(1)]
    #         goodness_per_label += [sum(goodness).unsqueeze(1)]
    #     goodness_per_label = torch.cat(goodness_per_label, 1)
    #     return goodness_per_label.argmax(1)

    def test(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            h = self.layer1(h)
            goodness += [h.pow(2).mean(1)]
            h = self.layer2(h)
            goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)


    # def train(self, x_pos, x_neg):
    #     h_pos, h_neg = x_pos, x_neg
    #     for i, layer in enumerate(self.layers):
    #         print('training layer', i, '...')
    #         h_pos, h_neg = layer.train(h_pos, h_neg)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        print('training layer', 0, '...')
        h_pos = self.layer1.train(h_pos, h_neg)
        print('training layer', 1, '...')
        h_neg = self.layer2.train(h_pos, h_neg)

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = optim.Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # x_direction = x_direction.to(DEVICE)
        print(f'x in cuda? : {x.is_cuda}, value of x :{x}, shape of x :{x.shape}')
        print(f'weight in cuda? : {self.weight.is_cuda}, type of x :{type(self.weight)}, value of x :{self.weight}, shape of x :{self.weight.shape}')
        matmul = torch.mm(x_direction, self.weight.T)
        bias = self.bias.unsqueeze(0)
        output = self.relu(matmul + bias)
        return output

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters, Client get_parameters {len(get_parameters(self.net))}")
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
        print(f'Parameter Size (For Client fitting): {len(parameters)}')
        set_parameters(self.net, parameters)
        
        x, y = next(iter(self.trainloader))
        # x, y = x.to(DEVICE), y.to(DEVICE)
        x_pos = overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])
        self.net.train(x_pos, x_neg)
        # print(f'x = {x.is_cuda}')
        # print(f'y = {y.is_cuda}')

        print('train error:', 1.0 - self.net.test(x).eq(y).float().mean().item())
        # self.net.train(self.net, self.trainloader, epochs=local_epochs)

        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        print(f'Parameter Size (For Client evaluation): {len(parameters)}')
        set_parameters(self.net, parameters)
        x, y = next(iter(self.valloader))
        loss = 0
        accuracy = 1.0 - self.net.test(x).eq(y).float().mean().item()
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    


def client_fn(cid) -> FlowerClient:
    net = Net([784, 500, 500]).to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net([784, 500, 500]).to(DEVICE)
    x, y = next(iter(valloaders[0]))
    # x, y = x.to(DEVICE), y.to(DEVICE)
    print(f'Parameter Size (For server evaluation): {len(parameters)}')
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss = 0
    accuracy = 1.0 - net.test(x).eq(y).float().mean().item()
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

trainloaders, valloaders, testloader = load_data(2)
params = get_parameters(Net([784, 500, 500]))

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=2,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    initial_parameters=fl.common.ndarrays_to_parameters(params),
    evaluate_fn=evaluate,
    on_fit_config_fn = fit_config,
)

client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
    strategy=strategy,
    client_resources=client_resources,
)

