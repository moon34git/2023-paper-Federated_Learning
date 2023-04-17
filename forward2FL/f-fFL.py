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
import torchvision.transforms as transforms

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 5

def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
         transforms.Lambda(lambda x: torch.flatten(x))]
    )
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
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
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(34))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(34))
        trainloaders.append(DataLoader(ds_train, batch_size=64, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=64))
    testloader = DataLoader(testset, batch_size=64)
   
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
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1])]


    def test(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
       

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = optim.Adam(self.parameters(), lr=0.01)
        self.threshold = 2.0
        self.num_epochs = 200

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        # print(f'x in cuda? : {x.is_cuda}, value of x :{x}, shape of x :{x.shape}')
        # print(f'weight in cuda? : {self.weight.is_cuda}, type of x :{type(self.weight)}, value of x :{self.weight}, shape of x :{self.weight.shape}')
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
    result1 = [val.cpu().numpy() for _, val in net.layers[0].state_dict().items()]
    result2 = [val.cpu().numpy() for _, val in net.layers[1].state_dict().items()]
    # print(f'get_parameters[0]: {result1}')
    # print(f'get_parameters[1]: {result2}')
    # print(f'summation of get_parameters[0] and get_parameters[1]: {result1 + result2}')
    # print(f'length: {len(result1 + result2)}')
    return result1 + result2
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    # print(f'set_parameters: {parameters}')
    # params_dict = zip(net.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # net.load_state_dict(state_dict, strict=True)
    params_dict1 = zip(net.layers[0].state_dict().keys(), parameters[:2])
    params_dict2 = zip(net.layers[1].state_dict().keys(), parameters[2:])
    state_dict1 = OrderedDict({k: torch.Tensor(v) for k, v in params_dict1})
    state_dict2 = OrderedDict({k: torch.Tensor(v) for k, v in params_dict2})
    net.layers[0].load_state_dict(state_dict1, strict=True)
    net.layers[1].load_state_dict(state_dict2, strict=True)

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
        # print(f'fit_params: {parameters[0][0][:5]}')
        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        
        x, y = next(iter(self.trainloader))
        x_pos = overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])
        self.net.train(x_pos, x_neg)

        print('train accuracy:', self.net.test(x).eq(y).float().mean().item())
        # self.net.train(self.net, self.trainloader, epochs=local_epochs)
        # print(f'After fit_params: {get_parameters(self.net)[0][0][:5]}')
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        x, y = next(iter(self.valloader))
        loss = 0
        accuracy = self.net.test(x).eq(y).float().mean().item()
        print(f"----------------------------------client evaluation loss {loss} / accuracy {accuracy}-----------------------------------")
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
    # x, y = next(iter(valloaders[0]))
    x, y = next(iter(testloader))
    # print(f'evaluate_params: {parameters[0][0][:5]}')
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss = 0
    accuracy = net.test(x).eq(y).float().mean().item()
    print(f"----------------------------------Server-side evaluation loss {loss} / accuracy {accuracy}-----------------------------------")
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        # "local_epochs": 1 if server_round < 2 else 2,  #
        "local_epochs": 1,
    }
    return config

trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
# trainloaders, valloaders, testloader = load_data(NUM_CLIENTS)
# Create an instance of the model and get the parameters

params = get_parameters(Net([784, 500, 500]))

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
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
    config=fl.server.ServerConfig(num_rounds = 3),  # Just three rounds
    strategy=strategy,
    client_resources=client_resources,
)
