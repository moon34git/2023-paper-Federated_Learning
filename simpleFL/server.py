import flwr as fl
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
import sys
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

server_start = time.time()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
batch_size = int(sys.argv[2])
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

MIN_AVAILABLE_CLIENTS = int(sys.argv[1])
NUM_ROUND = 5
NUM_EPOCHS = 1
file_n = 4968
# if file_n % MIN_AVAILABLE_CLIENTS > 0:
#     DIV = file_n // MIN_AVAILABLE_CLIENTS +1
# else:
#     DIV = file_n//MIN_AVAILABLE_CLIENTS
DIV = 0

def load_data(batch_size):
    with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/testset.pickle', 'rb') as tts:
        testset = pickle.load(tts)

    testloader = DataLoader(testset, batch_size = batch_size)
    num_examples = {"trainset" : 0, "testset" : len(testset)}

    return 0, testloader, num_examples

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    y_label = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_label.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = correct / total

    cr_matrix = classification_report(y_label, y_pred, digits = 4)
    cf_matrix = confusion_matrix(y_label, y_pred)

    with open('result.txt', 'w') as f:
        f.write("classification_report : \n"+ str(cr_matrix)+"\n")
        f.write("confusion_matrix : \n" +str(cf_matrix))
    f.close()

    return loss, accuracy

def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    def fit_config(rnd: int) -> Dict[str, str]:
        config = {"epoch": NUM_EPOCHS, "round": rnd, "file_n": file_n,
        "div": DIV, "n_round": NUM_ROUND, "n_clients": MIN_AVAILABLE_CLIENTS}
        return config
    return fit_config

def get_parameters(net) -> List[np.ndarray]:
    result = [val.cpu().numpy() for _, val in net.state_dict().items()]
    # print(f"Get on Server (0, get_parameter): {result[0][0][0][0][0]:>4f}")
    # print(f"Get on Server (7, get_parameter): {result[7][0]:>4f}\n")
    return result


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    # print(f"Set on Server (0, set_parameter): {parameters[0][0][0][0][0]:>4f}")
    # print(f"Set on Server (7, set_parameter): {parameters[7][0]:>4f}\n")

def evaluate(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    trainloader, testloader, num_examples = load_data(batch_size)
    net = model().to(DEVICE)
    # print(f"Set on Server (0, evaluate): {parameters[0][0][0][0][0]:>4f}")
    # print(f"Set on Server (7, evaluate): {parameters[7][0]:>4f}\n")
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}\n")
    return 0, {"err":0}


strategy = fl.server.strategy.FedAvg(
    fraction_fit = 1,
    min_fit_clients = MIN_AVAILABLE_CLIENTS,
    min_available_clients = MIN_AVAILABLE_CLIENTS,
    on_fit_config_fn = get_on_fit_config_fn(),
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(model())),
    evaluate_fn = evaluate
)



server_end = time.time()
print("-------------------SERVER EXECUTED---------------------")
print(f"Server Execution Time: {server_end - server_start}")
fl.server.start_server(
    config = fl.server.ServerConfig(num_rounds = NUM_ROUND), 
    server_address = '117.17.189.210:8080',
    strategy = strategy)

