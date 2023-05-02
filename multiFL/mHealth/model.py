import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle
import os
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('/home/jhmoon/venvFL/2023-paper-Federated_Learning/Data/mHealth/mhealth_raw_data.pickle', 'rb') as f:
    df = pickle.load(f)

from sklearn.utils import resample
 
df_majority = df[df.Activity==0]
df_minorities = df[df.Activity!=0]
 
df_majority_downsampled = resample(df_majority,n_samples=30000, random_state=42)
df = pd.concat([df_majority_downsampled, df_minorities])
df.Activity.value_counts()

#Dropping feature have data outside 98% confidence interval
df1 = df.copy()
for feature in df1.columns[:-1]:
  lower_range = np.quantile(df[feature],0.01)
  upper_range = np.quantile(df[feature],0.99)
  print(feature,'range:',lower_range,'to',upper_range)

  df1 = df1.drop(df1[(df1[feature]>upper_range) | (df1[feature]<lower_range)].index, axis=0)
  print('shape',df1.shape)

label_map = {
    0: 'Nothing',
    1: 'Standing still',  
    2: 'Sitting and relaxing', 
    3: 'Lying down',  
    4: 'Walking',  
    5: 'Climbing stairs',  
    6: 'Waist bends forward',
    7: 'Frontal elevation of arms', 
    8: 'Knees bending (crouching)', 
    9: 'Cycling', 
    10: 'Jogging', 
    11: 'Running', 
    12: 'Jump front & back' 
}

#spliting data into train and test set
print(df1.shape)
train = df1[(df1['subject'] != 'subject10') & (df1['subject'] != 'subject9')]
test = df1.drop(train.index, axis=0)
train.shape,test.shape


X_train = train.drop(['Activity','subject'],axis=1)
y_train = train['Activity']
X_test = test.drop(['Activity', 'subject'],axis=1)
y_test = test['Activity']
X_train.shape,y_train.shape,X_test.shape,y_test.shape


from scipy import stats

#function to create time series datset for seuence modeling
def create_dataset(X, y, time_steps, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(x)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

X_train,y_train = create_dataset(X_train, y_train, 100, step=50)
X_train.shape, y_train.shape

X_test,y_test = create_dataset(X_test, y_test, 100, step=50)
X_test.shape, y_test.shape
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=13)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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

model = ConvLSTM().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train1 = torch.from_numpy(X_train).float().to(DEVICE)
y_train1 = torch.from_numpy(y_train).long().to(DEVICE)

X_test1 = torch.from_numpy(X_test).float().to(DEVICE)
y_test1 = torch.from_numpy(y_test).long().to(DEVICE)

X_train1.shape, y_train1.shape, X_test1.shape, y_test1.shape

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# def train(model, train_loader, criterion, optimizer, device):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

# # Define the testing function
# def test(model, test_loader, criterion, device):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.to(device)
#             target = target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

def train(model, X_train, y_train, X_test, y_test, optimizer, loss_fn, epochs=10, device=DEVICE):
    train_acc = []
    test_acc = []
    train_loss =[]
    test_loss = []
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        if epoch == 1:
            print(X_train.shape, y_train.shape)
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train.squeeze())
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)
            train_acc.append(accuracy_score(y_train.cpu(), y_pred_train.cpu().argmax(1)))
            test_acc.append(accuracy_score(y_test.cpu(), y_pred_test.cpu().argmax(1)))
            train_loss.append(loss.item())
            print('Epoch: {} Train Loss: {:.4f} Train Acc: {:.4f} Test Acc: {:.4f}'.format(epoch, loss.item(), train_acc[-1], test_acc[-1]))
    return train_acc, test_acc, train_loss


train_acc, test_acc, train_loss = train(model, X_train1, y_train1, X_test1, y_test1, optimizer, loss_fn, epochs=100, device = DEVICE)

# plt.plot(train_acc, label='train', color = 'b')

# plt.plot(test_acc, label='test', color = 'r')

# # plt.plot(train_loss, label = 'train_loss', color = 'yellow')

# plt.legend()

# plt.show()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Define the LSTM model
# class LSTMClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         # Get the last output of the LSTM for each sequence
#         last_output = lstm_out[:, -1, :]
#         output = self.fc(last_output)
#         return output

# # Define the training function
# def train(model, train_loader, criterion, optimizer):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

# # Define the testing function
# def test(model, test_loader, criterion):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.to(device)
#             target = target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')

# # Define the main function

#     # Set the device
    

# # Define the parameters
# input_dim = 10
# hidden_dim = 20
# output_dim = 2
# lr = 0.01
# epochs = 10
# batch_size = 64

# # Create the model, criterion, optimizer
# model = LSTMClassifier(input_dim, hidden_dim, output_dim).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)

# # Generate some dummy data
# train_data = torch.randn(1000, 5, input_dim)
# train_target = torch.randint(0, output_dim, (1000,))
# test_data = torch.randn(100, 5, input_dim)
# test_target = torch.randint(0, output_dim, (100,))

# # Create the data loaders
# train_dataset = torch.utils.data.TensorDataset(train_data, train_target)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = torch.utils.data.TensorDataset(test_data, test_target)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Train the model
# for epoch in range(1, epochs + 1):
#     train(model, train_loader, criterion, optimizer)
#     test(model, test_loader, criterion)
