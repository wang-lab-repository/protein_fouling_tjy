import torch
import torch.nn as nn


# network infrastructure
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        f1 = 95
        f2 = 81
        dropout_rate1 = 0.05
        dropout_rate2 = 0.05

        self.conv1 = nn.Conv1d(1, 4, 5)
        self.conv2 = nn.Conv1d(4, 16, 4)
        self.pool1 = nn.MaxPool1d(5)
        self.pool2 = nn.MaxPool1d(3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(38, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, f2)
        self.fc4 = nn.Linear(f2, 2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout_rate1)
        self.drop2 = nn.Dropout(p=dropout_rate2)
        self.t1 = 1

    def forward(self, x):

        x_num = x[:, 0:6]
        x_cat = x[:, 6:]
        x_cat = x_cat.reshape(-1, 1, 49)

        x_cat = self.conv1(x_cat)
        x_cat = self.relu(x_cat)
        x_cat = self.pool1(x_cat)
        x_cat = self.conv2(x_cat)
        x_cat = self.relu(x_cat)
        x_cat = self.pool2(x_cat)
        x_cat = self.flat(x_cat)
        x = torch.cat([x_cat, x_num], axis=1)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        temp = self.relu(x)
        x = self.relu(self.fc3(temp))
        if self.t1 == 1:
            x = self.relu(self.fc4(x + temp))
        else:
            x = self.relu(self.fc4(x))
        return x
