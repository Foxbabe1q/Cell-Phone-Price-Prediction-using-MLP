import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
import time

device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


def create_dataset():
    data = pd.read_csv('手机价格预测.csv')
    x = data.iloc[:,:-1].values.astype('float32')
    y = data.iloc[:,-1].values.astype('int64')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88,stratify=y)
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    train_dataset = TensorDataset(torch.tensor(x_train).to(device), torch.tensor(y_train).to(device))
    test_dataset = TensorDataset(torch.tensor(x_test).to(device), torch.tensor(y_test).to(device))

    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))

class SimpleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, output_size)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = self.fc6(x)
        return x

def train():
    torch.manual_seed(0)
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    model = SimpleNet(input_size=input_dim, output_size=output_dim).to(device)

    optimizer = optim.Adam(params=model.parameters(),lr=0.0001)

    criterion = nn.CrossEntropyLoss()

    epochs = 50

    loss_list = []
    acc_list = []

    start_time = time.time()

    for epoch in range(epochs):
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        total_loss = 0.0
        num = 0
        start_time = time.time()
        total_correct = 0

        for x, y in dataloader:
            output = model(x)
            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(y)
            total_correct += (torch.argmax(output, dim=1)==y).sum().item()
            num += len(y)
        loss_list.append(total_loss)
        acc_list.append(total_correct/num)
        print("epoch:%d, loss:%.2f, time:%.2f" %(epoch+1,total_loss/num,time.time()-start_time))
    torch.save(model.state_dict(), 'model1.pt')

    fig = plt.figure(figsize=(6,4))
    axes1 = plt.subplot(1,2,1)
    axes2 = plt.subplot(1,2,2)
    axes1.plot(np.arange(1,epochs+1),loss_list)
    axes1.grid()
    axes1.set_title('loss')
    axes1.set_xlabel('epoch')
    axes1.set_ylabel('loss')
    axes2.plot(np.arange(1,epochs+1),acc_list)
    axes2.grid()
    axes2.set_title('accuracy')
    axes2.set_xlabel('epoch')
    axes2.set_ylabel('accuracy')
    fig.savefig('loss_acc1.png')
    plt.show()


def model_test():
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    model = SimpleNet(input_size=input_dim, output_size=output_dim)
    model.to(device)
    model.load_state_dict(torch.load('model1.pt'))
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    num = 0
    total_loss = 0.0

    for x, y in dataloader:

        with torch.no_grad():
            output = model(x)
            correct += (y==torch.argmax(output, dim=1)).sum()
            num+=len(y)
    print(f"acc: {correct/num}")



if __name__ == '__main__':
    train()
    model_test()
