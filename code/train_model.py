"""
Train classification model for MNIST
"""
import json
import pickle
import numpy as np
import time

# New imports
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from my_torch_model import Net

# New function
def train(model, device, train_loader, optimizer, epoch):
    log_interval = 100
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train_model():
    # Measure training time
    start_time = time.time()

    # Setting up network
    print("Setting up Params...")
    device = torch.device("cpu")
    batch_size = 64
    epochs = 3
    learning_rate = 0.01
    momentum = 0.5
    print("done.")

    # Load training data
    print("Load training data...")
    train_data = np.load('./data/processed_train_data.npy')

    # Divide loaded data-set into data and labels
    labels = torch.Tensor(train_data[:, 0]).long()
    data = torch.Tensor(train_data[:, 1:].reshape([train_data.shape[0], 1, 28, 28]))
    torch_train_data = torch.utils.data.TensorDataset(data, labels)
    train_loader = torch.utils.data.DataLoader(torch_train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    print("done.")

    # Define SVM classifier and train model
    print("Training model...")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    print("done.")

    # Save model as pkl
    print("Save model and training time metric...")
    with open("./data/model.pkl", 'wb') as f:
        pickle.dump(model, f)

    # End training time measurement
    end_time = time.time()

    # Create metric for model training time
    with open('./metrics/train_metric.json', 'w') as f:
        json.dump({'training_time': end_time - start_time}, f)
    print("done.")


if __name__ == '__main__':
    train_model()