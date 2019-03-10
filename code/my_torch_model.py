import torch.nn as nn
import torch.nn.functional as F

"""
This is a neural network consisting of 2 convolutional layers and 2 fully connected layers. Between every convolutional layer we apply a ReLU activation function, as well as 2d pooling. The tensor is then transformed to fit the shape of the fully connected layers. It then passes through the first fully connected layer followed by another ReLU, and finally the last fully connected layer. We apply a log_softmax, which results in a tensor with an estimation of the current input for each class. We will later take the maximum of all these estimates and use that as the classification result.
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)