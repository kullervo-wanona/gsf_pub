from sys import platform
if 'linux' in platform: 
    # import IPython.core.debugger
    # trace = IPython.core.debugger.Pdb.set_trace() #this one triggers the debugger
    from IPython.core.debugger import set_trace
    trace = set_trace
else:
    import ipdb
    trace = ipdb.set_trace

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch

# from DataLoaders.CelebA.CelebA64Loader import DataLoader
# data_loader = DataLoader(batch_size=10)
# data_loader.setup('Training', randomized=True, verbose=False)
# data_loader.setup('Test', randomized=False, verbose=False)
# _, _, batch = next(data_loader)

from DataLoaders.MNIST.MNISTLoader import DataLoader
data_loader = DataLoader(batch_size=10)
data_loader.setup('Training', randomized=True, verbose=True)
data_loader.setup('Test', randomized=True, verbose=True)
# data_loader.setup('Validation', randomized=True, verbose=True)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16*4*4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
for i, curr_batch_size, batch_np in data_loader:     
    image_th = torch.from_numpy(batch_np['Image'])
    label_th = torch.from_numpy(batch_np['Label'])

    # zero the parameter gradients
    optimizer.zero_grad()

    outputs = net(image_th)
    loss = criterion(outputs, label_th)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 20 == 19:    # print every 2000 mini-batches
        print(f'[{0 + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')








    