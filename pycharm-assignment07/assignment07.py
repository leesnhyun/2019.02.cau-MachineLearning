from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# import packages
# -----------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision

from torchvision import datasets, transforms
from torch.autograd import Variable
from random import shuffle

import argparse
import sys
import os
import numpy as np
import time
import datetime
import csv
import configparser
import argparse
import platform


IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100
IMAGE_CHANNEL = 1
DIMENSION = IMAGE_CHANNEL * IMAGE_HEIGHT * IMAGE_WIDTH

# global settings
# torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type('torch.cuda.DoubleTensor')
torch.set_printoptions(precision=16)
torch.cuda.set_device(0)

# setting check
print("current device : %s" % (torch.cuda.current_device()))
print("device count : %s" % (torch.cuda.device_count()))
print("device name : %s" % (torch.cuda.get_device_name(0)))
print("CUDA available? : %s" % (torch.cuda.is_available()))

batch_size = 300

class Linear(nn.Module):

    def __init__(self, num_classes=2):

        super(Linear, self).__init__()

        self.number_class = num_classes

        _size_image = 100 * 100
        _num1 = 100
        _num2 = 150

        self.fc1 = nn.Linear(_size_image, _num1, bias=True)
        self.fc2 = nn.Linear(_num1, _num2, bias=True)
        self.fc3 = nn.Linear(_num2, num_classes, bias=True)

        self.fc_layer1 = nn.Sequential(self.fc1, nn.ReLU(True))
        self.fc_layer2 = nn.Sequential(self.fc2, nn.ReLU(True))
        self.fc_layer3 = nn.Sequential(self.fc3, nn.Sigmoid())

        self.classifier = nn.Sequential(self.fc_layer1, self.fc_layer2, self.fc_layer3)

        self._initialize_weight()

    def _initialize_weight(self):
        for name, m in self._modules.items():
            if isinstance(m, nn.Linear):
                n = m.in_features
                m.weight.data.uniform_(- 1.0 / math.sqrt(n), 1.0 / math.sqrt(n))

                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

# -----------------------------------------------------------------------------
# load dataset
# -----------------------------------------------------------------------------

transform = transforms.Compose([  # transforms.Resize((256,256)),
    transforms.Grayscale(),
    # the code transforms.Graysclae() is for changing the size [3,100,100] to [1, 100, 100] (notice : [channel, height, width] )
    transforms.ToTensor(), ])

# train_data_path = 'relative path of training data set'
# change the valuse of batch_size, num_workers for your program
# if shuffle=True, the data reshuffled at every epoch
train_data_path = './horse-or-human/train'
validation_data_path = './horse-or-human/validation'

set_train = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
set_test = torchvision.datasets.ImageFolder(root=validation_data_path, transform=transform)

loader_train = torch.utils.data.DataLoader(
    dataset=set_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=5
)

loader_test = torch.utils.data.DataLoader(
    dataset=set_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=5
)

num_classes = 2

# -----------------------------------------------------------------------------
# load neural network model
# -----------------------------------------------------------------------------

model = Linear(num_classes=num_classes)

# -----------------------------------------------------------------------------
# Set the flag for using cuda
# -----------------------------------------------------------------------------
model.cuda()

# torch.backends.cudnn.benchmark = True
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -----------------------------------------------------------------------------
# optimization algorithm
# -----------------------------------------------------------------------------
learning_rate = 0.05
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=10e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=0.001, verbose=True)
objective = nn.CrossEntropyLoss()


# -----------------------------------------------------------------------------
# function for training the model
# -----------------------------------------------------------------------------

def train():
    # print('train the model at given epoch')

    loss_train = []

    model.train()

    for idx_batch, (data, target) in enumerate(loader_train):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).cuda(), Variable(target).cuda()

        optimizer.zero_grad()

        output = model(data)
        loss = objective(output, target)

        loss.backward()
        optimizer.step()

        loss_train_batch = loss.item() / len(data)
        loss_train.append(loss_train_batch)

    loss_train_mean = np.mean(loss_train)
    loss_train_std = np.std(loss_train)

    return {'loss_train_mean': loss_train_mean, 'loss_train_std': loss_train_std}


# -----------------------------------------------------------------------------
# function for testing the model
# -----------------------------------------------------------------------------

def test():
    # print('test the model at given epoch')
    loss_test = 0
    correct = 0

    model.eval()

    for idx_batch, (data, target) in enumerate(loader_test):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).cuda(), Variable(target).cuda()

        output = model(data)
        loss = objective(output, target)

        loss_test += loss.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss_test = loss_test / len(loader_test.dataset)
    accuracy_test = 100. * float(correct) / len(loader_test.dataset)

    return {'loss_test': loss_test, 'accuracy_test': accuracy_test}


# -----------------------------------------------------------------------------
# iteration for the epoch
# -----------------------------------------------------------------------------

loss_train_mean, loss_train_std, loss_test, accuracy_test = [], [], [], []

for e in range(1000):
    result_train = train()
    result_test = test()

    loss_train_mean.append(result_train['loss_train_mean'])
    loss_train_std.append(result_train['loss_train_std'])
    loss_test.append(result_test['loss_test'])
    accuracy_test.append(result_test['accuracy_test'])

    scheduler.step(result_test['accuracy_test'], e)

    print(result_train)
    print(result_test)
