# kuzu.py
# COMP9444, CSE, UNSW

# Student Name: Raymond Lu
# Student Number: z5277884
# submission date: July 12th

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        # The grayscale image resolution is 28x28
        self.linear = torch.nn.Linear(28*28,10)

    def forward(self,x):
        x = x.view(x.size(0),-1)
        y1 = self.linear(x)
        y_pred = F.log_softmax(y1,dim=0)
        return (y_pred)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.l1 = torch.nn.Linear(28*28,260)
        self.l2 = torch.nn.Linear(260,10)

    def forward(self, x):
        # return 0 # CHANGE CODE HERE
        x = x.view(x.size(0),-1)
        y1 = self.l1(x) 
        hid1 = F.tanh(y1) 
        y2 = self.l2(hid1)
        output = F.log_softmax(y2,dim=0) 
        return(output)

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.l1 = nn.Linear(2048, 256)
        self.l2 = nn.Linear(256,10)

    def forward(self, x):
        in_size = x.size(0)
        hid1 = F.relu(self.mp(self.conv1(x)))
        hid2 = F.relu(self.mp(self.conv2(hid1)))
        hid2 = hid2.view(in_size,-1)
        hid3 = F.relu(self.l1(hid2))
        output = F.log_softmax(self.l2(hid3),dim=0)
        return (output)
