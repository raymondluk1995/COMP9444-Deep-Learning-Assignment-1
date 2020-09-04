# spiral.py
# COMP9444, CSE, UNSW

# Student Name: Raymond Lu
# Student Number: z5277884
# submission date: July 12th

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self,num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(2,num_hid)
        self.l2 = nn.Linear(num_hid,1)
        self.hid1 = None 

    def forward(self,input):
        r = torch.norm(input,2,dim=1,keepdim=True)
        a = torch.atan2(input[:,1],input[:,0]).unsqueeze(1)
        i = torch.cat((r,a),1)
        self.hid1 = torch.tanh(self.l1(i))
        output = torch.sigmoid(self.l2(self.hid1))
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(2,num_hid) 
        self.l2 = nn.Linear(num_hid,num_hid) 
        self.l3 = nn.Linear(num_hid,1) 
        self.hid1 = None
        self.hid2 = None

    def forward(self, input):
        inpToHid1 = self.l1(input) 
        self.hid1 = torch.tanh(inpToHid1) 
        hid1ToHid2 = self.l2(self.hid1) 
        self.hid2 = torch.tanh(hid1ToHid2)  
        hid2ToOut = self.l3(self.hid2) 
        output = torch.sigmoid(hid2ToOut) 
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE
        self.lin_inp_hid1 = nn.Linear(2,num_hid) # lin stands for linear
        self.lin_inp_hid2 = nn.Linear(2,num_hid) 
        self.lin_inp_out  = nn.Linear(2,1) 
        self.lin_hid1_hid2 = nn.Linear(num_hid,num_hid) 
        self.lin_hid1_out = nn.Linear(num_hid,1) 
        self.lin_hid2_out = nn.Linear(num_hid,1)
        self.hid1 = None 
        self.hid2 = None 

    def forward(self, input):
        x_inpToHid1 = self.lin_inp_hid1(input) 
        x_inpToHid2 = self.lin_inp_hid2(input) 
        x_inpToOut = self.lin_inp_out(input) 
        self.hid1 = torch.tanh(x_inpToHid1) 
        x_inputOfHid2 = x_inpToHid2 + self.lin_hid1_hid2(self.hid1) 
        self.hid2 = torch.tanh(x_inputOfHid2)      
        x_inputOfOut = x_inpToOut + self.lin_hid1_out(self.hid1) + self.lin_hid2_out(self.hid2)
        output = torch.sigmoid(x_inputOfOut)
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad():
        net.eval() 
        ouput = net(grid) 
        if (net=="polar" and layer==2):
            raise ValueError("Polar Net does not have layer 2!")
        if(layer!=1 and layer!=2):
           raise ValueError("The value of layer should be 1 or 2!") 
        # In hidden layers, the output of tanh() is from -1 to 1. Therefore, value 0,
        # as the middle point, is the threshold.
        if(layer==1):
            pred = (net.hid1[:,node]>=0).float()
        else:
            pred = (net.hid2[:,node]>=0).float()
    
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')

