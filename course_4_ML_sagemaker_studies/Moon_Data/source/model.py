import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # your code here:
        # defining layers:
        self.fc_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        self.act_2 = nn.Sigmoid()
        
        # initializing layers:
        self.fc_1.weight = nn.init.xavier_uniform_(self.fc_1.weight, gain=nn.init.calculate_gain('relu'))
        self.fc_1.bias = nn.init.zeros_(self.fc_1.bias)
        self.fc_2.weight = nn.init.xavier_uniform_(self.fc_2.weight, gain=1.0)
        self.fc_2.bias = nn.init.zeros_(self.fc_2.bias)
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code here:
        x = self.act_1(self.fc_1(x))
        x = self.act_2(self.fc_2(x))
        
        return x