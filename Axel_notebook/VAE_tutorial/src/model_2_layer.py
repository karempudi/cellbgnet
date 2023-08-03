import torch
import torch.nn as nn
import torch.nn.functional as F

# https://debuggercafe.com/getting-started-with-variational-autoencoders-using-pytorch/ 

# define a simple linear VAE
class simple_nn(nn.Module):
    def __init__(self, in_features = 2, out_features = 2):
        super(simple_nn, self).__init__()
        print("Hej")

        self.features = in_features
        self.out_features = out_features
        # encoder
        self.ll1 = nn.Linear(in_features=in_features, out_features=10)
        
        self.ll2 = nn.Linear(in_features=10, out_features=out_features)
        # decoder 
  
    def forward(self, x):
        # encoding
        
        x = F.relu(self.ll1(x))
        out = self.ll2(x)
        
        return out
    
class bigger_nn(nn.Module):
    def __init__(self, in_features = 2, out_features = 2):
        super(bigger_nn, self).__init__()
        print("Hej")

        self.features = in_features
        self.out_features = out_features
        self.ll1 = nn.Linear(in_features=in_features, out_features=100)
        
        self.ll2 = nn.Linear(in_features=100, out_features=1000)
        self.ll3 = nn.Linear(in_features=1000, out_features=100)
        self.ll4 = nn.Linear(in_features=100, out_features=10)
        self.ll5 = nn.Linear(in_features=10, out_features=out_features)
        
    def forward(self, x):
        # encoding
        
        x = F.sigmoid(self.ll1(x))
        x = F.sigmoid(self.ll2(x))
        x = F.sigmoid(self.ll3(x))
        x = F.sigmoid(self.ll4(x))
        out = self.ll5(x)
        
        return out