import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class ffn(nn.Module):
    def __init__(self, d_model, d_ff = 2048):
        super(ffn, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self,x):
        fc1 = F.relu(self.fc1(x))
        return self.fc2(fc1)
    



    

    