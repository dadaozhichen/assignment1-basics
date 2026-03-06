import torch
import torch.nn as nn
from einops import einsum
import math 

class Linear(nn.Module):
    def __init__(self,in_feature,out_feature,
                 weights:torch.Tensor|None=None,device:torch.device|None = None,dtype:torch.dtype|None = None):
        super().__init__()
        if weights is not None:
            self.weights = nn.Parameter(weights)
        else:
            self.weights = nn.Parameter(torch.zeros(in_feature,out_feature))
            sigma = 2/(in_feature+out_feature)
            nn.init.trunc_normal_(self.weights,0,sigma,-math.sqrt(sigma),math.sqrt(sigma))
    
    def forward(self,x:torch.Tensor):
        return einsum(self.weights,x,"d_in d_out , ... d_in -> ... d_out")