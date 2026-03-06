import torch
import torch.nn as nn 
from einops import einsum,reduce
import math 

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,
                 weight:torch.Tensor|None = None,
                 device:torch.device|None = None,
                 dtype:torch.dtype|None = None):
        super().__init__()

        self.eps = eps
        self.d_model = d_model
        if weight is not None:
            self.weight = nn.Parameter(weight)
        else:
            self.weight = nn.Parameter(torch.randn(d_model))
            nn.init.trunc_normal_(self.weight)
        
        self.device = device
        self.dtype = dtype 

    def forward(self,x:torch.Tensor):
        #x:[batch_size,sequence_length,d_model]
        rms = reduce(x**2,'... d -> ... 1','mean') #[batch_size,sequence_length]
        rms = torch.sqrt(rms+self.eps)
        x_norm = x/rms 
        return einsum(x_norm,self.weight,'... d,d -> ... d')


