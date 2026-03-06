import torch
import torch.nn as nn 
from einops import einsum

class SiLU(nn.Module):
    def __init__(self,d_model:int,
                 device : torch.device|None = None,
                 dtype : torch.dtype|None = None):
        super().__init__()
        self.activate = nn.Sigmoid()
        self.device = device
        self.dtype = dtype 
    def forward(self,x:torch.Tensor):
        return einsum(self.activate(x),x,'... d,... d->... d')
    

class SwiGLU(nn.Module):
    def __init__(self, d_model:int,d_ff:int|None = None,
                 weight1:torch.Tensor|None = None,
                 weight2:torch.Tensor|None = None,
                 weight3:torch.Tensor|None = None):
        super().__init__()
        self.activate =SiLU(d_model)
        self.d_ff = d_ff if d_ff is not None else int(d_model*8/3)
        if weight1 is not None:
            self.weight1 = nn.Parameter(weight1)
        else:
            self.weight1 = nn.Parameter(torch.randn(self.d_ff,d_model)) 
        if weight2 is not None:
            self.weight2 = nn.Parameter(weight2) 
        else:
            self.weight2 = nn.Parameter(torch.randn(self.d_ff,d_model)) 
        if weight3 is not None:
            self.weight3 = nn.Parameter(weight3)
        else:
            self.weight3 = nn.Parameter(torch.randn(d_model,self.d_ff))

    def forward(self,x:torch.Tensor):
        x1 = self.activate(einsum(self.weight1,x,'df d,... d->... df'))
        x3 = einsum(self.weight3,x,'df d,... d->... df')
        mul = einsum(x1,x3,'... df,... df->... df')
        return einsum(self.weight2,mul,'d df,... df->... d')

        
        

