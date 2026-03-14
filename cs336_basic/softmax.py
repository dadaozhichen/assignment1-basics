import torch
import torch.nn as nn 


class softmax(nn.Module):
    def __init__(self,dim:int = -1) -> None:
        super().__init__()
        self.dim = dim 
    def forward(self,x:torch.Tensor):
        x_max = x.max(dim=self.dim,keepdim=True).values
        sum = (torch.exp(x-x_max)).sum(dim=self.dim,keepdim=True)
        ret = torch.exp(x-x_max)/sum 
        return ret 
