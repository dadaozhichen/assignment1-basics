import torch
import torch.nn as nn 
from einops import einsum 

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int
                 ,weights:torch.Tensor|None=None,device:torch.device|None=None,dtype:torch.dtype|None=None):
        super().__init__()
        if weights is not None:
            self.weights = nn.Parameter(weights)
        else:
            self.weights = nn.Parameter(torch.zeros(num_embeddings,embedding_dim))
            nn.init.trunc_normal_(self.weights,0,1,-3,3)

    def forward(self,token_ids:torch.Tensor):
        return self.weights[token_ids]
        #return einsum(self.weights,token_ids,"d_in d_out , ... d_in -> ... d_out")