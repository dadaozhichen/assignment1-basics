import torch
import torch.nn as nn
from einops import rearrange
from .linear import Linear

class RoPE(nn.Module):
    def __init__(self, theta:float,dim:int,max_seq_len:int) -> None:
        super().__init__()
        self.base = theta 
        self.dim = dim 
        inv_freq = 1.0/(self.base**(torch.arange(0,dim,2)/self.dim))
        self.register_buffer("inv_freq",inv_freq,persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self,seq_len):
        self.max_seq_len_cache = seq_len
        assert isinstance(self.inv_freq,torch.Tensor)
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t,self.inv_freq) #[max_seq_len,dim/2]
        emb = torch.cat((freqs,freqs),dim=-1) #[max_seq_len,dim]
        self.register_buffer("cos_cached",emb.cos(),persistent=False)
        self.register_buffer("sin_cache",emb.sin(),persistent=False) 
    
    def forward(self,x):



class transformer(nn.Module):
    def __init__(self, d_model,num_heads,d_ff,max_seq_len,theta) -> None:
        super().__init__()
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_ff = d_ff 
        self.Wk = Linear(d_model,d_model)
        self.Wq = Linear(d_model,d_model)
        self.Wv = Linear(d_model,d_model)
        self.rope = RoPE(theta,d_model,max_seq_len)
    def forward(self,x):
        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)
        token_position = torch.arange(self.d_model) 

        K = self.rope(K,token_position)