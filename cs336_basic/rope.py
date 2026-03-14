import torch
import torch.nn as nn 


class RoPE(nn.Module):
    def __init__(self,theta: float, d_k: int, max_seq_len: int|None = None, device=None):
        super().__init__()
        half_d = d_k//2
        dim_range = torch.arange(half_d,dtype=torch.float32)
        self.freqs = 1.0/(theta**(dim_range/half_d))
    
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        pos = token_positions.to(torch.float32).unsqueeze(-1)
        angle = pos*self.freqs
        cos = angle.cos()
        sin = angle.sin()
        x1 = x[...,0::2]
        x2 = x[...,1::2]
        rope_x1 = x1*cos - x2*sin
        rope_x2 = x1*sin + x2*cos
        rope_out = torch.empty_like(x)
        rope_out[...,0::2] = rope_x1
        rope_out[...,1::2] = rope_x2
        return rope_out

