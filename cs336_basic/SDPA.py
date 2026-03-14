import torch
import torch.nn as nn 
from einops import einsum 
from jaxtyping import Float,Bool 
from torch import Tensor 
from math import sqrt 
from .softmax import softmax
def ScaleDotProductAttention( Q: Float[Tensor, " ... queries d_k"],
                    K: Float[Tensor, " ... keys d_k"],
                    V: Float[Tensor, " ... values d_v"],
                    mask: Bool[Tensor, " ... queries keys"] | None = None,):
    d_k = Q.size()[-1]
    scores = einsum(K,Q,'... k d_k,... q d_k->... q k')/sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask,float('-inf')) #

    attn_weight = softmax(dim=-1)(scores)
    return einsum(attn_weight,V,'... q k,... k d ->... q d')

    

    

