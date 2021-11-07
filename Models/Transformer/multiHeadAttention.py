import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead
        
        assert (
            d_model % nhead == 0
        ), "Embedding size (d_model) needs to be divisible by number of heads"
        
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, query, key, value, d_k, mask=None, dropout=None):
    
        scores = torch.matmul(query, key.transpose(-2, -1)) /  np.sqrt(d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -np.inf)
        
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, value)
        return output

    def forward(self, query, key, value, mask=None):
        I = query.shape[0]
        
        # perform linear operation and split into N heads
        query = self.q_linear(query).view(I, -1, self.nhead, self.d_k)
        key = self.k_linear(key).view(I, -1, self.nhead, self.d_k)
        value = self.v_linear(value).view(I, -1, self.nhead, self.d_k)
        
        # transpose to get dimensions I * nhead * J * d_k
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # calculate attention
        scores = self.attention(query, key, value, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(I, -1, self.d_model)
        output = self.out(concat)
    
        return output

