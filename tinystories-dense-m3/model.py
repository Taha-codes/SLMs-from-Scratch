import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    # init builds the tools
    def __init__(self, d_embd, head_size, block_size, dropout):
        # init parameters are instructions for building things
        # They are temporary values only needed during construction
        super().__init__()

        # nn.Linear(in_features, out_features)
        # nn.Linear is the PyTorch version of multiply input by a  learned weight matrix

        #self.x attributes are things you keep and use later

        self.key = nn.Linear(d_embd, head_size, bias = False)    #key weight matrix
        self.query = nn.Linear(d_embd, head_size, bias = False)  #query weight matrix
        self.value = nn.Linear(d_embd, head_size, bias = False)  #value weight matrix
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'tril', 
            torch.tril(torch.ones(block_size, block_size))
        )
    
    def forward(self, x):
        # our input matrix which is the (B, T, C)
        B, T, C = x.shape

        k = self.key(x)     #(B, T, head_size)
        q = self.query(x)   #(B, T, head_size)
        v = self.value(x)   #(B, T, head_size)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5) #(B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, d_embd, block_size, dropout):
        super().__init__()

        self.Head = nn.ModuleList()
        self.W_o = nn.Linear(d_embd, d_embd)
        self.dropout = nn.Dropout(dropout)