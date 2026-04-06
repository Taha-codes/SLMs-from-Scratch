import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int): #where we define the layers and parameters
        super().__init__()
        self.embedding = nn.Embedding()
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)


