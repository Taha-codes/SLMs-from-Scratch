import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int): #where we define the layers and parameters
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        #pre conputing the positional matrix
        pe = torch.zeros(seq_len, d_model)

        positions = torch.arange(0, seq_len).unsqueeze(1)        # (seq_len, 1)
        div_term  = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))    # (d_model/2,)
        pe[:, 0::2] = torch.sin(positions * div_term)  # even dims
        pe[:, 1::2] = torch.cos(positions * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model) ← add batch dim

        self.register_buffer('pe', pe)  # fixed, not a learned parameter
    
    def forward(x, self):
        x = x + self.pe[:, :x.shape[1], :]  # add position to each token
        return self.dropout(x)

