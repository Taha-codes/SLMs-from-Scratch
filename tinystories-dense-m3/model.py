import torch
import torch.nn as nn
import math

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
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model) because we need to add the batch dim

        self.register_buffer('pe', pe)  # fixed, not a learned parameter
    
    def forward(x, self):
        x = x + self.pe[:, :x.shape[1], :]  # add position to each token
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "embedding dimension not divisible by number of heads"

        self.d_k = d_model // n_heads

    def forward(self, x, mask = None):

        B, T, _ = x.shape

        # first we project the input matrix X into the Q, K and V matrices
        # the input matrix X is (betch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # we split into h heads and swap the seq_len and n_heads dimensions for matrix multiply
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # calculate the scores
        scores = (Q @ K.transpose(-2, -1) / math.sqrt(self.d_k))

        # Apply mask if training
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax the scores to get the attention weights
        weights = torch.softmax(scores, dim=-1)

        # weighteed sum
        x = weights @ V

        #concatenate heads back together for W_o multiply
        x = x.transpose(1,2).contiguous().view(B, T, self.n_heads * self.d_k)
        
        return self.W_o(x)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        return self.layer2(x)

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.pre_norm1 = nn.LayerNorm(d_model)
        self.multi_head_att = MultiHeadAttention(d_model, n_heads, dropout)
        self.pre_norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        residual = x

        x = self.pre_norm1(x)
        x = self.multi_head_att(x, mask)
        x = self.dropout(x)

        x = x + residual

        residual = x

        x = self.pre_norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)

        x = x + residual

        return x

class GPT(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, 
                n_heads:int, d_ff:int, seq_len:int, num_layers:int, dropout: float=0.1):
        super().__init__()

        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, seq_len, dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.tok_emb(x)

        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x, self.mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits