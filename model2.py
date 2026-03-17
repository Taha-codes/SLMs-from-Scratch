import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Modlule):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size 
        self.embedding  = nn.Embedding(vocab_size, d_model) 

        # An embedding in pytorch is just a mapping between a number and a vector of dimension 512.
        # The embedding vector can be of any dimension, not necessarily 512.
        # I chose 512 just because I am following the implementation details of the paper.

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        # create a matrix of shape (sed_len, d_model)
        pe = torch.zeros(seq_len, d_model) 

        # create a vector that will represent the position of the word inside a sentence (seq_len)
        # Generate position indices
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #Tensor of shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Since we will have a batch of sentences
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps  = eps
        self.apha = nn.Parameter(torch.ones(1)) # We multiply alpha with the standardized x 
        self.bias = nn.Parameter(torch.zeros(1)) # After that we add this bias 

        def forward(self, x):
            mean = x.mean(dim = -1, keepdim = True)
            std  = x.std(dim  = -1, keepdim = True)
            return self.alpha * (x - mean) / (std * self.eps) + self.bias

class FeedForwardBlock(nn.Module): # the dimensions of this FF are: d_model -> d_ff = 2048 -> d_model
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, h: int):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Droupout(dropout)
        self.h       = h

        assert d_model % h == 0, "d_model must be divisible by the number of heads"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask):
        d_k = query.shape[-1]

        #(Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2. -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1) #(Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query   = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key     = self.w_k(k)
        value   = self.w_v(v)

        #(Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query   = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key     = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value   = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, attention_scores = self.attention(query, key, value, self.dropout, mask)

        #(Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm   = LayerNormalization()  
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_Attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_Attention_block
        self.feed_forward_block   = feed_forward_block
        self.residual_connections  = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x