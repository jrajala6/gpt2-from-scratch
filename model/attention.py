import torch 
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key =  nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value =  nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        batches, context_length, d_in = x.shape # (B, T, C)

        q = self.query(x) # (batches, context_length, d_in) --> (batches, context_length, d_out)
        k = self.key(x) # (batches, context_length, d_in) --> (batches, context_length, d_out)
        v = self.value(x) # (batches, context_length, d_in) --> (batches, context_length, d_out)

        attention_scores = q @ k.transpose(-2, -1) # (batches, context_length, d_out) @ (batched, d_out, context_length) = (B, T, T)
        attention_scores = attention_scores.masked_fill(self.mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores / q.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights) # only during training 

        context_vector = attention_weights @ v # (B, T, T) @ (B, T, d_out) --> (B, T, d_out): weighted sum given attention weight information 
        return context_vector

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.key =  nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value =  nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        batches, context_length, d_in = x.shape # (B, T, C)

        q = self.query(x) # (batches, context_length, d_in) --> (batches, context_length, d_out)
        k = self.key(x) # (batches, context_length, d_in) --> (batches, context_length, d_out)
        v = self.value(x) # (batches, context_length, d_in) --> (batches, context_length, d_out)

        q = q.view(batches, context_length, self.n_heads, self.head_dim) # (batches, context_length, n_heads, head_dim)
        k = k.view(batches, context_length, self.n_heads, self.head_dim) # (batches, context_length, n_heads, head_dim)
        v = v.view(batches, context_length, self.n_heads, self.head_dim) # (batches, context_length, n_heads, head_dim)

        q = q.transpose(1, 2) # (batches, n_heads, context_length, head_dim)
        k = k.transpose(1, 2) # (batches, n_heads, context_length, head_dim)
        v = v.transpose(1, 2) # (batches, n_heads, context_length, head_dim)

        attention_scores = q @ k.transpose(-2, -1) # (batches, n_heads, context_length, head_dim) @ (batches, n_heads, head_dim, context_length) --> (batches, n_heads, context_length, context_length)
        attention_scores = attention_scores.masked_fill(self.mask[:context_length , :context_length] == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores / q.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights) # only during training 

        context_vector = attention_weights @ v # (batches, n_heads, context_length, context_length) @ (batches, n_heads, context_length, head_dim) ->  (batches, n_heads, context_length, head_dim)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batches, context_length, -1)
        context_vector = self.out_proj(context_vector)
        return context_vector


"""
Inneficient Method
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, n_heads, dropout, qkv_bias=False):
        h_len = d_out // n_heads
        self.n_heads = n_heads
        self.heads = nn.ModuleList(
            [Attention(d_in, h_len, context_length, dropout, qkv_bias) for _ in range(n_heads)]
        )


    def forward(self, x):
        # head(x) --> (B, T, h_len) * n_head = (B, T, d_out)
        return torch.cat([head(x) for head in self.heads], dim=-1) 
"""
        
