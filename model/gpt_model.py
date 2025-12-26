import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .transformer_block import TransformerBlock
from .transformer_layers import LayerNorm

class GPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # get final logits

    def forward(self, x):
        batch_size, seq_len = x.shape # (B, T)
        tok_emb = self.tok_emb(x) # (B, T, C)
        pos_emb = self.pos_emb(torch.arange(seq_len))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


