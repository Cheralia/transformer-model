import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    block_size: int = 128     # Context window
    vocab_size: int = 50000    # Defined by tokenizer
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE).
    Solves entanglement by rotating Q/K vectors instead of adding positional vectors.
    """
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos[:seq_len, :], self.sin[:seq_len, :]

def apply_rotary_pos_emb(x, cos, sin):
    # x: [batch, seq_len, head_dim] -> we need to broadcast cos/sin
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    # rotate_half implementation
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_embd = config.n_embd  # <--- FIX 1: Save n_embd to self
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # RoPE module (The Fix)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        
        # <--- FIX 2: Use self.n_embd instead of config.n_embd
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape to (B, T, n_head, head_dim) for RoPE application
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # v goes straight to (B, H, T, D)

        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(T)
        
        q = q.transpose(1, 2) # (B, H, T, D)
        k = k.transpose(1, 2) # (B, H, T, D)
        
        # Adjust dimensions for the apply function which expects (..., T, D)
        q_flat = q.reshape(B * self.n_head, T, self.head_dim)
        k_flat = k.reshape(B * self.n_head, T, self.head_dim)
        
        q_rot = apply_rotary_pos_emb(q_flat, cos, sin).view(B, self.n_head, T, self.head_dim)
        k_rot = apply_rotary_pos_emb(k_flat, cos, sin).view(B, self.n_head, T, self.head_dim)

        # Attention
        att = (q_rot @ k_rot.transpose(-2, -1)) * (1.0 / math.sqrt(k_rot.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DisentangledTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Word Embeddings ONLY (No additive pos embeddings)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        x = self.blocks(x)
        x = self.ln_f(x)

        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss