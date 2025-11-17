import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, p, d_model=128, nhead=4, d_ff=256, nlayers=2, dropout=0.0):
        super().__init__()
        self.p = p
        self.seq_len = 3
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=d_ff, dropout=dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p)

    def forward(self, x):                      # x: (B,3) long
        h = self.tok(x) + self.pos            # (B,3,D)
        h = self.enc(h)                       # (B,3,D)
        cls = self.ln(h[:, -1, :])            # use CLS position (index 2)
        logits = self.head(cls)               # (B,p)
        return logits