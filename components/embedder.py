
import torch
from torch import nn


class Embedding(nn.Module):

    def __init__(self, vocab_size: int, model_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = nn.Parameter(torch.empty(vocab_size, model_dim))
        nn.init.normal_(self.embeddings, mean=0, std=1)

    def forward(self, x):
        return self.embeddings[x]
