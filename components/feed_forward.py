import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.project_up = nn.Linear(model_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project_down = nn.Linear(hidden_dim, model_dim)

    def forward(self, x: torch.Tensor):
        """
        Apply a simple two-layer neural network to x.
        :param x:
            (batch_size, seq_len, model_dim)
            float32
        :return:
            (batch_size, seq_len, model_dim)
            float32
        """
        embedding = self.project_up(x)
        embedding = self.act(embedding)
        embedding = self.dropout(embedding)
        return self.project_down(embedding)