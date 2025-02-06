import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Normalize the input token-wise, with learned weights and biases for each model dimension.
    """
    def __init__(self, model_dim: int, eps: float = 1e-5):
        super().__init__()

        # Used in multiplication - initialize to one.
        self.gamma = nn.Parameter(torch.ones(model_dim))

        # Used in addition - initialize to zero.
        self.beta = nn.Parameter(torch.zeros(model_dim))

        # Small number to prevent division by zero.
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalise the given layer (z-score) and then apply known weights and biases.
        :param x:
            (batch_size, seq_len, model_dim)
            float32
        :return:
            (batch_size, seq_len, model_dim)
            float32
        """
        mean = x.mean(-1, keepdim=True)
        # QUESTION whether or not to use Bessel's correction.
        std = x.std(-1, keepdim=True, unbiased=False) # Do not use Bessel's correction.
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


