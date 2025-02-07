import torch
from torch import nn
from utils.log_utils import setup_logger

logger = setup_logger('layer_norm')

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
        
        logger.debug(f"Initialized LayerNorm with model_dim={model_dim}, eps={eps}")

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
        logger.debug(f"Input tensor shape: {x.shape}")
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)  # Do not use Bessel's correction.
        
        logger.debug(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
        
        # Check for potential numerical instability
        if torch.any(std < self.eps):
            logger.warning("Very small standard deviation detected, potential numerical instability")
        
        normalized = self.gamma * (x - mean) / (std + self.eps) + self.beta
        logger.debug(f"Output tensor shape: {normalized.shape}")
        
        return normalized