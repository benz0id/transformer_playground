import torch
from torch import nn
from utils.log_utils import setup_logger

logger = setup_logger('feed_forward')

class FeedForward(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.project_up = nn.Linear(model_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project_down = nn.Linear(hidden_dim, model_dim)
        logger.debug(f"Initialized FeedForward with model_dim={model_dim}, hidden_dim={hidden_dim}, dropout={dropout}")

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
        logger.debug(f"Input tensor shape: {x.shape}")
        
        embedding = self.project_up(x)
        logger.debug(f"After project_up shape: {embedding.shape}")
        
        embedding = self.act(embedding)
        embedding = self.dropout(embedding)
        
        output = self.project_down(embedding)
        logger.debug(f"Output tensor shape: {output.shape}")
        
        return output