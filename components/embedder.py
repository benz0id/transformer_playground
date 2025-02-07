import torch
from torch import nn
from utils.log_utils import setup_logger

logger = setup_logger('embedder')

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = nn.Parameter(torch.empty(vocab_size, model_dim))
        nn.init.normal_(self.embeddings, mean=0, std=1)
        logger.debug(f"Initialized embeddings with vocab_size={vocab_size}, model_dim={model_dim}")

    def forward(self, x: torch.Tensor):
        """

        :param x:
            (batch_size, seq_len)
            float32
        :return:
            (batch_size, seq_len, model_dim)
            float32
        """
        logger.debug(f"Processing input tensor of shape {x.shape}")
        embeddings = self.embeddings[x]
        logger.debug(f"Generated embeddings of shape {embeddings.shape}")
        return embeddings