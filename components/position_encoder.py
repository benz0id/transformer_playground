import math
import torch
from torch import nn
from utils.log_utils import setup_logger

logger = setup_logger('position_encoder')

def get_positional_encoding(seq_len: int, model_dim: int) -> torch.tensor:
    """
    Generates a sinusoidal positional encoding.
    :param seq_len: The length of the sequence.
    :param model_dim: The model dimension.
    :return: A positional encoding
    """
    logger.debug(f"Generating positional encoding for seq_len={seq_len}, model_dim={model_dim}")
    
    encoding = torch.zeros(seq_len, model_dim, requires_grad=False)
    for pos in range(seq_len):
        for i in range(0, model_dim, 2):
            encoding[pos, i] = math.sin(pos / 10000 ** (i / model_dim))

        for i in range(1, model_dim, 2):
            encoding[pos, i] = math.cos(pos / 10000 ** (i / model_dim))

    logger.debug(f"Generated encoding of shape {encoding.shape}")
    return encoding

def get_positional_encoding_vectorized(seq_len: int, model_dim: int) -> torch.tensor:
    """
    Generates a sinusoidal positional encoding.
    :param seq_len: The length of the sequence.
    :param model_dim: The model dimension.
    :return: A positional encoding
    """
    logger.debug(f"Generating vectorized positional encoding for seq_len={seq_len}, model_dim={model_dim}")
    
    positions = torch.arange(0, seq_len).unsqueeze(-1)
    factors = (10000 ** (torch.arange(0, model_dim) / model_dim)).unsqueeze(0)

    vals = positions / factors

    vals[:, 0::2] = torch.sin(vals[:, 0::2])
    vals[:, 1::2] = torch.cos(vals[:, 1::2])

    vals.requires_grad = False
    logger.debug(f"Generated vectorized encoding of shape {vals.shape}")
    return vals

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 2000):
        super().__init__()
        logger.debug(f"Initializing PositionalEncoding with model_dim={model_dim}, max_len={max_len}")

        # Register as a buffer to become part of the model
        self.register_buffer('pe', get_positional_encoding(max_len, model_dim))
        logger.debug(f"Registered positional encoding buffer of shape {self.pe.shape}")

    def get_encoding(self, seq_len: int):
        """
        Gets position encodings.
        :param seq_len: Length of the given sequence.
        :return:
            (seq_len, model_dim)
            float32
        """
        logger.debug(f"Retrieving encoding for sequence length {seq_len}")
        encoding = self.pe[:seq_len,:]
        logger.debug(f"Retrieved encoding of shape {encoding.shape}")
        return encoding