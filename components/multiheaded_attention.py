import math
import torch
from torch import nn
from utils.log_utils import setup_logger

logger = setup_logger('multiheaded_attention')

class MultiHeadedAttention(nn.Module):
    """
    A multiheaded attention module.
    """
    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            dropout: float
    ):
        super().__init__()

        assert model_dim % num_heads == 0

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attention_dim = model_dim // num_heads

        self.keys_matrix = nn.Linear(model_dim, model_dim)
        self.queries_matrix = nn.Linear(model_dim, model_dim)
        self.values_matrix = nn.Linear(model_dim, model_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        logger.debug(f"Initialized MultiHeadedAttention with model_dim={model_dim}, "
                   f"num_heads={num_heads}, dropout={dropout}")

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        :param x:
            (batch_size, seq_len, model_dim)
            float32
        :param mask:
            (batch_size, seq_len, seq_len)
            bool
        :return:
            (batch_size, seq_len, model_dim)
            float32
        """
        batch_size, seq_length, model_dim = x.shape
        logger.debug(f"Input tensor shape: {x.shape}, mask shape: {mask.shape}")

        # Apply matrices to convert tokens into keys, queries, and values
        keys = self.keys_matrix(x)
        queries = self.queries_matrix(x)
        values = self.values_matrix(x)
        logger.debug(f"Generated keys, queries, values of shape {keys.shape}")

        # Partition keys, queries, and values across attention heads
        keys = keys.view(batch_size, seq_length, self.num_heads, model_dim // self.num_heads)
        queries = queries.view(batch_size, seq_length, self.num_heads, model_dim // self.num_heads)
        values = values.view(batch_size, seq_length, self.num_heads, model_dim // self.num_heads)
        logger.debug(f"Reshaped for attention heads: {keys.shape}")

        # Reshape to form matrix for each head
        keys = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        logger.debug(f"Permuted shapes: {keys.shape}")

        # Compute attention between keys and queries
        attention_weights = queries @ keys.transpose(-2, -1)
        logger.debug(f"Attention weights shape: {attention_weights.shape}")
        del keys, queries

        # Scale by attention dimension
        attention_weights = attention_weights / math.sqrt(self.attention_dim)

        # Apply mask - enforcing directionality
        mask = mask.unsqueeze(1)
        attention_weights.masked_fill_(mask, float('-inf'))

        # Apply softmax within each row of the attention matrix

        # Padded rows will all be NaN - So fix them
        attention_weights = attention_weights.masked_fill_(mask, 0)
        attention_weights = attention_weights.softmax(-1)

        # Check for numerical instabilities
        if torch.any(torch.isnan(attention_weights)):
            logger.warning("NaN values detected in attention weights")

        # Apply dropout to softmaxed attention
        attention_weights = self.attn_dropout(attention_weights)

        # Compute outputs as weighted sums of the values
        output = attention_weights @ values
        del attention_weights
        logger.debug(f"Output after attention: {output.shape}")

        # Merge attention heads
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(batch_size, seq_length, model_dim)
        output = self.out_dropout(output)
        logger.debug(f"Final output shape: {output.shape}")

        return output