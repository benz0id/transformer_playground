import math

import torch
from torch import nn


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

        # Apply matricies to convert tokens into keys, queries, and values.
        keys =    self.keys_matrix(x)
        queries = self.queries_matrix(x)
        values =  self.values_matrix(x)

        # Partition keys, queries, and values across attention heads. See *1
        keys    = keys.view(batch_size, seq_length, self.num_heads, model_dim // self.num_heads)
        queries = queries.view(batch_size, seq_length, self.num_heads, model_dim // self.num_heads)
        values  = values.view(batch_size, seq_length, self.num_heads, model_dim // self.num_heads)

        # Reshape to form matrix for each head. See *1
        keys    = keys.permute(0, 2, 1, 3)
        queries = queries.permute(0, 2, 1, 3)
        values  = values.permute(0, 2, 1, 3)

        # Compute attention between keys and queries.
        attention_weights = queries @ keys.transpose(-2, -1) # (batch_size, num_head, seq_len, seq_len)
        del keys, queries

        # Scale by attention dimension.
        attention_weights = attention_weights / math.sqrt(self.attention_dim)

        # Apply mask - enforcing directionality.
        mask = mask.unsqueeze(1)
        attention_weights.masked_fill_(mask, float('-inf'))

        # Apply softmax within each row of the attention matrix.
        attention_weights = attention_weights.softmax(-1)

        # Padded rows will all be NaN - So fix them
        attention_weights = attention_weights.masked_fill_(mask, 0)

        # Apply dropout to softmaxed attention to reduce reliance on certain attention patterns.
        attention_weights = self.attn_dropout(attention_weights)

        # Compute outputs as weighted sums of the values
        output = attention_weights @ values # (batch_size, num_head, seq_len, attention_dim)

        # Merge attention heads.
        output = output.permute(0, 2, 1, 3)
        output = output.view(batch_size, seq_length, model_dim)
        output = self.out_dropout(output)

        return output

""" 

=== Notes ===

*1

Reshape/view changes the shape of the tensor but maintains the order of the values (fill lowest dim first).
e.g. in a two dim matrix, order is left to right, top to bottom.

x = torch.tensor([[1, 2, 3],
          [4, 5, 6]])  # 2x3

y = x.reshape(3, 2)  # Becomes:
# [[1, 2],
#  [3, 4],
#  [5, 6]]
"""



