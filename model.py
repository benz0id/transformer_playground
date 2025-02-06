import torch
from torch import nn, clone

from components.feed_forward import FeedForward
from components.layer_norm import LayerNorm
from components.multiheaded_attention import MultiHeadedAttention
from components.position_encoder import PositionalEncoding
from components.embedder import Embedding

# TODO tune dropout to specific layers

class Layer(nn.Module):
    """
    A single layer of the transformer, implementing pre layer normalization.

    Input → Layer Norm → Self-Attention → Add → Layer Norm → FFN → Add
    """
    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            dropout: float,
            feed_forward_hidden_dim: int
    ):
        super().__init__(self)

        self.atn_layer_norm = LayerNorm(model_dim)
        self.attention = MultiHeadedAttention(model_dim, num_heads, dropout)
        self.ff_layer_norm = LayerNorm(model_dim)
        self.feed_forward = FeedForward(model_dim, feed_forward_hidden_dim, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param x:
            (batch_size, seq_len, model_dim)
            float32
        :param attention_mask:
            (batch_size, seq_len, seq_len)
            bool
            True -> mask this position.
        :return:
        """
        x = x + self.attention(self.atn_layer_norm(x), attention_mask)
        x = x + self.feed_forward(self.ff_layer_norm(x))
        return x



class Model(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            model_dim: int,
            dropout: float,
            max_len: int,
            num_layers: int,
            num_heads: int,
            feed_forward_hidden_dim: int
    ):
        super().__init__(self)
        self.num_layers = num_layers

        self.embed = Embedding(vocab_size, model_dim)
        self.position_encode = PositionalEncoding(model_dim, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            Layer(model_dim, num_heads, dropout, feed_forward_hidden_dim)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(model_dim, vocab_size)


    def forward(self, x: torch.tensor, attention_mask: torch.tensor):
        """
        :param x:
            (batch_size, seq_len)
            float32
        :param attention_mask:
            (batch_size, seq_len, seq_len)
            bool
            True -> mask this position.
        :return: logits in target dictionary.
            (batch_size, seq_len, vocab_size)
            float32
                """

        batch_size, seq_len, model_dim = x.shape

        # Get embeddings.
        token_embeddings = self.embed(x)

        # Get position encoding.
        position_embeddings = self.position_encode.get_encoding(seq_len)

        # Combine to produce input embeddings.
        x = token_embeddings + position_embeddings

        # Pass thorough layers.
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Compute logits.
        logits = self.output_projection(x)

        return logits








