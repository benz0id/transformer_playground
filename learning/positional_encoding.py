import numpy as np
import matplotlib.pyplot as plt

def visualize_positional_encoding(seq_len, d_model):
    """
    Visualizes the positional encodings used in Transformer models as a heatmap,
    with the axes flipped.

    Parameters:
        seq_len (int): The length of the sequence (number of positions).
        d_model (int): The dimension of the model (embedding size).

    This function computes the positional encodings as described in the paper
    "Attention Is All You Need" (Vaswani et al., 2017), and then displays
    them as a heatmap where the x-axis represents the sequence positions and the
    y-axis represents the embedding dimensions. The color intensity in the
    heatmap corresponds to the value of the positional encoding at that
    position and dimension.

    Example:
        visualize_positional_encoding(seq_len=50, d_model=512)
    """
    # Initialize the positional encoding matrix
    PE = np.zeros((seq_len, d_model))

    # Get the position indices (0 to seq_len-1)
    position = np.arange(seq_len)[:, np.newaxis]

    # Get the dimension indices (0 to d_model-1)
    i = np.arange(d_model)[np.newaxis, :]

    # Compute the angle rates
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    # Compute the angle radians
    angle_rads = position * angle_rates

    # Apply sin to even indices (0,2,4,...) and cos to odd indices (1,3,5,...)
    PE[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Even indices
    PE[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Odd indices

    # Transpose the positional encoding matrix to flip the axes
    PE_flipped = PE.T

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(PE_flipped, aspect='auto', cmap='viridis')
    plt.xlabel('Sequence Positions')
    plt.ylabel('Embedding Dimensions')
    plt.title('Positional Encoding Heatmap (Axes Flipped)')
    plt.colorbar(label='Encoding Value')
    plt.show()

if __name__ == '__main__':
    # Specify sequence length and model dimension
    sequence_length = 100
    model_dimension = 64

    # Visualize the positional encodings
    visualize_positional_encoding(seq_len=sequence_length, d_model=model_dimension)