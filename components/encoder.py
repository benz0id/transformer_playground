from pathlib import Path
from typing import List, Tuple

import torch

# Your encoding function as provided
AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3,
    'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15,
    'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': None,
}

"""
     === NOTES ===
Initial pass at the encoder and data loading helpers.

Future optimizations
- Vectorizing some of the encoding and mask creation operations.
- Writing code to precompute encoded tensors.
- Ensuring that we pass sequences of roughly similar length through in a single batch.
"""

def encode_protein_sequence(sequence: str, target_length: int):
    """
    Encode a protein sequence
    :param sequence: The sequence to be encoded.
    :param target_length: Determines amount of padding xor truncation.
    :return:
        one_hot
            (max_seq_len, 20)
            float32
            One hot encoded amino acid sequence, padded xor truncated to target_length.
        padding_mask
            (max_seq_len)
            bool
            Mask indicating location of padding.
        padding_mask_2d
            (max_seq_len, max_seq_len)
            bool
            Same as padding mask, but expanded to 2d.
            e.g
            [False, False, True] ->
            [[False, False, True],
             [False, False, True],
             [True,  True,  True]
        attention_mask
            (max_seq_len, max_seq_len)
            bool
            Upper triangular mask combined with padding_mask_2d.
    """
    # QUESTION You could save some memory by switching to an index rep, but would lose
    # potential to account for unknown amino acids with even distribution.
    one_hot_rep = torch.zeros(
         target_length, 20,
         dtype=torch.float32,
         requires_grad=False
    )
    num_to_fill = min((target_length, len(sequence)))

    for i in range(num_to_fill):
         ind = AA_MAP[sequence[i]]
         # In the case of unknown amino acids, might it make sense to use a uniform dist across all AAs?
         if ind is None:
             one_hot_rep[i, :] = 1 / 20
         else:
             one_hot_rep[i, ind] = 1

    # 1 -> True -> Apply mask here. False -> no mask here
    padding_mask = torch.ones(
        target_length,
        dtype=torch.float32,
        requires_grad=False
    )
    padding_mask[:num_to_fill] = 0
    padding_mask = padding_mask.bool()

    # QUESTION This transposition may be entirely unnecessary.
    # Need to work out where I am masking in order to account for padding.
    padding_mask_2d = padding_mask.unsqueeze(-1).expand(-1, target_length)
    padding_mask_2d = padding_mask_2d | padding_mask_2d.T

    attention_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool()
    attention_mask = attention_mask | padding_mask_2d

    return one_hot_rep, padding_mask, padding_mask_2d, attention_mask


def collate_protein_sequences(seqs: List[str]) \
        -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Converts a list of amino acid sequences into a one-hot encoded batch matrix.
    Also creates the masks required for operation.
    :param seqs: A list of amino acid sequences.
    :return:
        batch_one_hot
            (batch_size, max_seq_len, 20)
            float32
            One hot encoded amino acid sequences, padded to length of the longest sequence.
        batch_padding_mask
            (batch_size, max_seq_len)
            bool
            Mask indicating location of padding.
        batch_padding_mask_2d
            (batch_size, max_seq_len, max_seq_len)
            bool
            Same as padding mask, but expanded to 2d for each sequence.
            e.g
            [False, False, True] ->
            [[False, False, True],
             [False, False, True],
             [True,  True,  True]
        batch_attention_mask
            (batch_size, max_seq_len, max_seq_len)
            bool
            Upper triangular mask combined with padding_mask_2d for each sequence respectively.

    *Note* A value of 'True' at any position in the mask denotes that the respective
    position will be masked. 'False' denotes that it will not be.
    """
    lens = [len(seq) for seq in seqs]
    batch_size = max(lens)

    batch_one_hot = []
    batch_padding_mask = []
    batch_padding_mask_2d = []
    batch_attention_mask = []

    for seq in seqs:
        one_hot_rep, padding_mask, padding_mask_2d, attention_mask = (
            encode_protein_sequence(seq, batch_size))

        batch_one_hot.append(one_hot_rep)
        batch_padding_mask.append(padding_mask)
        batch_padding_mask_2d.append(padding_mask_2d)
        batch_attention_mask.append(attention_mask)

    # Stack the tensors to create batch tensors
    batch_one_hot = torch.stack(batch_one_hot)
    batch_padding_mask = torch.stack(batch_padding_mask)
    batch_padding_mask_2d = torch.stack(batch_padding_mask_2d)
    batch_attention_mask = torch.stack(batch_attention_mask)

    return batch_one_hot, batch_padding_mask, batch_padding_mask_2d, batch_attention_mask




