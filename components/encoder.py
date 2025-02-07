from typing import List, Tuple
import torch
from utils.log_utils import setup_logger

logger = setup_logger('encoder')

AA_MAP = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3,
    'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11,
    'P': 12, 'Q': 13, 'R': 14, 'S': 15,
    'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': None,
}

def encode_protein_sequence(
        sequence: str,
        target_length: int,
        truncate_mask: int = 0
):
    """
    Encode a protein sequence
    :param sequence: The sequence to be encoded.
    :param target_length: Determines amount of padding xor truncation.
    :param truncate_mask: Number of bases to prune from the end of the mask.
    :return:
        ind_rep
            (max_seq_len)
            int64
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
    logger.debug(f"Encoding sequence of length {len(sequence)} to target length {target_length}")
    
    ind_rep = torch.zeros(
         target_length,
         dtype=torch.int64,
         requires_grad=False
    )
    num_to_fill = min((target_length, len(sequence)))
    logger.debug(f"Will fill {num_to_fill} positions")

    unknown_aa_count = 0
    for i in range(num_to_fill):
         ind = AA_MAP[sequence[i]]
         if ind is None:
             ind_rep[i] = 20
             unknown_aa_count += 1
         else:
             ind_rep[i] = ind

    # if unknown_aa_count > 0:
        # logger.info(f"Encountered {unknown_aa_count} unknown amino acids in sequence")

    mask_len = target_length - truncate_mask
    padding_mask = torch.ones(
        mask_len,
        dtype=torch.float32,
        requires_grad=False
    )
    padding_mask[:num_to_fill] = 0
    padding_mask = padding_mask.bool()

    padding_mask_2d = padding_mask.unsqueeze(-1).expand(-1, mask_len)
    padding_mask_2d = padding_mask_2d | padding_mask_2d.T

    attention_mask = torch.triu(torch.ones(mask_len, mask_len), diagonal=1).bool()
    attention_mask = attention_mask | padding_mask_2d

    # logger.debug(f"Generated encodings and masks of shape {ind_rep.shape}")
    return ind_rep, padding_mask, padding_mask_2d, attention_mask

def collate_protein_sequences(seqs: List[str]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of amino acid sequences into a one-hot encoded batch matrix.
    Also creates the masks required for operation.
    :param seqs: A list of amino acid sequences.
    :return:
        batch_one_hot_x
            (batch_size, max_seq_len)
            int64
            One hot encoded amino acid sequences, padded to length of the longest sequence.
            Rightmost amino acid pruned.
        batch_one_hot_y
            (batch_size, max_seq_len, 20)
            int64
            Leftmost amino acid pruned.
        batch_padding_mask !DROPPED!
            (batch_size, max_seq_len)
            bool
            Mask indicating location of padding.
        batch_padding_mask_2d !DROPPED!
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
    """
    # logger.info(f"Collating batch of {len(seqs)} sequences")
    lens = [len(seq) for seq in seqs]
    batch_size = max(lens)
    # logger.debug(f"Max sequence length in batch: {batch_size}")

    batch_indices_x = []
    batch_indices_y = []
    # batch_padding_mask = []
    # batch_padding_mask_2d = []
    batch_attention_mask = []

    for i, seq in enumerate(seqs):
        # logger.debug(f"Processing sequence {i+1}/{len(seqs)}")
        one_hot_rep, padding_mask, padding_mask_2d, attention_mask = (
            encode_protein_sequence(seq, batch_size, truncate_mask=1))

        batch_indices_x.append(one_hot_rep[:-1,])
        batch_indices_y.append(one_hot_rep[1:,])
        # batch_padding_mask.append(padding_mask)
        # batch_padding_mask_2d.append(padding_mask_2d)
        batch_attention_mask.append(attention_mask)

    # Stack the tensors to create batch tensors
    batch_indices_x = torch.stack(batch_indices_x)
    batch_indices_y = torch.stack(batch_indices_y)
    # batch_padding_mask = torch.stack(batch_padding_mask)
    # batch_padding_mask_2d = torch.stack(batch_padding_mask_2d)
    batch_attention_mask = torch.stack(batch_attention_mask)

    return batch_indices_x.to('cuda'), batch_indices_y.to('cuda'), batch_attention_mask.to('cuda')