from pathlib import Path
from typing import List, Any, Iterator, Union

from torch.utils.data import Sampler
from tqdm import tqdm
import torch
from Bio import SeqIO


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            fasta_path: Path,
            verbose: bool = False,
            limit_dataset: Union[int, bool] = False):

        self._sequences = []
        self.lengths = []

        it = SeqIO.parse(fasta_path, 'fasta')
        if verbose:
            it = tqdm(
                it,
                desc='Parsing Sequences',
                mininterval=0.5
            )
            if limit_dataset:
                it.total = limit_dataset

        for i, record in enumerate(it):
            seq = str(record.seq)
            self._sequences.append(seq)
            self.lengths.append(seq)

            if limit_dataset and i > limit_dataset:
                break

    def __getitem__(self, item):
        return self._sequences[item]

    def __len__(self):
        return len(self._sequences)


class BucketingSampler(Sampler):
    """
    GPTo1 05-02-2025

    Sampler that groups sequences of similar lengths together to minimize padding.

    This sampler divides the dataset into chunks, sorts each chunk by sequence length
    to group similar lengths together, then divides sorted chunks into batches.

    This approach can help in reducing the amount of padding needed when batching sequences
    of varying lengths, which can improve computational efficiency.

    Args:
        data_source: The dataset to sample from. Must have a 'lengths' attribute,
                     which is a sequence of the lengths of each item in the dataset.
        batch_size: The number of samples in each batch.
        shuffle: Whether to shuffle the data indices before bucketing and batching.
        chunk_size: The number of samples in each chunk before sorting.
    """

    def __init__(self, data_source: Any, batch_size: int, shuffle: bool = True, chunk_size: int = 1000):
        """
        Initializes the BucketingSampler.

        Args:
            data_source: The dataset to sample from. Must have a 'lengths' attribute.
            batch_size: The number of samples per batch.
            shuffle: Whether to shuffle the data indices.
            chunk_size: The number of samples per chunk for bucketing.
        """
        super().__init__(data_source)
        self.data_source = data_source  # The dataset to sample from
        self.batch_size = batch_size    # The number of samples per batch
        self.shuffle = shuffle          # Whether to shuffle data before batching
        self.chunk_size = chunk_size    # The number of samples per chunk
        # Prepare the batches
        self.batches = self._prepare_batches()

    def _prepare_batches(self) -> List[List[int]]:
        """
        Prepares batches of indices.

        Returns:
            A list of batches, where each batch is a list of indices.
        """
        # Get indices of all data samples
        indices: List[int] = list(range(len(self.data_source)))
        # Get lengths of sequences; assumes data_source has a 'lengths' attribute
        lengths: List[int] = self.data_source.lengths

        # If shuffle is True, shuffle the indices
        if self.shuffle:
            # Generate a random permutation of indices
            rand_perm = torch.randperm(len(indices))
            # Reorder indices according to the random permutation
            indices = [indices[i] for i in rand_perm]

        # Divide indices into chunks of size 'chunk_size'
        chunks: List[List[int]] = [
            indices[i:i + self.chunk_size] for i in range(0, len(indices), self.chunk_size)
        ]

        batches: List[List[int]] = []

        for chunk in chunks:
            # Get the sequence lengths for each index in the chunk
            chunk_lengths: List[int] = [lengths[idx] for idx in chunk]
            # Sort the chunk indices by sequence length (ascending order)
            sorted_chunk: List[int] = [
                x for _, x in sorted(zip(chunk_lengths, chunk), key=lambda pair: pair[0])
            ]

            # Divide the sorted chunk into batches of size 'batch_size'
            for i in range(0, len(sorted_chunk), self.batch_size):
                batch: List[int] = sorted_chunk[i:i + self.batch_size]
                batches.append(batch)

        return batches  # List of batches, where each batch is a list of indices

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns an iterator over the batches.

        Yields:
            Batches of indices, where each batch is a list of indices.
        """
        if self.shuffle:
            # Shuffle the order of batches
            rand_perm = torch.randperm(len(self.batches))
            batches: List[List[int]] = [self.batches[i] for i in rand_perm]
        else:
            batches = self.batches

        # Yield each batch
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns:
            The total number of batches.
        """
        return len(self.batches)