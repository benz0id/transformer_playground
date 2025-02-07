import random
from pathlib import Path
from typing import List, Any, Iterator, Union

from torch.utils.data import Sampler
from tqdm import tqdm
import torch
from Bio import SeqIO

from utils.log_utils import setup_logger

logger = setup_logger('data_loading')

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
        self._prepare_batches()


    def _prepare_batches(self):
        """
        Prepares batches of indices.

        Returns:
            A list of batches, where each batch is a list of indices.
        """
        # Get lengths of sequences; assumes data_source has a 'lengths' attribute
        lengths = self.data_source.lengths

        len_to_inds = {}
        for ind, length in tqdm(enumerate(lengths), desc='Binning sequences'):
            if length in len_to_inds:
                len_to_inds[length].append(ind)
            else:
                len_to_inds[length] = [ind]

        self.ordered_inds = []
        for length in tqdm(len_to_inds, desc='Building index list'):
            self.ordered_inds.extend(len_to_inds[length])

        self.batch_inds = list(range(0, len(lengths) // self.batch_size))


    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns an iterator over the batches.

        Yields:
            Batches of indices, where each batch is a list of indices.
        """
        inds = self.batch_inds[:]
        if self.shuffle:
            logger.info('Shuffling batch indices.')
            random.shuffle(inds)

        # Yield each batch
        while inds:
            ind = inds.pop()
            start = ind * self.batch_size
            stop = (ind + 1) * self.batch_size
            yield self.ordered_inds[start: stop]

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns:
            The total number of batches.
        """
        return len(self.batches)


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def get_next_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            # Restart when we run out
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch