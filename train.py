from pathlib import Path

from torch.utils.data import DataLoader

from data_management.data_loading import SequenceDataset, BucketingSampler
from components.encoder import collate_protein_sequences


fasta_path = Path('ADD_FASTA_PATH_HERE')

batch_size = 50
chunk_size = 10000
dataloader_threads = 10
shuffle = True

# Training loop code snippets.
dataset = SequenceDataset(fasta_path)
sampler = BucketingSampler(dataset, batch_size, shuffle, chunk_size)
dataloader = DataLoader(
    dataset=dataset,
    batch_sampler=sampler,
    collate_fn=collate_protein_sequences,
    num_workers=dataloader_threads,
    pin_memory=True,
)



