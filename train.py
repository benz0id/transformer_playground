import multiprocessing
import pickle
import mmap
from multiprocessing import freeze_support

from tqdm import tqdm

from utils.log_utils import setup_logger
from data_management.helpers import MemoryLeakDetector

multiprocessing.set_start_method('spawn', force=True)
from torchviz import make_dot
from pathlib import Path

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data_management.data_loading import SequenceDataset, BucketingSampler, InfiniteDataLoader
from components.encoder import collate_protein_sequences
from model import Model
from transformers import get_linear_schedule_with_warmup

logger = setup_logger('training_loop')


def main():
    # === INSTANTIATE MODEL ===

    logger.info('Instantiating model.')

    model = Model(
        vocab_size=21,
        model_dim=64,
        dropout=0.1,
        max_len=5000,
        num_layers=4,
        num_heads=2,
        feed_forward_hidden_dim=512
    )

    # === DATA MANAGEMENT ===
    fasta_path = Path('/home/ben/peptide_design/datasets/uniprot_trembl.fasta')
    pickle_path = Path('/home/ben/peptide_design/datasets/uniprot_parsed.pickle')

    batch_size = 20
    chunk_size = 10000
    dataloader_threads = 2
    data_limit = False # 50000 # For testing purposes.
    shuffle = True

    logger.info('Loading dataset')
    # dataset = SequenceDataset(fasta_path, limit_dataset=data_limit, verbose=True)
    with open(pickle_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            dataset =  pickle.loads(mm)

    logger.info('Configuring sampling')
    sampler = BucketingSampler(dataset, batch_size, shuffle, chunk_size)

    logger.info('Configuring dataloader')
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=collate_protein_sequences,
        num_workers=dataloader_threads,
    )
    infinite_dataloader = InfiniteDataLoader(dataloader)

    # === TRAINING UTILITIES ===

    max_training_steps = 10000000
    accumulation_steps = 100

    # Claude 3.5 Sonnet 06/02/2025
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,  # Starting learning rate
        betas=(0.9, 0.999),  # Default Adam betas usually work well
        eps=1e-8,
        weight_decay=0.01  # Moderate weight decay since proteins have strong patterns
    )

    # Claude 3.5 Sonnet 06/02/2025
    num_warmup_steps = 1000  # Adjust based on your dataset size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_training_steps
    )

    cross_entropy = nn.CrossEntropyLoss()

    logger.info('Transferring model to GPU')
    model.to('cuda')
    model.train()

    detector = MemoryLeakDetector()
    logger.info('Beginning training loop')

    it1 = tqdm(
        range(max_training_steps // accumulation_steps),
        desc='Starting...',
        position=0,
    )
    desc_str = 'Training Step | loss={loss:.5f}'

    for step in it1:
        mean_loss = 0
        for acc_step in range(accumulation_steps):
            X, Y, mask = infinite_dataloader.get_next_batch()
            logits = model(X, mask)
            logits = logits.permute(0, 2, 1)
            loss = cross_entropy(logits, Y) / batch_size
            loss.backward()
            mean_loss += loss.item()

        if step == 0:
            make_dot(logits).render("model_graph", format="png")

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        mean_loss /= accumulation_steps
        it1.set_description(desc_str.format(loss=mean_loss))

if __name__ == '__main__':
    freeze_support()
    main()

