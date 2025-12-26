import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
import tiktoken

def create_dataloader(input_text, tokenizer, max_length=256, stride=128, batch_size=4, shuffle=True):
    dataset = Dataset(input_text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True
    )
    return dataloader

    