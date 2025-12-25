import torch
from torch.utils.data import DataLoader
from dataset import Dataset
import tiktoken

def create_dataloader(input_text, batch_size=4, max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_ecoding("gpt_2")
    dataset = Dataset(input_text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True
    )
    return dataloader

    