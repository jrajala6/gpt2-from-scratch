import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, input_text, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer(input_text)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i: i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1: i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)  

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


