import torch
from torch.utils.data import Dataset
import numpy as np
import pdb

class CustomDataset(Dataset):
    def __init__(self, dataset, seq_len,input_features,target_features,flag):
        self.df = dataset
        self.seq_len = seq_len
        self.input_features = input_features
        self.target_features = target_features
        self.flag = flag
        self.valid_indices = self.df.index[self.df[self.flag] == True].tolist()

    def __len__(self):
        return self.df[self.flag].sum()

    def __getitem__(self, idx):
        # Find the index of the row within the valid rows based on the flag
        idx_within_valid_rows = self.valid_indices[idx]

        # Extract the corresponding input and target sequences
        input_seq = self.df.loc[idx_within_valid_rows: idx_within_valid_rows + self.seq_len-1, self.input_features].values.astype(float)
        target_seq = self.df.loc[idx_within_valid_rows + self.seq_len, self.target_features].values.astype(float)

        # Convert sequences to tensors
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32)

        #flatten target_seq
        # target_seq = target_seq.view(-1)

        return input_seq, target_seq


