import torch
from torch.utils.data import Dataset
from .binidx import MMapIndexedDataset

class MMapDataset(Dataset):
    def __init__(self, data_prefix, req_len):
        self.data = MMapIndexedDataset(data_prefix)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        self.count = self.data_size // req_len
        self.req_len = req_len

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        data_chunk = self.data.get(idx=0, offset=idx * self.req_len, length=self.req_len).astype(int)
        input_ids = torch.tensor(data_chunk, dtype=torch.long)
        return {
            'input_ids':input_ids,
            'token_type_ids':torch.zeros_like(input_ids),
            'attention_mask':torch.ones_like(input_ids),
        }
