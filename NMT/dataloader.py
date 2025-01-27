import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file):
        self.src_data = self.load_file(src_file)
        self.tgt_data = self.load_file(tgt_file)
        assert len(self.src_data) == len(self.tgt_data), "Source and target files must have the same number of sentences."

    def load_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return [list(map(int, line.strip().split())) for line in f]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx]), torch.tensor(self.tgt_data[idx])

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch

def get_dataloaders(train_src, train_tgt, val_src, val_tgt, batch_size):
    train_dataset = TranslationDataset(train_src, train_tgt)
    val_dataset = TranslationDataset(val_src, val_tgt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
