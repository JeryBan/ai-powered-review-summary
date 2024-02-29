
from typing import List, Optional
import os

from torch.utils.data import Dataset, DataLoader

from torchtext.functional import to_tensor
from torchtext.vocab import Vocab
import torchtext.transforms as T

class CustomDataset(Dataset):
    '''Creates a torch.utils.data.Dataset and applies given transformations.'''
    def __init__(self, data: List[str], labels: list, vocab: Vocab):
        super().__init__()

        self.labels = torch.tensor(labels, dtype=torch.float)
        self.data = custom_transforms(data, vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
        

def train_test_split(data: List[str], labels: list, split_point: float = 0.7):
    '''Splits the data to train and test segments according to split_point.'''
    split = int(split_point * len(data))
    
    train_data = data[0 : split]
    train_labels = labels[0 : split]

    test_data = data[split : len(data) -1]
    test_labels = labels[split : len(labels) -1]

    return train_data, train_labels, test_data, test_labels

def create_dataloaders(train_dataset, test_dataset, batch_size: int):
    '''Creates train and test dataloaders of given batch size.'''
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=os.cpu_count(),
                                  shuffle=False)

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=os.cpu_count(),
                                  shuffle=False)

    return train_dataloader, test_dataloader

def custom_transforms(data: List[str], vocab: Vocab, max_seq_len: int = 200) -> torch.Tensor:
    '''Converts words to ids truncates and returns sentences as tensors.'''
    f = T.Truncate(max_seq_len=max_seq_len)
    
    word2id = [[vocab[word] for word in sentence.split(' ')] for sentence in data]
    word2id = f(word2id)
    
    return to_tensor(word2id, padding_value=0)
    
