
from typing import List, Optional
import os

from torch.utils.data import Dataset, DataLoader
from torchtext.functional import to_tensor

class CustomDataset(Dataset):
    '''Creates a torch.utils.data.Dataset and applies given transformations.'''
    def __init__(self, data: List[str], labels: list, transforms: Optional = None):
        super().__init__()

        self.labels = labels
        self.transforms = transforms

        if transforms:
            self.data = to_tensor(self.transforms(data), padding_value=1)
        else:
            self.data = to_tensor(data, padding_value=1)
       

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
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=os.cpu_count(),
                                  shuffle=False)

    return train_dataloader, test_dataloader
