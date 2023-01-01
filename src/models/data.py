import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import boto3 as boto

PATH = ""

s3_client = boto.client('s3')

class MammographyDataset(Dataset):

    def __init__(self, scan_ids: list, labels: dict, path :str):
        self.scan_ids = scan_ids
        self.labels = labels
        self.path = path

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        id = self.scan_ids[idx]
        tensor = torch.load(s3_client.get_object(bucket = 'mammographydata', key = self.path))
        labels = self.labels[id]

        return tensor.float(), labels


def sequential_train_test_split():
    train_scans, test_scans = None, None

    return train_scans, test_scans


def train_test_split():
    train_scans, test_scans = None, None

    return train_scans, test_scans


def get_train_test_dataset() -> Tuple[MammographyDataset, MammographyDataset]:

    training_set, test_set = None, None

    return training_set, test_set


def get_train_test_dataloader() -> Tuple[DataLoader, DataLoader]:

    training_generator, test_generator = None, None

    return training_generator, test_generator
