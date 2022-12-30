import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

PATH = ""


class MammographyDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return None

    def __getitem__(self):
        return None


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
