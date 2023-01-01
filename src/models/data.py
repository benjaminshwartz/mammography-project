import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import boto3 as boto
import random

PATH = ""

s3_resource= boto.resource('s3')

class MammographyDataset(Dataset):

    def __init__(self, patient_id: list, labels: dict, path :str = None):
        self.patient_ids = patient_id
        self.labels = labels
        self.path = path or 'DataSet/processed'

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        id = self.patient_ids[idx]
        
        LCC = torch.load(s3_resource.Bucket('mammographydata').download_file(key = f'{self.path}/{id}', Filename = 'LCC.pt'))
        LMLO = torch.load(s3_resource.Bucket('mammographydata').download_file(key = f'{self.path}/{id}', Filename = 'LMLO.pt'))
        RCC = torch.load(s3_resource.Bucket('mammographydata').download_file(key = f'{self.path}/{id}', Filename = 'RCC_flipped.pt'))
        RMLO = torch.load(s3_resource.Bucket('mammographydata').download_file(key = f'{self.path}/{id}', Filename = 'RMLO_flipped.pt'))

        assert(LCC.shape == LMLO.shape)
        assert(RCC.shape == RMLO.shape)
        assert(LCC.shape == RMLO.shape)

        tensor = torch.zeros(4,LCC.shape[0],LCC.shape[1])
        tensor[0] = LCC
        tensor[1] = LMLO
        tensor[2] = RCC
        tensor[3] = RMLO
        
        labels = (self.labels[id]['L'], self.labels[id]['R'])

        return tensor.float(), labels


def sequential_train_test_split(split: tuple, labels:dict):
    
    num_patients = len(labels.keys())
    training_num = int(split[0] * num_patients)
    training_patients = labels.keys()[:training_num]
    testing_patients = labels.keys()[training_num:]

    return training_patients, testing_patients


def random_train_test_split(split: tuple, labels:dict):
    patient_set = set(labels.keys())
    training_patients = set()

    num_patients = len(labels.keys())
    training_num = int(split[0] * num_patients)
    
    i = 0
    while i != num_patients:
        curr_patient = random.choice(list(patient_set-training_patients))
        training_patients.add(curr_patient)
        i += 1

    test_patients = patient_set - training_patients


    return list(training_patients), list(test_patients)


def get_train_test_dataset() -> Tuple[MammographyDataset, MammographyDataset]:

    training_set, test_set = None, None

    return training_set, test_set


def get_train_test_dataloader() -> Tuple[DataLoader, DataLoader]:

    training_generator, test_generator = None, None

    return training_generator, test_generator
