import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import boto3 as boto
import random
import pickle

PATH = ""

s3_resource= boto.resource('s3')

class MammographyDataset(Dataset):

    def __init__(self, patient_id: list, labels: dict, path :str = None):
        # print('initilizing the Mamographydataset class')
        self.patient_ids = patient_id
        self.labels = labels
        self.path = path or 'DataSet/processed'
        # print('finished initilizing the Mamographydataset class')
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        id = self.patient_ids[idx]
        
        #One image sample training path
        # path = 'smalldata'
        
        #Sample Training path
        path = 'dv_data'
        
        #Local path to all data
        # print(f'ID: {id}')
        # path = 'all_data'
        
        LCC = pickle.load((open(f'{path}/{id}/LCC.pt','rb')))
        LMLO = pickle.load((open(f'{path}/{id}/LMLO.pt','rb')))
        RCC = pickle.load((open(f'{path}/{id}/RCC_flipped.pt','rb')))
        RMLO = pickle.load((open(f'{path}/{id}/RMLO_flipped.pt','rb')))
        

        assert(LCC.shape == LMLO.shape)
        assert(RCC.shape == RMLO.shape)
        assert(LCC.shape == RMLO.shape)

        tensor = torch.zeros(4,LCC.shape[0],LCC.shape[1])
        tensor[0] = LCC
        tensor[1] = LMLO
        tensor[2] = RCC
        tensor[3] = RMLO
        
        labels = torch.tensor((self.labels[id]['L'] -1, self.labels[id]['R'] - 1))

        return tensor.float(), labels


def sequential_train_test_split(split: tuple, labels:dict):
    
    num_patients = len(labels.keys())
    training_num = int(split[0] * num_patients)
    training_patients = labels.keys()[:training_num]
    testing_patients = labels.keys()[training_num:]

    return training_patients, testing_patients


def random_train_test_split(split: tuple, labels:dict):
    
    # print('starting random split')
    patient_set = set(labels.keys())
    training_patients = set()

    num_patients = len(labels.keys())
    training_num = int(split[0] * num_patients)
    
    i = 0
    while i != training_num:
        # print('about to do random choice')
        curr_patient = random.choice(list(patient_set-training_patients))
        training_patients.add(curr_patient)
        i += 1

    test_patients = patient_set - training_patients

    return list(training_patients), list(test_patients)


def get_train_test_dataset(split: tuple, sequential: bool, path: str = None, batch_size: int = 1) -> Tuple[MammographyDataset, MammographyDataset]:
    
    #Smallest Sample Dic
    # dictionary = 'small_small_dic.pt'
    
    #Sample Dictionary
    dictionary = 'small_dic.pt'
    #
    #Real Dictionary
    # dictionary = 'label_dict.pt'

    label_dic = pickle.load(open(dictionary, 'rb'))
    if sequential:
        train, test = sequential_train_test_split(split,label_dic)
    else:
        train, test = random_train_test_split(split,label_dic)
        
        
    print(f'LEN TEST GEN BEFORE: {len(test)}')
    print(f'LEN TRAIN GEN BEFORE: {len(train)}')
    
    if len(test) % batch_size != 0:
        a = len(test) % batch_size
        test = test[:-a]
    
    if len(train) % batch_size != 0:
        b = len(train) % batch_size
        train = train[:-b]
        
    print(f'LEN TEST GEN: {len(test)}')
    print(f'LEN TRAIN GEN: {len(train)}')
    # print('Getting Training Set')
    train_set = MammographyDataset(train,label_dic,path)
    # print('Finished getting training set; Getting Testing Set')
    test_set = MammographyDataset(test,label_dic,path)
    # print('Finished getting test set; returning')

    return train_set, test_set


def get_train_test_dataloader(split: tuple, sequential: bool, batch: int, path:str = None) -> Tuple[DataLoader, DataLoader]:

    train_set, test_set = get_train_test_dataset(split,sequential,path, batch)
    # print('got train and test set properly')
    # print('creating the training generator')
    
    # print('Finished creating the training generator, creating the testing generator')
    
    
   
    # print('finished the testing generator; returning')
    
    
    training_generator = DataLoader(train_set, batch_size = batch, shuffle = True)
    test_generator = DataLoader(test_set, batch_size = batch, shuffle = True)


    return training_generator, test_generator
