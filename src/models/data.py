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
        #### REMEMBER TO CHANGE SELF.PATH WHEN CHANGING BETWEEN SMALL AND LARGE DATA SET
        
        # self.path = 'dv'
        self.path = 'processed'
        # print('finished initilizing the Mamographydataset class')
        CC_stats, MLO_stats = self.mean_and_variance()
        
        self.CC_mean, self.CC_std = CC_stats[0], CC_stats[1]
        self.MLO_mean, self.MLO_std = MLO_stats[0], MLO_stats[1]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        id = self.patient_ids[idx]
        
        #One image sample training path
        # path = 'smalldata'
        
        #Sample Training path
        # path = 'dv'
        # self.path = 'processed'
        
        #Local path to all data
        # print(f'ID: {id}')
        # path = 'all_data'
        # path = 'processed'
        
        LCC = pickle.load((open(f'{self.path}/{id}/LCC.pt','rb')))
        LMLO = pickle.load((open(f'{self.path}/{id}/LMLO.pt','rb')))
        RCC = pickle.load((open(f'{self.path}/{id}/RCC_flipped.pt','rb')))
        RMLO = pickle.load((open(f'{self.path}/{id}/RMLO_flipped.pt','rb')))

        LCC = (LCC - self.CC_mean)/self.CC_std
        LMLO = (LMLO - self.MLO_mean)/self.MLO_std
        RCC = (RCC - self.CC_mean)/ self.CC_std
        RMLO = (RMLO - self.MLO_mean) / self.MLO_std
        

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

    def mean_and_variance(self):
        print('Starting mean_and_variance')
        N = 0
        CC = 0
        MLO = 0
        squares_CC = 0
        squares_MLO = 0

        for patient in self.patient_ids:
            LCC = pickle.load((open(f'{self.path}/{patient}/LCC.pt','rb')))
            LMLO = pickle.load((open(f'{self.path}/{patient}/LMLO.pt','rb')))
            RCC = pickle.load((open(f'{self.path}/{patient}/RCC_flipped.pt','rb')))
            RMLO = pickle.load((open(f'{self.path}/{patient}/RMLO_flipped.pt','rb')))



            assert(LCC.numel() == LMLO.numel())
            assert(RCC.numel() == RMLO.numel())
            assert(LCC.numel() == RMLO.numel())
            
            N += LCC.numel()
            CC += (LCC.sum() + RCC.sum())
            MLO += (LMLO.sum() + RMLO.sum())

            squares_CC += (LCC** 2).sum() + (RCC** 2).sum()
            squares_MLO += (LMLO** 2).sum() + (RMLO** 2).sum()

        mean_CC = CC / N
        mean_MLO = MLO / N

        var_CC = (squares_CC - (mean_CC ** 2)/N) / (N - 1)
        var_MLO = (squares_MLO - (mean_MLO ** 2)/N) / (N - 1)

        std_CC = torch.sqrt(var_CC)
        std_MLO = torch.sqrt(var_MLO)
        print('Finishing mean_and_variance')
        return (mean_CC,std_CC),(mean_MLO,std_MLO)



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
    # dictionary = 'small_dic.pt'
    #
    #Real Dictionary
    dictionary = 'label_dict.pt'

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
