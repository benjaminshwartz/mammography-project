import boto3 as boto
import pickle
import torch
import sagemaker
import pandas as pd
import os
import numpy as np
import pydicom as dicom
from torch import nn

def import_data():
    s3 = boto.client('s3')
    sagemaker_session = sagemaker.Session()
    bucket = 'mammographydata'
    key_prefix = 'DataSet/dv/'
    download_area = 'dv_data' 
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket= bucket, Prefix= key_prefix)
    file_num = 0
    page_num = 0
    for page in pages:
        response = page['Contents']
        response = response[1:]
        for file in response:
            file_num += 1
            key = file['Key'].split('/')
            patient_ID = key[2]
            view = key[3]
            download_path = f'{download_area}/{patient_ID}'
            specific_prefix = f'{key_prefix}{patient_ID}/{view}'
            sagemaker_session.download_data(download_path, bucket, specific_prefix, extra_args=None)
            file_num += 1
            if file_num % 400 == 0:
                print(f'On file num: {file_num}')
        print(f'On Page: {page_num}')
        page_num += 1


def __main__():
    import_data()