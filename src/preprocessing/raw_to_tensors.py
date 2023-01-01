import pickle
import pydicom as dicom
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import torch


CSV_PATH = ""
RAW_PATH = ""
PREPROCESSED_PATH = ""
LABELS_PATH = ""

################################################################################
# parse csv summary file

df = pd.read_csv(f'{CSV_PATH}/breast-level_annotations.csv')
df = df[(df['height'] == 3518) & (df['width'] == 2800)]
df = df[~df['study_id'].isin(['dbca9d28baa3207b3187c4d07dc81a80'])]

################################################################################
#  CREATE LABEL DICT of the form {'patient': {'L': score, 'R': score} }

patients = list(df['study_id'].unique())
left_breast_birads = df[((df['laterality'] == 'L') & (
    df['view_position'] == 'CC'))]['breast_birads']
right_breast_birads = df[((df['laterality'] == 'R') & (
    df['view_position'] == 'CC'))]['breast_birads']

left_breast_birads = list(left_breast_birads.map(lambda x: int(x[-1])))
right_breast_birads = list(right_breast_birads.map(lambda x: int(x[-1])))

label_dict = dict()

i = 0
for patient in patients:
    left_label = left_breast_birads[i]
    right_label = right_breast_birads[i]
    label_dict[patient] = {'L': left_label, 'R': right_label}
    i += 1

label_dict


pickle.dump(LABELS_PATH, label_dict)

################################################################################
# create preprocessed data

if not os.path.exists(PREPROCESSED_PATH):
    os.mkdir(PREPROCESSED_PATH)

for patient in patients:
    curr_df = df[df['study_id'] == patient]
    lmlo_image_id = list(curr_df[((curr_df['laterality'] == 'L') & (
        curr_df['view_position'] == 'MLO'))]['image_id'])[0]
    lcc_image_id = list(curr_df[((curr_df['laterality'] == 'L') & (
        curr_df['view_position'] == 'CC'))]['image_id'])[0]
    rmlo_image_id = list(curr_df[((curr_df['laterality'] == 'R') & (
        curr_df['view_position'] == 'MLO'))]['image_id'])[0]
    rcc_image_id = list(curr_df[((curr_df['laterality'] == 'R') & (
        curr_df['view_position'] == 'CC'))]['image_id'])[0]

    lmlo_dicom = dicom.dcmread(
        f'{RAW_PATH}/image/{patient}/{lmlo_image_id}.dicom')
    lcc_dicom = dicom.dcmread(
        f'{RAW_PATH}/image/{patient}/{lcc_image_id}.dicom')
    rmlo_dicom = dicom.dcmread(
        f'{RAW_PATH}/image/{patient}/{rmlo_image_id}.dicom')
    rcc_dicom = dicom.dcmread(
        f'{RAW_PATH}/image/{patient}/{rcc_image_id}.dicom')

    lmlo_torch = torch.from_numpy(np.array(lmlo_dicom.pixel_array, dtype= np.float32))
    lcc_torch = torch.from_numpy(np.array(lcc_dicom.pixel_array, dtype = np.float32))
    rmlo_torch = torch.flip(torch.from_numpy(np.array(rmlo_dicom.pixel_array, dtype = np.float32)),(1,))
    rcc_torch = torch.flip(torch.from_numpy(np.array(rcc_dicom.pixel_array, dtype= np.float32),(1,)))

    pickle.dump(lmlo_torch, open(
        f'{PREPROCESSED_PATH}/{patient}/LMLO.pt', "wb"))
    pickle.dump(lcc_torch, open(f'{PREPROCESSED_PATH}/{patient}/LCC.pt', "wb"))
    pickle.dump(rmlo_torch, open(f'{PREPROCESSED_PATH}/{patient}/RMLO.pt', "wb"))
    pickle.dump(rcc_torch, open(f'{PREPROCESSED_PATH}/{patient}/RCC.pt', "wb"))
