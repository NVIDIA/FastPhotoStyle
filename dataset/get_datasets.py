import os
import numpy as np
import pandas as pd
import torch
import tifffile
from pathlib import Path
from .TransformedDataset import TransformedDataset


### Usage
"""
from dataset_name.get_datasets import get_datasets
datasets = get_datasets()

{k: len(v) for k, v in ds.items()}
#=> {'train': 670, 'val': 149, 'test': 148}

datasets['train'][0].keys()
#=> dict_keys(['file_t0_pan', 'file_t-1_pan', 'file_t-2_pan', 'file_t-3_pan', 'file_t0_ps', 'file_t-1_ps', 'file_t-2_ps', 'file_t-3_ps', 't0_pan', 't-1_pan', 't-2_pan', 't-3_pan', 't0_ps', 't-1_ps', 't-2_ps', 't-3_ps'])

datasets['train'][0]['file_t0_pan']
#=> 'path/to/color_transfer_1k_8bit/./images/10500100017E6600_i39019_j-7217_t0_pan.tif'

datasets['train'][0]['t0_pan']
#=> array([[54, 58, 59, ..., 70, 67, 65],
       [57, 57, 58, ..., 58, 63, 68],
       [58, 57, 57, ..., 55, 63, 68],
       ...,
       [57, 54, 55, ..., 49, 52, 54],
       [55, 58, 57, ..., 52, 51, 51],
       [56, 55, 52, ..., 55, 51, 53]], dtype=uint8)

### Only load some of the files to reduce disk reads when you're not using the full timestack:
datasets = get_datasets(csv_image_cols=['t0_pan', 't0_ps'])

datasets['train'][0].keys()
#=> dict_keys(['t0_pan', 't-1_pan', 't-2_pan', 't-3_pan', 't0_ps', 't-1_ps', 't-2_ps', 't-3_ps', 'file_t0_pan', 'file_t0_ps'])

datasets['train'][0]['t0_pan']
#=> array([[54, 58, 59, ..., 70, 67, 65],
       [57, 57, 58, ..., 58, 63, 68],
       [58, 57, 57, ..., 55, 63, 68],
       ...,
       [57, 54, 55, ..., 49, 52, 54],
       [55, 58, 57, ..., 52, 51, 51],
       [56, 55, 52, ..., 55, 51, 53]], dtype=uint8)
       
datasets['train'][0]['t-1_pan']
#=> './images/1030010048653D00_i39019_j-7217_t-1_pan.tif'
"""


dataset_root = Path(os.path.dirname(__file__))
def get_path_to_relative(relpath):
    return str(dataset_root/relpath)


# @gin.configurable
def filter_void(df):
    void_classes = [0,1]
    df['class_distribution'] = [eval(df.at[i, 'class_distribution']) for i in df.index]
    df['has_void'] = [np.sum(np.array(df.at[i, 'class_distribution'])[void_classes]) > 0 for i in df.index]
    return df[df['has_void'] == False].drop(columns=['has_void'])


# @gin.configurable
def get_csv_dataset(csv, modify_dataframe_fn=None, sample_cols=['x','y'], filepath_cols=['x'], split_col='set'):
    df = pd.read_csv(csv)
    if callable(modify_dataframe_fn):
        df = modify_dataframe_fn(df)
    if split_col in df.columns:
        df_by_set = {
            s: df[df['set'] == s]
        for s in df['set'].unique()}
    else:
        df_by_set = {
            'all': df
        }
    return {
        s: TransformedDataset(
            lambda row: {
                **row,
                **{
                    k: get_path_to_relative(v)
                for k, v in row.items() if k in filepath_cols},
            },
            df_set[sample_cols].to_dict('rows')
        )
    for s, df_set in df_by_set.items()}


# @gin.configurable
def get_image_dataset(file_dataset, image_keys, imread=tifffile.imread):
    return {
        k: TransformedDataset(
            lambda row: dict(
                [(k, v) for k, v in row.items()] +
                [(f'file_{k}', v) for k, v in row.items() if k in image_keys] +
                [(k, imread(v)) for k, v in row.items() if k in image_keys]
            ),
            ds
        )
    for k, ds in file_dataset.items()}


def combine_datasets(datasets):
    return {
        k: torch.utils.data.ConcatDataset([ds[k] for ds in datasets])
    for k in datasets[0].keys()}


def get_all_image_cols():
    return [
        't0_pan',
        't-1_pan',
        't-2_pan',
        't-3_pan',
        't0_ps',
        't-1_ps',
        't-2_ps',
        't-3_ps',
#         't0_ms',
#         't-1_ms',
#         't-2_ms',
#         't-3_ms',
    ]


# @gin.configurable
def get_datasets(csv=get_path_to_relative('manifest.csv'), datatype='images', csv_sample_cols=get_all_image_cols(), csv_image_cols=get_all_image_cols()):
    assert datatype in ['filepaths', 'images']
    
    dataset = get_csv_dataset(csv, sample_cols=csv_sample_cols, filepath_cols=csv_image_cols)
    if datatype == 'filepaths':
        # Useful to get a list of the filepaths without reading the files from disk.
        return dataset
    
    dataset = get_image_dataset(dataset, image_keys=csv_image_cols)
    
    return dataset


