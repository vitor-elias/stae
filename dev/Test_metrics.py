# HYPERPARAMETER TUNING FOR DL MODELS AVAILABLE IN PYOD
# AE, GAN

import numpy as np
import pandas as pd

import networkx as nx

import math
import os
import gc
import argparse
import torch
import optuna
import joblib
import sqlite3
import pickle
import warnings
import hashlib
import copy

from datetime import datetime
from tqdm import tqdm

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.anogan import AnoGAN
from torch.utils.data import DataLoader, TensorDataset

from optuna.samplers import TPESampler, GridSampler, BruteForceSampler

import source.nn.models as models
import source.utils.utils as utils
import source.utils.fault_detection as fd

from source.utils.utils import roc_params, compute_auc, get_auc, best_mcc, best_f1score

from importlib import reload
models = reload(models)
utils = reload(utils)

from pyprojroot import here
root_dir = str(here())

dataset_path = root_dir + "/data/datasets/"


def main():

    
    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)

    dataset_name = 'EGMS_anomaly'
    datasets = torch.load(dataset_path + f'{dataset_name}/Test/dataset.pt')
    model_dict = torch.load(root_dir + f'/outputs/Testing/model_dict_testing_{dataset_name}.pkl')

    model_names = ['AE', 'GCN2MLP', 'GCNAE', 'GConv2MLP', 'GConvAE', 'GUNet', 'RAE_GRU', 'RAE_LSTM']

    # Dictionary to store metrics for each model
    metrics_dict = {}

    for model_name in model_names:
        print(f"Computing metrics for {model_name}", flush=True)
        auc_list = []
        f1_list = []
        mcc_list = []
        
        for idx, dataset in enumerate(datasets):
            # Compute metrics for each dataset based on each label being true if any anomaly is present
            print(f"Processing dataset {idx+1}/{len(datasets)} for model {model_name}", flush=True)
            label = dataset['label'].any(axis=1)
            scores = model_dict[model_name]['scores'][idx]

            auc_seed = []
            f1_seed = []
            mcc_seed = []
            for seed in range(25):
                scores_seed = scores[seed]
                auc_seed.append(get_auc(scores_seed, label, resolution=101).round(3))
                f1_seed.append(best_f1score(scores_seed, label).round(3))
                mcc_seed.append(best_mcc(scores_seed, label).round(3))

            # Store metrics for the current dataset
            auc_list.append(auc_seed)
            f1_list.append(f1_seed)
            mcc_list.append(mcc_seed)

        
        # Store metrics and compute statistics
        metrics_dict[model_name] = {
            'mean_auc': np.mean(np.mean(auc_list,axis=0)).round(3),
            'std_auc': np.std(np.mean(auc_list,axis=0)).round(3),
            'malmo_auc': np.mean(np.mean(auc_list[:72],axis=0)).round(3),
            'oslo_auc': np.mean(np.mean(auc_list[-72:],axis=0)).round(3),
            'mean_f1': np.mean(np.mean(f1_list,axis=0)).round(3),
            'std_f1': np.std(np.mean(f1_list,axis=0)).round(3),
            'mean_mcc': np.mean(np.mean(mcc_list,axis=0)).round(3),
            'std_mcc': np.std(np.mean(mcc_list,axis=0)).round(3),
        }

    torch.save(metrics_dict, root_dir + f'/outputs/Testing/Test_metrics_{dataset_name}.pkl')

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":
    main()