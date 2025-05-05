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

graph_models = tuple([models.GCN2MLP, models.GConv2MLP, models.GCNAE, models.GConvAE, models.GUNet])

def dict_hash(d):
    return hashlib.md5(str(sorted(d.items())).encode()).hexdigest()

def pixel_mse(output,X):
    point_mse = torch.nn.MSELoss(reduction='none')
    return torch.mean(point_mse(output,X), axis=1)

def train_model(model, X, lr, n_epochs, edge_index=None, edge_weight=None, rng_seed=0):

    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    model.reset_parameters()

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        if isinstance(model, graph_models):
            output = model(X, edge_index=edge_index, edge_weight=edge_weight)
        elif isinstance(model, models.RAE):
            output = model(X.T.unsqueeze(0)).squeeze(0).T
        else:
            output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()

    return output


def main(args):

    # RUNNING
    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    
    # INITIALIZATION
    
    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)

    if 'Geological_anomaly' in args.datafile:
        dataset_name = 'Geological_anomaly'
    elif 'EGMS_anomaly' in args.datafile:
        dataset_name = 'EGMS_anomaly'

    # OBTAINING DATA
    datafile = args.datafile
    datasets = torch.load(dataset_path + datafile)
    model_dict = torch.load(root_dir + f'/outputs/Optuna_analysis/model_dict_{dataset_name}.pkl')
    
    if args.device=='auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    

    print(f'dataset: {dataset_name}', flush=True)
    print(f'on: {device}', flush=True)
    print(f'------', flush=True)


    gc.collect()

    for model_name in model_dict.keys():
        print(f'model: {model_name}', flush=True)

        score_list = []

        for iteration, dataset in enumerate(datasets):

            print(f'dataset: {iteration}/{len(datasets)}', flush=True) 

            data = dataset['data']
            label = dataset['label'].any(axis=1)
            X = torch.tensor(data).float().to(device)

            n_sensors = X.shape[0]
            n_timestamps = X.shape[1]

            if 'RAE' in model_name:
                model_params = {'n_features': n_sensors,
                                'latent_dim': model_dict[model_name]['trial_params']['latent_dim'],
                                'rnn_type': model_name.split('_')[-1],
                                'rnn_act': 'relu',
                                'device': device}
            
                model_type = 'RAE'

            elif model_name == 'GUNet':
                model_params = {'in_channels': n_timestamps,
                                'out_channels': n_timestamps,
                                'hidden_channels': model_dict[model_name]['trial_params']['hidden_channels'],
                                'depth': model_dict[model_name]['trial_params']['depth'],
                                'pool_ratios': model_dict[model_name]['trial_params']['pool_ratios']}
                model_type = 'GUNet'
            else:
                layer_dims = [n_timestamps]
                layer_dims.extend([ int(model_dict[model_name]['trial_params'][f'layer_dim_{i}'])
                                    for i in range(model_dict[model_name]['trial_params']['n_layers'])])
                model_params = {'layer_dims':layer_dims}
                model_type = model_name

            # print(model_params, flush=True)

            lr = model_dict[model_name]['trial_params']['lr']
            n_epochs = model_dict[model_name]['trial_params']['n_epochs']

            model_class = getattr(models, model_type)
            model = model_class(**model_params)
            model_orig = copy.deepcopy(model).to(device)

            edge_index, edge_weight = (None, None)
            if isinstance(model, graph_models):
                edge_index = dataset['edge_index'].to( device )
                edge_weight = dataset['edge_weight'].to( device )


            scores_seed = []
            for rng_seed in range(25):

                model = copy.deepcopy(model_orig)
                model = model.to(device)

                output = train_model(model, X, lr, n_epochs, edge_index, edge_weight, rng_seed)

                scores = pixel_mse(output, X).detach().cpu().numpy()
                scores_seed.append(scores)
            
            score_list.append(scores_seed)


        model_dict[model_name]['scores'] = score_list


    torch.save(model_dict, root_dir + f'/outputs/Testing/model_dict_seed_testing_{dataset_name}.pkl')

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--datafile', type=str, default='EGMS_anomaly/Test/dataset.pt')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)