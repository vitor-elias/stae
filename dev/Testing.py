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

def get_model_from_name(model_name, n_nodes, input_dim, params, device='cuda'):
    # Determine the model class and parameters
    if 'RAE' in model_name:
        rnn_type = 'GRU' if 'GRU' in model_name else 'LSTM'
        model_params = {
            'n_features': n_nodes,
            'latent_dim': params['latent_dim'],
            'rnn_type': rnn_type,
            'rnn_act': 'relu',
            'device': device
        }
        model_class = getattr(models, 'RAE')
    elif 'GUNet' in model_name:
        model_params = {
            'in_channels': input_dim,
            'out_channels': input_dim,
            'hidden_channels': params['hidden_channels'],
            'depth': params['depth'],
            'pool_ratios': params['pool_ratios']
        }
        model_class = getattr(models, 'GUNet')
    else:  # Covers AE, GCN, and GConv models
        layer_dims = [input_dim]
        current_dim = 2 * input_dim
        n_layers = params['n_layers']
        for i in range(n_layers):
            next_dim = params[f'layer_dim_{i}']
            layer_dims.append(int(next_dim))
            current_dim = next_dim
        model_params = {'layer_dims': layer_dims}
        model_class = getattr(models, model_name)
    
    # Instantiate the model
    model = model_class(**model_params).to(device)    

    return model

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


    model_names = ['AE', 'GCN2MLP', 'GCNAE', 'GConv2MLP', 'GConvAE', 'GUNet', 'RAE_GRU', 'RAE_LSTM']
    for model_name in model_names:
        print(f'model: {model_name}', flush=True)

        params = model_dict[model_name]['trial_params']
        score_list = []

        for iteration, dataset in enumerate(datasets):

            print(f'dataset: {iteration}/{len(datasets)}', flush=True) 

            data = dataset['data']
            label = dataset['label'].any(axis=1)
            X = torch.tensor(data).float().to(device)

            n_nodes = X.shape[0]
            input_dim = X.shape[1]

            lr = model_dict[model_name]['trial_params']['lr']
            n_epochs = model_dict[model_name]['trial_params']['n_epochs']

            edge_index, edge_weight = (None, None)

            scores_seed = []
            for seed in range(25):

                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)

                model = get_model_from_name(model_name, n_nodes, input_dim, params, device=device)
                model = model.to(device)

                if isinstance(model, graph_models):
                    edge_index = dataset['edge_index'].to( device )
                    edge_weight = dataset['edge_weight'].to( device )

                output = train_model(model, X, lr, n_epochs, edge_index, edge_weight, seed)

                scores= pixel_mse(output, X).detach().cpu().numpy()
                scores_seed.append(scores)
            
            score_list.append(scores_seed)

        model_dict[model_name]['scores'] = score_list

    torch.save(model_dict, root_dir + f'/outputs/Testing/model_dict_testing_{dataset_name}.pkl')

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--datafile', type=str, default='EGMS_anomaly/Test/dataset.pt')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)