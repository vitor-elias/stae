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

def main(args):

    # INITIALIZATION
    
    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)

    # OBTAINING DATA
    datafile = args.datafile
    datasets = torch.load(dataset_path + datafile)
    input_dim = datasets[0]['data'].shape[1]

    if args.device=='auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device


    datalog = 'TR_'
        
    if args.log_mod == '':
        log_file = args.log_dir + datalog + args.model + '.pkl'
    else:
        log_file = args.log_dir + datalog + args.model + '_' + args.log_mod + '.pkl'


    print(f'log dir: {args.log_dir + log_file}', flush=True)
    print(f'dataset: {args.datafile}', flush=True)
    print(f'model: {args.model}', flush=True)
    print(f'on: {device}', flush=True)
    print(f'------', flush=True)

    # DEFINITIONS

    def train_model(model, X, lr, n_epochs, edge_index=None, edge_weight=None, batch_size=None):

        rng_seed = 0
        torch.manual_seed(rng_seed)
        torch.cuda.manual_seed(rng_seed)
        np.random.seed(rng_seed)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        model.train()
        model.reset_parameters()

        if isinstance(model, models.RAE):

            if model.n_features != 1:
                for epoch in range(n_epochs):
                               
                    optimizer.zero_grad()
                    output = model(X.T.unsqueeze(0)).squeeze(0).T
                    loss = criterion(output, X)
                    loss.backward()
                    optimizer.step()
                
                return output


            else:

                dataset = TensorDataset(X, X)  # we want to reconstruct the same input
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                for epoch in range(n_epochs):
                    
                    for batch, _ in dataloader:
            
                        optimizer.zero_grad()
                        output = model(batch)
                        loss = criterion(output, batch)
                        loss.backward()
                        optimizer.step()

                return model(X)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            if isinstance(model, graph_models):
                output = model(X, edge_index=edge_index, edge_weight=edge_weight)
            else:
                output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
        return output

    def evaluate_model(model, datasets, lr, n_epochs, batch_size=None):

        auc_dataset = []

        it = 0
        for dataset in datasets:

            # print(f'Evaluating dataset {it}', flush=True)
            it+=1

            data = dataset['data']
            label = dataset['label'].any(axis=1) #label per pixel
            
            X = torch.tensor(data).float().to( next(model.parameters()).device )

            edge_index, edge_weight = (None, None)
            if isinstance(model, graph_models):
                edge_index = dataset['edge_index'].to( next(model.parameters()).device )
                edge_weight = dataset['edge_weight'].to( next(model.parameters()).device )

            if isinstance(model, models.RAE) and (model.n_features != 1):
                relevant_params = ['n_features', 'latent_dim', 'rnn_type', 'rnn_act', 'device']
                new_model_params = {key: getattr(model, key) for key in relevant_params}
                new_model_params['n_features'] = X.shape[0]
                model = models.RAE(**new_model_params)
                model.to(new_model_params['device'])

            output = train_model(model, X, lr, n_epochs, edge_index, edge_weight, batch_size)

            scores = pixel_mse(output, X).detach().cpu().numpy()

            auc = get_auc(scores, label, resolution=101).round(3)
            auc_dataset.append(auc)

        return auc_dataset

    def objective(trial):

        gc.collect()

        print(f"Trial: {trial.number}", flush=True)

        ### Parameters

        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True) 
        n_epochs = trial.suggest_int('n_epochs', 1, 1000, log=True)

        if args.model == 'RAE':
            model_params = {'n_features': 2,
                            'latent_dim': trial.suggest_int('latent_dim', 2, 6),
                            'rnn_type': args.log_mod,
                            'rnn_act': 'relu',
                            'device': device}
            # batch_size = 2**trial.suggest_int('batch_size',5,9)
            batch_size = 1024
        
        elif args.model == 'GUNet':
            model_params = {'in_channels': input_dim,
                            'out_channels': input_dim,
                            'hidden_channels': trial.suggest_int('hidden_channels', 2, input_dim*2),
                            'depth': trial.suggest_int('depth', 1, 5),
                            'pool_ratios': trial.suggest_float('pool_ratios', 0.3, 0.9)}
            batch_size = None
        else:
            # Layer dimensions
            layer_dims = [input_dim]
            current_dim = input_dim*2       
            n_layers = trial.suggest_int('n_layers', 2, 5)
            for i in range(n_layers):
                next_dim = trial.suggest_int(f'layer_dim_{i}', 2, current_dim)             # Non-increasing dimensions
                layer_dims.append(next_dim)
                current_dim = next_dim 
            ###
            model_params = {'layer_dims':layer_dims}
            batch_size = None

        print(trial.params, flush=True)

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()

        model_class = getattr(models, args.model)
        model = model_class(**model_params)
        model = model.to(device)

        auc_dataset = evaluate_model(model, datasets, lr, n_epochs, batch_size)
        
        trial.set_user_attr("auc_dataset", np.round(auc_dataset,3))
        
        return np.mean(auc_dataset).round(3)


    # RUNNING
    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    
    study = optuna.create_study(sampler=TPESampler(), direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=24,
                                                                   interval_steps=6))
    
    warnings.filterwarnings("ignore")
    study.set_metric_names(['auc'])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)
    

    if args.reuse:
        if os.path.isfile(log_file):
            print('Reusing previous study', flush=True)
            study = joblib.load(log_file)

    
    print(args.model)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    joblib.dump(study, log_file)

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AE')
    parser.add_argument('--datafile', type=str, default='pixel_detection/Oslo/training/dataset.pt')

    # parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/Geological_anomaly/')
    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/Phase_anomaly/')

    parser.add_argument('--log_mod', type=str, default='')
    parser.add_argument('--n_trials', type=int, default=1)
    
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    args = parser.parse_args()

    main(args)