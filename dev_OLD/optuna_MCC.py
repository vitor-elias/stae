# Hyperparameter training for the clustering method proposed by Bianchi (2020)

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
from scipy.stats import skew

from optuna.samplers import TPESampler

import source.nn.models as models
import source.utils.utils as utils
import source.utils.fault_detection as fd

from source.utils.utils import roc_params, compute_auc, get_auc, best_mcc, best_f1score

from importlib import reload
models = reload(models)
utils = reload(utils)

from pyprojroot import here
root_dir = str(here())

data_dir = os.path.expanduser('~/data/interim/')

epochs_list = [1,5,10,25,50,100,150,200,300,500,750,1000,1500]

def train_model(model, X, G, device, weight_loss, lr):

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    # Node coordinates
    A = torch.tensor(G.W.toarray()).float() #Using W as a float() tensor
    A = A.to(device)    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.reset_parameters()
    for epoch in range(1, 1+np.max(epochs_list)):

        optimizer.zero_grad()
        S, loss_mc, loss_o = model(X, A)
        loss = loss_mc + weight_loss*loss_o
        loss.backward()
        optimizer.step()

    return S

def evaluate_model(model, datasets, weight_loss, lr, device):

    auc_list = []
    f1_list = []
    mcc_list = []

    it = 0
    for dataset in datasets:

        print(f'Evaluating dataset {it}', flush=True)
        it+=1

        G = dataset['G']
        data = dataset['data']
        labels = dataset['labels']

        n_samples = 5

        for sample in range(n_samples):
            X = torch.tensor(data[sample]).float().to(device)
            label = labels[sample]

            S = train_model(model, X, G, device, weight_loss, lr)

            clusters = S.detach().cpu().softmax(dim=1).argmax(dim=1)
            unique_values, counts = clusters.unique(return_counts=True)
            value_counts = {value.item(): count.item() for value, count in zip(unique_values, counts)}
            count_tensor = clusters.cpu().clone().float().apply_(lambda x: value_counts[x])
            scores = 1 - (count_tensor/count_tensor.size()[0])

            auc_list.append(get_auc(scores, label).round(3))
            f1_list.append(best_f1score(scores, label).round(3))
            mcc_list.append(best_mcc(scores, label).round(3))

    return auc_list, f1_list, mcc_list


def main(args):

    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)

    def objective(trial):

        gc.collect()

        # Parameters     
        n_timestamps = datasets[0]['metadata']['T']
        weight_loss = trial.suggest_categorical('weight_loss', args.weight_loss)
        graphconv_n_feats = trial.suggest_categorical('graphconv_n_feats', args.graphconv_n_feats)
        n_clusters = trial.suggest_categorical('n_clusters', args.n_clusters)
        lr = trial.suggest_categorical('lr', args.lr)

        ###

        print(f"Trial: {trial.number}", flush=True)
        print(f"- N Clusters: {n_clusters}", flush=True)
        print(f"- Graph Convolution feats: {graphconv_n_feats}", flush=True)
        print(f"- Learing rate: {lr}", flush=True)
        print(f"- Weight loss: {weight_loss}", flush=True)

        ###

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()
            
        
        model = models.MCC(graphconv_n_feats, n_timestamps, n_clusters)
       
        model = model.to(device)

        auc_list, f1_list, mcc_list = evaluate_model(model, datasets, weight_loss, lr, device)
        
        trial.set_user_attr("f1", np.mean(f1_list).round(3))
        trial.set_user_attr("mcc", np.min(mcc_list).round(3))
        trial.set_user_attr("med", np.median(auc_list).round(3))
        trial.set_user_attr("auc_list", [round(elem, 2) for elem in auc_list])

        return np.mean(auc_list).round(3)

    if args.device=='auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    print(f'Using {device}', flush=True)
    print(f'log dir: {args.log_dir}', flush=True)
    print(f'dataset: {args.datafile}', flush=True)
    print(f'------', flush=True)

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    
    # OBTAINING DATA
    # datafile = 'synthetic_timeseries.pt'
    # datafile = 'synthetic_binary.pt'
    datafile = args.datafile
    datasets = torch.load(data_dir + datafile)[0:4]

    study = optuna.create_study(sampler=TPESampler(), direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=24,
                                                                   interval_steps=6))
    
    warnings.filterwarnings("ignore")
    study.set_metric_names(['auc'])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)

    if args.datafile == 'synthetic_timeseries.pt':
        datalog = 'ST_'
    elif args.datafile == 'synthetic_binary.pt':
        datalog = 'SB_'
    elif datafile == 'synthetic_InSAR.pt':
        datalog = 'SI_'

    if args.log_mod == '':
        log_file = args.log_dir + datalog + args.model + '.pkl'
    else:
        log_file = args.log_dir + datalog + args.model + '_' + args.log_mod + '.pkl'

    if args.reuse:
        if os.path.isfile(log_file):
            print('Reusing previous study', flush=True)
            study = joblib.load(log_file)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    joblib.dump(study, log_file)

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MCC')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/')
    parser.add_argument('--log_mod', type=str, default='')


    # PARAMETERS CONVOLUTION
    parser.add_argument('--n_clusters', type=float, nargs='+', default=[3,5,7,10])
    parser.add_argument('--weight_loss', type=float, nargs='+', default=[0.6, 0.8, 1])
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4])

    parser.add_argument('--graphconv_n_feats', type=int, nargs='+', default=[10, 20, 30, 40, 50])

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    args = parser.parse_args()

    main(args)
