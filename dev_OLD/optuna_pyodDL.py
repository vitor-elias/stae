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

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.anogan import AnoGAN

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

data_dir = os.path.expanduser('~/data/interim/')


def try_pyod(model, X):
    try:
        model.fit(X.cpu())
        return model.decision_scores_
    except Exception as e:
        print(f"Error in fitting model {model.__class__.__name__}: {str(e)}")
        return 0.5 * np.ones(len(X))

def evaluate_model(model, datasets):

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
            X = torch.tensor(data[sample]).float()
            label = labels[sample]

            scores = try_pyod(model, X)
            auc_list.append(get_auc(scores, label).round(3))
            f1_list.append(best_f1score(scores, label).round(3))
            mcc_list.append(best_mcc(scores, label).round(3))

    return auc_list, f1_list, mcc_list


def main(args):

    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)


    def objective(trial):

        gc.collect()

        print(f"Trial: {trial.number}", flush=True)

        # Parameters     

        if args.model == 'AE':
            batch_size = trial.suggest_categorical('batch_size', args.batch_size)
            N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
            lr = trial.suggest_categorical('lr', args.lr)
            dropout_rate = trial.suggest_categorical('dropout_rate', args.dropout_rate)

            first_layer = trial.suggest_categorical('init_layer', args.init_layer)
            n_layers = trial.suggest_categorical('n_layers', args.n_layers)
            reduction_layer = trial.suggest_categorical('reduction_layer', args.reduction_layer)
            hidden_neuron_list = [int(first_layer*reduction_layer**i )for i in range(n_layers)]

            print(f"- N epochs: {N_epochs}", flush=True)
            print(f"- Batch size: {batch_size}", flush=True)
            print(f"- lr: {lr}", flush=True)
            print(f"- dropout_rate: {dropout_rate}", flush=True)
            print(f"- Neuron list: {hidden_neuron_list}", flush=True)

        elif args.model == 'GAN':
            batch_size = trial.suggest_categorical('batch_size', args.batch_size)
            N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
            lr = trial.suggest_categorical('lr', args.lr)
            dropout_rate = trial.suggest_categorical('dropout_rate', args.dropout_rate)

            first_layer_G = trial.suggest_categorical('init_layer_G', args.init_layer_G)
            n_layers_G = trial.suggest_categorical('n_layers_G', args.n_layers_G)
            reduction_layer_G = trial.suggest_categorical('reduction_layer_G', args.reduction_layer_G)
            hidden_neuron_list_G = [int(first_layer_G*reduction_layer_G**i )for i in range(n_layers_G)]
            hidden_neuron_list_G.extend(hidden_neuron_list_G[:n_layers_G-1][::-1])

            first_layer_D = trial.suggest_categorical('init_layer_D', args.init_layer_D)
            n_layers_D = trial.suggest_categorical('n_layers_D', args.n_layers_D)
            reduction_layer_D = trial.suggest_categorical('reduction_layer_D', args.reduction_layer_D)
            hidden_neuron_list_D = [int(first_layer_D*reduction_layer_D**i )for i in range(n_layers_D)]

            print(f"- N epochs: {N_epochs}", flush=True)
            print(f"- Batch size: {batch_size}", flush=True)
            print(f"- lr: {lr}", flush=True)
            print(f"- dropout_rate: {dropout_rate}", flush=True)
            print(f"- G Neuron list: {hidden_neuron_list_G}", flush=True)
            print(f"- D Neuron list: {hidden_neuron_list_D}", flush=True)

        ###

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()

        if args.model=='AE':
            model = AutoEncoder(batch_size = batch_size,
                                epoch_num = N_epochs,
                                lr = lr,
                                dropout_rate = dropout_rate,
                                hidden_neuron_list = hidden_neuron_list,
                                device=device,
                                verbose=0
                                )
        elif args.model == 'GAN':
            model = AnoGAN( batch_size = batch_size,
                            epochs = N_epochs,
                            learning_rate = lr,
                            dropout_rate = dropout_rate,
                            G_layers = hidden_neuron_list_G,
                            D_layers = hidden_neuron_list_D,
                            device=device,
                            verbose=0
                            )
        
        auc_list, f1_list, mcc_list = evaluate_model(model, datasets)
        
        trial.set_user_attr("f1", np.mean(f1_list).round(3))
        trial.set_user_attr("mcc", np.min(mcc_list).round(3))
        trial.set_user_attr("med", np.median(auc_list).round(3))
        trial.set_user_attr("auc_list", [round(elem, 2) for elem in auc_list])

        return np.mean(auc_list).round(3)
    
    if args.device=='auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    print(f'log dir: {args.log_dir}', flush=True)
    print(f'dataset: {args.datafile}', flush=True)
    print(f'model: {args.model}', flush=True)
    print(f'------', flush=True)

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    
    # OBTAINING DATA
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

    
    print(args.model)

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    joblib.dump(study, log_file)

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":

    batch_list = [16, 32, 64, 128, 256]
    lr_list = [1e-4, 1e-3, 1e-2]
    epochs_list = [5, 10, 50, 100, 200]
    dropout_list = [0, 0.25, 0.5]

    # AE
    n_layers = [2, 3]
    init_layer = [32, 64, 128]
    reduction_layer = [0.25, 0.5]

    # GAN
    n_layers_G = [2, 3]
    init_layer_G = [10, 20, 30]
    reduction_layer_G = [0.5, 0.75]
    n_layers_D = [2, 3]
    init_layer_D = [10,20,30]
    reduction_layer_D = [0.25, 0.5, 0.75]


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='AE')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/')
    parser.add_argument('--log_mod', type=str, default='')
    parser.add_argument('--n_trials', type=int, default=1)
    
    parser.add_argument('--batch_size', type=int, nargs='+', default=batch_list)
    parser.add_argument('--N_epochs', type=int, nargs='+', default=epochs_list)
    parser.add_argument('--lr', type=float, nargs='+', default=lr_list)
    parser.add_argument('--dropout_rate', type=float, nargs='+', default=dropout_list)

    parser.add_argument('--n_layers', type=int, nargs='+', default=n_layers)
    parser.add_argument('--init_layer', type=int, nargs='+', default=init_layer)
    parser.add_argument('--reduction_layer', type=float, nargs='+', default=reduction_layer)

    parser.add_argument('--n_layers_D', type=int, nargs='+', default=n_layers_D)
    parser.add_argument('--n_layers_G', type=int, nargs='+', default=n_layers_G)
    parser.add_argument('--init_layer_D', type=int, nargs='+', default=init_layer_D)
    parser.add_argument('--init_layer_G', type=int, nargs='+', default=init_layer_G)
    parser.add_argument('--reduction_layer_D', type=float, nargs='+', default=reduction_layer_D)
    parser.add_argument('--reduction_layer_G', type=float, nargs='+', default=reduction_layer_G)


    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    args = parser.parse_args()

    main(args)
