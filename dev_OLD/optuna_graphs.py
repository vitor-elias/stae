# HYPERPARAMETER TUNING FOR GRAPH AUTOENCODERS AND GRAPH UNETS

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

epochs_list = [1,5,10,25,50,100,150,200,300,500,750,1000,1250]


def dict_hash(d):
    return hashlib.md5(str(sorted(d.items())).encode()).hexdigest()

def train_model(model, X, A, lr):

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    model_epoch = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss(reduction='mean')

    model.train()
    model.reset_parameters()

    for epoch in range(1, 1+np.max(epochs_list)):

        optimizer.zero_grad()
        output = model(X, A)
        loss = loss_function(X, output)
        loss.backward()
        optimizer.step()
        if epoch in epochs_list:
            model_epoch.append(model.state_dict().copy())

    return model_epoch

def evaluate_model(model, datasets, epochs, lr, device, db_conn):

    auc_list = []
    f1_list = []
    mcc_list = []

    db_cursor = db_conn.cursor()

    it = 0
    for dataset in datasets:

        print(f'Evaluating dataset {it}', flush=True)
        it+=1

        G = dataset['G']
        data = dataset['data']
        labels = dataset['labels']

        # Node coordinates
        A = torch.tensor(G.W.toarray()).float() #Using W as a float() tensor
        A = A.to(device)    

        n_samples = 5

        for sample in range(n_samples):
            X = torch.tensor(data[sample]).float().to(device)
            label = labels[sample]

            sim_dict = {'metadata':dataset['metadata'], 'model':model, 'sample':sample, 'lr':lr}
            sim_key = dict_hash(sim_dict)

            db_cursor.execute("SELECT model_epoch FROM graph_simulations WHERE hash=?", (sim_key,))
            row = db_cursor.fetchone()

            if row:
                model_epoch = pickle.loads(row[0])
            else:
                model_epoch = train_model(model, X, A, lr)
                db_cursor.execute("INSERT INTO graph_simulations (hash, model_epoch, sim_info) VALUES (?, ?, ?)",
                                ( sim_key, pickle.dumps(model_epoch), pickle.dumps(sim_dict) ))
                db_conn.commit()

            epochs_id = epochs_list.index(epochs)
            model_dict = model_epoch[epochs_id]
            model.load_state_dict(model_dict)

            model.eval()
            with torch.no_grad():
                Y = model(X, A)

            eval_function = torch.nn.MSELoss(reduction='none')
            scores = torch.mean(eval_function(X,Y), axis=1).cpu().detach().numpy()

            auc_list.append(get_auc(scores, label).round(3))
            f1_list.append(best_f1score(scores, label).round(3))
            mcc_list.append(best_mcc(scores, label).round(3))

    return auc_list, f1_list, mcc_list


def main(args):

    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)

    db_name = 'simulation_cache.db'
    db_path = root_dir + '/outputs/sql/' + db_name

    # Check if the database file exists
    if os.path.exists(db_path):
        if args.ow_db:
            print(f"The old database '{db_name}' is being overwritten.")
            os.remove(db_path)  # Delete the existing database file
        else:
            print(f"The old database '{db_name}' is being used.")
    else:
        print(f"The new database '{db_name}' is being created.")

    db_conn = sqlite3.connect(db_path)
    db_cursor = db_conn.cursor()
    db_cursor.execute('''CREATE TABLE IF NOT EXISTS graph_simulations (hash TEXT PRIMARY KEY, model_epoch BLOB, sim_info BLOB)''')
    db_conn.commit()

    def objective(trial):

        gc.collect()

        # Parameters
        n_timestamps = datasets[0]['metadata']['T']   
        N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
        lr = trial.suggest_categorical('lr', args.lr)

        if args.model == 'GAE':
            n_layers = trial.suggest_categorical('n_layers', args.n_layers)
            reduction = trial.suggest_categorical('reduction', args.reduction)
            print(f"Trial: {trial.number}", flush=True)
            print(f"- N Epochs: {N_epochs}", flush=True)
            print(f"- N Layers: {n_layers}", flush=True)
            print(f"- Reduction: {reduction}", flush=True)
            print(f"- Learing rate: {lr}", flush=True)
       
        elif args.model == 'GUNET':

            hidden_channels = trial.suggest_categorical('hidden_channels', args.hidden_channels)
            depth = trial.suggest_categorical('depth', args.depth)
            pool_ratio = trial.suggest_categorical('pool_ratio', args.pool_ratio)
            print(f"Trial: {trial.number}", flush=True)
            print(f"- N Epochs: {N_epochs}", flush=True)
            print(f"- Latent dim: {hidden_channels}", flush=True)
            print(f"- Depth: {depth}", flush=True)
            print(f"- Pool ratio: {pool_ratio}", flush=True)
            print(f"- Learing rate: {lr}", flush=True)
        ###

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()
            
        if args.model == 'GAE':
            model = models.GCNAE(n_timestamps, n_layers, reduction)
            if model.layer_dims[-1] <= 2:
                raise optuna.TrialPruned()
        elif args.model == 'GUNET':
            model = models.GUNET(n_timestamps, hidden_channels, n_timestamps, depth, pool_ratio)

        model = model.to(device)

        auc_list, f1_list, mcc_list = evaluate_model(model, datasets, N_epochs, lr, device, db_conn)
        
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
    parser.add_argument('--model', type=str, default='GAE')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/')
    parser.add_argument('--log_mod', type=str, default='')

    parser.add_argument('--N_epochs', type=int, nargs='+', default=[5,10,25,50,100,150,200,300,500,750,1000,1250])
    parser.add_argument('--n_layers', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--reduction', type=float, nargs='+', default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4])

    parser.add_argument('--hidden_channels', type=int, nargs='+', default=[2, 5, 15, 50])
    parser.add_argument('--depth', type=int, nargs='+', default=[2, 3, 4, 5])
    parser.add_argument('--pool_ratio', type=float, nargs='+', default=[0.25, 0.5, 0.75, 0.9])

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    parser.add_argument('--ow_db', action='store_true', default=False) 

    args = parser.parse_args()

    main(args)
