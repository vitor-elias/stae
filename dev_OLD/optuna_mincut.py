# Hyperparameter training for the proposed method

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

def skewscore(scores, skewth):
    sk = skew(scores)
    return (1-scores) if (sk<skewth) else scores

def dict_hash(d):
    return hashlib.md5(str(sorted(d.items())).encode()).hexdigest()

def train_model(model, X, G, device, weight_loss, lr):

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    loss_evo = []
    loss_mc_evo = []
    loss_o_evo = []
    S_partials = []

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
        loss_evo.append(loss.item())
        loss_mc_evo.append(loss_mc.item())
        loss_o_evo.append(loss_o.item())
        if epoch in epochs_list:
            S_partials.append(S)

    return S_partials, loss_mc_evo, loss_o_evo

def evaluate_model(model, datasets, epochs, weight_loss, lr, skewth, device, db_conn):

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

        n_samples = 5

        for sample in range(n_samples):
            X = torch.tensor(data[sample]).float().to(device)
            label = labels[sample]

            sim_dict = {'metadata':dataset['metadata'], 'model':model, 'sample':sample,
                        'weight':weight_loss, 'lr':lr}
            sim_key = dict_hash(sim_dict)

            db_cursor.execute("SELECT S_epoch FROM simulations WHERE hash=?", (sim_key,))
            row = db_cursor.fetchone()

            if row:
                S_epoch = pickle.loads(row[0])
            else:
                S_epoch = train_model(model, X, G, device, weight_loss, lr)[0]
                db_cursor.execute("INSERT INTO simulations (hash, S_epoch, sim_info) VALUES (?, ?, ?)",
                                ( sim_key, pickle.dumps(S_epoch), pickle.dumps(sim_dict) ))
                db_conn.commit()

            epochs_id = epochs_list.index(epochs)
            scores = S_epoch[epochs_id].detach().cpu().softmax(dim=1).max(dim=1)[0].numpy()
            scores = skewscore(scores, skewth)
            auc_list.append(get_auc(scores, label).round(3))
            f1_list.append(best_f1score(scores, label).round(3))
            mcc_list.append(best_mcc(scores, label).round(3))

    return auc_list, f1_list, mcc_list


def main(args):

    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)

    db_name = 'simulation_cache_b.db'
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
    db_cursor.execute('''CREATE TABLE IF NOT EXISTS simulations (hash TEXT PRIMARY KEY, S_epoch BLOB, sim_info BLOB)''')
    db_conn.commit()

    def objective(trial):

        gc.collect()

        # Parameters     
        n_timestamps = datasets[0]['metadata']['T']
        N_epochs = trial.suggest_categorical('N_epochs', args.N_epochs)
        weight_loss = trial.suggest_categorical('weight_loss', args.weight_loss)
        n_clusters = trial.suggest_categorical('n_clusters', args.n_clusters)
        skewth = trial.suggest_categorical('skewth', args.skewth)
        lr = trial.suggest_categorical('lr', args.lr)

        ###

        print(f"Trial: {trial.number}", flush=True)
        print(f"- N Epochs: {N_epochs}", flush=True)
        print(f"- N Clusters: {n_clusters}", flush=True)
        print(f"- Learing rate: {lr}", flush=True)
        print(f"- Weight loss: {weight_loss}", flush=True)
        print(f"- Skew Threshold: {skewth}", flush=True)

        ###

        if args.model == 'MCconv':
            conv_n_feats = trial.suggest_categorical('conv_n_feats', args.conv_n_feats)
            conv_kernel_size = trial.suggest_categorical('conv_kernel_size', args.conv_kernel_size)
            conv_stride = trial.suggest_categorical('conv_stride', args.conv_stride)
            print(f"- Convolution feats: {conv_n_feats}", flush=True)
            print(f"- Convolution kernel size: {conv_kernel_size}", flush=True)
            print(f"- Convolution stride: {conv_stride}", flush=True)

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()
            
        
        if args.model == 'MC':
            model = models.ClusterTS(n_timestamps, n_clusters)
        elif args.model == 'MCconv':
            model = models.ClusterTSconv(conv_n_feats, conv_kernel_size, conv_stride, n_timestamps, n_clusters)
        
        model = model.to(device)

        auc_list, f1_list, mcc_list = evaluate_model(model, datasets, N_epochs, weight_loss, lr, skewth, device, db_conn)
        
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

    if datafile == 'synthetic_timeseries.pt':
        datalog = 'ST_'
    elif datafile == 'synthetic_binary.pt':
        datalog = 'SB_'
    elif datafile == 'synthetic_InSAR.pt':
        datalog = 'SI_'
    elif datafile == 'synthetic_InSARfull.pt':
        datalog = 'SF_'


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
    parser.add_argument('--model', type=str, default='MC')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/')
    parser.add_argument('--log_mod', type=str, default='')


    # PARAMETERS CONVOLUTION
    parser.add_argument('--N_epochs', type=int, nargs='+', default=[5,10,25,50,100,150,200,300,500,750,1000,1250])
    parser.add_argument('--n_clusters', type=float, nargs='+', default=[3,5,7,10])
    parser.add_argument('--weight_loss', type=float, nargs='+', default=[0.6, 0.8, 1])
    parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4])
    parser.add_argument('--skewth', type=float, nargs='+', default=[-np.inf, 0, 0.15, 0.3, 0.45])

    parser.add_argument('--conv_n_feats', type=int, nargs='+', default=[3, 4, 5, 6])
    parser.add_argument('--conv_kernel_size', type=int, nargs='+', default=[10,20,30])
    parser.add_argument('--conv_stride', type=int, nargs='+', default=[5,10,15])

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    parser.add_argument('--ow_db', action='store_true', default=False) 

    args = parser.parse_args()

    main(args)
