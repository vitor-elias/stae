# HYPERPARAMETER TUNING FOR TRADITIONAL MODELS AVAILABLE IN PYOD
# KNN, COF, LOF, CBLOF, KPCA

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

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.kpca import KPCA

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

        if args.model in ('KNN', 'LOF', 'COF'):
            n_neighbors = trial.suggest_categorical('n_neighbors', args.n_neighbors)
            contamination = trial.suggest_categorical('contamination', args.contamination)
            print(f"- N Neighbors: {n_neighbors}", flush=True)
            print(f"- Contamination: {contamination}", flush=True)

        elif args.model == 'CBLOF':
            n_clusters = trial.suggest_categorical('n_clusters', args.n_clusters)
            contamination = trial.suggest_categorical('contamination', args.contamination)
            alpha = trial.suggest_categorical('alpha', args.alpha)
            beta = trial.suggest_categorical('beta', args.beta)
            print(f"- N Clusters: {n_clusters}", flush=True)
            print(f"- Contamination: {contamination}", flush=True)
            print(f"- Alpha: {alpha}", flush=True)
            print(f"- Beta: {beta}", flush=True)

        elif args.model == 'KPCA':
            n_components = trial.suggest_categorical('n_components', args.n_components)
            kernel = trial.suggest_categorical('kernel', args.kernel)
            print(f"- N Components: {n_components}", flush=True)
            print(f"- Kernel: {kernel}", flush=True)
        ###

        for completed_trial in trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            if completed_trial.params == trial.params:
                raise optuna.TrialPruned()

        if args.model=='KNN':
            model = KNN(method='mean', n_neighbors=n_neighbors, contamination=contamination)
        elif args.model=='LOF':
            model = LOF(n_neighbors=n_neighbors, contamination=contamination)
        elif args.model=='COF':
            model = COF(n_neighbors=n_neighbors, contamination=contamination)
        elif args.model=='CBLOF':
            model = CBLOF(n_clusters=n_clusters, contamination=contamination, alpha=alpha, beta=beta)
        elif args.model=='KPCA':
            model = KPCA(n_components=n_components, kernel=kernel)
        
        auc_list, f1_list, mcc_list = evaluate_model(model, datasets)
        
        trial.set_user_attr("f1", np.mean(f1_list).round(3))
        trial.set_user_attr("mcc", np.min(mcc_list).round(3))
        trial.set_user_attr("med", np.median(auc_list).round(3))
        trial.set_user_attr("auc_list", [round(elem, 2) for elem in auc_list])

        return np.mean(auc_list).round(3)

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

    study = optuna.create_study(sampler=BruteForceSampler(), direction='maximize',
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

    
    if args.model in ('KNN', 'LOF', 'COF'):
        n_trials = len(args.n_neighbors)*len(args.contamination)
    elif args.model == 'CBLOF':
        n_trials = len(args.n_clusters)*len(args.contamination)*len(args.alpha)*len(args.beta)
    elif args.model == 'KPCA':
        n_trials = len(args.n_components)*len(args.kernel)

    print(args.model)

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    joblib.dump(study, log_file)

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":

    neighbors_list = [20,25,30,35,40,45,50,55,60,65,70]
    contamination_list = [0.1,0.2,0.3,0.4,0.5]

    clusters_list = [5, 7, 10]
    alpha_list = [0.5, 0.75, 0.9]
    beta_list = [3, 5, 7]

    components_list = [10, 20, 30, 40, 50, 70, 100]
    kernel_list = ['poly', 'rbf', 'sigmoid']

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KNN')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/HP_training/')
    parser.add_argument('--log_mod', type=str, default='')
    
    parser.add_argument('--n_neighbors', type=int, nargs='+', default=neighbors_list)
    parser.add_argument('--contamination', type=float, nargs='+', default=contamination_list)

    parser.add_argument('--n_clusters', type=int, nargs='+', default=clusters_list)
    parser.add_argument('--alpha', type=float, nargs='+', default=alpha_list)
    parser.add_argument('--beta', type=int, nargs='+', default=beta_list)

    parser.add_argument('--n_components', type=int, nargs='+', default=components_list)
    parser.add_argument('--kernel', type=str, nargs='+', default=kernel_list)

    parser.add_argument('--reuse', action='store_true', default=True)
    parser.add_argument('--no-reuse', dest='reuse', action='store_false')

    args = parser.parse_args()

    main(args)
