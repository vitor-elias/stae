# TEST CODE FOR ALL PYOD MODELS, TRADITIONAL AND DL

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
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.anogan import AnoGAN

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

def test_model(model, datasets):

    graph_seed_list = []
    sample_list = []
    auc_list = []
    f1_list = []
    mcc_list = []

    it = 0
    for dataset in datasets:

        print(f'Testing dataset {it}', flush=True)
        it+=1

        G = dataset['G']
        data = dataset['data']
        labels = dataset['labels']

        graph_seed = dataset['metadata']['SEED']
        n_samples = dataset['metadata']['samples']

        for sample in range(n_samples):
            rng_seed = 0
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed(rng_seed)
            np.random.seed(rng_seed)

            X = torch.tensor(data[sample]).float()
            label = labels[sample]

            scores = scores = try_pyod(model, X)

            graph_seed_list.append(graph_seed)
            sample_list.append(sample)
            auc_list.append(get_auc(scores, label).round(3))
            f1_list.append(best_f1score(scores, label).round(3))
            mcc_list.append(best_mcc(scores, label).round(3))
    
    df_test = pd.DataFrame({'graph_seed':graph_seed_list,
                            'sample_id':sample_list,
                            'auc':auc_list,
                            'f1score':f1_list,
                            'mcc':mcc_list})

    return df_test


def main(args):

    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    
    if args.device=='auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)
    print(f'log dir: {args.log_dir}', flush=True)
    print(f'model: {args.model}', flush=True)
    print(f'dataset: {args.datafile}', flush=True)
    print(f'------', flush=True)


    # OBTAINING DATA
    datafile = args.datafile
    datasets = torch.load(data_dir + datafile)[5:]  #Testing on unused datasets (5+)

    if datafile == 'synthetic_timeseries.pt':
        datalog = 'ST_'
    elif datafile == 'synthetic_binary.pt':
        datalog = 'SB_'
    elif datafile == 'synthetic_InSAR.pt':
        datalog = 'SI_'

    study = joblib.load(root_dir+f'/outputs/HP_training/{datalog}{args.model}.pkl')
    best_params = study.best_params

    ###
    for item in best_params: print(item,': ', best_params[item])
    ###

    if args.model=='KNN':
        model = KNN(method='mean', n_neighbors=best_params['n_neighbors'], contamination=best_params['contamination'])
    elif args.model=='LOF':
        model = LOF(n_neighbors=best_params['n_neighbors'], contamination=best_params['contamination'])
    elif args.model=='COF':
        model = COF(n_neighbors=best_params['n_neighbors'], contamination=best_params['contamination'])
    elif args.model=='CBLOF':
        model = CBLOF(n_clusters=best_params['n_clusters'], contamination=best_params['contamination'],
                      alpha=best_params['alpha'], beta=best_params['beta'])
    elif args.model=='KPCA':
        model = KPCA(n_components=best_params['n_components'], kernel=best_params['kernel'])
    elif args.model == 'AE':
        hidden_neuron_list = [int(best_params['init_layer']*best_params['reduction_layer']**i )
                              for i in range(best_params['n_layers'])]

        model = AutoEncoder(batch_size=best_params['batch_size'],
                            epoch_num=best_params['N_epochs'],
                            lr=best_params['lr'],
                            dropout_rate=best_params['dropout_rate'],
                            hidden_neuron_list=hidden_neuron_list,
                            device=device,
                            verbose=0
                            )
    elif args.model == 'GAN':
        hidden_neuron_list_G = [int(best_params['init_layer_G']*best_params['reduction_layer_G']**i )
                              for i in range(best_params['n_layers_G'])]
        hidden_neuron_list_G.extend(hidden_neuron_list_G[:best_params['n_layers_G']-1][::-1])

        hidden_neuron_list_D = [int(best_params['init_layer_D']*best_params['reduction_layer_D']**i )
                              for i in range(best_params['n_layers_D'])]

        model = AnoGAN(batch_size=best_params['batch_size'],
                    epochs=best_params['N_epochs'],
                    learning_rate=best_params['lr'],
                    dropout_rate=best_params['dropout_rate'],
                    G_layers=hidden_neuron_list_G,
                    D_layers=hidden_neuron_list_D,
                    device=device,
                    verbose=0
                    )
        
    df_test = test_model(model, datasets)
       
    print(df_test[['auc','f1score','mcc']].mean(), flush=True)

    warnings.filterwarnings("ignore")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)

    if args.log_mod == '':
        log_file = args.log_dir + datalog + args.model + '.parq'
    else:
        log_file = args.log_dir + datalog + args.model + '_' + args.log_mod + '.parq'


    df_test.to_parquet(log_file)

    print(f"End: {datetime.now()}\n\n", flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='KNN')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/testing_mincut/')
    parser.add_argument('--log_mod', type=str, default='')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)
