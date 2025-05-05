# Test code for the proposed method

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

from scipy.stats import entropy, kurtosis, skew
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

def dscore(scores):
    smean = np.mean(scores)
    smin = np.min(scores)
    smax = np.max(scores)
    return (1-scores) if (smean-smin)>(smax-smean) else scores

def skewscore(scores, skewth):
    sk = skew(scores)
    return (1-scores) if (sk<skewth) else scores

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

def test_model(model, datasets, epochs, weight_loss, lr, skewth, device):

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
            X = torch.tensor(data[sample]).float().to(device)
            label = labels[sample]

            S_epoch = train_model(model, X, G, device, weight_loss, lr)[0]

            epochs_id = epochs_list.index(epochs)
            scores = S_epoch[epochs_id].detach().cpu().softmax(dim=1).max(dim=1)[0].numpy()

            scores = skewscore(scores, skewth)

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

    # Print the current date and time
    print(f"Start: {datetime.now()}\n", flush=True)
    print(f'Using {args.device}', flush=True)
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


    if args.model == 'MCconv':
        model = models.ClusterTSconv(conv1d_n_feats = best_params['conv_n_feats'],
                                     conv1d_kernel_size = best_params['conv_kernel_size'],
                                     conv1d_stride = best_params['conv_stride'],
                                     n_clusters = best_params['n_clusters'],
                                     n_timestamps = datasets[0]['metadata']['T'],
                                    )
   
    elif args.model == 'MC':
        model = models.ClusterTS(datasets[0]['metadata']['T'], n_clusters=best_params['n_clusters'])

    model = model.to(args.device)

    df_test = test_model(model, datasets,
                         epochs=best_params['N_epochs'],
                         weight_loss=best_params['weight_loss'],
                         lr=best_params['lr'],
                         skewth=best_params['skewth'],
                         device=args.device
                        )
       
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
    parser.add_argument('--model', type=str, default='MC')
    parser.add_argument('--datafile', type=str, default='synthetic_timeseries.pt')

    parser.add_argument('--log-dir', type=str, default=root_dir+'/outputs/testing_mincut/')
    parser.add_argument('--log_mod', type=str, default='')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(args)
