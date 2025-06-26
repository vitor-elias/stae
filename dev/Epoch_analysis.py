import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import contextily as cx
import plotly.graph_objects as go
import geopandas as gpd
import os
import matplotlib
import subprocess
import torch
import joblib
import glob
import copy

from IPython.display import display_html
from shapely.geometry import MultiPoint
from sklearn.cluster import KMeans
from tsmoothie import LowessSmoother, ExponentialSmoother
from pyprojroot import here
from scipy.spatial import ConvexHull
from datetime import datetime

import source.nn.models as models
import source.utils.utils as utils
import source.utils.fault_detection as fd

from source.utils.utils import roc_params, compute_auc, get_auc, best_mcc, best_f1score, otsuThresholding
from source.utils.utils import synthetic_timeseries
from source.utils.utils import plotly_signal

from importlib import reload
models = reload(models)
utils = reload(utils)
fd = reload(fd)

from pyprojroot import here
root_dir = str(here())

insar_dir = os.path.expanduser('~/data/raw/')
data_path = root_dir + '/data/interim/'
dataset_path = root_dir + "/data/datasets/"

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'font.family': 'DejaVu Serif'})

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)

device = 'cuda:2'

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
    return torch.mean(point_mse(output,X),axis=1)

def evaluate_epochs(model, X, lr, n_epochs, label, edge_index=None, edge_weight=None, rng_seed=0):
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    model.reset_parameters()

    loss_evolution = []
    auc_evolution = []

    if isinstance(model, models.RAE):
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = model(X.T.unsqueeze(0)).squeeze(0).T
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

            scores = pixel_mse(output, X).detach().cpu().numpy()
            auc = get_auc(scores, label, resolution=101).round(3)

            loss_evolution.append(loss.item())
            auc_evolution.append(auc)

    else:
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            if isinstance(model, graph_models):
                output = model(X, edge_index=edge_index, edge_weight=edge_weight)
            else:
                output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

            scores = pixel_mse(output, X).detach().cpu().numpy()
            auc = get_auc(scores, label, resolution=101).round(3)

            loss_evolution.append(loss.item())
            auc_evolution.append(auc)

    return loss_evolution, auc_evolution


def main():

    dataset_name = 'Geological_anomaly'

    print(f"Start: {datetime.now()}", flush=True)
    print(f"Dataset: {dataset_name}", flush=True)
    print(f'on: {device}', flush=True)
    print(f'------', flush=True)

    datasets = torch.load(dataset_path + f'{dataset_name}/Training/dataset.pt')
    model_dict = torch.load(root_dir + f'/outputs/Optuna_analysis/model_dict_{dataset_name}.pkl')

    model_names = ['AE', 'GCN2MLP', 'GCNAE', 'GConv2MLP', 'GConvAE', 'GUNet', 'RAE_GRU', 'RAE_LSTM']

    for model_name in model_names:
        # Iterate through datasets
        loss_dataset = []
        auc_dataset = []
        for idx, dataset in enumerate(datasets, start=1):
            print(f"\rProcessing dataset {idx}/{len(datasets)} for model {model_name}", end="", flush=True)

            data = dataset['data']
            label = dataset['label'].any(axis=1)

            X = torch.tensor(data).float().to(device)

            n_nodes = X.shape[0]
            input_dim = X.shape[1]
            params = model_dict[model_name]['trial_params']
            
            loss_seed = []
            auc_seed = []
            for seed in range(25):

                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                np.random.seed(seed)

                model = get_model_from_name(model_name, n_nodes, input_dim, params, device=device)

                edge_index, edge_weight = (None, None)
                if isinstance(model, graph_models):
                    edge_index = dataset['edge_index'].to(device)
                    edge_weight = dataset['edge_weight'].to(device)

                loss, auc = evaluate_epochs(model, X,
                                            lr=model_dict[model_name]['trial_params']['lr'],
                                            n_epochs=150,
                                            label=label,
                                            edge_index=edge_index,
                                            edge_weight=edge_weight,
                                            rng_seed=seed)
                loss_seed.append(loss)
                auc_seed.append(auc)

            loss_dataset.append(loss_seed)
            auc_dataset.append(auc_seed)

        print('\n')

        model_dict[model_name]['loss_evolution'] = loss_dataset
        model_dict[model_name]['auc_evolution'] = auc_dataset

    torch.save(model_dict, root_dir + f'/outputs/Optuna_analysis/Epochs_{dataset_name}.pkl')

    print(f"End: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()