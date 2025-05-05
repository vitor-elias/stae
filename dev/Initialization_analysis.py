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

device = 'cuda:3'

graph_models = tuple([models.GCN2MLP, models.GConv2MLP, models.GCNAE, models.GConvAE, models.GUNet])


def pixel_mse(output,X):
    point_mse = torch.nn.MSELoss(reduction='none')
    return torch.mean(point_mse(output,X),axis=1)

def evaluate_epochs(model, X, lr, n_epochs, label, edge_index=None, edge_weight=None, batch_size=None, rng_seed=0):
    
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
            # auc = get_auc(scores, label, resolution=101).round(3)

            # loss_evolution.append(loss.item())
            # auc_evolution.append(auc)

    return scores


def main():

    dataset_name = 'Geological_anomaly'
    datasets = torch.load(dataset_path + f'{dataset_name}/Training/dataset.pt')
    model_dict = torch.load(root_dir + f'/outputs/Optuna_analysis/model_dict_{dataset_name}.pkl')

    model_names = ['AE', 'GCN2MLP', 'GCNAE', 'GConv2MLP', 'GConvAE', 'GUNet', 'RAE_GRU', 'RAE_LSTM']

    auc_results = {}

    for model_name in model_names:  
        # Iterate through datasets
        loss_dataset = []
        auc_dataset = []
        for idx, dataset in enumerate(datasets, start=1):
            print(f"Processing dataset {idx}/{len(datasets)} for model {model_name}", flush=True)

            data = dataset['data']
            label = dataset['label'].any(axis=1)

            X = torch.tensor(data).float().to(device)

            auc_seed = []
            for seed in range(25):

                model = copy.deepcopy(model_dict[model_name]['model']).to(device)

                edge_index, edge_weight = (None, None)
                if isinstance(model, graph_models):
                    edge_index = dataset['edge_index'].to(next(model.parameters()).device)
                    edge_weight = dataset['edge_weight'].to(next(model.parameters()).device)
                if isinstance(model, models.RAE) and (model.n_features != 1):
                    relevant_params = ['n_features', 'latent_dim', 'rnn_type', 'rnn_act', 'device']
                    new_model_params = {key: getattr(model, key) for key in relevant_params}
                    new_model_params['n_features'] = X.shape[0]
                    new_model_params['device'] = device
                    model = models.RAE(**new_model_params)
                    model.to(new_model_params['device'])

                scores = evaluate_epochs(model, X,
                                            lr=model_dict[model_name]['trial_params']['lr'],
                                            n_epochs=model_dict[model_name]['trial_params']['n_epochs'],
                                            label=label,
                                            edge_index=edge_index,
                                            edge_weight=edge_weight,
                                            rng_seed=seed)

                auc = get_auc(scores, label, resolution=101).round(3)
                auc_seed.append(auc)

            auc_dataset.append(auc_seed)
        
        auc_results[model_name] = auc_dataset

    torch.save(auc_results, root_dir + f'/outputs/Optuna_analysis/initialization_{dataset_name}.pkl')


if __name__ == "__main__":
    main()