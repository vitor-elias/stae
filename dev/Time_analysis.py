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
import time 

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

def evaluate_time(model, X, lr, n_epochs, edge_index=None, edge_weight=None, batch_size=None):
    rng_seed = 0
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model.train()
    model.reset_parameters()

    start_time = time.time()
    if isinstance(model, models.RAE):
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = model(X.T.unsqueeze(0)).squeeze(0).T
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

            scores = pixel_mse(output, X).detach().cpu().numpy()


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

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_epoch = elapsed_time / n_epochs

    # Inference test

    model.eval()
    with torch.no_grad():
        inference_start_time = time.time()

        if isinstance(model, models.RAE):
            output = model(X.T.unsqueeze(0)).squeeze(0).T
        elif isinstance(model, graph_models):
            output = model(X, edge_index=edge_index, edge_weight=edge_weight)
        else:
            output = model(X)

        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

    return elapsed_time, time_per_epoch, inference_time

def main():

    dataset_name = 'EGMS_anomaly'
    datasets = torch.load(dataset_path + f'{dataset_name}/Training/dataset.pt')
    model_dict = torch.load(root_dir + f'/outputs/Optuna_analysis/model_dict_{dataset_name}.pkl')

    n_runs = 10

    for model_name in model_dict.keys():
        # Iterate through datasets
        time_total = []
        time_epoch = []
        time_inference = []
        for idx, dataset in enumerate(datasets, start=1):
            print(f"\rProcessing dataset {idx}/{len(datasets)} for model {model_name}", end="", flush=True)

            data = dataset['data']
            X = torch.tensor(data).float().to(device)

            model = model_dict[model_name]['model'].to(device)

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

            time_total_run = []
            time_epoch_run = []
            time_inference_run = []
            
            for run in range(n_runs):
                elapsed_time, time_per_epoch, inference_time = evaluate_time(model, X,
                                                            lr=model_dict[model_name]['trial_params']['lr'],
                                                            n_epochs=model_dict[model_name]['trial_params']['n_epochs'],
                                                            edge_index=edge_index,
                                                            edge_weight=edge_weight)

                time_total_run.append(elapsed_time)
                time_epoch_run.append(time_per_epoch)
                time_inference_run.append(inference_time)
            
            time_total.append(time_total_run)
            time_epoch.append(time_epoch_run)
            time_inference.append(time_inference_run)
            
        model_dict[model_name]['time_total'] = time_total
        model_dict[model_name]['time_epoch'] = time_epoch
        model_dict[model_name]['time_inference'] = time_inference

    torch.save(model_dict, root_dir + f'/outputs/Optuna_analysis/model_dict_times_{dataset_name}.pkl')


if __name__ == "__main__":
    main()