import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from ipywidgets import Output

import pygsp
import torch

from sklearn.metrics import mean_squared_error, confusion_matrix, auc, f1_score, matthews_corrcoef
from sklearn.cluster import KMeans

import source.utils.fault_detection as fd



########## VISUALIZATION ##############


def plotly_signal(G, signal, title=None, color_scale="Viridis", edge_width=0.5, node_size=10,
                  width=1000, height=1000):
    """
    Replicates pygsp's plot_signal using Plotly for a pygsp graph, using G.get_edge_list().
    
    Parameters:
    - G (pygsp.graphs.Graph): The input graph (with node coordinates).
    - signal (numpy array): The signal values to plot for each node.
    - title (str): Title of the plot.
    - color_scale (str): Color scale for the node colors.
    
    Returns:
    - A plotly graph.
    """
    if not hasattr(G, 'coords'):
        raise ValueError("Graph must have coordinates for visualization.")
    
    # Extract coordinates and signal values
    x, y = G.coords[:, 0], G.coords[:, 1]
    
    # Create scatter plot for the nodes
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=node_size,
            color=signal,  # Color node based on signal value
            colorscale=color_scale,  # Choose color scale
            colorbar=dict(title='Signal'),
            showscale=True,
            line_width=2
        ),
        text=[f"Node {i}: {signal_val}" for i, signal_val in enumerate(signal)],  # Hover text
        hoverinfo='text'
    )
    
    # Get edges from G.get_edge_list()
    edges = G.get_edge_list()
    edge_rows, edge_cols, weights = edges
    
    # Create edge trace for the graph
    edge_x, edge_y = [], []
    
    for i, j in zip(edge_rows, edge_cols):
        edge_x.extend([G.coords[i, 0], G.coords[j, 0], None])  # x-coordinates of edge
        edge_y.extend([G.coords[i, 1], G.coords[j, 1], None])  # y-coordinates of edge

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=edge_width, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create the figure and add the edge and node traces
    fig = go.Figure(data=[edge_trace, node_trace])

    # Set title and axis properties
    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=15, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width,
        height=height
    )

    fig.show()


def visualize_map(nodes_input, color='smoothed', size=None, size_max = 5, animation_frame=None, hover_data=[None],
                  colormap = px.colors.diverging.oxy, zoom=15, range_color = None, opacity=1, return_fig=False,
                  title=None, transparent=False, discrete_colormap = px.colors.qualitative.Light24, color_title=None,
                  figsize=(1200,800), renderer=None, mapbox_style='carto-positron', savehtml=None):
    
    columns = [col for col in ['pid', 'easting','northing',color, animation_frame]+hover_data if col is not None]
    columns = list(set(columns))

    if isinstance(size,str):
        columns.append(size)

    nodes = nodes_input[columns].copy()

    nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.easting,nodes.northing), crs='3035')
    nodes_gdf = nodes_gdf.to_crs('4326')

    if animation_frame is not None:
        nodes_gdf[animation_frame] = nodes_gdf[animation_frame].astype(str)
        nodes_gdf = nodes_gdf.sort_values([animation_frame, 'pid'])

    if range_color is None:
        range_color = (nodes_gdf[color].min(), nodes_gdf[color].max())

    if size is None:
        size = np.ones(nodes_gdf.shape[0])
        
    fig = px.scatter_mapbox(nodes_gdf, lat=nodes_gdf.geometry.y, lon=nodes_gdf.geometry.x,
                            hover_name = 'pid', hover_data = hover_data, opacity=opacity,
                            color=color, size=size, size_max = size_max,
                            mapbox_style=mapbox_style, animation_frame=animation_frame,
                            width=figsize[0], height=figsize[1], zoom=zoom, color_discrete_sequence=discrete_colormap,
                            color_continuous_scale=colormap, range_color=range_color,
                            )
    
    if color_title is None:
        color_title = color

    # cbar_y = 0.775 if animation_frame is not None else 0.9
    fig.update_layout(coloraxis={'colorbar': {'title': {'text':color_title, 'side':'right', 'font':{'size':16}},
                                            'tickfont':{'size':14},
                                            'len':1,
                                            'yanchor':'middle',
                                            # 'y':cbar_y,
                                            'thickness':10
                                            }})
    fig.update_layout(title=title)
    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    if savehtml is not None:
        fig.write_html(savehtml)

    if return_fig:
        return fig
    else:
        fig.show(renderer=renderer)



################# METRICS AND ANALYSIS ##########################

def roc_params(metric, label, interp=True, resolution=1001):
    fpr = []
    tpr = []
    thr = []
    thr_list = list(np.linspace(0, metric.max(),resolution))

    fp = 1
    ind = 0
    while fp > 0:
        threshold = thr_list[ind]
        ind += 1

        y = (metric>threshold)
        tn, fp, fn, tp = confusion_matrix(label, y).ravel()

        fpr.append( fp/(tn + fp) )
        tpr.append( tp/(tp + fn) )
        thr.append( threshold )

    while tp > 0:
        threshold = thr_list[ind]
        ind += 1
        y = (metric>threshold)
        tn, fp, fn, tp = confusion_matrix(label, y).ravel()

    
    fpr = fpr[::-1]
    tpr = tpr[::-1]
    thr = thr[::-1]

    if interp:
        fpr_base = np.linspace(0, 1, 101)
        tpr = list(np.interp(fpr_base, fpr, tpr))
        thr = list(np.interp(fpr_base, fpr, thr))
        fpr = list(fpr_base)

    fpr.insert(0, 0)
    tpr.insert(0, 0)
    thr.insert(0, threshold)

    return tpr, fpr, thr

def compute_auc(tpr, fpr):
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc

def get_auc(score, labels, resolution=1001):    
    tpr, fpr, _ = roc_params(metric=score, label=labels.reshape(-1,), interp=True, resolution=resolution)
    return compute_auc(tpr,fpr)

def best_precision_max_recall(metric, label, interp=True):
	thr_list = list(np.linspace(metric.min(), metric.max(),101))

	precision = []
	recall = []
	for threshold in thr_list[0:-1]:
		y = (metric>threshold)
		tn, fp, fn, tp = confusion_matrix(label, y).ravel()
		precision.append(tp/(fp+tp))
		recall.append(tp/(fn+tp))
	
	sorted_pairs = sorted(zip(precision, recall), key=lambda x: (x[1], x[0]), reverse=True)
	sorted_precision, sorted_recall = zip(*sorted_pairs)

	return sorted_precision[0].round(3), sorted_recall[0].round(3)

def best_f1score(metric, label, interp=True):
    thr_list = list(np.linspace(metric.min(), metric.max(),101))

    f1score = []
    for threshold in thr_list[0:-1]:
        y = (metric>threshold)
        f1score.append(f1_score(label, y))

    return np.max(f1score).round(3)

def best_mcc(metric, label, interp=True):
    thr_list = list(np.linspace(metric.min(), metric.max(),101))

    mcc = []
    for threshold in thr_list[0:-1]:
        y = (metric>threshold)
        mcc.append(matthews_corrcoef(label, y))

    return np.max(mcc).round(3)


def get_score(nodes, df_anomaly, S):

    nodes['pred'] = S.argmax(dim=1).cpu().numpy()
    nodes['score'] = S.softmax(dim=-1).detach().cpu().numpy().max(axis=1)
    nodes['anomaly'] = df_anomaly[['pid','anomaly']].groupby('pid').anomaly.max().values

    most_common_preds = nodes.query('anomaly!=0').groupby('anomaly')['pred'].apply(lambda x: x.mode()[0])

    nodes['new_pred'] = nodes['pred']
    nodes.loc[~nodes.pred.isin(most_common_preds.values),'new_pred'] = -1

    max_anomaly = nodes.groupby('new_pred')['anomaly'].transform('max')
    nodes.loc[nodes['new_pred'] != -1, 'new_pred'] = max_anomaly
    nodes.loc[nodes['new_pred'] == -1, 'new_pred'] = 0

    average = 'binary' if df_anomaly.anomaly.nunique()==2 else 'weighted'
    cluster_score = f1_score(y_true=nodes.anomaly, y_pred=nodes.new_pred, average=average)

    tpr, fpr, _ = roc_params(metric=nodes.score, label=(nodes.anomaly>0), interp=True)
    auc = compute_auc(tpr,fpr)

    return cluster_score, auc#, nodes

def otsuThresholding(img, num_thresholds=100):
    img = np.array(img)
    min_val, max_val = np.min(img), np.max(img)
    min_val, max_val = (0,1-(1/num_thresholds))
    # Generate threshold range for float values
    threshold_range = np.linspace(min_val, max_val, num_thresholds)
    criterias = np.array([_compute_otsu_criteria(img, th) for th in threshold_range])

    # Best threshold is the one minimizing the Otsu criteria

    if np.min(criterias)==np.inf:
        best_thresholds=np.array(0)
    else:        
        best_thresholds = threshold_range[np.where(criterias==np.min(criterias))]
    # best_threshold = threshold_range[np.argmin(criterias)]

    return best_thresholds, np.min(criterias)

def _compute_otsu_criteria(im, th):
    # Create the thresholded image
    thresholded_im = im >= th

    # Compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # If one of the classes is empty, that threshold is not considered
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # Find all pixels belonging to each class
    val_pixels1 = im[thresholded_im]
    val_pixels0 = im[~thresholded_im]

    # Compute variance of these classes
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0

    return weight0 * var0 + weight1 * var1


##################       DATA GENERATION         ####################################

def generate_synthetic_graph(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    N_grid_points = int(np.floor(0.9*N))
    N_rand_points = int(np.ceil(0.1*N))

    connected = False

    while not connected:

        # Generating grid points

        # Initialize starting position
        current_position = [0, 0]

        # Store visited points
        visited_points = set()
        visited_points.add(tuple(current_position))

        # Generate N points
        while len(visited_points) < N_grid_points:
            # Update the current position by taking a vertical or horizontal step
            # horizontal can be +-5, vertical can be +-15

            if np.random.rand()>0.5:
                current_position[0] += np.random.choice([-5, 5])
            else:
                current_position[1] += np.random.choice([-14, 14])

            # Add the new position to the set of visited points
            visited_points.add(tuple(current_position))

        points_list = list(visited_points)

        # Generating random points
        reference_pos = np.random.choice(a=N_grid_points, size=N_rand_points, replace=False)
        reference_points = [list(points_list[i]) for i in reference_pos]

        for point in reference_points:
            point[0] += 10*np.random.random()
            point[1] += 10*np.random.random()
            visited_points.add(tuple(point))
        

        # Convert the set of visited points to a list
        pos = np.array(list(visited_points))
        pos = pos + 0.5*np.random.randn(pos.shape[0], pos.shape[1])

        radius=15
        sigma=radius**2

        G = pygsp.graphs.NNGraph(pos,
                                NNtype='radius',
                                epsilon = radius, sigma = sigma,
                                center=False, rescale=False)
        connected = G.is_connected()

    return G


def generate_ramp(G, size):
    # Auxiliary function for generating data.
    # Returns array with "size" random ramps on the given graph. A ramp emulates regions with different behaviors.
    # Direction of each ramp is randomly horizontal or vertical
    # Slope, start and end positions of the ramp are random, within some constraints.
    
    pos = G.coords
    min_slope = 0.1
    max_slope = 1
    
    # Initialize an empty matrix to store ramp vectors
    ramp_matrix = np.zeros((len(pos), size))

    for i in range(size):
        # Randomly choose the direction of the ramp (0, horizontal or 1, vertical)
        if np.random.rand() > 0.5:
            direction = 0
        else:
            direction = 1

        # Generate a random slope within the specified maximum slope
        slope = np.random.uniform(low=min_slope, high=max_slope)

        # Calculate the minimum ramp length as a fraction of the peak-to-peak range
        min_slope_dist = 0.25 * pos[:, direction].ptp()

        # Determine the starting and ending positions
        start = pos[:, direction].min() + pos[:, direction].ptp() * np.random.rand() * 0.5
        end = start + min_slope_dist + (pos[:, direction].max() - start - min_slope_dist) * np.random.rand()

        # Generate the ramp vector based on the slope and the position within the range
        ramp = slope * (pos[:, direction] - start)

        # Applying a mask that defines where the ramp exists
        mask = (pos[:, direction] >= start) * (pos[:, direction] < end)
        ramp = ramp * mask

        # Set values beyond the end of the ramp to the maximum ramp value
        ramp[pos[:, direction] >= end] = ramp.max()

        # Store the ramp vector in the matrix
        ramp_matrix[:, i] = ramp

    return ramp_matrix


def generate_smooth(G, size):
    # Auxiliary function for generating data
    # Returns array with "size" smooth signals by filtering white noise with the two first eigenvectors of the graph
    
    w, V = np.linalg.eigh(G.L.toarray())

    # Low-pass filter
    h = np.ones(len(w))
    h[0] = 1
    h[1] = 0.1
    h[2:] = 0 

    # Generating and filtering white noise to create a smooth graph signal
    displacement = np.random.randn(G.N, size) 
    displacement = V @ np.diag(h) @ V.T @ displacement

    # Normalizing signal: average power sum(x^2)/N = 1
    displacement = np.sqrt(G.N)*displacement/(np.linalg.norm(displacement,axis=0))

    return displacement


def hfilter(G, cut=2):
    L = G.L.toarray()
    w, V = np.linalg.eigh(L)
    wh = np.ones(G.N)
    wh[w<cut] = 0
    Hh = V @ np.diag(wh) @ V.T
    return Hh

def generate_cluster_anomaly(df, nodes, G, data_size=10, partition=20, anomaly_level=10, n_anomalies=1):

    nodes['cluster'] = KMeans(n_clusters=partition, n_init='auto').fit_predict(nodes[['northing','easting']])

    df.drop('cluster', axis=1, inplace=True)
    df = df.merge(nodes[['pid','cluster']], how='left', on='pid')

    X = []
    label = []
    df_anomaly_list = []

    for sample in range(data_size):
        df_anomaly = df[['timestamp','pid','cluster','smoothed']].copy()
        df_anomaly['anomaly'] = 0

        anomalous_clusters = np.random.choice(nodes.cluster.unique(), size=n_anomalies)

        for index, cluster in enumerate(anomalous_clusters):
            index = index+1

            anomaly_sensor = (df_anomaly.cluster==cluster)
            anomaly_period = (df_anomaly.timestamp>'Jul 2020')&(df_anomaly.timestamp<'Jan 2021')
            anomaly_loc = anomaly_sensor&anomaly_period

            df_anomaly.loc[anomaly_loc, 'smoothed'] += anomaly_level
            df_anomaly.loc[anomaly_loc, 'anomaly'] = index
        
        X.append(df_anomaly.pivot(index='pid', columns='timestamp', values='smoothed').values)
        label.append(df_anomaly.pivot(index='pid', columns='timestamp', values='anomaly').values.max(axis=1))
        df_anomaly_list.append(df_anomaly)

    X = np.array(X)
    label = np.array(label)

    return X, label, df_anomaly_list

def generate_data(df_orig, graph_clusters, select_cluster=0, samples=10, anomalous_nodes=20,
                  anomaly_level=10, n_anomalies=1, noise=0, label_noise=False):

    df = df_orig.copy()

    df['graph_cluster'] = KMeans(n_clusters=graph_clusters, n_init=1, max_iter=2).fit_predict(df[['northing','easting']])
    sorted_clusters = df['graph_cluster'].value_counts().sort_values(ascending=False).index
    new_labels = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}
    df['graph_cluster'] = df['graph_cluster'].map(new_labels)

    df = df[df.graph_cluster==select_cluster].copy()

    df, nodes = fd.treat_nodes(df)
    _, nodes['subgraph'] = fd.NNGraph(nodes, radius=15, subgraphs=True)

    main_graph = nodes.subgraph.value_counts().index[0]
    nodes = nodes.query('subgraph==@main_graph').copy()
    G = fd.NNGraph(nodes, radius=15)
    df = df[df.pid.isin(nodes.pid.unique())].copy()

    anomaly_clusters = int(df.pid.nunique()/anomalous_nodes)

    nodes['cluster'] = KMeans(n_clusters=anomaly_clusters, n_init='auto').fit_predict(nodes[['northing','easting']])

    df.drop('cluster', axis=1, inplace=True)
    df = df.merge(nodes[['pid','cluster']], how='left', on='pid')

    X = []
    label = []
    df_anomaly_list = []

    for sample in range(samples):
        df_anomaly = df[['timestamp','pid','cluster','smoothed']].copy()
        df_anomaly['anomaly'] = 0

        anomalous_clusters = np.random.choice(nodes.cluster.unique(), size=n_anomalies)

        for index, cluster in enumerate(anomalous_clusters):
            index = index+1

            anomaly_sensor = (df_anomaly.cluster==cluster)
            anomaly_period = (df_anomaly.timestamp>'Jul 2020')&(df_anomaly.timestamp<'Jan 2021')
            anomaly_loc = anomaly_sensor&anomaly_period

            df_anomaly.loc[anomaly_loc, 'smoothed'] += anomaly_level
            df_anomaly.loc[anomaly_loc, 'anomaly'] = index


            if noise>0:
                # spreading noisy sensors across the other non-anomalous clusters
                all_clusters = list(nodes.cluster.unique())
                healthy_clusters = all_clusters.copy()
                healthy_clusters.remove(anomalous_clusters)
                healthy_sensors = nodes[nodes.cluster.isin(healthy_clusters)].pid.unique()
                n_noisy_sensors = int(np.ceil(noise*len(healthy_sensors)))

                noisy_pids = []
                for i in range(n_noisy_sensors):
                    cluster_noise = healthy_clusters[i%len(healthy_clusters)]
                    nodes_noise = nodes.query('cluster==@cluster_noise').pid.unique()

                    nodes_to_select = [node for node in nodes_noise if node not in noisy_pids]
                    noisy_pids.append(np.random.choice(nodes_to_select))

                noise_sensor = (df_anomaly.pid.isin(noisy_pids))
                noise_period = (df_anomaly.timestamp>'Jan 2020')&(df_anomaly.timestamp<'Mar 2021')
                noise_loc = noise_sensor&noise_period

                df_anomaly.loc[noise_loc, 'smoothed'] += anomaly_level
                if label_noise:
                    df_anomaly.loc[noise_loc, 'anomaly'] = -1 # Only for the case of n_anomalies=1            

        
        X.append(df_anomaly.pivot(index='pid', columns='timestamp', values='smoothed').values)
        label.append(df_anomaly.pivot(index='pid', columns='timestamp', values='anomaly').values.max(axis=1))
        df_anomaly_list.append(df_anomaly)

    X = np.array(X)
    label = np.array(label)

    return X, label, df_anomaly_list, G, nodes

def select_nodes(G, Number_of_nodes, mask=None):

    if ispygsp(G):
        A = G.A.toarray()
        if mask is not None:
            start_node = np.random.choice(np.where(mask)[0])
        else:
            start_node = np.random.choice(G.N)

    else:
        A = nx.adjacency_matrix(G).toarray()
        start_node = np.random.choice(G.number_of_nodes())

    
    selected_nodes = set([start_node])

    i = 1

    while len(selected_nodes)<Number_of_nodes:
        M = np.linalg.matrix_power(A,i) # matrix
        N = np.where(M[start_node,:])[0] #neighborhood
        X = M[start_node,np.where(M[start_node,:])[0]] #neighborhood values
        R = [(x**2)*np.random.rand() for x in X] #random ranking
        I = np.argsort(R)[::-1] # ranking indexes

        S = N[I] # ranked neighborhood

        for n in S:
            if len(selected_nodes) < Number_of_nodes:
                if mask is not None:
                    if np.isin(n, np.where(mask)[0]):
                        selected_nodes = selected_nodes.union([n])
                    
                else:
                    selected_nodes = selected_nodes.union([n])
        i=i+1

    return list(selected_nodes)

def select_walk(G, Number_of_nodes, walk_length=10):
    A = G.A.toarray()
    start_node = np.random.choice(G.N)

    selected_nodes = set([start_node])
    current_node = start_node

    while len(selected_nodes) < Number_of_nodes:
        # Perform a random walk starting from the last visited node
        for _ in range(walk_length):
            neighbors = np.where(A[current_node, :] > 0)[0]

            # Check if there are neighbors to walk to
            if len(neighbors) == 0:
                break  # Exit if the node has no neighbors

            # Randomly select the next step in the walk
            current_node = np.random.choice(neighbors)

            # Add the resulting node to the selected set if not already added
            selected_nodes.add(current_node)

            # Stop if we’ve reached the desired number of nodes
            if len(selected_nodes) >= Number_of_nodes:
                break

        # Restart the random walk from the start node if we still need more nodes
        current_node = start_node

    return list(selected_nodes)

def select_sparse(coords, N=None, r=30):
    num_nodes = coords.shape[0]
    selected_nodes = []

    if N is None:
        N = np.ceil(num_nodes*0.1).astype(int)

    while len(selected_nodes) < N:
        candidate = np.random.randint(num_nodes)
        is_valid = True

        if selected_nodes:
            distances = np.linalg.norm(coords[selected_nodes] - coords[candidate], axis=1)
            if np.any(distances < r):
                is_valid = False

        if is_valid:
            selected_nodes.append(candidate)

    return selected_nodes

def select_sparse_masked(coords, N=None, r=30, mask=None):

    num_nodes = coords.shape[0]
    all_indices = set(range(num_nodes))
    mask = set(mask) if mask is not None else set()
    available_indices = list(all_indices - mask)
    np.random.shuffle(available_indices)

    if N is None:
        N = max(1, int(np.ceil(num_nodes * 0.1)))

    selected_nodes = []

    for candidate in available_indices:
        if len(selected_nodes) >= N:
            break

        if not selected_nodes:
            selected_nodes.append(candidate)
            continue

        distances = np.linalg.norm(coords[selected_nodes] - coords[candidate], axis=1)
        if np.all(distances >= r):
            selected_nodes.append(candidate)

    return selected_nodes    


def select_radius(coords, radius):

    selected_nodes = []
    offset = coords.min(axis=0)
    ranges = (coords - offset).max(axis=0)

    minimum_selected_nodes = 10
    while(len(selected_nodes)<minimum_selected_nodes):

        # Generate a random location within the bounding box
        start_coords = offset + np.random.uniform([0.05*ranges[0], 0.05*ranges[1]], [0.95*ranges[0], 0.95*ranges[1]])

        distances = np.linalg.norm(coords - start_coords, axis=1)

        selected_nodes = np.where(distances<radius)[0]

    return list(selected_nodes), start_coords

def add_anomaly(node_positions, num_timestamps, anomaly_radius, onset, transient, deformation):

    # onset, transient and deformation are values in {1,2,3}
    # onset: 1 - anaomly starts at the first sample; 2 - anomaly starts at 1/3 of the timerange; 3 - at 2/3
    # transient: 1 - instantenous in time; 2 - takes 1 to 2 months; 3 - takes 6 to 24 months
    # deformation: 1 - abrupt in space; 2 - moderate; 3 (REMOVED) - smooth (the number of sigmas in the affected area)

    onset = (onset-1)*0.3

    if transient==2:
        transient=np.random.randint(5, 10)
    elif transient==3:
        transient=np.random.randint(30, 120)

    num_nodes = node_positions.shape[0]
    selected_nodes, anomaly_center = select_radius(node_positions, anomaly_radius)

    distances = np.linalg.norm(node_positions - anomaly_center, axis=1)
    sigma = distances[selected_nodes[-1]]/deformation
    diff = node_positions - anomaly_center

    anomaly = np.exp(- (diff[:,0]**2)/(2*sigma**2) - (diff[:,1]**2)/(2*sigma**2))

    onset_index = int(onset*num_timestamps)

    label = np.zeros((num_nodes, num_timestamps))
    label[selected_nodes, onset_index:] = 1

    transient_matrix = np.tile(np.arange(num_timestamps), (num_nodes, 1))

    transient_matrix = (transient_matrix - onset_index)/transient
    transient_matrix = transient_matrix*label
    transient_matrix[transient_matrix>1] = 1

    anomaly_matrix = transient_matrix*np.tile(anomaly.reshape((-1,1)), (1, num_timestamps))

    return anomaly_matrix, label

def add_phase_anomaly(node_positions, num_timestamps, anomalous_nodes):
    num_nodes = node_positions.shape[0]
    selected_nodes = select_sparse(node_positions, N=anomalous_nodes)

    label = np.zeros((num_nodes, num_timestamps))
    label[selected_nodes, :] = 1

    anomaly_matrix = np.zeros((num_nodes, num_timestamps))
    for node in selected_nodes:
        end = np.random.choice([-100,100])
        anomaly_matrix[node, :] = np.linspace(0, end, num_timestamps) + np.random.normal(0, 1, num_timestamps)
    return anomaly_matrix, label

def add_phase_anomaly_masked(node_positions, num_timestamps, anomalous_nodes, mask=None):
    num_nodes = node_positions.shape[0]
    selected_nodes = select_sparse_masked(node_positions, N=anomalous_nodes, mask=mask)

    label = np.zeros((num_nodes, num_timestamps))
    label[selected_nodes, :] = 1

    anomaly_matrix = np.zeros((num_nodes, num_timestamps))
    for node in selected_nodes:
        end = np.random.choice([-100,100])
        anomaly_matrix[node, :] = np.linspace(0, end, num_timestamps) + np.random.normal(0, 1, num_timestamps)
    return anomaly_matrix, label


def network_anomaly(G, data, n_anomalies):
    X = data.clone()
    anomalous_nodes = np.array(select_nodes(G, n_anomalies))
    label = torch.zeros(G.N if ispygsp(G) else G.number_of_nodes())
    label[anomalous_nodes] = 1

    anomalous_features = np.arange(0,50)

    X[anomalous_nodes[:,None], anomalous_features] = 1 - X[anomalous_nodes[:,None], anomalous_features]

    return X.numpy(), label.numpy()


def synthetic_timeseries(node_positions, num_timestamps, wave_params=None, gaussian_params=None,
                         anomaly_level=0.5, noise=0.1, seed=None):
    """
    Generate spatio-temporal data based on node positions and a specific function type.

    Parameters:
    - node_positions (numpy.ndarray): Array of shape (N, d) representing the positions of N nodes in d dimensions.
    - num_timestamps (int): Number of timestamps to generate.

    Returns:
    - data_matrix (numpy.ndarray): Spatio-temporal data matrix of shape (N, T), where N is the number of nodes and T is the number of timestamps.
    """

    if seed is not None:
        np.random.seed(seed)

    dim = node_positions.shape[1]
        
    if wave_params is None:
        wave_params = {'wave_source': np.random.random(dim), 'wave_speed': 0.05, 'frequency': 0.25,
                       'wave_amp':1}
    if gaussian_params is None:
        gaussian_params = {'gaussian_source': 0.1 + 0.8*np.random.random(dim), 'sigma':0.1,
                           'gaussian_amp':1}

    num_nodes = node_positions.shape[0]

    wave_source = wave_params.get('wave_source', np.random.random(dim))
    wave_speed = wave_params.get('wave_speed', 0.05)
    frequency = wave_params.get('frequency', 0.25)
    w_amp = wave_params.get('wave_amp', 1)
    
    gaussian_source = gaussian_params.get('gaussian_source', 0.1 + 0.8*np.random.random(dim))
    sigma = gaussian_params.get('sigma', 0.1)
    g_amp = gaussian_params.get('gaussian_amp', 0)

    center_t = np.random.randint(low=int(0.1*num_timestamps), high=int(0.9*num_timestamps))
    sigma_t = num_timestamps//10
    # time_window = [np.exp(-(t-center_t)**2/(2*sigma_t**2)) for t in range(num_timestamps)]


    data_matrix = np.zeros((num_nodes, num_timestamps))
    label = np.zeros(num_nodes)

    t = np.arange(num_timestamps)

    time_window = np.exp(-(t-center_t)**2/(2*sigma_t**2))
    spatial_decay = 1

    t_start = np.random.randint(low=int(0.1*num_timestamps), high=int(0.5*num_timestamps))
    anomaly_base = np.zeros(num_timestamps)
    anomaly_base[t_start:t_start+int(0.3*num_timestamps)] = anomaly_level

    for i, pos in enumerate(node_positions):
        d_wave = np.linalg.norm(pos - wave_source)
        d_gaussian = np.linalg.norm(pos - gaussian_source)

        healthy = w_amp*np.sin(2 * np.pi * frequency * (d_wave - wave_speed * t))  * np.exp(-spatial_decay * d_wave)
        gaussian = g_amp*np.exp(-d_gaussian**2 / (2 * sigma**2)) # Gaussian

        anomaly_mask = gaussian>0.5
        anomaly = anomaly_base*anomaly_mask

        label[i] = anomaly_mask.any()
        data_matrix[i, :] = healthy + anomaly

    normal_nodes = np.where(label==0)[0]
    noise_nodes = np.random.choice(normal_nodes, size=int(noise*len(normal_nodes)), replace=False)

    t_start = np.random.randint(low=int(0.1*num_timestamps), high=int(0.5*num_timestamps))
    anomaly_base = np.zeros(num_timestamps)
    anomaly_base[t_start:t_start+int(0.4*num_timestamps)] = anomaly_level
    
    for node in noise_nodes:
        data_matrix[node,:] += anomaly_base

    return data_matrix, label


def synthetic_binary(G, num_features, num_flips=10, noise=0.1, missing_ratio=0.2, seed=None):

    node_positions = G.coords
    dim = node_positions.shape[1]
    num_nodes = node_positions.shape[0]

    if seed is not None:
        np.random.seed(seed)
        
    data_matrix = np.tile(np.random.randint(low=0, high=2, size=num_features), (num_nodes,1))
    noise_indices = np.random.choice(num_nodes*num_features, int(noise*num_nodes*num_features), replace=False)
    data_matrix[np.unravel_index(noise_indices, data_matrix.shape)] ^=1

    anomalous_nodes = anomalous_nodes = np.array(select_nodes(G,int(0.15*num_nodes)))[:,None]
    anomalous_features = np.random.choice(num_features, num_flips, replace=False)

    label = np.zeros(num_nodes)
    label[anomalous_nodes] = 1
    
    data_matrix[anomalous_nodes, anomalous_features] ^= 1

    normal_nodes = np.where(label==0)[0]
    missing_nodes = np.random.choice(normal_nodes, size=int(missing_ratio*len(normal_nodes)), replace=False)

    
    for node in missing_nodes:
        data_matrix[node,:] = 0
            
    return data_matrix, label



###################### UTILITIES ########################

def sparsify_by_corr(Xo, A, top_k=4):
    """
    Sparsifies the adjacency matrix A by keeping only the top correlated connections among existing edges.
    
    Args:
    - Xo (torch.Tensor): The data matrix (N x d) where each row corresponds to the node features.
    - A (torch.Tensor): The adjacency matrix (N x N).
    - top_k (int): Number of top connections to retain per node based on correlation.
    
    Returns:
    - A_sparsified (torch.Tensor): The sparsified adjacency matrix.
    """
    # Compute the correlation matrix for the rows of Xo
    corr_matrix = Xo.corrcoef()
    
    # Replace self-correlation with a very small value (since we don't want self-loops)
    corr_matrix.fill_diagonal_(-float('inf'))

    # Initialize the sparsified adjacency matrix
    A_sparsified = torch.zeros_like(A).to(A.device)
    
    # Iterate through each node
    for i in range(A.shape[0]):
        # Find the existing edges for node i
        neighbors = (A[i] > 0).nonzero(as_tuple=True)[0]
        
        if len(neighbors) > 0:
            # Get the correlations for these neighbors
            neighbor_corrs = corr_matrix[i, neighbors]

            # Get the top k neighbors with the highest correlations
            top_k_indices = torch.topk(neighbor_corrs, min(top_k, len(neighbors)), largest=True).indices
            top_k_neighbors = neighbors[top_k_indices]

            # Retain only the top-k correlated edges in the adjacency matrix
            A_sparsified[i, top_k_neighbors] = A[i, top_k_neighbors]
    
    # Make the adjacency matrix symmetric (in case it's undirected)
    A_sparsified = torch.max(A_sparsified, A_sparsified.T)

    return A_sparsified


def connected_comm(N, Nc):
    conn = False
    while not conn:
        G = pygsp.graphs.Community(N=400, Nc=4, world_density=1/(40*(N/Nc)), size_ratio=1)
        conn = G.is_connected()
    return G

def mean_with_tuples(column):
    if isinstance(column.iloc[0], tuple):
        first_elements = [x[0] for x in column]
        second_elements = [x[1] for x in column]
        return (np.mean(first_elements).round(3), np.mean(second_elements).round(3))
    else:
        return column.mean().round(3)


def ispygsp(G):
    if isinstance(G, pygsp.graphs.Graph):
        return True
    elif isinstance(G, (nx.Graph, nx.DiGraph)):
        return False
    

####################################
def main():
    return 0


if __name__ == "__main__":
    main()



