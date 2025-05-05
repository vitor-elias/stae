import numpy as np

import math
import torch

from sklearn.cluster import KMeans
from collections import OrderedDict
from typing import Optional, Callable, Union, List

from torch import nn, Tensor
from torch.nn import Linear, Conv1d, LayerNorm, DataParallel, ReLU, Sequential, Parameter
from torch.nn.functional import glu

from torch_geometric.nn.models import GraphUNet
from torch_geometric.nn import GCNConv, GraphConv, TopKPooling
from torch_geometric.nn.dense import mincut_pool, dense_mincut_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import activation_resolver

from torch_geometric.utils import dense_to_sparse, scatter, add_remaining_self_loops, spmm
from torch_geometric.utils.repeat import repeat
from torch_geometric.utils import add_self_loops, remove_self_loops, to_torch_csr_tensor
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor, torch_sparse, PairTensor
from torch_geometric.nn.dense.linear import Linear as pygLinear
import source.nn.recurrent_autoencoder as rae

from torch_sparse import eye as speye

##############################################################################################################    

class AE(nn.Module):
    def __init__(self, layer_dims):
        super(AE, self).__init__()

        # Encoder
        encoder_layers = OrderedDict()
        for i in range(len(layer_dims) - 1):
            encoder_layers[f'linear_{i}'] = nn.Linear(layer_dims[i], layer_dims[i + 1])
            encoder_layers[f'relu_{i}'] = nn.ReLU()

        self.encoder = nn.Sequential(encoder_layers)

        # Decoder (mirror of the encoder)
        decoder_layers = OrderedDict()
        reversed_dims = layer_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers[f'linear_{i}'] = nn.Linear(reversed_dims[i], reversed_dims[i + 1])
            if i < len(reversed_dims) - 2:
                decoder_layers[f'relu_{i}'] = nn.ReLU()

        self.decoder = nn.Sequential(decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return o

    def reset_parameters(self):

        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


############

class GCNencoder(nn.Module):
    def __init__(self, layer_dims, conv_layer=GCNConv, conv_params=None):
        super(GCNencoder, self).__init__()

        if conv_params is None:
            conv_params = {}

        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(conv_layer(layer_dims[i], layer_dims[i+1], **conv_params))
            self.layers.append(nn.ReLU())

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            if isinstance(layer, tuple([GCNConv, GraphConv])):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class GCNdecoder(nn.Module):
    def __init__(self, layer_dims, conv_layer=GCNConv, conv_params=None):
        super(GCNdecoder, self).__init__()

        if conv_params is None:
            conv_params = {}

        reversed_dims = layer_dims[::-1]
        self.layers = nn.ModuleList()
        for i in range(len(reversed_dims) - 1):
            self.layers.append(conv_layer(reversed_dims[i], reversed_dims[i+1], **conv_params))
            if i < len(reversed_dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            if isinstance(layer, tuple([GCNConv, GraphConv])):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class GCN2MLP(nn.Module):
    def __init__(self, layer_dims, conv_layer=GCNConv, conv_params=None):
        super(GCN2MLP, self).__init__()

        self.encoder = GCNencoder(layer_dims, conv_layer, conv_params)

        decoder_layers = OrderedDict()
        reversed_dims = layer_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers[f'linear_{i}'] = nn.Linear(reversed_dims[i], reversed_dims[i + 1])
            if i < len(reversed_dims) - 2:
                decoder_layers[f'relu_{i}'] = nn.ReLU()

        self.decoder = nn.Sequential(decoder_layers)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z)
        return o

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class GConv2MLP(nn.Module):
    def __init__(self, layer_dims, conv_layer=GraphConv, conv_params=None):
        super(GConv2MLP, self).__init__()

        self.encoder = GCNencoder(layer_dims, conv_layer, conv_params)

        decoder_layers = OrderedDict()
        reversed_dims = layer_dims[::-1]
        for i in range(len(reversed_dims) - 1):
            decoder_layers[f'linear_{i}'] = nn.Linear(reversed_dims[i], reversed_dims[i + 1])
            if i < len(reversed_dims) - 2:
                decoder_layers[f'relu_{i}'] = nn.ReLU()

        self.decoder = nn.Sequential(decoder_layers)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z)
        return o

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class GCNAE(nn.Module):

    # Symmetric Graph Autoencoder with GCN layers

    def __init__(self, layer_dims, conv_layer=GCNConv, conv_params=None):        
        super(GCNAE, self).__init__()

        # GCN encoder
        self.encoder = GCNencoder(layer_dims, conv_layer, conv_params)
        self.decoder = GCNdecoder(layer_dims, conv_layer, conv_params)

    def forward(self, x, edge_index, edge_weight):

        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z, edge_index, edge_weight)

        return o

    def reset_parameters(self):

        self.encoder.reset_parameters()

        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class GConvAE(nn.Module):

    # Symmetric Graph Autoencoder with GraphConv layers

    def __init__(self, layer_dims, conv_layer=GraphConv, conv_params=None):        
        super(GConvAE, self).__init__()

        # GCN encoder
        self.encoder = GCNencoder(layer_dims, conv_layer, conv_params)
        self.decoder = GCNdecoder(layer_dims, conv_layer, conv_params)

    def forward(self, x, edge_index, edge_weight):

        z = self.encoder(x, edge_index, edge_weight)
        o = self.decoder(z, edge_index, edge_weight)

        return o

    def reset_parameters(self):

        self.encoder.reset_parameters()

        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()                



class GUNet(nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int,
        pool_ratios: Union[float, List[float]] = 0.5,
        sum_res: bool = True,
        act: Union[str, Callable] = 'relu',
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = activation_resolver(act)
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: OptTensor = None,
        edge_weight: Tensor = None,
    ) -> Tensor:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        assert edge_weight.dim() == 1
        assert edge_weight.size(0) == edge_index.size(1)

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight, x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x

    def augment_adj(self, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> PairTensor:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        adj = to_torch_csr_tensor(edge_index, edge_weight, size=(num_nodes, num_nodes))
        adj = (adj @ adj).to_sparse_coo()
        edge_index, edge_weight = adj.indices(), adj.values()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')


########

RAE = rae.RecurrentAE

##############################################################################################################
def main():
    return 0


if __name__ == "__main__":
    main()
