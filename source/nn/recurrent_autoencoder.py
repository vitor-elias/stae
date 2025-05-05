"""
Recurrent Autoencoder PyTorch implementation
https://github.com/PyLink88/Recurrent-Autoencoder
LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection

"""

import torch
import torch.nn as nn
from functools import partial

class RecurrentEncoder(nn.Module):
    """Recurrent encoder"""

    def __init__(self, n_features, latent_dim, rnn):
        super().__init__()

        self.rec_enc1 = rnn(n_features, latent_dim, batch_first=True)

    def forward(self, x):
        _, h_n = self.rec_enc1(x)

        return h_n

class RecurrentDecoder(nn.Module):
    """Recurrent decoder for RNN and GRU"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)
        # x = torch.zeros((h_0[0].size(0), seq_len, self.n_features), device=self.device)

        # Squeezing
        h_i = h_0.squeeze()

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i)

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i)

            if x_i.dim() == 1:
                x = torch.cat([x, x_i.unsqueeze(0)], axis=1)
            else:
                x = torch.cat([x, x_i], axis=1)    
            

        return x.view(-1, seq_len, self.n_features)


class RecurrentDecoderLSTM(nn.Module):
    """Recurrent decoder LSTM"""

    def __init__(self, latent_dim, n_features, rnn_cell, device):
        super().__init__()

        self.n_features = n_features
        self.device = device
        self.rec_dec1 = rnn_cell(n_features, latent_dim)
        self.dense_dec1 = nn.Linear(latent_dim, n_features)

    def forward(self, h_0, seq_len):
        # Initialize output
        x = torch.tensor([], device = self.device)

        # Squeezing
        h_i = [h.squeeze() for h in h_0]

        # Reconstruct first element with encoder output
        x_i = self.dense_dec1(h_i[0])

        # Reconstruct remaining elements
        for i in range(0, seq_len):
            h_i = self.rec_dec1(x_i, h_i)
            x_i = self.dense_dec1(h_i[0])
            
            if x_i.dim() == 1:
                x = torch.cat([x, x_i.unsqueeze(0)], axis=1)
            else:
                x = torch.cat([x, x_i], axis=1)    

        return x.view(-1, seq_len, self.n_features)


class RecurrentAE(nn.Module):
    """Recurrent autoencoder"""

    def __init__(self, n_features, latent_dim, rnn_type, rnn_act, device):
        super().__init__()

        # Encoder and decoder configuration
        self.rnn, self.rnn_cell = self.get_rnn_type(rnn_type, rnn_act)
        self.rnn_type = rnn_type
        self.rnn_act = rnn_act
        self.decoder = self.get_decoder(rnn_type)
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.device = device

        # Encoder and decoder
        self.encoder = RecurrentEncoder(self.n_features, self.latent_dim, self.rnn)
        self.decoder = self.decoder(self.latent_dim, self.n_features, self.rnn_cell, self.device)

    def forward(self, x):

        orig_dim = x.dim()
        if orig_dim == 2:
            x = x.unsqueeze(2)
        seq_len = x.shape[1]
        h_n = self.encoder(x)
        out = self.decoder(h_n, seq_len)

        if orig_dim == 2:
            return torch.flip(out, [1]).squeeze()
        else:
            return torch.flip(out, [1])

    @staticmethod
    def get_rnn_type(rnn_type, rnn_act=None):
        """Get recurrent layer and cell type"""
        if rnn_type == 'RNN':
            rnn = partial(nn.RNN, nonlinearity=rnn_act)
            rnn_cell = partial(nn.RNNCell, nonlinearity=rnn_act)

        else:
            rnn = getattr(nn, rnn_type)
            rnn_cell = getattr(nn, rnn_type + 'Cell')

        return rnn, rnn_cell

    @staticmethod
    def get_decoder(rnn_type):
        """Get recurrent decoder type"""
        if rnn_type == 'LSTM':
            decoder = RecurrentDecoderLSTM
        else:
            decoder = RecurrentDecoder
        return decoder
    
    def reset_parameters(self):

        for layer in self.encoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


##############################################################################################################
def main():
    return 0


if __name__ == "__main__":
    main()


# if __name__ == '__main__':

#     device = 'cuda'

#     # Adding random data
#     X = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.float32).unsqueeze(2)

#     # Model
#     model = RecurrentAE(n_features=1, latent_dim=4, rnn_type='GRU', rnn_act='relu', device=device)

#     # Encoder
#     h = model.encoder(X)
#     out =  model.decoder(h, seq_len = 10)
#     out = torch.flip(out, [1])

#     # Loss
#     loss = nn.L1Loss(reduction = 'mean')
#     l = loss(X, out)


