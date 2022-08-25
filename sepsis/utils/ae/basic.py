
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

class SAE(nn.Module):
    """Symmetric Autoencoder (SAE)

    It only allows to choose the layers. It would be nice to
    extend this class so that we can choose also the activation
    functions (Sigmoid, ReLu, Softmax), additional dropouts, ...
    """
    def __init__(self, layers):
        super(SAE, self).__init__()

        # Set layers
        self.layers = layers

        # Create encoder
        enc = []
        for prev, curr in zip(layers, layers[1:]):
            enc.append(nn.Linear(prev, curr))
            enc.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*enc)

        # Reversed layers
        rev = layers[::-1]

        # Create decoder
        dec = []
        for prev, curr in zip(rev, rev[1:]):
            dec.append(nn.Linear(prev, curr))
            dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        encoded = self.encoder(x.float())
        decoded = self.decoder(encoded)
        return decoded

    def transform(self, X):
        """Prepare data and compute embeddings."""
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy().astype(np.float32)
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        return self.encode_inputs(X)

    def fit(self, X, y):
        """We are using skorch for compatibility with scikits
           instead of using our own fit. However, we include
           this empty method in case we want to manualy create
           a pipeline with an AE as estimator. Note that this
           will have to be pretrained.
        """
        return self

    @torch.no_grad()
    def encode_inputs(self, x):
        """Compute the embeddings."""
        z = []
        for e in DataLoader(x, 16, shuffle=False):
            z.append(self.encoder(e))
        return torch.cat(z, dim=0).numpy()
