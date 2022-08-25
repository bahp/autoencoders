import pandas as pd
import numpy as np

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNet
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator

class SkorchAE(NeuralNet):
    def transform(self, X, *args, **kwargs):
        return self.module_.transform(X, *args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        """Overwritten.

        .. note: Fitting to X_ rather than y.
        """
        X_ = torch.from_numpy(X).float()
        return super().fit(X_, X_)

    def score(self, X, y):
        """Overwritten.

        This method contains the mean accuracy for classification
        problems and the coefficient of determination R^2 for a
        regression problem. For NeuralNet, needs to be implemented.

        X_ = torch.from_numpy(X).float()
        y_ = torch.from_numpy(y).float()
        loss2 = super().get_loss(X_, y_)
        """
        from skorch.scoring import loss_scoring
        return loss_scoring(self, X, X)

    def get_params(self, deep=True, **kwargs):
        """Overwritten.

        In order to get unique signatures when creating the pipelines,
        it is necessary to remove those parameters from NeuralNet that
        are created dynamically. Thus, in addition to avoid including
        the callbacks, we have to remove also the valid split as these
        are not part of the model hyperparameter configuration.

        'train_split':
            <skorch.dataset.ValidSplit object at 0x1544ffcd0>
        """
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        # Callback parameters are not returned by .get_params, needs
        # special treatment.
        #params_cb = self._get_params_callbacks(deep=deep)
        #params.update(params_cb)
        # don't include the following attributes
        to_exclude = {'_modules', '_criteria', '_optimizers', 'train_split'}
        return {key: val for key, val in params.items()
            if key not in to_exclude}


