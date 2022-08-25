# Generic
import numpy as np
import torch

from utils.ae.basic import SAE
from utils.ae.skorch import SkorchAE

# Specific
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.cross_decomposition import CCA
from umap import UMAP

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
# Features to use for training.
_FEATURES = {
    'set1': [
        'HCT',
        'HGB',
        'LY',
        'MCH',
        'MCHC',
        'MCV',
        'PLT',
        'RBC',
        'RDW',
        'WBC'
    ],
    'set2': [
        'HCT',
        'HGB',
        'PLT',
        'WBC'
    ]
}

_LABELS = {
    'set1': [
        'PersonID',
        'micro_code',
        'death',
        'day'
    ]
}

# Elements from sci-kits
_IMPUTERS = {
    'simp': SimpleImputer(),
    'iimp': IterativeImputer()
}
_SCALERS = {
    'std': StandardScaler(),
    'mmx': MinMaxScaler(),
    'rbt': RobustScaler(),
    'nrm': Normalizer(),
}
_METHODS = {
    'pca': PCA(n_components=2),
    'tsne': TSNE(n_components=2),
    'cca': CCA(n_components=2),
    'icaf': FastICA(n_components=2),
    'lda': LatentDirichletAllocation(n_components=2),
    'nmf': NMF(n_components=2),
    'pcak': KernelPCA(),
    'pcai': IncrementalPCA(),
    'iso': Isomap(n_components=2),
    'lle': LocallyLinearEmbedding(n_components=2),
    'mds': MDS(n_components=2, max_iter=100, n_init=1),
    'spe': SpectralEmbedding(n_components=2),
    'umap': UMAP(),
    'sae': SkorchAE(SAE, criterion=torch.nn.MSELoss)
}


def get_scaler(scaler=None):
    """Get the scaler"""
    if scaler is not None:
        if isinstance(scaler, str):
            return _SCALERS.get(scaler, None)
    return scaler
