# Generic
import numpy as np

# Specific
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE

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
    'cca': CCA(n_components=2)
}

def get_scaler(scaler=None):
    """Get the scaler"""
    if scaler is not None:
        if isinstance(scaler, str):
            return _SCALERS.get(scaler, None)
    return scaler
