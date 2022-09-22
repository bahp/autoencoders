# Generic
import numpy as np
import torch

# XGboost
import xgboost as xgb

from utils.ae.basic import SAE
from utils.ae.skorch import SkorchAE

# Umport umap
from umap import UMAP

# Import scikits metrics
from sklearn.metrics import make_scorer

# Specific
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Import scikts dimensionality reduction
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

# Import scikits classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

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
_SAMPLERS = {
    'ros': RandomOverSampler(),
    'rus': RandomUnderSampler(),
    'smt': SMOTE()
}

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

_CLASSIFIERS = {
    'gnb': GaussianNB(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'svm': SVC(),
    'ann': MLPClassifier(),
    'llr': LogisticRegression(),
    'etc': ExtraTreesClassifier(),
    'xgb': xgb.XGBClassifier(),
    # 'lgbm': lgbm.LGBMClassifier(),
    'rusboost': RUSBoostClassifier(),
    'bbc': BalancedBaggingClassifier(),
    'brfc': BalancedRandomForestClassifier(),
    'adaboost': AdaBoostClassifier(),
    'gradboost': GradientBoostingClassifier(),
    #'histboost': HistGradientBoostingClassifier(),
    #'vclf': VotingClassifier()
}

_METHODS = {
    # Transformers
    # ------------
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
    'sae': SkorchAE(SAE, criterion=torch.nn.MSELoss),

}

# Add calibrated estimators
def _calibrated_estimator(e):
    return CalibratedClassifierCV(base_estimator=e)

_CLASSIFIERS.update(
    {'c{0}'.format(k):_calibrated_estimator(v)
        for k,v in _CLASSIFIERS.items()})

# Combine all
_ALL = {}
_ALL.update(_SAMPLERS)
_ALL.update(_IMPUTERS)
_ALL.update(_SCALERS)
_ALL.update(_METHODS)
_ALL.update(_CLASSIFIERS)

def get_scaler(scaler=None):
    """Get the scaler"""
    if scaler is not None:
        if isinstance(scaler, str):
            return _SCALERS.get(scaler, None)
    return scaler


def get_features_upper(df):
    """"""
    cols = df.columns.tolist()
    cols = [c for c in cols if c.isupper()]
    return cols

def get_features_panels(df, panels):
    """"""
    return []