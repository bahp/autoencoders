"""
Author: Bernard
Description:

    This script creates the lstm matrices that should be
    imputed to an LSTM neural network. The shape of the
    matrices is (samples, timesteps, features) where features
    contains all the features and one additional variable
    th

    D_TUPLE: The days from microbiology sample to keep
    P_TUPLE: The days from microbiology samples for positive.
"""

# Libraries
import json
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

# Own
from utils.settings import _FEATURES
from utils.settings import _IMPUTERS
from utils.settings import _SCALERS
from utils.settings import _METHODS
from utils.lstm.preprocess import create_lstm_matrix


# --------------------------------------------------
# Constants
# --------------------------------------------------
def check_time_gaps(df, by, date):
    unit = pd.Timedelta(1, unit="d")
    return (df.groupby(by=by)[date].diff() > unit).sum() > 1

def resample_01(df, features, **kwargs):
    aux  = df.droplevel(0) \
        .resample(**kwargs) \
        .asfreq()
    #aux[features] = aux[features].ffill()
    aux.day = aux.day.interpolate()
    return aux

def score(x, features=None, day_s=-5, day_e=5):
    """

    Parameters
    ----------
    x
    features
    day_s
    day_e
    ratio

    Returns
    -------

    """
    if features is None:
        features = x.columns

    # Filter
    df_ = x.copy(deep=True)
    df_ = df_[(df_.day >= day_s)]
    df_ = df_[(df_.day <= day_e)]
    df_ = df_[features]

    # Score
    num = (~df_.isna()).sum().sum()
    den = len(features) * len(range(day_s, day_e+1))

    # Return
    return num/den


def filter_w_size(df, w=10):
    """Filter those patients with less than w observations"""
    size = df.groupby('PersonID').size()
    pids = size[size > w]
    return df[df.PersonID.isin(pids.index)]

def filter_p_density(df, day_s, day_e, features, min_density=0.75):
    """Filter those patients with less than d density

    Parameters
    ----------

    Returns
    -------
    """
    # Libraries
    from utils.pandas.apply import density_score

    # Compute
    density = rsmp.groupby('PersonID') \
        .apply(density_score, day_s=day_s,
               day_e=day_e, features=features) \
        .sort_values(ascending=False) \
        .rename('score')
    pids = density[density > min_density]
    return  rsmp[rsmp.PersonID.isin(pids.index)]


def create_fname(dws, dwe, pws, pwe, w, imputer, scaler, features):
    """

    Parameters
    ----------
    dws: int (data window start)
    dwe: int (data window end)
    pws: int (positive window start)
    pwe: int (positive window end)
    w: int (window)
    imputer: str
    scaler: str
    features: str

    Returns
    -------
    The filename.
    """
    return 'matrix.{dws}_{dwe}.{pws}_{pwe}.w{w}.{imputer}.{scaler}.{features}' \
        .format(dws=dws, dwe=dwe, pws=pws, pwe=pwe,
                w=w, imputer=imputer, scaler=scaler,
                features=features)

FEATURES_DICT = {
    'crp': [
        'CRP'
    ],
    #'bare': [
    #    'CRP',
    #    'HCT',
    #    'PLT',
    #    'WBC'
    #],
    #'fbc': [
    #    'BASO',
    #    'EOS',
    #    'HCT',
    #    'HGB',
    #    'LY',
    #    'MCH',
    #    'MCHC',
    #    'MCV',
    #    'MONO',
    #    'MPV',
    #    'NEUT',
    #    'NRBCA',
    #    'PLT',
    #    'RBC',
    #    'RDW',
    #    'WBC'
    #],
    #'wft': [
    #    'WFIO2',
    #    'WCL',
    #    'WG',
    #    'WHB',
    #    'WHBCO',
    #    'WHBMET',
    #    'WHBO2',
    #    'WHCT',
    #    'WHHB',
    #    'WICA',
    #    'WK',
    #    'WLAC',
    #    'WNA',
    #    'WPCO2',
    #    'WPH',
    #    'WPO2',
    #],
}

LABELS = [
    'pathogenic'
]

TIMESTAMP = datetime.now().strftime('%y%m%d-%H%M%S')

# --------------------------------------------------
# Step 00 - Load data
# --------------------------------------------------
# Define path
PATH = Path('./objects/datasets/set1/data.csv')

# Load data
data = pd.read_csv(PATH,
                   #nrows=1000,
                   dtype={'PersonID': 'str'},
                   parse_dates=['date_collected'])

# Keep raw copy
raw = data.copy(deep=True)

# Drop duplicates
data = data.drop_duplicates()


# --------------------------------------------------
# Step 01 - Preprocess data
# --------------------------------------------------

def get_dtuple():
    return (int(data.day.min()),
            int(data.day.max()))

"""
{
    'KEY': FEATURES_DICT.keys(),
    'IMPUTER': ['simp'],
    'SCALER': ['std'],
    'D_TUPLE': [(-10, 3)],
    'P_TUPLE': [(-10, 3), (-5, 3)],
    'WINDOW': [5, 8]
},
"""

# Config
param_grid = [
    {
        'KEY': FEATURES_DICT.keys(),
        'IMPUTER': ['simp'],
        'SCALER': ['std'],
        'D_TUPLE': [get_dtuple(), (-10, 3)],
        'P_TUPLE': [(-5, 3)],
        'WINDOW': [5]
    }
]

# Create grid
grid = list(ParameterGrid(param_grid))

# Loop
for i,d in enumerate(ParameterGrid(param_grid)):

    print("%s. %s" % (i, d))

    # Extract all
    FEATURES = FEATURES_DICT[d.get('KEY')]
    IMPUTER = d.get('IMPUTER', 'simp')
    SCALER = d.get('SCALER', 'std')
    WINDOW = d.get('WINDOW', 5)
    D_TUPLE = d.get('D_TUPLE')
    P_TUPLE = d.get('P_TUPLE')

    # Re-sample
    rsmp = data.copy(deep=True) \
        .set_index(['PersonID', 'date_collected']) \
        .groupby(level=0) \
        .apply(resample_01, features=FEATURES, rule='1D') \
        .reset_index()

    # Add pathogenic label
    rsmp['pathogenic'] = \
        (rsmp.micro_code != 'CNS') & \
        (rsmp.day >= P_TUPLE[0]) & \
        (rsmp.day <= P_TUPLE[1])

    # Filter those patients with less than w observations.
    # This might not be needed as this filter is done automatically
    # within the <create_lstm_matrix>. However, we do it here too
    # for clarity (for density and imputation).
    rsmp = filter_w_size(rsmp, w=WINDOW)

    # Filter those patients with less than 0.75 density.
    rsmp = filter_p_density(rsmp,
        day_s=D_TUPLE[0], day_e=D_TUPLE[1],
        features=FEATURES, min_density=0.75)

    # Split in train, validation and test
    # Note that this split is not random but sequential. For a
    # random split, replace df with the following piece of code
    # so that the DataFrame is "shuffled" first.
    #   --> df.sample(frac=1, random_state=42)
    fractions = np.array([0.6, 0.2, 0.2])
    df = pd.DataFrame(rsmp.PersonID.unique())
    train, validate, test = \
        np.split(df.copy(deep=True),
            [int(.6 * len(df)), int(.8 * len(df))])

    # Add information into the DataFrame.
    rsmp['set'] = None
    for name, df in [('train', train),
                     ('validate', validate),
                     ('test', test)]:
        idxs = rsmp.PersonID.isin(df.squeeze())
        rsmp.loc[idxs, 'set'] = name

    # Save mask with missing data
    mask_missing = rsmp.isnan()

    # Forward fill
    rsmp[FEATURES] = rsmp.groupby('PersonID')[FEATURES].ffill()

    # Divide
    train, validate, test = \
        rsmp.loc[rsmp.set == 'train', :], \
        rsmp.loc[rsmp.set == 'validate', :], \
        rsmp.loc[rsmp.set == 'test', :],

    # Fit pre-processing steps
    imputer = _IMPUTERS.get(IMPUTER).fit(train[FEATURES])
    scaler = _SCALERS.get(SCALER).fit(train[FEATURES])

    # Impute data
    rsmp[FEATURES] = imputer.transform(rsmp[FEATURES])
    rsmp[FEATURES] = scaler.transform(rsmp[FEATURES])

    # Filter by days
    rsmp = rsmp[(rsmp.day >= D_TUPLE[0])]
    rsmp = rsmp[(rsmp.day <= D_TUPLE[1])]

    # Double check
    print("Has time gaps: %s" % check_time_gaps(df=rsmp,
        by='PersonID', date='date_collected'))

    # Format matrix with shape (samples, timestamps, features)
    #matrix = create_lstm_matrix(rsmp,
    #    features=FEATURES + ['pathogenic'],  # , 'day'],
    #    groupby='PersonID',
    #    w=WINDOW)




    # -------------------------
    # Save
    # -------------------------
    # Workbench path
    WORKBENCH = Path('./objects/datasets/set1/data') / TIMESTAMP
    FOLDER = create_fname(dws=D_TUPLE[0],
        dwe=D_TUPLE[1],
        pws=P_TUPLE[0],
        pwe=P_TUPLE[1],
        w=WINDOW,
        imputer=IMPUTER,
        scaler=SCALER,
        features=d.get('KEY'))

    # Create directory
    (WORKBENCH / FOLDER).mkdir(parents=True, exist_ok=True)

    # Save pre-processers
    joblib.dump(imputer, WORKBENCH / FOLDER / 'imputer.save')
    joblib.dump(scaler, WORKBENCH / FOLDER / 'scaler.save')

    # Save mask missing
    mask_missing.to_csv(WORKBENCH / FOLDER / 'mask_missing.csv')

    # Save data with splits
    rsmp.to_csv(WORKBENCH / FOLDER / 'data.csv')

    # Save config
    with open(WORKBENCH / FOLDER / "config.json", "w") as outfile:
        json.dump(d, outfile)

    # Save
    for name, df in rsmp.groupby('set'):
        # Matrix shape (samples, timestamps, features)
        matrix = create_lstm_matrix(df,
            features=FEATURES + ['pathogenic'],  # 'day'
            groupby='PersonID',
            w=WINDOW)
        # Save
        np.save(WORKBENCH / FOLDER / ('%s.npy' % name), matrix)

    # Save numpy array
    #np.save(WORKBENCH / fname, matrix)

    # Save data with splits
    #rsmp.to_csv(WORKBENCH / ('%s.csv' % fname))

    # Log (samples, timesteps, features)
    #print("%s/%s. Saved <%s> with %s." %
    #      (i, len(grid), fname, str(matrix.shape)))