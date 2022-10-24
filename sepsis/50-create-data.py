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
from utils.utils import AttrDict
from utils.lstm.preprocess import create_lstm_matrix


# --------------------------------------------------
# Constants
# --------------------------------------------------
def check_one_date_sample_per_patient(df, patient='PersonID'):
    """The patient has one more than one micro sample.

    Parameters
    ----------
    df
    patient

    Returns
    -------

    """
    check = (df
        .groupby(patient) \
        .date_sample.nunique() > 1).sum()
    if check:
        raise Exception('date_sample', 'more than one!')


def check_time_gaps(df, by='PersonID', date='date_collected', unit='d'):
    """Check whether there are time gaps.
    
    Parameters
    ----------
    df
    by
    date
    unit

    Returns
    -------

    """
    unit = pd.Timedelta(1, unit=unit)
    check = (df.groupby(by=by)[date].diff() > unit).sum() > 1
    if check:
        raise Exception('date_collected', 'there is a gap!')


def resample(df, **kwargs):
    """Include missing rows by freq."""
    return df.droplevel(0) \
             .resample(**kwargs) \
             .asfreq()


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

def folder_name(d):
    """Create folder name"""
    return '-'.join([str(v) for k,v in d.items()])

DAY_RANGES_DICT = {
    'rng1': [-100, 100],
    'rng2': [-10, 1]
}

FEATURES_DICT = {
    'crp': [
        'CRP'
    ],
    'bare': [
        'CRP',
        'HCT',
        'PLT',
        'WBC'
    ],
    'fbc': [
        'BASO', 'EOS', 'HCT', 'HGB', 'LY','MCH', 'MCHC',
        'MCV', 'MONO', 'MPV', 'NEUT', 'NRBCA', 'PLT', 'RBC',
        'RDW', 'WBC'
    ],
    'wft': [
        'WFIO2', 'WCL', 'WG', 'WHB', 'WHBCO', 'WHBMET',
        'WHBO2', 'WHCT', 'WHHB', 'WICA', 'WK', 'WLAC',
        'WNA', 'WPCO2', 'WPH', 'WPO2',
    ],
}

LABELS = [
    'pathogenic'
]


TIMESTAMP = datetime.now().strftime('%y%m%d-%H%M%S')

# Workbench path
WORKBENCH = Path('./objects/datasets/set1/data') / TIMESTAMP

# --------------------------------------------------
# Step 00 - Load data
# --------------------------------------------------
# Define path
PATH = Path('./objects/datasets/set1/data.csv')

# Load data
data = pd.read_csv(PATH,
                   #nrows=1000,
                   dtype={'PersonID': 'str'},
                   parse_dates=['date_collected',
                                'date_sample'])

# Drop duplicates
data = data.drop_duplicates()




def get_dtuple():
    return (int(data.day.min()),
            int(data.day.max()))




# Create param grid
param_grid = [
    {
        'features_key': ['crp', 'bare', 'fbc', 'wft'],
        'imputer': ['simp'],
        'scaler': ['std'],
        'window': [10, 5],
        'abs_day_range_key': ['rng1', 'rng2'],
        'dns_day_range_key': ['rng2']    }
]

# Create grid
grid = list(ParameterGrid(param_grid))

# Configuration
config = AttrDict(dict(
    pid='PersonID',
    date='date_collected',
    resample_kws=dict(
        rule='1D'
    ),

    extra=dict(
        param_grid=param_grid,
        features_dict=FEATURES_DICT,
        day_range_dict=DAY_RANGES_DICT
    )
))




# ---------------------------
# Formatting
# ---------------------------
# Re-sample
rsmp = data.copy(deep=True) \
    .set_index([config.pid, config.date]) \
    .groupby(level=0) \
    .apply(resample, rule='1D') \
    .reset_index()

# Fill (ffill and bfill)
columns = [
    'date_sample',
    'pathogenic',
    'micro_code',
    'age',
    'sex'
]
rsmp[columns]= rsmp \
    .groupby(config.pid)[columns] \
    .ffill().bfill()

# Add day from date_sample
rsmp['day'] = (rsmp.date_collected - rsmp.date_sample).dt.days

# Add pathogenic window labels
for ds, de in [(-10, 3), (-5, 3), (-1, 3)]:
    rsmp['pathogenicw_%s_%s' % (ds, de)] = \
        (rsmp.micro_code != 'CNS') & \
        (rsmp.day >= ds) & \
        (rsmp.day <= de)

# Some useful checks.
check_one_date_sample_per_patient(rsmp, patient=config.pid)


# ------------------------------
# Train, validate, test division
# ------------------------------
# Note that this split is not random but sequential. For a
# random split, replace df with the following piece of code
# so that the DataFrame is "shuffled" first.
#   --> df.sample(frac=1, random_state=42)
fractions = np.array([0.6, 0.2, 0.2])
df = pd.DataFrame(rsmp[config.pid].unique())
train, validate, test = \
    np.split(df.copy(deep=True),
        [int(.8 * len(df)), int(.9 * len(df))])

# Add information into the DataFrame.
rsmp['set'] = None
for name, df in [('train', train),
                 ('validate', validate),
                 ('test', test)]:
    idxs = rsmp[config.pid].isin(df.squeeze())
    rsmp.loc[idxs, 'set'] = name



# ---------------------------
# Save information
# ---------------------------
# Create directory
WORKBENCH.mkdir(parents=True, exist_ok=True)

# Mask missing
mask = rsmp.isna()

# Save
rsmp.to_csv(WORKBENCH / 'data.csv', index=False)
mask.to_csv(WORKBENCH / 'mask.csv', index=False)


import sys
sys.exit()


# ------------------------------
# Create LSTM matrices
# ------------------------------
# Loop
for i,d in enumerate(ParameterGrid(param_grid)):

    print("%s. %s" % (i, d))

    # ---------------------------
    # Get information
    # ---------------------------
    # Get selected features
    FEATURES = FEATURES_DICT[d.get('features_key')]
    IMPUTER = d.get('imputer')
    SCALER = d.get('scaler')
    WINDOW = d.get('window')
    ABS_DAY_RANGE = DAY_RANGES_DICT[d.get('abs_day_range_key')]
    DNS_DAY_RANGE = DAY_RANGES_DICT[d.get('dns_day_range_key')]

    OUTCOMES = [c for c in rsmp.columns
        if c.startswith('pathogenic')]

    # Workbench path
    FOLDER = WORKBENCH / 'matrices' / folder_name(d)

    # Create directory
    FOLDER.mkdir(parents=True, exist_ok=True)


    # ---------------------------
    # Copy DataFrame
    # ---------------------------
    # Create auxiliary DataFrame
    aux = rsmp.copy(deep=True)


    # ---------------------------
    # Filtering
    # ---------------------------
    # This filtering is mostly needed for the temporal
    # approaches such as LSTMs. For the static approaches
    # it could be ignored.

    # Import own filter method
    from utils.utils import filter_data

    # Apply filtering
    filt = filter_data(rsmp,
        day_range=ABS_DAY_RANGE,
        window_size_kws=dict(
            min_w=WINDOW, col=config.pid),
        window_density_kws=dict(
            day_range=DNS_DAY_RANGE,
            features=FEATURES, min_density=0.75
        )
    )

    # Some useful checks.
    check_time_gaps(rsmp, by=config.pid,
                          date=config.date,
                          unit='d')


    # ---------------------------------
    # Create some default LSTM matrices
    # ---------------------------------
    # Filling
    aux[FEATURES] = aux.groupby(config.pid)[FEATURES].ffill()

    # Divide
    train, validate, test = \
        aux.loc[aux.set == 'train', :], \
        aux.loc[aux.set == 'validate', :], \
        aux.loc[aux.set == 'test', :],

    # Fit pre-processing steps
    imputer = _IMPUTERS.get(IMPUTER).fit(train[FEATURES])
    scaler = _SCALERS.get(SCALER).fit(train[FEATURES])

    # Impute data
    aux[FEATURES] = imputer.transform(aux[FEATURES])
    aux[FEATURES] = scaler.transform(aux[FEATURES])

    # Filter by days
    #rsmp = rsmp[(rsmp.day >= D_TUPLE[0])]
    #rsmp = rsmp[(rsmp.day <= D_TUPLE[1])]



    # -----------------------
    # Save
    # -----------------------
    # Create directory
    FOLDER.mkdir(parents=True, exist_ok=True)

    # Data
    aux[[config.pid, config.date, 'day'] + FEATURES + OUTCOMES] \
        .to_csv(FOLDER / 'prep.csv')

    # Save pre-processers
    joblib.dump(imputer, FOLDER / 'imputer.save')
    joblib.dump(scaler, FOLDER / 'scaler.save')

    # Save config
    config['current_grid'] = d
    with open(FOLDER / "config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)

    # Save
    for name, df in aux.groupby('set'):
        # Matrix shape (samples, timestamps, features)
        matrix = create_lstm_matrix(df,
            features=FEATURES + OUTCOMES,  # 'day'
            groupby=config.pid,
            w=WINDOW)
        print(name, matrix.shape)
        # Save
        np.save(FOLDER / ('%s.npy' % name), matrix)