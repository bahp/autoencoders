"""
Author: Bernard
Description:

"""

# Libraries
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
from sklearn.pipeline import Pipeline

# Own
from utils.settings import _FEATURES
from utils.settings import _IMPUTERS
from utils.settings import _SCALERS
from utils.settings import _METHODS
from utils.lstm.preprocess import create_lstm_matrix


# --------------------------------------------------
# Constants
# --------------------------------------------------
# Configuration
FEATURES = [
    'HCT',
    # 'LY',
    # 'MCV',
    # 'PLT',
    # 'RDW',
    'WBC',
    'CRP'
]

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
        'BASO'
        'EOS'
        'HCT'
        'HGB'
        'LY'
        'MCH'
        'MCHC'
        'MCV'
        'MONO'
        'MPV'
        'NEUT'
        'NRBCA'
        'PLT'
        'RBC'
        'RDW'
        'WBC'
    ],
    'wft': [
        'WFIO2',
        'WCL',
        'WG',
        'WHB',
        'WHBCO',
        'WHBMET',
        'WHBO2',
        'WHCT',
        'WHHB',
        'WICA',
        'WK',
        'WLAC',
        'WNA',
        'WPCO2',
        'WPH',
        'WPO2',
    ],
}

LABELS = [
    'pathogenic'
]



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

#data = data[data.PersonID == '100085']
#print(data)

# --------------------------------------------------
# Step 01 - Preprocess data
# --------------------------------------------------

def check_time_gaps(df, by, date):
    unit = pd.Timedelta(1, unit="d")
    return (df.groupby(by=by)[date].diff() > unit).sum() > 1

def resample_01(df, features, **kwargs):
    aux  = df.droplevel(0) \
        .resample(**kwargs) \
        .asfreq()
    aux[features] = aux[features].ffill()
    aux.day = aux.day.interpolate()
    return aux

# Config
KEY = 'wft'
FEATURES = FEATURES_DICT[KEY]
IMPUTER = 'simp'
SCALER = 'std'
DSTART = data.day.min()
DEND = data.day.max()
WINDOW = 5

# Resmple
rsmp = data.copy(deep=True) \
    .set_index(['PersonID', 'date_collected']) \
    .groupby(level=0) \
    .apply(resample_01, features=FEATURES, rule='1D') \
    .reset_index()

rsmp['pathogenic_v2'] = \
    (rsmp.micro_code != 'CNS') & \
    (rsmp.day >= -5) &  \
    (rsmp.day <= 3)

# Show
print("\nLoaded data:")
print(data[['PersonID', 'date_collected', 'day', 'micro_code'] + FEATURES + LABELS])
print("\nResampled data (ffilled):")
print(rsmp[['PersonID', 'date_collected', 'day'] + FEATURES + LABELS])
print("Has time gaps: %s" %
      check_time_gaps(df=rsmp, by='PersonID', date='date_collected'))

# Format data
rsmp[FEATURES] = _IMPUTERS.get(IMPUTER).fit_transform(rsmp[FEATURES])
rsmp[FEATURES] = _SCALERS.get(SCALER).fit_transform(rsmp[FEATURES])

print(rsmp)

# Filter by days
rsmp = rsmp[(rsmp.day >= DSTART)]
rsmp = rsmp[(rsmp.day <= DEND)]

# Format matrix with shape (samples, timestamps, features)
matrix = create_lstm_matrix(rsmp,
    features=FEATURES+['pathogenic_v2'], #, 'day'],
    groupby='PersonID',
    w=WINDOW)

# Show matrix
print("\nMatrix shape (samples, timesteps, features): %s" % str(matrix.shape))

"""
print(matrix[:, :, :])
X = matrix[:, :, :-1].astype('float32')
y = matrix[:, -1, -2:].astype('float32')
print(X)
print(y)
"""

# -------------------------
# Save
# -------------------------
# Workbench path
WORKBENCH = Path('./objects/datasets/set1/data')

# Create directory
WORKBENCH.mkdir(parents=True, exist_ok=True)

# Create filename
fname = 'matrix.%s.%s.%s.%s.%s.%s.v2.npy' % (
    DSTART, DEND, WINDOW, IMPUTER, SCALER, KEY
)

# Save
np.save(WORKBENCH / fname, matrix)