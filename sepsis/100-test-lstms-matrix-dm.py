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
# Step 00 - Load data
# --------------------------------------------------
# Define path
PATH = Path('../sepsis-damien/Working datasets/windows_7days.csv')

# Load data
data = pd.read_csv(PATH,
                   nrows=10000000,
                   dtype={'PersonID': 'str'},
                   parse_dates=['date_culture'])

# Keep raw copy
raw = data.copy(deep=True)

# Drop duplicates
data = data.drop_duplicates(subset=['id', 'date_diff'])

# Rename
data = data.rename(
    columns={
        'date_diff': 'day',
        'id': 'PersonID'
    }
)

print(data.head(20))

# --------------------------------------------------
# Step 01 - Preprocess data
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
FEATURES = [
    'Ward Lactate', 'Ward Glucose',
    'Ward sO2', 'White blood cell count, blood', 'Platelets',
    'Haemoglobin', 'Mean cell volume, blood', 'Haematocrit',
    'Mean cell haemoglobin conc, blood',
    'Mean cell haemoglobin level, blood', 'Red blood cell count, blood',
    'Red blood cell distribution width', 'Creatinine', 'Urea level, blood',
    'Potassium', 'Sodium', 'Neutrophils', 'Chloride', 'Lymphocytes',
    'Monocytes', 'Eosinophils',
    'C-Reactive Protein', 'Albumin', 'Alkaline Phosphatase',
    'Glucose POCT Strip Blood', 'Total Protein', 'Globulin',
    'Alanine Transaminase', 'Bilirubin', 'Prothrombin time', 'Fibrinogen (clauss)',
    'Procalcitonin', 'Ferritin', 'D-Dimer', 'sex', 'age'
]
#FEATURES = [
#    'Haematocrit',
#    'White blood cell count, blood',
#    'C-Reactive Protein'
#]
LABELS = [
    'pathogenic'
]


def check_time_gaps(df, by, date):
    unit = pd.Timedelta(1, unit="d")
    return (df.groupby(by=by)[date].diff() > unit).sum() > 1


def resample_01(df, **kwargs):
    return df.droplevel(0) \
        .resample(**kwargs) \
        .asfreq() \
        .ffill()


"""
print(data[['pathogenic', 'pathogenic_window']])

import sys
sys.exit()

aux = data[['PersonID', 'window'] + FEATURES + LABELS]

def reshape(x):
    print("")
    print(x)

    import sys
    sys.exit()

matrix = []

e = aux.copy(deep=True) \
    .set_index(['PersonID', 'window']) \
    .groupby(level=[0, 1]) \
    .apply(reshape) \
    .reset_index()

print(e)



rsmp = data.copy(deep=True) \
    .set_index(['PersonID', 'date_collected']) \
    .groupby(level=0) \
    .apply(resample_01, rule='1D') \
    .reset_index()

# Show
print("\nLoaded data:")
print(data[['PersonID', 'date_collected'] + FEATURES + LABELS])
print("\nResampled data (ffilled):")
print(rsmp[['PersonID', 'date_collected'] + FEATURES + LABELS])
print("Has time gaps: %s" %
      check_time_gaps(df=rsmp, by='PersonID', date='date_collected'))
"""
# Create steps
imputer = 'simp'
scaler = 'std'

# Format data
data[FEATURES] = _IMPUTERS.get(imputer).fit_transform(data[FEATURES])
data[FEATURES] = _SCALERS.get(scaler).fit_transform(data[FEATURES])

print(data)

# Format matrix with shape (samples, timestamps, features)
matrix = create_lstm_matrix(data,
                            features=FEATURES + LABELS,
                            groupby='PersonID',
                            w=5)

# Show matrix
print("\nMatrix shape (samples, timesteps, features): %s" % str(matrix.shape))

# Save
np.save('./100.matrix.simp.dm.pw.npy', matrix)
