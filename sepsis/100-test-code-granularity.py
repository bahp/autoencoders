# Libraries
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore", category=RuntimeWarning)

# ------------------------------------
# Constants
# ------------------------------------

COLUMNS = [
    'PersonID',
    'date_collected',
    'code',
    'examination_code'
]

EXAMINATION_CODES = [
    'WBS',
    'FBC',
    'CRP'
]

LABORATORY_CODES = [

]

PATIENTS = []

# Path
PATH = '../datasets/Sepsis/raw.csv'

# Read data
df = pd.read_csv(PATH,
    #nrows=10000,
    dtype={'PersonID': 'str'},
    parse_dates=[
        'patient_dob',
        'date_collected',
        'date_outcome',
        'date_sample'
    ]
)

# Filter
df = df[COLUMNS]
df = df.loc[df.examination_code.isin(EXAMINATION_CODES)]

# Ensure it is sorted
df = df.sort_values(by=['PersonID', 'date_collected'])

# Method
def granularity(x):
    return x \
        .diff().astype('timedelta64[h]') \
            .median()

# Compute time different between consecutive samples.
aux = df.groupby(['PersonID', 'code']) \
    .agg(['size', granularity]) \
    .unstack() \
    .describe() \
    .stack(level=2) \
    .unstack(level=0)

# Save to csv
aux.to_csv('../datasets/Sepsis/granularity.csv')