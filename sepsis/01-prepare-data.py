"""
Author: Bernard
Description:
"""

# Libraries
import pandas as pd
import numpy as np

from pathlib import Path
from utils.plot.heatmaps import plot_windows


# --------------------------------------------------
# Configuration
# --------------------------------------------------
# The output path
OUTPATH = Path('./objects/datasets/test-fbc-pct-crp-wbs')

# The output filename
FILENAME_DATA = 'data'

# Cleaning constants that will be used to filter the
# original data to keep the top N bio-markers and
# organisms with lesser number of missing inputs.
TOPN_BIO, TOPN_ORG = 15, 5

# To keep only those records with information
# n days before and n days after the microbiology
# sample was collected.
DAY_S, DAY_E, FILTER_DAY  = -1, 1, False

# Remove those test that have a zero value. Note that
# zero might be an outcome of a laboratory test in
# certain situations (patients who are immunocompromised)
FILTER_ZERO = False

# Keep only most common bio-markers
FILTER_BIO = False

# Keep only most common organisms
FILTER_ORG = False


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define paths
PATH = Path('./objects/datasets/data-fbc-pct-crp-wbs.csv')
DEATH = Path('./objects/datasets/deaths.csv')

# Load bio-markers
data = pd.read_csv(PATH,
    #nrows=100000,
    dtype={'PersonID': 'str'},
    parse_dates=['date_collected',  # pathology date
                 'date_sample',     # microbiology date
                 'date_outcome',    # outcome date
                 'patient_dob'])
data = data.replace({
    'micro_code': {
        'CONS': 'CNS'
    }
})


# Load deaths
deaths = pd.read_csv(DEATH,
    dtype={'PersonID': 'str'},
    parse_dates=['date_death'])
deaths = deaths.dropna(how='any')
deaths = deaths.drop(columns='Unnamed: 0')

# Combine deaths with data
data = data.merge(deaths, on='PersonID', how='left')

# Normalize dates
for d in ['date_collected',
          'date_sample',
          'date_outcome',
          'date_death',
          'patient_dob']:
    data[d] = data[d].dt.normalize()


data['death_days'] = (data.date_sample - data.date_death).dt.days
data['death'] = data.death_days.abs() < 600
data.death = data.death.astype(int)
data.PersonID = data.PersonID.astype(str)

# If before it will be about 24h gaps.
#data['day'] = data.date

"""
# Normalize dates
for d in ['date_collected',
          'date_sample',
          'date_outcome',
          'date_death',
          'patient_dob']:
    data[d] = data[d].dt.normalize()
"""

# -------------------------------------
# Step 00: Quick checks
# -------------------------------------
# Count organisms per patient
micro_count = data[['PersonID', 'micro_code']] \
    .drop_duplicates() \
    .groupby('PersonID') \
    .count()

# Count date_sample per patient
dsample_count = data[['PersonID', 'date_sample']] \
    .drop_duplicates() \
    .groupby('PersonID')\
    .count()

# Show
print("Patients with more than one organism: %s" %
      (micro_count.values > 1).sum())
print("Patients with more than one date_sample: %s" %
      (dsample_count.values > 1).sum())




# Compute the day.
data['day'] = (data.date_collected - data.date_sample).dt.days


#data.to_csv('./objects/data.death.csv')

# Show
print("\nData:")
print(data)
print("\nDtypes:")
print(data.dtypes)
print("\nCodes:")
print(data.code.value_counts())
print("\nOrganisms:")
print(data.micro_code.value_counts())

# Check patients have multiple organisms
# Check patients have multiple sample dates
# Check ...

# Variables
top_bio = data.code \
    .value_counts() \
    .head(TOPN_BIO) \
    .index.tolist()

top_org = data.micro_code \
    .value_counts() \
    .head(TOPN_ORG) \
    .index.tolist()

# ----------------------------
# Filters
# ----------------------------
# These might not be necessary since we just
# want to plot in the 2D space a point which
# represents X days worth of data.

# Remove PersonID is NaN.
data = data.dropna(how='any', subset=['PersonID'])

# Keep only those records with information
# n days before the microbiology sample was
# collected.
if FILTER_DAY:
    data = data[(data.day >= DAY_S) & (data.day <= DAY_E)]

# Filter those records in which result <= 0.
if FILTER_ZERO:
    data = data[data.result > 0]

# Keep only top N bio-markers.
if FILTER_BIO:
    data = data[data.code.isin(top_bio)]

# Keep only top N organisms.
if FILTER_ORG:
    data = data[data.micro_code.isin(top_org)]

# Keep only patients with all the data. Or
# maybe keep patients with at least n days
# worth of data.

# Brief summary.
print("\nInformation:")
print("Shape:    %s" % str(data.shape))
print("Patients: %s" % data.PersonID.nunique())

# Create folder if it does not exist
OUTPATH.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Step 01: Pivot
# ----------------------------
# Pivot table.
piv = pd.pivot_table(data,
    values=['result'],
    index=['PersonID', 'date_collected'],
    columns=['code'])

# Basic formatting
piv.columns = piv.columns.droplevel(0)
piv = piv.reset_index()
#piv.date_collected = piv.date_collected.dt.normalize()
piv = piv.drop_duplicates(subset=['PersonID', 'date_collected'])
piv = piv.set_index(['PersonID', 'date_collected'])


#piv.index = piv.index.normalize()
#data = data.set_index(['date_collected', 'PersonID'])
#piv = piv.merge(data)
#piv = piv.join(data[['micro_code', 'death', 'day', 'death_days']], how='left')




# ---------------------------
# Delta transformation data
# ---------------------------
# Libraries
from utils.sklearn.preprocessing import DeltaTransformer

# Create transformer object
delta = DeltaTransformer(by='PersonID',
    date='date_collected', keep=True,
    periods=[1,2], method='diff',
    resample_params={'rule': '1D'},
    function_params={'fill_method': 'ffill'})

# Transform
df_diff = delta.fit_transform(piv.reset_index())

# Create transformer object
delta = DeltaTransformer(by='PersonID',
    date='date_collected', keep=True,
    periods=[1,2], method='pct_change',
    resample_params={'rule': '1D'},
    function_params={'fill_method': 'ffill'})

# Transform (pct_change)
df_pctc = delta.fit_transform(piv.reset_index())



# ---------------------
# Step 03: Phenotypes
# ---------------------

def a2b_map(df, c1, c2):
    """"""
    aux = df[[c1, c2]] \
        .dropna(how='any') \
        .drop_duplicates()
    if (aux.groupby(c1).size() > 1).any():
        print("   Warning! There are multiple values of %s for %s."
            % (c2, c1))
    return dict(zip(aux[c1], aux[c2]))

def add_metadata(df, data):
    """Includes metadata information.

     - micro_code: str
        The code of the organism
     - date_sample: date
        The date the microbiology sample was collected
     - day_to_death
        Number of days till patient death (from date_collected)
     - death
        Whether the patient died

    Parameters
    ----------

    Returns
    -------
    """
    # Reset index
    df = df.reset_index()

    # Add microorganism code
    d = a2b_map(data, 'PersonID', 'micro_code')
    df['micro_code'] = df.PersonID.map(d)

    # Add date_sample
    d = a2b_map(data, 'PersonID', 'date_sample')
    df['date_sample'] = df.PersonID.map(d)

    # Add date_death
    d = a2b_map(data, 'PersonID', 'date_death')
    df['date_death'] = df.PersonID.map(d)

    # Add day
    df['day'] = (df.date_collected - df.date_sample).dt.days
    df['idx_to_sample'] = (df.date_collected - df.date_sample).dt.days
    df['idx_to_death'] = (df.date_collected - df.date_death).dt.days
    df['death'] = df.idx_to_death.abs() < 10

    # Add phenotypes
    df['pathogenic'] = df.micro_code != 'CNS'

    # Return
    return df

# Include metadata and phenotype
df_diff = add_metadata(df_diff, data)
df_pctc = add_metadata(df_pctc, data)

# Save
df_diff.to_csv(Path(OUTPATH) / ('%s.csv' % FILENAME_DATA))
df_diff.to_csv(Path(OUTPATH) / ('%s.diff.csv' % FILENAME_DATA))
df_pctc.to_csv(Path(OUTPATH) / ('%s.pctc.csv' % FILENAME_DATA))


# --------------------------------------
# Add phenotypes
# --------------------------------------
# Add pathogenic column
#rsmp['pathogenic'] = rsmp.micro_code != 'CNS'

# Log
print("\nFinal:")
#print(rsmp)

# Save
#rsmp.to_csv(Path(OUTPATH) / ('%s.csv' % FILENAME_DATA))


import sys
sys.exit()

# --------------------------------------
# Compute aggregated DataFrame
# --------------------------------------
# ..note: It is compute the min, max, median for the
#         whole stay of the patient. However, this should
#         be computed one the interesting period (e.g.
#         -5 to 0 days from date_sample) is selected.

# Libraries
from utils.sklearn.preprocessing import AggregateTransformer

# Define functions
aggmap = {k: ['min', 'max', 'median']
    for k in [
        'HCT',
        'HGB',
        'LY',
        'PLT',
        'WBC',
        'MCH',
        'MCHC',
        'MCV',
        'PLT',
        'RBC',
        'RDW',
        ]}
aggmap['pathogenic'] = 'max'

# Create transformer
agg = AggregateTransformer(by='PersonID',
    aggmap=aggmap, include=list(aggmap.keys()))

# Transform data
agg = agg.fit_transform(rsmp)

# Save
agg.to_csv(Path(OUTPATH) / ('%s.agg.csv' % FILENAME_DATA))
