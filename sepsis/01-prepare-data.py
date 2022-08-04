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
# Cleaning constants that will be used to filter the
# original data to keep the top N bio-markers and
# organisms with lesser number of missing inputs.
TOPN_BIO, TOPN_ORG = 15, 5

# To keep only those records with information
# n days before ad n days after the microbiology
# sample was collected.
DAY_S, DAY_E = -5, 1

# This variable enables the computation of the data
# report using the pandas-profiling package. Note that
# it is computed on the pivoted dataset.
COMPUTE_REPORT = False
PLOT_PIVOT = False
PLOT_RESAMPLE = False


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = './objects/datasets/data.csv'
DEATH = './objects/datasets/deaths.csv'

# Load data
data = pd.read_csv(PATH,
    #nrows=100000,
    dtype={'PersonID': 'str'},
    parse_dates=['date_collected',  # pathology date
                 'date_sample',     # microbiology date
                 'date_outcome',    # ???
                 'patient_dob'])

# Load deaths
deaths = pd.read_csv(DEATH,
    dtype={'PersonID': 'str'},
    parse_dates=['date_death'])
deaths = deaths.dropna(how='any')
deaths = deaths.drop(columns='Unnamed: 0')

# Combine deaths with data
data = data.merge(deaths, on='PersonID')
data['death_days'] = (data.date_sample - data.date_death).dt.days
data['death'] = data.death_days.abs() < 600
data.PersonID = data.PersonID.astype(str)

# If before it will be about 24h gaps.
#data['day'] = data.date

# Normalize dates
for d in ['date_collected',
          'date_sample',
          'date_outcome',
          'date_death',
          'patient_dob']:
    data[d] = data[d].dt.normalize()


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
data = data[(data.day >= DAY_S) & (data.day <= DAY_E)]

# Keep only patients with all the data. Or
# maybe keep patients with at least n days
# worth of data.

# Keep only top N bio-markers.
data = data[data.code.isin(top_bio)]

# Keep only top N organisms.
data = data[data.micro_code.isin(top_org)]

# Brief summary.
print("\nInformation:")
print("Shape:    %s" % str(data.shape))
print("Patients: %s" % data.PersonID.nunique())


# ----------------------------
# Step 01: Pivot
# ----------------------------
# Pivot table.
piv = pd.pivot_table(data,
    values=['result'],
    index=['PersonID', 'date_collected'],
    columns=['code'])




#a = piv.resample('d', on=1).mean()



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


if PLOT_PIVOT:
    fig = plot_windows(piv, x=top_bio, title='PIVOTED')


# ----------------------
# Step 02: Resample
# ----------------------
"""
The method resample_01 is filling missing values for each
patient using the forward fill. We could also top the amount
of consecutive empty days that are allowed to be filled.
"""

def resample_01(df, ffill=True):
    return df.droplevel(0) \
        .resample('1D').asfreq() \
        .ffill() # filling missing!

# Re-sample
rsmp = piv \
    .groupby(level=0) \
    .apply(resample_01) \

# Log
print("\nResampled:")
print(rsmp)

# Plot
if PLOT_RESAMPLE:
    fig = plot_windows(rsmp, x=top_bio, title='RESAMPLED')


# ---------------------
# Step 03: Phenotypes
# ---------------------
"""
This section adds information that is of interest:

 - micro_code: str
    The code of the organism
    
 - date_sample: date
    The date the microbiology sample was collected
    
 - day_to_death
    Number of days till patient death (from date_collected)
    
 - death
    Whether the patient died
"""

def a2b_map(df, c1, c2):
    """"""
    aux = df[[c1, c2]] \
        .dropna(how='any') \
        .drop_duplicates()
    print(aux.groupby(c1).count() > 1)
    if (aux.groupby(c1).size() > 1).any():
        print("   Warning! There are multiple values of %s for %s." % (c2, c1))
    return dict(zip(aux[c1], aux[c2]))

# Reset index
rsmp = rsmp.reset_index()

# Add microorganism code
d = a2b_map(data, 'PersonID', 'micro_code')
rsmp['micro_code'] = rsmp.PersonID.map(d)

# Add date_sample
d = a2b_map(data, 'PersonID', 'date_sample')
rsmp['date_sample'] = rsmp.PersonID.map(d)

# Add date_death
d = a2b_map(data, 'PersonID', 'date_death')
rsmp['date_death'] = rsmp.PersonID.map(d)

# Add day
rsmp['day'] = (rsmp.date_collected - rsmp.date_sample).dt.days
rsmp['idx_to_sample'] = (rsmp.date_collected - rsmp.date_sample).dt.days
rsmp['idx_to_death'] = (rsmp.date_collected - rsmp.date_death).dt.days
rsmp['death'] = rsmp.idx_to_death.abs() < 90

# Log
print("\nFinal:")
print(rsmp)

# Filename
filename = 'tidy'

# Save
rsmp.to_csv('./objects/datasets/%s.csv' % filename)


# -------------------------
# Step 05: Pandas profiling
# -------------------------
# Libraries
from pandas_profiling import ProfileReport

if COMPUTE_REPORT:

    # Create report
    profile = ProfileReport(rsmp,
        title="Sepsis Dataset",
        explorative=False,
        minimal=True)

    # Save report
    profile.to_file("./objects/datasets/report-%s.html" % filename)