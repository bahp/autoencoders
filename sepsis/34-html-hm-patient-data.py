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



"""
# Cleaning constants that will be used to filter the
# original data to keep the top N bio-markers and
# organisms with lesser number of missing inputs.
TOPN_BIO, TOPN_ORG = 15, 5

# To keep only those records with information
# n days before and n days after the microbiology
# sample was collected.
DAY_S, DAY_E, FILTER_DAY = -1, 1, False

# Remove those test that have a zero value. Note that
# zero might be an outcome of a laboratory test in
# certain situations (patients who are immunocompromised)
FILTER_ZERO = False

# Keep only most common bio-markers
FILTER_BIO = False

# Keep only most common organisms
FILTER_ORG = False

# This variable enables the computation of the data
# report using the pandas-profiling package. Note that
# it is computed on the pivoted dataset.
COMPUTE_REPORT = True
PLOT_PIVOT = True
PLOT_RESAMPLE = True

# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = Path('./objects/datasets/data.csv')
DEATH = './objects/datasets/deaths.csv'

# Load data
data = pd.read_csv(PATH,
                   #nrows=5000,
                   dtype={'PersonID': 'str'},
                   parse_dates=['date_collected',  # pathology date
                                'date_sample',  # microbiology date
                                'date_outcome',  # outcome date
                                'patient_dob'])

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
# data['day'] = data.date


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
    .groupby('PersonID') \
    .count()

# Show
print("Patients with more than one organism: %s" %
      (micro_count.values > 1).sum())
print("Patients with more than one date_sample: %s" %
      (dsample_count.values > 1).sum())

# Compute the day.
data['day'] = (data.date_collected - data.date_sample).dt.days

# data.to_csv('./objects/data.death.csv')

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
DATE = 'date_collected'

# Pivot table.
piv = pd.pivot_table(data,
                     values=['result'],
                     index=['PersonID', DATE], # date_collected
                     columns=['code'])

# a = piv.resample('d', on=1).mean()


# Basic formatting
piv.columns = piv.columns.droplevel(0)
piv = piv.reset_index()
# piv.date_collected = piv.date_collected.dt.normalize()
piv = piv.drop_duplicates(subset=['PersonID', DATE])
piv = piv.set_index(['PersonID', DATE])
# piv.index = piv.index.normalize()

# data = data.set_index(['date_collected', 'PersonID'])

# piv = piv.merge(data)
# piv = piv.join(data[['micro_code', 'death', 'day', 'death_days']], how='left')


# Save
#fig = plot_windows(piv, x=top_bio, title='PIVOTED')
#fig.write_html(OUTPATH / 'data.pivoted.html')



def resample_01(df, ffill=True):
    return df.droplevel(0) \
        .resample('1D').asfreq() \
        #.ffill() # filling missing!

# Re-sample
rsmp = piv \
    .groupby(level=0) \
    .apply(resample_01) \


def a2b_map(df, c1, c2):
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
rsmp['death'] = rsmp.idx_to_death.abs() < 10


# Log
print("\nFinal:")
print(rsmp)

from utils.settings import _SCALERS
aux = rsmp.copy(deep=True)


rows=10
cols=10
x = top_bio
"""

from utils.settings import _SCALERS

# The output path
OUTPATH = Path('./objects/datasets/test')

aux = pd.read_csv(OUTPATH / 'data.csv')

print(aux)

F = ['WBC',
     'PLT',
     'HGB',
     'MCV',
     'HCT',
     'MCHC',
     'MCH',
     'RBC',
     'RDW',
     'LY' ,
     'MPV',
     'NE' ,
     'BA' ,
     'EO' ,
     'MO' ,
     'NRBCA',
     'NEUT',
     'BASO',
     'EOS',
     'MONO',
     'HCT_d1',
     'PLT_d1']


aux[F] = _SCALERS['mmx'].fit_transform(aux[F])


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

#aux = aux[aux.PersonID.isin(['105053'])]

# Get person score
person_score = aux.groupby('PersonID') \
    .apply(score, day_s=-5, day_e=5, features=F) \
    .sort_values(ascending=False)

# Add score to aux
#aux = aux.merge(person_score.to_frame(), on='PersonID', how='left')

#print(aux)



# Filter
aux = aux[aux.PersonID.isin( \
    person_score[person_score >= 0.75].index)]


# Show
print("\nPerson score (window completeness):")
print(person_score)
print("\nOver 0.75 completeness: %s" % aux.PersonID.nunique())






# -----------------------------
# Display
# -----------------------------
# Libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Compute groups
g = list(aux.groupby('PersonID'))[:50]

# Configure
COLS = 8
ROWS = (len(g) // COLS) + 1
SUBPLOT_TITLES = ['%s' % i for i in range(ROWS*COLS)]

# Show information
print("\nFigure grid: (%s, %s)" % (ROWS, COLS))

# Create Figure
fig = make_subplots(rows=ROWS, cols=COLS,
    shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.02,
    subplot_titles=SUBPLOT_TITLES)

# Display
for i, (name, df_) in enumerate(g):

    # Break
    if i > (ROWS * COLS) - 1:
        break

    # Variables
    x = F
    #y = df_.index.get_level_values(1).day.astype(str)
    y = df_.day
    z = df_[x]
    row = (i // COLS) + 1
    col = (i % COLS) + 1
    title = '%s (%.2f)' % (name, person_score[name])

    # Display heat-map
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            #zmin=g[x].to_numpy().min(),
            #zmax=g[x].to_numpy().max(),
            coloraxis='coloraxis',
        ),
        row=row, col=col)

    """
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5, y=1.3, showarrow=False,
        text=name, row=row, col=col)
    """

    # Update text (hacks)
    fig.layout.annotations[i].update(text=title)


# Update axes
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
    tickmode='linear',
    tickfont_size=8
)
fig.update_xaxes(
    tickmode='linear',
    tickfont_size=8
)
fig.update_layout(
    title=dict(
        text="Completeness of patient data (%s patients over 0.75 cut-off)" \
             % aux.PersonID.nunique(),
        x=0.5
    ),
    height=400*ROWS,
    width=200*COLS,
    coloraxis=dict(
        colorscale='Viridis'),
    showlegend=False
)
fig.update_coloraxes(
    colorscale='Viridis'
)

fig.show()
fig.write_html(OUTPATH / '05.hm.patient.data.html')