# Libraries
import argparse
import pandas as pd
import numpy as np

from pathlib import Path

# -------------------------
# Constants
# -------------------------
# When displaying the co-occurrence matrix, it indicates
# whether the whole date and time should match, or whether
# we should only consider the date.
USE_DATETIME = False

# Define paths
PATH = Path('./objects/datasets/data-all.csv')
PATH = Path('../datasets/Sepsis/raw.csv')
PATH = Path('../datasets/Sepsis/data-set1.csv')

# -------------------------
# Parameters
# -------------------------
# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, nargs='?',
                    const=PATH, default=PATH,
                    help="data path.")
args = parser.parse_args()


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Load bio-markers
data = pd.read_csv(PATH,
    #nrows=10000,
    dtype={'PersonID': 'str'},
    parse_dates=['date_collected',  # pathology date
                 'date_sample',     # microbiology date
                 'date_outcome',    # outcome date
                 'patient_dob'],
    usecols=[
        'PersonID',
        'date_collected',
        'date_sample',
        'date_outcome',
        'patient_dob',
        'code',
        'examination_code',
        'result',
        'unit'
    ])

# .. note: There are values for the same patient, datetime
#          of collection and bio-marker with different results
#          ... which one should we keep?
data = data.drop_duplicates(
    subset=['PersonID', 'date_collected', 'code'],
    keep='last'
)

# Keep only FBC
data = data[data.examination_code.isin([
    'FBC',
    'CRP',
    'PCT',
    'WBS',
    'PLT',
    'UE',
    'FER',
    'FIB',
    'BONE',
    'COAG',
    'BIL',
    'ALT',
    'DDIMER'
])]


#print(data)##

aux = data[['code', 'examination_code']].drop_duplicates()
aux.to_html('here.html')
#print(aux)

#import sys
#sys.exit()
# Do some formatting.
#data = data.replace({
#    'micro_code': {
#        'CONS': 'CNS'
#    }
#})

# Variables
top_bio = data.code \
    .value_counts() \
    .head(55) \
    #.index.tolist()

#top_org = data.micro_code \
#    .value_counts() \
#    .head(50) \
    #.index.tolist()

# Show
print("\nShape:")
print(data.shape)
print("\nShow top:")
print(top_bio)
#print(top_org)

# Keep only top N and set rest as OTHER.
idxs = data.code.isin(top_bio.index.tolist())
data.loc[~idxs, 'code'] = 'OTHER'


# -----------------------------------------
# Display PROPORTIONS
# -----------------------------------------
# Libraries
import plotly.express as px

# Compute counts
counts = data.code.value_counts()

# Plot
fig = px.pie(counts,
    values=counts.values,
    names=counts.index,
    title='Biomarker proportions'
)
fig.show()

# Compute the number of patients with each
# of the microorganisms to see how frequent
# they are in the data.
if 'micro_code' in data:

    counts = data.drop_duplicates(
        subset=[
            'PersonID',
            'micro_code'
        ]
    ).micro_code.value_counts().head(50)

    # Plot
    fig = px.pie(counts,
        values=counts.values,
        names=counts.index,
        title='Microorganism proportions (# patients)'
    )
    fig.show()


# -----------------------------------------
# Display CO-OCCURRENCE
# -----------------------------------------
# Remove other
data = data.loc[~data.code.isin(['OTHER'])]

# Normalize date
# .. note: When normalizing the data, those bio-markers
#          which are sampled frequently (e.g. hourly)
#          will appear with less frequency since different
#          hours for a same date will be merge to such date.
if not USE_DATETIME:
    data.date_collected = data.date_collected.dt.normalize()

# Create pivot table
piv = pd.pivot_table(data,
    values=['result'],
    index=['PersonID', 'date_collected'],
    columns=['examination_code', 'code'])

# Format pivot table
piv = piv.notna().astype(int)
piv.columns = piv.columns.droplevel(0)
piv.index = piv.index.droplevel(0)

#Compute co-occurrence (numpy)
# v.T.dot(v) # numpy
# np.einsum('ij,ik->jk', df, df)

# Compute co-occurrence
coocc = piv.T.dot(piv)
coocc_pct = (coocc / np.diag(coocc)) * 100
#coocc_pct = coocc.div(np.diag(coocc)) * 100


def to_flat_index(m):
    return ['-'.join(col).rstrip('_') for col in m]

def to_flat_matrix(m):
    aux = m.copy(deep=True)
    aux.columns = to_flat_index(aux.columns.values)
    aux.index = to_flat_index(aux.index.values)
    return aux

# Format to plot (panel and bio-marker code)
#aux1 = to_flat_matrix(coocc)
#aux2 = to_flat_matrix(coocc_pct)


# Format to plot (only bio-marker code)
aux1 = coocc.copy(deep=True) \
    .droplevel(0, axis=0) \
    .droplevel(0, axis=1)
aux2 = coocc_pct.copy(deep=True) \
    .droplevel(0, axis=0) \
    .droplevel(0, axis=1)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

if not check_symmetric(coocc.to_numpy()):
    print("[Warning] Co-occurrence matrix (ratio) is not symmetric!")

if not check_symmetric(coocc_pct.to_numpy(), rtol=1, atol=1):
    print("[Warning] Co-occurrence matrix (pct) is not symmetric!")

# .. note: It makes sense that the co-occurrence matrix in pct
#          is not symmetric since it is divided by the corresponding
#          column (e.g. alb-crp, alb/crp  crp/alb). We would need to
#          keep the maximum of the two rather than the diagonal if
#          we want a symmetric table.

# Show (co-occurrence)
fig = px.imshow(aux1,
    color_continuous_scale='Reds',
    #zmin=-100.0, zmax=100.0,
    text_auto='.0f'
)

fig.update_layout(
    title=dict(
        text=str('Co-occurrence (#)'),
        x=0.5
    ),
    yaxis=dict(
        tickmode='linear',
        tickfont=dict(size=8)
    ),
    xaxis = dict(
        tickmode='linear',
        tickfont=dict(size=8)
    )
)
fig.write_html(Path(args.path) / 'graphs' / '05.hm.cooccurrence.html')
fig.show()

# Show (co-occurrence percent)
fig = px.imshow(aux2,
    color_continuous_scale='Reds',
    #zmin=-100.0, zmax=100.0,
    text_auto='.0f'
)

fig.update_layout(
    title=dict(
        text=str('Co-occurrence (%)'),
        x=0.5
    ),
    yaxis=dict(
        tickmode='linear',
        tickfont=dict(size=8)
    ),
    xaxis = dict(
        tickmode='linear',
        tickfont=dict(size=8)
    )
)
fig.show()
fig.write_html(Path(args.path) / 'graphs' / '05.hm.cooccurrence.pct.html')






# -------------------------------------
# Cluster MAP
# ------------------------------------
# Libraries
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=0.5)

# Panel names
N_names = coocc_pct.index \
    .get_level_values('examination_code') \
    .unique()

# Number of different panels
N = len(N_names)

# Create network (panel) colors
network_pal = sns.husl_palette(N, s=0.45)
network_lut = dict(zip(map(str, N_names), network_pal))
networks = coocc_pct.columns.get_level_values('examination_code')
network_colors = pd.Series(networks, index=coocc_pct.columns) \
    .map(network_lut)

# Display cluster map (co-occurrence #)
g1 = sns.clustermap(coocc, cmap="Reds",
    row_colors=network_colors, #col_colors=network_colors,
    dendrogram_ratio=(.1, .1),
    cbar_pos=(.01, .02, .03, .2),
    annot=False, fmt=".0f", annot_kws={"fontsize":6}, # Add numbers
    row_cluster=False, col_cluster=False,            # Enable clustering
    linewidths=.10, figsize=(10, 10))
g1.ax_row_dendrogram.set_visible(False)
g1.ax_col_dendrogram.set_visible(False)
g1.fig.suptitle('Co-occurrence matrix (#)')

# Display cluster map (co-occurrence %)
g2 = sns.clustermap(coocc_pct, cmap="Reds",#vlag
    row_colors=network_colors, #col_colors=network_colors,
    dendrogram_ratio=(.1, .1),
    cbar_pos=(.01, .02, .03, .2),
    annot=True, fmt=".0f", annot_kws={"fontsize":6}, # Add numbers
    row_cluster=False, col_cluster=False,            # Enable clustering
    linewidths=.10, figsize=(10, 10))
g2.ax_row_dendrogram.set_visible(False)
g2.ax_col_dendrogram.set_visible(False)
g2.fig.suptitle('Co-occurrence matrix (%)')

# Show
plt.show()


# -----------------------
# Manually group
# -----------------------
"""
# Libraries
from sklearn import cluster

# Numpy
a = coocc_pct.to_numpy()

# Compute
model = cluster.AgglomerativeClustering(n_clusters=2, affinity="euclidean").fit(a)
new_order = np.argsort(model.labels_)
ordered_dist = a[new_order] # can be your original matrix instead of dist[]
ordered_dist = ordered_dist[:,new_order]

# Show
fig = px.imshow(ordered_dist,
    color_continuous_scale='Reds',
    #zmin=-100.0, zmax=100.0,
    text_auto='.0f'
)
fig.show()
"""

