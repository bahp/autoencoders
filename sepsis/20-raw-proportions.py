# Libraries
import pandas as pd
import numpy as np

from pathlib import Path


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define paths
PATH = Path('./objects/datasets/data-all.csv')

# Load bio-markers
data = pd.read_csv(PATH,
    #nrows=1000,
    dtype={'PersonID': 'str'},
    parse_dates=['date_collected',  # pathology date
                 'date_sample',     # microbiology date
                 'date_outcome',    # outcome date
                 'patient_dob'])

# .. note: There are values for the same patient, datetime
#          of collection and bio-marker with different results
#          ... which one should we keep?
data = data.drop_duplicates(
    subset=['PersonID', 'date_collected', 'code'],
    keep='last'
)

# Do some formatting.
data = data.replace({
    'micro_code': {
        'CONS': 'CNS'
    }
})

# Variables
top_bio = data.code \
    .value_counts() \
    .head(55) \
    #.index.tolist()

top_org = data.micro_code \
    .value_counts() \
    .head(50) \
    #.index.tolist()

# Show
print("\nShape:")
print(data.shape)
print("\nShow top:")
print(top_bio)
print(top_org)

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

# Compute counts
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
# Create pivot table
piv = pd.pivot_table(data,
    values=['result'],
    index=['PersonID', 'date_collected'],
    columns=['code'])

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
fig = px.imshow(coocc,
    color_continuous_scale='Reds',
    #zmin=-100.0, zmax=100.0,
    text_auto='.0f'
)

fig.update_layout(
    title=dict(
        text=str('Co-occurrence'),
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

# Show (co-occurrence percent)
fig = px.imshow(coocc_pct,
    color_continuous_scale='Reds',
    #zmin=-100.0, zmax=100.0,
    text_auto='.0f'
)

fig.update_layout(
    title=dict(
        text=str('Co-occurrence'),
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



# -------------------------------------
# Cluster MAP
# ------------------------------------
# Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Display cluster map
sns.clustermap(coocc_pct, center=0, cmap="vlag",
    #row_colors=network_colors, col_colors=network_colors,
    dendrogram_ratio=(.1, .2),
    cbar_pos=(.02, .32, .03, .2),
    linewidths=.75, figsize=(10, 10))

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

