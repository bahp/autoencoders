#
import argparse
import pandas as pd
from pathlib import Path

# The output path
PATH = './objects/datasets/test'

# -------------------------
# Parameters
# -------------------------
# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, nargs='?',
                    const=PATH, default=PATH,
                    help="path containing grid-search files.")
args = parser.parse_args()



# Load data
df = pd.read_csv(Path(args.path) / 'data.csv')

# --------------------------------------
# Biomarkers
# --------------------------------------
# Count
counts = df.count().sort_values(ascending=False)

# Show
print(df)
print(counts)


# --------------------------------------
# Organisms
# --------------------------------------
# Count #records per organism
count_org_records = df.micro_code \
    .value_counts() \
    .sort_values(ascending=False)

# Count #patients per organism
count_org_patient = \
    df[['PersonID', 'micro_code']] \
        .drop_duplicates().micro_code \
        .value_counts() \
        .sort_values(ascending=False)

# Count average number of days till death from
# the date of collection of the microbiology
# sample.

# Create Data-Frame
count_org = pd.DataFrame()
count_org['patients'] = count_org_patient
count_org['records'] = count_org_records
count_org['ratio'] = count_org.records / count_org.patients

# Show
print(count_org)

# Save
#count_org.to_csv(Path(args.path) / 'graphs' / '02.counts.organism.csv')
count_org.to_html(Path(args.path) / 'graphs' / '02.counts.organism.html')


# ---------------------------------------
#
# ---------------------------------------


# ---------------------------
# Correlation
# ---------------------------
# Keep
keep = [c for c in df.columns if c.isupper()]

# Remove some columns
aux = df[keep].copy(deep=True)

# Compute correlation
corr = aux.corr()

# Show
print(corr)

"""
# Display correlation
sns.set(rc={'figure.figsize':(16,8)})
f = sns.heatmap(corr, annot=True, fmt='.2g',
    cmap='coolwarm', annot_kws={"size": 8}) \
    .set(title='Correlation')

# Save
plt.tight_layout()
plt.savefig(Path(args.path) / 'graphs' / '02.correlation.png')
"""

# Libraries
import plotly.graph_objects as go
import plotly.express as px

# Reverse corr
#corr = corr[corr.columns[::-1]].T

# Plot using imshow
fig = px.imshow(corr*100,
    color_continuous_scale='RdBu_r',
    zmin=-100.0, zmax=100.0,
    text_auto='.0f')

fig.update_layout(
    title=dict(
        text='Correlation (Pearson)',
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
fig.write_html(Path(args.path) / 'graphs' / '02.correlation.html')

"""
# Plot
fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=corr.T.index,
        y=corr.T.columns,
        z=df.corr(),
        type='heatmap',
        colorscale='RdBu_r',
    )
)

fig.update_layout(
    title='Correlation (Pearson)'
)
fig.data[0].update(zmin=-1.0, zmax=1.0)
fig.show()
"""


# -----------------------------------------
# Display CO-OCCURRENCE
# -----------------------------------------
# Libraries
import numpy as np

# When displaying the co-occurrence matrix, it indicates
# whether the whole date and time should match, or whether
# we should only consider the date.
USE_DATETIME = False

# Normalize date
# .. note: When normalizing the data, those bio-markers
#          which are sampled frequently (e.g. hourly)
#          will appear with less frequency since different
#          hours for a same date will be merge to such date.
if not USE_DATETIME:
    df.date_collected = pd.to_datetime(df.date_collected)
    df.date_collected = df.date_collected.dt.normalize()

# Keep
keep = [c for c in df.columns if c.isupper()]

# Format pivot table
piv = df[keep].notna().astype(int)

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

def to_cluster(m):
    # Libraries
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=4, affinity="euclidean")
    # Compute
    model = model.fit(m)
    new_order = np.argsort(model.labels_)
    print(new_order)
    print(m.columns)
    #ordered_dist = a[new_order]  # can be your original matrix instead of dist[]
    #ordered_dist = ordered_dist[:, new_order]
    print(m)
    print(m.iloc[:, new_order])
    return m.iloc[:, new_order]

# Format to plot (panel and bio-marker code)
#aux1 = to_flat_matrix(coocc)
#aux2 = to_flat_matrix(coocc_pct)


# Format to plot (only bio-marker code)
aux1 = coocc.copy(deep=True)
aux2 = coocc_pct.copy(deep=True)

# Agglomerative
#aux1 = to_cluster(aux1)
#aux2 = to_cluster(aux2)

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