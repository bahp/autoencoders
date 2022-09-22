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
count_org.to_csv(Path(args.path) / '02.counts.organism.csv')



# ---------------------------------------
#
# ---------------------------------------


# ---------------------------
# Correlation
# ---------------------------
# Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Remove some columns
aux = df.copy(deep=True) \
    .drop(columns=[
        'Index',
        'Unnamed: 0',
        'PersonID',
        'idx_to_sample',
        'idx_to_death'],
        errors='ignore')

# Compute correlation
corr = aux.corr()

# Show
print(corr)

# Display correlation
sns.set(rc={'figure.figsize':(16,8)})
f = sns.heatmap(corr, annot=True, fmt='.2g',
    cmap='coolwarm', annot_kws={"size": 8}) \
    .set(title='Correlation')

# Save
plt.tight_layout()
plt.savefig(Path(args.path) / '02.correlation.png')


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
    title='Correlation (Pearson)',
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
fig.write_html(Path(args.path) / '02.correlation.html')

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