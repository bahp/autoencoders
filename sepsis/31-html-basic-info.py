#
import pandas as pd
from pathlib import Path

# The output path
OUTPATH = Path('./objects/datasets/test')

# Load data
df = pd.read_csv(OUTPATH / 'data.csv')

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
count_org.to_csv(OUTPATH / '02.counts.organism.csv')



# ---------------------------------------
#
# ---------------------------------------


# ---------------------------
# Correlation
# ---------------------------
# Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation
corr = df.corr()

# Display correlation
sns.set(rc={'figure.figsize':(16,8)})
f = sns.heatmap(corr, annot=True, fmt='.2g',
    cmap='coolwarm', annot_kws={"size": 8}) \
    .set(title='Correlation')

# Save
plt.tight_layout()
plt.savefig(OUTPATH / '02.correlation.png')

"""
# Libraries
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=corr.index,
        y=corr.columns,
        z = df.corr(),
        type = 'heatmap',
        colorscale = 'Viridis'
    )
)

fig.show()
"""