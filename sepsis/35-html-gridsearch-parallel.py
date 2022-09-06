import pandas as pd
from pathlib import Path

# How to display files.
DISPLAY_TOGETHER = False

# Path to load grid-search results.
PATH = Path('objects/results/classification/')


# Load all DataFrames
dfs = [
    (f, pd.read_csv(f))
        for f in PATH.glob('**/gridsearch-*.csv')
]

# Combine into one
if DISPLAY_TOGETHER:
    dfs = [('combined', pd.concat([e[1] for e in dfs]))]

# -------------------------
#
# -------------------------
# Libraries
import plotly.express as px

dimensions = [e for e in dfs[0][1].columns
    if 'mean' in e]

dimensions = [
    'gmean',
    'roc_auc',
    'sens',
    'spec',
    'support0',
    'support1'
]

# Manual
dimensions = [
    'mean_test_gmean',
    'mean_test_roc_auc',
    'mean_test_sens',
    'mean_test_spec',
    'mean_test_support0',
    'mean_test_support1',

    'mean_train_gmean',
    'mean_train_roc_auc',
    'mean_train_sens',
    'mean_train_spec',
    'mean_train_support0',
    'mean_train_support1'
]

# Create labels
def create_label(x):
    x = x.replace('mean_', '')
    x = x.replace('_', ' ')
    return x.title()
labels = {e:create_label(e) for e in dimensions}

# Plot
for name, df in dfs:
    fig = px.parallel_coordinates(df,
        color="mean_test_gmean",
        dimensions=dimensions,
        labels=labels,
        color_continuous_scale=px.colors.diverging.Tealrose,
        #color_continuous_midpoint=2
        title=str(name)
    )
    fig.show()
    fig.write_html(name.parent / '01.parallel.gridsearch.html')