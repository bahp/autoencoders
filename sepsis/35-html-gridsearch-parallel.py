# Generic
import argparse
import pandas as pd

# Specific
from pathlib import Path

# -----------------------
# Constants
# -----------------------
# How to display files.
DISPLAY_TOGETHER = True

# Define keyword
KEYWORD = 'gridsearch'
#KEYWORD = 'checkpoint'

# Manual
DIMENSIONS = [
    'folder',
    'method',
    'size',

    'mean_test_support0',
    'mean_test_support1',
    'mean_test_gmean',
    'mean_test_roc_auc',
    'mean_test_sens',
    'mean_test_spec',

    'mean_train_support0',
    'mean_train_support1',
    'mean_train_gmean',
    'mean_train_roc_auc',
    'mean_train_sens',
    'mean_train_spec',
]

# COLOR
COLOR = 'mean_test_gmean'

# Path to load grid-search results.
PATH = Path('objects/results/classification/normal/220906-191306')
PATH = Path('objects/results/classification/delta/220907-122220')



# ----------------------------------
# Load arguments
# ----------------------------------
# Load
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, nargs='?',
                    const=PATH, default=PATH,
                    help="path containing grid-search files.")
parser.add_argument("--method", type=str, nargs='?',
                    const=None, default=None,
                    help="filter by metho.")
parser.add_argument("--keyword", type=str, nargs='?',
                    const=KEYWORD, default=KEYWORD,
                    help="prefix of the files (e.g. gridsearch).")
parser.add_argument("--combine", action='store_true',
                    help='whether to combine in a single graph.')
parser.add_argument("--color", type=str, nargs='?',
                    const=COLOR, default=COLOR,
                    help="column to compute colors.")
args = parser.parse_args()


# -----------------------
# Load data
# -----------------------
# Load all DataFrames
dfs = [(f, pd.read_csv(f))
    for f in Path(args.path) \
        .glob('**/%s-*.csv' % args.keyword)
]

# Combine into one
if args.combine:
    dfs = [('combined', pd.concat([e[1] for e in dfs]))]

# Add columns
for name, df in dfs:
    if 'folder' in df:
        df['method'] = df.folder.str.split("-").str[-1]
        df['size'] = df.folder.map(df.groupby('folder').size())


# -------------------------
# Visualise
# -------------------------
# Libraries
import plotly.express as px
import plotly.graph_objs as go
from utils.plotly.parcoords import get_dimensions

# Plot
for name, df in dfs:

    dnames = DIMENSIONS
    if args.method is not None:
        df = df[df.method == args.method]
        p = [e for e in df.columns if e.startswith('param_')]
        dnames = DIMENSIONS + p

    # Variables
    cols = [d for d in dnames if d in df]
    dims = get_dimensions(df[cols])
    color = df[args.color] if args.color in df else None

    # Display
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=color,
                colorscale='Tealrose',
                showscale=True,
                #cmin=,
                #cmax=
            ),
            labelangle=-45,
            dimensions=dims
        )
    )
    fig['layout'].update(
        margin=dict(l=150, r=150),
        title=dict(
            text=str(name),
            x=0.5, y=0.03
        )
    )
    fig.show()
    #fig.write_html(name.parent / '01.parallel.gridsearch.html')


    """
    fig = px.parallel_coordinates(df_,
        color="mean_test_gmean",
        dimensions=dimensions,
        #labels=labels,
        color_continuous_scale=px.colors.diverging.Tealrose,
        #color_continuous_midpoint=2
        title=str(name)
    )
    """