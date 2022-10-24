# Generic
import yaml
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
PATH = Path('objects/results/classification/normal/220923-192623')
#PATH = Path('objects/results/classification/delta/220915-091843')



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
                    help="filter by method.")
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
"""
# Create index column
table = pd.DataFrame()
for path, df in dfs:
    with open(path.with_suffix('.yaml')) as file:
        info = pd.json_normalize(yaml.full_load(file))
        info['folder'] = path.parent.stem
        info['link'] = '''<a href="./hiplot.%s.%s.html" target="_blank"> Link </a>''' % (
            path.parent.parent.stem,
            path.parent.stem)
        table = table.append(info)


# Filter information
table = table[[
    'features',
    'filter.day.start',
    'filter.day.end',
    'strategy',
    'search.strategy',
    'folder',
    'link'
]]
"""

# Output
OUTPUT = Path('./parallel')

# Create directory
OUTPUT.mkdir(parents=True, exist_ok=True)

"""
# Save
table.to_html(OUTPUT / 'hiplot.table.html',
    index=False, justify='center',
    escape=False)
"""



# Add path information
for path, df in dfs:
    #print("Loaded file... %s" % path)
    df['path'] = path
    df['strategy'] = path.parent.parent.stem
    if 'folder' in df:
        df['method'] = df.folder.str.split("-").str[-1]
        df['size'] = df.folder.map(df.groupby('folder').size())

# Combine into one
if args.combine:
    aux = pd.concat([e[1] for e in dfs])
    aux = aux.reset_index(drop=True)
    dfs = [('combined', aux)]

# Add columns
#for name, df in dfs:
#    print(name)
#    if 'folder' in df:
#        df['method'] = df.folder.str.split("-").str[-1]
#        df['size'] = df.folder.map(df.groupby('folder').size())

#import sys
#sys.exit()
# -------------------------
# Visualise
# -------------------------
# Libraries
import plotly.express as px
import plotly.graph_objs as go
import hiplot as hip

from utils.plotly.parcoords import get_dimensions

# Plot
for name, df in dfs:

    dnames = DIMENSIONS
    if args.method is not None:
        df = df[df.method == args.method]
        df = df.dropna(axis=1, how='all')
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
    #fig.show()
    #fig.write_html(name.parent / '01.parallel.gridsearch.html')


    if args.combine:
        fname = 'hiplot.combined.html'
    else:
        fname = 'hiplot.%s.%s.html' % (
            name.parent.parent.stem,
            name.parent.stem)

    # Create hiplot.
    hip.Experiment.from_iterable(
        df[cols + ['strategy']].round(decimals=3) \
            .to_dict(orient='records')
    ).to_html(OUTPUT / fname)


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