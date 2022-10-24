# Libraries
import yaml
import pickle
import pandas as pd
import numpy as np
#ssimport utils.settings

from pathlib import Path
from utils.utils import AttrDict

# Create data
data = [
    { 'name':'VRE', 'tp':1, 'tn': 2, 'tp':3, 'fn':4 },
    { 'name':'SAUR', 'tp':1, 'tn': 2, 'tp':3, 'fn':4 },
    { 'name':'PAUR', 'tp':1, 'tn': 2, 'tp':3, 'fn':4 }
]

# Data
df = pd.DataFrame(data)

# Show
print(df)

# Path
path = Path('./objects/results/classification/normal/')
folder = '221004-154750'
filename_yaml = 'gridsearch-%s.yaml' % folder
filename_model = 'grid17-rus-simp-mmx-xgb-idx40.pkl'
filename_model = 'grid12-simp-std-xgb-idx40.pkl'

# Create filepath
filepath_data = path / folder / 'data' / 'data.csv'
filepath_yaml = path / folder / filename_yaml
filepath_pickle = path / folder / 'models' / filename_model

# Load configuration from file
with open(filepath_yaml) as file:
    CONFIG = AttrDict(yaml.full_load(file))

# Load model
with open(filepath_pickle, 'rb') as file:
    model = pickle.load(file)


# ----------------------------------
# Load data
# ----------------------------------
# Load data from csv file.
data = pd.read_csv(filepath_data)

# Make predictions
data['y_true'] = data.pathogenic
data['y_pred'] = model.predict(data[CONFIG.features])
data['y_prob'] = model.predict_proba(data[CONFIG.features])[:,1]

print(data)
print(data.columns)

def f(x):
    """"""
    # Library
    from sklearn.metrics import confusion_matrix
    # Compute
    try:
        tn, fp, fn, tp = confusion_matrix(x.y_true, x.y_pred).ravel()
    except Exception as e:
        tn, fp, fn, tp = [0, 0, 0, 0]

    #normalize = 'all'

    # Return
    return pd.Series({
        'tn':tn, 'fp':fp, 'fn':fn, 'tp':tp
    })


def custom_metrics_(y, y_pred, y_prob, n=1000):
    """This method computes the metrics.

    Parameters
    ----------
    y_true: np.array (dataframe)
        Array with original data (X).
    y_pred: np.array
        Array with transformed data (y_emb).
    y: np.array (dataframe)
        Array with the outcomes
    n: int
        The number of samples to use for distance metrics.

    Returns
    -------
    dict-like
        Dictionary with the scores.
    """
    # Libraries
    from scipy.stats import gmean
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support

    try:

        # Compute confusion matrix (binary only)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        # Compute precision recall and support.
        prec, recall, fscore, support = \
            precision_recall_fscore_support(y, y_pred)

        # Show classification report
        #print(classification_report(y, y_pred, zero_division=0))

        # Create dictionary
        d = {}
        #d['report'] = classification_report(y, y_pred)
        d['accuracy'] = accuracy_score(y, y_pred)
        #d['roc_auc'] = roc_auc_score(y, y_prob)
        d['sens'] = recall_score(y, y_pred, pos_label=1)
        d['spec'] = recall_score(y, y_pred, pos_label=0)
        d['gmean'] = gmean([d['sens'], d['spec']])
        d['tn'] = tn
        d['tp'] = tp
        d['fp'] = fp
        d['fn'] = fn
        d['count'] = len(y)

        for i,v in enumerate(support):
            d['support%s' % i] = v

    except Exception as e:
        d = {}

    # Return
    return pd.Series(d)

def metrics(x):
    """"""
    return custom_metrics_(x.y_true, x.y_pred, x.y_prob)


# Compute confusion matrix
aux = data.groupby('micro_code').apply(metrics)
#aux['count'] = aux.sum(axis=1)
#aux = aux.sort_values('count', ascending=False)
aux = aux.unstack()
aux = aux.sort_values('count', ascending=False)
print(aux)


aux = data.groupby('day').apply(f)
#aux['count'] = aux.sum(axis=1)
#aux = aux.sort_values('count', ascending=False)
print(aux)


def p25(x):
    return np.nanpercentile(x, 25)

def p75(x):
    return np.nanpercentile(x, 75)

aux = data.groupby('day').y_prob \
    .agg(['mean', 'std', p25, p75])

print("\nDaily:")
print(aux)




# ----------------------
# Visualise box - micro
# ----------------------
# Library
import plotly.express as px

# Display
fig = px.box(data, x='micro_code', y='y_prob', color='y_true')
fig.update_traces(quartilemethod="linear")
fig.show()

# ----------------------
# Visualise box - day
# ----------------------
# Library
import plotly.express as px

# Display
fig = px.box(data, x='day', y='y_prob', color='y_true', points='all')
fig.update_traces(quartilemethod="linear")
fig.show()

# ----------------------
# Visualise violin - day
# ----------------------
# Library
import plotly.express as px

# Display
fig = px.violin(data, x='day', y='y_prob',
    color='y_true', box=True, points='all')
fig.update_traces(quartilemethod="linear")
fig.show()

import sys
sys.exit()

















# ----------------------------------
# Visualise
# ----------------------------------
# Libraries
from itertools import cycle
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from utils.plotly.colors import rgb_to_rgba
from utils.plotly.colors import hex_to_rgba
from utils.plotly.colors import n_colorscale

# Colors (viridis, twilight, turbo, ...)
#col_cycle = cycle(px.colors.qualitative.Alphabet)
#col_cycle = cycle(px.colors.sequential.Viridis)
#col_cycle = cycle(px.colors.cyclical.Twilight)
col_cycle = cycle(n_colorscale("viridis", n=10))

# Compute groups
#g = aux.groupby(level=[0, 2], axis=1)

# Configure
COLS = 1
ROWS = 1
SUBPLOT_TITLES = ['%s' % i for i in range(ROWS*COLS)]

# Show information
print("\nFigure grid: (%s, %s)" % (ROWS, COLS))

# Create Figure
fig = make_subplots(rows=ROWS, cols=COLS,
    shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.002,
    subplot_titles=SUBPLOT_TITLES)

# Variables
x = aux.index
y = aux['mean']
y_upper = aux.p75
y_lower = aux.p25
row = (1 // COLS) + 1
col = (1 % COLS) + 1
color = next(col_cycle)
#title = name[1] if i < COLS else ''

print(y)

# Plot
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        line=dict(color=color),
        mode='lines',  # +markers',
        #name=str(name),
        showlegend=False
    ), row=1, col=1
)

fig.add_trace(
    go.Scatter(
        name='Upper Bound',
        x=x,
        y=y_upper,
        mode='lines',
        marker=dict(color=color),
        line=dict(width=0),
        showlegend=False
    ), row=1, col=1
)

fig.add_trace(
    go.Scatter(
        name='Lower Bound',
        x=x,
        y=y_lower,
        marker=dict(color=color),
        line=dict(width=0),
        mode='lines',
        fillcolor=rgb_to_rgba(color, 0.25),
        fill='tonexty',
        showlegend=False,
        opacity=0.5
    ), row=1, col=1
)

fig.show()

import sys
sys.exit()

# Display
for i, (name, df_) in enumerate(aux):

    # Format DataFrame
    df_ = df_.droplevel(level=2, axis=1)

    print(df_)

    # Variables
    x = df_.index
    y = df_[name[0]]['mean']
    y_upper = df_[name[0]].p75
    y_lower = df_[name[0]].p25
    row = (i // COLS) + 1
    col = (i % COLS) + 1
    color = next(col_cycle)
    title = name[1] if i < COLS else ''

    # Show
    #print("%2s. %s => (%s, %s)" % (i, name, row, col))

    # Plot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color=color),
            mode='lines', #+markers',
            name=str(name),
            showlegend=False
        ), row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            name='Upper Bound',
            x=x,
            y=y_upper,
            mode='lines',
            marker=dict(color=color),
            line=dict(width=0),
            showlegend=False
        ), row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            name='Lower Bound',
            x=x,
            y=y_lower,
            marker=dict(color=color),
            line=dict(width=0),
            mode='lines',
            fillcolor=rgb_to_rgba(color, 0.25),
            fill='tonexty',
            showlegend=False,
            opacity=0.5
        ), row=row, col=col
    )

    # Update text (hacks)
    fig.layout.annotations[i].update(text=title)

    if col == COLS:
        fig.layout['yaxis%s'% (i+1)].title = name[0]
        fig.layout['yaxis%s'% (i+1)].side = 'right'



# Update x-axes
fig.update_xaxes(
    showgrid=False,
    showline=True,
    gridwidth=0.25,
    gridcolor='lightgray',
    tickmode='linear',
    tick0=0,   # ??
    showticklabels=False,
    tickformat=',.2%',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='#666666',
    dtick=5,
)

# Update layout
fig.update_layout(
    height=150*ROWS,
    width=150*COLS,
    title=dict(
        text="Bio-marker TS (mean, p25, p75) by Microorganism",
        x=0.5
    ),
    plot_bgcolor="white"
)

# Show
fig.show()
