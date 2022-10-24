# Libraries
import pandas as pd
import tensorflow as tf

from pathlib import Path

from utils.settings import _IMPUTERS
from utils.settings import _SCALERS
from utils.lstm.preprocess import create_lstm_matrix


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Define path
PATH = Path('./objects/datasets/set1/data.csv')

# Load data
data = pd.read_csv(PATH,
                   #nrows=10000,
                   dtype={'PersonID': 'str'},
                   parse_dates=['date_collected'])

# Keep raw copy
raw = data.copy(deep=True)

# Drop duplicates
data = data.drop_duplicates()

print(data)

# --------------------------------------------------
# Load model
# --------------------------------------------------
# Define path
MPATH = Path('./objects/results/classification/lstm')
#MPATH = MPATH / '221010-180326/matrix.-31_30.-31_30.w8.simp.std.wft'
#MPATH = MPATH / '221011-144226/matrix.-31_30.-31_30.w8.simp.std.wft'
#MPATH = MPATH / '221011-144226/matrix.-10_3.-5_3.w8.simp.std.wft'
MPATH = MPATH / '221012-181435/matrix.-10_3.-10_3.w5.simp.std.wft'
MPATH = MPATH / 'model.h5'

model = tf.keras.models.load_model(MPATH)

print("Loading model... Done")

from utils.settings import _FEATURES

FEATURES = [
    'WFIO2',
    'WCL',
    'WG',
    'WHB',
    'WHBCO',
    'WHBMET',
    'WHBO2',
    'WHCT',
    'WHHB',
    'WICA',
    'WK',
    'WLAC',
    'WNA',
    'WPCO2',
    'WPH',
    'WPO2',
]

FEATURES = _FEATURES['wft']
LABELS = [
    'pathogenic'
]

def resample_01(df, features, **kwargs):
    aux  = df.droplevel(0) \
        .resample(**kwargs) \
        .asfreq()
    aux[features] = aux[features].ffill()
    aux.day = aux.day.interpolate()
    return aux

# Resample
rsmp = data.copy(deep=True) \
    .set_index(['PersonID', 'date_collected']) \
    .groupby(level=0) \
    .apply(resample_01, features=FEATURES, rule='1D') \
    .reset_index()

# Format data
rsmp[FEATURES] = _IMPUTERS.get('simp').fit_transform(rsmp[FEATURES])
rsmp[FEATURES] = _SCALERS.get('std').fit_transform(rsmp[FEATURES])

rsmp['id'] = rsmp.PersonID

# Format matrix with shape (samples, timestamps, features)
matrix = create_lstm_matrix(rsmp,
    features=FEATURES + ['pathogenic', 'id', 'day'],
    groupby='PersonID',
    w=5)

# Create X and y
X = matrix[:, :, :-3].astype('float32')
y = matrix[:, -1, -3:].astype('float32')



# Compute probabilities
y_prob = model.predict(X)



# ------------------------------------------
# Plot scatter (median, p25, p75)
# ------------------------------------------
# Libraries
import numpy as np

def p25(x):
    return np.nanpercentile(x, 25)

def p75(x):
    return np.nanpercentile(x, 75)

print("Plotting 'probability time-series for each outcome' ...")

# Create DataFrame
df = pd.DataFrame(np.hstack((y, y_prob)),
    columns=['y_true', 'id', 'day', 'y_prob'])

# Compute metrics
agg = df[['y_prob', 'y_true', 'day']] \
    .groupby(['day', 'y_true']) \
    .agg(['mean', 'std', p25, p75]) \
    .droplevel(0, axis=1) \
    .reset_index()

print(agg)


# Libraries
import plotly.graph_objects as go

from itertools import cycle
from plotly.subplots import make_subplots
from utils.plotly.colors import rgb_to_rgba
from utils.plotly.colors import hex_to_rgba
from utils.plotly.colors import n_colorscale

# Cycle
col_cycle = cycle(n_colorscale("viridis", n=2))

# Create Figure
fig = make_subplots(rows=2, cols=1,
    shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.002,
    subplot_titles=['non-pathogenic', 'pathogenic'])

for i, (name, df_) in enumerate(agg.groupby(['y_true'])):

    # Variables
    x = df_['day']
    y = df_['mean']
    y_upper = df_['p75']
    y_lower = df_['p25']
    row = i + 1
    col = 1
    color = next(col_cycle)
    #title = name[1] if i < COLS else ''

    print(x)

    # Plot
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(
                color=color,
                shape='spline',
                smoothing=1.3
            ),
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
            #marker=dict(color=color),
            line=dict(
                width=0,
                shape='spline',
                smoothing=1.3
            ),
            showlegend=False
        ), row=row, col=col
    )

    fig.add_trace(
        go.Scatter(
            name='Lower Bound',
            x=x,
            y=y_lower,
            #marker=dict(color=color),
            line=dict(
                width=0,
                shape='spline',
                smoothing=1.3
            ),
            mode='lines',
            fillcolor=rgb_to_rgba(color, 0.25),
            fill='tonexty',
            showlegend=False,
            opacity=0.5
        ), row=row, col=col
    )

# Update x-axes
fig.update_xaxes(
    showgrid=False,
    showline=True,
    gridwidth=0.25,
    gridcolor='lightgray',
    tickmode='linear',
    tick0=0,   # ??
    showticklabels=True,
    #tickformat=',.2%',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='#666666',
    dtick=5,
    range=[-10, 10]
)

fig.update_yaxes(
    range=[0, 1]
)

# Update layout
fig.update_layout(
    #height=150*ROWS,
    #width=150*COLS,
    title=dict(
        text="Probabilities (mean, p25, p75) by outcome",
        x=0.5
    ),
    #yaxis_range=[0, 1],
    plot_bgcolor="white"
)

fig.show()


import plotly.express as px

fig = px.box(df, x='day', y='y_prob', color='y_true', points='all')

fig.show()

"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

grid = sns.FacetGrid(df, col="id",
    hue="y_true", palette="tab20c", col_wrap=4,
    height=1.5)

# Draw a horizontal line to show the starting point
grid.refline(y=0, linestyle=":")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "day", "prob", marker="o")

for ax in grid.axes.flat:
    ax.axvline(x=0, color='r', linestyle=':')

# Adjust the tick positions and labels
#grid.set(xticks=np.arange(5), yticks=[-3, 3],
#         xlim=(-.5, 4.5), ylim=(-3.5, 3.5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)

plt.show()
"""