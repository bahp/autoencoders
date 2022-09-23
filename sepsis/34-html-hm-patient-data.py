"""
Author: Bernard
Description:
"""

# Libraries
import pandas as pd
import numpy as np

from pathlib import Path
from utils.plot.heatmaps import plot_windows

# --------------------------------------------------
# Configuration
# --------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

# The output path
PATH = Path('./objects/datasets/test')


# -------------------------
# Parameters
# -------------------------
# Libraries
import argparse

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, nargs='?',
                    const=PATH, default=PATH,
                    help="path containing grid-search files.")
args = parser.parse_args()


# Read data
aux = pd.read_csv(Path(args.path) / 'data.csv')

# Show
print("\n")
print(aux)

# Get all columns which are uppercase.
F = [c for c in aux.columns.tolist() if c.isupper()]

# Normalise features
aux[F] = MinMaxScaler().fit_transform(aux[F])


def score(x, features=None, day_s=-5, day_e=5):
    """

    Parameters
    ----------
    x
    features
    day_s
    day_e
    ratio

    Returns
    -------

    """
    if features is None:
        features = x.columns

    # Filter
    df_ = x.copy(deep=True)
    df_ = df_[(df_.day >= day_s)]
    df_ = df_[(df_.day <= day_e)]
    df_ = df_[features]

    # Score
    num = (~df_.isna()).sum().sum()
    den = len(features) * len(range(day_s, day_e+1))

    # Return
    return num/den

#aux = aux[aux.PersonID.isin([105053, 2506109])]
#aux = aux[aux.PersonID.isin([2506109])]
#print(aux)

# Get person score
DAY_S, DAY_E = -5, 5
person_score = aux.groupby('PersonID') \
    .apply(score, day_s=DAY_S, day_e=DAY_E, features=F) \
    .sort_values(ascending=False) \
    .rename('score')

# Add score to aux
#aux = aux.merge(person_score.to_frame(),
#    on='PersonID', how='left') \
#    .sort_values(by='score', ascending=False)

# Filter
aux = aux[aux.PersonID.isin( \
    person_score[person_score >= 0.50].index)]

# Show
print("\nPerson score (window completeness):")
print(person_score)
print("\nOver 0.75 completeness: %s" % aux.PersonID.nunique())


# -----------------------------
# Display
# -----------------------------
# Libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Compute groups
g = list(aux.groupby('PersonID'))[:50]

# Configure
COLS = 3
ROWS = (len(g) // COLS) + 1
SUBPLOT_TITLES = ['%s' % i for i in range(ROWS*COLS)]

# Show information
print("\nFigure grid: (%s, %s)" % (ROWS, COLS))

# Create Figure
fig = make_subplots(rows=ROWS, cols=COLS,
    shared_yaxes=True, shared_xaxes=False,
    horizontal_spacing=0.02,
    vertical_spacing=0.02,
    subplot_titles=SUBPLOT_TITLES)

# Display
#for i, (name, df_) in enumerate(g):

for i,name in enumerate(person_score.index[:100]):

    print("Plotting...", i, name)

    df_ = aux[aux.PersonID==name]

    # Break
    if i > (ROWS * COLS) - 1:
        break

    # Variables
    x = F
    #y = df_.index.get_level_values(1).day.astype(str)
    y = df_.day
    z = df_[x]
    row = (i // COLS) + 1
    col = (i % COLS) + 1
    title = '%s (%.2f)' % (name, person_score[name])

    # Display heat-map
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            #zmin=g[x].to_numpy().min(),
            #zmax=g[x].to_numpy().max(),
            coloraxis='coloraxis',
        ),
        row=row, col=col)




    """
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5, y=1.3, showarrow=False,
        text=name, row=row, col=col)
    """

    # Update text (hacks)
    fig.layout.annotations[i].update(text=title)


#fig.add_hline(y=DAY_S)
#fig.add_hline(y=DAY_E)
fig.add_hrect(y0=DAY_S, y1=DAY_E, line_width=0,
    fillcolor="red", opacity=0.2)

# Update axes
fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
    tickmode='linear',
    tickfont_size=8
)
fig.update_xaxes(
    tickmode='linear',
    tickfont_size=8
)
fig.update_layout(
    title=dict(
        text="Completeness of patient data (%s patients over 0.75 cut-off)" \
             % aux.PersonID.nunique(),
        x=0.5
    ),
    height=500*ROWS,
    width=500*COLS,
    coloraxis=dict(
        colorscale='Viridis'),
    showlegend=False
)
fig.update_coloraxes(
    colorscale='Viridis'
)

fig.show()
fig.write_html(Path(args.path) / 'graphs' / '05.hm.patient.data.html')