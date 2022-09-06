"""

Description:

    Displays a grid where <col> is the microorganism and <row> is
    the feature (bio-marker). For each subplot, it plots the
    mean value for each day (where 0 represents day of microbiology
    sample collected). It also includes the p25 and p75 interval.

"""

# Libraries
import numpy as np
import pandas as pd

from pathlib import Path

def rgb_to_rgba(value, alpha=1.0):
    """

    Parameters
    ----------
    value: str
        The input RGB value.
    alpha: float [0,1]
        The transparency in range [0, 1].

    Returns
    -------
    RGBA Value
    """
    return f"rgba{str(value)[3:-1]}, {alpha})"

def hex_to_rgba(value, alpha=1.0):
    """

    Parameters
    ----------
    value: str
        The hexadecimal value.
    alpha: float [0, 1]
        The transparency in range [0, 1].

    Returns
    -------
    RGBA value
    """
    from plotly.colors import hex_to_rgb
    rgb = hex_to_rgb(value)
    rgb = f"rgb{str(rgb)}"
    return rgb_to_rgba(rgb, alpha)

def n_colorscale(cmap="turbo", n=10):
    """"""
    from plotly.colors import sample_colorscale
    return sample_colorscale(cmap, [i / (n-1) for i in range(n)])

def p25(x):
    return np.nanpercentile(x, 25)

def p75(x):
    return np.nanpercentile(x, 75)



# Constants
# ---------

# Features
FEATURES = [
    'HCT', 'WBC', 'LY', 'MCV', 'PLT', 'MCH', 'MCHC', 'HCT_d1', 'PLT_d1'
]
#FEATURES = FEATURES + \
#    ['%s_diff'%e for e in FEATURES ]

# Group by
GROUPBY = ['day', 'micro_code']

# Keep top bio
TOP_BIO = 5

# Path
PATH = Path('./objects/datasets/test')




# ----------------------------------
# Main
# ----------------------------------
# Load data
df = pd.read_csv(PATH / 'data.csv')

# Show
print("\nData:")
print(df)

# Count number of samples per micro.
top_org = df.micro_code \
    .value_counts() \
    .head(TOP_BIO) \
    .index.tolist()

# Filter
df = df[df.micro_code.isin(top_org)]

# Compute metrics
aux = df[FEATURES + GROUPBY] \
    .groupby(GROUPBY) \
    .agg(['mean', 'std', p25, p75]) \
    .unstack()

# Show
print("\nTransformed:")
print(aux)


# ----------------------------------
# Visualise
# ----------------------------------
# Libraries
from itertools import cycle
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# Colors (viridis, twilight, turbo, ...)
#col_cycle = cycle(px.colors.qualitative.Alphabet)
#col_cycle = cycle(px.colors.sequential.Viridis)
#col_cycle = cycle(px.colors.cyclical.Twilight)
col_cycle = cycle(n_colorscale("viridis", n=5))

# Compute groups
g = aux.groupby(level=[0, 2], axis=1)

# Configure
COLS = df.micro_code.nunique()
ROWS = (len(g) // COLS) + 1
SUBPLOT_TITLES = ['%s' % i for i in range(ROWS*COLS)]

# Show information
print("\nFigure grid: (%s, %s)" % (ROWS, COLS))

# Create Figure
fig = make_subplots(rows=ROWS, cols=COLS,
    shared_yaxes=True, shared_xaxes=True,
    horizontal_spacing=0.02,
    vertical_spacing=0.02,
    subplot_titles=SUBPLOT_TITLES)

# Display
for i, (name, df_) in enumerate(g):

    # Format DataFrame
    df_ = df_.droplevel(level=2, axis=1)

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
fig.write_html(PATH / '04.ts.biomarker.grid.html')