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
    'HCT', 'WBC', 'LY', 'MCV', 'PLT', 'MCH', 'MCHC'
]
#FEATURES = FEATURES + \
#    ['%s_diff'%e for e in FEATURES ]

# Group by
GROUPBY = ['day', 'micro_code']

# Path
PATH = './objects/datasets/test/data-v5.csv'




# ----------------------------------
# Main
# ----------------------------------
# Load data
df = pd.read_csv(PATH)

# Show
print("\nData:")
print(df)

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
col_cycle = cycle(n_colorscale("viridis", n=5))

# Compute groups
g = aux.groupby(level=0, axis=1)

# Configure
COLS = df.micro_code.nunique()
ROWS = (len(g) // COLS) + 1
SUBPLOT_TITLES = ['%s' % i for i in range(ROWS*COLS)]

# Show information
print("\nFigure grid: (%s, %s)" % (ROWS, COLS))

# Create Figure
fig = make_subplots(rows=ROWS, cols=COLS,
    shared_yaxes=False, shared_xaxes=True,
    horizontal_spacing=0.1,
    vertical_spacing=0.05,
    subplot_titles=SUBPLOT_TITLES)

# Display
for i, (name, df_) in enumerate(g):

    # Variables
    x = df_.index
    y = df_[name]['mean']
    y_upper = df_[name].p75
    y_lower = df_[name].p25
    row = (i // COLS) + 1
    col = (i % COLS) + 1
    title = name

    showlegend = True if i==0 else False

    # Show
    #print("%2s. %s => (%s, %s)" % (i, name, row, col))

    for org in y.columns:

        # Plot
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y[org],
                line=dict(color=next(col_cycle)),
                mode='lines', #+markers',
                name=str(org),
                showlegend=showlegend
            ), row=row, col=col
        )

    # Update text (hacks)
    fig.layout.annotations[i].update(text=title)




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
    #height=30*len(g),
    #width=800,
    title=dict(
        text="Bio-marker TS (mean, p25, p75) by Microorganism",
        x=0.5
    ),
    plot_bgcolor="white"
)

# Show
fig.show()
#fig.write_html(OUTPATH / 'data.pivoted.html')