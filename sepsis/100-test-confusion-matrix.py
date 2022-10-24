# General
import pandas as pd
import numpy as np

from pathlib import Path

# ------------------------------------------
# Methods
# ------------------------------------------

def create_data(n_samples, n_classes):
    """

    Parameters
    ----------
    n_samples
    n_classes

    Returns
    -------

    """
    # Create data
    day = np.random.randint(-3, 0, n_samples)
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples, n_classes)
    y_pred = np.random.randint(0, 2, n_samples)

    # Create prob column names
    pcols = ['p%s' % i for i in range(y_prob.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(columns=[
        'day', 'y_true', 'y_pred'
    ])

    # Fill DataFrame
    df.day = day
    df.y_true = y_true
    df.y_pred = y_pred
    df[pcols] = y_prob

    return df


def cmt(x):
    """Confusion matrix type

    Parameters
    ----------
    x: series
        It must contain y_true and y_pred

    Returns
    -------

    """
    if bool(x.y_true) & bool(x.y_pred):
        return 'TP'
    if bool(x.y_true) & ~bool(x.y_pred):
        return 'FN'
    if ~bool(x.y_true) & bool(x.y_pred):
        return 'FP'
    if ~bool(x.y_true) & ~bool(x.y_pred):
        return 'TN'
    return np.NaN


# Create data
#df = create_data(n_samples=1000, n_classes=2)

# Load data
base = Path('./objects/results/classification/delta/')
folder = '220906-120028'
filename = 'grid0-idx23.csv'
df = pd.read_csv(base / folder / filename)

# Add y_type
df['y_type'] = df.apply(cmt, axis=1)

# Show
print(df)

#a = df.groupby(['day', 'y_type']).size().unstack()

g = df.groupby(['day', 'y_type'])
print(len(g))
#print(a)



# ----------------------------------
# Visualise
# ----------------------------------
# Libraries
from itertools import cycle
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go



# Distribution (type)
# --------------------
fig = px.histogram(df, x="p1", color="y_type",
                   marginal="rug", # box, violin, rug
                   title="Distributions by type",
                   hover_data=df.columns)
fig.show()


# Distribution (day, type)
# ------------------------
fig = px.box(df, x="y_type", y="p1", color="day",
             notched=True, # used notched shape
             #facet_row='day',
             #facet_col='y_type',
             title="Distributions by day and type",
             category_orders={'day': sorted(df.day.unique())},
             hover_data=["day", "PersonID"])
fig.show()

# Daily distribution
# --------------------
# Colors (viridis, twilight, turbo, ...)
#col_cycle = cycle(px.colors.qualitative.Alphabet)
col_cycle = cycle(px.colors.sequential.Viridis)
#col_cycle = cycle(px.colors.cyclical.Twilight)
#col_cycle = cycle(n_colorscale("viridis", n=5))

# Compute groups
g = df.groupby(['day', 'y_type'])

# Configure
COLS = df.y_type.nunique()
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

    # Variables
    x = df_.day
    y = df_.p1
    row = (i // COLS) + 1
    col = (i % COLS) + 1
    color = next(col_cycle)
    title = name[1] if i < COLS else ''

    # Show
    #print("%2s. %s => (%s, %s)" % (i, name, row, col))


    # Plot
    """
    fig.add_trace(
        go.Histogram(
            x=y,
            #line=dict(color=color),
            #mode='lines', #+markers',
            name=str(name),
            nbinsx=50,
            marker_color=color,
            showlegend=False
        ), row=row, col=col
    )

    """
    fig.add_trace(
        go.Box(
            y=y,
            name=str(name[0]),  # if [0] deleted then scalated
            #boxpoints='all',
            #jitter=0.5,
            #whiskerwidth=0.2,
            #fillcolor=color,
            #marker_size=2,
            #line_width=1,
            #line=dict(color=color),
            #mode='lines', #+markers',
            marker_color=color,
            showlegend=False,
        ), row=row, col=col
    )



    # Update text (hacks)
    fig.layout.annotations[i].update(text=title)

    if col == COLS:
        fig.layout['yaxis%s'% (i+1)].title = str(name[0])
        fig.layout['yaxis%s'% (i+1)].side = 'right'



# Update x-axes
fig.update_xaxes(
    showgrid=True,
    showline=False,
    gridwidth=0.25,
    gridcolor='lightgray',
    tickmode='linear',
    tick0=0,   # ??
    showticklabels=True,
    #tickformat=',.2%',
    zeroline=False,
    zerolinewidth=1,
    zerolinecolor='#666666',
    dtick=10,
)

#fig.update_traces(orientation='h')

# Update layout
fig.update_layout(
    height=150*ROWS,
    width=200*COLS,
    title=dict(
        text="Bio-marker TS (mean, p25, p75) by Microorganism",
        x=0.5
    ),
    #xaxis_range=[0, 100],
    plot_bgcolor="white"
)


# Show
fig.show()