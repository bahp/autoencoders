def plot_windows(data, x=None, rows=11, cols=11, scaler='rbt',
        title=''):
    """This method plots a grid with windows.

    Parameters
    ----------

    Returns
    -------
    fig: plotly.figure

    """
    # Libraries

    import plotly.graph_objects as go
    import plotly.express as px
    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots


    # Copy data
    aux = data.copy(deep=True)

    # Apply scaler
    #if scaler is not None:
    #    if isinstance(scaler, str):
    #        scaler = _SCALERS.get(scaler)
    #   aux[x] = scaler.fit_transform(aux[x])

    from utils.settings import _SCALERS
    aux[x] = _SCALERS['mmx'].fit_transform(aux[x])


    # Figure
    fig = make_subplots(rows=rows, cols=cols,
        #vertical_spacing=,
        #horizontal_spacing=,
        #subplot_titles=
        shared_xaxes=True)

    # Display
    for i, (name, g) in enumerate(aux.groupby(level=0)):
        # Break
        if i > (rows * cols) - 1:
            break
        # Compute indexes
        row, col = (i // cols) + 1, (i % cols) + 1
        # Display heatmap
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=g.index.get_level_values(1).day.astype(str),
                z=g[x],
                #zmin=g[x].to_numpy().min(),
                #zmax=g[x].to_numpy().max(),
                coloraxis='coloraxis'
            ),
            row=row, col=col)
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=0.5, y=1.3, showarrow=False,
            text=name, row=row, col=col)

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
        title=title,
        height=150*10,
        width=150*10,
        coloraxis=dict(
            colorscale='Viridis'),
        showlegend=False
    )
    fig.update_coloraxes(
        colorscale='Viridis'
    )
    # Show
    fig.show()

    # Return
    return fig





