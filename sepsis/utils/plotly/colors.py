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
    """

    Parameters
    ----------
    cmap
    n

    Returns
    -------

    """
    from plotly.colors import sample_colorscale
    return sample_colorscale(cmap, [i / (n-1) for i in range(n)])