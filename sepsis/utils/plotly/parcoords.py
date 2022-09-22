
# Create labels
def get_dimension_label(x):
    x = x.split('__')[-1]
    x = x.replace('mean_', '')
    x = x.replace('param_', '')
    x = x.replace('_', ' ')
    return x.title()

def get_dimensions(df):
    """Get parallel coordinates.

    .. note: Needed to work with non numeric values.
    .. note: [dummy[v] for v in df[c]] raises a warning.

    Parameters
    ----------
    df: pd.DataFrame
        The pandas DataFrame with all the columns.
    """
    # Libraries
    from pandas.api.types import is_string_dtype
    from pandas.api.types import is_numeric_dtype

    # Loop
    dimensions = []
    for c in df.columns:
        if is_numeric_dtype(df[c]):
            d_ = dict(
               label=get_dimension_label(c),
               range=(df[c].min(), df[c].max()),
               values=df[c].tolist()
            )
        else:
            unique = df[c].unique()
            dummy = dict(zip(unique, range(len(unique))))
            df[c] = [dummy[v] for v in df[c]]
            d_ = dict(
                label=get_dimension_label(c),
                values=df[c].tolist(),
                tickvals=list(dummy.values()),
                ticktext=list(dummy.keys())
            )
        dimensions.append(d_)

    # Return
    return dimensions