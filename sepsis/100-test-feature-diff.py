#
import pandas as pd
from pathlib import Path

# The output path
OUTPATH = Path('./objects/datasets/test')

# Load data
df = pd.read_csv(OUTPATH / 'data.csv')

# Filter by days
df = df[(df.day >= -5)]
df = df[(df.day <= 1)]

# Show
print(df)


def delta(x, features=None, periods=1):
    """Computes delta.
    """
    aux = x[features].diff(periods=periods)
    aux.columns = ['%s_d%s' % (e, periods)
        for e in aux.columns]
    return aux

df_ = df.copy(deep=True)
df_ = df_.groupby('PersonID') \
    .apply(delta, features=['HCT', 'PLT'])
df_ = df_.dropna(how='all')

aux = pd.concat([df, df_], axis=1)

# Show
print("Added:")
print(aux)