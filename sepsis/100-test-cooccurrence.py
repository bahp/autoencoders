# Libraries
import pandas as pd
import numpy as np

# Create data
data = [
    [1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1]
]

counts = [
    [1000, 2, 3, 4],
    [2, 2000, 3, 4],
    [3, 3, 2500, 4],
    [4, 3, 3, 500]
]

df = pd.DataFrame(data)
df2 = pd.DataFrame(counts)

# Data
print(df)


# Compute co-occurence
coocc = df.T.dot(df)
coocc_pct = (coocc / np.diag(coocc)) * 100
print(np.diag(coocc))

print("\nCo-occurrence (count)")
print(coocc)
print("\nCo-occurence (%)")
print(coocc_pct)

print("\n\n\n")
print(np.diag(df2))
print(df2)
aux = (df2 / np.diag(df2)) * 100
print(aux)