# Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Specific
from pathlib import Path

# Define history path
PATH = Path('./objects/results/classification/lstm')
#PATH = PATH / '221012-181435'
#PATH = PATH / 'matrix.-10_3.-5_3.w5.simp.std.crp'

# List all interesting files
files = list(PATH.glob('**/history.csv'))

# Loop for al te files.
for i, filepath in enumerate(files):

    # ---------------------------------------------
    # Plot history
    # ---------------------------------------------
    # Load history
    history = pd.read_csv(filepath, index_col=0)

    # Get features
    features = [
        c for c in history.columns
            if not c.startswith('val_')
    ]

    # Create plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    axes = axes.flatten()

    # Loop
    for i, f in enumerate(features):
        axes[i].plot(history[f], label='train')
        axes[i].plot(history['val_%s' % f], label='test')
        axes[i].set_title(f)
        axes[i].legend()

    # Format
    plt.legend()
    plt.suptitle(str(filepath))
    plt.tight_layout()

    # Create folder and save figure
    root = (filepath.parent / 'graphs')
    root.mkdir(parents=True, exist_ok=True)
    plt.savefig(root / 'history.png')

# Show
plt.show()