# Libraries
import pandas as pd
import argparse
from pathlib import Path

# Constant
PATH = './objects/datasets/test'



# -------------------------
# Parameters
# -------------------------
# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, nargs='?',
                    const=PATH, default=PATH,
                    help="path containing grid-search files.")
args = parser.parse_args()

# Load data
data = pd.read_csv(Path(args.path) / 'data.csv')



# -------------------------
# Pandas profiling
# -------------------------
# Libraries
from pandas_profiling import ProfileReport

# Create report
profile = ProfileReport(data,
    title="Sepsis Dataset",
    explorative=False,
    minimal=True)

# Save report
profile.to_file(Path(args.path) / '01.data.report.html')