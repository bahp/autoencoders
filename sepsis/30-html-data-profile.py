# Libraries
import pandas as pd
from pathlib import Path

# Constant
PATH = Path('./objects/datasets/test')

# Load data
data = pd.read_csv(PATH / 'data.csv')

# -------------------------
# Step 05: Pandas profiling
# -------------------------
# Libraries
from pandas_profiling import ProfileReport

# Create report
profile = ProfileReport(data,
    title="Sepsis Dataset",
    explorative=False,
    minimal=True)

# Save report
profile.to_file(PATH / '01.data.report.html')