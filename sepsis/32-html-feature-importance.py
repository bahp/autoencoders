import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Configure
OUTPATH = Path('./objects/datasets/test')

FEATURES = [
    'PLT',
    'HGB',
    'HCT',
    'LY',
    'MCH',
    'MCHC',
    'MCV',
    'RBC',
    'RDW',
    'WBC'
]

# Load data
df = pd.read_csv(OUTPATH / 'data.csv')

# Remove NaN
df = df.dropna(how='any', subset=FEATURES)



# ---------------------------
# Feature importance
# ---------------------------
# Libraries
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def show_importance(model, importance, features, ax=None):
    """Displays the importance"""

    print("\n\n%s:" % model)
    for i, v in enumerate(importance):
        print('Feature %0d: %5s, Score: %.5f' % (i, features[i], v))

    # plot feature importance
    if ax is None:
        fig, ax = plt.subplots(figsize=[5, 4])
    ax.bar(range(len(importance)), importance)
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels(features, rotation=45, fontdict={'fontsize':3})
    ax.set_title(model.__class__.__name__)


def get_importances(m, X, y, scoring='accuracy'):
    """Compute permutation"""
    # Fit model
    m.fit(X, y)
    # Compute permutation importance
    perm = permutation_importance(m, X, y, scoring=scoring)
    # Compute single importance
    sing = None
    if hasattr(m, 'coef_'):
        sing = m.coef_[0]
    if hasattr(m, 'feature_importances_'):
        sing =  m.feature_importances_
    # Return
    return [sing, perm]



# Add class
df['label'] = LabelEncoder().fit_transform(df.micro_code)
df['label'] = df.pathogenic

# Data
X = df[FEATURES]
y = df.label

# Transform
#X = StandardScaler().fit_transform(X)
X = MinMaxScaler().fit_transform(X)

# Models
models = [
    LogisticRegression(),     # coef_
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
    #KNeighborsClassifier(),
    #SVC(),
    #MLPClassifier(),
    GaussianNB()
]

# Create figure
fig, axes = plt.subplots(3, 2, sharex=True)
axes = axes.flatten()

# Loop for each model
for i,m in enumerate(models):
    importances = get_importances(m, X, y, scoring='accuracy')
    show_importance(m, importances[1].importances_mean, FEATURES, ax=axes[i])

# Configure axes
for n, ax in enumerate(axes):
    ax.set_xticklabels(FEATURES, rotation=90,
        fontdict={'fontsize':8})

#
plt.title("Feature importance")

# Save
plt.tight_layout()
plt.savefig(OUTPATH / '03.feature.importance.png')

# Show
plt.show()