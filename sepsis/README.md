
01-prepare-data.py
------------------

02-ls2d-basic
-------------

This is a basic script to compute encoding using different 
dimensionality reduction algorithms. The main purpose is
to use it quick testing.


03-ls2d-loop-gscv.py
--------------------

This script performs GridSearch for the different transformers and hyperparameter
configurations specified in the .yaml file and stores the computes metrics on
a .csv file. By default it loads the 03-ls2d-loop.yaml configuration file.

```
$ python 03-ls2d-loop-gscv.py --yaml <filename>
```

An example of the configuration file is shown below

```
# Location of the data.
datapath: ./objects/datasets/tidy.csv

# Path to store the models.
outpath: ./objects/results/baseline/

# Features to be used for training. The features will
# be automatically sorted alphabetically. Thus, the
# order does not matter.
features:
  - HCT
  - HGB
  - LY
  - MCH
  - MCHC
  - MCV
  - PLT
  - RBC
  - RDW
  - WBC

# Columns that will be considered as targets. These are
# used to compute metrics such as the GMM (Gaussian
# Mixture Models).
targets:
  - micro_code
  - death

# The parameters to create the ParameterGrid to create and
# evaluate different hyper-parameter configurations. Please
# ensure that the hyper-parameters ae specified, otherwise
# a default empty dictionary will be returned.
params:
  pca:
    n_components: [2]
```

It is important to defined the transformers, scalers and 
imputers previously in utils.settings.py so that they can
be loaded by using an acronym.

```
_IMPUTERS = {
    'simp': SimpleImputer()
}

_SCALERS = {
    'std': StandardScaler()
}

_METHODS = {
    'pca': PCA()
}
```

The metrics that will be computed are specified in the 
script itself. Look at the functions custom_metrics().
03-ls2d-loop.yaml.

Note

03-ls2d-loop-encode.csv
-----------------------

This script is equivalent to the 03-ls2d-loop-encode.csv but
it creates a folder with the following structure:

```
pca-simp-mmx
  |- hyper-00
     |- data.csv       # Original data and encoded features
     |- thumbnail.png  # previsualisation
     |- model.pickle   # Optional
     |- matrix.npy     # Optional
  |- hyper-01
```

It is possible to quickly visualise the data by opening the
utils.apps.visualiser.html file in the browser and loading
the corresponding data.csv file.
