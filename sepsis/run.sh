#!/bin/sh

# -----------------------
# Create data
# -----------------------
#python 01-prepare-data.py

# -----------------------
# Some visualisations
# -----------------------

#FOLDER = "Hello World"

python 30-html-data-profile.py --path ./objects/datasets/set1
python 31-html-basic-info.py --path ./objects/datasets/set1
#python 32-feature-importance.py
python 33-html-ts-biomarker-grid.py --path ./objects/datasets/set1
python 34-html-hm-patient-data.py --path ./objects/datasets/set1

# -----------------------
# Compute
# -----------------------

#python 04-bclass-loop-gscv.py --yaml ./yaml/04.bclass.grid.normal.yaml
#python 04-bclass-loop-gscv.py --yaml ./yaml/04.bclass.grid.delta.diff.yaml
#python 04-bclass-loop-gscv.py --yaml ./yaml/04.bclass.grid.delta.pctc.yaml

#python 04-bclass-loop-gscv.py --yaml ./yaml/04.bclass.bayes.normal.yaml
#python 04-bclass-loop-gscv.py --yaml ./yaml/04.bclass.bayes.delta.diff.yaml

#python 04-bclass-loop-gscv.py --yaml ./yaml/04.wbs.bclass.grid.normal.yaml


#python 35-html-gridsearch-parallel.py --path ./objects/results/classification/normal/220914-201243


#for d in */ ; do
#    [ -L "${d%/}" ] && continue
#    echo "$d"
#done


