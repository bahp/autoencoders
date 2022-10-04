#!/bin/sh

# -----------------------
# Create data
# -----------------------
#python 01-prepare-data.py

# -----------------------
# Some visualisations
# -----------------------
python 30-html-data-profile.py --path ./objects/datasets/set1
python 31-html-basic-info.py --path ./objects/datasets/set1
python 32-feature-importance.py --path ./objects/datasets/set1
python 33-html-ts-biomarker-grid.py --path ./objects/datasets/set1
python 34-html-hm-patient-data.py --path ./objects/datasets/set1


# -----------------------
# Run workbenches
# -----------------------
# run00 (damien) - Using the following features:
#  - CRP
#  - HCT, HGB, PLT, RBC, RDW, WBC
#  - CL, CRE, K, UREA
#  - ALB
#  - ALP
#  - BIL
#  - WFIO2, WCL, WG, WHB, WHBCO, WHBMET, WHBO2, WHCT
#    WHHB, WICA, WK, WLAC, WNA, WPCO2, WPH, WPO2, WSO2
#  - age, gender

python 04-bclass-loop-gscv.py --yaml ./yaml/run00/04.bclass.bayes.agg.yaml

# run01 (combo) - Using the following features:
#
#  - CRP
#  - HCT, HGB, PLT, RBC, RDW, WBC
#  - CL, CRE, K, UREA
#  - ALB
#  - ALP
#  - BIL

python 04-bclass-loop-gscv.py --yaml ./yaml/run01/04.bclass.grid.normal.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run01/04.bclass.grid.agg.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run01/04.bclass.bayes.normal.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run01/04.bclass.bayes.agg.yaml

# run02 (fbc) - Using the following features:
#
#  - HCT, HGB, LY, MCH, MCHC, MCV, PLT, RBC, RDW, WBC

python 04-bclass-loop-gscv.py --yaml ./yaml/run02/04.bclass.grid.normal.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run02/04.bclass.grid.agg.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run02/04.bclass.bayes.normal.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run02/04.bclass.bayes.agg.yaml

# run03 (wbs) - Using the following features:
#
#  - WFIO2, WCL, WG, WHB, WHBCO, WHBMET, WHBO2, WHCT
#    WHHB, WICA, WK, WLAC, WNA, WPCO2, WPH, WPO2, WSO2

python 04-bclass-loop-gscv.py --yaml ./yaml/run03/04.bclass.grid.normal.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run03/04.bclass.grid.agg.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run03/04.bclass.bayes.normal.yaml
python 04-bclass-loop-gscv.py --yaml ./yaml/run03/04.bclass.bayes.agg.yaml

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


