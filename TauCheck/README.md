# Tau Check

## Instructions

Works in lxplus as of 2025.Aug.23 in el9 through condor.
Works with MiniAOD inputs (GEN inputs need some tweaks in `rivet.py`)

```
#!/usr/bin/env bash

source setup.sh # minimal setup to get rucio env

# create input collection
# now with diboson pythia8 samples
# modify inputs dictionary in the main function for other datasets
python3 collect.py

# create job directories based on txt files created in inputs directory
# this will automatically submit the condor jobs for you
# final output will be <DatasetName>.yoda
# modify config dictionary in the job_config function for maxEvents modification, default 500000
python3 submit.py

# collect final histograms using usual rivet interface commands
# check [link](https://rivet.hepforge.org/doc) for more details
rivet-mkhtml job_<DatasetName1>/<DatasetName1>.yoda job_<DatasetName2>/<DatasetName2>.yoda ...
```
