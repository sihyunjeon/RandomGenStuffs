#!/usr/bin/env bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
if [ -d "CMSSW_14_2_0" ]; then
    echo "CMSSW_14_2_0 already exists."
else
    cmsrel CMSSW_14_2_0
fi
cd CMSSW_14_2_0/src
cmsenv
cd ../../

if [ -f "MyProxy" ]; then
    echo "MyProxy exists, running rivet"
    mkdir hide_job
    cd hide_job
    mv ../rivet.py ./
    mv ../inputs.txt ./
    cmsRun rivet.py
    mv out.yoda ../
else
    source /cvmfs/cms.cern.ch/rucio/setup.sh
    voms-proxy-init --voms cms
fi
